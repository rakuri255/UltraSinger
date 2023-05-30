import getopt
import copy
import os
import sys
import Levenshtein
import librosa

from moduls import os_helper
from moduls.Audio.vocal_chunks import export_chunks_from_ultrastar_data, convert_audio_to_mono_wav, \
    export_chunks_from_transcribed_data, remove_silence_from_transcribtion_data, convert_wav_to_mp3, \
    export_transcribed_data_to_csv
from moduls.Audio.youtube import download_youtube_video, download_youtube_audio, get_youtube_title, download_youtube_thumbnail
from moduls.Audio.separation import separate_audio
from moduls.Audio.denoise import ffmpeg_reduce_noise
from moduls.Midi import midi_creator
from moduls.Midi.midi_creator import convert_frequencies_to_notes, most_frequent, create_midi_notes_from_pitched_data
from moduls.Pitcher.pitcher import get_frequency_with_high_confidence, get_pitch_with_crepe_file
from moduls.Ultrastar import ultrastar_parser, ultrastar_converter, ultrastar_writer, ultrastar_score_calculator
from moduls.Speech_Recognition.Vosk import transcribe_with_vosk
from moduls.Speech_Recognition.hyphenation import hyphenation, language_check
from moduls.Speech_Recognition.Whisper import transcribe_with_whisper
from moduls.Log import PRINT_ULTRASTAR, print_red_highlighted_text, print_blue_highlighted_text, \
    print_gold_highlighted_text, \
    print_light_blue_highlighted_text
from matplotlib import pyplot as plt
from moduls.Ultrastar.ultrastar_txt import UltrastarTxt
from Settings import Settings
from tqdm import tqdm
from moduls.DeviceDetection.device_detection import get_available_device
from time import sleep

settings = Settings()


def get_confidence(pitched_data, threshold):
    # todo: replace get_frequency_with_high_conf from pitcher
    conf_t = []
    conf_f = []
    conf_c = []
    for i in range(len(pitched_data.times)):
        if pitched_data.confidence[i] > threshold:
            conf_t.append(pitched_data.times[i])
            conf_f.append(pitched_data.frequencies[i])
            conf_c.append(pitched_data.confidence[i])
    return conf_t, conf_f, conf_c


def convert_ultrastar_note_numbers(midi_notes):
    print(f"{PRINT_ULTRASTAR} Creating Ultrastar notes from midi data")

    ultrastar_note_numbers = []
    for i in range(len(midi_notes)):
        note_number_librosa = librosa.note_to_midi(midi_notes[i])
        pitch = ultrastar_converter.midi_note_to_ultrastar_note(note_number_librosa)
        ultrastar_note_numbers.append(pitch)
        # todo: Progress?
        # print("Note: " + midi_notes[i] + " midi_note: " + str(note_number_librosa) + ' pitch: ' + str(pitch))
    return ultrastar_note_numbers


def pitch_each_chunk_with_crepe(directory):
    print(f"{PRINT_ULTRASTAR} Pitching each chunk with {print_blue_highlighted_text('crepe')}")

    midi_notes = []
    for filename in sorted([f for f in os.listdir(directory) if f.endswith('.wav')],
                           key=lambda x: int(x.split("_")[1])):
        filepath = os.path.join(directory, filename)
        # todo: stepsize = duration? then when shorter than "it" it should take the duration. Otherwise there a more notes
        pitched_data = get_pitch_with_crepe_file(filepath, settings.crepe_step_size, settings.crepe_model_capacity)
        conf_f = get_frequency_with_high_confidence(pitched_data.frequencies, pitched_data.confidence)

        notes = convert_frequencies_to_notes(conf_f)
        note = most_frequent(notes)[0][0]

        midi_notes.append(note)
        # todo: Progress?
        # print(filename + " f: " + str(mean))

    return midi_notes


def add_hyphen_to_data(transcribed_data, hyphen_words):
    data = []

    for i in range(len(transcribed_data)):
        if not hyphen_words[i]:
            data.append(transcribed_data[i])
        else:
            chunk_duration = transcribed_data[i].end - transcribed_data[i].start
            chunk_duration = chunk_duration / (len(hyphen_words[i]))

            next_start = transcribed_data[i].start
            for j in range(len(hyphen_words[i])):
                dup = copy.copy(transcribed_data[i])
                dup.start = next_start
                next_start = transcribed_data[i].end - chunk_duration * (len(hyphen_words[i]) - 1 - j)
                dup.end = next_start
                dup.word = hyphen_words[i][j]
                dup.is_hyphen = True
                data.append(dup)

    return data


def get_bpm_from_data(data, sr):
    onset_env = librosa.onset.onset_strength(y=data, sr=sr)
    wav_tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

    print(f"{PRINT_ULTRASTAR} BPM is {print_blue_highlighted_text(str(round(wav_tempo[0], 2)))}")
    return wav_tempo[0]


def get_bpm_from_file(wav_file):
    data, sr = librosa.load(wav_file, sr=None)
    return get_bpm_from_data(data, sr)


def correct_words(recognized_words, word_list_file):
    with open(word_list_file, 'r') as f:
        text = f.read()
    word_list = text.split()

    for i, rec_word in enumerate(recognized_words):
        if rec_word.word in word_list:
            continue
        else:
            closest_word = min(word_list, key=lambda x: Levenshtein.distance(rec_word.word, x))
            print(recognized_words[i].word + " - " + closest_word)
            recognized_words[i].word = closest_word
    return recognized_words


def print_help():
    m = '''
    UltraSinger.py [opt] [mode] [transcription] [pitcher] [extra]
    
    [opt]
    -h      This help text.
    -i      Ultrastar.txt
            audio like .mp3, .wav, youtube link
    -o      Output folder
    
    [mode]
    ## INPUT is audio ##
    default  Creates all
            
    (-u      Create ultrastar txt file) # In Progress
    (-m      Create midi file) # In Progress
    (-s      Create sheet file) # In Progress
    
    ## INPUT is ultrastar.txt ##
    default  Creates all

    (-r      repitch Ultrastar.txt (input has to be audio)) # In Progress
    (-p      Check pitch of Ultrastar.txt input) # In Progress
    (-m      Create midi file) # In Progress

    [transcription]
    --whisper   (default) tiny|base|small|medium|large
    --vosk      Needs model
    
    [extra]
    (-k                     Keep audio chunks) # In Progress
    --hyphenation           (default) True|False
    --disable_separation    True|False
    --disable_karaoke       True|False
    
    [pitcher]
    --crepe  (default) tiny|small|medium|large|full
    '''
    print(m)


def plot(input_file, vosk_transcribed_data, midi_notes):
    pitched_data = get_pitch_with_crepe_file(input_file, settings.crepe_step_size, settings.crepe_model_capacity)
    t, f, c = get_confidence(pitched_data, 0.4)

    plt.ylim(0, 600)
    plt.xlim(0, 50)
    plt.plot(t, f, linewidth=0.1)
    plt.savefig(os.path.join("test", "crepe_0.4.png"))

    for i in range(len(vosk_transcribed_data)):
        nf = librosa.note_to_hz(midi_notes[i])
        plt.plot([vosk_transcribed_data[i].start, vosk_transcribed_data[i].end], [nf, nf], linewidth=1, alpha=0.5)
    plt.savefig(os.path.join("test", "pit.png"), dpi=2000)


def remove_unecessary_punctuations(transcribed_data):
    punctuation = ".,"
    for i in range(len(transcribed_data)):
        transcribed_data[i].word = transcribed_data[i].word.translate({ord(i): None for i in punctuation})


def hyphenate_each_word(language, transcribed_data):
    hyphenated_word = []
    lang_region = language_check(language)
    if lang_region is None:
        print(f"{PRINT_ULTRASTAR} {print_red_highlighted_text('Error in hyphenation for language ')} {print_blue_highlighted_text(language)} {print_red_highlighted_text(', maybe you want to disable it?')}")
        return None

    sleep(.1)
    for i in tqdm(range(len(transcribed_data))):
        hyphenated_word.append(hyphenation(transcribed_data[i].word, lang_region))
    return hyphenated_word


def print_support():
    print()
    print(f"{PRINT_ULTRASTAR} {print_gold_highlighted_text('Do you like UltraSinger? And want it to be even better? Then help with your')} {print_light_blue_highlighted_text('support')}{print_gold_highlighted_text('!')}")
    print(f"{PRINT_ULTRASTAR} See project page -> https://github.com/rakuri255/UltraSinger")
    print(f"{PRINT_ULTRASTAR} {print_gold_highlighted_text('This will help alot to keep this project alive and improved.')}")


def run():
    isAudio = ".txt" not in settings.input_file_path
    ultrastar_class = None
    real_bpm = None

    if not isAudio:  # Parse Ultrastar txt
        print(f"{PRINT_ULTRASTAR} {print_gold_highlighted_text('re-pitch mode')}")
        basename_without_ext, real_bpm, song_output, ultrastar_audio_input_path, ultrastar_class = parse_ultrastar_txt()
    elif settings.input_file_path.startswith('https:'):  # Youtube
        print(f"{PRINT_ULTRASTAR} {print_gold_highlighted_text('full automatic mode')}")
        basename_without_ext, song_output, ultrastar_audio_input_path = download_from_youtube()
    else:  # Audio File
        print(f"{PRINT_ULTRASTAR} {print_gold_highlighted_text('full automatic mode')}")
        basename_without_ext, song_output, ultrastar_audio_input_path = setup_audio_input_file()

    cache_path = os.path.join(song_output, 'cache')
    settings.mono_audio_path = os.path.join(cache_path, basename_without_ext + '.wav')
    os_helper.create_folder(cache_path)

    # Separate vocal from audio
    audio_separation_path = separate_vocal_from_audio(basename_without_ext, cache_path, ultrastar_audio_input_path)

    # Denoise vocal audio
    denoise_vocal_audio(basename_without_ext, cache_path)

    # Audio transcription
    transcribed_data = None
    language = None
    if isAudio:
        language, transcribed_data = transcribe_audio(transcribed_data)
        remove_unecessary_punctuations(transcribed_data)
        transcribed_data = remove_silence_from_transcribtion_data(settings.mono_audio_path, transcribed_data)

        if settings.hyphenation:
            hyphen_words = hyphenate_each_word(language, transcribed_data)
            if hyphen_words is not None:
                transcribed_data = add_hyphen_to_data(transcribed_data, hyphen_words)

        # todo: do we need to correct words?
        # lyric = 'input/faber_lyric.txt'
        # --corrected_words = correct_words(vosk_speech, lyric)

    # Create audio chunks
    if settings.create_audio_chunks:
        create_audio_chunks(cache_path, isAudio, transcribed_data, ultrastar_audio_input_path, ultrastar_class)

    # Pitch the audio
    midi_notes, pitched_data, ultrastar_note_numbers = pitch_audio(isAudio, transcribed_data, ultrastar_class)

    # Create plot
    if isAudio and settings.create_plot:
        plot(ultrastar_audio_input_path, transcribed_data, midi_notes)

    # Write Ultrastar txt
    if isAudio:
        real_bpm, ultrastar_file_output = create_ultrastar_txt_from_automation(audio_separation_path,
                                                                               basename_without_ext,
                                                                               song_output, transcribed_data,
                                                                               ultrastar_audio_input_path,
                                                                               ultrastar_note_numbers, language)
    else:
        ultrastar_file_output = create_ultrastar_txt_from_ultrastar_data(song_output, ultrastar_class,
                                                                         ultrastar_note_numbers)

    # Calc Points
    ultrastar_class, simple_score, accurate_score = calculate_score_points(isAudio, pitched_data, ultrastar_class, ultrastar_file_output)

    # Add calculated score to Ultrastar txt
    ultrastar_writer.add_score_to_ultrastar_txt(ultrastar_file_output, simple_score)

    # Midi
    if settings.create_midi:
        create_midi_file(isAudio, real_bpm, song_output, ultrastar_class)

    # Print Support
    print_support()


def transcribe_audio(transcribed_data):
    if settings.transcriber == "whisper":
        transcribed_data, language = transcribe_with_whisper(settings.mono_audio_path, settings.whisper_model,
                                                             settings.device)
    else:  # vosk
        transcribed_data = transcribe_with_vosk(settings.mono_audio_path, settings.vosk_model_path)
        # todo: make language selectable
        language = 'en'
    return language, transcribed_data


def separate_vocal_from_audio(basename_without_ext, cache_path, ultrastar_audio_input_path):
    audio_separation_path = os.path.join(cache_path, "separated", "htdemucs", basename_without_ext)
    if settings.use_separated_vocal or settings.create_karaoke:
        separate_audio(ultrastar_audio_input_path, cache_path)
    if settings.use_separated_vocal:
        vocals_path = os.path.join(audio_separation_path, "vocals.wav")
        convert_audio_to_mono_wav(vocals_path, settings.mono_audio_path)
    else:
        convert_audio_to_mono_wav(ultrastar_audio_input_path, settings.mono_audio_path)
    return audio_separation_path


def calculate_score_points(isAudio, pitched_data, ultrastar_class, ultrastar_file_output):
    if isAudio:
        ultrastar_class = ultrastar_parser.parse_ultrastar_txt(ultrastar_file_output)
        simple_score, accurate_score = ultrastar_score_calculator.calculate_score(pitched_data, ultrastar_class)
        ultrastar_score_calculator.print_score_calculation(simple_score, accurate_score)
    else:
        print(f"{PRINT_ULTRASTAR} {print_blue_highlighted_text('Score of original Ultrastar txt')}")
        simple_score, accurate_score = ultrastar_score_calculator.calculate_score(pitched_data, ultrastar_class)
        ultrastar_score_calculator.print_score_calculation(simple_score, accurate_score)
        print(f"{PRINT_ULTRASTAR} {print_blue_highlighted_text('Score of re-pitched Ultrastar txt')}")
        ultrastar_class = ultrastar_parser.parse_ultrastar_txt(ultrastar_file_output)
        simple_score, accurate_score = ultrastar_score_calculator.calculate_score(pitched_data, ultrastar_class)
        ultrastar_score_calculator.print_score_calculation(simple_score, accurate_score)
    return ultrastar_class, simple_score, accurate_score


def create_ultrastar_txt_from_ultrastar_data(song_output, ultrastar_class, ultrastar_note_numbers):
    output_repitched_ultrastar = os.path.join(song_output, ultrastar_class.title + '.txt')
    ultrastar_writer.create_repitched_txt_from_ultrastar_data(settings.input_file_path, ultrastar_note_numbers,
                                                              output_repitched_ultrastar)
    return output_repitched_ultrastar


def create_ultrastar_txt_from_automation(audio_separation_path, basename_without_ext, song_output,
                                         transcribed_data, ultrastar_audio_input_path, ultrastar_note_numbers,
                                         language):
    ultrastar_header = UltrastarTxt()
    ultrastar_header.title = basename_without_ext
    ultrastar_header.artist = basename_without_ext
    ultrastar_header.mp3 = basename_without_ext + ".mp3"
    ultrastar_header.video = basename_without_ext + ".mp4"
    ultrastar_header.language = language
    cover = basename_without_ext + " [CO].jpg"
    ultrastar_header.cover = cover if os_helper.check_file_exists(os.path.join(song_output, cover)) else None

    real_bpm = get_bpm_from_file(ultrastar_audio_input_path)
    ultrastar_file_output = os.path.join(song_output, basename_without_ext + '.txt')
    ultrastar_writer.create_ultrastar_txt_from_automation(transcribed_data, ultrastar_note_numbers,
                                                          ultrastar_file_output,
                                                          ultrastar_header, real_bpm)
    if settings.create_karaoke:
        no_vocals_path = os.path.join(audio_separation_path, "no_vocals.wav")
        title = basename_without_ext + " [Karaoke]"
        ultrastar_header.title = title
        ultrastar_header.mp3 = title + ".mp3"
        karaoke_output_path = os.path.join(song_output, title)
        karaoke_audio_output_path = karaoke_output_path + ".mp3"
        convert_wav_to_mp3(no_vocals_path, karaoke_audio_output_path)
        karaoke_txt_output_path = karaoke_output_path + ".txt"
        ultrastar_writer.create_ultrastar_txt_from_automation(transcribed_data, ultrastar_note_numbers,
                                                              karaoke_txt_output_path,
                                                              ultrastar_header, real_bpm)
    return real_bpm, ultrastar_file_output


def setup_audio_input_file():
    basename = os.path.basename(settings.input_file_path)
    basename_without_ext = os.path.splitext(basename)[0]
    song_output = os.path.join(settings.output_file_path, basename_without_ext)
    os_helper.create_folder(song_output)
    os_helper.copy(settings.input_file_path, song_output)
    ultrastar_audio_input_path = os.path.join(song_output, basename)
    return basename_without_ext, song_output, ultrastar_audio_input_path


FILENAME_REPLACEMENTS = (('?:"', ""), ("<", "("), (">", ")"), ("/\\|*", "-"))


def sanitize_filename(fname: str) -> str:
    for old, new in FILENAME_REPLACEMENTS:
        for char in old:
            fname = fname.replace(char, new)
    if fname.endswith("."):
        fname = fname.rstrip(" .")  # Windows does not like trailing periods
    return fname


def download_from_youtube():
    title = get_youtube_title(settings.input_file_path)
    basename_without_ext = sanitize_filename(title)
    basename = basename_without_ext + '.mp3'
    song_output = os.path.join(settings.output_file_path, basename_without_ext)
    os_helper.create_folder(song_output)
    download_youtube_audio(settings.input_file_path, basename_without_ext, song_output)
    download_youtube_video(settings.input_file_path, basename_without_ext, song_output)
    download_youtube_thumbnail(settings.input_file_path, basename_without_ext, song_output)
    ultrastar_audio_input_path = os.path.join(song_output, basename)
    return basename_without_ext, song_output, ultrastar_audio_input_path


def parse_ultrastar_txt():
    ultrastar_class = ultrastar_parser.parse_ultrastar_txt(settings.input_file_path)
    real_bpm = ultrastar_converter.ultrastar_bpm_to_real_bpm(float(ultrastar_class.bpm.replace(',', '.')))
    ultrastar_mp3_name = ultrastar_class.mp3
    basename_without_ext = os.path.splitext(ultrastar_mp3_name)[0]
    dirname = os.path.dirname(settings.input_file_path)
    ultrastar_audio_input_path = os.path.join(dirname, ultrastar_mp3_name)
    song_output = os.path.join(settings.output_file_path, ultrastar_class.artist + ' - ' + ultrastar_class.title)
    return basename_without_ext, real_bpm, song_output, ultrastar_audio_input_path, ultrastar_class


def create_midi_file(isAudio, real_bpm, song_output, ultrastar_class):
    print(f"{PRINT_ULTRASTAR} Creating Midi with {print_blue_highlighted_text('pretty_midi')}")
    if isAudio:
        voice_instrument = [midi_creator.convert_ultrastar_to_midi_instrument(ultrastar_class)]
        midi_output = os.path.join(song_output, ultrastar_class.title + '.mid')
        midi_creator.instruments_to_midi(voice_instrument, real_bpm, midi_output)
    else:
        voice_instrument = [midi_creator.convert_ultrastar_to_midi_instrument(ultrastar_class)]
        midi_output = os.path.join(song_output, ultrastar_class.title + '.mid')
        midi_creator.instruments_to_midi(voice_instrument, real_bpm, midi_output)


def pitch_audio(isAudio, transcribed_data, ultrastar_class):
    # todo: chunk pitching as option?
    # midi_notes = pitch_each_chunk_with_crepe(chunk_folder_name)
    pitched_data = get_pitch_with_crepe_file(settings.mono_audio_path, settings.crepe_step_size,
                                             settings.crepe_model_capacity)
    if isAudio:
        start_times = []
        end_times = []
        for i in range(len(transcribed_data)):
            start_times.append(transcribed_data[i].start)
            end_times.append(transcribed_data[i].end)
        midi_notes = create_midi_notes_from_pitched_data(start_times, end_times, pitched_data)

    else:
        midi_notes = create_midi_notes_from_pitched_data(ultrastar_class.startTimes, ultrastar_class.endTimes,
                                                         pitched_data)
    ultrastar_note_numbers = convert_ultrastar_note_numbers(midi_notes)
    return midi_notes, pitched_data, ultrastar_note_numbers


def create_audio_chunks(cache_path, isAudio, transcribed_data, ultrastar_audio_input_path, ultrastar_class):
    audio_chunks_path = os.path.join(cache_path, settings.audio_chunk_folder_name)
    os_helper.create_folder(audio_chunks_path)
    if isAudio:  # and csv
        csv_filename = os.path.join(audio_chunks_path, "_chunks.csv")
        export_chunks_from_transcribed_data(settings.mono_audio_path, transcribed_data, audio_chunks_path)
        export_transcribed_data_to_csv(transcribed_data, csv_filename)
    else:
        export_chunks_from_ultrastar_data(ultrastar_audio_input_path, ultrastar_class, audio_chunks_path)


def denoise_vocal_audio(basename_without_ext, cache_path):
    denoised_path = os.path.join(cache_path, basename_without_ext + '_denoised.wav')
    ffmpeg_reduce_noise(settings.mono_audio_path, denoised_path)
    settings.mono_audio_path = denoised_path


def main(argv):
    init_settings(argv)
    run()
    # todo: cleanup
    sys.exit()


def init_settings(argv):
    short = "hi:o:amv:"
    long = ["ifile=", "ofile=", "crepe=", "vosk=", "whisper=", "hyphenation=", "disable_separation=",
            "disable_karaoke=", "create_audio_chunks="]
    opts, args = getopt.getopt(argv, short, long)
    if len(opts) == 0:
        print_help()
        sys.exit()
    for opt, arg in opts:
        if opt == '-h':
            print_help()
            sys.exit()
        elif opt in ("-i", "--ifile"):
            settings.input_file_path = arg
        elif opt in ("-o", "--ofile"):
            settings.output_file_path = arg
        elif opt in ("--whisper"):
            settings.transcriber = "whisper"
            settings.whisper_model = arg
        elif opt in ("--vosk"):
            settings.transcriber = 'vosk'
            settings.vosk_model_path = arg
        elif opt in ("--crepe"):
            settings.crepe_model_capacity = arg
        elif opt in ("--hyphenation"):
            settings.hyphenation = arg
        elif opt in ("--disable_separation"):
            settings.use_separated_vocal = not arg
        elif opt in ("--disable_karaoke"):
            settings.create_karaoke = not arg
        elif opt in ("--create_audio_chunks"):
            settings.create_audio_chunks = arg
    if settings.output_file_path == '':
        if settings.input_file_path.startswith('https:'):
            dirname = os.getcwd()
        else:
            dirname = os.path.dirname(settings.input_file_path)
        settings.output_file_path = os.path.join(dirname, 'output')
    settings.device = get_available_device()


if __name__ == "__main__":
    main(sys.argv[1:])
