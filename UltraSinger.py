"""UltraSinger uses AI to automatically create UltraStar song files"""

import copy
import getopt
import os
import sys
from time import sleep

import Levenshtein
import librosa
from matplotlib import pyplot as plt
from tqdm import tqdm

from modules import os_helper
from modules.Audio.denoise import ffmpeg_reduce_noise
from modules.Audio.separation import separate_audio
from modules.Audio.vocal_chunks import (
    convert_audio_to_mono_wav,
    convert_wav_to_mp3,
    export_chunks_from_transcribed_data,
    export_chunks_from_ultrastar_data,
    export_transcribed_data_to_csv,
    remove_silence_from_transcribtion_data,
)
from modules.Audio.youtube import (
    download_youtube_audio,
    download_youtube_thumbnail,
    download_youtube_video,
    get_youtube_title,
)
from modules.DeviceDetection.device_detection import get_available_device
from modules.Log import (
    PRINT_ULTRASTAR,
    print_blue_highlighted_text,
    print_gold_highlighted_text,
    print_light_blue_highlighted_text,
    print_red_highlighted_text,
)
from modules.Midi import midi_creator
from modules.Midi.midi_creator import (
    convert_frequencies_to_notes,
    create_midi_notes_from_pitched_data,
    most_frequent,
)
from modules.Pitcher.pitcher import (
    get_frequency_with_high_confidence,
    get_pitch_with_crepe_file,
)
from modules.Speech_Recognition.hyphenation import hyphenation, language_check
from modules.Speech_Recognition.Vosk import transcribe_with_vosk
from modules.Speech_Recognition.Whisper import transcribe_with_whisper
from modules.Ultrastar import (
    ultrastar_converter,
    ultrastar_parser,
    ultrastar_score_calculator,
    ultrastar_writer,
)
from modules.Ultrastar.ultrastar_txt import UltrastarTxt
from Settings import Settings
from modules.musicbrainz_client import get_music_infos

settings = Settings()


def get_confidence(pitched_data, threshold):
    """Docstring"""
    # todo: replace get_frequency_with_high_conf from pitcher
    conf_t = []
    conf_f = []
    conf_c = []
    for i in enumerate(pitched_data.times):
        if pitched_data.confidence[i] > threshold:
            conf_t.append(pitched_data.times[i])
            conf_f.append(pitched_data.frequencies[i])
            conf_c.append(pitched_data.confidence[i])
    return conf_t, conf_f, conf_c


def convert_ultrastar_note_numbers(midi_notes):
    """Docstring"""
    print(f"{PRINT_ULTRASTAR} Creating Ultrastar notes from midi data")

    ultrastar_note_numbers = []
    for i in enumerate(midi_notes):
        note_number_librosa = librosa.note_to_midi(midi_notes[i])
        pitch = ultrastar_converter.midi_note_to_ultrastar_note(
            note_number_librosa
        )
        ultrastar_note_numbers.append(pitch)
        # todo: Progress?
        # print(
        #    f"Note: {midi_notes[i]} midi_note: {str(note_number_librosa)} pitch: {str(pitch)}"
        # )
    return ultrastar_note_numbers


def pitch_each_chunk_with_crepe(directory):
    """Docstring"""
    print(
        f"{PRINT_ULTRASTAR} Pitching each chunk with {print_blue_highlighted_text('crepe')}"
    )

    midi_notes = []
    for filename in sorted(
        [f for f in os.listdir(directory) if f.endswith(".wav")],
        key=lambda x: int(x.split("_")[1]),
    ):
        filepath = os.path.join(directory, filename)
        # todo: stepsize = duration? then when shorter than "it" it should take the duration. Otherwise there a more notes
        pitched_data = get_pitch_with_crepe_file(
            filepath, settings.crepe_step_size, settings.crepe_model_capacity
        )
        conf_f = get_frequency_with_high_confidence(
            pitched_data.frequencies, pitched_data.confidence
        )

        notes = convert_frequencies_to_notes(conf_f)
        note = most_frequent(notes)[0][0]

        midi_notes.append(note)
        # todo: Progress?
        # print(filename + " f: " + str(mean))

    return midi_notes


def add_hyphen_to_data(transcribed_data, hyphen_words):
    """Docstring"""
    data = []

    for i in enumerate(transcribed_data):
        if not hyphen_words[i]:
            data.append(transcribed_data[i])
        else:
            chunk_duration = transcribed_data[i].end - transcribed_data[i].start
            chunk_duration = chunk_duration / (len(hyphen_words[i]))

            next_start = transcribed_data[i].start
            for j in enumerate(hyphen_words[i]):
                dup = copy.copy(transcribed_data[i])
                dup.start = next_start
                next_start = transcribed_data[i].end - chunk_duration * (
                    len(hyphen_words[i]) - 1 - j
                )
                dup.end = next_start
                dup.word = hyphen_words[i][j]
                dup.is_hyphen = True
                data.append(dup)

    return data


def get_bpm_from_data(data, sampling_rate):
    """Docstring"""
    onset_env = librosa.onset.onset_strength(y=data, sr=sampling_rate)
    wav_tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sampling_rate)

    print(
        f"{PRINT_ULTRASTAR} BPM is {print_blue_highlighted_text(str(round(wav_tempo[0], 2)))}"
    )
    return wav_tempo[0]


def get_bpm_from_file(wav_file):
    """Docstring"""
    data, sampling_rate = librosa.load(wav_file, sr=None)
    return get_bpm_from_data(data, sampling_rate)


def correct_words(recognized_words, word_list_file):
    """Docstring"""
    with open(word_list_file, "r", encoding="utf-8") as file:
        text = file.read()
    word_list = text.split()

    for i, rec_word in enumerate(recognized_words):
        if rec_word.word in word_list:
            continue

        closest_word = min(
            word_list, key=lambda x: Levenshtein.distance(rec_word.word, x)
        )
        print(recognized_words[i].word + " - " + closest_word)
        recognized_words[i].word = closest_word
    return recognized_words


def print_help():
    """Docstring"""
    help_string = """
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
    """
    print(help_string)


def plot(input_file, vosk_transcribed_data, midi_notes):
    """Docstring"""
    pitched_data = get_pitch_with_crepe_file(
        input_file, settings.crepe_step_size, settings.crepe_model_capacity
    )
    conf_t, conf_f, conf_c = get_confidence(pitched_data, 0.4)

    plt.ylim(0, 600)
    plt.xlim(0, 50)
    plt.plot(conf_t, conf_f, linewidth=0.1)
    plt.savefig(os.path.join("test", "crepe_0.4.png"))

    for i in enumerate(vosk_transcribed_data):
        note_frequency = librosa.note_to_hz(midi_notes[i])
        plt.plot(
            [vosk_transcribed_data[i].start, vosk_transcribed_data[i].end],
            [note_frequency, note_frequency],
            linewidth=1,
            alpha=0.5,
        )
    plt.savefig(os.path.join("test", "pit.png"), dpi=2000)


def remove_unecessary_punctuations(transcribed_data):
    """Docstring"""
    punctuation = ".,"
    for i in enumerate(transcribed_data):
        transcribed_data[i].word = transcribed_data[i].word.translate(
            {ord(i): None for i in punctuation}
        )


def hyphenate_each_word(language, transcribed_data):
    """Docstring"""
    hyphenated_word = []
    lang_region = language_check(language)
    if lang_region is None:
        print(
            f"{PRINT_ULTRASTAR} {print_red_highlighted_text('Error in hyphenation for language ')} {print_blue_highlighted_text(language)} {print_red_highlighted_text(', maybe you want to disable it?')}"
        )
        return None

    sleep(0.1)
    for i in tqdm(enumerate(transcribed_data)):
        hyphenated_word.append(
            hyphenation(transcribed_data[i].word, lang_region)
        )
    return hyphenated_word


def print_support():
    """Docstring"""
    print()
    print(
        f"{PRINT_ULTRASTAR} {print_gold_highlighted_text('Do you like UltraSinger? And want it to be even better? Then help with your')} {print_light_blue_highlighted_text('support')}{print_gold_highlighted_text('!')}"
    )
    print(
        f"{PRINT_ULTRASTAR} See project page -> https://github.com/rakuri255/UltraSinger"
    )
    print(
        f"{PRINT_ULTRASTAR} {print_gold_highlighted_text('This will help alot to keep this project alive and improved.')}"
    )


def run():
    """Docstring"""
    is_audio = ".txt" not in settings.input_file_path
    ultrastar_class = None
    real_bpm = None

    if not is_audio:  # Parse Ultrastar txt
        print(
            f"{PRINT_ULTRASTAR} {print_gold_highlighted_text('re-pitch mode')}"
        )
        (
            basename_without_ext,
            real_bpm,
            song_output,
            ultrastar_audio_input_path,
            ultrastar_class,
        ) = parse_ultrastar_txt()
    elif settings.input_file_path.startswith("https:"):  # Youtube
        print(
            f"{PRINT_ULTRASTAR} {print_gold_highlighted_text('full automatic mode')}"
        )
        (
            basename_without_ext,
            song_output,
            ultrastar_audio_input_path,
        ) = download_from_youtube()
    else:  # Audio File
        print(
            f"{PRINT_ULTRASTAR} {print_gold_highlighted_text('full automatic mode')}"
        )
        (
            basename_without_ext,
            song_output,
            ultrastar_audio_input_path,
        ) = setup_audio_input_file()

    cache_path = os.path.join(song_output, "cache")
    settings.mono_audio_path = os.path.join(
        cache_path, basename_without_ext + ".wav"
    )
    os_helper.create_folder(cache_path)

    # Get additional data for song
    get_music_infos(basename_without_ext)

    # Separate vocal from audio
    audio_separation_path = separate_vocal_from_audio(
        basename_without_ext, cache_path, ultrastar_audio_input_path
    )

    # Denoise vocal audio
    denoise_vocal_audio(basename_without_ext, cache_path)

    # Audio transcription
    transcribed_data = None
    language = None
    if is_audio:
        language, transcribed_data = transcribe_audio(transcribed_data)
        remove_unecessary_punctuations(transcribed_data)
        transcribed_data = remove_silence_from_transcribtion_data(
            settings.mono_audio_path, transcribed_data
        )

        if settings.hyphenation:
            hyphen_words = hyphenate_each_word(language, transcribed_data)
            if hyphen_words is not None:
                transcribed_data = add_hyphen_to_data(
                    transcribed_data, hyphen_words
                )

        # todo: do we need to correct words?
        # lyric = 'input/faber_lyric.txt'
        # --corrected_words = correct_words(vosk_speech, lyric)

    # Create audio chunks
    if settings.create_audio_chunks:
        create_audio_chunks(
            cache_path,
            is_audio,
            transcribed_data,
            ultrastar_audio_input_path,
            ultrastar_class,
        )

    # Pitch the audio
    midi_notes, pitched_data, ultrastar_note_numbers = pitch_audio(
        is_audio, transcribed_data, ultrastar_class
    )

    # Create plot
    if is_audio and settings.create_plot:
        plot(ultrastar_audio_input_path, transcribed_data, midi_notes)

    # Write Ultrastar txt
    if is_audio:
        real_bpm, ultrastar_file_output = create_ultrastar_txt_from_automation(
            audio_separation_path,
            basename_without_ext,
            song_output,
            transcribed_data,
            ultrastar_audio_input_path,
            ultrastar_note_numbers,
            language,
        )
    else:
        ultrastar_file_output = create_ultrastar_txt_from_ultrastar_data(
            song_output, ultrastar_class, ultrastar_note_numbers
        )

    # Calc Points
    ultrastar_class, simple_score, accurate_score = calculate_score_points(
        is_audio, pitched_data, ultrastar_class, ultrastar_file_output
    )

    # Add calculated score to Ultrastar txt
    ultrastar_writer.add_score_to_ultrastar_txt(
        ultrastar_file_output, simple_score
    )

    # Midi
    if settings.create_midi:
        create_midi_file(is_audio, real_bpm, song_output, ultrastar_class)

    # Print Support
    print_support()


def get_unused_song_output_dir(path):
    """Docstring"""
    # check if dir exists and add (i) if it does
    i = 1
    if os_helper.check_if_folder_exists(path):
        path = f"{path} ({i})"
    else:
        return path

    while os_helper.check_if_folder_exists(path):
        path = path.replace(f"({i - 1})", f"({i})")
        i += 1
        if i > 999:
            print(
                f"{PRINT_ULTRASTAR} {print_red_highlighted_text('Error: Could not create output folder! (999) is the maximum number of tries.')}"
            )
            sys.exit(1)
    return path


def transcribe_audio(transcribed_data):
    """Docstring"""
    if settings.transcriber == "whisper":
        transcribed_data, language = transcribe_with_whisper(
            settings.mono_audio_path, settings.whisper_model, settings.device
        )
    else:  # vosk
        transcribed_data = transcribe_with_vosk(
            settings.mono_audio_path, settings.vosk_model_path
        )
        # todo: make language selectable
        language = "en"
    return language, transcribed_data


def separate_vocal_from_audio(
    basename_without_ext, cache_path, ultrastar_audio_input_path
):
    """Docstring"""
    audio_separation_path = os.path.join(
        cache_path, "separated", "htdemucs", basename_without_ext
    )
    if settings.use_separated_vocal or settings.create_karaoke:
        separate_audio(ultrastar_audio_input_path, cache_path)
    if settings.use_separated_vocal:
        vocals_path = os.path.join(audio_separation_path, "vocals.wav")
        convert_audio_to_mono_wav(vocals_path, settings.mono_audio_path)
    else:
        convert_audio_to_mono_wav(
            ultrastar_audio_input_path, settings.mono_audio_path
        )
    return audio_separation_path


def calculate_score_points(
    is_audio, pitched_data, ultrastar_class, ultrastar_file_output
):
    """Docstring"""
    if is_audio:
        ultrastar_class = ultrastar_parser.parse_ultrastar_txt(
            ultrastar_file_output
        )
        (
            simple_score,
            accurate_score,
        ) = ultrastar_score_calculator.calculate_score(
            pitched_data, ultrastar_class
        )
        ultrastar_score_calculator.print_score_calculation(
            simple_score, accurate_score
        )
    else:
        print(
            f"{PRINT_ULTRASTAR} {print_blue_highlighted_text('Score of original Ultrastar txt')}"
        )
        (
            simple_score,
            accurate_score,
        ) = ultrastar_score_calculator.calculate_score(
            pitched_data, ultrastar_class
        )
        ultrastar_score_calculator.print_score_calculation(
            simple_score, accurate_score
        )
        print(
            f"{PRINT_ULTRASTAR} {print_blue_highlighted_text('Score of re-pitched Ultrastar txt')}"
        )
        ultrastar_class = ultrastar_parser.parse_ultrastar_txt(
            ultrastar_file_output
        )
        (
            simple_score,
            accurate_score,
        ) = ultrastar_score_calculator.calculate_score(
            pitched_data, ultrastar_class
        )
        ultrastar_score_calculator.print_score_calculation(
            simple_score, accurate_score
        )
    return ultrastar_class, simple_score, accurate_score


def create_ultrastar_txt_from_ultrastar_data(
    song_output, ultrastar_class, ultrastar_note_numbers
):
    """Docstring"""
    output_repitched_ultrastar = os.path.join(
        song_output, ultrastar_class.title + ".txt"
    )
    ultrastar_writer.create_repitched_txt_from_ultrastar_data(
        settings.input_file_path,
        ultrastar_note_numbers,
        output_repitched_ultrastar,
    )
    return output_repitched_ultrastar


def create_ultrastar_txt_from_automation(
    audio_separation_path,
    basename_without_ext,
    song_output,
    transcribed_data,
    ultrastar_audio_input_path,
    ultrastar_note_numbers,
    language,
):
    """Docstring"""
    ultrastar_header = UltrastarTxt()
    ultrastar_header.title = basename_without_ext
    ultrastar_header.artist = basename_without_ext
    ultrastar_header.mp3 = basename_without_ext + ".mp3"
    ultrastar_header.video = basename_without_ext + ".mp4"
    ultrastar_header.language = language
    cover = basename_without_ext + " [CO].jpg"
    ultrastar_header.cover = (
        cover
        if os_helper.check_file_exists(os.path.join(song_output, cover))
        else None
    )

    real_bpm = get_bpm_from_file(ultrastar_audio_input_path)
    ultrastar_file_output = os.path.join(
        song_output, basename_without_ext + ".txt"
    )
    ultrastar_writer.create_ultrastar_txt_from_automation(
        transcribed_data,
        ultrastar_note_numbers,
        ultrastar_file_output,
        ultrastar_header,
        real_bpm,
    )
    if settings.create_karaoke:
        no_vocals_path = os.path.join(audio_separation_path, "no_vocals.wav")
        title = basename_without_ext + " [Karaoke]"
        ultrastar_header.title = title
        ultrastar_header.mp3 = title + ".mp3"
        karaoke_output_path = os.path.join(song_output, title)
        karaoke_audio_output_path = karaoke_output_path + ".mp3"
        convert_wav_to_mp3(no_vocals_path, karaoke_audio_output_path)
        karaoke_txt_output_path = karaoke_output_path + ".txt"
        ultrastar_writer.create_ultrastar_txt_from_automation(
            transcribed_data,
            ultrastar_note_numbers,
            karaoke_txt_output_path,
            ultrastar_header,
            real_bpm,
        )
    return real_bpm, ultrastar_file_output


def setup_audio_input_file():
    """Docstring"""
    basename = os.path.basename(settings.input_file_path)
    basename_without_ext = os.path.splitext(basename)[0]
    song_output = os.path.join(settings.output_file_path, basename_without_ext)
    song_output = get_unused_song_output_dir(song_output)
    os_helper.create_folder(song_output)
    os_helper.copy(settings.input_file_path, song_output)
    ultrastar_audio_input_path = os.path.join(song_output, basename)
    return basename_without_ext, song_output, ultrastar_audio_input_path


FILENAME_REPLACEMENTS = (('?:"', ""), ("<", "("), (">", ")"), ("/\\|*", "-"))


def sanitize_filename(fname: str) -> str:
    """Docstring"""
    for old, new in FILENAME_REPLACEMENTS:
        for char in old:
            fname = fname.replace(char, new)
    if fname.endswith("."):
        fname = fname.rstrip(" .")  # Windows does not like trailing periods
    return fname


def download_from_youtube():
    """Docstring"""
    title = get_youtube_title(settings.input_file_path)
    basename_without_ext = sanitize_filename(title)
    basename = basename_without_ext + ".mp3"
    song_output = os.path.join(settings.output_file_path, basename_without_ext)
    song_output = get_unused_song_output_dir(song_output)
    os_helper.create_folder(song_output)
    download_youtube_audio(
        settings.input_file_path, basename_without_ext, song_output
    )
    download_youtube_video(
        settings.input_file_path, basename_without_ext, song_output
    )
    download_youtube_thumbnail(
        settings.input_file_path, basename_without_ext, song_output
    )
    ultrastar_audio_input_path = os.path.join(song_output, basename)
    return basename_without_ext, song_output, ultrastar_audio_input_path


def parse_ultrastar_txt():
    """Docstring"""
    ultrastar_class = ultrastar_parser.parse_ultrastar_txt(
        settings.input_file_path
    )
    real_bpm = ultrastar_converter.ultrastar_bpm_to_real_bpm(
        float(ultrastar_class.bpm.replace(",", "."))
    )
    ultrastar_mp3_name = ultrastar_class.mp3
    basename_without_ext = os.path.splitext(ultrastar_mp3_name)[0]
    dirname = os.path.dirname(settings.input_file_path)
    ultrastar_audio_input_path = os.path.join(dirname, ultrastar_mp3_name)
    song_output = os.path.join(
        settings.output_file_path,
        ultrastar_class.artist + " - " + ultrastar_class.title,
    )
    song_output = get_unused_song_output_dir(song_output)
    os_helper.create_folder(song_output)
    return (
        basename_without_ext,
        real_bpm,
        song_output,
        ultrastar_audio_input_path,
        ultrastar_class,
    )


def create_midi_file(is_audio, real_bpm, song_output, ultrastar_class):
    """Docstring"""
    print(
        f"{PRINT_ULTRASTAR} Creating Midi with {print_blue_highlighted_text('pretty_midi')}"
    )
    if is_audio:
        voice_instrument = [
            midi_creator.convert_ultrastar_to_midi_instrument(ultrastar_class)
        ]
        midi_output = os.path.join(song_output, ultrastar_class.title + ".mid")
        midi_creator.instruments_to_midi(
            voice_instrument, real_bpm, midi_output
        )
    else:
        voice_instrument = [
            midi_creator.convert_ultrastar_to_midi_instrument(ultrastar_class)
        ]
        midi_output = os.path.join(song_output, ultrastar_class.title + ".mid")
        midi_creator.instruments_to_midi(
            voice_instrument, real_bpm, midi_output
        )


def pitch_audio(is_audio, transcribed_data, ultrastar_class):
    """Docstring"""
    # todo: chunk pitching as option?
    # midi_notes = pitch_each_chunk_with_crepe(chunk_folder_name)
    pitched_data = get_pitch_with_crepe_file(
        settings.mono_audio_path,
        settings.crepe_step_size,
        settings.crepe_model_capacity,
    )
    if is_audio:
        start_times = []
        end_times = []
        for i in enumerate(transcribed_data):
            start_times.append(transcribed_data[i].start)
            end_times.append(transcribed_data[i].end)
        midi_notes = create_midi_notes_from_pitched_data(
            start_times, end_times, pitched_data
        )

    else:
        midi_notes = create_midi_notes_from_pitched_data(
            ultrastar_class.startTimes, ultrastar_class.endTimes, pitched_data
        )
    ultrastar_note_numbers = convert_ultrastar_note_numbers(midi_notes)
    return midi_notes, pitched_data, ultrastar_note_numbers


def create_audio_chunks(
    cache_path,
    is_audio,
    transcribed_data,
    ultrastar_audio_input_path,
    ultrastar_class,
):
    """Docstring"""
    audio_chunks_path = os.path.join(
        cache_path, settings.audio_chunk_folder_name
    )
    os_helper.create_folder(audio_chunks_path)
    if is_audio:  # and csv
        csv_filename = os.path.join(audio_chunks_path, "_chunks.csv")
        export_chunks_from_transcribed_data(
            settings.mono_audio_path, transcribed_data, audio_chunks_path
        )
        export_transcribed_data_to_csv(transcribed_data, csv_filename)
    else:
        export_chunks_from_ultrastar_data(
            ultrastar_audio_input_path, ultrastar_class, audio_chunks_path
        )


def denoise_vocal_audio(basename_without_ext, cache_path):
    """Docstring"""
    denoised_path = os.path.join(
        cache_path, basename_without_ext + "_denoised.wav"
    )
    ffmpeg_reduce_noise(settings.mono_audio_path, denoised_path)
    settings.mono_audio_path = denoised_path


def main(argv):
    """Docstring"""
    init_settings(argv)
    run()
    # todo: cleanup
    sys.exit()


def init_settings(argv):
    """Docstring"""
    short = "hi:o:amv:"
    long = [
        "ifile=",
        "ofile=",
        "crepe=",
        "vosk=",
        "whisper=",
        "hyphenation=",
        "disable_separation=",
        "disable_karaoke=",
        "create_audio_chunks=",
    ]
    opts, args = getopt.getopt(argv, short, long)
    if len(opts) == 0:
        print_help()
        sys.exit()
    for opt, arg in opts:
        if opt == "-h":
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
            settings.transcriber = "vosk"
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
    if settings.output_file_path == "":
        if settings.input_file_path.startswith("https:"):
            dirname = os.getcwd()
        else:
            dirname = os.path.dirname(settings.input_file_path)
        settings.output_file_path = os.path.join(dirname, "output")
    settings.device = get_available_device()


if __name__ == "__main__":
    main(sys.argv[1:])
