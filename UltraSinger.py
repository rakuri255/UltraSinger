import getopt
import copy
import os
import sys
import Levenshtein
import librosa
import numpy as np
import math
import shutil
import re

from moduls.Audio.vocal_chunks import export_chunks_from_ultrastar_data, convert_audio_to_mono_wav, \
    export_chunks_from_transcribed_data, remove_silence_from_transcribtion_data
from moduls.Audio.youtube import download_youtube_video, download_youtube_audio, get_youtube_title
from moduls.Midi import midi_creator
from moduls.Pitcher.pitcher import get_frequency_with_high_confidence, get_pitch_with_crepe_file
from moduls.Ultrastar import ultrastar_parser, ultrastar_converter, ultrastar_writer
from moduls.Speech_Recognition.Vosk import transcribe_with_vosk, export_transcribed_data_to_csv
from moduls.Speech_Recognition.hyphenation import hyphenation, language_check
from moduls.Speech_Recognition.Whisper import transcribe_with_whisper
from moduls.os_helper import create_folder
from moduls.Audio.separation import separate_audio
from matplotlib import pyplot as plt
from collections import Counter
from Settings import Settings

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


def convert_frequencies_to_notes(frequency):
    notes = []
    for f in frequency:
        notes.append(librosa.hz_to_note(float(f)))
    return notes


def most_frequent(ar):
    return Counter(ar).most_common(1)


def convert_ultrastar_note_numbers(midi_notes):
    print("Creating Ultrastar notes from midi data")

    ultrastar_note_numbers = []
    for i in range(len(midi_notes)):
        note_number_librosa = librosa.note_to_midi(midi_notes[i])
        pitch = ultrastar_converter.midi_note_to_ultrastar_note(note_number_librosa)
        ultrastar_note_numbers.append(pitch)
        # todo: Progress?
        # print("Note: " + midi_notes[i] + " midi_note: " + str(note_number_librosa) + ' pitch: ' + str(pitch))
    return ultrastar_note_numbers


def pitch_each_chunk_with_crepe(directory):
    print("Pitching each chunk with crepe.")

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


def find_nearest_index(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


def add_hyphen_to_data(transcribed_data, hyphen_words):
    data = []

    for i in range(len(transcribed_data)):
        if not hyphen_words[i]:
            data.append(transcribed_data[i])
        else:
            chunk_duration = transcribed_data[i].end - transcribed_data[i].start
            chunk_duration = chunk_duration / (len(hyphen_words[i]))

            for j in range(len(hyphen_words[i])):
                dup = copy.copy(transcribed_data[i])
                dup.start = transcribed_data[i].start + chunk_duration * j
                dup.end = transcribed_data[i].end - chunk_duration * (len(hyphen_words[i]) - 1 - j)
                dup.word = hyphen_words[i][j]
                dup.is_hyphen = True
                data.append(dup)

    return data


def create_midi_notes_from_pitched_data(start_times, end_times, pitched_data):
    print("Creating midi notes from pitched data")

    midi_notes = []

    for i in range(len(start_times)):
        start_time = start_times[i]
        end_time = end_times[i]

        s = find_nearest_index(pitched_data.times, start_time)
        e = find_nearest_index(pitched_data.times, end_time)

        if s == e:
            f = [pitched_data.frequencies[s]]
            c = [pitched_data.confidence[s]]
        else:
            f = pitched_data.frequencies[s:e]
            c = pitched_data.confidence[s:e]

        conf_f = get_frequency_with_high_confidence(f, c)

        notes = convert_frequencies_to_notes(conf_f)

        note = most_frequent(notes)[0][0]

        midi_notes.append(note)
        # todo: Progress?
        # print(filename + " f: " + str(mean))
    return midi_notes


def get_bpm_from_data(data, sr):
    onset_env = librosa.onset.onset_strength(y=data, sr=sr)
    wav_tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

    print("BPM is " + str(wav_tempo[0]))
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


def do_ultrastar_stuff():
    # Parse Ultrastar txt
    ultrastar_class = ultrastar_parser.parse_ultrastar_txt(settings.input_file_path)
    real_bpm = ultrastar_converter.ultrastar_bpm_to_real_bpm(float(ultrastar_class.bpm.replace(',', '.')))

    # todo: duplicate code
    ultrastar_mp3_name = ultrastar_class.mp3
    basename_without_ext = os.path.splitext(ultrastar_mp3_name)[0]
    dirname = os.path.dirname(settings.input_file_path)

    ultrastar_audio_input_path = dirname + '/' + ultrastar_mp3_name
    song_output = settings.output_file_path + '/' + ultrastar_class.artist + ' - ' + ultrastar_class.title
    cache_path = song_output + '/cache'
    settings.mono_audio_path = cache_path + '/' + basename_without_ext + '.wav'
    create_folder(cache_path)

    audio_separation_path = cache_path + "/separated/htdemucs/" + basename_without_ext
    if settings.use_separated_vocal or settings.create_karaoke:
        separate_audio(ultrastar_audio_input_path, cache_path)

    if settings.use_separated_vocal:
        vocals_path = audio_separation_path + "/vocals.wav"
        convert_audio_to_mono_wav(vocals_path, settings.mono_audio_path)
    else:
        convert_audio_to_mono_wav(ultrastar_audio_input_path, settings.mono_audio_path)

    # todo: here we have to remove all silance, and dont need to store it!
    # ultrastar_class = remove_silence_from_ultrastar_data(ultrastar_audio_input_path, ultrastar_class)

    if settings.create_audio_chunks:
        audio_chunks_path = cache_path + '/' + settings.audio_chunk_folder_name
        create_folder(audio_chunks_path)
        export_chunks_from_ultrastar_data(ultrastar_audio_input_path, ultrastar_class, audio_chunks_path)

    # Pitch the audio
    # todo: chunk pitching as option?
    # midi_notes = pitch_each_chunk_with_crepe(chunk_folder_name)
    pitched_data = get_pitch_with_crepe_file(settings.mono_audio_path, settings.crepe_step_size,
                                             settings.crepe_model_capacity)

    midi_notes = create_midi_notes_from_pitched_data(ultrastar_class.startTimes, ultrastar_class.endTimes, pitched_data)
    ultrastar_note_numbers = convert_ultrastar_note_numbers(midi_notes)

    # Create new repitched ultrastar txt
    output_repitched_ultrastar = song_output + '/' + ultrastar_class.title + '.txt'
    ultrastar_writer.create_repitched_txt_from_ultrastar_data(settings.input_file_path, ultrastar_note_numbers,
                                                              output_repitched_ultrastar)

    if settings.create_midi:
        voice_instrument = [midi_creator.convert_ultrastar_to_midi_instrument(ultrastar_class)]
        midi_output = song_output + '/' + ultrastar_class.title + '.mid'
        midi_creator.instruments_to_midi(voice_instrument, real_bpm, midi_output)


def plot(input_file, vosk_transcribed_data, midi_notes):
    pitched_data = get_pitch_with_crepe_file(input_file, settings.crepe_step_size, settings.crepe_model_capacity)
    t, f, c = get_confidence(pitched_data, 0.4)

    plt.ylim(0, 600)
    plt.xlim(0, 50)
    plt.plot(t, f, linewidth=0.1)
    plt.savefig('test/crepe_0.4.png')

    for i in range(len(vosk_transcribed_data)):
        nf = librosa.note_to_hz(midi_notes[i])
        plt.plot([vosk_transcribed_data[i].start, vosk_transcribed_data[i].end], [nf, nf], linewidth=1, alpha=0.5)
    plt.savefig('test/pit.png', dpi=2000)


def remove_unecessary_punctuations(transcribed_data):
    punctuation = ".,"
    for i in range(len(transcribed_data)):
        transcribed_data[i].word = transcribed_data[i].word.translate({ord(i): None for i in punctuation})


def hyphenate_each_word(language, transcribed_data):
    print("Hyphenation each word")
    lang_region = language_check(language)
    hyphenated_word = []
    for i in range(len(transcribed_data)):
        hyphenated_word.append(hyphenation(transcribed_data[i].word, lang_region))
    return hyphenated_word


def do_audio_stuff():
    # Youtube
    if settings.input_file_path.startswith('https:'):
        title = get_youtube_title(settings.input_file_path)

        basename_without_ext = re.sub('[^A-Za-z0-9. _-]+', '', title).strip()
        basename = basename_without_ext + '.mp3'

        song_output = settings.output_file_path + '/' + basename_without_ext
        create_folder(song_output)
        download_youtube_audio(settings.input_file_path, basename_without_ext, song_output)
        download_youtube_video(settings.input_file_path, basename_without_ext, song_output)

    else:
        basename = os.path.basename(settings.input_file_path)
        basename_without_ext = os.path.splitext(basename)[0]
        song_output = settings.output_file_path + '/' + basename_without_ext
        create_folder(song_output)
        #todo: to OS helper
        shutil.copy(settings.input_file_path,
                    song_output)  # dst can be a folder; use shutil.copy2() to preserve timestamp

    ultrastar_audio_input_path = song_output + '/' + basename
    cache_path = song_output + '/cache'
    settings.mono_audio_path = cache_path + '/' + basename_without_ext + '.wav'
    create_folder(cache_path)

    audio_separation_path = cache_path + "/separated/htdemucs/" + basename_without_ext
    if settings.use_separated_vocal or settings.create_karaoke:
        separate_audio(ultrastar_audio_input_path, cache_path)

    if settings.use_separated_vocal:
        vocals_path = audio_separation_path + "/vocals.wav"
        convert_audio_to_mono_wav(vocals_path, settings.mono_audio_path)
    else:
        convert_audio_to_mono_wav(ultrastar_audio_input_path, settings.mono_audio_path)

    # Audio transcription
    if settings.transcriber == "whisper":
        transcribed_data, language = transcribe_with_whisper(settings.mono_audio_path, settings.whisper_model)
    else:  # vosk
        transcribed_data = transcribe_with_vosk(settings.mono_audio_path, settings.vosk_model_path)
        # todo: make language selectable
        language = 'en'

    remove_unecessary_punctuations(transcribed_data)

    transcribed_data = remove_silence_from_transcribtion_data(settings.mono_audio_path, transcribed_data)

    if settings.hyphenation:
        hyphen_words = hyphenate_each_word(language, transcribed_data)
        transcribed_data = add_hyphen_to_data(transcribed_data, hyphen_words)

    if settings.create_audio_chunks:
        audio_chunks_path = cache_path + '/' + settings.audio_chunk_folder_name
        create_folder(audio_chunks_path)
        csv_filename = audio_chunks_path + "/_chunks.csv"
        export_chunks_from_transcribed_data(settings.mono_audio_path, transcribed_data, audio_chunks_path)
        export_transcribed_data_to_csv(transcribed_data, csv_filename)

    # todo: do we need to correct words?
    # lyric = 'input/faber_lyric.txt'
    # --corrected_words = correct_words(vosk_speech, lyric)

    # Pitch detection
    # todo: chunk pitching as option?
    # midi_notes = pitch_each_chunk_with_crepe(chunk_folder_name)
    pitched_data = get_pitch_with_crepe_file(settings.mono_audio_path, settings.crepe_step_size,
                                             settings.crepe_model_capacity)

    start_times = []
    end_times = []
    for i in range(len(transcribed_data)):
        start_times.append(transcribed_data[i].start)
        end_times.append(transcribed_data[i].end)
    midi_notes = create_midi_notes_from_pitched_data(start_times, end_times, pitched_data)

    ultrastar_note_numbers = convert_ultrastar_note_numbers(midi_notes)

    if settings.create_plot:
        plot(ultrastar_audio_input_path, transcribed_data, midi_notes)

    # Ultrastar txt creation
    real_bpm = get_bpm_from_file(ultrastar_audio_input_path)
    ultrastar_file_output = song_output + '/' + basename_without_ext + '.txt'
    ultrastar_writer.create_ultrastar_txt_from_automation(transcribed_data, ultrastar_note_numbers,
                                                          ultrastar_file_output,
                                                          basename_without_ext, real_bpm)

    if settings.create_midi:
        ultrastar_class = ultrastar_parser.parse_ultrastar_txt(ultrastar_file_output)
        voice_instrument = [midi_creator.convert_ultrastar_to_midi_instrument(ultrastar_class)]
        midi_output = song_output + '/' + ultrastar_class.title + '.mid'
        midi_creator.instruments_to_midi(voice_instrument, real_bpm, midi_output)


def main(argv):
    short = "hi:o:amv:"
    long = ["ifile=", "ofile=", "crepe_model=", "vosk=", "whisper=", "hyphenation=", "disable_separation="]

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

    if settings.output_file_path == '':
        if settings.input_file_path.startswith('https:'):
            dirname = os.getcwd()
        else:
            dirname = os.path.dirname(settings.input_file_path)
        settings.output_file_path = dirname + '/output'

    if ".txt" in settings.input_file_path:
        do_ultrastar_stuff()
    else:
        do_audio_stuff()

    sys.exit()

    # todo: cleanup


if __name__ == "__main__":
    main(sys.argv[1:])
