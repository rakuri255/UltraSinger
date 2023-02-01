import getopt
import os
import sys
import Levenshtein
import librosa
import numpy as np
import math

from moduls.Audio.vocal_chunks import export_chunks_from_ultrastar_data, convert_audio_to_mono_wav
from moduls.Midi import midi_creator
from moduls.Pitcher.pitcher import get_frequency_with_high_confidence, get_pitch_with_crepe_file
from moduls.Ultrastar import ultrastar_parser, ultrastar_converter, ultrastar_writer
from moduls.Speech_Recognition.Vosk import transcribe_with_vosk, export_vosk_data_to_csv
from moduls.Audio.vocal_chunks import export_chunks_from_vosk_data
from moduls.os_helper import create_folder
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


def create_midi_notes_from_pitched_data(start_times, end_times, pitched_data):
    print("Create midi notes from pitched data")

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
    UltraSinger.py [opt] [mode] [rec model] [pitcher] [extra]
    
    [opt]
    -h      This help text.
    
    -i      Ultrastar.txt
            audio like .mp3, .wav, youtube link
    
    -o      Output folder
    
    [mode]
    ## INPUT is audio ##
    -a      Is default
            Creates all
    -u      Create ultrastar txt file
    -m      Create midi file
    -s      Create sheet file
    
    ## INPUT is ultrastar.txt ##
    -a    Is default
            Creates all
    -r      repitch Ultrastar.txt (input has to be audio)
    -p    Check pitch of Ultrastar.txt input
    -m      Create midi file

    [rec model]
    -v     vosk model path
      
    [extra]
    -k      Keep audio chunks
    
    [pitcher]
    -crepe  default
    '''
    print(m)


def do_ultrastar_stuff():
    # Parse Ultrastar txt
    ultrastar_class = ultrastar_parser.parse_ultrastar_txt(settings.input_file_path)
    real_bpm = ultrastar_converter.ultrastar_bpm_to_real_bpm(float(ultrastar_class.bpm.replace(',', '.')))

    # todo: duplicate code
    dirname = os.path.dirname(settings.input_file_path)
    settings.mono_audio_path = 'output/mono.wav'
    create_folder('output')
    convert_audio_to_mono_wav(dirname + '/' + ultrastar_class.mp3, settings.mono_audio_path)

    # todo: input folder
    ultrastar_audio_input_path = 'input/' + ultrastar_class.mp3.replace('\n', '')
    # todo: here we have to remove all silance, and dont need to store it!
    # ultrastar_class = remove_silence_from_ultrastar_data(ultrastar_audio_input_path, ultrastar_class)

    if settings.create_audio_chunks:
        create_folder(settings.audio_chunk_folder_name)
        export_chunks_from_ultrastar_data(ultrastar_audio_input_path, ultrastar_class, settings.audio_chunk_folder_name)

    # Pitch the audio
    # todo: chunk pitching as option?
    # midi_notes = pitch_each_chunk_with_crepe(chunk_folder_name)
    pitched_data = get_pitch_with_crepe_file(settings.mono_audio_path, settings.crepe_step_size, settings.crepe_model_capacity)

    midi_notes = create_midi_notes_from_pitched_data(ultrastar_class.startTimes, ultrastar_class.endTimes, pitched_data)
    ultrastar_note_numbers = convert_ultrastar_note_numbers(midi_notes)

    # Create new repitched ultrastar txt
    # todo: output
    output_repitched_ultrastar = 'output/ultrastar_repitch.txt'
    ultrastar_writer.create_repitched_txt_from_ultrastar_data(settings.input_file_path, ultrastar_note_numbers,
                                                              output_repitched_ultrastar)

    if settings.create_midi:
        # todo: remove .txt
        voice_instrument = [midi_creator.convert_ultrastar_to_midi_instrument(ultrastar_class)]
        # todo: output
        midi_output = settings.input_file_path.replace('.txt', '.mid')
        midi_output = midi_output.replace('input', 'output')
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


def do_audio_stuff():
    output_mono_audio = 'output/mono.wav'
    create_folder('output')
    # todo: different sample rates for different models
    convert_audio_to_mono_wav(settings.input_file_path, output_mono_audio)

    # Audio transcription
    vosk_transcribed_data = transcribe_with_vosk(output_mono_audio, settings.audio_chunk_folder_name, settings.vosk_model_path)

    if settings.create_audio_chunks:
        create_folder(settings.audio_chunk_folder_name)
        csv_filename = settings.audio_chunk_folder_name + "/_chunks.csv"
        export_chunks_from_vosk_data(output_mono_audio, vosk_transcribed_data, settings.audio_chunk_folder_name)
        export_vosk_data_to_csv(vosk_transcribed_data, csv_filename)

    # todo: do we need to correct words?
    # lyric = 'input/faber_lyric.txt'
    # --corrected_words = correct_words(vosk_speech, lyric)

    # Pitch detection
    # todo: chunk pitching as option?
    # midi_notes = pitch_each_chunk_with_crepe(chunk_folder_name)
    pitched_data = get_pitch_with_crepe_file(output_mono_audio, settings.crepe_step_size, settings.crepe_model_capacity)

    start_times = []
    end_times = []
    for i in range(len(vosk_transcribed_data)):
        start_times.append(vosk_transcribed_data[i].start)
        end_times.append(vosk_transcribed_data[i].end)

    midi_notes = create_midi_notes_from_pitched_data(start_times, end_times, pitched_data)
    ultrastar_note_numbers = convert_ultrastar_note_numbers(midi_notes)

    if settings.create_plot:
        plot(settings.input_file_path, vosk_transcribed_data, midi_notes)

    # Ultrastar txt creation
    real_bpm = get_bpm_from_file(settings.input_file_path)
    # todo: filename from audio / yt title
    ultrastar_file_output = 'output/ultrastar.txt'
    ultrastar_writer.create_txt_from_transcription(vosk_transcribed_data, ultrastar_note_numbers, ultrastar_file_output,
                                                   real_bpm)

    if settings.create_midi:
        ultrastar_class = ultrastar_parser.parse_ultrastar_txt(ultrastar_file_output)

        # todo: remove .txt
        voice_instrument = [midi_creator.convert_ultrastar_to_midi_instrument(ultrastar_class)]
        # todo: output. remove file extansion
        midi_output = settings.input_file_path.replace('.wav', '.mid')
        midi_output = midi_output.replace('input', 'output')
        midi_creator.instruments_to_midi(voice_instrument, real_bpm, midi_output)


def main(argv):
    short = "hi:o:amv:"
    long = ["ifile=", "ofile="]

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
            settings.audio_chunk_folder_name = arg
        elif opt in ("-v"):
            settings.vosk_model_path = arg

    if ".txt" in settings.input_file_path:
        do_ultrastar_stuff()
    else:
        do_audio_stuff()

    sys.exit()

    # todo: cleanup


if __name__ == "__main__":
    main(sys.argv[1:])
