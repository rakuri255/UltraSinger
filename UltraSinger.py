import getopt
import os
import sys
import Levenshtein
import librosa

from moduls.Audio.vocal_chunks import export_chunks_from_ultrastar_data, convert_audio_to_mono_wav
from moduls.Midi import midi_creator
from moduls.Pitcher.pitcher import get_frequency_with_high_confidance, get_pitch_with_crepe_file
from moduls.Ultrastar import ultrastar_parser, ultrastar_converter, ultrastar_writer
from moduls.Speech_Recognition.Vosk import transcribe_with_vosk
from moduls.os_helper import create_folder
from matplotlib import pyplot as plt
from collections import Counter


def get_confidance(t, f, c, threashold):
    # todo: replace get_frequency_with_high_conf from pitcher
    conf_t = []
    conf_f = []
    conf_c = []
    for i in range(len(t)):
        if c[i] > threashold:
            conf_t.append(t[i])
            conf_f.append(f[i])
            conf_c.append(c[i])
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
        t, f, c = get_pitch_with_crepe_file(filepath, 10)
        conf_f = get_frequency_with_high_confidance(f, c)

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


def do_ultrastar_stuff(input_file, chunk_folder_name, do_create_midi):
    # Parse Ultrastar txt
    ultrastar_class = ultrastar_parser.parse_ultrastar_txt(input_file)
    real_bpm = ultrastar_converter.ultrastar_bpm_to_real_bpm(float(ultrastar_class.bpm.replace(',', '.')))

    # Create audio chunks from timing table
    # todo: input folder
    ultrastar_audio_input_path = 'input/' + ultrastar_class.mp3.replace('\n', '')
    # todo: check if chunk folder is empty
    # todo: here we have to remove all silance, and dont need to store it!
    # ultrastar_class = remove_silence_from_ultrastar_data(ultrastar_audio_input_path, ultrastar_class)
    export_chunks_from_ultrastar_data(ultrastar_audio_input_path, ultrastar_class, chunk_folder_name)

    # Pitch the audio
    midi_notes = pitch_each_chunk_with_crepe(chunk_folder_name)
    ultrastar_note_numbers = convert_ultrastar_note_numbers(midi_notes)

    # Create new repitched ultrastar txt
    # todo: output
    output_repitched_ultrastar = 'output/ultrastar_repitch.txt'
    ultrastar_writer.create_repitched_txt_from_ultrastar_data(input_file, ultrastar_note_numbers,
                                                              output_repitched_ultrastar)

    if do_create_midi:
        # todo: remove .txt
        voice_instrument = [midi_creator.convert_ultrastar_to_midi_instrument(ultrastar_class)]
        # todo: output
        midi_output = input_file.replace('.txt', '.mid')
        midi_output = midi_output.replace('input', 'output')
        midi_creator.instruments_to_midi(voice_instrument, real_bpm, midi_output)


def plot(input_file, vosk_transcribed_data, midi_notes):
    t, f, c = get_pitch_with_crepe_file(input_file, 10)
    t, f, c = get_confidance(t, f, c, 0.4)

    plt.ylim(0, 600)
    plt.xlim(0, 50)
    plt.plot(t, f, linewidth=0.1)
    plt.savefig('test/crepe_0.4.png')

    for i in range(len(vosk_transcribed_data)):
        nf = librosa.note_to_hz(midi_notes[i])
        plt.plot([vosk_transcribed_data[i].start, vosk_transcribed_data[i].end], [nf, nf], linewidth=1, alpha=0.5)
    plt.savefig('test/pit.png', dpi=2000)


def do_audio_stuff(input_file, chunk_folder_name, model_path, do_create_midi):
    # Audio transcription
    # todo: convert to mono wav
    vosk_transcribed_data = transcribe_with_vosk(input_file, chunk_folder_name, model_path)
    # todo: do we need to correct words?
    # lyric = 'input/faber_lyric.txt'
    # --corrected_words = correct_words(vosk_speech, lyric)

    # Pitch detection
    midi_notes = pitch_each_chunk_with_crepe(chunk_folder_name)
    ultrastar_note_numbers = convert_ultrastar_note_numbers(midi_notes)

    export_plot = False
    if export_plot:
        plot(input_file, vosk_transcribed_data, midi_notes)

    # Ultrastar txt creation
    real_bpm = get_bpm_from_file(input_file)
    # todo: filename from audio / yt title
    ultrastar_file_output = 'output/ultrastar.txt'
    ultrastar_writer.create_txt_from_transcription(vosk_transcribed_data, ultrastar_note_numbers, ultrastar_file_output,
                                                   real_bpm)

    if do_create_midi:
        ultrastar_class = ultrastar_parser.parse_ultrastar_txt(ultrastar_file_output)

        # todo: remove .txt
        voice_instrument = [midi_creator.convert_ultrastar_to_midi_instrument(ultrastar_class)]
        # todo: output. remove file extansion
        midi_output = input_file.replace('.wav', '.mid')
        midi_output = midi_output.replace('input', 'output')
        midi_creator.instruments_to_midi(voice_instrument, real_bpm, midi_output)


def main(argv):
    input_file = ''
    chunk_folder_name = 'output/audio-chunks'
    model_path = ''  # "models/vosk-model-small-en-us-0.15"

    short = "hi:o:amv:"
    long = ["ifile=", "ofile="]

    do_create_midi = True

    opts, args = getopt.getopt(argv, short, long)

    if len(opts) == 0:
        print_help()
        sys.exit()

    for opt, arg in opts:
        if opt == '-h':
            print_help()
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-o", "--ofile"):
            chunk_folder_name = arg
        elif opt in ("-v"):
            model_path = arg

    if ".txt" in input_file:
        do_ultrastar_stuff(input_file, chunk_folder_name, do_create_midi)
    else:
        # todo: args for disable
        output_mono_audio = 'output/mono.wav'
        create_folder('output')
        create_folder(chunk_folder_name)
        # todo: different sample rates for different models
        convert_audio_to_mono_wav(input_file, output_mono_audio)
        input_file = output_mono_audio
        do_audio_stuff(input_file, chunk_folder_name, model_path, do_create_midi)

    sys.exit()

    # todo: cleanup


if __name__ == "__main__":
    main(sys.argv[1:])
