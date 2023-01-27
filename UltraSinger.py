import getopt
import os
import statistics
import sys
import Levenshtein
import librosa

from moduls.Audio.vocal_chunks import export_chunks_from_ultrastar_data, convert_audio_to_mono_wav
from moduls.Midi import midi_creator
from moduls.Pitcher import pitcher
from moduls.Pitcher.pitcher import get_frequency_with_high_confidance
from moduls.Ultrastar import ultrastar_parser, ultrastar_converter, ultrastar_writer
from moduls.Speech_Recognition.Vosk import transcribe_with_vosk


def convert_frequencies_to_notes(frequency):
    notes = []
    for f in frequency:
        notes.append(librosa.hz_to_note(float(f)))
    return notes


def pitch_each_chunk_with_crepe(directory):
    print("Pitching each chunk with crepe.")
    _pitcher = pitcher

    frequency = []
    for filename in sorted([f for f in os.listdir(directory) if f.endswith('.wav')],
                           key=lambda x: int(x.split("_")[1])):
        filepath = os.path.join(directory, filename)
        t, f, c = _pitcher.get_pitch_with_crepe_file(filepath, 100)
        conf_f = get_frequency_with_high_confidance(f, c)
        mean = statistics.mean(conf_f)
        frequency.append(mean)
        # todo: Progress?
        # print(filename + " f: " + str(mean))

    midi_notes = convert_frequencies_to_notes(frequency)
    note_numbers = []
    for i in range(len(midi_notes)):
        note_number_librosa = librosa.note_to_midi(midi_notes[i])
        pitch = ultrastar_converter.midi_note_to_ultrastar_note(note_number_librosa)
        note_numbers.append(pitch)
        # todo: Progress?
        # print("Note: " + midi_notes[i] + " midi_note: " + str(note_number_librosa) + ' pitch: ' + str(pitch))

    return note_numbers


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
    export_chunks_from_ultrastar_data(ultrastar_audio_input_path, ultrastar_class, chunk_folder_name)

    # Pitch the audio
    note_numbers = pitch_each_chunk_with_crepe(chunk_folder_name)

    # Create new repitched ultrastar txt
    # todo: output
    output_repitched_ultrastar = 'output/ultrastar_repitch.txt'
    ultrastar_writer.create_repitched_txt_from_ultrastar_data(input_file, note_numbers, output_repitched_ultrastar)

    if do_create_midi:
        # todo: remove .txt
        voice_instrument = [midi_creator.convert_ultrastar_to_midi_instrument(ultrastar_class)]
        # todo: output
        midi_output = input_file.replace('.txt', '.mid')
        midi_output = midi_output.replace('input', 'output')
        midi_creator.instruments_to_midi(voice_instrument, real_bpm, midi_output)


def do_audio_stuff(input_file, chunk_folder_name, model_path, do_create_midi):
    # Audio transcription
    # todo: convert to mono wav
    list_of_vosk_words = transcribe_with_vosk(input_file, chunk_folder_name, model_path)
    # todo: do we need to correct words?
    # lyric = 'input/faber_lyric.txt'
    # --corrected_words = correct_words(vosk_speech, lyric)

    # Pitch detection
    note_numbers = pitch_each_chunk_with_crepe(chunk_folder_name)

    # Ultrastar txt creation
    real_bpm = get_bpm_from_file(input_file)
    # todo: filename from audio / yt title
    ultrastar_file_output = 'output/ultrastar.txt'
    ultrastar_writer.create_txt_from_transcription(list_of_vosk_words, note_numbers, ultrastar_file_output, real_bpm)

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

    # todo: args for disable
    output_mono_audio = 'output/mono.wav'
    convert_audio_to_mono_wav(input_file, output_mono_audio)
    input_file = output_mono_audio

    if ".txt" in input_file:
        do_ultrastar_stuff(input_file, chunk_folder_name, do_create_midi)
    else:
        do_audio_stuff(input_file, chunk_folder_name, model_path, do_create_midi)

    sys.exit()

    # todo: cleanup


if __name__ == "__main__":
    main(sys.argv[1:])
