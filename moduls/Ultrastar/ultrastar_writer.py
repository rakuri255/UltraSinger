from moduls.Ultrastar.ultrastar_converter import real_bpm_to_ultrastar_bpm, second_to_beat
import re


def create_txt_from_transcription(list_of_vosk_words, note_numbers, ultrastar_file_output, bpm=120):
    print("Creating {} from transcription.".format(ultrastar_file_output))

    # todo: Optimize multiplication
    multiplication = 32
    with open(ultrastar_file_output, 'w') as f:
        gap = list_of_vosk_words[0].start
        ultrastar_bpm = real_bpm_to_ultrastar_bpm(bpm) * multiplication

        # todo: #MP3, #ARTIST, #TITLE
        f.write('#AUTHOR: UltraSinger' + '\n')
        f.write('#BPM: ' + str(ultrastar_bpm) + '\n')  # not the real BPM!
        f.write('#GAP: ' + str(int(gap * 1000)) + '\n')

        # Write the singing part
        for i in range(len(list_of_vosk_words)):
            start_time = (list_of_vosk_words[i].start - gap) * multiplication
            end_time = (list_of_vosk_words[i].end - list_of_vosk_words[i].start) * multiplication
            start_beat = second_to_beat(start_time, bpm)
            duration = second_to_beat(end_time, bpm)

            # : 10 10 10 w
            # ':'   start midi part
            # 'n1'  start at real beat
            # 'n2'  duration at real beat
            # 'n3'  pitch where 0 == C4
            # 'w'   lyric
            f.write(': ')
            f.write(str(round(start_beat)) + ' ')
            f.write(str(round(duration)) + ' ')
            f.write(str(note_numbers[i]) + ' ')
            f.write(list_of_vosk_words[i].word + ' ')
            f.write('\n')

            # detect silence between words
            if i < len(list_of_vosk_words) - 1:
                silence = list_of_vosk_words[i + 1].start - list_of_vosk_words[i].end
            else:
                silence = 0

            if silence > 0.3:
                # - 10
                # '-' end of current sing part
                # 'n1' show next at time in real beat
                f.write('- ')
                show_next = second_to_beat(list_of_vosk_words[i].end - gap, bpm) * multiplication
                f.write(str(round(show_next)))
                f.write('\n')


def create_repitched_txt_from_ultrastar_data(input_file, note_numbers, output_repitched_ultrastar):
    # todo: just add '_repitched' to input_file
    print("Creating repitched ultrastar txt -> {}_repitch.txt".format(input_file))

    # todo: to reader
    file = open(input_file, 'r')
    txt = file.readlines()

    i = 0
    # todo: just add '_repitched' to input_file
    with open(output_repitched_ultrastar, 'w') as f:
        for line in txt:
            if line.startswith(':'):
                parts = re.findall(r'\S+|\s+', line)
                # between are whitespaces
                # [0] :
                # [2] start beat
                # [4] duration
                # [6] pitch
                # [8] word
                parts[6] = str(note_numbers[i])
                delimiter = ''
                f.write(delimiter.join(parts))
                i += 1
            else:
                f.write(line)


class UltraStarWriter:
    pass
