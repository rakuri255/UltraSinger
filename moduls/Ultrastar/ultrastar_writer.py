from moduls.Ultrastar.ultrastar_converter import real_bpm_to_ultrastar_bpm, second_to_beat
from moduls.Log import PRINT_ULTRASTAR
import re


def create_ultrastar_txt_from_automation(transcribed_data, note_numbers, ultrastar_file_output, title, bpm=120):
    print(PRINT_ULTRASTAR + " Creating {} from transcription.".format(ultrastar_file_output))

    # todo: Optimize multiplication
    multiplication = 32
    with open(ultrastar_file_output, 'w') as f:
        gap = transcribed_data[0].start
        ultrastar_bpm = real_bpm_to_ultrastar_bpm(bpm) * multiplication

        f.write('#ARTIST:' + title + '\n')
        f.write('#TITLE:' + title + '\n')
        f.write('#CREATOR:UltraSinger' + '\n')
        f.write('#FIXER:YOUR NAME' + '\n')
        f.write('#MP3:' + title + '.mp3\n')
        f.write('#VIDEO:' + title + '.mp4\n')
        f.write('#BPM:' + str(ultrastar_bpm) + '\n')  # not the real BPM!
        f.write('#GAP:' + str(int(gap * 1000)) + '\n')

        # Write the singing part
        for i in range(len(transcribed_data)):
            start_time = (transcribed_data[i].start - gap) * multiplication
            end_time = (transcribed_data[i].end - transcribed_data[i].start) * multiplication
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
            f.write(transcribed_data[i].word)
            f.write('\n')

            # detect silence between words
            if i < len(transcribed_data) - 1:
                silence = transcribed_data[i + 1].start - transcribed_data[i].end
            else:
                silence = 0

            if silence > 0.3:
                # - 10
                # '-' end of current sing part
                # 'n1' show next at time in real beat
                f.write('- ')
                show_next = second_to_beat(transcribed_data[i].end - gap, bpm) * multiplication
                f.write(str(round(show_next)))
                f.write('\n')
        f.write('E')


def create_repitched_txt_from_ultrastar_data(input_file, note_numbers, output_repitched_ultrastar):
    # todo: just add '_repitched' to input_file
    print(PRINT_ULTRASTAR + " Creating repitched ultrastar txt -> {}_repitch.txt".format(input_file))

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
