from moduls.Ultrastar.ultrastar_txt import UltrastarTxt
from moduls.Ultrastar.ultrastar_converter import get_end_time_from_ultrastar, get_start_time_from_ultrastar


def parse_ultrastar_txt(input_file):
    print("Parse ultrastar txt -> {}".format(input_file))

    file = open(input_file, 'r')
    txt = file.readlines()

    ultrastar_class = UltrastarTxt()
    count = 0

    # Strips the newline character
    for line in txt:
        count += 1
        if line.startswith('#'):
            if line.startswith('#ARTIST'):
                ultrastar_class.artist = line.split(':')[1].replace('\n', '')
            elif line.startswith('#TITLE'):
                ultrastar_class.title = line.split(':')[1].replace('\n', '')
            elif line.startswith('#MP3'):
                ultrastar_class.mp3 = line.split(':')[1].replace('\n', '')
            elif line.startswith('#GAP'):
                ultrastar_class.gap = line.split(':')[1].replace('\n', '')
            elif line.startswith('#BPM'):
                ultrastar_class.bpm = line.split(':')[1].replace('\n', '')
        elif line.startswith(('F', ':', '*', 'R', 'G')):
            parts = line.split()
            # [0] F : * R G
            # [1] start beat
            # [2] duration
            # [3] pitch
            # [4] word
            if parts[0] and parts[1] and parts[2] and parts[3] and parts[4]:
                ultrastar_class.noteType.append(parts[0])
                ultrastar_class.startBeat.append(parts[1])
                ultrastar_class.durations.append(parts[2])
                ultrastar_class.pitches.append(parts[3])
                ultrastar_class.words.append(parts[4])
                # do always as last
                pos = len(ultrastar_class.startBeat) - 1
                ultrastar_class.startTimes.append(get_start_time_from_ultrastar(ultrastar_class, pos))
                ultrastar_class.endTimes.append(get_end_time_from_ultrastar(ultrastar_class, pos))
                # todo: Progress?
                # print(parts[4])

    return ultrastar_class


