from moduls.Ultrastar.ultrastar_txt import UltrastarTxt


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
                ultrastar_class.artist = line.split(':')[1]
            elif line.startswith('#MP3'):
                ultrastar_class.mp3 = line.split(':')[1]
            elif line.startswith('#GAP'):
                ultrastar_class.gap = line.split(':')[1]
            elif line.startswith('#BPM'):
                ultrastar_class.bpm = line.split(':')[1]
        elif line.startswith(':'):
            parts = line.split()
            # [0] :
            # [1] start beat
            # [2] duration
            # [3] pitch
            # [4] word
            if parts[0] and parts[1] and parts[2] and parts[3] and parts[4]:
                ultrastar_class.startTimes.append(parts[1])
                ultrastar_class.durations.append(parts[2])
                ultrastar_class.pitches.append(parts[3])
                ultrastar_class.words.append(parts[4])
                # todo: Progress?
                # print(parts[4])

    return ultrastar_class


