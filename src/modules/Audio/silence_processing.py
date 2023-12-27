"""Silence processing module"""

import librosa
from pydub import AudioSegment, silence

from modules.console_colors import ULTRASINGER_HEAD
from modules.Speech_Recognition.TranscribedData import TranscribedData

def remove_silence_from_transcription_data(audio_path: str, transcribed_data: list[TranscribedData]) -> list[
    TranscribedData]:
    """Remove silence from given transcription data"""

    print(
        f"{ULTRASINGER_HEAD} Removing silent start and ending, from transcription data"
    )

    silence_timestamps = get_silence_sections(audio_path)
    a = remove_silence2(silence_timestamps, transcribed_data)
    return a

    #audio, sample_rate = librosa.load(audio_path, sr=None)
    #remove_silence(audio, sample_rate, transcribed_data)
    #return transcribed_data


def get_silence_sections(audio_path: str,
                         min_silence_len=50,
                         silence_thresh=-50) -> list[tuple[float, float]]:
    y = AudioSegment.from_wav(audio_path)
    s = silence.detect_silence(y, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    s = [((start / 1000), (stop / 1000)) for start, stop in s]  # convert to sec
    return s


def remove_silence2(silence_parts_list: list[tuple[float, float]], transcribed_data: list[TranscribedData]):
    new_transcribed_data = []

    for data in transcribed_data:
        new_transcribed_data.append(data)

        origin_end = data.end
        was_split = False

        for silence_start, silence_end in silence_parts_list:

            # |    ****    | silence
            # |  **    **  | data
            # |0 1 2 3 4 5 | time
            if silence_start > origin_end or silence_end < data.start:
                continue

            # |    **  **    | silence
            # |  **********  | data
            # |0 1 2 3 4 5 6 | time
            if silence_start >= data.start and silence_end <= origin_end:
                next_index = silence_parts_list.index((silence_start, silence_end)) + 1
                if next_index < len(silence_parts_list) and silence_parts_list[next_index][0] < origin_end:
                    split_end = silence_parts_list[next_index][0]

                    if silence_parts_list[next_index][1] >= origin_end:
                        split_word = "~ "
                        is_hyphen = False
                    else:
                        split_word = "~"
                        is_hyphen = True
                else:
                    split_end = origin_end
                    split_word = "~ "
                    is_hyphen = False

                split_data = TranscribedData({"conf": data.conf, "word": split_word, "end": split_end, "start": silence_end, "is_hyphen": is_hyphen})

                if not was_split:
                    data.end = silence_start

                    if data.end - data.start < 0.1:
                        data.start = silence_end
                        data.end = split_end
                        continue

                    if split_data.end - split_data.start <= 0.1:
                        continue

                    data.is_hyphen = True

                    # Remove last whitespace from the data.word
                    if data.word[-1] == " ":
                        data.word = data.word[:-1]

                if split_data.end - split_data.start > 0.1:
                    was_split = True
                    new_transcribed_data.append(split_data)
                elif split_word == "~ " and data.is_hyphen:
                    if new_transcribed_data[-1].word[-1] != " ":
                        new_transcribed_data[-1].word += " "
                    new_transcribed_data[-1].is_hyphen = False

                continue

            # |    ****  | silence
            # |     **   | data
            # |0 1 2 3 4 | time
            if silence_start < data.start and silence_end > origin_end:
                new_transcribed_data.remove(data)
                break

            # |    ****    | silence
            # |      ****  | data
            # |0 1 2 3 4 5 | time
            if silence_start < data.start:
                data.start = silence_end

            # |    ****  | silence
            # |  ****    | data
            # |0 1 2 3 4 | time
            if silence_end > origin_end:
                data.end = silence_start

            # |    ****  | silence
            # |  **      | data
            # |0 1 2 3 4 | time
            if silence_start > origin_end:
                # Nothing to do with this word anymore, go to next word
                break
    return new_transcribed_data


def remove_silence(audio, sample_rate, transcribed_data: list[TranscribedData]):
    for i, data in enumerate(transcribed_data):
        start_time = data.start
        end_time = data.end
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        chunk = audio[start_sample:end_sample]

        # todo: why 5 works good? It should be 40db ?!?
        # max_dB = librosa.amplitude_to_db(chunk, ref=np.max)
        silence_threshold = 5
        onsets = librosa.effects.split(
            chunk, top_db=silence_threshold, frame_length=2048, hop_length=100
        )

        # Get the duration of the first and last silent intervals
        if len(onsets) > 0:
            first_silence = onsets[0][0]
            last_silence = len(chunk) - onsets[-1][1]

            first_silence_duration = librosa.samples_to_time(
                first_silence, sr=sample_rate
            )
            last_silence_duration = librosa.samples_to_time(
                last_silence, sr=sample_rate
            )
        else:
            first_silence_duration = 0
            last_silence_duration = 0

        data.start = (
                data.start + first_silence_duration
        )
        data.end = (
                data.end - last_silence_duration
        )
