"""Silence processing module"""
import librosa
import soundfile as sf

from pydub import AudioSegment, silence

from modules.console_colors import ULTRASINGER_HEAD
from modules.Speech_Recognition.TranscribedData import TranscribedData

def remove_silence_from_transcription_data(audio_path: str, transcribed_data: list[TranscribedData]) -> list[
    TranscribedData]:
    """Remove silence from given transcription data"""

    print(
        f"{ULTRASINGER_HEAD} Removing silent parts from transcription data"
    )

    silence_timestamps = get_silence_sections(audio_path)
    data = remove_silence(silence_timestamps, transcribed_data)
    return data


def get_silence_sections(audio_path: str,
                         min_silence_len=50,
                         silence_thresh=-50) -> list[tuple[float, float]]:
    y = AudioSegment.from_wav(audio_path)
    s = silence.detect_silence(y, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    s = [((start / 1000), (stop / 1000)) for start, stop in s]  # convert to sec
    return s


def remove_silence(silence_parts_list: list[tuple[float, float]], transcribed_data: list[TranscribedData]):
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
                        is_word_end = True
                    else:
                        split_word = "~"
                        is_word_end = False
                else:
                    split_end = origin_end
                    split_word = "~ "
                    is_word_end = True

                split_data = TranscribedData(confidence=data.confidence, word=split_word, end=split_end, start=silence_end, is_word_end=is_word_end)

                if not was_split:
                    data.end = silence_start

                    if data.end - data.start < 0.1:
                        data.start = silence_end
                        data.end = split_end
                        continue

                    if split_data.end - split_data.start <= 0.1:
                        continue

                    data.is_word_end = False

                    # Remove last whitespace from the data.word
                    if data.word[-1] == " ":
                        data.word = data.word[:-1]

                if split_data.end - split_data.start > 0.1:
                    was_split = True
                    new_transcribed_data.append(split_data)
                elif split_word == "~ " and not data.is_word_end:
                    if new_transcribed_data[-1].word[-1] != " ":
                        new_transcribed_data[-1].word += " "
                    new_transcribed_data[-1].is_word_end = True

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


def mute_no_singing_parts(mono_output_path, mute_output_path):
    print(
        f"{ULTRASINGER_HEAD} Mute audio parts with no singing"
    )
    silence_sections = get_silence_sections(mono_output_path)
    y, sr = librosa.load(mono_output_path, sr=None)
    # Mute the parts of the audio with no singing
    for i in silence_sections:
        # Define the time range to mute

        start_time = i[0]  # Start time in seconds
        end_time = i[1]  # End time in seconds

        # Convert time to sample indices
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        y[start_sample:end_sample] = 0
    sf.write(mute_output_path, y, sr)
