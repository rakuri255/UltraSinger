"""Silence processing module"""

import librosa
from pydub import AudioSegment, silence

from modules.console_colors import ULTRASINGER_HEAD
from modules.Speech_Recognition.TranscribedData import TranscribedData


def remove_silence_from_transcription_data(audio_path: str, transcribed_data: list[TranscribedData]) -> list[TranscribedData]:
    """Remove silence from given transcription data"""

    print(
        f"{ULTRASINGER_HEAD} Removing silent start and ending, from transcription data"
    )

    #audio, sample_rate = librosa.load(audio_path, sr=None)
    y = AudioSegment.from_wav(audio_path)
    silence_timestamps = get_silence_sections(y)
    a = remove_silence2(silence_timestamps, transcribed_data)
    #remove_silence(audio, sample_rate, transcribed_data)

    return a

def get_silence_sections(y,
                         min_silence_len=50,
                         silence_thresh=-50) -> list[tuple[float, float]]:

    s = silence.detect_silence(y, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    s = [((start / 1000), (stop / 1000)) for start, stop in s]  # convert to sec
    return s

def remove_silence2(timeList: list[tuple[float, float]], transcribed_data: list[TranscribedData]):

    new_transcribed_data = []
    for i, data in enumerate(transcribed_data):
        new_transcribed_data.append(data)

        for j in range(len(timeList)):
            silence_start = timeList[j][0]
            silence_end = timeList[j][1]

            if silence_start >= data.end or silence_end <= data.start:
                continue
            if silence_start > data.start and silence_end < data.end:
                print(f"{ULTRASINGER_HEAD} Splitting \"{data.word}\" because it was in silence")
                splitted = TranscribedData({
                    "conf": data.conf,
                    "word": "~",
                    "end": data.end,
                    "start": silence_end
                })
                data.end = silence_start
                new_transcribed_data.append(splitted)
                break
            if silence_start < data.start and silence_end > data.end:
                new_transcribed_data.remove(data)
                print(f"{ULTRASINGER_HEAD} Removed \"{data.word}\" because it was in silence")
                break
            if silence_start < data.start:
                data.start = silence_end
                print(f"{ULTRASINGER_HEAD} Removed silence in the start of word \"{data.word}\"")
            if silence_end > data.end:
                data.end = silence_start
                print(f"{ULTRASINGER_HEAD} Removed silence in the end of word \"{data.word}\"")
            if silence_start > data.end:
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