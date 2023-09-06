"""Silence processing module"""

import librosa

from modules.console_colors import ULTRASINGER_HEAD
from modules.Speech_Recognition.TranscribedData import TranscribedData


def remove_silence_from_transcription_data(audio_path: str, transcribed_data: list[TranscribedData]) -> list[TranscribedData]:
    """Remove silence from given transcribtion data"""

    print(
        f"{ULTRASINGER_HEAD} Removing silent start and ending, from transcription data"
    )

    audio, sample_rate = librosa.load(audio_path, sr=None)

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

    return transcribed_data
