"""Convert audio to other formats"""

from pydub import AudioSegment
import librosa
import soundfile as sf

from modules.console_colors import ULTRASINGER_HEAD


def convert_audio_to_mono_wav(input_file: str, output_file: str) -> None:
    """Convert audio to mono wav"""
    print(f"{ULTRASINGER_HEAD} Converting audio for AI")

    y, sr = librosa.load(input_file, sr=None)

    if len(y.shape) > 1 and y.shape[0] > 1:
        y = librosa.to_mono(y)

    sf.write(output_file, y, sr)


def convert_wav_to_mp3(input_file: str, output_file: str) -> None:
    """Convert wav to mp3"""
    print(f"{ULTRASINGER_HEAD} Converting wav to mp3")

    sound = AudioSegment.from_wav(input_file)
    sound.export(output_file, format="mp3")
