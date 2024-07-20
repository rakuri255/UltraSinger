"""Convert audio to other formats"""

from pydub import AudioSegment
import librosa
import soundfile as sf

from modules.console_colors import ULTRASINGER_HEAD


def convert_audio_to_mono_wav(input_file_path: str, output_file_path: str) -> None:
    """Convert audio to mono wav"""
    print(f"{ULTRASINGER_HEAD} Converting audio for AI")
    y, sr = librosa.load(input_file_path, mono=True, sr=None)
    sf.write(output_file_path, y, sr)


def convert_wav_to_mp3(input_file_path: str, output_file_path: str) -> None:
    """Convert wav to mp3"""
    print(f"{ULTRASINGER_HEAD} Converting wav to mp3. -> {output_file_path}")

    sound = AudioSegment.from_wav(input_file_path)
    sound.export(output_file_path, format="mp3")
