"""Convert audio to other formats"""

from pydub import AudioSegment

from src.modules.console_colors import ULTRASINGER_HEAD


def convert_audio_to_mono_wav(input_file: str, output_file: str) -> None:
    """Convert audio to mono wav"""
    print(f"{ULTRASINGER_HEAD} Converting audio for AI")

    if ".mp3" in input_file:
        sound = AudioSegment.from_mp3(input_file)
    elif ".wav" in input_file:
        sound = AudioSegment.from_wav(input_file)
    else:
        raise ValueError("data format not supported")

    sound = sound.set_channels(1)
    sound.export(output_file, format="wav")


def convert_wav_to_mp3(input_file: str, output_file: str) -> None:
    """Convert wav to mp3"""
    print(f"{ULTRASINGER_HEAD} Converting wav to mp3")

    sound = AudioSegment.from_wav(input_file)
    sound.export(output_file, format="mp3")
