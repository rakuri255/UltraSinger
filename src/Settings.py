from dataclasses import dataclass, field
from typing import Optional

from dataclasses_json import dataclass_json

from modules.Audio.separation import DemucsModel
from modules.Speech_Recognition.Whisper import WhisperModel


@dataclass_json
@dataclass
class Settings:
    APP_VERSION = "0.0.12-dev2"

    APP_VERSION: str = "0.0.12-dev1"

    create_midi: bool = True
    create_plot: bool = False
    create_audio_chunks: bool = False
    hyphenation: bool = True
    use_separated_vocal: bool = True
    create_karaoke: bool = True
    ignore_audio: bool = False
    input_file_is_ultrastar_txt: bool = False # todo: to process_data
    keep_cache: bool = False

    # Process data Paths
    input_file_path: str = ""
    output_folder_path: str = ""

    language: Optional[str] = None
    format_version: str = "1.0.0"

    # Demucs
    demucs_model: str = DemucsModel.HTDEMUCS  # htdemucs|htdemucs_ft|htdemucs_6s|hdemucs_mmi|mdx|mdx_extra|mdx_q|mdx_extra_q|SIG

    # Whisper
    transcriber: str = "whisper"  # whisper
    whisper_model: str = WhisperModel.LARGE_V2  # Multilingual model tiny|base|small|medium|large-v1|large-v2|large-v3
    # English-only model tiny.en|base.en|small.en|medium.en
    whisper_align_model: Optional[str] = None   # Model for other languages from huggingface.co e.g -> "gigant/romanian-wav2vec2"
    whisper_batch_size: int = 16   # reduce if low on GPU mem
    whisper_compute_type: Optional[str] = None   # change to "int8" if low on GPU mem (may reduce accuracy)

    # Pitch
    crepe_model_capacity: str = "full"  # tiny|small|medium|large|full
    crepe_step_size: int = 10 # in miliseconds
    pitch_loudness_threshold: int = -60

    # Device
    pytorch_device: str = "cpu"  # cpu|cuda
    tensorflow_device: str = "cpu"  # cpu|cuda
    force_cpu: bool = False
    force_whisper_cpu: bool = False
    force_crepe_cpu: bool = False

    # MuseScore
    musescore_path: Optional[str] = None

    # UltraSinger Evaluation Configuration
    test_songs_input_folder: Optional[str] = None
    cache_override_path: Optional[str] = None
    skip_cache_vocal_separation: bool = False
    skip_cache_denoise_vocal_audio: bool = False
    skip_cache_transcription: bool = False
    skip_cache_pitch_detection: bool = False
    calculate_score: bool = True
