from dataclasses import dataclass

from dataclasses_json import dataclass_json

from modules.Audio.separation import DemucsModel
from modules.Speech_Recognition.Whisper import WhisperModel
from modules.Ultrastar.ultrastar_txt import FormatVersion


@dataclass_json
@dataclass
class Settings:

    APP_VERSION = "0.0.13.dev16"
    CONFIDENCE_THRESHOLD = 0.6
    CONFIDENCE_PROMPT_TIMEOUT = 4

    create_midi = True
    create_plot = False
    create_audio_chunks = False
    hyphenation = True
    use_separated_vocal = True
    create_karaoke = True
    ignore_audio = False
    input_file_is_ultrastar_txt = False # todo: to process_data
    keep_cache = False
    interactive_mode = False
    user_ffmpeg_path = ""
    quantize_to_key = True  # Quantize notes to the detected key of the song

    # Process data Paths
    input_file_path = ""
    output_folder_path = ""
    
    language = None
    format_version = FormatVersion.V1_2_0

    # Demucs
    demucs_model = DemucsModel.HTDEMUCS  # htdemucs|htdemucs_ft|htdemucs_6s|hdemucs_mmi|mdx|mdx_extra|mdx_q|mdx_extra_q|SIG

    # Whisper
    transcriber = "whisper"  # whisper
    whisper_model = WhisperModel.LARGE_V2  # Multilingual model tiny|base|small|medium|large-v1|large-v2|large-v3
    # English-only model tiny.en|base.en|small.en|medium.en
    whisper_align_model = None   # Model for other languages from huggingface.co e.g -> "gigant/romanian-wav2vec2"
    whisper_batch_size = 16   # reduce if low on GPU mem
    whisper_compute_type = None   # change to "int8" if low on GPU mem (may reduce accuracy)
    keep_numbers = False

    # Device
    pytorch_device = 'cpu'  # cpu|cuda
    force_cpu = False
    force_whisper_cpu = False

    # MuseScore
    musescore_path = None

    # yt-dlp
    cookiefile = None

    # Denoise
    denoise_noise_reduction = 20  # Noise reduction in dB (0.01-97, default: 20). Previous default was 70 which destroyed vocal nuances needed by Whisper.
    denoise_noise_floor = -80  # Noise floor in dB (-80 to -20, default: -80)
    denoise_track_noise = True  # Enable adaptive noise floor tracking

    # UltraSinger Evaluation Configuration
    test_songs_input_folder = None
    cache_override_path = None #"C:\\UltraSinger\\test_output"
    skip_cache_vocal_separation = False
    skip_cache_denoise_vocal_audio = False
    skip_cache_transcription = False
    skip_cache_pitch_detection = False
    calculate_score = True