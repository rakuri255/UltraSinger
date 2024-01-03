class Settings:
    APP_VERSION = "0.0.8"

    create_midi = True
    create_plot = False
    create_audio_chunks = False
    hyphenation = True
    use_separated_vocal = True
    create_karaoke = True

    input_file_path = ""
    output_file_path = ""
    processing_audio_path = ""

    language = None
    format_version = "1.0.0"

    # Transcribe
    audio_chunk_folder_name = "audio-chunks"

    # Whisper
    transcriber = "whisper"  # whisper
    whisper_model = "large-v2"  # Multilingual model tiny|base|small|medium|large-v1|large-v2
    # English-only model tiny.en|base.en|small.en|medium.en
    whisper_align_model = None   # Model for other languages from huggingface.co e.g -> "gigant/romanian-wav2vec2"
    whisper_batch_size = 16   # reduce if low on GPU mem
    whisper_compute_type = None   # change to "int8" if low on GPU mem (may reduce accuracy)

    # Pitch
    crepe_model_capacity = "full"  # tiny|small|medium|large|full
    crepe_step_size = 10 # in miliseconds

    # Device
    pytorch_device = 'cpu'  # cpu|cuda
    tensorflow_device = 'cpu'  # cpu|cuda
    force_cpu = False
    force_whisper_cpu = False
    force_crepe_cpu = False
