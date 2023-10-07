class Settings:
    create_midi = True
    create_plot = False
    create_audio_chunks = False
    hyphenation = True
    use_separated_vocal = True
    create_karaoke = True
    ignore_audio = False
    input_file_is_ultrastar_txt = False

    input_file_path = ""
    output_file_path = ""
    mono_audio_path = ""

    language = None

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
    pitch_loudness_threshold = -60

    # Device
    pytorch_device = 'cpu'  # cpu|cuda
    tensorflow_device = 'cpu'  # cpu|cuda
    force_cpu = False

    # UltraSinger Evaluation Configuration
    test_songs_input_folder = None
    cache_override_path = None
    skip_cache_vocal_separation = False
    skip_cache_denoise_vocal_audio = False
    skip_cache_transcription = False
    skip_cache_pitch_detection = False
