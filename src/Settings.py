class Settings:
    create_midi = True
    create_plot = False
    create_audio_chunks = False
    hyphenation = True
    use_separated_vocal = True
    create_karaoke = True

    input_file_path = ""
    output_file_path = ""
    mono_audio_path = ""

    language = None

    # Transcribe
    audio_chunk_folder_name = "audio-chunks"

    # Whisper
    transcriber = "whisper"  # whisper|vosk
    whisper_model = "large-v2"  # Multilingual model tiny|base|small|medium|large-v1|large-v2
    # English-only model tiny.en|base.en|small.en|medium.en
    whisper_align_model = None   # Model for other languages from huggingface.co e.g -> "gigant/romanian-wav2vec2"
    whisper_batch_size = 16   # reduce if low on GPU mem
    whisper_compute_type = None   # change to "int8" if low on GPU mem (may reduce accuracy)

    # Vosk
    vosk_model_path = ""  # "models/vosk-model-small-en-us-0.15"

    # Pitch
    crepe_model_capacity = "full"  # tiny|full
    crepe_step_size = 10 # in miliseconds
    crepe_batch_size = None # torchcrepe defaults to 2048, reduce if low on GPU mem

    # Device
    device = 'cpu'  # cpu|cuda
    force_whisper_cpu = False
    force_separation_cpu = False
    force_crepe_cpu = False
