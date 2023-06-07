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

    # Transcribe
    audio_chunk_folder_name = "audio-chunks"

    # Whisper
    transcriber = "whisper"  # whisper|vosk
    whisper_model = "large-v2"  # Multilingual model tiny|base|small|medium|large-v1|large-v2
    # English-only model tiny.en|base.en|small.en|medium.en

    # Vosk
    vosk_model_path = ""  # "models/vosk-model-small-en-us-0.15"

    # Pitch
    crepe_model_capacity = "full"  # tiny|small|medium|large|full
    crepe_step_size = 10

    # Device
    device = 'cpu'  # cpu|cuda
    force_whisper_cpu = False
    force_separation_cpu = False

