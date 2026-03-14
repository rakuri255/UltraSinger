"""Common Prints"""

from modules.console_colors import ULTRASINGER_HEAD, gold_highlighted, light_blue_highlighted


def print_help() -> None:
    """Print help text"""
    help_string = """
    UltraSinger.py [opt] [mode] [transcription] [pitcher] [extra]
    
    [opt]
    -h      This help text.
    -i      Ultrastar.txt
            audio/video like .mp3, .mp4, .wav, youtube link
    -o      Output folder
    
    [mode]
    ## INPUT is audio ##
    default (Full Automatic Mode) - Creates all, depending on command line options
    --interactive - Interactive Mode. All options are asked at runtime for easier configuration
    
    # Single file creation selection is in progress, you currently getting all!
    (-u      Create ultrastar txt file) # In Progress
    (-m      Create midi file) # In Progress
    (-s      Create sheet file) # In Progress
    
    ## INPUT is ultrastar.txt ##
    default  Creates all


    [separation]
    # Default is htdemucs
    --demucs              Model name htdemucs|htdemucs_ft|htdemucs_6s|hdemucs_mmi|mdx|mdx_extra|mdx_q|mdx_extra_q >> ((default) is htdemucs)

    [transcription]
    # Default is whisper
    --whisper               Multilingual model > tiny|base|small|medium|large-v1|large-v2|large-v3  >> ((default) is large-v2)
                            English-only model > tiny.en|base.en|small.en|medium.en
    --whisper_align_model   Use other languages model for Whisper provided from huggingface.co
    --language              Override the language detected by whisper for alignment and hyphenation.
                            Default: auto-detect. WARNING: setting this for non-matching songs
                            will degrade alignment quality (e.g. --language en for German songs).
    --whisper_batch_size    Reduce if low on GPU mem >> ((default) is 16)
    --whisper_compute_type  Change to "int8" if low on GPU mem (may reduce accuracy) >> ((default) is "float16" for cuda devices, "int8" for cpu)
    --keep_numbers          Numbers will be transcribed as numerics instead of as words >> True|False >> ((default) is False)
    
    [extra]
    --bpm                   Override auto-detected BPM with a manual value (e.g., --bpm 340)
    --disable_hyphenation   Disable word hyphenation. Hyphenation is enabled by default.
    --disable_separation    Disable track separation. Track separation is enabled by default.
    --disable_karaoke       Disable creation of karaoke style txt file. Karaoke is enabled by default.
    --disable_onset_correction  Disable onset-based timing correction. Enabled by default.
    --create_audio_chunks   Enable creation of audio chunks. Audio chunks are disabled by default.
    --keep_cache            Keep cache folder after creation. Cache folder is removed by default.
    --plot                  Enable creation of plots. Plots are disabled by default.
    --disable_quantization  Disable key quantization. Key quantization is enabled by default and removes slides and out-of-key notes.
    --disable_vocal_center  Disable vocal-centre octave correction. Enabled by default.
    --format_version        0.3.0|1.0.0|1.1.0|1.2.0 >> ((default) is 1.2.0)
    --musescore_path        path to MuseScore executable
    --keep_numbers          Transcribe numbers as digits and not words
    --ffmpeg                Path to ffmpeg and ffprobe executable

    [denoise]
    --denoise_nr            Noise reduction in dB (0.01-97). Lower preserves more vocal detail. >> ((default) is 20)
    --denoise_nf            Noise floor in dB (-80 to -20) >> ((default) is -80)
    --disable_denoise_track_noise  Disable adaptive noise floor tracking >> ((default) tracking is enabled)

    [yt-dlp]
    --cookiefile            File name where cookies should be read from and dumped to.
    
    [llm]
    --llm_correct           Enable LLM-based lyric correction (requires API key)
    --llm_api_base_url      OpenAI-compatible API base URL >> ((default) is https://api.openai.com/v1)
    --llm_api_key           API key for LLM service (or set LLM_API_KEY env var)
    --llm_model             LLM model name >> ((default) is gpt-4o-mini)

    [device]
    --force_cpu             Force all steps to be processed on CPU.
    --force_whisper_cpu     Force whisper transcription to be processed on CPU.
    """
    print(help_string)


def print_support() -> None:
    """Print support text"""
    print()
    print(
        f"{ULTRASINGER_HEAD} {gold_highlighted('Do you like UltraSinger? Want it to be even better? Then help with your')} {light_blue_highlighted('support')}{gold_highlighted('!')}"
    )
    print(
        f"{ULTRASINGER_HEAD} See project page -> https://github.com/rakuri255/UltraSinger"
    )
    print(
        f"{ULTRASINGER_HEAD} {gold_highlighted('This will help a lot to keep this project alive and improved.')}"
    )


def print_version(app_version: str) -> None:
    """Print version text"""
    versiontext = (f"UltraSinger Version: {app_version}")    
    starquant = "*" * len(versiontext) # set star number to length of version
    print()
    print(f"{ULTRASINGER_HEAD} {gold_highlighted(starquant)}")
    print(f"{ULTRASINGER_HEAD} {gold_highlighted('UltraSinger Version:')} {light_blue_highlighted(app_version)}")
    print(f"{ULTRASINGER_HEAD} {gold_highlighted(starquant)}")
