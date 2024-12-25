"""Common Prints"""

from modules.console_colors import ULTRASINGER_HEAD, gold_highlighted, light_blue_highlighted


def print_help() -> None:
    """Print help text"""
    help_string = """
    UltraSinger.py [opt] [mode] [transcription] [pitcher] [extra]
    
    [opt]
    -h      This help text.
    -i      Ultrastar.txt
            audio like .mp3, .wav, youtube link
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
    --language              Override the language detected by whisper, does not affect transcription but steps after transcription
    --whisper_batch_size    Reduce if low on GPU mem >> ((default) is 16)
    --whisper_compute_type  Change to "int8" if low on GPU mem (may reduce accuracy) >> ((default) is "float16" for cuda devices, "int8" for cpu)
    --keep_numbers          Numbers will be transcribed as numerics instead of as words >> True|False >> ((default) is False)
    
    [pitcher]
    # Default is crepe
    --crepe            tiny|full >> ((default) is full)
    --crepe_step_size  unit is miliseconds >> ((default) is 10)
    
    [extra]
    --disable_hyphenation   Disable word hyphenation. Hyphenation is enabled by default.
    --disable_separation    Disable track separation. Track separation is enabled by default.
    --disable_karaoke       Disable creation of karaoke style txt file. Karaoke is enabled by default.
    --create_audio_chunks   Enable creation of audio chunks. Audio chunks are disabled by default.
    --keep_cache            Keep cache folder after creation. Cache folder is removed by default.
    --plot                  Enable creation of plots. Plots are disabled by default.
    --format_version        0.3.0|1.0.0|1.1.0|1.2.0 >> ((default) is 1.2.0)
    --musescore_path        path to MuseScore executable
    --keep_numbers          Transcribe numbers as digits and not words
    --ffmpeg                Path to ffmpeg and ffprobe executable

    [yt-dlp]
    --cookiefile            File name where cookies should be read from and dumped to.
    
    [device]
    --force_cpu             Force all steps to be processed on CPU.
    --force_whisper_cpu     Force whisper transcription to be processed on CPU.
    --force_crepe_cpu       Force crepe pitch detection to be processed on CPU.
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
