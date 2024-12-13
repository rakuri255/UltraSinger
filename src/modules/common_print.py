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
    default  Creates all
    
    # Single file creation selection is in progress, you currently getting all!
    (-u      Create ultrastar txt file) # In Progress
    (-m      Create midi file) # In Progress
    (-s      Create sheet file) # In Progress
    
    ## INPUT is ultrastar.txt ##
    default  Creates all

    # Single selection is in progress, you currently getting all!
    (-r      repitch Ultrastar.txt (input has to be audio)) # In Progress
    (-p      Check pitch of Ultrastar.txt input) # In Progress
    (-m      Create midi file) # In Progress

    [transcription]
    # Default is whisper
    --whisper               Multilingual model > tiny|base|small|medium|large-v1|large-v2  >> ((default) is large-v2
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
    --hyphenation           True|False >> ((default) is True)
    --disable_separation    True|False >> ((default) is False)
    --disable_karaoke       True|False >> ((default) is False)
    --create_audio_chunks   True|False >> ((default) is False)
    --keep_cache            True|False >> ((default) is False)
    --plot                  True|False >> ((default) is False)
    --format_version        0.3.0|1.0.0|1.1.0 >> ((default) is 1.0.0)
    --musescore_path        path to MuseScore executable
    
    [device]
    --force_cpu             True|False >> ((default) is False)  All steps will be forced to cpu
    --force_whisper_cpu     True|False >> ((default) is False)  Only whisper will be forced to cpu
    --force_crepe_cpu       True|False >> ((default) is False)  Only crepe will be forced to cpu
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
    print()
    print(f"{ULTRASINGER_HEAD} {gold_highlighted('*****************************')}")
    print(f"{ULTRASINGER_HEAD} {gold_highlighted('UltraSinger Version:')} {light_blue_highlighted(app_version)}")
    print(f"{ULTRASINGER_HEAD} {gold_highlighted('*****************************')}")
