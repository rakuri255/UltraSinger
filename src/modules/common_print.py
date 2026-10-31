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
    # Default is bs_roformer
    --separator-engine, --separator  Vocal separation engine > bs_roformer|mel_roformer|htdemucs|scnet|demucs >> ((default) is bs_roformer)
    --dereverb            Surgical de-reverb/de-echo post-filter > none|uvr_dereverb|kim_vocal_2 >> ((default) is none)
    --demucs              Model name htdemucs|htdemucs_ft|htdemucs_6s|hdemucs_mmi|mdx|mdx_extra|mdx_q|mdx_extra_q >> ((default) is htdemucs)
    --bs_roformer_model   BS-Roformer model checkpoint >> ((default) is model_bs_roformer_ep_317_sdr_12.9755.ckpt)

    [transcription & alignment]
    # Default is whisper_v2 and wav2vec2
    --asr-engine          ASR Engine > whisper_v2|whisper_turbo >> ((default) is whisper_v2)
    --aligner-engine      Phonetic Aligner > wav2vec2|mms_fa >> ((default) is wav2vec2)
    --whisper             Multilingual model > tiny|base|small|medium|large-v1|large-v2|large-v3|large-v3-turbo >> ((default) is large-v2)
                          English-only model > tiny.en|base.en|small.en|medium.en
    --whisper_align_model Use other languages model for Whisper provided from huggingface.co
    --language            Override the language detected by whisper, does not affect transcription but steps after transcription
    --whisper_batch_size  Reduce if low on GPU mem >> ((default) is 16)
    --whisper_compute_type Change to "int8" if low on GPU mem (may reduce accuracy) >> ((default) is "float16" for cuda devices, "int8" for cpu)
    --keep_numbers        Numbers will be transcribed as numerics instead of as words >> True|False >> ((default) is False)
    
    [pitcher & midi]
    # Default pitcher is rmvpe, default midi-engine is basic_pitch
    --pitch-engine, --pitcher Pitch detection engine > rmvpe|crepe|swiftf0|basic-pitch >> ((default) is rmvpe)
    --midi-engine         MIDI transcription engine > basic_pitch|some|omnizart >> ((default) is basic_pitch)
    --magenta-humanize    Apply MusicVAE / Performance RNN humanization (micro-timing and expressive dynamics)
    --crepe_model         CREPE model capacity > tiny|small|medium|large|full >> ((default) is full)
    
    [duet]
    --duet                Enable duet mode (automatically separates song into P1 and P2 singers)
    --duet-engine         Duet diarization engine > mfcc_kmeans|ecapa_embeddings >> ((default) is mfcc_kmeans)
    --p1_name             Display name for Player 1 (default is "Singer 1")
    --p2_name             Display name for Player 2 (default is "Singer 2")
    
    [extra]
    --disable_hyphenation   Disable word hyphenation. Hyphenation is enabled by default.
    --disable_separation    Disable track separation. Track separation is enabled by default.
    --disable_karaoke       Disable creation of karaoke style txt file. Karaoke is enabled by default.
    --create_audio_chunks   Enable creation of audio chunks. Audio chunks are disabled by default.
    --keep_cache            Keep cache folder after creation. Cache folder is removed by default.
    --plot                  Enable creation of plots. Plots are disabled by default.
    --quantize_to_key       Quantize notes to the detected musical key. This removes slides and out-of-key notes.
    --format_version        0.3.0|1.0.0|1.1.0|1.2.0 >> ((default) is 1.2.0)
    --musescore_path        path to MuseScore executable
    --genre                 Specify musical genre (e.g. "Hip Hop/R&B", "Rock", "Pop"). Also auto-detected from GENRE.txt
    --keep_numbers          Transcribe numbers as digits and not words
    --disable_onset_snapping Disable acoustic onset snapping. Onset snapping is enabled by default.
    --onset_window_ms       Maximum window in ms to snap syllables to detected onsets >> ((default) is 120)
    --bpm                   Manually specify song BPM (overrides automatic BPM detection)
    --octave_offset         Manual octave shift integer e.g. -1, 0, 1 >> ((default) is auto)
    --disable_auto_octave   Disable automatic octave centering on UltraStar staff lines
    --disable_trim_energy   Disable RMS voice energy tail trimming
    --disable_click_track   Disable generation of [Sync_Test].mp3 verification audio
    --disable_auto_calibrate_gap Disable auto-calibration of first syllable start (#GAP) to acoustic onset
    --duet                  Enable duet mode (automatically separates song into P1 and P2 singers)
    --p1_name               Display name for Player 1 (default is "Singer 1")
    --p2_name               Display name for Player 2 (default is "Singer 2")
    --keep_input_names      Preserve exact input folder / file names verbatim without normalization.
    --disable_musicbrainz   Disable MusicBrainz online metadata queries.
    --to_whisper_out        Only transcribe with Whisper and save to /input/.../whisper.txt without generating /output.
    --disable_melismas      Disable melisma pitch splitting (forces 1 clean note per syllable, eliminating ~ notes).
    --melisma_threshold     Minimum semitone deviation to trigger a melisma split (default: 1.8, filters out vibrato).
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
