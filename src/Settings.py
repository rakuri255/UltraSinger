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
    onset_snapping = True  # Snap syllable timings to detected spectral onsets
    onset_window_ms = 45.0  # Max distance in ms to snap syllables within phrases (strict 40-50ms)
    phrase_snap_window_ms = 120.0  # Max distance in ms to snap phrase starters and post-silence words (tarareos, verses)
    bpm_override = None  # User-specified BPM override
    octave_offset = None  # User-specified octave offset (e.g. -1, 0, 1) or None for auto
    auto_octave = True  # Automatically center notes within UltraStar staff
    trim_energy = True  # Trim note tails using RMS energy falloff
    create_click_track = True  # Generate [Sync_Test].mp3 with audible clicks on each syllable
    auto_calibrate_gap = True  # Calibrate #GAP to acoustic vocal onset
    max_calibrate_ms = 500.0  # Max discrepancy in ms allowed for auto-calibrating #GAP (safe limit against intro bleed)
    disable_melismas = False  # Disable melisma splitting (force 1 clean note per syllable, eliminating ~)
    melisma_threshold = 1.8  # Minimum semitones change to trigger melisma (filters vibrato)
    melisma_min_duration = 0.160  # Minimum duration (s) of a melisma sub-note
    duet = False  # Enable automatic duet separation (P1 and P2)
    p1_name = "Singer 1"  # Name for Player 1
    p2_name = "Singer 2"  # Name for Player 2
    keep_input_names = False  # Keep exact input folder/filename without hyphen normalization
    disable_musicbrainz = False  # Disable MusicBrainz online metadata queries
    to_whisper_out = False  # Only generate Whisper transcription to /input/.../whisper.txt and exit
    autocorrect = False  # Standalone fast auto-correction mode (align bars, pitch, and GAP from vocals)

    # Process data Paths
    input_file_path = ""
    output_folder_path = ""
    lyrics_file_path = None  # User-provided reference lyrics file (e.g. lyrics.txt)
    genre_file_path = None  # User-provided or auto-detected GENRE.txt
    genre = "Pop"  # Musical genre (e.g. from GENRE.txt, CLI --genre, or fallback "Pop")
    
    language = None
    format_version = FormatVersion.V1_2_0

    # Multi-Engine Architecture Settings
    separator_engine = "bs_roformer"  # bs_roformer|mel_roformer|htdemucs|scnet
    dereverb = "none"  # none|uvr_dereverb|kim_vocal_2
    asr_engine = "whisper_v2"  # whisper_v2|whisper_turbo
    aligner_engine = "wav2vec2"  # wav2vec2|mms_fa
    pitch_engine = "rmvpe"  # rmvpe|crepe|swiftf0|basic-pitch
    midi_engine = "basic_pitch"  # basic_pitch|some|omnizart
    magenta_humanize = False  # Enable MusicVAE / Performance RNN micro-timing and dynamics
    duet_engine = "mfcc_kmeans"  # mfcc_kmeans|ecapa_embeddings

    # Separation (Demucs or SOTA BS-Roformer)
    separator = "demucs"  # demucs|bs-roformer|mel_roformer|scnet|htdemucs
    demucs_model = DemucsModel.HTDEMUCS  # htdemucs|htdemucs_ft|htdemucs_6s|hdemucs_mmi|mdx|mdx_extra|mdx_q|mdx_extra_q|SIG
    bs_roformer_model = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

    # Pitch Extraction (SwiftF0, Crepe, RMVPE)
    pitcher = "swiftf0"  # swiftf0|crepe|rmvpe
    crepe_model_capacity = "full"
    crepe_step_size = 10  # ms

    # Whisper
    transcriber = "whisper"  # whisper
    whisper_model = WhisperModel.LARGE_V2  # Multilingual model tiny|base|small|medium|large-v1|large-v2|large-v3
    # English-only model tiny.en|base.en|small.en|medium.en
    whisper_align_model = None   # Model for other languages from huggingface.co e.g -> "gigant/romanian-wav2vec2"
    whisper_batch_size = 4  # reduce if low on GPU mem
    whisper_compute_type = None   # change to "int8" if low on GPU mem (may reduce accuracy)
    keep_numbers = False

    # Device
    pytorch_device = 'cpu'  # cpu|cuda
    force_cpu = False
    force_whisper_cpu = False
    force_crepe_cpu = False

    # MuseScore
    musescore_path = None

    # yt-dlp
    cookiefile = None

    # UltraSinger Evaluation Configuration
    test_songs_input_folder = None
    cache_override_path = None #"C:\\UltraSinger\\test_output"
    skip_cache_vocal_separation = False
    skip_cache_denoise_vocal_audio = False
    skip_cache_transcription = False
    skip_cache_pitch_detection = False
    calculate_score = True