"""UltraSinger uses AI to automatically create UltraStar song files"""

import copy
import getopt
import os
import re
import sys
import Levenshtein
import librosa

from packaging import version

from modules import os_helper
from modules.init_interactive_mode import init_settings_interactive
from modules.Audio.denoise import denoise_vocal_audio
from modules.Audio.separation import separate_vocal_from_audio
from modules.Audio.vocal_chunks import (
    create_audio_chunks_from_transcribed_data,
    create_audio_chunks_from_ultrastar_data,
)
from modules.Audio.key_detector import detect_key_from_audio, get_allowed_notes_for_key
from modules.Audio.silence_processing import remove_silence_from_transcription_data, mute_no_singing_parts
from modules.Audio.separation import DemucsModel
from modules.Audio.convert_audio import convert_audio_to_mono_wav, convert_audio_format
from modules.Audio.youtube import (
    download_from_youtube,
)
from modules.Audio.bpm import get_bpm_from_file
from modules.Audio.onset_snapping import snap_transcribed_data_to_onsets
from modules.Audio.voice_activity_trimmer import trim_syllable_ends_with_energy
from modules.Audio.vocal_validator import validate_and_calibrate_gap
from modules.Audio.sync_click_track import create_sync_click_track
from modules.Audio.duet_separator import separate_duet_phrases

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    gold_highlighted,
    red_highlighted,
    green_highlighted,
    cyan_highlighted,
    bright_green_highlighted,
)
from modules.Midi.midi_creator import (
    create_midi_segments_from_transcribed_data,
    create_repitched_midi_segments_from_ultrastar_txt,
    create_midi_file,
)
from modules.Midi.MidiSegment import MidiSegment
from modules.Midi.note_length_calculator import get_thirtytwo_note_second, get_sixteenth_note_second
from modules.Pitcher.pitcher import (
    get_pitch_with_file,
)
from modules.Pitcher.pitched_data import PitchedData
from modules.Speech_Recognition.TranscriptionResult import TranscriptionResult
from modules.Speech_Recognition.hyphenation import (
    hyphenate_each_word,
)
from modules.Speech_Recognition.Whisper import transcribe_with_whisper
from modules.Speech_Recognition.genre_profiles import resolve_genre_profile
from modules.Ultrastar import (
    ultrastar_writer,
)
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Speech_Recognition.Whisper import WhisperModel
from modules.Ultrastar.ultrastar_score_calculator import Score, calculate_score_points
from modules.Ultrastar.ultrastar_txt import FILE_ENCODING, FormatVersion
from modules.Ultrastar.coverter.ultrastar_txt_converter import from_ultrastar_txt, \
    create_ultrastar_txt_from_midi_segments, create_ultrastar_txt_from_automation
from modules.Ultrastar.ultrastar_parser import parse_ultrastar_txt
from modules.common_print import print_support, print_help, print_version
from modules.os_helper import check_file_exists, get_unused_song_output_dir
from modules.plot import create_plots
from modules.musicbrainz_client import search_musicbrainz
from modules.sheet import create_sheet
from modules.ProcessData import ProcessData, ProcessDataPaths, MediaInfo
from modules.DeviceDetection.device_detection import check_gpu_support
from modules.Image.image_helper import save_image
from modules.ffmpeg_helper import (
    is_ffmpeg_available,
    get_ffmpeg_and_ffprobe_paths,
    is_video_file,
    separate_audio_video,
)

from Settings import Settings

settings = Settings()


def align_syllables_by_chars(data: TranscribedData, syllables: list[str]) -> list[tuple[float, float]]:
    """Align syllables using WhisperX character timestamps if available."""
    if not data.chars or len(syllables) <= 1:
        return None

    syllable_times = []
    current_char_idx = 0
    for s_idx, syl in enumerate(syllables):
        syl_clean = re.sub(r"[^\w]", "", syl)
        syl_len = len(syl_clean) or len(syl.strip())
        end_char_idx = min(current_char_idx + syl_len, len(data.chars))
        syl_chars = [c for c in data.chars[current_char_idx:end_char_idx] if "start" in c and "end" in c]

        if syl_chars:
            s_start = float(syl_chars[0]["start"])
            s_end = float(syl_chars[-1]["end"])
        else:
            s_start = None
            s_end = None

        syllable_times.append((s_start, s_end))
        current_char_idx = end_char_idx

    # Resolve boundaries into contiguous, non-inverted segments with semi-Markov duration floor (Phase 1)
    resolved = []
    curr_start = data.start
    min_syl_dur = 0.045
    for s_idx, (s_start, s_end) in enumerate(syllable_times):
        is_last = (s_idx == len(syllables) - 1)
        actual_start = s_start if (s_start is not None and s_start >= curr_start) else curr_start

        if is_last:
            actual_end = max(actual_start + min_syl_dur, data.end)
        elif s_idx + 1 < len(syllable_times) and syllable_times[s_idx + 1][0] is not None:
            actual_end = max(actual_start + min_syl_dur, syllable_times[s_idx + 1][0])
        elif s_end is not None and s_end > actual_start:
            actual_end = max(actual_start + min_syl_dur, s_end)
        else:
            actual_end = actual_start + 0.06

        if actual_end <= actual_start:
            actual_end = actual_start + min_syl_dur

        resolved.append((round(actual_start, 3), round(actual_end, 3)))
        curr_start = actual_end

    return resolved


def add_hyphen_to_data(
        transcribed_data: list[TranscribedData], hyphen_words: list[list[str]]
):
    """Add hyphen to transcribed data return new data list"""
    new_data = []

    for i, data in enumerate(transcribed_data):
        if not hyphen_words[i]:
            data.is_word_end = True
            if not data.word.endswith(" "):
                data.word += " "
            new_data.append(data)
        else:
            char_times = align_syllables_by_chars(data, hyphen_words[i])
            if char_times:
                for j, syl_word in enumerate(hyphen_words[i]):
                    s_start, s_end = char_times[j]
                    dup = copy.copy(data)
                    dup.start = s_start
                    dup.end = s_end
                    dup.is_hyphen = True
                    is_last = (j == len(hyphen_words[i]) - 1)
                    dup.is_word_end = is_last
                    dup.is_line_end = is_last and getattr(data, "is_line_end", False)
                    dup.word = (syl_word.rstrip(" ") + " ") if is_last else syl_word.rstrip(" ")
                    new_data.append(dup)
            else:
                # Equation III: Dennis Klatt Structural Phonetic Weighting when char_times is None
                vowels_set = set("aeiouáéíóúü")
                syl_weights = []
                for j_idx, s_word in enumerate(hyphen_words[i]):
                    s_clean = s_word.strip().lower()
                    n_v = max(1, sum(1 for c in s_clean if c in vowels_set))
                    n_c = sum(1 for c in s_clean if c.isalpha() and c not in vowels_set)
                    is_last_syl = (j_idx == len(hyphen_words[i]) - 1)
                    w_syl = 3.0 * n_v + 1.0 * n_c + (1.5 if is_last_syl else 0.0)
                    syl_weights.append(w_syl)
                total_syl_w = max(0.1, sum(syl_weights))

                total_duration = max(0.04, data.end - data.start)
                next_start = data.start
                for j in enumerate(hyphen_words[i]):
                    hyphenated_word_index = j[0]
                    dup = copy.copy(data)
                    syl_dur = total_duration * (syl_weights[hyphenated_word_index] / total_syl_w)
                    dup.start = next_start
                    next_start = data.end if (hyphenated_word_index == len(hyphen_words[i]) - 1) else (next_start + syl_dur)
                    dup.end = next_start
                    dup.is_hyphen = True
                    is_last = (hyphenated_word_index == len(hyphen_words[i]) - 1)
                    dup.is_word_end = is_last
                    dup.is_line_end = is_last and getattr(data, "is_line_end", False)
                    syl = hyphen_words[i][hyphenated_word_index]
                    dup.word = (syl.rstrip(" ") + " ") if is_last else syl.rstrip(" ")
                    new_data.append(dup)

    return new_data


# Todo: Unused
def correct_words(recognized_words, word_list_file):
    """Docstring"""
    with open(word_list_file, "r", encoding="utf-8") as file:
        text = file.read()
    word_list = text.split()

    for i, rec_word in enumerate(recognized_words):
        if rec_word.word in word_list:
            continue

        closest_word = min(
            word_list, key=lambda x: Levenshtein.distance(rec_word.word, x)
        )
        print(recognized_words[i].word + " - " + closest_word)
        recognized_words[i].word = closest_word
    return recognized_words


def remove_unecessary_punctuations(transcribed_data: list[TranscribedData]) -> None:
    """Remove unnecessary punctuations (including parentheses and question/exclamation marks) while preserving apostrophes."""
    punctuation = ".,()[]{}:;¿?¡!\"“”«»"
    trans_table = {ord(ch): None for ch in punctuation}
    for data in transcribed_data:
        has_trailing_space = data.word.endswith(" ")
        cleaned = data.word.translate(trans_table).strip()
        if cleaned:
            data.word = (cleaned + " ") if (has_trailing_space or getattr(data, "is_word_end", True)) else cleaned


def rebalance_monosyllable_prepositions(transcribed_data: list[TranscribedData]) -> list[TranscribedData]:
    """
    Rebalance acoustic durations between unstressed monosyllabic prepositions/conjunctions
    (e.g., 'a', 'de', 'en', 'y', 'o') and subsequent rhyming pronouns/monosyllables (e.g., 'mi', 'mí', 'ti', 'tu', 'yo').

    In singing, Whisper frequently stretches open vowels ('a') across the entire sustained note
    and compresses the subsequent stressed syllable ('mí') into trailing dead air.
    This restores natural human singing prosody: the preposition lasts ~0.20-0.26s, and the tonic syllable
    holds the remainder of the sustained melodic vibration.
    """
    if not transcribed_data or len(transcribed_data) < 2:
        return transcribed_data

    UNSTRESSED_PREPS = {"a", "de", "en", "y", "o"}
    TONIC_TARGETS = {"mi", "mí", "ti", "tu", "yo", "él", "dios", "sol"}

    for i in range(len(transcribed_data) - 1):
        curr_td = transcribed_data[i]
        next_td = transcribed_data[i + 1]

        w_curr = curr_td.word.strip().lower().rstrip(".,;!?:")
        w_next = next_td.word.strip().lower().rstrip(".,;!?:")

        curr_dur = float(curr_td.end) - float(curr_td.start)
        next_dur = float(next_td.end) - float(next_td.start)

        if w_curr in UNSTRESSED_PREPS and w_next in TONIC_TARGETS:
            if curr_dur >= 0.38 or (curr_dur >= 0.30 and next_dur <= 0.15):
                prep_target_dur = 0.24  # Standard sung duration of an unstressed entrance preposition (~6 beats at ~380 BPM)
                prep_target_dur = min(curr_dur * 0.30, prep_target_dur)
                prep_target_dur = max(0.18, prep_target_dur)

                split_point = float(curr_td.start) + prep_target_dur
                orig_curr_end = float(curr_td.end)
                orig_next_end = float(next_td.end)

                new_next_end = max(orig_curr_end, orig_next_end)

                # Bound new_next_end to avoid stretching notes into multi-second voids
                max_allowed_dur = max(next_dur, 1.80)
                if (new_next_end - split_point) > max_allowed_dur:
                    new_next_end = split_point + max_allowed_dur

                if i + 2 < len(transcribed_data):
                    after_next_start = float(transcribed_data[i + 2].start)
                    if new_next_end > after_next_start:
                        new_next_end = max(split_point + 0.10, after_next_start - 0.02)

                curr_td.end = split_point
                next_td.start = split_point
                next_td.end = new_next_end
                print(
                    f"{ULTRASINGER_HEAD} {cyan_highlighted('Prosody Rebalance:')} "
                    f"'{curr_td.word.strip()}' shortened ({curr_dur:.2f}s -> {prep_target_dur:.2f}s), "
                    f"'{next_td.word.strip()}' extended to hold melody ({next_dur:.2f}s -> {new_next_end - split_point:.2f}s)"
                )

    return transcribed_data


def run() -> tuple[str, Score, Score]:
    """The processing function of this program"""
    if settings.autocorrect:
        from modules.Audio.ultrastar_autocorrect import autocorrect_song, find_song_folders_in_dir
        force_crepe_cpu = getattr(settings, "force_crepe_cpu", False)
        device = "cpu" if settings.force_cpu or (force_crepe_cpu and settings.pitcher == "crepe") else settings.pytorch_device
        pitcher = getattr(settings, "pitcher", "crepe")

        target_folders = find_song_folders_in_dir(settings.input_file_path)
        if not target_folders:
            raise FileNotFoundError(f"No UltraStar song folders with .txt files found in: {settings.input_file_path}")

        # Check if multiple song subfolders were detected (Batch Mode)
        is_batch = len(target_folders) > 1 or (
            len(target_folders) == 1 and os.path.abspath(target_folders[0]) != os.path.abspath(settings.input_file_path)
        )

        if is_batch:
            print(f"{ULTRASINGER_HEAD} Detected batch autocorrect folder with {blue_highlighted(str(len(target_folders)))} song(s).")
            reports = []
            for idx, s_folder in enumerate(target_folders):
                print(f"\n{ULTRASINGER_HEAD} ===================================================")
                print(f"{ULTRASINGER_HEAD} Autocorrecting song [{idx + 1}/{len(target_folders)}]: {blue_highlighted(os.path.basename(s_folder))}")
                print(f"{ULTRASINGER_HEAD} ===================================================")
                try:
                    rep = autocorrect_song(s_folder, pitcher_name=pitcher, device=device)
                    reports.append((os.path.basename(s_folder), rep))
                except Exception as e:
                    print(f"{ULTRASINGER_HEAD} {red_highlighted('Error processing')} {s_folder}: {e}")

            # Consolidated Batch QA Summary Table
            if reports:
                print(f"\n{ULTRASINGER_HEAD} ===================================================================")
                print(f"{ULTRASINGER_HEAD}              {cyan_highlighted('BATCH AUTOCORRECT & QA SUMMARY')}")
                print(f"{ULTRASINGER_HEAD} ===================================================================")
                for s_name, rep in reports:
                    score_diff = rep.corrected_score - rep.original_score
                    score_str = f"{rep.original_score:,} -> {green_highlighted(f'{rep.corrected_score:,}')} ({score_diff:+d})"
                    gap_str = f"GAP: {rep.gap_shift_ms:+.0f}ms"
                    cov_str = f"Cov: {rep.timing_coverage:.1f}%"
                    pitch_str = f"Pitch: {rep.pitch_accuracy:.1f}%"
                    print(f"{ULTRASINGER_HEAD} * {blue_highlighted(s_name)}: {score_str} | {gap_str} | {cov_str} | {pitch_str}")
                print(f"{ULTRASINGER_HEAD} ===================================================================")
        else:
            autocorrect_song(settings.input_file_path, pitcher_name=pitcher, device=device)

        return "", Score(), Score()

    # Check if input path is a batch directory containing multiple song folders
    if os.path.isdir(settings.input_file_path) and find_song_audio_in_dir(settings.input_file_path) is None:
        subdirs = [os.path.join(settings.input_file_path, d) for d in sorted(os.listdir(settings.input_file_path))
                   if os.path.isdir(os.path.join(settings.input_file_path, d))]
        song_dirs = [d for d in subdirs if find_song_audio_in_dir(d) is not None]
        if song_dirs:
            print(f"{ULTRASINGER_HEAD} Detected batch folder with {blue_highlighted(str(len(song_dirs)))} song(s).")
            last_result = None
            orig_input = settings.input_file_path
            orig_output = settings.output_folder_path
            orig_lyrics = settings.lyrics_file_path
            orig_genre = settings.genre
            orig_genre_file = settings.genre_file_path
            orig_language = settings.language
            for idx, s_dir in enumerate(song_dirs):
                print(f"\n{ULTRASINGER_HEAD} ===================================================")
                print(f"{ULTRASINGER_HEAD} Processing song [{idx + 1}/{len(song_dirs)}]: {blue_highlighted(os.path.basename(s_dir))}")
                print(f"{ULTRASINGER_HEAD} ===================================================")
                settings.input_file_path = s_dir
                settings.output_folder_path = orig_output
                settings.lyrics_file_path = None
                settings.genre = orig_genre
                settings.genre_file_path = None
                settings.language = orig_language
                last_result = run_single_song()
            settings.input_file_path = orig_input
            settings.output_folder_path = orig_output
            settings.lyrics_file_path = orig_lyrics
            settings.genre = orig_genre
            settings.genre_file_path = orig_genre_file
            settings.language = orig_language
            return last_result

    return run_single_song()


def run_single_song() -> tuple[str, Score, Score]:
    """Process a single song through the UltraSinger pipeline"""
    #List selected options (can add more later)
    if settings.keep_numbers:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Numbers will be transcribed as numerics (i.e. 1, 2, 3, etc.)')}")
    if settings.create_plot:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Plot will be created')}")
    if settings.keep_cache:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Cache folder will not be deleted')}")
    if settings.create_audio_chunks:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Audio chunks will be created')}")
    if not settings.create_karaoke:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Karaoke txt will not be created')}")
    if not settings.use_separated_vocal:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Vocals will not be separated')}")
    if not settings.hyphenation:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Hyphenation will not be applied')}")
    if settings.quantize_to_key:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Notes will be quantized to the detected musical key')}")
    if not settings.onset_snapping:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Onset snapping will not be applied')}")
    if settings.separator != "demucs":
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted(f'Vocal separator: {settings.separator}')}")
    if settings.pitcher != "swiftf0":
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted(f'Pitch extractor: {settings.pitcher}')}")

    process_data = InitProcessData()

    process_data.process_data_paths.cache_folder_path = (
        os.path.join(settings.output_folder_path, "cache")
        if settings.cache_override_path is None
        else settings.cache_override_path
    )

    # Create process audio
    process_data.process_data_paths.processing_audio_path = CreateProcessAudio(process_data)

    # Get BPM from wav file
    if not settings.input_file_is_ultrastar_txt:
        if settings.bpm_override is not None:
            process_data.media_info.bpm = float(settings.bpm_override)
            print(f"{ULTRASINGER_HEAD} Using manual BPM {blue_highlighted(str(round(process_data.media_info.bpm, 2)))}")
        else:
            bpm_audio_path = (
                process_data.process_data_paths.audio_output_file_path
                if os_helper.check_file_exists(process_data.process_data_paths.audio_output_file_path)
                else process_data.process_data_paths.processing_audio_path
            )
            process_data.media_info.bpm = get_bpm_from_file(bpm_audio_path)

    # Detect key
    detected_key, detected_mode = detect_key_from_audio(process_data.process_data_paths.processing_audio_path)
    if process_data.media_info.music_key is None:
        process_data.media_info.music_key = f"{detected_key} {detected_mode}"

    # Audio transcription (with auto-detection of lyrics language to prevent cross-language batch translation)
    lyrics_file_check = getattr(process_data, "lyrics_file_path", None) or settings.lyrics_file_path
    if lyrics_file_check and os_helper.check_file_exists(lyrics_file_check):
        from modules.Speech_Recognition.lyrics_aligner import detect_language_from_lyrics
        detected_lyrics_lang = detect_language_from_lyrics(lyrics_file_check, settings.language)
        if detected_lyrics_lang and detected_lyrics_lang != settings.language:
            print(
                f"{ULTRASINGER_HEAD} {gold_highlighted('Lyrics Language Guard:')} "
                f"Reference lyrics detected as '{blue_highlighted(detected_lyrics_lang)}' "
                f"(overriding CLI '--language {settings.language}')."
            )
            settings.language = detected_lyrics_lang

    process_data.media_info.language = settings.language
    if not settings.ignore_audio:
        TranscribeAudio(process_data)

    if settings.to_whisper_out:
        target_dir = os.path.dirname(settings.input_file_path)
        input_whisper_out = os.path.join(target_dir, "whisper.txt")
        export_whisper_transcription_to_file(process_data.transcribed_data, input_whisper_out)
        print(f"\n{ULTRASINGER_HEAD} {green_highlighted('Success:')} Whisper transcription saved to {blue_highlighted(input_whisper_out)}")
        print(f"{ULTRASINGER_HEAD} {cyan_highlighted('Option --to_whisper_out active:')} Skipping pitch extraction, MIDI, and output generation.")
        return "", None, None

    # Split syllables into segments
    if not settings.ignore_audio:
        process_data.transcribed_data = rebalance_monosyllable_prepositions(process_data.transcribed_data)
        process_data.transcribed_data = split_syllables_into_segments(process_data.transcribed_data,
                                                                  process_data.media_info.bpm)

    # Create audio chunks
    if settings.create_audio_chunks:
        create_audio_chunks(process_data)

    # Pitch audio
    process_data.pitched_data = pitch_audio(process_data.process_data_paths)

    # Allowed keys for quantization
    allowed_notes_for_key = None
    if settings.quantize_to_key and not settings.ignore_audio:
        allowed_notes_for_key = get_allowed_notes_for_key(detected_key, detected_mode)

    # Create Midi_Segments
    if not settings.ignore_audio:
        active_genre_str = process_data.media_info.genre or settings.genre or "Pop"
        active_genre_prof = resolve_genre_profile(active_genre_str)
        effective_melisma_thresh = max(
            float(settings.melisma_threshold),
            active_genre_prof.vibrato_tolerance_semitones + 0.3,
        )
        process_data.midi_segments = create_midi_segments_from_transcribed_data(
            process_data.transcribed_data,
            process_data.pitched_data,
            allowed_notes_for_key,
            disable_melismas=settings.disable_melismas,
            melisma_threshold=effective_melisma_thresh,
            melisma_min_duration=settings.melisma_min_duration,
            vibrato_freq_range=active_genre_prof.vibrato_freq_range,
            vibrato_max_amplitude_st=active_genre_prof.vibrato_max_amplitude_st,
        )
    else:
        process_data.midi_segments = create_repitched_midi_segments_from_ultrastar_txt(process_data.pitched_data,
                                                                                       process_data.parsed_file)

    # Merge syllable segments
    if not settings.ignore_audio:
        process_data.midi_segments, process_data.transcribed_data = merge_syllable_segments(process_data.midi_segments,
                                                                                        process_data.transcribed_data,
                                                                                        process_data.media_info.bpm)

    # Magenta humanization (MusicVAE / Performance RNN expressive dynamics and timing)
    if settings.magenta_humanize and not settings.ignore_audio:
        from modules.Midi.Extractors import apply_magenta_humanize
        audio_dur = getattr(process_data.media_info, "duration", None)
        process_data.midi_segments = apply_magenta_humanize(
            process_data.midi_segments,
            audio_duration=audio_dur,
        )

    # Create plot
    if settings.create_plot:
        create_plots(process_data, settings.output_folder_path)

    # Create Ultrastar txt
    accurate_score, simple_score, ultrastar_file_output = CreateUltraStarTxt(process_data)

    # Create Midi
    if settings.create_midi:
        vocal_source = (
            process_data.process_data_paths.vocals_audio_file_path
            if os_helper.check_file_exists(process_data.process_data_paths.vocals_audio_file_path)
            else process_data.process_data_paths.processing_audio_path
        )
        create_midi_file(
            process_data.media_info.bpm,
            settings.output_folder_path,
            process_data.midi_segments,
            process_data.basename,
            audio_path=vocal_source,
            pitched_data=process_data.pitched_data,
        )

    # Sheet music
    create_sheet(process_data.midi_segments, settings.output_folder_path,
                 process_data.process_data_paths.cache_folder_path, settings.musescore_path, process_data.basename,
                 process_data.media_info)

    # Automated Headless Visual QA Plotter
    try:
        from modules.Audio.visual_qa_plotter import generate_visual_qa_plot
        vocal_source = (
            process_data.process_data_paths.vocals_audio_file_path
            if os_helper.check_file_exists(process_data.process_data_paths.vocals_audio_file_path)
            else process_data.process_data_paths.processing_audio_path
        )
        generate_visual_qa_plot(
            txt_path=ultrastar_file_output,
            vocal_audio_path=vocal_source,
            output_folder=settings.output_folder_path,
            pitched_data=process_data.pitched_data,
            pitcher_name=settings.pitcher,
            device=settings.pytorch_device,
        )
    except Exception as qa_err:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Visual QA Plot warning:')} {qa_err}")

    # Cleanup
    if not settings.keep_cache:
        remove_cache_folder(process_data.process_data_paths.cache_folder_path)

    # Print Support
    print_support()
    return ultrastar_file_output, simple_score, accurate_score


def split_syllables_into_segments(
        transcribed_data: list[TranscribedData],
        real_bpm: float) -> list[TranscribedData]:
    """Pass syllables intact to MIDI creation where acoustic F0 melisma segmentation will occur."""
    return transcribed_data


def merge_syllable_segments(midi_segments: list[MidiSegment],
                            transcribed_data: list[TranscribedData],
                            real_bpm: float) -> tuple[list[MidiSegment], list[TranscribedData]]:
    """Preserve all syllable segments intact so each syllable is sung as a distinct note."""
    return midi_segments, transcribed_data


def create_audio_chunks(process_data):
    if not settings.ignore_audio:
        create_audio_chunks_from_transcribed_data(
            process_data.process_data_paths,
            process_data.transcribed_data)
    else:
        create_audio_chunks_from_ultrastar_data(
            process_data.process_data_paths,
            process_data.parsed_file
        )


def find_song_audio_in_dir(directory: str) -> str | None:
    """Find the main audio/video file inside a song folder."""
    if not os.path.isdir(directory):
        return None
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    candidates = []
    audio_exts = ('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.opus', '.mp4', '.mkv')
    ignored = ('[vocals]', '[instrumental]', '[sync_test]', '[karaoke]')
    for f in files:
        f_lower = f.lower()
        if any(f_lower.endswith(ext) for ext in audio_exts):
            if not any(ign in f_lower for ign in ignored):
                candidates.append(os.path.join(directory, f))
    if not candidates:
        return None
    folder_name = os.path.basename(os.path.normpath(directory)).lower()
    for c in candidates:
        if os.path.splitext(os.path.basename(c))[0].lower() == folder_name:
            return c
    return candidates[0]


def find_lyrics_in_dir(directory: str) -> str | None:
    """Find lyrics.txt in a directory."""
    if not os.path.isdir(directory):
        return None
    for name in ["lyrics.txt", "letra.txt", "letras.txt", "lyrics.lrc"]:
        p = os.path.join(directory, name)
        if os.path.isfile(p):
            return p
    return None


def find_genre_in_dir(directory: str) -> str | None:
    """Find GENRE.txt or genre.txt in a directory."""
    if not os.path.isdir(directory):
        return None
    for name in ["GENRE.txt", "genre.txt", "GENRE", "genre"]:
        p = os.path.join(directory, name)
        if os.path.isfile(p):
            return p
    return None


def parse_genre_file(file_path: str) -> str:
    """Read and clean musical genre from file, supporting '#GENRE: Rock' or raw 'Rock'."""
    try:
        with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
            content = f.read().strip()
        clean = re.sub(r"^#?genre\s*:\s*", "", content, flags=re.IGNORECASE).strip()
        first_line = [l.strip() for l in clean.splitlines() if l.strip()]
        return first_line[0] if first_line else "Pop"
    except Exception:
        return "Pop"


def InitProcessData():
    if not settings.input_file_path.startswith("https:") and not os.path.exists(settings.input_file_path):
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Error: Input file or directory does not exist:')} {blue_highlighted(settings.input_file_path)}")
        sys.exit(1)

    # Check if input path is a folder containing an audio file directly (e.g. /input/Artist - Song)
    if os.path.isdir(settings.input_file_path):
        audio_file = find_song_audio_in_dir(settings.input_file_path)
        if audio_file:
            lyrics_path = find_lyrics_in_dir(settings.input_file_path)
            if lyrics_path:
                settings.lyrics_file_path = lyrics_path
            genre_path = find_genre_in_dir(settings.input_file_path)
            if genre_path:
                settings.genre_file_path = genre_path
                settings.genre = parse_genre_file(genre_path)
                print(f"{ULTRASINGER_HEAD} Found musical genre file: {blue_highlighted(settings.genre)} ({genre_path})")
            settings.input_file_path = audio_file
    elif os.path.isfile(settings.input_file_path):
        parent_dir = os.path.dirname(settings.input_file_path)
        if not settings.lyrics_file_path:
            lyrics_path = find_lyrics_in_dir(parent_dir)
            if lyrics_path:
                settings.lyrics_file_path = lyrics_path
                print(f"{ULTRASINGER_HEAD} Found reference lyrics file: {blue_highlighted(lyrics_path)}")
        if not settings.genre_file_path:
            genre_path = find_genre_in_dir(parent_dir)
            if genre_path:
                settings.genre_file_path = genre_path
                settings.genre = parse_genre_file(genre_path)
                print(f"{ULTRASINGER_HEAD} Found musical genre file: {blue_highlighted(settings.genre)} ({genre_path})")

    settings.input_file_is_ultrastar_txt = settings.input_file_path.endswith(".txt")
    if settings.input_file_is_ultrastar_txt:
        # Parse Ultrastar txt
        (
            basename,
            settings.output_folder_path,
            audio_file_path,
            ultrastar_class,
            audio_extension,
        ) = parse_ultrastar_txt(settings.input_file_path, settings.output_folder_path)
        process_data = from_ultrastar_txt(ultrastar_class)
        process_data.basename = basename
        process_data.process_data_paths.audio_output_file_path = audio_file_path
        process_data.media_info.audio_extension = audio_extension
        # todo: ignore transcribe
        settings.ignore_audio = True

    elif settings.input_file_path.startswith("https:"):
        # Youtube
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Full Automatic Mode')}")
        process_data = ProcessData()
        (
            process_data.basename,
            settings.output_folder_path,
            process_data.process_data_paths.audio_output_file_path,
            process_data.media_info
        ) = download_from_youtube(settings.input_file_path, settings.output_folder_path, settings.cookiefile)
    else:
        # Audio/Video File
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Full Automatic Mode')}")
        process_data = ProcessData()
        (
            process_data.basename,
            settings.output_folder_path,
            process_data.process_data_paths.audio_output_file_path,
            process_data.media_info,
        ) = infos_from_audio_video_input_file()
        process_data.lyrics_file_path = settings.lyrics_file_path

    if settings.genre and (not process_data.media_info.genre or process_data.media_info.genre == "Pop"):
        process_data.media_info.genre = settings.genre
    return process_data


def export_whisper_transcription_to_file(transcribed_data, target_file_path: str) -> None:
    """Format and write pure Whisper transcribed words to whisper.txt."""
    if not transcribed_data:
        with open(target_file_path, "w", encoding="utf-8") as f:
            f.write("")
        return

    lines = []
    current_words = []

    for i, td in enumerate(transcribed_data):
        word = td.word.strip()
        if not word:
            continue
        current_words.append(word)

        should_break = False
        if i < len(transcribed_data) - 1:
            silence_gap = transcribed_data[i + 1].start - td.end
            # Break line if there is a silence pause >= 0.7s, sentence punctuation, or line length >= 9 words
            if silence_gap >= 0.7:
                should_break = True
            elif any(word.endswith(p) for p in [".", "?", "!", ";"]) and len(current_words) >= 3:
                should_break = True
            elif len(current_words) >= 9:
                should_break = True
        else:
            should_break = True

        if should_break:
            lines.append(" ".join(current_words))
            current_words = []

    if current_words:
        lines.append(" ".join(current_words))

    content = "\n".join(lines) + "\n"
    with open(target_file_path, "w", encoding="utf-8") as f:
        f.write(content)


def TranscribeAudio(process_data):
    transcription_result = transcribe_audio(process_data.process_data_paths.cache_folder_path,
                                            process_data.process_data_paths.processing_audio_path)

    if process_data.media_info.language is None:
        process_data.media_info.language = transcription_result.detected_language

    process_data.transcribed_data = transcription_result.transcribed_data

    # Save pure raw Whisper transcription to output/<Song>/whisper.txt before forced alignment
    if settings.output_folder_path and os_helper.check_if_folder_exists(settings.output_folder_path):
        whisper_out_path = os.path.join(settings.output_folder_path, "whisper.txt")
        export_whisper_transcription_to_file(transcription_result.transcribed_data, whisper_out_path)
        print(f"{ULTRASINGER_HEAD} Raw Whisper transcription saved to {blue_highlighted(whisper_out_path)}")
    lyrics_file = getattr(process_data, "lyrics_file_path", None) or settings.lyrics_file_path
    if not settings.to_whisper_out and lyrics_file and os_helper.check_file_exists(lyrics_file):
        from modules.Speech_Recognition.lyrics_aligner import load_lyrics_file, align_transcribed_data_with_lyrics
        print(f"{ULTRASINGER_HEAD} Applying forced alignment with reference lyrics: {blue_highlighted(lyrics_file)}")
        lyrics_words = load_lyrics_file(lyrics_file)
        vocal_align_audio = (
            process_data.process_data_paths.vocals_audio_file_path
            if os_helper.check_file_exists(process_data.process_data_paths.vocals_audio_file_path)
            else process_data.process_data_paths.processing_audio_path
        )
        process_data.transcribed_data = align_transcribed_data_with_lyrics(
            process_data.transcribed_data,
            lyrics_words,
            audio_path=vocal_align_audio,
            language=process_data.media_info.language,
            device=settings.pytorch_device,
        )

    # Pre-hyphenation word-level vocal tail trimming:
    # Cuts off hallucinated silence tails from words (e.g. Whisper stretching 'nena' across 5.5s of dead air)
    # ensuring syllables are strictly contained within actual acoustic vibration.
    if settings.trim_energy and not settings.ignore_audio:
        pre_trim_audio_path = (
            process_data.process_data_paths.vocals_audio_file_path
            if os_helper.check_file_exists(process_data.process_data_paths.vocals_audio_file_path)
            else process_data.process_data_paths.processing_audio_path
        )
        process_data.transcribed_data = trim_syllable_ends_with_energy(
            pre_trim_audio_path,
            process_data.transcribed_data,
        )

    # Hyphen
    # Todo: Is it really unnecessary?
    remove_unecessary_punctuations(process_data.transcribed_data)
    if settings.hyphenation:
        hyphen_words = hyphenate_each_word(process_data.media_info.language, process_data.transcribed_data)

        if hyphen_words is not None:
            process_data.transcribed_data = add_hyphen_to_data(process_data.transcribed_data, hyphen_words)

    process_data.transcribed_data = remove_silence_from_transcription_data(
        process_data.process_data_paths.processing_audio_path, process_data.transcribed_data
    )

    if settings.onset_snapping and not settings.ignore_audio:
        onset_audio_path = (
            process_data.process_data_paths.vocals_audio_file_path
            if os_helper.check_file_exists(process_data.process_data_paths.vocals_audio_file_path)
            else process_data.process_data_paths.processing_audio_path
        )
        genre_prof = resolve_genre_profile(settings.genre)
        process_data.transcribed_data = snap_transcribed_data_to_onsets(
            onset_audio_path,
            process_data.transcribed_data,
            snap_window_ms=settings.onset_window_ms,
            phrase_snap_window_ms=settings.phrase_snap_window_ms,
            rhythm_grid_mode=getattr(genre_prof, "rhythm_grid_mode", "straight"),
        )

    if settings.trim_energy and not settings.ignore_audio:
        trim_audio_path = (
            process_data.process_data_paths.vocals_audio_file_path
            if os_helper.check_file_exists(process_data.process_data_paths.vocals_audio_file_path)
            else process_data.process_data_paths.processing_audio_path
        )
        process_data.transcribed_data = trim_syllable_ends_with_energy(
            trim_audio_path,
            process_data.transcribed_data,
        )

    if not settings.ignore_audio:
        vocal_path = (
            process_data.process_data_paths.vocals_audio_file_path
            if os_helper.check_file_exists(process_data.process_data_paths.vocals_audio_file_path)
            else process_data.process_data_paths.processing_audio_path
        )
        validate_and_calibrate_gap(
            vocal_path,
            process_data.transcribed_data,
            auto_calibrate=settings.auto_calibrate_gap,
            max_calibrate_ms=settings.max_calibrate_ms,
        )
        process_data.transcribed_data = rebalance_monosyllable_prepositions(process_data.transcribed_data)


def CreateUltraStarTxt(process_data: ProcessData):
    # Move instrumental and vocals
    if settings.create_karaoke and version.parse(settings.format_version.value) < version.parse(
            FormatVersion.V1_1_0.value):
        karaoke_output_path = os.path.join(settings.output_folder_path, process_data.basename + " [Karaoke]." + process_data.media_info.audio_extension)
        convert_audio_format(process_data.process_data_paths.instrumental_audio_file_path, karaoke_output_path)

    if version.parse(settings.format_version.value) >= version.parse(FormatVersion.V1_1_0.value):
        instrumental_output_path = os.path.join(settings.output_folder_path,
                                                process_data.basename + " [Instrumental]." + process_data.media_info.audio_extension)
        convert_audio_format(process_data.process_data_paths.instrumental_audio_file_path, instrumental_output_path)
        vocals_output_path = os.path.join(settings.output_folder_path, process_data.basename + " [Vocals]." + process_data.media_info.audio_extension)
        convert_audio_format(process_data.process_data_paths.vocals_audio_file_path, vocals_output_path)

    # Create Ultrastar txt
    phrase_player_ids = None
    if settings.duet and not settings.ignore_audio:
        vocal_source = (
            process_data.process_data_paths.vocals_audio_file_path
            if os_helper.check_file_exists(process_data.process_data_paths.vocals_audio_file_path)
            else process_data.process_data_paths.processing_audio_path
        )
        silence_split_dur = ultrastar_writer.calculate_silent_beat_length(process_data.midi_segments)
        phrase_player_ids, _, settings.p1_name, settings.p2_name = separate_duet_phrases(
            vocal_source,
            process_data.midi_segments,
            silence_split_dur,
            p1_name=settings.p1_name,
            p2_name=settings.p2_name,
            engine=settings.duet_engine,
        )

    if not settings.ignore_audio:
        ultrastar_file_output = create_ultrastar_txt_from_automation(
            process_data.basename,
            settings.output_folder_path,
            process_data.midi_segments,
            process_data.media_info,
            settings.format_version,
            settings.create_karaoke,
            settings.APP_VERSION,
            settings.octave_offset,
            settings.auto_octave,
            settings.duet,
            settings.p1_name,
            settings.p2_name,
            phrase_player_ids,
        )
    else:
        ultrastar_file_output = create_ultrastar_txt_from_midi_segments(
            settings.output_folder_path, settings.input_file_path, process_data.media_info.title,
            process_data.midi_segments
        )

    # Calc Points
    simple_score = None
    accurate_score = None
    if settings.calculate_score:
        simple_score, accurate_score = calculate_score_points(process_data, ultrastar_file_output)

    # Add calculated score to Ultrastar txt
    #Todo: Missing Karaoke
    ultrastar_writer.add_score_to_ultrastar_txt(ultrastar_file_output, simple_score)

    # Create Sync Test Audio (click track for alignment verification)
    if settings.create_click_track and not settings.ignore_audio:
        click_track_output_path = os.path.join(
            settings.output_folder_path,
            process_data.basename + " [Sync_Test]." + process_data.media_info.audio_extension
        )
        vocal_source = (
            vocals_output_path
            if 'vocals_output_path' in locals() and os_helper.check_file_exists(vocals_output_path)
            else process_data.process_data_paths.vocals_audio_file_path
        )
        create_sync_click_track(
            vocal_source,
            click_track_output_path,
            process_data.midi_segments,
        )

    return accurate_score, simple_score, ultrastar_file_output


def CreateProcessAudio(process_data) -> str:
    # Set processing audio to cache file
    process_data.process_data_paths.processing_audio_path = os.path.join(
        process_data.process_data_paths.cache_folder_path, process_data.basename + ".wav"
    )
    os_helper.create_folder(process_data.process_data_paths.cache_folder_path)

    # Separate vocal from audio
    audio_separation_folder_path = separate_vocal_from_audio(
        process_data.process_data_paths.cache_folder_path,
        process_data.process_data_paths.audio_output_file_path,
        settings.use_separated_vocal,
        settings.create_karaoke,
        settings.pytorch_device,
        settings.demucs_model,
        settings.skip_cache_vocal_separation,
        separator_name=settings.separator_engine or settings.separator,
        bs_roformer_model=settings.bs_roformer_model,
        dereverb_engine=settings.dereverb,
    )
    process_data.process_data_paths.vocals_audio_file_path = os.path.join(audio_separation_folder_path, "vocals.wav")
    process_data.process_data_paths.instrumental_audio_file_path = os.path.join(audio_separation_folder_path,
                                                                                "no_vocals.wav")

    # Free GPU VRAM after separation
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if settings.use_separated_vocal:
        input_path = process_data.process_data_paths.vocals_audio_file_path
    else:
        input_path = process_data.process_data_paths.audio_output_file_path

    # Denoise vocal audio
    denoised_output_path = os.path.join(
        process_data.process_data_paths.cache_folder_path, process_data.basename + "_denoised.wav"
    )
    denoise_vocal_audio(input_path, denoised_output_path, settings.skip_cache_denoise_vocal_audio)

    # Convert to mono audio
    mono_output_path = os.path.join(
        process_data.process_data_paths.cache_folder_path, process_data.basename + "_mono.wav"
    )
    convert_audio_to_mono_wav(denoised_output_path, mono_output_path)

    # Mute silence sections
    mute_output_path = os.path.join(
        process_data.process_data_paths.cache_folder_path, process_data.basename + "_mute.wav"
    )
    mute_no_singing_parts(mono_output_path, mute_output_path)

    # Define the audio file to process
    return mute_output_path


def transcribe_audio(cache_folder_path: str, processing_audio_path: str) -> TranscriptionResult:
    """Transcribe audio with AI"""
    transcription_result = None
    whisper_align_model_string = None
    if settings.transcriber == "whisper":
        if not settings.whisper_align_model is None:
            whisper_align_model_string = settings.whisper_align_model.replace("/", "_")
        whisper_device = "cpu" if settings.force_whisper_cpu else settings.pytorch_device
        genre_cache_string = re.sub(r"[^\w\-]", "_", str(settings.genre or "Pop"))
        aligner_str = str(settings.aligner_engine or "wav2vec2")
        transcription_config = f"{settings.transcriber}_{settings.whisper_model.value}_{whisper_device}_{whisper_align_model_string}_{settings.whisper_batch_size}_{settings.whisper_compute_type}_{settings.language}_{genre_cache_string}_{aligner_str}"
        transcription_path = os.path.join(cache_folder_path, f"{transcription_config}.json")
        cached_transcription_available = check_file_exists(transcription_path)
        if settings.skip_cache_transcription or not cached_transcription_available:
            transcription_result = transcribe_with_whisper(
                processing_audio_path,
                settings.whisper_model,
                whisper_device,
                settings.whisper_align_model,
                settings.whisper_batch_size,
                settings.whisper_compute_type,
                settings.language,
                settings.keep_numbers,
                genre=settings.genre,
                aligner_engine=settings.aligner_engine,
            )
            with open(transcription_path, "w", encoding=FILE_ENCODING) as file:
                file.write(transcription_result.to_json())
        else:
            print(f"{ULTRASINGER_HEAD} {green_highlighted('cache')} reusing cached transcribed data")
            with open(transcription_path) as file:
                json = file.read()
                transcription_result = TranscriptionResult.from_json(json)
    else:
        raise NotImplementedError
    return transcription_result


def extract_artist_title_from_name(name: str) -> tuple[str | None, str | None]:
    """
    Extracts (artist, title) from a filename or directory name.
    Supports flexible separators: ' - ', ' – ', ' — ', ' _ ', '- ', ' -', or unspaced '-' (with >= 2 chars on each side).
    """
    if not name:
        return None, None
    name = name.strip()

    # 1. Spaced separators
    for sep in [" - ", " – ", " — ", " _ ", "- ", " -"]:
        if sep in name:
            parts = name.split(sep, 1)
            p1, p2 = parts[0].strip(), parts[1].strip()
            if p1 and p2:
                return p1, p2

    # 2. Unspaced hyphen / dash e.g. 'Artist-Title' (requiring >= 2 characters on each side)
    import re
    match = re.match(r"^(.{2,}?)[-–—](.{2,})$", name)
    if match:
        p1, p2 = match.group(1).strip(), match.group(2).strip()
        if p1 and p2:
            return p1, p2

    return None, name


def infos_from_audio_video_input_file() -> tuple[str, str, str, MediaInfo]:
    """Infos from audio/video input file"""
    basename = os.path.basename(settings.input_file_path)
    basename_without_ext = os.path.splitext(basename)[0].strip()
    parent_dir = os.path.basename(os.path.dirname(settings.input_file_path)).strip()

    artist = None
    title = None

    generic_dirs = {"input", "entrada", "output", "salida", "cache", "music", "audio"}

    # Priority 1: Check audio/video filename
    parsed_artist, parsed_title = extract_artist_title_from_name(basename_without_ext)
    if parsed_artist:
        artist, title = parsed_artist, parsed_title
    else:
        # Priority 2: If filename had no artist separator, check parent directory name
        if parent_dir and parent_dir.lower() not in generic_dirs:
            parent_artist, parent_title = extract_artist_title_from_name(parent_dir)
            if parent_artist:
                artist, title = parent_artist, parent_title
            else:
                title = parsed_title or parent_title or basename_without_ext
        else:
            title = parsed_title or basename_without_ext

    user_has_explicit_artist = (artist is not None and len(artist.strip()) > 0)
    user_has_explicit_title = (title is not None and len(title.strip()) > 0)

    # Search Musicbrainz for metadata (year, cover, genre) unless disabled
    if settings.disable_musicbrainz:
        from modules.musicbrainz_client import SongInfo
        song_info = SongInfo(title=title or basename_without_ext, artist=artist or "Unknown Artist")
    else:
        song_info = search_musicbrainz(title, artist)

    # NEVER overwrite user's explicit artist or title with MusicBrainz fuzzy results!
    if user_has_explicit_artist:
        song_info.artist = artist
    if user_has_explicit_title:
        song_info.title = title

    artist_name = song_info.artist or "Unknown Artist"
    title_name = song_info.title or title or "Unknown Title"

    if settings.keep_input_names:
        # Keep exact input folder name or file basename verbatim
        if parent_dir and parent_dir.lower() not in generic_dirs:
            basename_without_ext = parent_dir
        else:
            basename_without_ext = os.path.splitext(basename)[0].strip()
    else:
        basename_without_ext = f"{artist_name} - {title_name}"

    song_folder_output_path = os.path.join(settings.output_folder_path, basename_without_ext)
    if not (settings.keep_cache and os_helper.check_if_folder_exists(song_folder_output_path)):
        song_folder_output_path = get_unused_song_output_dir(song_folder_output_path)
    os_helper.create_folder(song_folder_output_path)

    extension = os.path.splitext(basename)[1]
    if is_video_file(settings.input_file_path):
        print(f"{ULTRASINGER_HEAD} Video file detected - separating audio and video")

        video_with_audio_basename = f"{basename_without_ext}{extension}"
        video_with_audio_path = os.path.join(song_folder_output_path, video_with_audio_basename)
        os_helper.copy(settings.input_file_path, video_with_audio_path)

        # Separate audio and video
        ultrastar_audio_input_path, final_video_path, audio_ext, video_ext = separate_audio_video(
            video_with_audio_path, basename_without_ext, song_folder_output_path
        )
    else:
        # Audio file
        basename_with_ext = f"{basename_without_ext}{extension}"
        audio_ext = extension.lstrip('.')
        video_ext = None
        os_helper.copy(settings.input_file_path, song_folder_output_path)
        os_helper.rename(
            os.path.join(song_folder_output_path, os.path.basename(settings.input_file_path)),
            os.path.join(song_folder_output_path, basename_with_ext),
        )
        ultrastar_audio_input_path = os.path.join(song_folder_output_path, basename_with_ext)

    # Todo: Read ID3 tags
    if song_info.cover_image_data is not None:
        save_image(song_info.cover_image_data, basename_without_ext, song_folder_output_path)

    effective_genre = settings.genre if (settings.genre_file_path or settings.genre != "Pop") else (song_info.genres or settings.genre or "Pop")

    return (
        basename_without_ext,
        song_folder_output_path,
        ultrastar_audio_input_path,
        MediaInfo(
            artist=song_info.artist,
            title=song_info.title,
            year=song_info.year,
            genre=effective_genre,
            cover_url=song_info.cover_url,
            audio_extension=audio_ext,
            video_extension=video_ext
        ),
    )


def pitch_audio(
        process_data_paths: ProcessDataPaths) -> PitchedData:
    """Pitch audio"""

    pitching_config = f"{settings.pitcher}_{settings.ignore_audio}"
    pitched_data_path = os.path.join(process_data_paths.cache_folder_path, f"{pitching_config}.json")
    cache_available = check_file_exists(pitched_data_path)

    if settings.skip_cache_pitch_detection or not cache_available:
        force_crepe_cpu = getattr(settings, "force_crepe_cpu", False)
        device = "cpu" if settings.force_cpu or (force_crepe_cpu and settings.pitcher == "crepe") else settings.pytorch_device
        pitcher_kwargs = {}
        if settings.pitcher == "crepe":
            pitcher_kwargs["model_capacity"] = settings.crepe_model_capacity
        pitched_data = get_pitch_with_file(
            process_data_paths.processing_audio_path,
            pitcher_name=settings.pitcher,
            device=device,
            **pitcher_kwargs
        )

        # Free GPU VRAM after pitch extraction
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        pitched_data_json = pitched_data.to_json()
        with open(pitched_data_path, "w", encoding=FILE_ENCODING) as file:
            file.write(pitched_data_json)
    else:
        print(f"{ULTRASINGER_HEAD} {green_highlighted('cache')} reusing cached pitch data")
        with open(pitched_data_path) as file:
            json = file.read()
            pitched_data = PitchedData.from_json(json)

    return pitched_data


def main(argv: list[str]) -> None:
    """Main function"""
    print_version(settings.APP_VERSION)
    init_settings(argv)
    check_requirements()
    if settings.interactive_mode:
        init_settings_interactive(settings)
    run()
    sys.exit()


def check_requirements() -> None:
    if not settings.force_cpu:
        settings.pytorch_device = check_gpu_support()
    print(f"{ULTRASINGER_HEAD} ----------------------")

    if not is_ffmpeg_available(settings.user_ffmpeg_path):
        print(
            f"{ULTRASINGER_HEAD} {red_highlighted('Error:')} {blue_highlighted('FFmpeg')} {red_highlighted('is not available. Provide --ffmpeg ‘path’ or install FFmpeg with PATH')}")
        sys.exit(1)
    else:
        ffmpeg_path, ffprobe_path = get_ffmpeg_and_ffprobe_paths()
        print(f"{ULTRASINGER_HEAD} {blue_highlighted('FFmpeg')} - using {red_highlighted(ffmpeg_path)}")
        print(f"{ULTRASINGER_HEAD} {blue_highlighted('FFprobe')} - using {red_highlighted(ffprobe_path)}")

    print(f"{ULTRASINGER_HEAD} ----------------------")

def remove_cache_folder(cache_folder_path: str) -> None:
    """Remove cache folder"""
    os_helper.remove_folder(cache_folder_path)


def init_settings(argv: list[str]) -> Settings:
    """Init settings"""
    # Sanitize argv: strip leading/trailing whitespace, non-breaking spaces (\xa0), and filter empty tokens
    argv = [arg.strip(" \t\r\n\xa0") for arg in argv if arg.strip(" \t\r\n\xa0")]
    long, short = arg_options()
    opts, args = getopt.gnu_getopt(argv, short, long)
    if len(opts) == 0:
        print_help()
        sys.exit()
    for opt, arg in opts:
        if opt == "-h":
            print_help()
            sys.exit()
        elif opt in ("-i", "--ifile"):
            settings.input_file_path = arg
        elif opt in ("-o", "--ofile"):
            settings.output_folder_path = arg
        elif opt in ("--whisper"):
            settings.transcriber = "whisper"

            #Addition of whisper model choice. Added error handling for unknown models.
            try:
                settings.whisper_model = WhisperModel(arg)
            except ValueError as ve:
                print(f"{ULTRASINGER_HEAD} The model {arg} is not a valid whisper model selection. Please use one of the following models: {blue_highlighted(', '.join([m.value for m in WhisperModel]))}")
                sys.exit()
        elif opt in ("--whisper_align_model"):
            settings.whisper_align_model = arg
        elif opt in ("--whisper_batch_size"):
            settings.whisper_batch_size = int(arg)
        elif opt in ("--whisper_compute_type"):
            settings.whisper_compute_type = arg
        elif opt in ("--keep_numbers"):
            settings.keep_numbers = True
        elif opt in ("--language"):
            settings.language = arg
        elif opt in ("--crepe"):
            settings.crepe_model_capacity = arg
        elif opt in ("--crepe_step_size"):
            settings.crepe_step_size = int(arg)
        elif opt in ("--plot"):
            settings.create_plot = True
        elif opt in ("--midi"):
            settings.create_midi = True
        elif opt in ("--disable_hyphenation"):
            settings.hyphenation = False
        elif opt in ("--disable_separation"):
            settings.use_separated_vocal = False
        elif opt in ("--disable_karaoke"):
            settings.create_karaoke = False
        elif opt in ("--create_audio_chunks"):
            settings.create_audio_chunks = arg
        elif opt in ("--ignore_audio"):
            settings.ignore_audio = True
        elif opt in ("--force_cpu"):
            settings.force_cpu = True
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        elif opt in ("--force_whisper_cpu"):
            settings.force_whisper_cpu = True
        elif opt in ("--force_crepe_cpu"):
            settings.force_crepe_cpu = True
        elif opt in ("--format_version"):
            if arg == FormatVersion.V0_3_0.value:
                settings.format_version = FormatVersion.V0_3_0
            elif arg == FormatVersion.V1_0_0.value:
                settings.format_version = FormatVersion.V1_0_0
            elif arg == FormatVersion.V1_1_0.value:
                settings.format_version = FormatVersion.V1_1_0
            elif arg == FormatVersion.V1_2_0.value:
                settings.format_version = FormatVersion.V1_2_0
            else:
                print(
                    f"{ULTRASINGER_HEAD} {red_highlighted('Error: Format version')} {blue_highlighted(arg)} {red_highlighted('is not supported.')}"
                )
                sys.exit(1)
        elif opt in ("--keep_cache"):
            settings.keep_cache = True
        elif opt in ("--musescore_path"):
            settings.musescore_path = arg
        #Addition of demucs model choice. Work seems to be needed to make sure syntax is same for models. Added error handling for unknown models
        elif opt in ("--demucs"):
            try:
                settings.demucs_model = DemucsModel(arg)
            except ValueError as ve:
                print(f"{ULTRASINGER_HEAD} The model {arg} is not a valid demucs model selection. Please use one of the following models: {blue_highlighted(', '.join([m.value for m in DemucsModel]))}")
                sys.exit()
        elif opt in ("--cookiefile"):
            settings.cookiefile = arg
        elif opt in ("--interactive"):
            settings.interactive_mode = True
        elif opt in ("--quantize_to_key"):
            settings.quantize_to_key = arg
        elif opt in ("--ffmpeg"):
            settings.user_ffmpeg_path = arg
        elif opt in ("--disable_onset_snapping"):
            settings.onset_snapping = False
        elif opt in ("--onset_window_ms"):
            settings.onset_window_ms = float(arg)
        elif opt in ("--bpm"):
            settings.bpm_override = float(arg)
        elif opt in ("--octave_offset"):
            settings.octave_offset = int(arg)
        elif opt in ("--disable_auto_octave"):
            settings.auto_octave = False
        elif opt in ("--disable_trim_energy"):
            settings.trim_energy = False
        elif opt in ("--disable_click_track"):
            settings.create_click_track = False
        elif opt in ("--create_click_track"):
            settings.create_click_track = True
        elif opt in ("--disable_auto_calibrate_gap"):
            settings.auto_calibrate_gap = False
        elif opt in ("--duet"):
            settings.duet = True
        elif opt in ("--p1_name"):
            settings.p1_name = arg
        elif opt in ("--p2_name"):
            settings.p2_name = arg
        elif opt in ("--separator"):
            from modules.Audio.Separators import SeparatorFactory
            valid_separators = SeparatorFactory.available_separators()
            clean_arg = arg.lower().strip().replace("-", "_")
            if clean_arg not in [s.replace("-", "_") for s in valid_separators] and clean_arg not in ["bs_roformer", "roformer", "mel_band_roformer", "melband", "scnet", "scnet_large", "htdemucs"]:
                print(f"{ULTRASINGER_HEAD} {red_highlighted('Invalid separator:')} {blue_highlighted(arg)}. Options: {', '.join(valid_separators)}")
                sys.exit(1)
            settings.separator = arg.lower()
            settings.separator_engine = clean_arg
            if clean_arg in ["mel_band_roformer", "melband", "mel_roformer"]:
                settings.bs_roformer_model = "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"
        elif opt in ("--separator_engine", "--separator-engine"):
            from modules.Audio.Separators import SeparatorFactory
            valid_separators = SeparatorFactory.available_separators()
            arg_clean = arg.lower().strip().replace("-", "_")
            if arg_clean not in [s.replace("-", "_") for s in valid_separators]:
                print(f"{ULTRASINGER_HEAD} {red_highlighted('Invalid separator-engine:')} {blue_highlighted(arg)}. Options: {', '.join(valid_separators)}")
                sys.exit(1)
            settings.separator_engine = arg_clean
            settings.separator = arg_clean
        elif opt in ("--dereverb"):
            settings.dereverb = arg.lower().strip()
        elif opt in ("--asr_engine", "--asr-engine"):
            settings.asr_engine = arg.lower().strip()
            if "turbo" in settings.asr_engine:
                settings.whisper_model = WhisperModel.LARGE_V3_TURBO
            elif "v2" in settings.asr_engine or "large-v2" in settings.asr_engine:
                settings.whisper_model = WhisperModel.LARGE_V2
        elif opt in ("--aligner_engine", "--aligner-engine"):
            settings.aligner_engine = arg.lower().strip()
        elif opt in ("--pitch_engine", "--pitch-engine"):
            from modules.Pitcher.Extractors import PitcherFactory
            valid_pitchers = PitcherFactory.available_pitchers()
            arg_clean = arg.lower().strip().replace("-", "_")
            if arg_clean not in [p.replace("-", "_") for p in valid_pitchers]:
                print(f"{ULTRASINGER_HEAD} {red_highlighted('Invalid pitch-engine:')} {blue_highlighted(arg)}. Options: {', '.join(valid_pitchers)}")
                sys.exit(1)
            settings.pitch_engine = arg_clean
            settings.pitcher = arg_clean
        elif opt in ("--midi_engine", "--midi-engine"):
            from modules.Midi.Extractors import MidiExtractorFactory
            valid_extractors = MidiExtractorFactory.available_extractors()
            arg_clean = arg.lower().strip().replace("-", "_")
            if arg_clean not in [m.replace("-", "_") for m in valid_extractors]:
                print(f"{ULTRASINGER_HEAD} {red_highlighted('Invalid midi-engine:')} {blue_highlighted(arg)}. Options: {', '.join(valid_extractors)}")
                sys.exit(1)
            settings.midi_engine = arg_clean
        elif opt in ("--magenta_humanize", "--magenta-humanize"):
            settings.magenta_humanize = True
        elif opt in ("--duet_engine", "--duet-engine"):
            settings.duet_engine = arg.lower().strip()
        elif opt in ("--pitcher"):
            from modules.Pitcher.Extractors import PitcherFactory
            valid_pitchers = PitcherFactory.available_pitchers()
            if arg.lower() not in valid_pitchers and arg.lower() not in ["swift-f0"]:
                print(f"{ULTRASINGER_HEAD} {red_highlighted('Invalid pitcher:')} {blue_highlighted(arg)}. Options: {', '.join(valid_pitchers)}")
                sys.exit(1)
            settings.pitcher = arg.lower()
            settings.pitch_engine = arg.lower()
        elif opt in ("--bs_roformer_model"):
            settings.bs_roformer_model = arg
        elif opt in ("--crepe_model"):
            settings.crepe_model_capacity = arg
        elif opt in ("--keep_input_names"):
            settings.keep_input_names = True
        elif opt in ("--disable_musicbrainz"):
            settings.disable_musicbrainz = True
        elif opt in ("--to_whisper_out"):
            settings.to_whisper_out = True
        elif opt in ("--disable_melismas"):
            settings.disable_melismas = True
        elif opt in ("--melisma_threshold"):
            settings.melisma_threshold = float(arg)
        elif opt in ("--autocorrect", "--autocorreccion"):
            settings.autocorrect = True
        elif opt in ("--skip_cache_transcription"):
            settings.skip_cache_transcription = True
        elif opt in ("--genre"):
            settings.genre = arg.strip()
    if settings.output_folder_path == "":
        if settings.input_file_path.startswith("https:"):
            dirname = os.getcwd()
        else:
            dirname = os.path.dirname(settings.input_file_path)
        settings.output_folder_path = os.path.join(dirname, "output")

    return settings


#For convenience, made True/False options into noargs
def arg_options():
    short = "hi:o:amv:"
    long = [
        "ifile=",
        "ofile=",
        "crepe=",
        "crepe_step_size=",
        "demucs=",
        "whisper=",
        "whisper_align_model=",
        "whisper_batch_size=",
        "whisper_compute_type=",
        "language=",
        "genre=",
        "plot",
        "midi",
        "disable_hyphenation",
        "disable_separation",
        "disable_karaoke",
        "create_audio_chunks",
        "ignore_audio",
        "force_cpu",
        "force_whisper_cpu",
        "force_crepe_cpu",
        "format_version=",
        "keep_cache",
        "musescore_path=",
        "keep_numbers",
        "quantize_to_key",
        "interactive",
        "cookiefile=",
        "ffmpeg=",
        "disable_onset_snapping",
        "onset_window_ms=",
        "bpm=",
        "octave_offset=",
        "disable_auto_octave",
        "disable_trim_energy",
        "disable_click_track",
        "create_click_track",
        "disable_auto_calibrate_gap",
        "duet",
        "p1_name=",
        "p2_name=",
        "separator=",
        "separator_engine=",
        "separator-engine=",
        "dereverb=",
        "asr_engine=",
        "asr-engine=",
        "aligner_engine=",
        "aligner-engine=",
        "pitch_engine=",
        "pitch-engine=",
        "midi_engine=",
        "midi-engine=",
        "magenta_humanize",
        "magenta-humanize",
        "duet_engine=",
        "duet-engine=",
        "pitcher=",
        "bs_roformer_model=",
        "crepe_model=",
        "keep_input_names",
        "disable_musicbrainz",
        "to_whisper_out",
        "disable_melismas",
        "melisma_threshold=",
        "autocorrect",
        "autocorreccion",
        "skip_cache_transcription",
    ]
    return long, short

if __name__ == "__main__":
    main(sys.argv[1:])
