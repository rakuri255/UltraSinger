"""Ultrastar writer module"""

import re
import numpy as np
try:
    import librosa
except Exception:
    librosa = None
try:
    import langcodes
except Exception:
    langcodes = None
from packaging import version

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted
from modules.Ultrastar.coverter.ultrastar_converter import (
    real_bpm_to_ultrastar_bpm,
    second_to_beat, )
from modules.Ultrastar.coverter.ultrastar_midi_converter import convert_midi_note_to_ultrastar_note
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue, UltrastarTxtTag, UltrastarTxtNoteTypeTag, \
    FILE_ENCODING
try:
    from modules.Ultrastar.ultrastar_score_calculator import Score
except Exception:
    Score = object
from modules.Midi.MidiSegment import MidiSegment


def calculate_optimal_octave_offset(midi_segments: list[MidiSegment]) -> int:
    """Calculate the best octave offset so the song's notes are centered on the UltraStar staff (between -12 and +12)."""
    if not midi_segments:
        return 0
    import pretty_midi
    midi_numbers = []
    for s in midi_segments:
        try:
            clean_note = str(s.note).replace("♯", "#").replace("♭", "b")
            m = pretty_midi.note_name_to_number(clean_note)
            midi_numbers.append(m)
        except Exception:
            pass
    if not midi_numbers:
        return 0
    median_midi = float(np.median(midi_numbers))
    current_center = median_midi - 48
    optimal_offset = -int(round(current_center / 12.0))
    return optimal_offset


def get_multiplier(real_bpm: float) -> int:
    """Calculates the musical binary multiplier (powers of 2: 8, 16, 4, 2, 1) for UltraStar BPM.

    Using a power of 2 ensures musical note subdivisions (quarter=8, 8th=4, 16th=2, 32nd=1 beat)
    map to exact integer beats with 0 fractional quantization error.
    Standard UltraStar songs use BPM around 200-420.
    """
    if real_bpm <= 0:
        raise Exception("BPM is 0")

    # Target UltraStar BPM between 180 and 420 with power of 2 multiplier
    for m in [8, 16, 4, 2, 1]:
        if 180.0 <= real_bpm * m <= 420.0:
            return m
    return 8


def get_language_name(language: str) -> str:
    """Creates the language name from the language code"""
    if langcodes is not None:
        try:
            return langcodes.Language.make(language=language).display_name()
        except Exception:
            return language
    return language


def create_ultrastar_txt(
        midi_segments: list[MidiSegment],
        ultrastar_file_output: str,
        ultrastar_class: UltrastarTxtValue,
        real_bpm: float,
        octave_offset: int = None,
        auto_octave: bool = True,
        is_duet: bool = False,
        p1_name: str = "Singer 1",
        p2_name: str = "Singer 2",
        phrase_player_ids: list[int] = None) -> None:
    """Creates an Ultrastar txt file from the automation data"""

    print(f"{ULTRASINGER_HEAD} Creating UltraStar file {ultrastar_file_output}")

    if octave_offset is None and auto_octave:
        octave_offset = calculate_optimal_octave_offset(midi_segments)
    elif octave_offset is None:
        octave_offset = 0

    if octave_offset != 0:
        print(f"{ULTRASINGER_HEAD} Staff centering: applying {blue_highlighted(str(octave_offset))} octave(s) offset to align notes inside pentagram")

    ultrastar_bpm = real_bpm_to_ultrastar_bpm(real_bpm)
    multiplication = get_multiplier(ultrastar_bpm)
    ultrastar_bpm = round(ultrastar_bpm * multiplication, 2)
    silence_split_duration = calculate_silent_beat_length(midi_segments)

    with open(ultrastar_file_output, "w", encoding=FILE_ENCODING) as file:
        gap = midi_segments[0].start

        if version.parse(ultrastar_class.version) >= version.parse("1.0.0"):
            file.write(f"#{UltrastarTxtTag.VERSION.value}:{ultrastar_class.version}\n")
        file.write(f"#{UltrastarTxtTag.ENCODING.value}:UTF8\n")
        file.write(f"#{UltrastarTxtTag.ARTIST.value}:{ultrastar_class.artist}\n")
        file.write(f"#{UltrastarTxtTag.TITLE.value}:{ultrastar_class.title}\n")
        genre_val = (ultrastar_class.genre or "Pop").strip()
        file.write(f"#{UltrastarTxtTag.GENRE.value}:{genre_val}\n")
        if is_duet:
            file.write(f"#{UltrastarTxtTag.P1.value}:{p1_name}\n")
            file.write(f"#{UltrastarTxtTag.P2.value}:{p2_name}\n")
        if ultrastar_class.year is not None:
            file.write(f"#{UltrastarTxtTag.YEAR.value}:{ultrastar_class.year}\n")
        if ultrastar_class.language is not None:
            file.write(f"#{UltrastarTxtTag.LANGUAGE.value}:{get_language_name(ultrastar_class.language)}\n")
        if ultrastar_class.cover is not None:
            file.write(f"#{UltrastarTxtTag.COVER.value}:{ultrastar_class.cover}\n")
        if version.parse(ultrastar_class.version) >= version.parse("1.2.0"):
            if ultrastar_class.coverUrl is not None:
                file.write(f"#{UltrastarTxtTag.COVERURL.value}:{ultrastar_class.coverUrl}\n")
        if ultrastar_class.background is not None:
            file.write(f"#{UltrastarTxtTag.BACKGROUND.value}:{ultrastar_class.background}\n")
        file.write(f"#{UltrastarTxtTag.MP3.value}:{ultrastar_class.mp3}\n")
        if version.parse(ultrastar_class.version) >= version.parse("1.1.0"):
            file.write(f"#{UltrastarTxtTag.AUDIO.value}:{ultrastar_class.audio}\n")
            if ultrastar_class.vocals is not None:
                file.write(f"#{UltrastarTxtTag.VOCALS.value}:{ultrastar_class.vocals}\n")
            if ultrastar_class.instrumental is not None:
                file.write(f"#{UltrastarTxtTag.INSTRUMENTAL.value}:{ultrastar_class.instrumental}\n")
        if ultrastar_class.video is not None:
            file.write(f"#{UltrastarTxtTag.VIDEO.value}:{ultrastar_class.video}\n")
        if ultrastar_class.videoGap is not None:
            file.write(f"#{UltrastarTxtTag.VIDEOGAP.value}:{ultrastar_class.videoGap}\n")
        if version.parse(ultrastar_class.version) >= version.parse("1.2.0"):
            if ultrastar_class.videoUrl is not None:
                file.write(f"#{UltrastarTxtTag.VIDEOURL.value}:{ultrastar_class.videoUrl}\n")
        file.write(f"#{UltrastarTxtTag.BPM.value}:{ultrastar_bpm}\n")  # not the real BPM!
        file.write(f"#{UltrastarTxtTag.GAP.value}:{int(gap * 1000)}\n")
        file.write(f"#{UltrastarTxtTag.RESOLUTION.value}:4\n")
        if version.parse(ultrastar_class.version) >= version.parse("1.1.0"):
            if ultrastar_class.tags is not None:
                file.write(f"#{UltrastarTxtTag.TAGS.value}:{ultrastar_class.tags}\n")
        file.write(f"#{UltrastarTxtTag.CREATOR.value}:{ultrastar_class.creator}\n")
        file.write(f"#{UltrastarTxtTag.COMMENT.value}:{ultrastar_class.comment}\n")

        # Filter out truly empty dummy segments (preserve valid melisma notes with '~')
        filtered_segments: list[MidiSegment] = []
        for s in midi_segments:
            txt = str(s.word).strip()
            if txt:
                filtered_segments.append(s)
            elif filtered_segments:
                # If truly empty segment had trailing space or line end, transfer to real preceding note
                if str(s.word).endswith(" "):
                    filtered_segments[-1].is_word_end = True
                    if not filtered_segments[-1].word.endswith(" "):
                        filtered_segments[-1].word += " "
                if getattr(s, "is_line_end", False):
                    filtered_segments[-1].is_line_end = True

        midi_segments = filtered_segments
        has_lyrics_line_breaks = any(getattr(s, "is_line_end", False) for s in midi_segments)

        # Seconds to beat conversion factor strictly from written UltraStar BPM (zero drift guarantee)
        seconds_to_beat_factor = (4.0 * ultrastar_bpm) / 60.0

        def write_player_track(segments_for_player: list[MidiSegment]):
            previous_end_beat = 0
            words_in_line = 0

            for i, midi_segment in enumerate(segments_for_player):
                t_start_rel = float(midi_segment.start) - float(gap)
                t_end_rel = float(midi_segment.end) - float(gap)

                start_beat_exact = t_start_rel * seconds_to_beat_factor
                end_beat_exact = t_end_rel * seconds_to_beat_factor

                start_beat_int = int(round(start_beat_exact))
                end_beat_int = int(round(end_beat_exact))

                raw_word = str(midi_segment.word)
                is_melisma = (raw_word.strip() == "~" or raw_word == "" or raw_word == " ")
                if not raw_word.strip() and not is_melisma:
                    continue

                if i < len(segments_for_player) - 1:
                    silence = float(segments_for_player[i + 1].start) - float(midi_segment.end)
                else:
                    silence = 0.0

                min_dur_beats = 1 if is_melisma else max(1, int(round(0.060 * seconds_to_beat_factor)))
                start_beat_int = max(start_beat_int, previous_end_beat)
                duration_int = max(min_dur_beats, end_beat_int - start_beat_int)
                previous_end_beat = start_beat_int + duration_int

                has_explicit_space = raw_word.endswith(" ")
                has_word_end_attr = getattr(midi_segment, "is_word_end", False)

                if is_melisma:
                    # USDX standard: tied continuation notes use '~ ' (word end) or '~' (intra-word)
                    if has_explicit_space or has_word_end_attr:
                        clean_word = "~ "
                        is_word_end = True
                    else:
                        clean_word = "~"
                        is_word_end = False
                else:
                    clean_text = raw_word.rstrip(" ")
                    if has_explicit_space or has_word_end_attr:
                        clean_word = clean_text + " "
                        is_word_end = True
                    elif silence >= 0.08 or i == len(segments_for_player) - 1:
                        clean_word = clean_text + " "
                        is_word_end = True
                    else:
                        clean_word = clean_text
                        is_word_end = False

                ultrastar_pitch = convert_midi_note_to_ultrastar_note(midi_segment, octave_offset)
                ultrastar_pitch = max(-24, min(24, ultrastar_pitch))
                note_tag = (
                    UltrastarTxtNoteTypeTag.FREESTYLE.value
                    if getattr(midi_segment, "is_freestyle", False)
                    else UltrastarTxtNoteTypeTag.NORMAL.value
                )
                line = (
                    f"{note_tag} "
                    f"{str(start_beat_int)} "
                    f"{str(duration_int)} "
                    f"{str(ultrastar_pitch)} "
                    f"{clean_word}\n"
                )
                file.write(line)

                if is_word_end:
                    words_in_line += 1

                is_phrase_end = False
                if i != len(segments_for_player) - 1 and is_word_end:
                    if has_lyrics_line_breaks:
                        if getattr(midi_segment, "is_line_end", False):
                            if words_in_line >= 2 or silence >= 0.15:
                                is_phrase_end = True
                        elif words_in_line >= 10 or silence >= 2.0:
                            is_phrase_end = True
                    elif silence_split_duration > 0:
                        if (silence >= 0.65 and words_in_line >= 2) or (silence >= 1.8 and words_in_line >= 1):
                            is_phrase_end = True
                        elif words_in_line >= 3 and silence >= silence_split_duration:
                            is_phrase_end = True
                        elif words_in_line >= 5 and silence >= 0.20:
                            is_phrase_end = True
                        elif words_in_line >= 7 and silence >= 0.10:
                            is_phrase_end = True
                        elif words_in_line >= 9:
                            is_phrase_end = True
                        elif any(clean_word.rstrip().endswith(p) for p in ["?", "!", "."]) and words_in_line >= 3 and silence >= 0.20:
                            is_phrase_end = True

                if i != len(segments_for_player) - 1 and is_phrase_end:
                    t_break = (float(midi_segment.end) + float(segments_for_player[i + 1].start)) / 2.0
                    break_beat_exact = (t_break - float(gap)) * seconds_to_beat_factor
                    break_beat_int = int(round(break_beat_exact))
                    break_beat_int = max(previous_end_beat, break_beat_int)

                    next_start_beat = int(round((float(segments_for_player[i + 1].start) - float(gap)) * seconds_to_beat_factor))
                    if break_beat_int >= next_start_beat:
                        break_beat_int = previous_end_beat

                    linebreak = f"{UltrastarTxtTag.LINEBREAK.value} {str(break_beat_int)}\n"
                    file.write(linebreak)
                    words_in_line = 0

            # Ensure final phrase has a clean line break before next block in duet mode
            if is_duet and segments_for_player and words_in_line > 0:
                final_break_beat = previous_end_beat + 4
                file.write(f"{UltrastarTxtTag.LINEBREAK.value} {str(final_break_beat)}\n")

        # Write singing tracks
        if is_duet:
            from modules.Audio.duet_separator import group_midi_segments_into_phrases
            phrases = group_midi_segments_into_phrases(midi_segments, silence_split_duration)

            if phrase_player_ids:
                for p_idx, p in enumerate(phrases):
                    pid = phrase_player_ids[p_idx] if p_idx < len(phrase_player_ids) else 1
                    for seg in p:
                        seg.player_id = pid

            p1_segments = [s for s in midi_segments if getattr(s, "player_id", 1) in (1, 3)]
            p2_segments = [s for s in midi_segments if getattr(s, "player_id", 1) in (2, 3)]
            if not p2_segments and p1_segments:
                p2_segments = p1_segments

            # UltraStar Deluxe Official Duet Specification:
            # Entire Player 1 track block, followed by entire Player 2 track block
            file.write("P 1\n")
            if p1_segments:
                write_player_track(p1_segments)
            file.write("P 2\n")
            if p2_segments:
                write_player_track(p2_segments)
            file.write(f"{UltrastarTxtTag.FILE_END.value}\n")
        else:
            write_player_track(midi_segments)
            file.write(f"{UltrastarTxtTag.FILE_END.value}\n")


def deviation(silence_parts):
    """Calculates the deviation of the silence parts"""
    if len(silence_parts) < 5:
        return 0

    sorted_parts = sorted(silence_parts)
    filtered_parts = [part for part in sorted_parts if part < sorted_parts[-1]]
    sum_parts = sum(filtered_parts)
    if sum_parts == 0:
        return 0

    mean = sum_parts / len(filtered_parts)
    return mean


def calculate_silent_beat_length(midi_segments: list[MidiSegment]) -> float:
    """Calculate pause threshold for linebreaks so sentences are displayed naturally."""
    print(f"{ULTRASINGER_HEAD} Calculating silence parts for linebreaks.")

    if len(midi_segments) < 5:
        return 0.45

    # Only look at pauses between completed words
    word_silences = []
    for i, data in enumerate(midi_segments):
        if i < len(midi_segments) - 1:
            pause = float(midi_segments[i + 1].start) - float(data.end)
            if str(data.word).endswith(" ") and pause > 0.15:
                word_silences.append(pause)

    if not word_silences:
        return 0.45

    median_pause = float(np.median(word_silences))
    return max(0.38, min(0.70, median_pause))


def create_repitched_txt_from_ultrastar_data(
        input_file: str, note_numbers: list[int], output_repitched_ultrastar: str
) -> None:
    """Creates a repitched ultrastar txt file from the original one"""
    # todo: just add '_repitched' to input_file
    print(f"{ULTRASINGER_HEAD} Creating repitched ultrastar txt -> {input_file}_repitch.txt")

    # todo: to reader
    with open(input_file, "r", encoding=FILE_ENCODING) as file:
        txt = file.readlines()

    i = 0
    # todo: just add '_repitched' to input_file
    with open(output_repitched_ultrastar, "w", encoding=FILE_ENCODING) as file:
        for line in txt:
            if line.startswith(f"{UltrastarTxtNoteTypeTag.NORMAL.value} "):
                parts = re.findall(r"\S+|\s+", line)
                # between are whitespaces
                # [0] :
                # [2] start beat
                # [4] duration
                # [6] pitch
                # [8] word
                parts[6] = str(note_numbers[i])
                delimiter = ""
                file.write(delimiter.join(parts))
                i += 1
            else:
                file.write(line)


def add_score_to_ultrastar_txt(ultrastar_file_output: str, score: Score) -> None:
    """Adds the score to the ultrastar txt file"""
    with open(ultrastar_file_output, "r", encoding=FILE_ENCODING) as file:
        text = file.read()
    text = text.split("\n")

    for i, line in enumerate(text):
        if line.startswith(f"#{UltrastarTxtTag.COMMENT.value}:"):
            text[
                i
            ] = f"{line} | Score: total: {score.score}, notes: {score.notes} line: {score.line_bonus}, golden: {score.golden}"
            break

        if line.startswith((
                f"{UltrastarTxtNoteTypeTag.FREESTYLE.value} ",
                f"{UltrastarTxtNoteTypeTag.NORMAL.value} ",
                f"{UltrastarTxtNoteTypeTag.GOLDEN.value} ",
                f"{UltrastarTxtNoteTypeTag.RAP.value} ",
                f"{UltrastarTxtNoteTypeTag.RAP_GOLDEN.value} ")):
            text.insert(
                i,
                f"#{UltrastarTxtTag.COMMENT.value}: UltraSinger [GitHub] | Score: total: {score.score}, notes: {score.notes} line: {score.line_bonus}, golden: {score.golden}",
            )
            break

    text = "\n".join(text)

    with open(ultrastar_file_output, "w", encoding=FILE_ENCODING) as file:
        file.write(text)


class UltraStarWriter:
    """Docstring"""


COMMON_GENRE_ACRONYMS = {
    "edm": "EDM",
    "r&b": "R&B",
    "dj": "DJ",
    "ost": "OST",
    "idm": "IDM",
    "lo-fi": "Lo-Fi",
    "hi-nrg": "Hi-NRG",
    "k-pop": "K-Pop",
    "j-pop": "J-Pop",
    "c-pop": "C-Pop",
}


def _capitalize_genre_segment(seg: str) -> str:
    seg_clean = seg.strip()
    if not seg_clean:
        return ""
    seg_lower = seg_clean.lower()
    if seg_lower in COMMON_GENRE_ACRONYMS:
        return COMMON_GENRE_ACRONYMS[seg_lower]
    if seg_clean.isupper() and len(seg_clean) <= 4:
        return seg_clean
    return seg_clean.capitalize()


def format_separated_string(data: str) -> str:
    """Format and clean a delimiter-separated string (e.g. genre, tags) into standard comma-separated Title Case."""
    if not data:
        return ""
    temp = re.sub(r'[;/]', ',', data)
    tokens = [s.strip() for s in temp.split(',') if s.strip()]

    formatted_tokens = []
    for token in tokens:
        if token.lower() in COMMON_GENRE_ACRONYMS:
            formatted_tokens.append(COMMON_GENRE_ACRONYMS[token.lower()])
            continue

        space_words = token.split(' ')
        formatted_space_words = []
        for sw in space_words:
            if not sw:
                continue
            if '-' in sw:
                dash_words = sw.split('-')
                formatted_dash = '-'.join(_capitalize_genre_segment(dw) for dw in dash_words)
                formatted_space_words.append(formatted_dash)
            else:
                formatted_space_words.append(_capitalize_genre_segment(sw))

        formatted_tokens.append(' '.join(formatted_space_words))

    return ', '.join(formatted_tokens)
