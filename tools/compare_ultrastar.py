"""Compare two UltraStar TXT files and report quality metrics.

Developer tool for evaluating generated UltraStar files against reference files.
Handles format differences automatically (v1.2.0 pitch convention vs legacy raw MIDI).

Usage:
    uv run python tools/compare_ultrastar.py generated.txt reference.txt
    uv run python tools/compare_ultrastar.py generated.txt reference.txt --json
    uv run python tools/compare_ultrastar.py generated.txt reference.txt --detailed
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median, mean


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Note:
    """A single note parsed from an UltraStar TXT file."""
    note_type: str          # ':', '*', 'F', 'R', 'G'
    start_beat: int
    duration: int
    pitch: int              # raw pitch value from the file
    midi: int               # normalised MIDI note number
    word: str               # lyric text (may be '~' for continuation)
    start_ms: float         # start time in milliseconds
    end_ms: float           # end time in milliseconds


@dataclass
class WordGroup:
    """A word with all its continuation notes grouped together."""
    word: str               # the displayable word (without '~' suffixes)
    notes: list[Note] = field(default_factory=list)
    midi: int = 0           # primary MIDI pitch (first note)
    start_ms: float = 0.0   # start time of the first note
    end_ms: float = 0.0     # end time of the last note


@dataclass
class ParsedFile:
    """Result of parsing an UltraStar TXT file."""
    path: str
    version: str | None     # None for legacy files
    bpm: float              # header BPM value
    gap: float              # header GAP in milliseconds
    notes: list[Note] = field(default_factory=list)
    word_groups: list[WordGroup] = field(default_factory=list)
    linebreak_count: int = 0
    headers: dict = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Complete comparison metrics."""
    # Pitch
    pitch_diffs: list[int] = field(default_factory=list)
    pitch_median: float = 0.0
    pitch_mean: float = 0.0
    pitch_exact_pct: float = 0.0
    pitch_within_2_pct: float = 0.0

    # Intervals
    interval_diffs: list[int] = field(default_factory=list)
    interval_exact_pct: float = 0.0
    interval_within_2_pct: float = 0.0

    # Timing (ms)
    timing_diffs: list[float] = field(default_factory=list)
    timing_median: float = 0.0
    timing_mean: float = 0.0
    timing_within_100: float = 0.0
    timing_within_200: float = 0.0
    timing_within_500: float = 0.0

    # Lyrics
    matched_pairs: int = 0
    gen_word_count: int = 0
    ref_word_count: int = 0
    lyrics_precision: float = 0.0
    lyrics_recall: float = 0.0

    # Structure
    gen_note_count: int = 0
    ref_note_count: int = 0
    gen_linebreaks: int = 0
    ref_linebreaks: int = 0

    # Meta
    gen_bpm: float = 0.0
    ref_bpm: float = 0.0
    gen_gap: float = 0.0
    ref_gap: float = 0.0
    gen_version: str | None = None
    ref_version: str | None = None

    # Detail (word-level pairs)
    word_pairs: list[dict] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialise to JSON string."""
        d = {
            "pitch": {
                "median_diff": self.pitch_median,
                "mean_diff": self.pitch_mean,
                "exact_pct": self.pitch_exact_pct,
                "within_2st_pct": self.pitch_within_2_pct,
                "count": len(self.pitch_diffs),
            },
            "intervals": {
                "exact_pct": self.interval_exact_pct,
                "within_2st_pct": self.interval_within_2_pct,
                "count": len(self.interval_diffs),
            },
            "timing": {
                "median_ms": self.timing_median,
                "mean_ms": self.timing_mean,
                "within_100ms_pct": self.timing_within_100,
                "within_200ms_pct": self.timing_within_200,
                "within_500ms_pct": self.timing_within_500,
                "count": len(self.timing_diffs),
            },
            "lyrics": {
                "matched_pairs": self.matched_pairs,
                "gen_words": self.gen_word_count,
                "ref_words": self.ref_word_count,
                "precision": self.lyrics_precision,
                "recall": self.lyrics_recall,
            },
            "structure": {
                "gen_notes": self.gen_note_count,
                "ref_notes": self.ref_note_count,
                "gen_linebreaks": self.gen_linebreaks,
                "ref_linebreaks": self.ref_linebreaks,
            },
            "meta": {
                "gen_bpm": self.gen_bpm,
                "ref_bpm": self.ref_bpm,
                "gen_gap": self.gen_gap,
                "ref_gap": self.ref_gap,
                "gen_version": self.gen_version,
                "ref_version": self.ref_version,
            },
        }
        return json.dumps(d, indent=2)

    def to_summary(self, detailed: bool = False) -> str:
        """Human-readable summary."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("UltraStar Comparison Report")
        lines.append("=" * 60)

        lines.append(f"\nFormat: generated={self.gen_version or 'legacy'}"
                     f"  reference={self.ref_version or 'legacy'}")
        lines.append(f"BPM:    generated={self.gen_bpm:.2f}"
                     f"  reference={self.ref_bpm:.2f}")
        lines.append(f"GAP:    generated={self.gen_gap:.0f}ms"
                     f"  reference={self.ref_gap:.0f}ms"
                     f"  (diff={self.gen_gap - self.ref_gap:+.0f}ms)")

        lines.append(f"\n--- Pitch ({len(self.pitch_diffs)} matched words) ---")
        lines.append(f"  Median offset:  {self.pitch_median:+.1f} semitones")
        lines.append(f"  Mean offset:    {self.pitch_mean:+.1f} semitones")
        lines.append(f"  Exact (0 ST):   {self.pitch_exact_pct:.1f}%")
        lines.append(f"  Within +/-2 ST: {self.pitch_within_2_pct:.1f}%")

        lines.append(f"\n--- Intervals ({len(self.interval_diffs)} consecutive pairs) ---")
        lines.append(f"  Exact:          {self.interval_exact_pct:.1f}%")
        lines.append(f"  Within +/-2 ST: {self.interval_within_2_pct:.1f}%")

        lines.append(f"\n--- Timing ({len(self.timing_diffs)} pairs, excl. 1st word) ---")
        lines.append(f"  Median:         {self.timing_median:+.0f}ms")
        lines.append(f"  Mean:           {self.timing_mean:+.0f}ms")
        lines.append(f"  Within 100ms:   {self.timing_within_100:.1f}%")
        lines.append(f"  Within 200ms:   {self.timing_within_200:.1f}%")
        lines.append(f"  Within 500ms:   {self.timing_within_500:.1f}%")

        lines.append("\n--- Lyrics ---")
        lines.append(f"  Matched pairs:  {self.matched_pairs}")
        lines.append(f"  Generated:      {self.gen_word_count} words")
        lines.append(f"  Reference:      {self.ref_word_count} words")
        lines.append(f"  Precision:      {self.lyrics_precision:.1f}%")
        lines.append(f"  Recall:         {self.lyrics_recall:.1f}%")

        lines.append("\n--- Structure ---")
        lines.append(f"  Notes:          {self.gen_note_count} gen"
                     f" / {self.ref_note_count} ref")
        lines.append(f"  Linebreaks:     {self.gen_linebreaks} gen"
                     f" / {self.ref_linebreaks} ref")

        if detailed and self.word_pairs:
            lines.append("\n--- Word-level detail (first 50) ---")
            lines.append(f"{'Gen Word':<15} {'Ref Word':<15}"
                         f" {'PitchD':>6} {'TimeD':>8} {'GenMIDI':>7}"
                         f" {'RefMIDI':>7}")
            lines.append("-" * 70)
            for wp in self.word_pairs[:50]:
                lines.append(
                    f"{wp['gen_word']:<15} {wp['ref_word']:<15}"
                    f" {wp['pitch_diff']:>+6d} {wp['timing_diff']:>+8.0f}ms"
                    f" {wp['gen_midi']:>7d} {wp['ref_midi']:>7d}"
                )

        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_header_value(line: str) -> str:
    """Extract the value after the first colon in a header line."""
    idx = line.index(":")
    return line[idx + 1:].strip()


def _parse_gap(raw: str) -> float:
    """Parse GAP value (may use comma as decimal separator). Returns ms."""
    return float(raw.replace(",", "."))


def _parse_bpm(raw: str) -> float:
    """Parse BPM value (may use comma as decimal separator)."""
    return float(raw.replace(",", "."))


def _beat_to_ms(beat: int, bpm_header: float, gap_ms: float) -> float:
    """Convert a beat position to milliseconds.

    UltraStar BPM in the header is 1/4 of the real BPM.
    Formula: ms = gap_ms + (beat * 60000) / (bpm_header * 4)
    """
    real_bpm = bpm_header * 4
    if real_bpm == 0:
        return gap_ms
    return gap_ms + (beat * 60_000) / real_bpm


def _pitch_to_midi(pitch: int, version: str | None) -> int:
    """Convert a file pitch value to a standard MIDI note number.

    - UltraSinger v1.2.0 format: pitch is relative (0 = C4 = MIDI 48)
      so midi = pitch + 48
    - Legacy / standard UltraStar format: pitch IS the MIDI note number
    """
    if version is not None and _version_ge(version, "1.2.0"):
        return pitch + 48
    return pitch


def _version_ge(version: str, threshold: str) -> bool:
    """Check if *version* >= *threshold* using simple numeric comparison.

    Handles shorthand versions ("1.2" treated as "1.2.0") and suffixed
    versions ("1.2.0-beta" extracts 1.2.0).  Pads the shorter tuple with
    trailing zeros so the comparison is always well-defined.
    """
    def _to_tuple(v: str) -> tuple[int, ...]:
        parts = [int(m) for m in re.findall(r"\d+", v.split("-")[0])]
        if not parts:
            raise ValueError(f"no numeric components in {v!r}")
        return tuple(parts)

    try:
        v_parts = _to_tuple(version)
        t_parts = _to_tuple(threshold)
        # Pad shorter tuple with zeros
        max_len = max(len(v_parts), len(t_parts))
        v_padded = v_parts + (0,) * (max_len - len(v_parts))
        t_padded = t_parts + (0,) * (max_len - len(t_parts))
        return v_padded >= t_padded
    except (ValueError, AttributeError):
        return False


_NOTE_TYPES = {":", "*", "F", "R", "G"}


def parse_ultrastar(path: str | Path) -> ParsedFile:
    """Parse an UltraStar TXT file and return normalised data."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    headers: dict[str, str] = {}
    version: str | None = None
    raw_bpm: str = "0"
    raw_gap: str = "0"
    raw_notes: list[tuple[str, int, int, int, str]] = []
    linebreak_count = 0

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Header
        if line.startswith("#"):
            tag_match = re.match(r"#([A-Z_]+):(.*)", line, re.IGNORECASE)
            if tag_match:
                tag = tag_match.group(1).upper()
                value = tag_match.group(2).strip()
                headers[tag] = value
                if tag == "VERSION":
                    version = value
                elif tag == "BPM":
                    raw_bpm = value
                elif tag == "GAP":
                    raw_gap = value
            continue

        # End marker
        if line.upper() == "E":
            break

        # Linebreak
        if line.startswith("-"):
            linebreak_count += 1
            continue

        # Note line
        parts = line.split(None, 4)
        if len(parts) >= 4 and parts[0] in _NOTE_TYPES:
            try:
                note_type = parts[0]
                start_beat = int(parts[1])
                duration = int(parts[2])
                pitch = int(parts[3])
            except ValueError:
                continue  # skip malformed note lines gracefully
            word = parts[4] if len(parts) > 4 else ""
            raw_notes.append((note_type, start_beat, duration, pitch, word))

    bpm = _parse_bpm(raw_bpm)
    gap_ms = _parse_gap(raw_gap)

    notes: list[Note] = []
    for note_type, start_beat, duration, pitch, word in raw_notes:
        midi = _pitch_to_midi(pitch, version)
        start_ms = _beat_to_ms(start_beat, bpm, gap_ms)
        end_ms = _beat_to_ms(start_beat + duration, bpm, gap_ms)
        notes.append(Note(
            note_type=note_type,
            start_beat=start_beat,
            duration=duration,
            pitch=pitch,
            midi=midi,
            word=word,
            start_ms=start_ms,
            end_ms=end_ms,
        ))

    word_groups = _group_words(notes)

    return ParsedFile(
        path=str(path),
        version=version,
        bpm=bpm,
        gap=gap_ms,
        notes=notes,
        word_groups=word_groups,
        linebreak_count=linebreak_count,
        headers=headers,
    )


def _group_words(notes: list[Note]) -> list[WordGroup]:
    """Group continuation notes (~) with their parent word.

    Merges continuation text into the parent word so that split words
    like ``Hel`` + ``~lo`` produce the group word ``Hello``.
    """
    groups: list[WordGroup] = []
    for note in notes:
        clean_word = note.word.strip()
        if clean_word.startswith("~") and groups:
            # Continuation of previous word — merge suffix text
            suffix = clean_word.lstrip("~")
            groups[-1].notes.append(note)
            if suffix:
                groups[-1].word += suffix
            groups[-1].end_ms = note.end_ms
        else:
            wg = WordGroup(
                word=clean_word,
                notes=[note],
                midi=note.midi,
                start_ms=note.start_ms,
                end_ms=note.end_ms,
            )
            groups.append(wg)
    return groups


# ---------------------------------------------------------------------------
# Word alignment
# ---------------------------------------------------------------------------

def _normalise_word(word: str) -> str:
    """Normalise a word for fuzzy matching.

    Applies NFKD Unicode decomposition to convert accented characters
    (e.g. 'é' → 'e') before stripping non-alphanumeric characters.
    """
    # Decompose accented characters and remove combining diacritical marks
    decomposed = unicodedata.normalize("NFKD", word)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", stripped.lower())


def align_words(
    gen_groups: list[WordGroup],
    ref_groups: list[WordGroup],
) -> list[tuple[int, int]]:
    """Align generated word groups to reference word groups.

    Returns list of (gen_index, ref_index) pairs.
    Uses SequenceMatcher on normalised word sequences for robust matching.
    """
    gen_words = [_normalise_word(g.word) for g in gen_groups]
    ref_words = [_normalise_word(g.word) for g in ref_groups]

    # Filter out empty words
    gen_valid = [(i, w) for i, w in enumerate(gen_words) if w]
    ref_valid = [(i, w) for i, w in enumerate(ref_words) if w]

    gen_seq = [w for _, w in gen_valid]
    ref_seq = [w for _, w in ref_valid]

    matcher = SequenceMatcher(None, gen_seq, ref_seq, autojunk=False)
    pairs: list[tuple[int, int]] = []

    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            gi = gen_valid[block.a + k][0]
            ri = ref_valid[block.b + k][0]
            pairs.append((gi, ri))

    return pairs


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _pct(count: int, total: int) -> float:
    """Safe percentage calculation."""
    return (count / total * 100) if total > 0 else 0.0


def compare(gen_path: str | Path, ref_path: str | Path) -> ComparisonResult:
    """Compare a generated UltraStar file against a reference."""
    gen = parse_ultrastar(gen_path)
    ref = parse_ultrastar(ref_path)

    pairs = align_words(gen.word_groups, ref.word_groups)

    result = ComparisonResult()

    # Meta
    result.gen_bpm = gen.bpm
    result.ref_bpm = ref.bpm
    result.gen_gap = gen.gap
    result.ref_gap = ref.gap
    result.gen_version = gen.version
    result.ref_version = ref.version

    # Structure
    result.gen_note_count = len(gen.notes)
    result.ref_note_count = len(ref.notes)
    result.gen_linebreaks = gen.linebreak_count
    result.ref_linebreaks = ref.linebreak_count

    # Lyrics — use same non-empty word filter as align_words()
    gen_valid_words = [g for g in gen.word_groups if _normalise_word(g.word)]
    ref_valid_words = [g for g in ref.word_groups if _normalise_word(g.word)]
    result.gen_word_count = len(gen_valid_words)
    result.ref_word_count = len(ref_valid_words)
    result.matched_pairs = len(pairs)
    result.lyrics_precision = _pct(len(pairs), len(gen_valid_words))
    result.lyrics_recall = _pct(len(pairs), len(ref_valid_words))

    if not pairs:
        return result

    # Pitch & Timing
    pitch_diffs: list[int] = []
    timing_diffs: list[float] = []
    word_pair_details: list[dict] = []

    for idx, (gi, ri) in enumerate(pairs):
        gw = gen.word_groups[gi]
        rw = ref.word_groups[ri]

        pd = gw.midi - rw.midi
        pitch_diffs.append(pd)

        td = gw.start_ms - rw.start_ms
        # Skip first matched word for timing (GAP offset)
        if idx > 0:
            timing_diffs.append(td)

        word_pair_details.append({
            "gen_word": gw.word,
            "ref_word": rw.word,
            "gen_midi": gw.midi,
            "ref_midi": rw.midi,
            "pitch_diff": pd,
            "timing_diff": td,
        })

    result.pitch_diffs = pitch_diffs
    result.word_pairs = word_pair_details

    if pitch_diffs:
        result.pitch_median = float(median(pitch_diffs))
        result.pitch_mean = mean(pitch_diffs)
        result.pitch_exact_pct = _pct(
            sum(1 for d in pitch_diffs if d == 0), len(pitch_diffs))
        result.pitch_within_2_pct = _pct(
            sum(1 for d in pitch_diffs if abs(d) <= 2), len(pitch_diffs))

    # Intervals (consecutive matched pairs)
    interval_diffs: list[int] = []
    for k in range(len(pairs) - 1):
        gi1, ri1 = pairs[k]
        gi2, ri2 = pairs[k + 1]
        gen_interval = gen.word_groups[gi2].midi - gen.word_groups[gi1].midi
        ref_interval = ref.word_groups[ri2].midi - ref.word_groups[ri1].midi
        interval_diffs.append(gen_interval - ref_interval)

    result.interval_diffs = interval_diffs
    if interval_diffs:
        result.interval_exact_pct = _pct(
            sum(1 for d in interval_diffs if d == 0), len(interval_diffs))
        result.interval_within_2_pct = _pct(
            sum(1 for d in interval_diffs if abs(d) <= 2),
            len(interval_diffs))

    # Timing
    result.timing_diffs = timing_diffs
    if timing_diffs:
        result.timing_median = float(median(timing_diffs))
        result.timing_mean = mean(timing_diffs)
        result.timing_within_100 = _pct(
            sum(1 for d in timing_diffs if abs(d) <= 100), len(timing_diffs))
        result.timing_within_200 = _pct(
            sum(1 for d in timing_diffs if abs(d) <= 200), len(timing_diffs))
        result.timing_within_500 = _pct(
            sum(1 for d in timing_diffs if abs(d) <= 500), len(timing_diffs))

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for comparing two UltraStar TXT files."""
    parser = argparse.ArgumentParser(
        description="Compare two UltraStar TXT files (generated vs reference)")
    parser.add_argument("generated", help="Path to generated UltraStar TXT")
    parser.add_argument("reference", help="Path to reference UltraStar TXT")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--detailed", action="store_true",
                        help="Show per-word comparison table")
    args = parser.parse_args()

    result = compare(args.generated, args.reference)

    if args.json:
        print(result.to_json())
    else:
        print(result.to_summary(detailed=args.detailed))


if __name__ == "__main__":
    main()
