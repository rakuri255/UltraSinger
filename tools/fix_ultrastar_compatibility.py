"""Fix UltraStar song compatibility: audio format conversion and encoding normalization.

Scans a directory tree for UltraStar TXT files and fixes two categories of
compatibility issues:

1. Audio Format Conversion:
   UltraStar Play only supports mp3, ogg, and wav. Songs with incompatible
   audio formats (aac, flac, opus, m4a, wma, etc.) are converted to OGG Vorbis
   using FFmpeg, and the TXT file references are updated accordingly.
   Original audio files are preserved (never deleted).

2. Encoding Normalization (--normalize-encoding):
   TXT files in legacy encodings (CP1252, Latin-1, etc.) are converted to UTF-8
   for universal compatibility across all UltraStar derivatives (USDX, UltraStar
   Play, Performous, Vocaluxe). The deprecated #ENCODING: tag is removed.

Usage:
    uv run python tools/fix_ultrastar_compatibility.py "D:\\UltraStar\\Songs"
    uv run python tools/fix_ultrastar_compatibility.py "D:\\UltraStar\\Songs" --normalize-encoding
    uv run python tools/fix_ultrastar_compatibility.py "D:\\UltraStar\\Songs" --normalize-encoding --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Ensure stdout can handle Unicode on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMPATIBLE_FORMATS = {"mp3", "ogg", "wav"}

# UltraStar TXT tags that reference audio files
AUDIO_TAGS = ("#MP3:", "#AUDIO:", "#VOCALS:", "#INSTRUMENTAL:")

# Tags used to identify a file as UltraStar TXT (at least one must be present)
IDENTITY_TAGS = ("#TITLE:", "#ARTIST:", "#MP3:", "#AUDIO:")

# Encoding fallback chain for reading UltraStar TXT files.
# Many older song collections use Windows-1252 or Latin-1 encoding.
_TXT_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")

TXT_ENCODING_WRITE = "utf-8"


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _read_txt_lines(path: Path) -> tuple[list[str], str]:
    """Read a TXT file trying multiple encodings.

    Returns (lines, encoding_used).  The fallback chain is:
      1. UTF-8 (with BOM)  — modern files
      2. Windows-1252      — Western European (ö ü ä é è ñ ł etc.)
      3. Latin-1           — never fails (maps all 256 byte values)

    We detect "bad" UTF-8 decoding by checking for the replacement
    character U+FFFD which appears when errors="replace" substitutes
    undecodable bytes.
    """
    raw = path.read_bytes()

    for enc in _TXT_ENCODINGS:
        try:
            text = raw.decode(enc)
            # If the codec had to replace bytes we get U+FFFD — try next
            if "\ufffd" in text and enc != _TXT_ENCODINGS[-1]:
                continue
            return text.splitlines(keepends=True), enc
        except (UnicodeDecodeError, LookupError):
            continue

    # Should never reach here (latin-1 never raises), but just in case
    return raw.decode("latin-1").splitlines(keepends=True), "latin-1"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AudioFileInfo:
    """Info about a single audio file referenced in a TXT."""
    tag: str                # e.g. "#MP3:"
    filename: str           # e.g. "song.aac"
    extension: str          # e.g. "aac"
    full_path: Path         # absolute path to the audio file
    compatible: bool        # True if extension in COMPATIBLE_FORMATS


@dataclass
class SongInfo:
    """Parsed metadata from one UltraStar TXT file."""
    txt_path: Path
    song_dir: Path
    artist: str = ""
    title: str = ""
    audio_files: list[AudioFileInfo] = field(default_factory=list)
    file_encoding: str = "utf-8"          # detected encoding of the TXT file
    has_encoding_tag: bool = False         # True if #ENCODING: tag is present

    @property
    def display_name(self) -> str:
        if self.artist and self.title:
            return f"{self.artist} - {self.title}"
        if self.title:
            return self.title
        return self.txt_path.stem


@dataclass
class FileConversion:
    """Record of a single file conversion."""
    tag: str                # e.g. "#MP3:"
    old_filename: str       # e.g. "song.aac"
    new_filename: str       # e.g. "song.ogg"
    old_ext: str            # e.g. "aac"


@dataclass
class ConversionResult:
    """Result of processing one song."""
    song_info: SongInfo
    conversions: list[FileConversion] = field(default_factory=list)
    status: str = "skipped"         # "converted", "skipped", "error", "already_compatible"
    error_msg: str | None = None
    warnings: list[str] = field(default_factory=list)
    encoding_normalized: bool = False      # True if encoding was converted to UTF-8
    encoding_from: str | None = None       # original encoding if normalized
    encoding_tag_removed: bool = False     # True if #ENCODING: tag was removed


# ---------------------------------------------------------------------------
# TXT discovery and parsing
# ---------------------------------------------------------------------------

def is_ultrastar_txt(path: Path) -> bool:
    """Check if a file is an UltraStar TXT by looking for identity tags."""
    try:
        lines, _ = _read_txt_lines(path)
        for i, line in enumerate(lines):
            if i >= 50:
                break
            line_upper = line.strip().upper()
            for tag in IDENTITY_TAGS:
                if line_upper.startswith(tag):
                    return True
    except OSError:
        pass
    return False


def find_ultrastar_txts(root: Path) -> list[Path]:
    """Recursively find all UltraStar TXT files under root."""
    txts = []
    for path in sorted(root.rglob("*.txt")):
        if path.is_file() and is_ultrastar_txt(path):
            txts.append(path)
    return txts


def parse_song_info(txt_path: Path) -> SongInfo | None:
    """Parse audio tags and metadata from an UltraStar TXT file."""
    song_dir = txt_path.parent
    info = SongInfo(txt_path=txt_path, song_dir=song_dir)

    try:
        lines, detected_enc = _read_txt_lines(txt_path)
    except OSError as e:
        print(f"  WARNING: Cannot read {txt_path}: {e}", file=sys.stderr)
        return None

    info.file_encoding = detected_enc

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue

        # Extract tag and value — split only on first ':'
        colon_idx = stripped.find(":")
        if colon_idx < 0:
            continue

        tag_part = stripped[: colon_idx + 1].upper()  # e.g. "#MP3:"
        value = stripped[colon_idx + 1:].strip()

        if tag_part == "#ENCODING:":
            info.has_encoding_tag = True

        if not value:
            continue

        if tag_part == "#ARTIST:":
            info.artist = value
        elif tag_part == "#TITLE:":
            info.title = value
        elif tag_part in ("#MP3:", "#AUDIO:", "#VOCALS:", "#INSTRUMENTAL:"):
            ext = _get_extension(value)
            full_path = song_dir / value
            info.audio_files.append(AudioFileInfo(
                tag=tag_part,
                filename=value,
                extension=ext,
                full_path=full_path,
                compatible=ext in COMPATIBLE_FORMATS,
            ))

    return info


def _get_extension(filename: str) -> str:
    """Extract lowercase extension without dot from a filename."""
    _, ext = os.path.splitext(filename)
    return ext.lstrip(".").lower()


# ---------------------------------------------------------------------------
# FFmpeg conversion
# ---------------------------------------------------------------------------

def check_ffmpeg() -> str:
    """Return the ffmpeg executable path or raise if not found."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ERROR: ffmpeg not found in PATH. Please install FFmpeg.", file=sys.stderr)
        sys.exit(1)
    return ffmpeg


def convert_to_ogg(ffmpeg_path: str, input_path: Path, output_path: Path) -> None:
    """Convert an audio file to OGG Vorbis using FFmpeg."""
    cmd = [
        ffmpeg_path,
        "-i", str(input_path),
        "-y",
        "-loglevel", "error",
        "-codec:a", "libvorbis",
        "-q:a", "6",               # High quality VBR (~192 kbps)
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr.strip()}")


# ---------------------------------------------------------------------------
# TXT file update
# ---------------------------------------------------------------------------

def update_txt_tags(txt_path: Path, replacements: dict[str, str]) -> None:
    """Replace audio filenames in TXT tags.

    Reads the file with encoding auto-detection, writes back in UTF-8.

    Args:
        txt_path: Path to the UltraStar TXT file.
        replacements: Mapping of old_filename → new_filename.
    """
    lines, _ = _read_txt_lines(txt_path)

    updated = False
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            colon_idx = stripped.find(":")
            if colon_idx >= 0:
                tag_part = stripped[: colon_idx + 1].upper()
                value = stripped[colon_idx + 1:].strip()
                if tag_part in ("#MP3:", "#AUDIO:", "#VOCALS:", "#INSTRUMENTAL:"):
                    if value in replacements:
                        # Preserve original tag case from the file
                        original_tag = stripped[: colon_idx + 1]
                        line = f"{original_tag}{replacements[value]}\n"
                        updated = True
        new_lines.append(line)

    if updated:
        with open(txt_path, "w", encoding=TXT_ENCODING_WRITE, newline="") as f:
            f.writelines(new_lines)


# ---------------------------------------------------------------------------
# Encoding normalization
# ---------------------------------------------------------------------------

def _is_utf8_encoding(enc: str) -> bool:
    """Check if an encoding name represents UTF-8 (with or without BOM)."""
    return enc.lower().replace("-", "").replace("_", "") in ("utf8", "utf8sig")


def normalize_txt_encoding(txt_path: Path) -> tuple[bool, bool]:
    """Re-encode a TXT file to UTF-8 and remove the deprecated #ENCODING: tag.

    Returns (encoding_changed, encoding_tag_removed).
    """
    lines, detected_enc = _read_txt_lines(txt_path)

    encoding_changed = not _is_utf8_encoding(detected_enc)
    encoding_tag_removed = False

    # Filter out #ENCODING: lines (deprecated since format v1.0.0)
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            colon_idx = stripped.find(":")
            if colon_idx >= 0:
                tag_part = stripped[: colon_idx + 1].upper()
                if tag_part == "#ENCODING:":
                    encoding_tag_removed = True
                    continue  # skip this line
        new_lines.append(line)

    # Only write if something actually changed
    if encoding_changed or encoding_tag_removed:
        with open(txt_path, "w", encoding=TXT_ENCODING_WRITE, newline="") as f:
            f.writelines(new_lines)

    return encoding_changed, encoding_tag_removed


# ---------------------------------------------------------------------------
# Song processing
# ---------------------------------------------------------------------------

def process_song(
    song_info: SongInfo,
    ffmpeg_path: str,
    dry_run: bool,
    normalize_encoding: bool = False,
) -> ConversionResult:
    """Process a single song: check audio compatibility and convert if needed.

    If normalize_encoding is True, also re-encode non-UTF-8 TXT files to UTF-8
    and remove the deprecated #ENCODING: tag.
    """
    result = ConversionResult(song_info=song_info)

    # --- Encoding normalization ---
    needs_encoding_fix = normalize_encoding and (
        not _is_utf8_encoding(song_info.file_encoding)
        or song_info.has_encoding_tag
    )

    if needs_encoding_fix:
        if not _is_utf8_encoding(song_info.file_encoding):
            result.encoding_normalized = True
            result.encoding_from = song_info.file_encoding
        if song_info.has_encoding_tag:
            result.encoding_tag_removed = True

        if not dry_run:
            try:
                normalize_txt_encoding(song_info.txt_path)
            except Exception as e:
                result.warnings.append(f"Encoding normalization failed: {e}")
                result.encoding_normalized = False
                result.encoding_tag_removed = False

    # --- Audio format conversion ---
    if not song_info.audio_files:
        if needs_encoding_fix:
            # No audio tags but encoding was fixed — that's still a useful result
            result.status = "already_compatible"
        else:
            result.status = "error"
            result.error_msg = "No audio tags found in TXT"
        return result

    # Check which files need conversion
    needs_conversion: list[AudioFileInfo] = []
    for af in song_info.audio_files:
        if af.compatible:
            continue

        # Check if audio file exists
        if not af.full_path.is_file():
            result.warnings.append(f"{af.tag} {af.filename} — file not found, skipping")
            continue

        needs_conversion.append(af)

    if not needs_conversion:
        result.status = "already_compatible"
        return result

    # Build replacements: group by actual file (multiple tags may reference the same file)
    file_replacements: dict[str, str] = {}  # old_filename → new_filename
    conversions_to_do: list[tuple[AudioFileInfo, Path]] = []  # (info, ogg_output_path)

    for af in needs_conversion:
        if af.filename in file_replacements:
            # Already planned for conversion (another tag references the same file)
            new_filename = file_replacements[af.filename]
            result.conversions.append(FileConversion(
                tag=af.tag,
                old_filename=af.filename,
                new_filename=new_filename,
                old_ext=af.extension,
            ))
            continue

        stem = af.full_path.stem
        ogg_filename = f"{stem}.ogg"
        ogg_path = af.full_path.parent / ogg_filename

        file_replacements[af.filename] = ogg_filename
        conversions_to_do.append((af, ogg_path))

        result.conversions.append(FileConversion(
            tag=af.tag,
            old_filename=af.filename,
            new_filename=ogg_filename,
            old_ext=af.extension,
        ))

    if dry_run:
        result.status = "converted"  # would be converted
        return result

    # Perform actual conversions
    try:
        for af, ogg_path in conversions_to_do:
            if ogg_path.is_file():
                # OGG already exists (previous run?) — skip conversion but still update TXT
                result.warnings.append(
                    f"{ogg_path.name} already exists, skipping conversion"
                )
            else:
                convert_to_ogg(ffmpeg_path, af.full_path, ogg_path)

        # Update TXT file with new filenames
        update_txt_tags(song_info.txt_path, file_replacements)
        result.status = "converted"

    except Exception as e:
        result.status = "error"
        result.error_msg = str(e)

    return result


# ---------------------------------------------------------------------------
# Statistics output
# ---------------------------------------------------------------------------

def print_summary(
    results: list[ConversionResult],
    dry_run: bool,
    normalize_encoding: bool = False,
) -> None:
    """Print conversion summary statistics."""
    total = len(results)
    compatible = sum(1 for r in results if r.status == "already_compatible")
    converted = sum(1 for r in results if r.status == "converted")
    errors = sum(1 for r in results if r.status == "error")
    skipped = sum(1 for r in results if r.status == "skipped")

    enc_normalized = sum(1 for r in results if r.encoding_normalized)
    enc_tag_removed = sum(1 for r in results if r.encoding_tag_removed)

    print()
    if dry_run:
        print("=" * 55)
        print("  DRY RUN — No changes were made")
        print("=" * 55)
    else:
        print("=" * 55)
        print("  Conversion Summary")
        print("=" * 55)

    print(f"  Songs scanned:          {total:>5}")
    print(f"  Already compatible:     {compatible:>5}")
    if dry_run:
        print(f"  Audio would convert:    {converted:>5}")
    else:
        print(f"  Audio converted:        {converted:>5}")
    if errors:
        print(f"  Errors:                 {errors:>5}")
    if skipped:
        print(f"  Skipped (no audio):     {skipped:>5}")

    if normalize_encoding:
        print("  " + "-" * 40)
        if dry_run:
            print(f"  Encoding would fix:     {enc_normalized:>5}")
            if enc_tag_removed:
                print(f"  #ENCODING: would drop:  {enc_tag_removed:>5}")
        else:
            print(f"  Encoding normalized:    {enc_normalized:>5}")
            if enc_tag_removed:
                print(f"  #ENCODING: removed:     {enc_tag_removed:>5}")

        # Show encoding distribution
        enc_counts: dict[str, int] = {}
        for r in results:
            enc = r.song_info.file_encoding
            enc_counts[enc] = enc_counts.get(enc, 0) + 1
        if enc_counts:
            print("  " + "-" * 40)
            print("  Encoding distribution:")
            for enc_name in sorted(enc_counts, key=lambda e: -enc_counts[e]):
                count = enc_counts[enc_name]
                marker = "" if _is_utf8_encoding(enc_name) else " ←"
                print(f"    {enc_name:<15} {count:>5}{marker}")

    print("=" * 55)

    # Print warnings
    for r in results:
        for w in r.warnings:
            print(f"  WARNING: [{r.song_info.display_name}] {w}")

    # Print errors
    for r in results:
        if r.status == "error":
            print(f"  ERROR:   [{r.song_info.display_name}] {r.error_msg}")


def print_song_list(
    results: list[ConversionResult],
    dry_run: bool,
    normalize_encoding: bool = False,
) -> None:
    """Print the list of converted (or would-be-converted) songs."""
    # Audio conversions
    converted = [r for r in results if r.status == "converted"]
    if converted:
        print()
        if dry_run:
            print(f"=== Songs that would be converted ({len(converted)}) ===")
        else:
            print(f"=== Converted songs ({len(converted)}) ===")

        for i, r in enumerate(converted, 1):
            print(f"  [{i:>3}] {r.song_info.display_name}")
            for c in r.conversions:
                print(f"        {c.tag[1:]} {c.old_filename} \u2192 {c.new_filename}")

        print()

    # Encoding normalizations (only when --normalize-encoding is active)
    if normalize_encoding:
        enc_fixed = [r for r in results if r.encoding_normalized]
        if enc_fixed:
            print()
            if dry_run:
                print(f"=== Songs with encoding to normalize ({len(enc_fixed)}) ===")
            else:
                print(f"=== Songs with encoding normalized ({len(enc_fixed)}) ===")

            for i, r in enumerate(enc_fixed, 1):
                extra = ""
                if r.encoding_tag_removed:
                    extra = " + #ENCODING: removed"
                print(
                    f"  [{i:>3}] {r.song_info.display_name}"
                    f"  ({r.encoding_from} \u2192 utf-8{extra})"
                )

            print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fix_ultrastar_compatibility",
        description=(
            "Fix UltraStar song compatibility: convert incompatible audio formats "
            "to OGG Vorbis and optionally normalize TXT file encoding to UTF-8."
        ),
        epilog=(
            "UltraStar Play supports mp3, ogg, and wav. "
            "Other formats (aac, flac, opus, m4a, wma, ...) are converted to OGG Vorbis. "
            "Original audio files are never deleted. "
            "Use --normalize-encoding to also convert legacy text encodings "
            "(CP1252, Latin-1, etc.) to UTF-8 for universal compatibility."
        ),
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Root directory containing UltraStar song folders",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be changed, without making any modifications",
    )
    parser.add_argument(
        "--normalize-encoding",
        action="store_true",
        help=(
            "Convert non-UTF-8 TXT files to UTF-8 and remove deprecated "
            "#ENCODING: tags. UTF-8 is supported by all UltraStar derivatives."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.directory.resolve()

    if not root.is_dir():
        print(f"ERROR: Directory not found: {root}", file=sys.stderr)
        return 1

    # Check FFmpeg availability early
    ffmpeg_path = check_ffmpeg()

    mode_parts = []
    if args.dry_run:
        mode_parts.append("DRY RUN")
    if args.normalize_encoding:
        mode_parts.append("encoding normalization enabled")
    prefix = f"[{', '.join(mode_parts)}] " if mode_parts else ""
    print(f"{prefix}Scanning {root} ...")

    # Discover UltraStar TXT files
    txt_files = find_ultrastar_txts(root)
    if not txt_files:
        print("No UltraStar TXT files found.")
        return 0

    print(f"Found {len(txt_files)} UltraStar TXT file(s).")

    # Parse and process each song
    results: list[ConversionResult] = []
    for txt_path in txt_files:
        song_info = parse_song_info(txt_path)
        if song_info is None:
            continue
        result = process_song(
            song_info, ffmpeg_path, args.dry_run,
            normalize_encoding=args.normalize_encoding,
        )
        results.append(result)

        # Progress indicator for conversions
        if not args.dry_run:
            if result.status == "converted":
                print(f"  Audio converted: {result.song_info.display_name}")
            if result.encoding_normalized:
                print(
                    f"  Encoding fixed:  {result.song_info.display_name}"
                    f"  ({result.encoding_from} → utf-8)"
                )

    # Output statistics
    print_summary(results, args.dry_run, args.normalize_encoding)
    print_song_list(results, args.dry_run, args.normalize_encoding)

    return 0


if __name__ == "__main__":
    sys.exit(main())
