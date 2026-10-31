"""Lyric synchronization benchmark against human-curated UltraStar references.

Compares a generated UltraStar .txt (UltraSinger output) with a production/human
reference .txt (e.g. ``/ProducionTestMusic/<song>/<song>.txt``) in absolute seconds,
independent of the BPM/GAP each file uses.

Usage (inside the container, from /app/UltraSinger/src):
    python3 -m modules.Benchmark.sync_benchmark <reference.txt> <generated.txt> [--json out.json]

Metrics (see implementation_plan.md, Phase 1):
    - onset MAE (median |dt|), % onsets within 50 / 100 ms
    - line-start median error
    - temporal IoU, lexical coverage, syllabification agreement
    - crushed-note rate, pitch accuracy (mod 12), lexical freestyle rate
    - #GAP error, linear drift (slope of t_gen = a + b * t_ref)
    - composite score 0-10:
        S = 10 * (0.35 P50 + 0.20 IoU + 0.15 C_lex + 0.15 C_syl + 0.10 P_pitch + 0.05 (1 - R_crush))
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple


ENCODINGS = ("utf-8-sig", "utf-16", "utf-8", "cp1252", "latin-1")


@dataclass
class SyllableNote:
    """A sung syllable expressed in absolute seconds."""
    text: str
    norm: str
    t_start: float
    t_end: float
    pitch: int
    note_type: str
    line_idx: int
    is_line_start: bool = False


@dataclass
class UltraStarSong:
    """Minimal UltraStar representation for benchmarking."""
    bpm: float
    gap_ms: float
    notes: List[SyllableNote] = field(default_factory=list)


def _read_text(path: str) -> str:
    """Read text trying several encodings (production files may be ISO-8859/UTF-16)."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    for enc in ENCODINGS:
        if enc == "utf-16":
            continue
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def normalize_syllable(text: str) -> str:
    """Lowercase, strip accents and non-alphanumerics."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    no_acc = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", no_acc)


def parse_ultrastar(path: str) -> UltraStarSong:
    """Parse an UltraStar file into absolute-time syllables. '~' notes extend the previous syllable."""
    content = _read_text(path)
    headers = {}
    raw_notes: List[Tuple[str, int, int, int, str, int]] = []
    line_idx = 0
    for raw in content.splitlines():
        line = raw.rstrip("\r\n")
        if not line:
            continue
        if line.startswith("#"):
            parts = line[1:].split(":", 1)
            if len(parts) == 2:
                headers[parts[0].strip().upper()] = parts[1].strip()
            continue
        if line[0] == "-":
            line_idx += 1
            continue
        if line[0] == "E":
            break
        if line[0] in (":", "*", "F", "R", "G"):
            parts = line.split(" ", 4)
            if len(parts) < 4:
                continue
            try:
                s_b, d_b, p = int(parts[1]), int(parts[2]), int(parts[3])
            except ValueError:
                continue
            word = parts[4] if len(parts) == 5 else ""
            raw_notes.append((parts[0], s_b, d_b, p, word, line_idx))

    bpm = float(headers.get("BPM", "120").replace(",", "."))
    gap = float(headers.get("GAP", "0").replace(",", "."))
    beat_sec = 60.0 / (4.0 * bpm)
    song = UltraStarSong(bpm=bpm, gap_ms=gap)

    last_line = -1
    for n_type, s_b, d_b, p, word, l_idx in raw_notes:
        t_s = gap / 1000.0 + s_b * beat_sec
        t_e = t_s + d_b * beat_sec
        norm = normalize_syllable(word)
        if not norm and song.notes:
            # Melisma '~' (or empty): extend previous syllable
            song.notes[-1].t_end = max(song.notes[-1].t_end, t_e)
            continue
        if not norm:
            continue
        song.notes.append(
            SyllableNote(
                text=word,
                norm=norm,
                t_start=t_s,
                t_end=t_e,
                pitch=p,
                note_type=n_type,
                line_idx=l_idx,
                is_line_start=(l_idx != last_line),
            )
        )
        last_line = l_idx
    return song


def _syl_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    # cheap similarity: ratio of common prefix/suffix chars
    from difflib import SequenceMatcher
    sim = SequenceMatcher(None, a, b).ratio()
    return 0.5 if sim >= 0.6 else 1.0


def align_syllables(ref: List[SyllableNote], gen: List[SyllableNote], gap_cost: float = 0.7) -> List[Tuple[int, int, float]]:
    """Needleman-Wunsch alignment of syllable sequences. Returns list of (i_ref, j_gen, cost) matches.

    A temporal band (|t_ref - t_gen| < 60 s) keeps the matrix sparse and avoids matching
    repeated choruses minutes apart.
    """
    n, m = len(ref), len(gen)
    inf = float("inf")
    # Use concatenated-letter alignment fallback: both sequences may syllabify differently,
    # so substitution between partially similar syllables is allowed at cost 0.5.
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        dp[i][0] = i * gap_cost
        bt[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = j * gap_cost
        bt[0][j] = 2
    for i in range(1, n + 1):
        r = ref[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            g = gen[j - 1]
            if abs(r.t_start - g.t_start) > 60.0:
                sub = inf
            else:
                sub = prev[j - 1] + _syl_cost(r.norm, g.norm)
            dele = prev[j] + gap_cost
            ins = row[j - 1] + gap_cost
            best, move = sub, 0
            if dele < best:
                best, move = dele, 1
            if ins < best:
                best, move = ins, 2
            row[j] = best
            bt[i][j] = move
    i, j = n, m
    pairs: List[Tuple[int, int, float]] = []
    while i > 0 or j > 0:
        mv = bt[i][j]
        if i > 0 and j > 0 and mv == 0:
            pairs.append((i - 1, j - 1, _syl_cost(ref[i - 1].norm, gen[j - 1].norm)))
            i, j = i - 1, j - 1
        elif i > 0 and (mv == 1 or j == 0):
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


@dataclass
class BenchmarkReport:
    """Synchronization benchmark result."""
    n_ref_syllables: int
    n_gen_syllables: int
    onset_median_abs_ms: float
    onset_mean_abs_ms: float
    pct_within_50ms: float
    pct_within_100ms: float
    line_start_median_abs_ms: float
    temporal_iou: float
    lexical_coverage: float
    syllabification_agreement: float
    crushed_note_rate: float
    pitch_accuracy_mod12: float
    lexical_freestyle_rate: float
    gap_error_ms: float
    drift_slope: float
    drift_intercept_ms: float
    structural_outliers: List[dict]
    composite_score: float
    ref_bpm: float = 0.0
    ref_gap_ms: float = 0.0
    gen_bpm: float = 0.0
    gen_gap_ms: float = 0.0
    duration_ratio_median: float = 1.0


def _median(vals: List[float]) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = len(s) // 2
    return s[k] if len(s) % 2 else 0.5 * (s[k - 1] + s[k])


def compare_songs(ref: UltraStarSong, gen: UltraStarSong) -> BenchmarkReport:
    """Compute all synchronization metrics between a reference and a generated song."""
    pairs = align_syllables(ref.notes, gen.notes)
    exact_or_close = [(i, j) for i, j, c in pairs if c <= 0.5]
    exact = [(i, j) for i, j, c in pairs if c == 0.0]

    deltas = [gen.notes[j].t_start - ref.notes[i].t_start for i, j in exact_or_close]
    abs_d = [abs(d) for d in deltas]
    n_ref = max(1, len(ref.notes))

    p50 = sum(1 for d in abs_d if d <= 0.050) / n_ref
    p100 = sum(1 for d in abs_d if d <= 0.100) / n_ref

    line_abs = [abs(gen.notes[j].t_start - ref.notes[i].t_start) for i, j in exact_or_close if ref.notes[i].is_line_start]

    inter = union = 0.0
    for i, j in exact_or_close:
        r, g = ref.notes[i], gen.notes[j]
        ov = max(0.0, min(r.t_end, g.t_end) - max(r.t_start, g.t_start))
        un = max(r.t_end, g.t_end) - min(r.t_start, g.t_start)
        inter += ov
        union += max(un, 1e-6)
    iou = inter / union if union > 0 else 0.0

    lex_cov = len(exact_or_close) / n_ref
    syl_agree = len(exact) / n_ref

    crushed = sum(1 for g in gen.notes if (g.t_end - g.t_start) < 0.060 and len(g.norm) > 1)
    crush_rate = crushed / max(1, len(gen.notes))

    pitch_hits = 0
    pitch_total = 0
    for i, j in exact_or_close:
        r, g = ref.notes[i], gen.notes[j]
        if r.note_type in ("F", "R") or g.note_type in ("F", "R"):
            continue
        d = abs(g.pitch - r.pitch) % 12
        d = min(d, 12 - d)
        pitch_total += 1
        if d <= 1:
            pitch_hits += 1
    pitch_acc = pitch_hits / max(1, pitch_total)

    lex_f = sum(1 for g in gen.notes if g.note_type == "F" and len(g.norm) >= 1)
    lex_f_rate = lex_f / max(1, len(gen.notes))

    # Linear drift regression t_gen = a + b * t_ref (on close matches)
    xs = [ref.notes[i].t_start for i, j in exact_or_close]
    ys = [gen.notes[j].t_start for i, j in exact_or_close]
    slope, intercept = 1.0, 0.0
    if len(xs) >= 2:
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        if sxx > 0:
            slope = sxy / sxx
            intercept = my - slope * mx

    outliers = []
    for i, j in exact_or_close:
        d = gen.notes[j].t_start - ref.notes[i].t_start
        if abs(d) > 2.0:
            outliers.append({
                "ref_line": ref.notes[i].line_idx,
                "syllable": ref.notes[i].text.strip(),
                "t_ref": round(ref.notes[i].t_start, 3),
                "t_gen": round(gen.notes[j].t_start, 3),
                "delta_s": round(d, 3),
            })

    score = 10.0 * (
        0.35 * p50 + 0.20 * iou + 0.15 * lex_cov + 0.15 * syl_agree + 0.10 * pitch_acc + 0.05 * (1.0 - crush_rate)
    )

    return BenchmarkReport(
        n_ref_syllables=len(ref.notes),
        n_gen_syllables=len(gen.notes),
        onset_median_abs_ms=round(_median(abs_d) * 1000.0, 1) if abs_d else float("nan"),
        onset_mean_abs_ms=round(sum(abs_d) / len(abs_d) * 1000.0, 1) if abs_d else float("nan"),
        pct_within_50ms=round(p50 * 100.0, 2),
        pct_within_100ms=round(p100 * 100.0, 2),
        line_start_median_abs_ms=round(_median(line_abs) * 1000.0, 1) if line_abs else float("nan"),
        temporal_iou=round(iou, 4),
        lexical_coverage=round(lex_cov * 100.0, 2),
        syllabification_agreement=round(syl_agree * 100.0, 2),
        crushed_note_rate=round(crush_rate * 100.0, 2),
        pitch_accuracy_mod12=round(pitch_acc * 100.0, 2),
        lexical_freestyle_rate=round(lex_f_rate * 100.0, 2),
        gap_error_ms=round(abs(gen.gap_ms - ref.gap_ms), 1),
        drift_slope=round(slope, 5),
        drift_intercept_ms=round(intercept * 1000.0, 1),
        structural_outliers=outliers[:200],
        composite_score=round(score, 3),
        ref_bpm=ref.bpm,
        ref_gap_ms=ref.gap_ms,
        gen_bpm=gen.bpm,
        gen_gap_ms=gen.gap_ms,
        duration_ratio_median=round(
            _median([
                (gen.notes[j].t_end - gen.notes[j].t_start) / max(1e-3, ref.notes[i].t_end - ref.notes[i].t_start)
                for i, j in exact_or_close
            ]) if exact_or_close else 1.0,
            3,
        ),
    )


def run_benchmark(reference_path: str, generated_path: str, json_out: Optional[str] = None) -> BenchmarkReport:
    """Parse both files, compare and optionally dump JSON."""
    ref = parse_ultrastar(reference_path)
    gen = parse_ultrastar(generated_path)
    report = compare_songs(ref, gen)
    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    return report


def format_report(report: BenchmarkReport, title: str = "") -> str:
    """Human-readable table."""
    rows = [
        ("BPM / GAP (ref)", f"{report.ref_bpm} / {report.ref_gap_ms} ms"),
        ("BPM / GAP (gen)", f"{report.gen_bpm} / {report.gen_gap_ms} ms"),
        ("Syllables (ref / gen)", f"{report.n_ref_syllables} / {report.n_gen_syllables}"),
        ("Duration ratio median", f"{report.duration_ratio_median}"),
        ("Onset median |dt|", f"{report.onset_median_abs_ms} ms"),
        ("Onset mean |dt|", f"{report.onset_mean_abs_ms} ms"),
        ("Onsets <= 50 ms", f"{report.pct_within_50ms} %"),
        ("Onsets <= 100 ms", f"{report.pct_within_100ms} %"),
        ("Line start median |dt|", f"{report.line_start_median_abs_ms} ms"),
        ("Temporal IoU", f"{report.temporal_iou}"),
        ("Lexical coverage", f"{report.lexical_coverage} %"),
        ("Syllabification agreement", f"{report.syllabification_agreement} %"),
        ("Crushed notes", f"{report.crushed_note_rate} %"),
        ("Pitch accuracy (mod 12)", f"{report.pitch_accuracy_mod12} %"),
        ("Freestyle notes", f"{report.lexical_freestyle_rate} %"),
        ("#GAP error", f"{report.gap_error_ms} ms"),
        ("Drift slope / intercept", f"{report.drift_slope} / {report.drift_intercept_ms} ms"),
        ("Structural outliers (>2s)", f"{len(report.structural_outliers)}"),
        ("COMPOSITE SCORE", f"{report.composite_score} / 10"),
    ]
    width = max(len(r[0]) for r in rows)
    out = [f"=== Sync Benchmark {title} ==="]
    out += [f"  {k.ljust(width)} : {v}" for k, v in rows]
    return "\n".join(out)


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    json_out = None
    if "--json" in argv:
        k = argv.index("--json")
        json_out = argv[k + 1] if k + 1 < len(argv) else None
        argv = argv[:k] + argv[k + 2:]
    rep = run_benchmark(argv[0], argv[1], json_out)
    print(format_report(rep, title=argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
