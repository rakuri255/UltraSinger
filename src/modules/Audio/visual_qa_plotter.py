"""
Visual QA Plotter Module (Headless)
Genera gráficos de validación de sincronización y afinación entre UltraStar .txt
y el audio vocal F0 sin dependencias de entorno gráfico (Agg backend).
"""

import os
import sys
import gc
import math
import numpy as np

# Configurar Matplotlib en modo estrictamente desatendido (Headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import librosa

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, green_highlighted, red_highlighted
from modules.Pitcher.pitched_data import PitchedData


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_note_name(midi_val: int) -> str:
    """Convierte un valor MIDI a nombre musical legible (ej. 60 -> C4)."""
    octave = (midi_val // 12) - 1
    note = NOTE_NAMES[midi_val % 12]
    return f"{note}{octave}"


class UltraStarNoteData:
    def __init__(self, note_type: str, start_beat: float, duration_beats: float, pitch: int, word: str, player: int = 1):
        self.note_type = note_type
        self.start_beat = start_beat
        self.duration_beats = duration_beats
        self.pitch = pitch
        self.word = word
        self.player = player
        self.midi_note = pitch + 48
        self.start_sec = 0.0
        self.duration_sec = 0.0
        self.end_sec = 0.0


def parse_ultrastar_for_qa(txt_path: str) -> tuple[dict[str, str], list[UltraStarNoteData]]:
    """Lee el archivo .txt de UltraStar y calcula los tiempos absolutos en segundos."""
    headers = {}
    notes = []
    current_player = 1

    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue

            if line_str.startswith("#"):
                parts = line_str[1:].split(":", 1)
                if len(parts) == 2:
                    headers[parts[0].upper()] = parts[1].strip()
            elif line_str.upper().startswith(("P 1", "P1")):
                current_player = 1
            elif line_str.upper().startswith(("P 2", "P2")):
                current_player = 2
            elif line_str.startswith((":", "*", "F", "R", "G")):
                parts = line_str.split(None, 4)
                if len(parts) >= 4:
                    n_type = parts[0]
                    s_beat = float(parts[1])
                    d_beat = float(parts[2])
                    pitch = int(parts[3])
                    word = parts[4] if len(parts) > 4 else ""
                    notes.append(UltraStarNoteData(n_type, s_beat, d_beat, pitch, word, player=current_player))

    bpm = float(headers.get("BPM", "120").replace(",", "."))
    gap_ms = float(headers.get("GAP", "0").replace(",", "."))

    beat_sec = 15.0 / bpm if bpm > 0 else 0.125
    gap_sec = gap_ms / 1000.0

    for n in notes:
        n.start_sec = gap_sec + (n.start_beat * beat_sec)
        n.duration_sec = n.duration_beats * beat_sec
        n.end_sec = n.start_sec + n.duration_sec

    return headers, notes


def prepare_f0_curve(
    pitched_data: PitchedData | None,
    vocal_audio_path: str | None,
    pitcher_name: str = "rmvpe",
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Prepara los arrays de tiempo y semitonos MIDI de F0 con manejo de silencios."""
    if pitched_data is None and vocal_audio_path and os.path.isfile(vocal_audio_path):
        from modules.Pitcher.pitcher import get_pitch_with_file
        pitched_data = get_pitch_with_file(vocal_audio_path, pitcher_name=pitcher_name, device=device)

    if not pitched_data or len(pitched_data.times) == 0:
        return np.array([]), np.array([])

    times = np.array(pitched_data.times, dtype=np.float64)
    freqs = np.array(pitched_data.frequencies, dtype=np.float64)
    confs = np.array(pitched_data.confidence, dtype=np.float64)

    # Convertir frecuencias válidas a notas MIDI continuas
    midi_f0 = np.full_like(freqs, np.nan)
    valid_mask = (confs >= 0.35) & (freqs >= 65.0) & (freqs <= 1200.0)

    if np.any(valid_mask):
        midi_f0[valid_mask] = 12.0 * np.log2(freqs[valid_mask] / 440.0) + 69.0

    # Insertar cortes NaN en saltos temporales mayores a 35ms para evitar líneas cruzadas en silencios
    step_diffs = np.diff(times, prepend=times[0])
    midi_f0[step_diffs > 0.04] = np.nan

    return times, midi_f0


def generate_visual_qa_plot(
    txt_path: str,
    vocal_audio_path: str | None = None,
    output_folder: str | None = None,
    pitched_data: PitchedData | None = None,
    pitcher_name: str = "rmvpe",
    device: str = "cuda",
    seconds_per_row: float = 16.0,
    dpi: int = 180,
) -> str | None:
    """
    Genera el gráfico de Control de Calidad Visual paginado en subplots verticales.
    Guarda obligatoriamente en: /output/<Nombre_Cancion>/png/<Nombre_Cancion>_[QA_Visual].png
    """
    if not os.path.isfile(txt_path):
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Visual QA Error:')} Archivo no encontrado: {txt_path}")
        return None

    try:
        headers, notes = parse_ultrastar_for_qa(txt_path)
        if not notes:
            print(f"{ULTRASINGER_HEAD} {red_highlighted('Visual QA:')} No se encontraron notas en {txt_path}")
            return None

        # Determinar nombre base y carpeta de destino
        song_title = headers.get("TITLE", os.path.splitext(os.path.basename(txt_path))[0])
        song_artist = headers.get("ARTIST", "")
        song_folder = output_folder or os.path.dirname(os.path.abspath(txt_path))
        song_dir_name = os.path.basename(song_folder.rstrip("/\\"))

        # Preparar curva continua F0
        f0_times, f0_midis = prepare_f0_curve(
            pitched_data=pitched_data,
            vocal_audio_path=vocal_audio_path,
            pitcher_name=pitcher_name,
            device=device,
        )

        # Determinar rango temporal total
        last_note_end = max(n.end_sec for n in notes) if notes else 0.0
        audio_end = float(f0_times[-1]) if len(f0_times) > 0 else 0.0
        total_duration = max(last_note_end + 1.5, audio_end)

        num_rows = max(1, int(math.ceil(total_duration / seconds_per_row)))

        # Determinar rango tonal común (Y-axis)
        all_note_midis = [n.midi_note for n in notes]
        valid_f0 = f0_midis[~np.isnan(f0_midis)] if len(f0_midis) > 0 else np.array([])

        min_midi = min(all_note_midis) if all_note_midis else 48
        max_midi = max(all_note_midis) if all_note_midis else 72
        if len(valid_f0) > 0:
            min_midi = min(min_midi, int(np.floor(np.percentile(valid_f0, 2))))
            max_midi = max(max_midi, int(np.ceil(np.percentile(valid_f0, 98))))

        y_min = max(24, min_midi - 2)
        y_max = min(96, max_midi + 3)

        # Configuración de la figura Matplotlib en modo oscuro
        fig_height = max(4.0, 2.2 * num_rows)
        fig, axes = plt.subplots(
            nrows=num_rows,
            ncols=1,
            figsize=(18, fig_height),
            facecolor="#121216",
            sharey=True,
        )

        if num_rows == 1:
            axes = [axes]

        header_title = f"{song_artist} - {song_title}" if song_artist else song_title
        fig.suptitle(
            f"Control de Calidad Visual (QA) | {header_title}\n"
            f"Barras UltraStar (Hitboxes) vs Curva F0 ({pitcher_name.upper()})",
            fontsize=13,
            fontweight="bold",
            color="#E0E0E0",
            y=0.995,
        )

        # Renderizar cada fila (subplot de 15 a 20 segundos)
        for row_idx, ax in enumerate(axes):
            t_row_start = row_idx * seconds_per_row
            t_row_end = t_row_start + seconds_per_row

            ax.set_facecolor("#1A1A22")
            ax.set_xlim(t_row_start, t_row_end)
            ax.set_ylim(y_min, y_max)

            # Rejilla de semitonos estilo Piano Roll
            for m in range(y_min, y_max + 1):
                is_c = (m % 12 == 0)
                is_black_key = (m % 12) in [1, 3, 6, 8, 10]
                if is_black_key:
                    ax.axhspan(m - 0.5, m + 0.5, color="#15151C", alpha=0.6, zorder=0)
                grid_color = "#3A3A48" if is_c else "#252530"
                grid_alpha = 0.7 if is_c else 0.4
                ax.axhline(m, color=grid_color, linestyle="--", linewidth=0.6, alpha=grid_alpha, zorder=1)

            # 1. Dibujar Hitboxes de UltraStar
            is_duet = any(n.player == 2 for n in notes)
            row_notes = [n for n in notes if n.end_sec >= t_row_start and n.start_sec <= t_row_end]
            for n in row_notes:
                if n.player == 2:
                    face_col = "#FF9F43" if n.note_type == "*" else ("#A55EEA" if n.note_type == ":" else "#FD79A8")
                    edge_col = "#EE5253" if n.note_type == "*" else ("#8854D0" if n.note_type == ":" else "#E84393")
                else:
                    face_col = "#FFD166" if n.note_type == "*" else ("#00B4D8" if n.note_type == ":" else "#9D4EDD")
                    edge_col = "#FFB703" if n.note_type == "*" else ("#0077B6" if n.note_type == ":" else "#7B2CBF")
                alpha_val = 0.75 if n.note_type == "*" else 0.55

                # Rectángulo de la barra
                rect = patches.Rectangle(
                    (n.start_sec, n.midi_note - 0.4),
                    n.duration_sec,
                    0.8,
                    linewidth=1.2,
                    edgecolor=edge_col,
                    facecolor=face_col,
                    alpha=alpha_val,
                    zorder=2,
                )
                ax.add_patch(rect)

                # Texto de la sílaba
                txt_x = n.start_sec + 0.05
                if t_row_start <= txt_x <= t_row_end:
                    clean_word = n.word.strip()
                    if clean_word:
                        prefix = f"[P{n.player}] " if is_duet else ""
                        txt_obj = ax.text(
                            txt_x,
                            n.midi_note,
                            f"{prefix}{clean_word}",
                            fontsize=8,
                            fontweight="bold",
                            color="white",
                            va="center",
                            zorder=4,
                        )
                        txt_obj.set_path_effects(
                            [path_effects.withStroke(linewidth=2.5, foreground="#0A0A0E")]
                        )

            # 2. Superponer Curva Continua F0
            if len(f0_times) > 0:
                f0_mask = (f0_times >= t_row_start - 0.1) & (f0_times <= t_row_end + 0.1)
                if np.any(f0_mask):
                    ax.plot(
                        f0_times[f0_mask],
                        f0_midis[f0_mask],
                        color="#FF0055",
                        linewidth=1.7,
                        alpha=0.92,
                        zorder=3,
                        label="F0 Acústico" if row_idx == 0 else "",
                    )

            # Configuración de Ejes
            ticks_x = np.arange(t_row_start, t_row_end + 0.01, 2.0)
            ax.set_xticks(ticks_x)
            ax.set_xticklabels([f"{int(t // 60):02d}:{int(t % 60):02d}" for t in ticks_x], color="#8E8E9B", fontsize=7.5)

            # Eje Y musical (solo en las notas C y F para evitar saturación visual)
            y_ticks = [m for m in range(y_min, y_max + 1) if (m % 12 == 0) or (m % 12 == 5)]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([midi_to_note_name(m) for m in y_ticks], color="#A0A0B0", fontsize=7.5)

            ax.tick_params(axis="x", colors="#555566", labelsize=7)
            ax.tick_params(axis="y", colors="#555566", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#2E2E3C")

            # Etiqueta de rango de fila
            time_badge = f"[{int(t_row_start // 60):02d}:{int(t_row_start % 60):02d} - {int(t_row_end // 60):02d}:{int(t_row_end % 60):02d}]"
            ax.text(
                t_row_start + 0.2,
                y_max - 0.7,
                time_badge,
                fontsize=7.5,
                color="#666677",
                fontweight="bold",
                zorder=5,
            )

        # Ajuste de espacio entre subplots
        plt.tight_layout(rect=[0.02, 0.01, 0.98, 0.98])
        plt.subplots_adjust(hspace=0.35)

        # 3. Exportación Estricta a /output/<Nombre_Cancion>/png/<Nombre_Cancion>_[QA_Visual].png
        png_dir = os.path.join(song_folder, "png")
        os.makedirs(png_dir, exist_ok=True)
        final_png_path = os.path.join(png_dir, f"{song_dir_name}_[QA_Visual].png")

        fig.savefig(
            final_png_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        plt.close(fig)
        gc.collect()

        print(f"{ULTRASINGER_HEAD} {green_highlighted('Visual QA:')} Gráfico generado en -> {blue_highlighted(final_png_path)}")
        return final_png_path

    except Exception as e:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Error al generar Visual QA Plot:')} {e}")
        return None
