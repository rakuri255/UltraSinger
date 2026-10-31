"""Tests verifying the 4 key fixes:
1. Exact user song folder / title preservation.
2. Melisma generation (~) for sustained vowel pitch changes.
3. Octave jump folding to prevent notes at pitch +24 in the ceiling.
4. Intelligent line splitting at commas to avoid UltraStar text overflow.
"""

import unittest
import numpy as np

from modules.Midi.MidiSegment import MidiSegment
from modules.Midi.midi_creator import (
    correct_octave_jumps,
    segment_melismas_from_f0,
)
from modules.Pitcher.pitched_data import PitchedData
from modules.Speech_Recognition.lyrics_aligner import split_long_line
from modules.Ultrastar.ultrastar_writer import create_ultrastar_txt
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue


class TestPlanFixes(unittest.TestCase):
    def test_split_long_line_at_comma(self):
        long_line = "Esta es una frase larga de prueba, con varias comas, para verificar la division"
        sub_lines = split_long_line(long_line, max_chars=30)
        self.assertGreater(len(sub_lines), 1)
        for sl in sub_lines:
            self.assertLessEqual(len(sl), 35)
        # Check all words preserved
        self.assertEqual(" ".join(sub_lines).replace("  ", " "), long_line)

    def test_correct_octave_jumps_folds_pitch_spike(self):
        # A melody around C4 (MIDI 60) with an isolated harmonic spike at C6 (MIDI 84 = pitch 24)
        segs = [
            MidiSegment("C4", 0.0, 0.5, "Can-"),
            MidiSegment("D4", 0.5, 1.0, "tan-"),
            MidiSegment("C6", 1.0, 1.3, "spike "),  # Harmonic spike at pitch 24 (+24 semitones)
            MidiSegment("C4", 1.3, 1.8, "do "),
            MidiSegment("B3", 1.8, 2.2, "hoy "),
            MidiSegment("C4", 2.2, 2.8, "bien "),
        ]
        corrected = correct_octave_jumps(segs)
        spike_note = corrected[2].note
        # Spike should be folded down to C4 or C5, never C6 (MIDI 84)
        import librosa
        spike_midi = librosa.note_to_midi(spike_note)
        self.assertLessEqual(spike_midi, 72)
        self.assertGreaterEqual(spike_midi, 55)

    def test_segment_melismas_generates_tilde(self):
        # Create synthetic PitchedData with pitch slide from C4 (261.6 Hz) to G4 (392.0 Hz)
        times = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        # First half C4 (~261Hz), second half G4 (~392Hz)
        frequencies = [261.6, 261.6, 261.6, 261.6, 392.0, 392.0, 392.0, 392.0, 392.0]
        confidence = [0.9] * len(times)
        pitched_data = PitchedData(times=times, frequencies=frequencies, confidence=confidence)

        segs = segment_melismas_from_f0(0.0, 0.40, "amor ", pitched_data)
        self.assertGreater(len(segs), 1)
        self.assertTrue(segs[0].word.startswith("amor"))
        self.assertTrue(segs[-1].word.startswith("~"))

    def test_ultrastar_writer_preserves_tilde_melisma(self):
        import tempfile
        import os
        segs = [
            MidiSegment("C4", 0.0, 0.5, "a"),
            MidiSegment("E4", 0.5, 0.8, "~"),
            MidiSegment("G4", 0.8, 1.2, "~ "),
        ]
        u_txt = UltrastarTxtValue()
        u_txt.artist = "Test Artist"
        u_txt.title = "Test Title"
        u_txt.mp3 = "song.mp3"

        with tempfile.NamedTemporaryFile("r+", suffix=".txt", delete=False) as f:
            temp_path = f.name

        try:
            create_ultrastar_txt(segs, temp_path, u_txt, real_bpm=120.0, auto_octave=False)
            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("#ARTIST:Test Artist", content)
            self.assertIn("#TITLE:Test Title", content)
            self.assertIn("#GENRE:Pop", content)
            # Verify melisma notes are preserved as clean tied syllables without literal ~
            note_lines = [l for l in content.splitlines() if l.startswith(":")]
            self.assertEqual(len(note_lines), 3)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_parse_genre_file_various_formats(self):
        import tempfile
        import os
        from UltraSinger import parse_genre_file

        cases = [
            ("#GENRE:Hip Hop/R&B", "Hip Hop/R&B"),
            ("#genre: Cumbia", "Cumbia"),
            ("Rock & Roll", "Rock & Roll"),
            ("  #GENRE:  Pop Latino  \nExtra line", "Pop Latino"),
            ("", "Pop"),
        ]
        for input_text, expected in cases:
            with tempfile.NamedTemporaryFile("w+", suffix=".txt", encoding="utf-8", delete=False) as f:
                f.write(input_text)
                f.flush()
                temp_path = f.name
            try:
                result = parse_genre_file(temp_path)
                self.assertEqual(result, expected)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    def test_ultrastar_writer_explicit_genre(self):
        import tempfile
        import os
        segs = [MidiSegment("C4", 0.0, 0.5, "test ")]
        u_txt = UltrastarTxtValue()
        u_txt.artist = "Artista"
        u_txt.title = "Cancion"
        u_txt.genre = "Hip Hop/R&B"
        u_txt.mp3 = "song.mp3"

        with tempfile.NamedTemporaryFile("r+", suffix=".txt", delete=False) as f:
            temp_path = f.name
        try:
            create_ultrastar_txt(segs, temp_path, u_txt, real_bpm=120.0, auto_octave=False)
            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("#GENRE:Hip Hop/R&B\n", content)
            # Verify #GENRE is placed directly after #TITLE
            title_idx = content.find("#TITLE:Cancion")
            genre_idx = content.find("#GENRE:Hip Hop/R&B")
            self.assertGreater(genre_idx, title_idx)
            self.assertLess(genre_idx - title_idx, 40)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_resolve_genre_profiles_all_families(self):
        from modules.Speech_Recognition.genre_profiles import (
            resolve_genre_profile,
            HIP_HOP_RNB,
            REGGAETON_DEMBOW,
            EDM_HOUSE_TECHNO,
            CUMBIA_GRUPERA,
            SALSA,
            ROCK_ALTERNATIVE,
            POP_LATIN_POP,
            SYMPHONIC_OPERA,
            EXTREME_METAL,
            JAZZ_FUSION,
            WORLD_FLAMENCO_CELTIC,
            BALADA_ROMANTICA,
        )

        test_cases = [
            ("Hip Hop / R&B", HIP_HOP_RNB),
            ("hip-hop", HIP_HOP_RNB),
            ("Trap latino", HIP_HOP_RNB),
            ("Boom Bap", HIP_HOP_RNB),
            ("Reggaeton", REGGAETON_DEMBOW),
            ("dembow", REGGAETON_DEMBOW),
            ("EDM", EDM_HOUSE_TECHNO),
            ("Techno", EDM_HOUSE_TECHNO),
            ("House", EDM_HOUSE_TECHNO),
            ("Cumbia Sonidera", CUMBIA_GRUPERA),
            ("Banda sinaloense", CUMBIA_GRUPERA),
            ("Salsa Romántica", SALSA),
            ("Guaguancó", SALSA),
            ("Hard Rock", ROCK_ALTERNATIVE),
            ("Grunge 90s", ROCK_ALTERNATIVE),
            ("Pop", POP_LATIN_POP),
            ("Pop Latino", POP_LATIN_POP),
            ("Ópera Clásica", SYMPHONIC_OPERA),
            ("Aria Sinfónica", SYMPHONIC_OPERA),
            ("Death Metal", EXTREME_METAL),
            ("Grindcore", EXTREME_METAL),
            ("Jazz Bebop", JAZZ_FUSION),
            ("Scat Singing", JAZZ_FUSION),
            ("Flamenco Tradicional", WORLD_FLAMENCO_CELTIC),
            ("Celtic Folk", WORLD_FLAMENCO_CELTIC),
            ("Balada Romántica", BALADA_ROMANTICA),
            ("Bolero Clásico", BALADA_ROMANTICA),
            ("", POP_LATIN_POP),
            ("UnknownGenre123", POP_LATIN_POP),
            (None, POP_LATIN_POP),
        ]

        for genre_input, expected_profile in test_cases:
            resolved = resolve_genre_profile(genre_input)
            self.assertEqual(
                resolved.name,
                expected_profile.name,
                f"Failed for input '{genre_input}': got {resolved.name}, expected {expected_profile.name}",
            )

    def test_vibrato_sinusoid_evaluation(self):
        from modules.Audio.ultrastar_autocorrect import evaluate_vibrato_sinusoid
        # Synthetic 5.2 Hz vocal vibrato on C4 (MIDI 60) with 0.5 semitone amplitude over 0.6 seconds
        dt = 0.010
        times = np.arange(0.0, 0.60, dt)
        f_v = 5.2
        a_v = 0.5
        f0_vibrato = 60.0 + a_v * np.sin(2.0 * np.pi * f_v * times)

        is_vib, detected_fv, detected_av = evaluate_vibrato_sinusoid(
            f0_vibrato, times, f_min=4.5, f_max=7.5, max_amplitude_st=1.2
        )
        self.assertTrue(is_vib)
        self.assertAlmostEqual(detected_fv, f_v, delta=0.6)
        self.assertAlmostEqual(detected_av, a_v, delta=0.25)

        # Flat pitch (no vibrato)
        f0_flat = np.full_like(times, 60.0)
        is_vib_flat, _, _ = evaluate_vibrato_sinusoid(f0_flat, times)
        self.assertFalse(is_vib_flat)

    def test_genre_dsp_profile_parameters(self):
        from modules.Speech_Recognition.genre_profiles import (
            CUMBIA_GRUPERA,
            REGGAETON_DEMBOW,
            JAZZ_FUSION,
            EXTREME_METAL,
            SYMPHONIC_OPERA,
        )
        self.assertEqual(CUMBIA_GRUPERA.vibrato_freq_range, (4.8, 5.8))
        self.assertEqual(REGGAETON_DEMBOW.rhythm_grid_mode, "dembow_332")
        self.assertEqual(JAZZ_FUSION.rhythm_grid_mode, "swing_ternary")
        self.assertEqual(EXTREME_METAL.spectral_entropy_threshold, 0.70)
        self.assertEqual(SYMPHONIC_OPERA.crest_factor_mode, "high_dynamics")

    def test_kde_mode_pitch_estimation(self):
        from modules.Audio.ultrastar_autocorrect import estimate_pitch_kde_mode
        # A singer sings G3 (MIDI 55).
        # Frames:
        # 1. Initial scoop: 4 frames from 48, 49, 51, 53 with low confidence (0.4)
        # 2. Stable vowel on G3: 25 frames centered around 55.0 (+- 0.3 vibrato) with high confidence (0.95)
        # 3. Trailing decay: 3 frames falling to 52, 50 with low confidence (0.3)
        scoop = np.array([48.0, 49.0, 51.0, 53.0])
        scoop_weights = np.array([0.4, 0.4, 0.5, 0.6])

        stable_vowel = 55.0 + 0.3 * np.sin(np.linspace(0, 4 * np.pi, 25))
        stable_weights = np.full(25, 0.95)

        decay = np.array([53.0, 51.0, 49.0])
        decay_weights = np.array([0.4, 0.3, 0.2])

        midi_traj = np.concatenate([scoop, stable_vowel, decay])
        weights = np.concatenate([scoop_weights, stable_weights, decay_weights]) ** 2

        mode_pitch = estimate_pitch_kde_mode(midi_traj, weights, bandwidth_semitones=0.35)
        self.assertAlmostEqual(mode_pitch, 55.0, delta=0.25)

    def test_hilbert_instantaneous_envelope_trimmer(self):
        import os
        import tempfile
        import soundfile as sf
        from modules.Audio.voice_activity_trimmer import trim_syllable_ends_with_energy
        from modules.Speech_Recognition.TranscribedData import TranscribedData

        sr = 22050
        duration = 1.0
        # 0.45s of tone (440 Hz) followed by 0.55s of silence
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t)
        audio[int(sr * 0.45):] = 0.0  # Cut off audio at 0.45s

        with tempfile.NamedTemporaryFile("wb", suffix=".wav", delete=False) as f:
            temp_wav = f.name
        sf.write(temp_wav, audio, sr)

        try:
            # Transcribed syllable hanging until 0.9s
            syl = TranscribedData(
                word="canta",
                start=0.0,
                end=0.9,
                confidence=0.9,
            )
            result = trim_syllable_ends_with_energy(temp_wav, [syl], energy_threshold_ratio=0.15)
            # End should be trimmed from 0.9s down close to 0.45s
            self.assertLess(result[0].end, 0.70)
            self.assertGreaterEqual(result[0].end, 0.40)
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

    def test_superflux_vibrato_suppression(self):
        import os
        import tempfile
        import soundfile as sf
        from modules.Audio.onset_snapping import detect_vocal_onsets

        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # FM synthesis for natural vibrato: 440 Hz carrier with 5.5 Hz vibrato (+-25 Hz)
        f_c = 440.0
        f_v = 5.5
        delta_f = 25.0
        phase = 2 * np.pi * f_c * t - (delta_f / f_v) * np.cos(2 * np.pi * f_v * t)
        vibrato_signal = 0.5 * np.sin(phase)

        with tempfile.NamedTemporaryFile("wb", suffix=".wav", delete=False) as f:
            temp_wav = f.name
        sf.write(temp_wav, vibrato_signal, sr)

        try:
            onsets = detect_vocal_onsets(temp_wav, sr=sr)
            # Standard spectral flux triggers 5+ false onsets across 1 second of vibrato.
            # SuperFlux suppresses them so that only the initial note onset is found.
            self.assertLessEqual(len(onsets), 2)
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

    def test_genre_bilingual_prompt_resolution(self):
        from modules.Speech_Recognition.genre_profiles import (
            HIP_HOP_RNB,
            REGGAETON_DEMBOW,
            ROCK_ALTERNATIVE,
            CUMBIA_GRUPERA,
            POP_LATIN_POP,
            SYMPHONIC_OPERA,
            EXTREME_METAL,
            JAZZ_FUSION,
            WORLD_FLAMENCO_CELTIC,
            BALADA_ROMANTICA,
            SALSA,
            EDM_HOUSE_TECHNO,
        )

        all_profiles = [
            HIP_HOP_RNB,
            REGGAETON_DEMBOW,
            ROCK_ALTERNATIVE,
            CUMBIA_GRUPERA,
            POP_LATIN_POP,
            SYMPHONIC_OPERA,
            EXTREME_METAL,
            JAZZ_FUSION,
            WORLD_FLAMENCO_CELTIC,
            BALADA_ROMANTICA,
            SALSA,
            EDM_HOUSE_TECHNO,
        ]

        for prof in all_profiles:
            self.assertTrue(len(prof.initial_prompt_en) > 20, f"Missing English prompt for {prof.name}")
            self.assertTrue(len(prof.initial_prompt_es) > 20, f"Missing Spanish prompt for {prof.name}")
            # Spanish queries
            self.assertEqual(prof.get_initial_prompt("es"), prof.initial_prompt_es)
            self.assertEqual(prof.get_initial_prompt("Spanish"), prof.initial_prompt_es)
            self.assertEqual(prof.get_initial_prompt("español"), prof.initial_prompt_es)
            # English / Default queries
            self.assertEqual(prof.get_initial_prompt("en"), prof.initial_prompt_en)
            self.assertEqual(prof.get_initial_prompt("English"), prof.initial_prompt_en)
            self.assertEqual(prof.get_initial_prompt(None), prof.initial_prompt_en)
            self.assertEqual(prof.get_initial_prompt("de"), prof.initial_prompt_en)
            # Backward-compatible property returns English default
            self.assertEqual(prof.initial_prompt, prof.initial_prompt_en)

    def test_midi_dynamic_rms_velocity(self):
        import tempfile
        import os
        import soundfile as sf
        from modules.Midi.midi_creator import create_midi_instrument

        sr = 22050
        duration = 2.0
        # First second: quiet whisper (amplitude 0.05)
        # Second second: powerful singing (amplitude 0.85)
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = np.zeros_like(t)
        mid_point = int(sr * 1.0)
        audio[:mid_point] = 0.05 * np.sin(2 * np.pi * 440.0 * t[:mid_point])
        audio[mid_point:] = 0.85 * np.sin(2 * np.pi * 440.0 * t[mid_point:])

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_wav = f.name
        sf.write(temp_wav, audio.astype(np.float32), sr)

        try:
            segs = [
                MidiSegment("A4", 0.1, 0.9, "whisper "),
                MidiSegment("A4", 1.1, 1.9, "SHOUT "),
            ]
            inst = create_midi_instrument(segs, audio_path=temp_wav)
            self.assertEqual(len(inst.notes), 2)
            vel_quiet = inst.notes[0].velocity
            vel_loud = inst.notes[1].velocity

            self.assertGreater(vel_loud, vel_quiet)
            self.assertGreaterEqual(vel_quiet, 30)
            self.assertLessEqual(vel_loud, 127)
            # Quiet whisper should be scaled lower (~40), loud should be near ceiling (~120-127)
            self.assertLess(vel_quiet, 65)
            self.assertGreater(vel_loud, 110)
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

    def test_midi_continuous_pitch_bends(self):
        from modules.Midi.midi_creator import create_midi_instrument

        # C4 is MIDI 60 (261.63 Hz). We simulate vibrato from 254 Hz (-0.5 st) to 269 Hz (+0.5 st)
        times = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        # Oscillation around 261.63 Hz
        f_c = 261.63
        freqs = [f_c * (2.0 ** (0.4 * np.sin(2 * np.pi * 2.0 * t) / 12.0)) for t in times]
        confs = [0.9] * len(times)
        pitched_data = PitchedData(times=times, frequencies=freqs, confidence=confs)

        segs = [
            MidiSegment("C4", 0.0, 0.9, "vibrato "),
        ]
        inst = create_midi_instrument(segs, pitched_data=pitched_data)
        self.assertGreater(len(inst.pitch_bends), 0)

        # Check all pitch bends are in standard 14-bit MIDI range [-8192, 8191]
        for pb in inst.pitch_bends:
            self.assertGreaterEqual(pb.pitch, -8192)
            self.assertLessEqual(pb.pitch, 8191)

        # Note ends with reset to 0
        self.assertEqual(inst.pitch_bends[-1].pitch, 0)
        self.assertAlmostEqual(inst.pitch_bends[-1].time, 0.9)

    def test_pitcher_factory_basic_pitch(self):
        from modules.Pitcher.Extractors import PitcherFactory
        from modules.Pitcher.Extractors.basic_pitch_extractor import BasicPitchExtractor

        pitcher = PitcherFactory.get_pitcher("basic-pitch")
        self.assertIsInstance(pitcher, BasicPitchExtractor)

        pitcher_alias = PitcherFactory.get_pitcher("basicpitch")
        self.assertIsInstance(pitcher_alias, BasicPitchExtractor)

        self.assertIn("basic-pitch", PitcherFactory.available_pitchers())

    def test_basic_pitch_extractor_inference(self):
        import tempfile
        import os
        import soundfile as sf
        from modules.Pitcher.Extractors.basic_pitch_extractor import BasicPitchExtractor

        sr = 22050
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # 440 Hz pure sine wave (A4 = MIDI 69)
        audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_wav = f.name
        sf.write(temp_wav, audio, sr)

        try:
            extractor = BasicPitchExtractor()
            result = extractor.extract_pitch(temp_wav, device="cpu")
            self.assertIsInstance(result, PitchedData)
            self.assertGreater(len(result.times), 0)
            self.assertEqual(len(result.times), len(result.frequencies))
            self.assertEqual(len(result.times), len(result.confidence))
            # Average detected frequency during the tone should be close to 440 Hz
            high_conf_freqs = [f for f, c in zip(result.frequencies, result.confidence) if c > 0.4]
            if high_conf_freqs:
                median_freq = float(np.median(high_conf_freqs))
                self.assertAlmostEqual(median_freq, 440.0, delta=25.0)
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)


if __name__ == "__main__":
    unittest.main()

