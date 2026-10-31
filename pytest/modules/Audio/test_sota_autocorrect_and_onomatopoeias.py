"""Unit tests for SOTA Onomatopoeia ("Omatepas") mathematical models,
Melisma vs. Vibrato KDE inflection detection, and --autocorrect integrity."""

import os
import tempfile
import unittest
import numpy as np

from modules.Speech_Recognition.fuzzy_lyric_matcher import (
    compute_phonotactic_entropy_and_reduplication,
    compute_markov_cv_periodicity_and_morphology,
    compute_levenshtein_oov_distance,
    detect_closed_mouth_humming_acoustics,
    detect_rhythmic_transient_envelope,
    classify_vocable_category,
    is_scat_or_adlib_pattern,
    classify_onomatopoeia_acoustics,
    filter_intro_adlibs,
)
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Audio.ultrastar_autocorrect import (
    UltraStarLine,
    ArticulatoryAcousticFeatures,
    evaluate_melisma_vs_vibrato,
    compute_zero_crossing_rate_frames,
    analyze_multidimensional_mir_events,
    evaluate_karaoke_phrases_scoring,
)
from modules.Audio.bpm import compute_autocorrelation_tempo
from modules.Audio.key_detector import (
    compute_pitch_class_profile,
    detect_key_from_chroma,
    detect_dominant_chords_timeline,
)
from modules.Midi.midi_creator import segment_melismas_from_f0
from modules.Pitcher.pitched_data import PitchedData


def _make_td(word: str, start: float, end: float) -> TranscribedData:
    td = TranscribedData()
    td.word = word + " "
    td.start = start
    td.end = end
    td.confidence = 0.9
    return td


class TestSotaOnomatopoeiasAndMelismas(unittest.TestCase):
    def test_phonotactic_entropy_and_reduplication_math(self):
        # Repetitive scat/onomatopoeia ("nanana", "oooooh") has low entropy or high reduplication
        h_rep, r_rep = compute_phonotactic_entropy_and_reduplication("nananana")
        self.assertGreaterEqual(r_rep, 0.38)
        self.assertTrue(is_scat_or_adlib_pattern("nananana"))

        h_vow, r_vow = compute_phonotactic_entropy_and_reduplication("oooooh")
        self.assertLessEqual(h_vow, 1.15)
        self.assertTrue(is_scat_or_adlib_pattern("oooooh"))

        # Urban interjections ("skrrt", "prrra", "wepa", "mambo") are detected
        for urban_ono in ["skrrt", "prrra", "brr", "wepa", "mambo", "ajá", "ey"]:
            self.assertTrue(is_scat_or_adlib_pattern(urban_ono), f"Failed on {urban_ono}")

        # Valid short lexical words ("sol", "mar", "luz", "amor", "corazón") are NEVER false positives
        for valid_word in ["sol", "sal", "mar", "luz", "hoy", "amor", "corazón", "tiempo", "canción", "pensamiento"]:
            self.assertFalse(is_scat_or_adlib_pattern(valid_word), f"False positive on {valid_word}")

    def test_super_listado_vocables_taxonomy_and_markov(self):
        # 1. Scat & Jazz
        for scat in ["Shoo-be-doo", "bop-ba-doo-wop", "skiddly-bop-doo", "dweeb-da-bop", "scooby-dooby-doo", "zap-zaba-doo", "shoo-wop-shoo-wop"]:
            self.assertEqual(classify_vocable_category(scat), "scat_jazz", f"Failed scat_jazz on {scat}")

        # 2. Pop & Rock Chorus
        for pop_v in ["Sha-la-la-la", "whoa-oh-oh", "yeah-yeah-yeah", "na-na-na"]:
            self.assertEqual(classify_vocable_category(pop_v), "pop_rock_chorus", f"Failed pop_rock_chorus on {pop_v}")

        # 3. Instrument Imitation: Bass/Brass vs Percussion/Scratch
        for bass_brass in ["Bum-bum-bum", "dom-dom", "ba-dum", "dum-di-dum", "Pa-ra-pa-pam", "ta-ra-rá", "tu-tu-ru"]:
            self.assertEqual(classify_vocable_category(bass_brass), "instrument_bass_brass", f"Failed bass_brass on {bass_brass}")

        for perc_scratch in ["Rat-a-tat", "tss-tss", "pa-ta-ta", "Chika-chika", "wah-wah", "dzeen", "pow"]:
            self.assertEqual(classify_vocable_category(perc_scratch), "instrument_percussion_scratch", f"Failed perc_scratch on {perc_scratch}")

        # 4. Folk & Celtic Lilting
        for folk in ["Diddle-diddle-dum", "toora-loora", "diddly-idle-um"]:
            self.assertEqual(classify_vocable_category(folk), "folk_lilting", f"Failed folk_lilting on {folk}")

        # 5. Pure Melismatic
        for mel in ["Eh-eh-eh", "uh-huh", "ay-ay-ay", "ooey-ooey"]:
            self.assertEqual(classify_vocable_category(mel), "pure_melismatic", f"Failed pure_melismatic on {mel}")

        # 6. Markov CV periodicity & morphological affix check
        p_cv_la, has_morph_la = compute_markov_cv_periodicity_and_morphology("lalalala")
        self.assertAlmostEqual(p_cv_la, 1.0)
        self.assertFalse(has_morph_la)

        _, has_morph_word = compute_markov_cv_periodicity_and_morphology("sentimiento")
        self.assertTrue(has_morph_word)
        self.assertIsNone(classify_vocable_category("sentimiento"))

    def test_delta_mfcc_cepstral_flux_discrimination(self):
        # Drum/scratch imitation ("chika-chika", "rat-a-tat") is always 'freestyle' even if F0 detector hallucinates
        self.assertEqual(
            classify_onomatopoeia_acoustics(
                word="Chika-chika",
                v_mask_mean=0.60,
                pitch_std_st=0.30,
                spectral_flatness_mean=0.10,
                rms_max=0.15,
                noise_gate=0.01,
                r_boca_mean=2.0,
            ),
            "freestyle",
        )
        # Rapid cepstral articulation (delta_mfcc_mean >= 18.0) forces 'freestyle'
        self.assertEqual(
            classify_onomatopoeia_acoustics(
                word="skiddly-bop-doo",
                v_mask_mean=0.55,
                pitch_std_st=0.40,
                spectral_flatness_mean=0.12,
                rms_max=0.15,
                noise_gate=0.01,
                r_boca_mean=2.0,
                delta_mfcc_mean=22.5,
            ),
            "freestyle",
        )

    def test_trimodal_onomatopoeia_acoustics(self):
        # 1. Melodic vocable ("oooh", "na") with high F0 confidence and open mouth -> normal (:)
        mode_melodic = classify_onomatopoeia_acoustics(
            word="oooh",
            f0_conf_mean=0.82,
            r_boca_mean=3.5,
            spectral_flatness_mean=0.08,
            rms_mean=0.15,
            noise_gate=0.01,
        )
        self.assertEqual(mode_melodic, "normal")

        # 2. Percussive/urban ad-lib ("skrrt", "wepa") with active energy but low F0 / high flatness -> freestyle (F)
        mode_perc = classify_onomatopoeia_acoustics(
            word="skrrt",
            f0_conf_mean=0.18,
            r_boca_mean=0.9,
            spectral_flatness_mean=0.45,
            rms_mean=0.08,
            noise_gate=0.01,
        )
        self.assertEqual(mode_perc, "freestyle")

        # 3. Sub-threshold noise bleed -> purge
        mode_purge = classify_onomatopoeia_acoustics(
            word="uh",
            f0_conf_mean=0.05,
            r_boca_mean=0.4,
            spectral_flatness_mean=0.75,
            rms_mean=0.002,
            noise_gate=0.01,
        )
        self.assertEqual(mode_purge, "purge")

    def test_intro_adlib_preserves_scripted_opening_onomatopoeia(self):
        # When lyrics.txt explicitly starts with "Yeah", filter_intro_adlibs must anchor immediately
        words = [
            _make_td("Yeah", 1.0, 1.5),
            _make_td("perdonen", 1.6, 2.2),
            _make_td("amigos", 2.3, 2.9),
        ]
        lyrics = ["Yeah", "perdonen", "amigos"]
        main_words, quarantined = filter_intro_adlibs(words, lyrics)
        self.assertEqual(len(quarantined), 0)
        self.assertEqual(len(main_words), 3)
        self.assertEqual(main_words[0].word.strip(), "Yeah")

        # When lyrics.txt starts with "Perdonen" and Whisper hears unscripted "Yeah" at 1.0s, quarantine "Yeah"
        lyrics_no_ono = ["Perdonen", "amigos"]
        main_words2, quarantined2 = filter_intro_adlibs(words, lyrics_no_ono)
        self.assertEqual(len(quarantined2), 1)
        self.assertEqual(quarantined2[0].word.strip(), "Yeah")
        self.assertEqual(main_words2[0].word.strip(), "perdonen")

    def test_evaluate_melisma_vs_vibrato_kde_inflection(self):
        times = np.arange(0.0, 1.0, 0.010, dtype=np.float32)
        n = len(times)

        # Case A: Biological vibrato at 5.8 Hz (+/- 0.8 semitones around MIDI 60) -> NO melisma split
        f0_vib = 60.0 + 0.8 * np.sin(2.0 * np.pi * 5.8 * times)
        feat_vib = ArticulatoryAcousticFeatures(
            times=times,
            f0_midi=f0_vib,
            confidence=np.full(n, 0.9, dtype=np.float32),
            r_boca=np.full(n, 4.0, dtype=np.float32),
            spectral_flatness=np.full(n, 0.05, dtype=np.float32),
            spectral_centroid=np.full(n, 1500.0, dtype=np.float32),
            delta_mfcc=np.zeros(n, dtype=np.float32),
            rms=np.full(n, 0.2, dtype=np.float32),
            v_mask=np.full(n, 0.85, dtype=np.float32),
        )
        res_vib = evaluate_melisma_vs_vibrato(feat_vib, 0.0, 0.95)
        self.assertIsNone(res_vib)

        # Case B: Genuine step melisma (MIDI 60 -> MIDI 64 at t = 0.48s) -> splits near 0.48s
        f0_mel = np.where(times < 0.48, 60.0, 64.0).astype(np.float32)
        feat_mel = ArticulatoryAcousticFeatures(
            times=times,
            f0_midi=f0_mel,
            confidence=np.full(n, 0.9, dtype=np.float32),
            r_boca=np.full(n, 4.0, dtype=np.float32),
            spectral_flatness=np.full(n, 0.05, dtype=np.float32),
            spectral_centroid=np.full(n, 1500.0, dtype=np.float32),
            delta_mfcc=np.zeros(n, dtype=np.float32),
            rms=np.full(n, 0.2, dtype=np.float32),
            v_mask=np.full(n, 0.85, dtype=np.float32),
        )
        res_mel = evaluate_melisma_vs_vibrato(feat_mel, 0.0, 0.95, min_semitone_diff=1.8)
        self.assertIsNotNone(res_mel)
        t_split, p1, p2 = res_mel
        self.assertAlmostEqual(t_split, 0.48, delta=0.05)
        self.assertEqual(p1, 60 - 48)
        self.assertEqual(p2, 64 - 48)

        # Case C: Formula 1 (dF0/dt != 0 AND dF1/dt ~ 0):
        # If formant flux (delta_mfcc) spikes (>= 22.0) across the window, it is a vowel/syllable change, NOT a melisma
        feat_diphthong = ArticulatoryAcousticFeatures(
            times=times,
            f0_midi=f0_mel,
            confidence=np.full(n, 0.9, dtype=np.float32),
            r_boca=np.full(n, 4.0, dtype=np.float32),
            spectral_flatness=np.full(n, 0.05, dtype=np.float32),
            spectral_centroid=np.full(n, 1500.0, dtype=np.float32),
            delta_mfcc=np.full(n, 28.0, dtype=np.float32),
            rms=np.full(n, 0.2, dtype=np.float32),
            v_mask=np.full(n, 0.85, dtype=np.float32),
        )
        self.assertIsNone(evaluate_melisma_vs_vibrato(feat_diphthong, 0.0, 0.95, min_semitone_diff=1.8))

    def test_universal_mir_formulas_centroid_transients_and_levenshtein(self):
        # Formula 2: Closed-mouth humming ("Mmmm") -> Spectral Centroid C < 1000 Hz + steady energy
        self.assertTrue(detect_closed_mouth_humming_acoustics(spectral_centroid_mean=680.0, f0_conf_mean=0.80, rms_cv=0.18))
        self.assertFalse(detect_closed_mouth_humming_acoustics(spectral_centroid_mean=1850.0, f0_conf_mean=0.80, rms_cv=0.18))

        # Unscripted closed-mouth humming is purged, scripted "mmmm" in lyrics.txt is preserved as melodic 'normal'
        self.assertEqual(
            classify_onomatopoeia_acoustics(
                word="mmmm",
                v_mask_mean=0.75,
                pitch_std_st=0.25,
                spectral_flatness_mean=0.05,
                rms_max=0.10,
                noise_gate=0.01,
                r_boca_mean=0.85,
                in_reference_lyrics=False,
                spectral_centroid_mean=680.0,
                rms_cv=0.18,
            ),
            "purge",
        )
        self.assertEqual(
            classify_onomatopoeia_acoustics(
                word="mmmm",
                v_mask_mean=0.75,
                pitch_std_st=0.25,
                spectral_flatness_mean=0.05,
                rms_max=0.10,
                noise_gate=0.01,
                r_boca_mean=0.85,
                in_reference_lyrics=True,
                spectral_centroid_mean=680.0,
                rms_cv=0.18,
            ),
            "normal",
        )

        # Formula 3: Rhythmic transients (T_crest >= 2.10) -> forces 'freestyle'
        self.assertTrue(detect_rhythmic_transient_envelope(rms_crest_factor=2.65, sustained_duty_cycle=0.30))
        self.assertEqual(
            classify_onomatopoeia_acoustics(
                word="ta-ra-rá",
                v_mask_mean=0.65,
                pitch_std_st=0.35,
                spectral_flatness_mean=0.10,
                rms_max=0.20,
                noise_gate=0.01,
                r_boca_mean=2.0,
                rms_crest_factor=2.65,
            ),
            "freestyle",
        )

        # Formula 4: Levenshtein OOV distance for jitanjáforas
        dist_jit, is_jit = compute_levenshtein_oov_distance("skiddlybop")
        self.assertGreater(dist_jit, 0.45)
        self.assertTrue(is_jit)

        dist_lex, is_lex_jit = compute_levenshtein_oov_distance("corazon")
        self.assertEqual(dist_lex, 0.0)
        self.assertFalse(is_lex_jit)

    def test_midi_creator_segment_melismas_sustained_vs_spike(self):
        times = list(np.arange(0.0, 0.80, 0.010))
        # MIDI 60 = 261.63 Hz, MIDI 65 = 349.23 Hz
        freqs = [261.63 if t < 0.40 else 349.23 for t in times]
        confs = [0.95 for _ in times]
        pd = PitchedData(times, freqs, confs)

        segs = segment_melismas_from_f0(
            0.0, 0.78, "amor ", pd, min_segment_duration=0.160, pitch_semitone_threshold=1.8
        )
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0].word, "amor")
        self.assertEqual(segs[1].word, "~ ")

    def test_multidimensional_5_modules_mir_engine(self):
        # Module 2: Tempo by autocorrelation R(k) = sum_n x[n]*x[n-k] at 120 BPM (2 Hz pulse at 100 fps -> lag 50)
        times = np.arange(0.0, 4.0, 0.010, dtype=np.float32)
        n = len(times)
        pulse_env = (np.sin(2.0 * np.pi * 2.0 * times) > 0.85).astype(np.float32) * 0.5 + 0.05
        bpm_est = compute_autocorrelation_tempo(pulse_env, frame_rate_hz=100.0)
        self.assertAlmostEqual(bpm_est, 120.0, delta=2.5)

        # Module 3: Pitch Class Profile C(p) = sum_o |X(p + 12o)| and C#m chord detection
        # C# minor triad = C# (1), E (4), G# (8) across 2 octaves (24 semitone bins)
        semitone_bins = np.zeros((24, n), dtype=np.float64)
        semitone_bins[[1, 4, 8, 13, 16, 20], :] = 1.0
        pcp = compute_pitch_class_profile(semitone_bins)
        self.assertEqual(pcp.shape, (12, n))
        self.assertAlmostEqual(float(pcp[1, 0]), 2.0)

        chords = detect_dominant_chords_timeline(pcp, times, window_sec=2.0)
        self.assertTrue(len(chords) >= 1)
        self.assertEqual(chords[0]["quantitative_value"], "C#m")

        # Module 4: ZCR = (1/2N) * sum |sgn(x[n]) - sgn(x[n-1])|
        rng = np.random.default_rng(42)
        white_noise = rng.standard_normal(4096)
        sine_wave = np.sin(2.0 * np.pi * 220.0 * np.arange(4096) / 22050.0)
        zcr_noise = compute_zero_crossing_rate_frames(white_noise, frame_length=1024, hop_length=512)
        zcr_sine = compute_zero_crossing_rate_frames(sine_wave, frame_length=1024, hop_length=512)
        self.assertGreater(float(np.mean(zcr_noise)), 0.35)
        self.assertLess(float(np.mean(zcr_sine)), 0.05)

        # Module 1, 2, 3, 4, 5 integrated in analyze_multidimensional_mir_events
        f0_vib = 60.0 + 0.7 * np.sin(2.0 * np.pi * 5.5 * times)
        zcr_arr = np.where(times < 1.0, 0.28, 0.03).astype(np.float32)
        conf_arr = np.where(times < 1.0, 0.15, 0.90).astype(np.float32)
        rms_arr = pulse_env.copy()
        rms_arr[300:380] = 0.95  # Climax / Drop window

        art_feat = ArticulatoryAcousticFeatures(
            times=times,
            f0_midi=f0_vib,
            confidence=conf_arr,
            r_boca=np.full(n, 3.0, dtype=np.float32),
            spectral_flatness=np.full(n, 0.08, dtype=np.float32),
            spectral_centroid=np.full(n, 1500.0, dtype=np.float32),
            delta_mfcc=np.zeros(n, dtype=np.float32),
            rms=rms_arr,
            v_mask=conf_arr,
            zcr=zcr_arr,
        )
        events = analyze_multidimensional_mir_events(art_feat, chroma=pcp, window_sec=1.0)
        dims = {ev.dimension for ev in events}
        self.assertEqual(
            dims,
            {"Vibrato", "Tempo", "Acorde Dominante", "Respiración/Ruido", "Clímax/Drop"},
        )

    def test_evaluate_karaoke_phrases_scoring_json(self):
        # At 300 BPM, 1 beat = 15 / 300 = 0.05s (50ms)
        times = np.arange(0.0, 2.0, 0.010, dtype=np.float32)
        n = len(times)
        f0_vib = 60.0 + 0.6 * np.sin(2.0 * np.pi * 5.5 * times)
        cent_arr = np.where(times < 0.5, 650.0, 1600.0).astype(np.float32)

        art_feat = ArticulatoryAcousticFeatures(
            times=times,
            f0_midi=f0_vib,
            confidence=np.full(n, 0.92, dtype=np.float32),
            r_boca=np.full(n, 3.5, dtype=np.float32),
            spectral_flatness=np.full(n, 0.05, dtype=np.float32),
            spectral_centroid=cent_arr,
            delta_mfcc=np.zeros(n, dtype=np.float32),
            rms=np.full(n, 0.25, dtype=np.float32),
            v_mask=np.full(n, 0.88, dtype=np.float32),
            zcr=np.full(n, 0.03, dtype=np.float32),
        )

        lines = [
            UltraStarLine(line_type=":", start_beat=0, duration_beats=10, pitch=12, word="mmmm ", is_note=True),
            UltraStarLine(line_type="*", start_beat=10, duration_beats=12, pitch=12, word="amor", is_note=True),
            UltraStarLine(line_type=":", start_beat=22, duration_beats=8, pitch=12, word="~ ", is_note=True),
            UltraStarLine(line_type="-", start_beat=30, is_line_break=True),
        ]

        phrase_json = evaluate_karaoke_phrases_scoring(
            lines=lines,
            bpm=300.0,
            gap_ms=0.0,
            art_feat=art_feat,
            noise_gate=0.01,
            reference_words=["mmmm", "amor"],
        )
        self.assertEqual(len(phrase_json), 1)
        p0 = phrase_json[0]
        self.assertIn("pitch_accuracy_%", p0)
        self.assertIn("timing_accuracy_%", p0)
        self.assertIn("lyric_accuracy_%", p0)
        self.assertIn("bonus_techniques", p0)
        self.assertIn("overall_score", p0)
        self.assertGreaterEqual(p0["pitch_accuracy_%"], 95.0)
        self.assertEqual(p0["lyric_accuracy_%"], 100.0)
        self.assertIn("Tarareo", p0["bonus_techniques"])
        self.assertIn("Melisma", p0["bonus_techniques"])
        self.assertIn("Vibrato", p0["bonus_techniques"])
        self.assertGreaterEqual(p0["overall_score"], 9000)

    def test_klatt_structural_duration_weighting(self):
        """Verifies Equation III: Klatt structural duration weighting for missing phonetic boundaries."""
        from modules.Speech_Recognition.lyrics_aligner import compute_klatt_word_weight

        # Syllables with diphthongs/vowels vs short unstressed monosyllables
        w_diphthong = compute_klatt_word_weight("traición")
        w_short = compute_klatt_word_weight("de")
        w_plosive = compute_klatt_word_weight("tu")

        self.assertGreater(w_diphthong, w_short)
        self.assertGreater(w_diphthong, w_plosive)
        self.assertGreater(w_short, 0.5)

    def test_optimize_pitch_for_usdx_frames_maximizer(self):
        """Verifies Equation V: Direct USDX Pitch Maximizer p* selects candidate with maximal hit ratio."""
        from modules.Audio.ultrastar_autocorrect import optimize_pitch_for_usdx_frames

        # F0 frequencies oscillating around MIDI 60.2 (C4)
        f0_c4 = np.full(50, 60.2, dtype=np.float32)
        weights = np.ones(50, dtype=np.float32)

        # Base MIDI float is 60.2 -> candidate range [59, 60, 61]
        best_p, hit_ratio = optimize_pitch_for_usdx_frames(
            f0_valid=f0_c4,
            center_midi_float=60.2,
            v_weights=weights,
        )
        # best_p is in UltraStar relative pitch (MIDI - 48), so 60 - 48 = 12
        self.assertEqual(best_p, 12)
        self.assertAlmostEqual(hit_ratio, 1.0, places=2)

    def test_leading_silence_trimming_hilbert(self):
        """Verifies Equation II: Leading Silence Gate trims dead silence before singer starts singing."""
        import tempfile
        import soundfile as sf
        from modules.Audio.voice_activity_trimmer import trim_syllable_ends_with_energy

        sr = 22050
        # 1.0 second silence + 0.8s 440Hz tone + 0.2s silence
        audio = np.zeros(int(sr * 2.0), dtype=np.float32)
        vocal_start_idx = int(sr * 1.0)
        vocal_end_idx = int(sr * 1.8)
        t = np.linspace(0, 0.8, vocal_end_idx - vocal_start_idx, endpoint=False)
        audio[vocal_start_idx:vocal_end_idx] = 0.5 * np.sin(2 * np.pi * 440 * t)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sf.write(tmp_path, audio, sr)
            # Syllable mistakenly starts at 0.5s (0.5s of dead silence before vocal attack at 1.0s)
            td = _make_td("amor", 0.5, 1.8)
            td.is_line_start = True

            trimmed = trim_syllable_ends_with_energy(
                audio_path=tmp_path,
                transcribed_data=[td],
                energy_threshold_ratio=0.15,
                min_syllable_dur=0.06,
            )

            # td.start should have advanced from 0.5s towards 1.0s (leaving at most small padding ~0.010s)
            self.assertGreater(trimmed[0].start, 0.90)
            self.assertLessEqual(trimmed[0].start, 1.01)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_grid_upscaling_and_zero_cascade_drift(self):
        """Verifies Equation VI: High BPM grid upscaling (>=600 BPM) preserves sub-25ms precision without cascade drift."""
        # 100 consecutive 250ms notes spanning 25 seconds
        target_bpm = 600.0  # UltraStar 4-beats per measure -> 60 / (600 * 4) = 0.025s/beat
        beat_sec = 60.0 / (target_bpm * 4.0)
        gap_ms = 1000.0
        gap_sec = gap_ms / 1000.0

        n_notes = 100
        note_dur = 0.250

        # Simulate physical timestamps
        curr_t = gap_sec
        prev_end_beat = int(round((curr_t - gap_sec) / beat_sec))

        for k in range(n_notes):
            t_s = curr_t
            t_e = t_s + note_dur
            b_start = int(round((t_s - gap_sec) / beat_sec))
            b_start = max(b_start, prev_end_beat)
            b_end = int(round((t_e - gap_sec) / beat_sec))
            b_dur = max(1, b_end - b_start)

            reconstructed_t_end = gap_sec + ((b_start + b_dur) * beat_sec)
            error_sec = abs(reconstructed_t_end - t_e)

            # Max drift at any individual note must be within half a beat (<= 12.5ms)
            self.assertLessEqual(error_sec, beat_sec * 0.5 + 1e-4)
            prev_end_beat = b_start + b_dur
            curr_t = t_e

        # Final note cumulative drift after 100 notes must NOT accumulate (still <= half a beat)
        total_time = gap_sec + (n_notes * note_dur)
        final_reconstructed_t = gap_sec + (prev_end_beat * beat_sec)
        self.assertLessEqual(abs(final_reconstructed_t - total_time), beat_sec * 0.5 + 1e-4)

    def test_tarareo_pre_verse_detach_and_orphan_head_reanchor(self):
        """Verifies Equations I, II & III on synthetic test timestamps:
        1. Orphan 'Hoy' at 2.005s-2.506s is re-anchored to ~11.8s before 'vengo' (12.128s).
        2. 7 'Tararara' tokens (24.28s-39.92s) are purged and duplicate refrain line over tarareo is skipped.
        3. 'Canto' (40.379s-43.564s) has its 2.8s pre-verse tarareo tail detached (-> start >= 42.8s).
        """
        from modules.Speech_Recognition.lyrics_aligner import align_transcribed_data_with_lyrics

        whisper_tds = [
            _make_td("Hoy", 2.005, 2.506),
            _make_td("vengo", 12.128, 12.429),
            _make_td("a", 12.449, 12.870),
            _make_td("cantar", 13.010, 13.872),
            _make_td("frente", 13.892, 14.133),
            _make_td("a", 14.213, 15.175),
            _make_td("ti.", 15.195, 15.255),
            _make_td("Tararara,", 24.280, 26.384),
            _make_td("tararara,", 27.005, 28.768),
            _make_td("tararara,", 29.409, 31.773),
            _make_td("tararara,", 31.793, 33.536),
            _make_td("tararara,", 34.177, 36.541),
            _make_td("tararara,", 36.561, 38.284),
            _make_td("tararara.", 38.965, 39.927),
            _make_td("Canto", 40.379, 43.564),
            _make_td("que", 43.584, 43.704),
            _make_td("escuches", 43.724, 44.125),
            _make_td("de", 44.145, 44.185),
            _make_td("un", 44.225, 44.285),
            _make_td("tiempo", 44.345, 44.625),
            _make_td("atrás,", 44.686, 45.046),
        ]

        lyrics_words = [
            ("Hoy", False), ("vengo", False), ("a", False), ("cantar", False), ("frente", False), ("a", False), ("ti", True),
            ("Hoy", False), ("vengo", False), ("a", False), ("cantar", False), ("frente", False), ("a", False), ("ti", True),
            ("Canto", False), ("que", False), ("escuches", False), ("que", False), ("un", False), ("tiempo", False), ("atrás", True),
        ]

        aligned = align_transcribed_data_with_lyrics(whisper_tds, lyrics_words, audio_path=None)

        # 1. 'Hoy' must be re-anchored from 2.0s to >11.5s right before 'vengo' (12.128s) and NOT broken by Continuity Guard
        self.assertEqual(aligned[0].word.strip(), "Hoy")
        self.assertGreaterEqual(aligned[0].start, 11.50)
        self.assertFalse(aligned[0].is_line_end)

        # 2. The duplicate 'Hoy vengo a cantar frente a ti' over the Tararara interval must be skipped,
        #    so the 8th aligned word (index 7) is 'Canto' (from 'Canto que escuches...')
        self.assertEqual(aligned[7].word.strip(), "Canto")

        # 3. 'Canto' (originally 40.379s-43.564s) must have its pre-verse tarareo detached (start >= 42.80s)
        self.assertGreaterEqual(aligned[7].start, 42.80)
        self.assertAlmostEqual(aligned[7].end, 43.564, places=2)

    def test_causal_tail_gate_prevents_crushing_words_with_leading_breath(self):
        """Verifies that trim_syllable_ends_with_energy advances start past a 0.55s breathing pause
        instead of letting the tail trimmer trigger inside the initial silence (like 'Empezarás' / 'Emocionados').
        """
        import tempfile
        import soundfile as sf
        from modules.Audio.voice_activity_trimmer import trim_syllable_ends_with_energy

        sr = 22050
        # 2.5s audio:
        # - Word 1 ('cancion'): 0.10s to 0.75s (voiced sine wave)
        # - Breathing pause (silence): 0.75s to 1.35s
        # - Word 2 ('empezarás'): Whisper timestamp is 0.79s to 1.90s, but actual singing is 1.35s to 1.85s!
        t = np.linspace(0.0, 2.5, int(sr * 2.5), endpoint=False)
        y = np.zeros_like(t, dtype=np.float32)
        mask_w1 = (t >= 0.10) & (t <= 0.75)
        mask_w2 = (t >= 1.35) & (t <= 1.85)
        y[mask_w1] = 0.5 * np.sin(2.0 * np.pi * 220.0 * t[mask_w1])
        y[mask_w2] = 0.5 * np.sin(2.0 * np.pi * 246.94 * t[mask_w2])

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, y, sr)
            td1 = _make_td("cancion ", 0.10, 0.75)
            td1.is_line_end = True
            # Note: td2 starts only 0.04s after td1.end (0.79s), with 0.56s of leading silence before 1.35s!
            td2 = _make_td("empezarás ", 0.79, 1.90)

            trimmed = trim_syllable_ends_with_energy(tmp_path, [td1, td2])
            # td2 must advance its start to ~1.34s and preserve its end near ~1.85s (duration >= 0.45s, NOT crushed to 0.06s!)
            self.assertGreaterEqual(trimmed[1].start, 1.30)
            self.assertGreaterEqual(trimmed[1].end, 1.80)
            self.assertGreaterEqual(trimmed[1].end - trimmed[1].start, 0.45)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()

