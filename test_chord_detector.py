"""
Unit and Accuracy Tests for ChordDetector
Verifies detection accuracy on all 24 major/minor keys, realistic chord progressions,
silence handling, temporal segment querying, and execution speed.
"""

import unittest
import time
from pathlib import Path
import numpy as np

from chord_detector import (
    ChordDetector,
    ChordTimeline,
    PITCH_CLASSES,
)


def synthesize_note(freq: float, duration: float, sr: int = 22050) -> np.ndarray:
    """Synthesizes a musical note with natural harmonic overtones and envelope."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Fundamental + 2nd, 3rd, 4th harmonics
    harmonics = [1.0, 0.5, 0.3, 0.15]
    sig = np.zeros_like(t)
    for i, h in enumerate(harmonics, start=1):
        sig += h * np.sin(2 * np.pi * freq * i * t)

    # Attack-Decay-Sustain envelope
    attack_samples = int(0.05 * sr)
    env = np.ones_like(t)
    if attack_samples > 0:
        env[:attack_samples] = np.linspace(0, 1, attack_samples)
    sig *= env
    return sig


def midi_to_freq(midi_pitch: int) -> float:
    return 440.0 * (2.0 ** ((midi_pitch - 69.0) / 12.0))


def synthesize_chord(root_pc: int, is_minor: bool, duration: float = 1.0, octave: int = 4, sr: int = 22050) -> np.ndarray:
    """Synthesizes a standard triad chord."""
    base_midi = 12 * (octave + 1) + root_pc  # e.g. root_pc=0 (C), octave=4 -> 60 (C4)
    third_offset = 3 if is_minor else 4
    fifth_offset = 7

    r_note = synthesize_note(midi_to_freq(base_midi), duration, sr)
    t_note = synthesize_note(midi_to_freq(base_midi + third_offset), duration, sr)
    f_note = synthesize_note(midi_to_freq(base_midi + fifth_offset), duration, sr)

    chord_sig = r_note + t_note + f_note
    return chord_sig


class TestChordDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = ChordDetector(sr=22050)

    def test_all_12_major_chords(self):
        """Tests that each of the 12 Major chords is accurately identified."""
        correct = 0
        dur = 1.0
        sr = 22050

        print("\n--- Testing All 12 Major Chords ---")
        for pc_idx, name in enumerate(PITCH_CLASSES):
            audio = synthesize_chord(pc_idx, is_minor=False, duration=dur, sr=sr)
            audio /= np.max(np.abs(audio))
            timeline = self.detector.detect_from_audio(audio, sr=sr)

            # Query chord in the middle of duration
            seg = timeline.get_chord_at(0.5)
            self.assertIsNotNone(seg)
            is_match = (seg.chord == name)
            if is_match:
                correct += 1
            print(f"  Target: {name:<3} | Detected: {seg.chord:<3} ({int(seg.confidence*100)}%) | {'PASS' if is_match else 'FAIL'}")
            self.assertEqual(seg.chord, name, f"Failed for Major chord {name}")

        self.assertEqual(correct, 12)

    def test_all_12_minor_chords(self):
        """Tests that each of the 12 Minor chords is accurately identified."""
        correct = 0
        dur = 1.0
        sr = 22050

        print("\n--- Testing All 12 Minor Chords ---")
        for pc_idx, name in enumerate(PITCH_CLASSES):
            expected_name = name + "m"
            audio = synthesize_chord(pc_idx, is_minor=True, duration=dur, sr=sr)
            audio /= np.max(np.abs(audio))
            timeline = self.detector.detect_from_audio(audio, sr=sr)

            seg = timeline.get_chord_at(0.5)
            self.assertIsNotNone(seg)
            is_match = (seg.chord == expected_name)
            if is_match:
                correct += 1
            print(f"  Target: {expected_name:<4} | Detected: {seg.chord:<4} ({int(seg.confidence*100)}%) | {'PASS' if is_match else 'FAIL'}")
            self.assertEqual(seg.chord, expected_name, f"Failed for Minor chord {expected_name}")

        self.assertEqual(correct, 12)

    def test_classic_progression_c_g_am_f(self):
        """Tests sequential progression: C -> G -> Am -> F with transition boundaries."""
        sr = 22050
        dur_per_chord = 1.5

        # Progressions: C (0), G (7), Am (9 minor), F (5)
        c_chord = synthesize_chord(0, is_minor=False, duration=dur_per_chord, sr=sr)
        g_chord = synthesize_chord(7, is_minor=False, duration=dur_per_chord, sr=sr)
        am_chord = synthesize_chord(9, is_minor=True, duration=dur_per_chord, sr=sr)
        f_chord = synthesize_chord(5, is_minor=False, duration=dur_per_chord, sr=sr)

        full_audio = np.concatenate([c_chord, g_chord, am_chord, f_chord])
        full_audio /= np.max(np.abs(full_audio))

        t0 = time.time()
        timeline = self.detector.detect_from_audio(full_audio, sr=sr)
        detection_time = time.time() - t0

        print(f"\n--- Testing Progression C -> G -> Am -> F ({len(full_audio)/sr:.1f}s audio processed in {detection_time*1000:.1f}ms) ---")
        timeline.print_summary()

        # Check detected chord at midpoints
        seg_c = timeline.get_chord_at(0.75)
        seg_g = timeline.get_chord_at(2.25)
        seg_am = timeline.get_chord_at(3.75)
        seg_f = timeline.get_chord_at(5.25)

        self.assertEqual(seg_c.chord, "C")
        self.assertEqual(seg_g.chord, "G")
        self.assertEqual(seg_am.chord, "Am")
        self.assertEqual(seg_f.chord, "F")

    def test_silence_detection(self):
        """Tests that silence or zero energy is correctly classified as 'N' (No Chord)."""
        sr = 22050
        silence = np.zeros(int(sr * 1.5), dtype=np.float32)
        timeline = self.detector.detect_from_audio(silence, sr=sr)

        seg = timeline.get_chord_at(0.5)
        self.assertIsNotNone(seg)
        self.assertEqual(seg.chord, "N")

    def test_json_serialization(self):
        """Verifies JSON output matches API requirements."""
        sr = 22050
        chord = synthesize_chord(0, is_minor=False, duration=1.0, sr=sr)
        timeline = self.detector.detect_from_audio(chord, sr=sr)

        json_str = timeline.to_json()
        self.assertIn('"chord": "C"', json_str)
        self.assertIn('"duration"', json_str)
        self.assertIn('"segments"', json_str)


if __name__ == "__main__":
    unittest.main()
