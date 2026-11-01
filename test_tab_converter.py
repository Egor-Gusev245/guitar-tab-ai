import unittest
from tab_converter import (
    NoteEvent,
    FretPosition,
    GuitarFretboard,
    TablatureSolver,
    AsciiTabFormatter,
    STANDARD_TUNING,
)


class TestTabConverter(unittest.TestCase):
    def setUp(self):
        self.fretboard = GuitarFretboard()
        self.solver = TablatureSolver(fretboard=self.fretboard)
        self.formatter = AsciiTabFormatter(max_line_width=60)

    def test_single_notes_candidates(self):
        # E2 (MIDI 40) is open 6th string
        cands_e2 = self.fretboard.get_candidates_for_pitch(40)
        self.assertTrue(any(c.string == 6 and c.fret == 0 for c in cands_e2))

        # E4 (MIDI 64) can be open 1st string, 5th fret 2nd string, etc.
        cands_e4 = self.fretboard.get_candidates_for_pitch(64)
        pos_set = {(c.string, c.fret) for c in cands_e4}
        self.assertIn((1, 0), pos_set)
        self.assertIn((2, 5), pos_set)
        self.assertIn((3, 9), pos_set)
        self.assertIn((4, 14), pos_set)

    def test_open_c_major_chord(self):
        # C Major chord in open position:
        # Notes: C3 (48), E3 (52), G3 (55), C4 (60), E4 (64)
        # Expected fingering:
        # String 5: fret 3 (C3)
        # String 4: fret 2 (E3)
        # String 3: fret 0 (G3)
        # String 2: fret 1 (C4)
        # String 1: fret 0 (E4)
        notes = [
            NoteEvent(pitch=48, start_time=0.0, end_time=1.0),
            NoteEvent(pitch=52, start_time=0.01, end_time=1.0),
            NoteEvent(pitch=55, start_time=0.02, end_time=1.0),
            NoteEvent(pitch=60, start_time=0.03, end_time=1.0),
            NoteEvent(pitch=64, start_time=0.04, end_time=1.0),
        ]

        solved = self.solver.solve(notes)
        self.assertEqual(len(solved), 1)

        slice_info, fingering = solved[0]
        # Check string assignments
        self.assertEqual(fingering[48], FretPosition(string=5, fret=3))
        self.assertEqual(fingering[52], FretPosition(string=4, fret=2))
        self.assertEqual(fingering[55], FretPosition(string=3, fret=0))
        self.assertEqual(fingering[60], FretPosition(string=2, fret=1))
        self.assertEqual(fingering[64], FretPosition(string=1, fret=0))

        tab_str = self.formatter.format(solved)
        print("\nGenerated C Major Tab:\n" + tab_str)
        self.assertIn("e|-0-|", tab_str)
        self.assertIn("B|-1-|", tab_str)
        self.assertIn("G|-0-|", tab_str)
        self.assertIn("D|-2-|", tab_str)
        self.assertIn("A|-3-|", tab_str)

    def test_sequential_notes_riff(self):
        # A simple riff (e.g. E minor pentatonic in open position)
        # E2 (40), G2 (43), A2 (45), B2 (47), D3 (50), E3 (52)
        pitches = [40, 43, 45, 47, 50, 52]
        notes = [
            NoteEvent(pitch=p, start_time=i * 0.5, end_time=(i + 1) * 0.5)
            for i, p in enumerate(pitches)
        ]

        solved = self.solver.solve(notes)
        self.assertEqual(len(solved), 6)

        tab_str = self.formatter.format(solved)
        print("\nGenerated Riff Tab:\n" + tab_str)
        self.assertTrue(len(tab_str) > 0)

    def test_multi_digit_frets_alignment(self):
        # High notes requiring two-digit frets:
        # E5 (76): String 1, fret 12
        # G5 (79): String 1, fret 15
        notes = [
            NoteEvent(pitch=76, start_time=0.0, end_time=0.5),
            NoteEvent(pitch=79, start_time=0.6, end_time=1.0),
        ]
        solved = self.solver.solve(notes)
        tab_str = self.formatter.format(solved)
        print("\nGenerated High Frets Tab:\n" + tab_str)

        # Ensure all lines in each block have the exact same character length
        for block in tab_str.strip().split("\n\n"):
            lines = [l for l in block.split("\r\n") if l.strip()] if "\r\n" in block else [l for l in block.split("\n") if l.strip()]
            if lines:
                first_len = len(lines[0])
                for line in lines:
                    self.assertEqual(len(line), first_len, f"Line length mismatch: {line}")

    def test_basic_pitch_integration(self):
        from tab_converter import basic_pitch_to_tab

        # Simulated Basic Pitch output: (start_s, end_s, pitch_midi, amplitude, pitch_bends)
        fake_bp_events = [
            (0.00, 0.40, 40.0, 0.85, []),     # Low E (E2)
            (0.45, 0.85, 45.0, 0.90, []),     # A2
            (0.90, 1.30, 50.0, 0.88, []),     # D3
            (0.95, 0.96, 70.0, 0.10, []),     # Artifact: very short + low confidence (should be filtered)
        ]

        tab = basic_pitch_to_tab(fake_bp_events, min_confidence=0.2, min_duration=0.05)
        print("\nGenerated from Basic Pitch Events:\n" + tab)
        self.assertIn("E|-0-", tab)
        self.assertIn("A|----0-", tab)
        self.assertIn("D|-------0-|", tab)


if __name__ == "__main__":
    unittest.main()
