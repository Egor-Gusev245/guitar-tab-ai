"""
Guitar Tablature Converter Module
Converts MIDI note events (pitch, start_time, end_time) into guitar tablature in standard tuning.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
import math


@dataclass
class NoteEvent:
    pitch: int          # MIDI note number (e.g. 40 = E2, 60 = C4)
    start_time: float   # Start time in seconds
    end_time: float     # End time in seconds
    velocity: float = 1.0  # Normalized velocity or confidence (0.0 to 1.0)


@dataclass(frozen=True)
class FretPosition:
    string: int  # 1 to 6 (1 = high e, 6 = low E)
    fret: int    # 0 to max_fret (0 = open string)


# Standard Guitar Tuning: String 1 (high E) to String 6 (low E)
# String 1: E4 (64)
# String 2: B3 (59)
# String 3: G3 (55)
# String 4: D3 (50)
# String 5: A2 (45)
# String 6: E2 (40)
STANDARD_TUNING: Dict[int, int] = {
    1: 64,  # e4
    2: 59,  # B3
    3: 55,  # G3
    4: 50,  # D3
    5: 45,  # A2
    6: 40,  # E2
}

STRING_NAMES = {
    1: "e",
    2: "B",
    3: "G",
    4: "D",
    5: "A",
    6: "E",
}


class GuitarFretboard:
    def __init__(
        self,
        tuning: Optional[Dict[int, int]] = None,
        max_fret: int = 22,
        max_hand_span: int = 4,
    ):
        self.tuning = tuning or STANDARD_TUNING
        self.max_fret = max_fret
        self.max_hand_span = max_hand_span

    def get_candidates_for_pitch(self, pitch: int) -> List[FretPosition]:
        """Find all possible (string, fret) positions for a single MIDI pitch."""
        candidates = []
        for string_idx, open_pitch in self.tuning.items():
            fret = pitch - open_pitch
            if 0 <= fret <= self.max_fret:
                candidates.append(FretPosition(string=string_idx, fret=fret))
        return candidates


@dataclass
class ChordSlice:
    start_time: float
    end_time: float
    notes: List[NoteEvent]


class TablatureSolver:
    def __init__(
        self,
        fretboard: Optional[GuitarFretboard] = None,
        onset_tolerance: float = 0.045,  # 45ms window to treat notes as a chord/slice
    ):
        self.fretboard = fretboard or GuitarFretboard()
        self.onset_tolerance = onset_tolerance

    def cluster_into_slices(self, notes: List[NoteEvent]) -> List[ChordSlice]:
        """Group notes that start at approximately the same time into chord slices."""
        if not notes:
            return []

        sorted_notes = sorted(notes, key=lambda n: (n.start_time, n.pitch))
        slices: List[ChordSlice] = []
        
        current_cluster: List[NoteEvent] = [sorted_notes[0]]
        cluster_start = sorted_notes[0].start_time

        for note in sorted_notes[1:]:
            if note.start_time - cluster_start <= self.onset_tolerance:
                current_cluster.append(note)
            else:
                max_end = max(n.end_time for n in current_cluster)
                slices.append(ChordSlice(start_time=cluster_start, end_time=max_end, notes=current_cluster))
                current_cluster = [note]
                cluster_start = note.start_time

        if current_cluster:
            max_end = max(n.end_time for n in current_cluster)
            slices.append(ChordSlice(start_time=cluster_start, end_time=max_end, notes=current_cluster))

        return slices

    def _get_chord_fingerings(self, pitches: List[int]) -> List[Dict[int, FretPosition]]:
        """
        Generate ergonomically valid fingering assignments for a set of simultaneous pitches.
        Returns a list of dicts mapping pitch -> FretPosition.
        """
        unique_pitches = sorted(list(set(pitches)))
        if not unique_pitches:
            return []

        # Find candidates for each pitch
        pitch_candidates = [
            (p, self.fretboard.get_candidates_for_pitch(p))
            for p in unique_pitches
        ]

        # If any pitch cannot be played on the guitar in standard tuning, ignore or drop it
        pitch_candidates = [(p, cands) for p, cands in pitch_candidates if cands]
        if not pitch_candidates:
            return []

        valid_fingerings: List[Dict[int, FretPosition]] = []

        def backtrack(index: int, current_assignment: Dict[int, FretPosition], used_strings: Set[int]):
            if index == len(pitch_candidates):
                valid_fingerings.append(current_assignment.copy())
                return

            pitch, candidates = pitch_candidates[index]
            for cand in candidates:
                if cand.string in used_strings:
                    continue  # A string cannot play two notes at once

                # Ergonomic check: fret span check for fretted notes (fret > 0)
                current_frets = [pos.fret for pos in current_assignment.values() if pos.fret > 0]
                if cand.fret > 0:
                    current_frets.append(cand.fret)

                if current_frets:
                    span = max(current_frets) - min(current_frets)
                    if span > self.fretboard.max_hand_span:
                        continue  # Stretch is too wide for comfortable hand position

                # Branch
                used_strings.add(cand.string)
                current_assignment[pitch] = cand
                backtrack(index + 1, current_assignment, used_strings)
                # Unbranch
                used_strings.remove(cand.string)
                del current_assignment[pitch]

        backtrack(0, {}, set())
        return valid_fingerings

    @staticmethod
    def _fingering_position(fingering: Dict[int, FretPosition]) -> Optional[float]:
        """Compute the average fretted position for a fingering."""
        frets = [pos.fret for pos in fingering.values() if pos.fret > 0]
        if not frets:
            return None
        return sum(frets) / len(frets)

    def _static_cost(self, fingering: Dict[int, FretPosition]) -> float:
        """
        Cost of a single chord fingering in isolation:
        - Penalizes frets far up the neck (prefer lower/natural guitar frets 0-10).
        - Penalizes wide fret span.
        """
        frets = [pos.fret for pos in fingering.values() if pos.fret > 0]
        cost = 0.0

        if frets:
            span = max(frets) - min(frets)
            cost += span * 2.0  # Compact chords are easier

            # Preference for natural frets 0-10; progressive penalty for high frets
            avg_fret = sum(frets) / len(frets)
            if avg_fret > 10:
                cost += (avg_fret - 10) * 2.5
            if avg_fret > 14:
                cost += (avg_fret - 14) * 5.0
            else:
                cost += avg_fret * 0.08  # Slight preference for lower frets

        return cost

    def _transition_cost(
        self,
        prev_fingering: Dict[int, FretPosition],
        curr_fingering: Dict[int, FretPosition],
    ) -> float:
        """Cost of moving hand and strings from previous chord/note to current one."""
        # Open strings require no hand positioning
        if all(p.fret == 0 for p in curr_fingering.values()):
            return 0.0

        prev_pos = self._fingering_position(prev_fingering)
        curr_pos = self._fingering_position(curr_fingering)

        cost = 0.0
        if prev_pos is not None and curr_pos is not None:
            # Hand movement along the fretboard
            fret_distance = abs(curr_pos - prev_pos)
            cost += fret_distance * 1.5

        # Penalize unnecessary wide string jumps to maintain single-string / adjacent melody flow
        prev_strs = [p.string for p in prev_fingering.values()]
        curr_strs = [p.string for p in curr_fingering.values()]
        if prev_strs and curr_strs:
            avg_prev_str = sum(prev_strs) / len(prev_strs)
            avg_curr_str = sum(curr_strs) / len(curr_strs)
            cost += abs(avg_curr_str - avg_prev_str) * 0.4

        return cost

    def solve(self, notes: List[NoteEvent]) -> List[Tuple[ChordSlice, Dict[int, FretPosition]]]:
        """
        Finds optimal string/fret assignment across the sequence of notes using Viterbi DP.
        Returns a list of pairs: (ChordSlice, mapping of pitch -> FretPosition).
        """
        slices = self.cluster_into_slices(notes)
        if not slices:
            return []

        # Generate candidate fingerings for each slice
        slice_candidates: List[List[Dict[int, FretPosition]]] = []
        for sl in slices:
            pitches = [n.pitch for n in sl.notes]
            cands = self._get_chord_fingerings(pitches)
            if not cands:
                # If no valid fingering was found for full chord (e.g. impossible polyphony),
                # fallback to single note assignments
                fallback_cands = []
                for p in pitches:
                    p_cands = self.fretboard.get_candidates_for_pitch(p)
                    if p_cands:
                        fallback_cands.append({p: p_cands[0]})
                cands = fallback_cands if fallback_cands else [{}]
            slice_candidates.append(cands)

        # Dynamic Programming / Viterbi
        # dp[t][i] = minimum cost to reach state i at slice t
        num_slices = len(slices)
        dp = [{} for _ in range(num_slices)]
        backtrack = [{} for _ in range(num_slices)]

        # Initialize t = 0
        for i, cand in enumerate(slice_candidates[0]):
            dp[0][i] = self._static_cost(cand)
            backtrack[0][i] = None

        # Step forward t = 1 .. T-1
        for t in range(1, num_slices):
            curr_cands = slice_candidates[t]
            prev_cands = slice_candidates[t - 1]

            for j, curr_cand in enumerate(curr_cands):
                static_cost = self._static_cost(curr_cand)
                min_cost = float("inf")
                best_prev = None

                for i, prev_cand in enumerate(prev_cands):
                    prev_cost = dp[t - 1][i]
                    trans_cost = self._transition_cost(prev_cand, curr_cand)
                    total_cost = prev_cost + trans_cost + static_cost

                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_prev = i

                dp[t][j] = min_cost
                backtrack[t][j] = best_prev

        # Backtrack best path
        best_last_idx = min(dp[num_slices - 1], key=dp[num_slices - 1].get)
        path = [best_last_idx]

        for t in range(num_slices - 1, 0, -1):
            prev_idx = backtrack[t][path[-1]]
            path.append(prev_idx)

        path.reverse()

        result = []
        for t, idx in enumerate(path):
            result.append((slices[t], slice_candidates[t][idx]))

        return result


class AsciiTabFormatter:
    """Renders solved guitar positions into readable ASCII tablature."""

    def __init__(self, max_line_width: int = 70):
        self.max_line_width = max_line_width

    def format(
        self,
        solved_slices: List[Tuple[ChordSlice, Dict[int, FretPosition]]],
    ) -> str:
        """
        Formats list of (ChordSlice, pitch_to_fret_pos) into multi-line ASCII tablature.
        """
        if not solved_slices:
            return "No notes to display."

        lines = {s: [] for s in range(1, 7)}
        line_lengths = 0
        output_blocks = []

        def init_block():
            return {s: [f"{STRING_NAMES[s]}|"] for s in range(1, 7)}

        current_block = init_block()
        current_block_len = 2  # e.g. "e|" is 2 characters

        for sl, fingering in solved_slices:
            # Map string -> fret string (e.g. "0", "12", "-")
            string_frets: Dict[int, Optional[str]] = {s: None for s in range(1, 7)}
            for pos in fingering.values():
                string_frets[pos.string] = str(pos.fret)

            # Determine column width needed (2 chars if fret >= 10, else 1 char, plus spacing)
            max_digits = max(
                (len(f) for f in string_frets.values() if f is not None),
                default=1,
            )
            col_width = max(max_digits, 1)

            # Check if adding this column exceeds max_line_width
            # We add col_width + 1 dashes
            col_addition_len = col_width + 2

            if current_block_len + col_addition_len > self.max_line_width:
                # Close current block and start a new one
                for s in range(1, 7):
                    current_block[s].append("|\n")
                output_blocks.append(
                    "".join("".join(current_block[s]) for s in range(1, 7))
                )
                current_block = init_block()
                current_block_len = 2

            # Append the fret numbers or dashes
            for s in range(1, 7):
                fret_str = string_frets[s]
                if fret_str is not None:
                    # Pad fret string to match col_width with trailing dash
                    padded = fret_str.ljust(col_width, "-")
                    current_block[s].append(f"-{padded}-")
                else:
                    # Fill with dashes
                    current_block[s].append("-" * (col_width + 2))

            current_block_len += col_addition_len

        # Close the final block
        for s in range(1, 7):
            current_block[s].append("|\n")
        output_blocks.append(
            "".join("".join(current_block[s]) for s in range(1, 7))
        )

        return "\n".join(output_blocks)


def from_basic_pitch_events(
    events: List[Tuple],
    min_confidence: float = 0.2,
    min_duration: float = 0.05,
) -> List[NoteEvent]:
    """
    Converts raw note events from Spotify Basic Pitch into NoteEvent objects.
    Each event in basic-pitch is typically: (start_s, end_s, pitch_midi, amplitude, [pitch_bends]).
    Filters out noise/ghost notes based on duration and confidence.
    """
    notes = []
    for ev in events:
        start_s = ev[0]
        end_s = ev[1]
        pitch = int(round(ev[2]))
        amp = ev[3] if len(ev) > 3 else 1.0

        # Filter artifacts
        if end_s - start_s < min_duration:
            continue
        if amp < min_confidence:
            continue

        notes.append(NoteEvent(
            pitch=pitch,
            start_time=start_s,
            end_time=end_s,
            velocity=float(amp),
        ))
    return notes


def notes_to_tab(
    notes: List[NoteEvent],
    max_line_width: int = 70,
    onset_tolerance: float = 0.045,
    max_hand_span: int = 4,
) -> str:
    """
    High-level convenience function:
    Takes a list of NoteEvent objects and returns an ASCII tablature string.
    """
    fretboard = GuitarFretboard(max_hand_span=max_hand_span)
    solver = TablatureSolver(fretboard=fretboard, onset_tolerance=onset_tolerance)
    solved = solver.solve(notes)
    formatter = AsciiTabFormatter(max_line_width=max_line_width)
    return formatter.format(solved)


def basic_pitch_to_tab(
    events: List[Tuple],
    min_confidence: float = 0.2,
    min_duration: float = 0.05,
    max_line_width: int = 70,
) -> str:
    """
    End-to-end convenience function:
    Takes raw Spotify Basic Pitch events and outputs ASCII tablature.
    """
    notes = from_basic_pitch_events(
        events,
        min_confidence=min_confidence,
        min_duration=min_duration,
    )
    return notes_to_tab(notes, max_line_width=max_line_width)

