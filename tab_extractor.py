"""
Tab Extractor Module
Fast, robust transcription using Spotify Basic Pitch ONNX model and guitar tab solver.
"""

from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
import soundfile as sf
import soxr
import pretty_midi

import basic_pitch.inference as bpi
import basic_pitch.note_creation as infer
from basic_pitch.constants import (
    AUDIO_SAMPLE_RATE,
    AUDIO_N_SAMPLES,
    FFT_HOP,
)

from tab_converter import (
    NoteEvent,
    notes_to_tab,
)


@dataclass
class TranscriptionResult:
    audio_path: Path
    ascii_tab: str
    notes: List[NoteEvent]
    midi_data: Optional[pretty_midi.PrettyMIDI]
    raw_events: List[Tuple]

    def save_tab(self, output_path: Path) -> None:
        """Saves ASCII tablature to a text file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.ascii_tab, encoding="utf-8")
        print(f"[Tab] Saved tablature to {output_path}")

    def save_midi(self, output_path: Path) -> None:
        """Saves MIDI transcription to a .mid file."""
        if self.midi_data is None:
            # Build pretty_midi object from notes
            pm = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(program=25)  # Acoustic guitar
            for n in self.notes:
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=int(n.velocity * 100),
                        pitch=n.pitch,
                        start=n.start_time,
                        end=n.end_time,
                    )
                )
            pm.instruments.append(inst)
            self.midi_data = pm

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.midi_data.write(str(output_path))
        print(f"[MIDI] Saved MIDI to {output_path}")


def _fast_windowed_audio(audio_path: Path, overlap_len: int, hop_size: int):
    """Fast audio loader without librosa dependency."""
    data, sr = sf.read(str(audio_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # stereo to mono
    if sr != AUDIO_SAMPLE_RATE:
        data = soxr.resample(data, sr, AUDIO_SAMPLE_RATE)

    original_length = len(data)
    padded_data = np.pad(data, (overlap_len // 2, 0), mode="constant")

    for window, window_time in bpi.window_audio_file(padded_data, hop_size):
        yield np.expand_dims(window, axis=0), window_time, original_length


def clean_guitar_notes(
    notes: List[NoteEvent],
    merge_gap: float = 0.12,
    deoctave: bool = True,
) -> List[NoteEvent]:
    """
    Cleans and consolidates raw polyphonic note detections:
    1. Merges consecutive duplicate notes of identical pitch separated by small gaps (< merge_gap).
    2. De-octave filter: suppresses acoustic upper octave harmonics (+12 / +24 semitones)
       commonly produced by electric guitar pickups, 12-string guitars, or distortion/fuzz.
    """
    if not notes:
        return []

    # Step 1: Merge consecutive notes of identical pitch
    notes_by_pitch = sorted(notes, key=lambda n: (n.pitch, n.start_time))
    merged: List[NoteEvent] = []

    for n in notes_by_pitch:
        if not merged:
            merged.append(NoteEvent(pitch=n.pitch, start_time=n.start_time, end_time=n.end_time, velocity=n.velocity))
            continue
        prev = merged[-1]
        if prev.pitch == n.pitch and (n.start_time - prev.end_time) <= merge_gap:
            # Extend sustain of the existing note
            prev.end_time = max(prev.end_time, n.end_time)
            prev.velocity = max(prev.velocity, n.velocity)
        else:
            merged.append(NoteEvent(pitch=n.pitch, start_time=n.start_time, end_time=n.end_time, velocity=n.velocity))

    if not deoctave:
        merged.sort(key=lambda n: (n.start_time, n.pitch))
        return merged

    # Step 2: Harmonic / Octave overtone suppression
    filtered: List[NoteEvent] = []
    for b in merged:
        is_harmonic = False
        for a in merged:
            if a is b:
                continue
            # Check if b is 1 or 2 octaves above fundamental a
            if (b.pitch - a.pitch) in (12, 24):
                overlap = min(a.end_time, b.end_time) - max(a.start_time, b.start_time)
                b_dur = b.end_time - b.start_time
                if b_dur > 0 and overlap > 0.04 and (overlap / b_dur) > 0.30:
                    is_harmonic = True
                    break
        if not is_harmonic:
            filtered.append(b)

    filtered.sort(key=lambda n: (n.start_time, n.pitch))
    return filtered


class GuitarTabExtractor:
    def __init__(
        self,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        min_note_duration: float = 0.08,
        min_confidence: float = 0.42,
        max_line_width: int = 70,
        max_hand_span: int = 4,
    ):
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.min_note_duration = min_note_duration
        self.min_confidence = min_confidence
        self.max_line_width = max_line_width
        self.max_hand_span = max_hand_span

    def transcribe(self, guitar_audio_path: Path) -> TranscriptionResult:
        """
        Runs Spotify Basic Pitch ONNX inference and generates guitar tablature.
        """
        guitar_audio_path = Path(guitar_audio_path).resolve()
        if not guitar_audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {guitar_audio_path}")

        # Monkey-patch audio reader for high performance
        bpi.get_audio_input = _fast_windowed_audio

        # Step 1: Run ONNX neural net inference
        model_output = bpi.run_inference(
            guitar_audio_path,
            bpi.ICASSP_2022_MODEL_PATH,
        )

        # Step 2: Polyphonic note extraction
        min_note_len_frames = max(1, int(round(self.min_note_duration * AUDIO_SAMPLE_RATE / FFT_HOP)))
        raw_notes = infer.output_to_notes_polyphonic(
            model_output["note"],
            model_output["onset"],
            onset_thresh=self.onset_threshold,
            frame_thresh=self.frame_threshold,
            min_note_len=min_note_len_frames,
            infer_onsets=True,
            max_freq=None,
            min_freq=None,
        )

        # Step 3: Convert frame indices to seconds and MIDI pitches
        # Note: output_to_notes_polyphonic already returns absolute MIDI pitch (freq_idx + MIDI_OFFSET)
        notes: List[NoteEvent] = []
        raw_events: List[Tuple] = []

        for start_frame, end_frame, pitch_val, amplitude in raw_notes:
            start_s = float(start_frame * FFT_HOP / AUDIO_SAMPLE_RATE)
            end_s = float(end_frame * FFT_HOP / AUDIO_SAMPLE_RATE)
            midi_pitch = int(pitch_val)

            # Filter guitar range: E2 (40) to E6 (88)
            if not (40 <= midi_pitch <= 88):
                continue

            if amplitude < self.min_confidence or (end_s - start_s) < self.min_note_duration:
                continue

            notes.append(
                NoteEvent(
                    pitch=midi_pitch,
                    start_time=start_s,
                    end_time=end_s,
                    velocity=float(amplitude),
                )
            )
            raw_events.append((start_s, end_s, midi_pitch, float(amplitude), None))

        # Step 4: Clean, de-bounce and de-octave notes
        cleaned_notes = clean_guitar_notes(notes, merge_gap=0.12, deoctave=True)

        # Step 5: Solve guitar fretboard positions & render ASCII tab
        ascii_tab = notes_to_tab(
            cleaned_notes,
            max_line_width=self.max_line_width,
            max_hand_span=self.max_hand_span,
        )

        return TranscriptionResult(
            audio_path=guitar_audio_path,
            ascii_tab=ascii_tab,
            notes=cleaned_notes,
            midi_data=None,
            raw_events=raw_events,
        )
