"""
Direct end-to-end test script.
"""

import time
import numpy as np
import soundfile as sf
import soxr

import basic_pitch.inference as bpi
import basic_pitch.note_creation as infer
from basic_pitch.constants import AUDIO_SAMPLE_RATE, FFT_HOP

from tab_converter import NoteEvent, notes_to_tab


def fast_get_audio_input(audio_path, overlap_len, hop_size):
    data, sr = sf.read(str(audio_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != AUDIO_SAMPLE_RATE:
        data = soxr.resample(data, sr, AUDIO_SAMPLE_RATE)
    padded = np.pad(data, (overlap_len // 2, 0))
    for window, window_time in bpi.window_audio_file(padded, hop_size):
        yield np.expand_dims(window, axis=0), window_time, len(data)


def run():
    audio_path = "C:/Users/egorg/.gemini/antigravity/scratch/guitar_tabs_mvp/sample_riff.wav"

    t0 = time.time()
    print(">>> 1. Loading audio and running Basic Pitch ONNX inference...")
    bpi.get_audio_input = fast_get_audio_input
    model_output = bpi.run_inference(audio_path, bpi.ICASSP_2022_MODEL_PATH)
    t_inf = time.time() - t0
    print(f"    Inference completed in {t_inf:.3f} s")

    t1 = time.time()
    print(">>> 2. Extracting notes...")
    min_note_len_frames = 11
    raw_notes = infer.output_to_notes_polyphonic(
        model_output["note"],
        model_output["onset"],
        onset_thresh=0.5,
        frame_thresh=0.3,
        min_note_len=min_note_len_frames,
        infer_onsets=True,
        max_freq=None,
        min_freq=None,
    )

    notes = []
    for start_frame, end_frame, pitch_idx, amplitude in raw_notes:
        start_s = float(start_frame * FFT_HOP / AUDIO_SAMPLE_RATE)
        end_s = float(end_frame * FFT_HOP / AUDIO_SAMPLE_RATE)
        midi_pitch = int(pitch_idx + 21)

        # Filter guitar range: E2 (40) to E6 (88)
        if not (40 <= midi_pitch <= 88):
            continue
        notes.append(
            NoteEvent(
                pitch=midi_pitch,
                start_time=start_s,
                end_time=end_s,
                velocity=float(amplitude),
            )
        )

    notes.sort(key=lambda n: (n.start_time, n.pitch))
    t_notes = time.time() - t1
    print(f"    Extracted {len(notes)} notes in {t_notes:.3f} s:")
    for n in notes:
        print(f"      - Note MIDI {n.pitch} at {n.start_time:.2f}s - {n.end_time:.2f}s (confidence: {n.velocity:.2f})")

    t2 = time.time()
    print(">>> 3. Solving guitar fretboard fingering (Standard Tuning EADGBE)...")
    ascii_tab = notes_to_tab(notes)
    t_tab = time.time() - t2
    print(f"    Fretboard solver completed in {t_tab:.3f} s")

    total_time = time.time() - t0
    print("\n" + "=" * 40)
    print(f"TEST RESULT (Total time: {total_time:.3f} s)")
    print("=" * 40)
    print(ascii_tab)
    print("=" * 40)


if __name__ == "__main__":
    run()
