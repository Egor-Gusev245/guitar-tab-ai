"""
Generates a realistic multi-chord demo song for the Web UI.
Progression: C Major -> G Major -> A Minor -> F Major (3 seconds each = 12 seconds).
"""

import numpy as np
import soundfile as sf
from pathlib import Path


def synthesize_strum(pitches, duration=3.0, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = np.zeros_like(t)

    # 4 strums per chord (e.g. at 0.0s, 0.75s, 1.5s, 2.25s)
    strum_times = [0.0, 0.75, 1.5, 2.25]
    for st_time in strum_times:
        st_idx = int(st_time * sr)
        rem_dur = duration - st_time
        t_sub = np.linspace(0, rem_dur, int(sr * rem_dur), endpoint=False)

        for i, p in enumerate(pitches):
            freq = 440.0 * (2.0 ** ((p - 69.0) / 12.0))
            decay = np.exp(-t_sub * (1.8 + 0.1 * i))
            pluck = (
                np.sin(2 * np.pi * freq * t_sub)
                + 0.6 * np.sin(2 * np.pi * 2 * freq * t_sub)
                + 0.3 * np.sin(2 * np.pi * 3 * freq * t_sub)
                + 0.15 * np.sin(2 * np.pi * 4 * freq * t_sub)
            ) * decay

            # Strum delay per string
            string_delay = int(i * 0.015 * sr)
            if string_delay < len(t_sub):
                padded = np.pad(pluck[:len(t_sub) - string_delay], (string_delay, 0))
                sig[st_idx : st_idx + len(padded)] += padded

    return sig


def build_demo_song():
    sr = 22050
    # Standard guitar voicings:
    # C: C3(48), E3(52), G3(55), C4(60), E4(64)
    # G: G2(43), B2(47), D3(50), G3(55), B3(59), G4(67)
    # Am: A2(45), E3(52), A3(57), C4(60), E4(64)
    # F: F2(41), C3(48), F3(53), A3(57), C4(60), F4(65)
    voicings = [
        [48, 52, 55, 60, 64],       # C
        [43, 47, 50, 55, 59, 67],   # G
        [45, 52, 57, 60, 64],       # Am
        [41, 48, 53, 57, 60, 65],   # F
    ]

    parts = [synthesize_strum(v, duration=3.0, sr=sr) for v in voicings]
    full_audio = np.concatenate(parts)
    full_audio = full_audio / np.max(np.abs(full_audio)) * 0.9

    out_path = Path(__file__).resolve().parent / "sample_song.wav"
    sf.write(str(out_path), full_audio, sr)
    print(f"Created demo song: {out_path} ({len(full_audio)/sr:.1f}s)")


if __name__ == "__main__":
    build_demo_song()
