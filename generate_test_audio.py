"""
Generates a short synthetic audio file for testing the pipeline end-to-end.
"""

import numpy as np
import soundfile as sf
from pathlib import Path


def generate_pluck(freq: float, duration: float, sr: int = 44100) -> np.ndarray:
    """Simple Karplus-Strong / additive synthesis of a plucked guitar string."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Add fundamental + harmonics
    harmonics = [1.0, 0.6, 0.4, 0.2, 0.1]
    signal = np.zeros_like(t)
    for i, h in enumerate(harmonics, start=1):
        decay = np.exp(-t * (3.0 * i))
        signal += h * np.sin(2 * np.pi * freq * i * t) * decay

    # Normalize
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal)) * 0.8
    return signal


def create_sample_riff(output_path: Path):
    sr = 44100
    # Riff: E2 (82.4 Hz), G2 (98 Hz), A2 (110 Hz), D3 (146.8 Hz)
    notes = [
        (82.41, 0.6),
        (97.99, 0.6),
        (110.0, 0.6),
        (146.83, 0.8),
    ]

    riff_data = []
    for freq, dur in notes:
        pluck = generate_pluck(freq, dur, sr)
        riff_data.append(pluck)

    audio = np.concatenate(riff_data)
    # Stereo
    stereo_audio = np.column_stack((audio, audio))
    sf.write(str(output_path), stereo_audio, sr)
    print(f"Created sample guitar riff: {output_path} ({len(audio)/sr:.2f}s)")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "sample_riff.wav"
    create_sample_riff(out)
