"""
Advanced Stress and Real-World Scenario Tests for ChordDetector
Tests:
1. Real guitar-like voicings with multiple octaves (Open chords: E, A, D, G, C, Em, Am, Dm).
2. Noise robustness (SNR test with percussive/broadband noise).
3. Benchmark: processing time for a full 3-minute audio track.
"""

import time
import numpy as np
from chord_detector import ChordDetector, PITCH_CLASSES


def synthesize_guitar_voicing(string_pitches, duration=1.5, sr=22050):
    """
    Synthesizes realistic guitar chord voicings across 6 strings.
    string_pitches: list of MIDI pitches for sounding strings.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = np.zeros_like(t)

    # Strum delay between strings (~25ms arpeggiation)
    strum_delay = 0.02
    for i, p in enumerate(string_pitches):
        offset = int(i * strum_delay * sr)
        freq = 440.0 * (2.0 ** ((p - 69.0) / 12.0))
        # Pluck decay
        decay = np.exp(-t * (1.5 + 0.1 * i))
        pluck = (
            np.sin(2 * np.pi * freq * t)
            + 0.5 * np.sin(2 * np.pi * 2 * freq * t)
            + 0.3 * np.sin(2 * np.pi * 3 * freq * t)
            + 0.15 * np.sin(2 * np.pi * 4 * freq * t)
        ) * decay
        if offset > 0:
            pluck = np.pad(pluck[:-offset], (offset, 0))
        sig += pluck

    return sig / np.max(np.abs(sig))


def run_advanced_tests():
    detector = ChordDetector(sr=22050)
    sr = 22050

    print("=" * 65)
    print("ADVANCED CHORD RECOGNITION TESTS: REAL GUITAR VOICINGS")
    print("=" * 65)

    # Standard open guitar chord fingerings (MIDI numbers)
    # E open: E2(40), B2(47), E3(52), G#3(56), B3(59), E4(64)
    # Em open: E2(40), B2(47), E3(52), G3(55), B3(59), E4(64)
    # A open: A2(45), E3(52), A3(57), C#4(61), E4(64)
    # Am open: A2(45), E3(52), A3(57), C4(60), E4(64)
    # C open: C3(48), E3(52), G3(55), C4(60), E4(64)
    # G open: G2(43), B2(47), D3(50), G3(55), B3(59), G4(67)
    # D open: D3(50), A3(57), D4(62), F#4(66)
    # Dm open: D3(50), A3(57), D4(62), F4(65)

    guitar_voicings = {
        "E": [40, 47, 52, 56, 59, 64],
        "Em": [40, 47, 52, 55, 59, 64],
        "A": [45, 52, 57, 61, 64],
        "Am": [45, 52, 57, 60, 64],
        "C": [48, 52, 55, 60, 64],
        "G": [43, 47, 50, 55, 59, 67],
        "D": [50, 57, 62, 66],
        "Dm": [50, 57, 62, 65],
    }

    correct = 0
    for target_chord, pitches in guitar_voicings.items():
        audio = synthesize_guitar_voicing(pitches, duration=1.5, sr=sr)
        tl = detector.detect_from_audio(audio, sr=sr)
        detected = tl.get_chord_at(0.75).chord
        conf = tl.get_chord_at(0.75).confidence
        passed = (detected == target_chord)
        if passed:
            correct += 1
        status = "PASS" if passed else "FAIL"
        print(f"  Open Voicing: {target_chord:<3} | Detected: {detected:<3} ({int(conf*100)}%) | {status}")

    print(f"\nGuitar Voicings Accuracy: {correct}/{len(guitar_voicings)} ({correct/len(guitar_voicings)*100:.1f}%)")

    # Test 2: Noise Robustness (SNR = 12dB with added random noise)
    print("\n" + "=" * 65)
    print("NOISE ROBUSTNESS TEST (Adding broadband background noise)")
    print("=" * 65)
    noise_correct = 0
    np.random.seed(42)
    for target_chord, pitches in guitar_voicings.items():
        clean = synthesize_guitar_voicing(pitches, duration=1.5, sr=sr)
        noise = np.random.normal(0, 0.15, len(clean)).astype(np.float32)  # noise ~ 15% amplitude
        noisy_audio = clean + noise
        noisy_audio /= np.max(np.abs(noisy_audio))

        tl = detector.detect_from_audio(noisy_audio, sr=sr)
        detected = tl.get_chord_at(0.75).chord
        conf = tl.get_chord_at(0.75).confidence
        passed = (detected == target_chord)
        if passed:
            noise_correct += 1
        print(f"  Noisy {target_chord:<3} | Detected: {detected:<3} ({int(conf*100)}%) | {'PASS' if passed else 'FAIL'}")

    print(f"\nNoisy Audio Accuracy: {noise_correct}/{len(guitar_voicings)} ({noise_correct/len(guitar_voicings)*100:.1f}%)")

    # Test 3: Benchmark on full 3-minute song duration
    print("\n" + "=" * 65)
    print("PERFORMANCE BENCHMARK (Simulating 3-minute song = 180 seconds)")
    print("=" * 65)
    # Generate 180s repeating progression: Em - C - G - D (each 2 seconds)
    prog_chords = [
        guitar_voicings["Em"],
        guitar_voicings["C"],
        guitar_voicings["G"],
        guitar_voicings["D"],
    ]
    chunks = []
    for _ in range(180 // 8):
        for pitches in prog_chords:
            chunks.append(synthesize_guitar_voicing(pitches, duration=2.0, sr=sr))

    song_audio = np.concatenate(chunks)
    song_dur = len(song_audio) / sr

    t0 = time.time()
    song_timeline = detector.detect_from_audio(song_audio, sr=sr)
    benchmark_time = time.time() - t0

    rt_factor = song_dur / benchmark_time
    print(f"  Total audio length : {song_dur:.1f} seconds (3.0 minutes)")
    print(f"  Processing time    : {benchmark_time:.2f} seconds")
    print(f"  Speedup factor     : {rt_factor:.1f}x real-time! (Takes ~{benchmark_time*1000/3:.0f}ms per minute of music)")
    print(f"  Segments detected  : {len(song_timeline.segments)}")
    print("=" * 65)


if __name__ == "__main__":
    run_advanced_tests()
