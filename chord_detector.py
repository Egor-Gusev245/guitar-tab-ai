"""
Chord Detector Module for Harmony Recognition
Extracts Chroma features from audio and detects 24 Major/Minor chords + Silence (N).
Pure NumPy & SciPy implementation for blazing-fast speed and compatibility.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
import json
import numpy as np
import scipy.signal as signal
import soundfile as sf
import soxr


PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class ChordSegment:
    start: float       # Start time in seconds
    end: float         # End time in seconds
    chord: str         # Chord name (e.g. 'Am', 'C', 'N')
    confidence: float  # Score (0.0 to 1.0)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict:
        return {
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "duration": round(float(self.duration), 3),
            "chord": self.chord,
            "confidence": round(float(self.confidence), 3),
        }


@dataclass
class ChordTimeline:
    segments: List[ChordSegment]
    duration: float

    def get_chord_at(self, time_s: float) -> Optional[ChordSegment]:
        """Returns the active chord segment at the given timestamp."""
        if not self.segments:
            return None
        # Binary search or linear search (timeline has few hundred segments at most)
        for seg in self.segments:
            if seg.start <= time_s < seg.end:
                return seg
        # Edge case: exactly at or beyond the last segment
        if time_s >= self.segments[-1].start:
            return self.segments[-1]
        return self.segments[0]

    def to_list(self) -> List[Dict]:
        return [s.to_dict() for s in self.segments]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            {
                "duration": round(float(self.duration), 3),
                "total_segments": len(self.segments),
                "segments": self.to_list(),
            },
            indent=indent,
            ensure_ascii=False,
        )

    def save_json(self, output_path: Union[str, Path]) -> None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        print(f"[Chord] Timeline saved to {p}")

    def print_summary(self) -> None:
        """Pretty prints chord progression summary."""
        print(f"\n--- Chord Progression Timeline ({len(self.segments)} segments, {self.duration:.1f}s) ---")
        line = []
        for s in self.segments:
            if s.chord != "N":
                line.append(f"[{s.start:5.1f}s - {s.end:5.1f}s] {s.chord:<4} ({int(s.confidence*100)}%)")
            else:
                line.append(f"[{s.start:5.1f}s - {s.end:5.1f}s] N (silence)")
        print("\n".join(line))
        print("----------------------------------------------------------------------")


class ChordDetector:
    """
    Robust harmony analyzer based on Pitch Class Profiles (Chroma) and
    mean-centered Pearson correlation against harmonic chord templates.
    """

    def __init__(
        self,
        sr: int = 22050,
        n_fft: int = 8192,
        hop_length: int = 2048,
        f_min: float = 65.4,     # C2 note
        f_max: float = 2100.0,   # C7 note
        min_chord_duration: float = 0.35,  # Minimum seconds a chord must sustain
        silence_threshold_db: float = -42.0,
        confidence_threshold: float = 0.35,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.min_chord_duration = min_chord_duration
        self.silence_threshold_db = silence_threshold_db
        self.confidence_threshold = confidence_threshold

        # Precompute chroma filterbank and chord templates
        self._filterbank = self._build_chroma_filterbank()
        self._chord_names, self._templates_centered, self._templates_norm = self._build_templates()

    def _build_chroma_filterbank(self) -> np.ndarray:
        """Constructs a Gaussian-smoothed, frequency-weighted chroma filterbank."""
        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.sr)
        valid_idx = np.where((freqs >= self.f_min) & (freqs <= self.f_max))[0]

        midi_pitches = 12.0 * np.log2(freqs[valid_idx] / 440.0) + 69.0
        f_valid = freqs[valid_idx]

        # Cosine bell curve across log-frequencies to emphasize middle octaves
        log_f = np.log2(f_valid)
        log_min, log_max = np.log2(self.f_min), np.log2(self.f_max)
        spectral_window = np.sin(np.pi * (log_f - log_min) / (log_max - log_min)) ** 2

        W = np.zeros((12, len(freqs)), dtype=np.float32)
        sigma = 0.45  # semitones width

        for pc in range(12):
            diff = (midi_pitches - pc) % 12
            diff = np.minimum(diff, 12 - diff)
            weights = np.exp(-0.5 * (diff / sigma) ** 2) * spectral_window
            W[pc, valid_idx] = weights

        # Normalize filterbank columns
        col_sums = W.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        W /= col_sums
        return W

    def _build_templates(self) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Builds mean-centered templates for 24 Major & Minor triads.
        Major triad: Root (1.0), Maj3 (+4, 0.8), 5th (+7, 0.8)
        Minor triad: Root (1.0), Min3 (+3, 0.8), 5th (+7, 0.8)
        """
        chord_names: List[str] = []
        raw_templates: List[np.ndarray] = []

        base_maj = np.array([1.0, 0, 0, 0, 0.8, 0, 0, 0.8, 0, 0, 0, 0], dtype=np.float32)
        base_min = np.array([1.0, 0, 0, 0.8, 0, 0, 0, 0.8, 0, 0, 0, 0], dtype=np.float32)

        # 12 Major chords
        for root in range(12):
            chord_names.append(PITCH_CLASSES[root])
            raw_templates.append(np.roll(base_maj, root))

        # 12 Minor chords
        for root in range(12):
            chord_names.append(PITCH_CLASSES[root] + "m")
            raw_templates.append(np.roll(base_min, root))

        templates = np.array(raw_templates, dtype=np.float32)  # shape (24, 12)
        # Mean-center each template for Pearson correlation
        t_centered = templates - templates.mean(axis=1, keepdims=True)
        t_norm = np.linalg.norm(t_centered, axis=1, keepdims=True)
        t_norm[t_norm == 0] = 1.0

        return chord_names, t_centered, t_norm

    def extract_chroma(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the chromagram, time grid, and RMS energy (dB) of the audio signal.
        Returns:
            chroma: (12, n_frames) normalized chromagram
            times: (n_frames,) time timestamps
            rms_db: (n_frames,) frame RMS level in dB
        """
        # Ensure 1D float32
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        # Compute STFT
        _, times, Zxx = signal.stft(
            audio,
            fs=self.sr,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            boundary="zeros",
        )
        mag = np.abs(Zxx)

        # Log compression of magnitude: log(1 + 10 * mag)
        mag_compressed = np.log1p(10.0 * mag)

        # Project to chroma
        raw_chroma = self._filterbank @ mag_compressed  # (12, n_frames)

        # Smooth chroma over time with a short moving average (5 frames ~ 0.4s)
        kernel_size = 5
        kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
        smooth_chroma = np.zeros_like(raw_chroma)
        for i in range(12):
            smooth_chroma[i] = signal.medfilt(raw_chroma[i], kernel_size=kernel_size)

        # Normalize chroma per frame (L2 norm)
        norms = np.linalg.norm(smooth_chroma, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_chroma = smooth_chroma / norms

        # Frame RMS energy in dB
        frame_len = self.n_fft
        hop = self.hop_length
        num_frames = len(times)
        rms = np.zeros(num_frames, dtype=np.float32)

        for i in range(num_frames):
            center = int(times[i] * self.sr)
            start = max(0, center - frame_len // 2)
            end = min(len(audio), center + frame_len // 2)
            chunk = audio[start:end]
            if len(chunk) > 0:
                rms[i] = np.sqrt(np.mean(chunk ** 2))

        max_rms = np.max(rms) if len(rms) > 0 and np.max(rms) > 0 else 1.0
        rms_norm = rms / max_rms
        rms_db = 20.0 * np.log10(np.maximum(rms_norm, 1e-5))

        return normalized_chroma, times, rms_db

    def detect_frames(
        self,
        chroma: np.ndarray,
        rms_db: np.ndarray,
    ) -> List[Tuple[str, float]]:
        """
        Matches each chroma frame with chord templates using Pearson correlation.
        Returns list of (chord_name, confidence).
        """
        n_frames = chroma.shape[1]
        results: List[Tuple[str, float]] = []

        # Mean-center chroma frames
        c_centered = chroma - chroma.mean(axis=0, keepdims=True)  # (12, n_frames)
        c_norm = np.linalg.norm(c_centered, axis=0, keepdims=True)  # (1, n_frames)
        c_norm[c_norm == 0] = 1.0

        # Pearson correlation matrix: (24, n_frames)
        # corr[chord_i, frame_j] = (t_centered[chord_i] @ c_centered[:, frame_j]) / (t_norm * c_norm)
        corr_matrix = (self._templates_centered @ c_centered) / (self._templates_norm * c_norm)

        best_chord_indices = np.argmax(corr_matrix, axis=0)
        best_scores = np.max(corr_matrix, axis=0)

        for j in range(n_frames):
            # Check for silence or low energy
            if rms_db[j] < self.silence_threshold_db or best_scores[j] < self.confidence_threshold:
                results.append(("N", 0.0))
            else:
                chord = self._chord_names[best_chord_indices[j]]
                conf = float(np.clip(best_scores[j], 0.0, 1.0))
                results.append((chord, conf))

        return results

    def segment_timeline(
        self,
        frame_chords: List[Tuple[str, float]],
        times: np.ndarray,
        total_duration: float,
    ) -> ChordTimeline:
        """
        Merges consecutive frames into sustained chord segments and prunes short flickers.
        """
        if not frame_chords:
            return ChordTimeline(segments=[], duration=total_duration)

        # Step 1: Initial grouping of consecutive identical chords
        raw_segments: List[ChordSegment] = []
        cur_chord, cur_conf = frame_chords[0]
        cur_start = 0.0
        frame_dur = self.hop_length / self.sr
        conf_acc = [cur_conf]

        for i in range(1, len(frame_chords)):
            chord, conf = frame_chords[i]
            t = times[i]
            if chord == cur_chord:
                conf_acc.append(conf)
            else:
                raw_segments.append(
                    ChordSegment(
                        start=cur_start,
                        end=t,
                        chord=cur_chord,
                        confidence=float(np.mean(conf_acc)),
                    )
                )
                cur_chord = chord
                cur_start = t
                conf_acc = [conf]

        raw_segments.append(
            ChordSegment(
                start=cur_start,
                end=total_duration,
                chord=cur_chord,
                confidence=float(np.mean(conf_acc)),
            )
        )

        # Step 2: Prune transient flickers shorter than min_chord_duration
        # Merge very short segments into the dominant neighbor
        pruned_segments: List[ChordSegment] = []
        for i, seg in enumerate(raw_segments):
            if seg.duration < self.min_chord_duration and pruned_segments and i + 1 < len(raw_segments):
                prev_seg = pruned_segments[-1]
                next_seg = raw_segments[i + 1]
                # Merge into previous if matching or if prev has higher confidence
                if prev_seg.chord == next_seg.chord:
                    prev_seg.end = next_seg.end
                    prev_seg.confidence = max(prev_seg.confidence, next_seg.confidence)
                else:
                    # Absorb into the longer or more confident neighbor
                    if prev_seg.duration >= next_seg.duration:
                        prev_seg.end = seg.end
                    else:
                        seg.chord = next_seg.chord
                        seg.confidence = next_seg.confidence
                        pruned_segments.append(seg)
            else:
                # Merge with previous if same chord
                if pruned_segments and pruned_segments[-1].chord == seg.chord:
                    pruned_segments[-1].end = seg.end
                    pruned_segments[-1].confidence = max(pruned_segments[-1].confidence, seg.confidence)
                else:
                    pruned_segments.append(seg)

        # Final pass: Ensure continuous coverage up to total_duration
        if pruned_segments:
            pruned_segments[0].start = 0.0
            pruned_segments[-1].end = max(pruned_segments[-1].end, total_duration)

        return ChordTimeline(segments=pruned_segments, duration=total_duration)

    def detect_from_audio(self, audio: np.ndarray, sr: int) -> ChordTimeline:
        """Detects chord timeline directly from an audio array."""
        if sr != self.sr:
            audio = soxr.resample(audio, sr, self.sr)
        duration = len(audio) / self.sr
        chroma, times, rms_db = self.extract_chroma(audio)
        frame_chords = self.detect_frames(chroma, rms_db)
        return self.segment_timeline(frame_chords, times, duration)

    def detect_from_file(self, audio_path: Union[str, Path]) -> ChordTimeline:
        """Reads audio file and returns the full chord progression timeline."""
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        data, file_sr = sf.read(str(audio_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)

        return self.detect_from_audio(data, file_sr)
