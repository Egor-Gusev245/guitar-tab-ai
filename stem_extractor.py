"""
Demucs Stem Extractor Module
Handles audio stem separation with a focus on guitar extraction and efficient caching.
"""

import os
import hashlib
from pathlib import Path
from typing import Optional, Tuple
import soundfile as sf
import torch
import demucs.api


CACHE_DIR = Path(__file__).resolve().parent / ".cache"


def get_audio_file_hash(audio_path: Path) -> str:
    """Computes a quick hash of the audio file based on path, size, and modification time."""
    stat = audio_path.stat()
    hash_str = f"{audio_path.name}_{stat.st_size}_{stat.st_mtime}"
    return hashlib.md5(hash_str.encode("utf-8")).hexdigest()[:12]


def slice_audio(
    input_path: Path,
    output_path: Path,
    start_time: float,
    end_time: float,
) -> None:
    """Efficiently slices a section of an audio file using soundfile seek."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(input_path)) as f:
        sr = f.samplerate
        start_frame = max(0, int(start_time * sr))
        end_frame = min(len(f), int(end_time * sr))
        frames_to_read = end_frame - start_frame
        if frames_to_read <= 0:
            raise ValueError(f"Invalid time range: start={start_time}s, end={end_time}s")

        f.seek(start_frame)
        data = f.read(frames_to_read)
        sf.write(str(output_path), data, sr)


def check_stems_exist(audio_path: Path, cache_dir: Optional[Path] = None) -> Optional[dict]:
    """Checks if separated guitar and backing stems already exist in cache."""
    c_dir = cache_dir or CACHE_DIR
    file_hash = get_audio_file_hash(audio_path)
    song_cache = c_dir / f"{audio_path.stem}_{file_hash}"
    guitar_path = song_cache / "guitar.wav"
    no_guitar_path = song_cache / "no_guitar.wav"
    if guitar_path.exists() and no_guitar_path.exists():
        return {
            "guitar": guitar_path,
            "no_guitar": no_guitar_path,
            "cache_dir_name": song_cache.name,
        }
    return None


class GuitarStemExtractor:
    def __init__(
        self,
        model_name: str = "htdemucs_6s",
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._separator: Optional[demucs.api.Separator] = None

    @property
    def separator(self) -> demucs.api.Separator:
        """Lazy loader for Demucs Separator."""
        if self._separator is None:
            print(f"[Demucs] Loading model '{self.model_name}' on {self.device}...")
            self._separator = demucs.api.Separator(
                model=self.model_name,
                device=self.device,
            )
        return self._separator

    def separate_full_stems(
        self,
        audio_path: Path,
        force_recompute: bool = False,
    ) -> dict:
        """
        Separates full audio into:
        1. 'guitar.wav' - isolated guitar stem.
        2. 'no_guitar.wav' - backing track (vocals, drums, bass, piano, other).
        Returns dict with paths and relative URLs.
        """
        audio_path = Path(audio_path).resolve()
        file_hash = get_audio_file_hash(audio_path)
        song_cache = self.cache_dir / f"{audio_path.stem}_{file_hash}"
        guitar_stem_path = song_cache / "guitar.wav"
        no_guitar_stem_path = song_cache / "no_guitar.wav"

        if guitar_stem_path.exists() and no_guitar_stem_path.exists() and not force_recompute:
            print(f"[Cache] Reusing full separated stems from {song_cache.name}")
            return {
                "guitar": guitar_stem_path,
                "no_guitar": no_guitar_stem_path,
                "cache_dir_name": song_cache.name,
            }

        song_cache.mkdir(parents=True, exist_ok=True)
        print(f"[Demucs] Running full separation on {audio_path.name}...")
        origin, separated = self.separator.separate_audio_file(str(audio_path))

        if "guitar" not in separated:
            raise RuntimeError(f"Model '{self.model_name}' did not produce a 'guitar' stem.")

        # 1. Save guitar stem
        guitar_tensor = separated["guitar"]
        demucs.api.save_audio(
            guitar_tensor,
            str(guitar_stem_path),
            samplerate=self.separator.samplerate,
        )

        # 2. Sum and save backing track (all non-guitar stems)
        other_tensors = [tensor for name, tensor in separated.items() if name != "guitar"]
        if other_tensors:
            no_guitar_tensor = sum(other_tensors)
            demucs.api.save_audio(
                no_guitar_tensor,
                str(no_guitar_stem_path),
                samplerate=self.separator.samplerate,
            )

        print(f"[Demucs] Saved guitar.wav and no_guitar.wav to {song_cache}")
        return {
            "guitar": guitar_stem_path,
            "no_guitar": no_guitar_stem_path,
            "cache_dir_name": song_cache.name,
        }

    def get_full_guitar_stem(
        self,
        audio_path: Path,
        force_recompute: bool = False,
    ) -> Path:
        """
        Extracts and caches the full isolated guitar stem from an audio file.
        Returns the path to the cached 'guitar.wav'.
        """
        audio_path = Path(audio_path).resolve()
        file_hash = get_audio_file_hash(audio_path)
        song_cache = self.cache_dir / f"{audio_path.stem}_{file_hash}"
        guitar_stem_path = song_cache / "guitar.wav"

        if guitar_stem_path.exists() and not force_recompute:
            print(f"[Cache] Reusing existing guitar stem: {guitar_stem_path.name}")
            return guitar_stem_path

        song_cache.mkdir(parents=True, exist_ok=True)
        print(f"[Demucs] Separating full track: {audio_path.name}...")
        origin, separated = self.separator.separate_audio_file(str(audio_path))

        if "guitar" not in separated:
            raise RuntimeError(
                f"Model '{self.model_name}' did not produce a 'guitar' stem. "
                f"Available stems: {list(separated.keys())}"
            )

        # Save guitar stem
        guitar_tensor = separated["guitar"]
        demucs.api.save_audio(
            guitar_tensor,
            str(guitar_stem_path),
            samplerate=self.separator.samplerate,
        )
        print(f"[Demucs] Saved guitar stem to {guitar_stem_path}")
        return guitar_stem_path

    def get_guitar_snippet(
        self,
        audio_path: Path,
        start_time: float,
        end_time: float,
        full_track_demux: bool = False,
    ) -> Path:
        """
        Extracts a guitar snippet between start_time and end_time.
        - If full_track_demux is True or full guitar stem is already cached:
          slices from the full cached guitar stem.
        - Otherwise (Fast Mode): slices the raw audio first, then runs Demucs only
          on that short snippet (much faster on CPU!).
        """
        audio_path = Path(audio_path).resolve()
        file_hash = get_audio_file_hash(audio_path)
        song_cache = self.cache_dir / f"{audio_path.stem}_{file_hash}"
        full_guitar_path = song_cache / "guitar.wav"

        snippet_name = f"guitar_{start_time:.2f}_{end_time:.2f}.wav"
        snippet_path = song_cache / snippet_name

        # 1. If full guitar stem already exists or full demux requested
        if full_guitar_path.exists() or full_track_demux:
            if not full_guitar_path.exists():
                full_guitar_path = self.get_full_guitar_stem(audio_path)
            print(f"[Slicing] Slicing guitar [{start_time:.2f}s - {end_time:.2f}s] from cached stem...")
            slice_audio(full_guitar_path, snippet_path, start_time, end_time)
            return snippet_path

        # 2. Fast Snippet Mode: Demux only the requested segment with small padding
        song_cache.mkdir(parents=True, exist_ok=True)
        padding = 0.5  # 500ms context padding for Demucs
        padded_start = max(0.0, start_time - padding)
        padded_end = end_time + padding

        raw_snippet = song_cache / f"raw_snippet_{padded_start:.2f}_{padded_end:.2f}.wav"
        slice_audio(audio_path, raw_snippet, padded_start, padded_end)

        print(
            f"[Demucs] Fast Snippet Mode: separating {padded_end - padded_start:.1f}s section..."
        )
        _, separated = self.separator.separate_audio_file(str(raw_snippet))
        if "guitar" not in separated:
            raise RuntimeError("Model did not return a 'guitar' stem.")

        padded_guitar_path = song_cache / f"padded_guitar_{padded_start:.2f}_{padded_end:.2f}.wav"
        demucs.api.save_audio(
            separated["guitar"],
            str(padded_guitar_path),
            samplerate=self.separator.samplerate,
        )

        # Slice out the exact [start_time, end_time] from the padded guitar audio
        offset_start = start_time - padded_start
        duration = end_time - start_time
        slice_audio(
            padded_guitar_path,
            snippet_path,
            offset_start,
            offset_start + duration,
        )

        # Clean up temporary raw snippet
        if raw_snippet.exists():
            raw_snippet.unlink()

        return snippet_path
