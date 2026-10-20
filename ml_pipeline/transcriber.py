"""
Модуль распознавания нот из аудиозаписи (Transcriber).
Оптимизирован для классического гитарного строя Standard E:
частотный диапазон от 80 Гц (E2 ~ 82.4 Гц) до 1350 Гц (24 лад первой струны).
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import math


class AudioTranscriber:
    """
    Класс транскрибации аудиофайла в нотные события на базе Librosa (pYIN + Onset Detection).
    """

    def __init__(self, min_freq: float = 80.0, max_freq: float = 1350.0):
        # 80 Гц строго отсекает низкочастотный гул и басовые шумы ниже ноты Ми
        self.min_freq = min_freq
        self.max_freq = max_freq

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        audio_path = str(Path(audio_path).resolve())

        try:
            import librosa
            import numpy as np
        except ImportError:
            raise RuntimeError(
                "Библиотека librosa не установлена. Выполните: pip install librosa soundfile numpy scipy"
            )

        print(f"[*] Анализ аудио через Librosa (pYIN, диапазон Standard E: {self.min_freq}-{self.max_freq} Гц)...")
        y, sr = librosa.load(audio_path, sr=22050, mono=True)

        # 1. Определение моментов атак медиатора/пальцев (Onsets)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        # 2. Оценка основной частоты нот через pYIN
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=self.min_freq,
            fmax=self.max_freq,
            sr=sr,
            frame_length=2048,
        )
        times = librosa.times_like(f0, sr=sr)

        if len(onset_times) == 0:
            onset_times = np.array([0.0])

        duration = librosa.get_duration(y=y, sr=sr)
        all_boundaries = np.append(onset_times, duration)

        formatted_notes = []

        # 3. Квантование интервалов между атаками
        for i in range(len(all_boundaries) - 1):
            t_start = all_boundaries[i]
            t_end = min(all_boundaries[i + 1], t_start + 2.0)

            mask = (times >= t_start) & (times < t_end) & (voiced_flag == True)
            f0_segment = f0[mask]

            if len(f0_segment) > 0:
                median_freq = float(np.median(f0_segment))
                if median_freq >= self.min_freq:
                    # MIDI нота
                    midi_pitch = int(round(69 + 12 * math.log2(median_freq / 440.0)))

                    formatted_notes.append({
                        "pitch": midi_pitch,
                        "start": round(float(t_start), 3),
                        "end": round(float(t_end), 3),
                        "amplitude": 0.8
                    })

        print(f"[+] Распознано нотных событий: {len(formatted_notes)}")
        return formatted_notes

    @staticmethod
    def get_demo_riff(riff_name: str = "nothing_else_matters") -> List[Dict[str, Any]]:
        """
        Встроенные партии гитары в Standard E строе.
        """
        if riff_name == "smoke_on_the_water":
            # Deep Purple - Smoke on the water (Standard E)
            pattern = [
                (50, 0.0, 0.4), (53, 0.5, 0.9), (55, 1.0, 1.4),
                (50, 1.5, 1.9), (53, 2.0, 2.4), (56, 2.5, 2.7), (55, 2.8, 3.4),
                (50, 3.5, 3.9), (53, 4.0, 4.4), (55, 4.5, 4.9),
                (53, 5.0, 5.4), (50, 5.5, 6.2)
            ]
        elif riff_name == "pentatonic":
            # Блюзовая пентатоника Am
            pitches = [57, 60, 62, 63, 64, 67, 69, 67, 64, 63, 62, 60, 57]
            pattern = []
            t = 0.0
            for p in pitches:
                pattern.append((p, t, t + 0.35))
                t += 0.4
        else:
            # Metallica - Nothing Else Matters (Открытое арпеджио Standard E)
            pattern = [
                (40, 0.0, 0.5),  # 6-я E
                (55, 0.5, 1.0),  # 3-я G
                (59, 1.0, 1.5),  # 2-я B
                (64, 1.5, 2.0),  # 1-я e
                (59, 2.0, 2.5),  # 2-я B
                (55, 2.5, 3.0),  # 3-я G
                (40, 3.0, 3.5),
                (55, 3.5, 4.0),
                (59, 4.0, 4.5),
                (64, 4.5, 5.0),
                (59, 5.0, 5.5),
                (55, 5.5, 6.0),
            ]

        return [
            {"pitch": p, "start": s, "end": e, "amplitude": 0.8}
            for p, s, e in pattern
        ]
