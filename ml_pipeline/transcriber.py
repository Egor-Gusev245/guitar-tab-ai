"""
Модуль распознавания нот из аудиозаписи (Transcriber).

Использует модель Spotify Basic Pitch (полифоническое распознавание нот)
для преобразования изолированного гитарного аудио в список нотных событий (нота, тайминг, громкость).
Также содержит синтетический генератор тестовых партий для быстрой отладки без загрузки весов.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import math


class AudioTranscriber:
    """
    Класс транскрибации аудиофайла в нотные события.
    """

    def __init__(self, onset_threshold: float = 0.5, frame_threshold: float = 0.3):
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold

    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Транскрибирует аудио в список нот с помощью Spotify Basic Pitch:
        Возвращает:
            [
                {"pitch": 52, "start": 0.5, "end": 0.9, "amplitude": 0.8},
                ...
            ]
        """
        try:
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH
        except ImportError:
            print("[!] Библиотека basic-pitch не установлена.")
            print("    Установите: pip install basic-pitch")
            raise

        audio_path = str(Path(audio_path).resolve())
        print(f"[*] Анализ аудио через Spotify Basic Pitch: {Path(audio_path).name}...")

        # Запуск инференса Basic Pitch
        model_output, midi_data, note_events = predict(
            audio_path,
            onset_threshold=self.onset_threshold,
            frame_threshold=self.frame_threshold,
            minimum_note_length=50,  # мс
            minimum_frequency=80.0,   # нижняя граница гитары (E2 ~ 82 Hz)
            maximum_frequency=1200.0  # верхняя граница гитары
        )

        formatted_notes = []
        for start, end, pitch, amp, _ in note_events:
            formatted_notes.append({
                "pitch": int(round(pitch)),
                "start": round(float(start), 3),
                "end": round(float(end), 3),
                "amplitude": round(float(amp), 2)
            })

        # Сортируем по времени начала
        formatted_notes.sort(key=lambda x: x["start"])
        print(f"[+] Распознано нот: {len(formatted_notes)}")
        return formatted_notes

    @staticmethod
    def get_demo_riff(riff_name: str = "pentatonic") -> List[Dict[str, Any]]:
        """
        Генератор эталонных тестовых гитарных партий для тестирования
        алгоритма аппликатуры и генератора табов без необходимости
        тяжелых вычислений нейросетей.
        """
        if riff_name == "smoke_on_the_water":
            # E4=64, G4=67, A4=69, Bb4=70...
            # Играется на 4 струне D (50): 50(открытая), 53(лад 3), 55(лад 5)...
            pattern = [
                (50, 0.0, 0.4),  # D
                (53, 0.5, 0.9),  # F
                (55, 1.0, 1.4),  # G
                (50, 1.5, 1.9),  # D
                (53, 2.0, 2.4),  # F
                (56, 2.5, 2.7),  # G#
                (55, 2.8, 3.4),  # G
                (50, 3.5, 3.9),  # D
                (53, 4.0, 4.4),  # F
                (55, 4.5, 4.9),  # G
                (53, 5.0, 5.4),  # F
                (50, 5.5, 6.2),  # D
            ]
        elif riff_name == "nothing_else_matters":
            # Знаменитое открытое арпеджио: E2, G3, B3, E4, B3, G3
            # MIDI: E2=40, G3=55, B3=59, E4=64
            pattern = [
                (40, 0.0, 0.5),  # 6-я открытая E
                (55, 0.5, 1.0),  # 3-я открытая G
                (59, 1.0, 1.5),  # 2-я открытая B
                (64, 1.5, 2.0),  # 1-я открытая e
                (59, 2.0, 2.5),  # 2-я B
                (55, 2.5, 3.0),  # 3-я G
                (40, 3.0, 3.5),
                (55, 3.5, 4.0),
                (59, 4.0, 4.5),
                (64, 4.5, 5.0),
                (59, 5.0, 5.5),
                (55, 5.5, 6.0),
            ]
        else:
            # Блюзовая пентатоника Am: A3(57), C4(60), D4(62), Eb4(63), E4(64), G4(67), A4(69)
            pitches = [57, 60, 62, 63, 64, 67, 69, 67, 64, 63, 62, 60, 57]
            pattern = []
            t = 0.0
            for p in pitches:
                pattern.append((p, t, t + 0.35))
                t += 0.4

        return [
            {"pitch": p, "start": s, "end": e, "amplitude": 0.8}
            for p, s, e in pattern
        ]
