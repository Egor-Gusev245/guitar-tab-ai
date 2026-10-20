"""
Модуль трансляции нот в гитарную аппликатуру (Fretboard Mapper).

Использует алгоритм динамического программирования для поиска
наиболее эргономичного положения кисти гитариста на грифе (минимизация
перемещений между ладами и избегание физически неудобных скачков).
"""

from typing import List, Tuple, Dict, Any, Optional
import math


class GuitarFretboard:
    """
    Модель грифа 6-струнной гитары в стандартном строе (Standard E).
    Номера струн:
        1: e (E4, MIDI 64) - первая (самая тонкая)
        2: B (B3, MIDI 59)
        3: G (G3, MIDI 55)
        4: D (D3, MIDI 50)
        5: A (A2, MIDI 45)
        6: E (E2, MIDI 40) - шестая (самая толстая)
    """

    # Стандартный строй (MIDI номера открытых струн)
    # Индекс 0 -> 1-я струна (E4), ..., Индекс 5 -> 6-я струна (E2)
    TUNING_MIDI = [64, 59, 55, 50, 45, 40]
    STRING_NAMES = ["e", "B", "G", "D", "A", "E"]
    MAX_FRETS = 24

    def __init__(self, tuning: Optional[List[int]] = None, max_frets: int = 24):
        self.tuning = tuning if tuning else self.TUNING_MIDI
        self.max_frets = max_frets

    def get_possible_positions(self, midi_pitch: int) -> List[Tuple[int, int]]:
        """
        Возвращает все возможные позиции (номер_струны_0_based, лад) для заданной MIDI ноты.
        """
        positions = []
        for string_idx, open_pitch in enumerate(self.tuning):
            fret = midi_pitch - open_pitch
            if 0 <= fret <= self.max_frets:
                positions.append((string_idx, fret))
        return positions

    def transition_cost(
        self,
        pos1: Tuple[int, int],
        pos2: Tuple[int, int],
        prev_hand_fret: int = 0
    ) -> float:
        """
        Функция штрафа (стоимости перехода) между двумя последовательными нотами.
        Оценивает физическую сложность перемещения руки гитариста:
        - Расстояние между ладами (смещение позиции руки)
        - Смена струн
        - Открытые струны (лад 0) не требуют смещения кисти
        """
        s1, f1 = pos1
        s2, f2 = pos2

        # Если одна из нот - открытая струна (лад 0), смещать руку не обязательно
        if f1 == 0 and f2 == 0:
            return abs(s2 - s1) * 0.5
        if f1 == 0:
            return abs(f2 - prev_hand_fret) * 1.0 + abs(s2 - s1) * 0.5
        if f2 == 0:
            return abs(s2 - s1) * 0.5

        # Расстояние между ладами (основной фактор)
        fret_diff = abs(f2 - f1)

        # Скачок через струны
        string_diff = abs(s2 - s1)

        # Нелинейный штраф за слишком большой растяг пальцев (> 4 ладов)
        stretch_penalty = 0.0
        if fret_diff > 4:
            stretch_penalty = (fret_diff - 4) ** 2 * 3.0

        return fret_diff * 1.5 + string_diff * 0.8 + stretch_penalty

    def optimize_sequence(self, note_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Алгоритм динамического программирования (Viterbi-подобный)
        для поиска оптимального пути по грифу для цепочки нот.

        note_events: список словарей с полями:
            - 'pitch': int (MIDI pitch)
            - 'start': float (время начала в сек)
            - 'end': float (время окончания)
        """
        if not note_events:
            return []

        # 1. Поиск допустимых позиций для каждой ноты
        stages = []
        for note in note_events:
            pitch = int(round(note["pitch"]))
            positions = self.get_possible_positions(pitch)
            if not positions:
                # Если нота вне диапазона гитары, транспонируем в октаву
                if pitch < self.tuning[-1]:
                    positions = self.get_possible_positions(pitch + 12)
                elif pitch > self.tuning[0] + self.max_frets:
                    positions = self.get_possible_positions(pitch - 12)

            if not positions:
                # Если все еще нет, ставим крайнюю допустимую
                positions = [(0, min(self.max_frets, max(0, pitch - self.tuning[0])))]

            stages.append(positions)

        # 2. Динамическое программирование (DP)
        # dp[i][j] = минимальная стоимость пути до j-й позиции на шаге i
        num_notes = len(note_events)
        dp = []
        backtrack = []

        # Инициализация первого шага (предпочитаем позиции ближе к 0-5 ладам)
        first_step_costs = []
        for string_idx, fret in stages[0]:
            initial_cost = fret * 0.5  # предпочтение началу грифа
            first_step_costs.append(initial_cost)
        dp.append(first_step_costs)
        backtrack.append([-1] * len(stages[0]))

        # Основной цикл по нотам
        for i in range(1, num_notes):
            current_positions = stages[i]
            prev_positions = stages[i - 1]

            step_costs = []
            step_backtrack = []

            for curr_idx, curr_pos in enumerate(current_positions):
                min_cost = float("inf")
                best_prev_idx = -1

                for prev_idx, prev_pos in enumerate(prev_positions):
                    trans_cost = self.transition_cost(prev_pos, curr_pos)
                    total_cost = dp[i - 1][prev_idx] + trans_cost

                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_prev_idx = prev_idx

                step_costs.append(min_cost)
                step_backtrack.append(best_prev_idx)

            dp.append(step_costs)
            backtrack.append(step_backtrack)

        # 3. Восстановление оптимального пути (Backtracking)
        last_step_costs = dp[-1]
        best_last_idx = min(range(len(last_step_costs)), key=lambda k: last_step_costs[k])

        optimal_path = [stages[-1][best_last_idx]]
        curr_backtrack_idx = best_last_idx

        for i in range(num_notes - 1, 0, -1):
            curr_backtrack_idx = backtrack[i][curr_backtrack_idx]
            optimal_path.append(stages[i - 1][curr_backtrack_idx])

        optimal_path.reverse()

        # Формируем итоговые аннотированные ноты
        result = []
        for i, note in enumerate(note_events):
            string_idx, fret = optimal_path[i]
            result.append({
                "pitch": note["pitch"],
                "start": note.get("start", 0.0),
                "end": note.get("end", 0.0),
                "string": string_idx + 1,  # 1-based (1..6)
                "string_name": self.STRING_NAMES[string_idx],
                "fret": fret,
            })

        return result
