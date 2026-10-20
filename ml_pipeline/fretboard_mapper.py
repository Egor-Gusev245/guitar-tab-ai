"""
Модуль трансляции нот в гитарную аппликатуру (Fretboard Mapper).
Стандартизирован под классический строй гитары Standard E (E A D G B E).

Использует алгоритм динамического программирования для поиска
наиболее эргономичного положения руки гитариста на грифе.
"""

from typing import List, Tuple, Dict, Any, Optional


class GuitarFretboard:
    """
    Модель грифа 6-струнной гитары в классическом строе Standard E:
        1-я струна: e (E4, MIDI 64, ~329.6 Гц)
        2-я струна: B (B3, MIDI 59, ~246.9 Гц)
        3-я струна: G (G3, MIDI 55, ~196.0 Гц)
        4-я струна: D (D3, MIDI 50, ~146.8 Гц)
        5-я струна: A (A2, MIDI 45, ~110.0 Гц)
        6-я струна: E (E2, MIDI 40, ~82.4 Гц)
    """

    TUNING_MIDI = [64, 59, 55, 50, 45, 40]
    STRING_NAMES = ["e", "B", "G", "D", "A", "E"]
    TUNING_NAME = "Standard E (E A D G B E)"
    MAX_FRETS = 24
    LOWEST_PITCH = 40  # E2

    def __init__(self):
        self.tuning = self.TUNING_MIDI
        self.string_names = self.STRING_NAMES
        self.tuning_name = self.TUNING_NAME

    def get_possible_positions(self, midi_pitch: int) -> List[Tuple[int, int]]:
        """
        Возвращает допустимые пары (номер_струны_0_based, лад) для заданной MIDI-ноты.
        """
        positions = []
        for string_idx, open_pitch in enumerate(self.tuning):
            fret = midi_pitch - open_pitch
            if 0 <= fret <= self.MAX_FRETS:
                positions.append((string_idx, fret))
        return positions

    def transition_cost(
        self,
        pos1: Tuple[int, int],
        pos2: Tuple[int, int],
        prev_hand_fret: int = 0
    ) -> float:
        """
        Функция штрафа (стоимости перехода) между двумя нотами.
        Минимизирует смещение руки по ладам и неестественные скачки через струны.
        """
        s1, f1 = pos1
        s2, f2 = pos2

        # Открытая струна (лад 0) не требует перемещения позиции кисти
        if f1 == 0 and f2 == 0:
            return abs(s2 - s1) * 0.5
        if f1 == 0:
            return abs(f2 - prev_hand_fret) * 1.0 + abs(s2 - s1) * 0.5
        if f2 == 0:
            return abs(s2 - s1) * 0.5

        fret_diff = abs(f2 - f1)
        string_diff = abs(s2 - s1)

        # Штраф за слишком широкий растяг пальцев (> 4 ладов)
        stretch_penalty = 0.0
        if fret_diff > 4:
            stretch_penalty = (fret_diff - 4) ** 2 * 3.0

        return fret_diff * 1.5 + string_diff * 0.8 + stretch_penalty

    def optimize_sequence(self, note_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Алгоритм динамического программирования (Viterbi) для поиска
        оптимальной аппликатуры цепочки нот на грифе Standard E.
        """
        if not note_events:
            return []

        stages = []
        for note in note_events:
            pitch = int(round(note["pitch"]))

            # Если нота ниже E2 (например, басовый шум), приводим к открытой 6-й струне E2
            if pitch < self.LOWEST_PITCH:
                pitch = self.LOWEST_PITCH

            positions = self.get_possible_positions(pitch)

            # Если нота выше 24 лада 1-й струны, транспонируем на октаву вниз
            if not positions and pitch > self.tuning[0] + self.MAX_FRETS:
                positions = self.get_possible_positions(pitch - 12)

            if not positions:
                positions = [(0, min(self.MAX_FRETS, max(0, pitch - self.tuning[0])))]

            stages.append(positions)

        num_notes = len(note_events)
        dp = []
        backtrack = []

        # Первый шаг: предпочтение естественной позиции на первых 0-5 ладах
        first_step_costs = [fret * 0.5 for _, fret in stages[0]]
        dp.append(first_step_costs)
        backtrack.append([-1] * len(stages[0]))

        # Динамическое программирование
        for i in range(1, num_notes):
            current_positions = stages[i]
            prev_positions = stages[i - 1]

            step_costs = []
            step_backtrack = []

            for curr_pos in current_positions:
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

        # Восстановление пути (Backtracking)
        last_step_costs = dp[-1]
        best_last_idx = min(range(len(last_step_costs)), key=lambda k: last_step_costs[k])

        optimal_path = [stages[-1][best_last_idx]]
        curr_backtrack_idx = best_last_idx

        for i in range(num_notes - 1, 0, -1):
            curr_backtrack_idx = backtrack[i][curr_backtrack_idx]
            optimal_path.append(stages[i - 1][curr_backtrack_idx])

        optimal_path.reverse()

        result = []
        for i, note in enumerate(note_events):
            string_idx, fret = optimal_path[i]
            result.append({
                "pitch": note["pitch"],
                "start": note.get("start", 0.0),
                "end": note.get("end", 0.0),
                "string": string_idx + 1,
                "string_name": self.string_names[string_idx],
                "fret": fret,
            })

        return result
