"""
Модуль форматирования табулатуры (Tab Formatter).

Преобразует список нот с позициями (струна, лад, тайминг)
в текстовый формат ASCII-табулатуры и JSON для веб-плеера (alphaTab).
"""

from typing import List, Dict, Any
import json


class TabFormatter:
    """
    Генератор ASCII-табулатур и структурированных данных.
    """
    STRING_NAMES = ["e", "B", "G", "D", "A", "E"]

    @staticmethod
    def to_ascii(
        mapped_notes: List[Dict[str, Any]],
        notes_per_line: int = 32,
        time_resolution: float = 0.25
    ) -> str:
        """
        Преобразует размеченные ноты в классическую текстовую табулатуру:
        e|---------------------------------|
        B|---------------------------------|
        G|------0---2---0------------------|
        D|--2---------------2--------------|
        A|---------------------------------|
        E|---------------------------------|
        """
        if not mapped_notes:
            return "Пустая табулатура (ноты не обнаружены)."

        # Сортируем ноты по времени начала
        sorted_notes = sorted(mapped_notes, key=lambda n: n.get("start", 0.0))

        # Группируем ноты по временным слотам (квантование)
        min_time = sorted_notes[0].get("start", 0.0)
        max_time = sorted_notes[-1].get("end", sorted_notes[-1].get("start", 0.0) + 0.5)

        total_steps = int((max_time - min_time) / time_resolution) + 1
        total_steps = max(total_steps, len(sorted_notes) * 2)

        # 6 струн (индексы 0..5: 0=e, ..., 5=E)
        grid = {s: ["-"] * total_steps for s in range(6)}

        # Заполняем сетку
        for note in sorted_notes:
            s_idx = note["string"] - 1  # 0..5
            start = note.get("start", 0.0)
            step = int((start - min_time) / time_resolution)
            step = min(step, total_steps - 1)

            fret_str = str(note["fret"])
            # Если лад двузначный (12, 14), занимаем два символа
            for offset, char in enumerate(fret_str):
                if step + offset < total_steps:
                    grid[s_idx][step + offset] = char

        # Разбиваем на строки фиксированной длины (такты/строфы)
        lines_output = []
        for line_start in range(0, total_steps, notes_per_line):
            line_end = min(line_start + notes_per_line, total_steps)

            lines_output.append("")
            for s_idx in range(6):
                s_name = TabFormatter.STRING_NAMES[s_idx]
                chunk = "".join(grid[s_idx][line_start:line_end])
                lines_output.append(f"{s_name}|{chunk}|")

        return "\n".join(lines_output)

    @staticmethod
    def to_json(mapped_notes: List[Dict[str, Any]], title: str = "Guitar Tab") -> str:
        """
        Экспорт в JSON для передачи на фронтенд / alphaTab.
        """
        payload = {
            "title": title,
            "tuning": ["E4", "B3", "G3", "D3", "A2", "E2"],
            "notes_count": len(mapped_notes),
            "notes": mapped_notes
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)
