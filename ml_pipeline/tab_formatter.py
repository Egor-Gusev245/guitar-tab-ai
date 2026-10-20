"""
Модуль форматирования табулатуры (Tab Formatter).
Генерирует классическую ASCII-табулатуру для строя Standard E и JSON для alphaTab.
"""

from typing import List, Dict, Any
import json


class TabFormatter:
    """
    Генератор ASCII-табулатур в строе Standard E.
    """
    STRING_NAMES = ["e", "B", "G", "D", "A", "E"]
    TUNING_NAME = "Standard E (E A D G B E)"

    @classmethod
    def to_ascii(
        cls,
        mapped_notes: List[Dict[str, Any]],
        notes_per_line: int = 32,
        time_resolution: float = 0.25
    ) -> str:
        """
        Преобразует ноты в классическую табулатуру:
        e|---------------------------------|
        B|---------------------------------|
        G|------0---2---0------------------|
        D|--2---------------2--------------|
        A|---------------------------------|
        E|0--------------------------------|
        """
        if not mapped_notes:
            return f"Tuning: {cls.TUNING_NAME}\nПустая табулатура (ноты не обнаружены)."

        sorted_notes = sorted(mapped_notes, key=lambda n: n.get("start", 0.0))

        min_time = sorted_notes[0].get("start", 0.0)
        max_time = sorted_notes[-1].get("end", sorted_notes[-1].get("start", 0.0) + 0.5)

        total_steps = int((max_time - min_time) / time_resolution) + 1
        total_steps = max(total_steps, len(sorted_notes) * 2)

        # 6 струн
        grid = {s: ["-"] * total_steps for s in range(6)}

        for note in sorted_notes:
            s_idx = note["string"] - 1  # 0..5
            start = note.get("start", 0.0)
            step = int((start - min_time) / time_resolution)
            step = min(step, total_steps - 1)

            fret_str = str(note["fret"])
            for offset, char in enumerate(fret_str):
                if step + offset < total_steps:
                    grid[s_idx][step + offset] = char

        lines_output = [f"Tuning: {cls.TUNING_NAME}"]

        for line_start in range(0, total_steps, notes_per_line):
            line_end = min(line_start + notes_per_line, total_steps)

            lines_output.append("")
            for s_idx in range(6):
                s_name = cls.STRING_NAMES[s_idx].ljust(2)
                chunk = "".join(grid[s_idx][line_start:line_end])
                lines_output.append(f"{s_name}|{chunk}|")

        return "\n".join(lines_output)

    @classmethod
    def to_json(cls, mapped_notes: List[Dict[str, Any]], title: str = "Guitar Tab") -> str:
        payload = {
            "title": title,
            "tuning": cls.TUNING_NAME,
            "string_names": cls.STRING_NAMES,
            "notes_count": len(mapped_notes),
            "notes": mapped_notes
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)
