"""
Модуль разделения аудиодорожек с помощью Meta Demucs.

Использует модель `htdemucs_6s` (6 stems: drums, bass, other, vocals, guitar, piano)
для выделения изолированной партии гитары (guitar.wav) из полного аудиомикса.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


class DemucsRunner:
    """
    Обертка для запуска Demucs через Python API или CLI интерфейс.
    """

    def __init__(self, model_name: str = "htdemucs_6s", output_dir: str = "separated"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def separate_audio(
        self,
        audio_path: str,
        two_stems: Optional[str] = "guitar",
        device: str = "auto"
    ) -> str:
        """
        Запускает разделение аудиофайла.
        Возвращает путь к изолированному гитарному файлу (guitar.wav).

        audio_path: путь к входному файлу (.mp3, .wav, .flac)
        two_stems: если указано "guitar", ускоряет обработку, разделяя на гитару и остальное
        """
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")

        print(f"[*] Запуск Demucs [{self.model_name}] для файла: {audio_path.name}...")

        # Команда CLI demucs
        cmd = [
            sys.executable, "-m", "demucs.separate",
            "-n", self.model_name,
            "-o", str(self.output_dir),
        ]

        if two_stems:
            cmd.extend(["--two-stems", two_stems])

        if device != "auto":
            cmd.extend(["-d", device])

        cmd.append(str(audio_path))

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            print("[+] Demucs успешно завершил разделение!")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            err_msg = e.stderr if hasattr(e, "stderr") and e.stderr else str(e)
            print(f"[-] Ошибка выполнения Demucs: {err_msg}")
            raise RuntimeError(
                "Не удалось выполнить Demucs. Убедитесь, что установлены demucs, torch и ffmpeg:\n"
                "pip install demucs torch torchaudio"
            ) from e

        # Ищем получившийся файл гитары
        track_name = audio_path.stem
        guitar_file = self.output_dir / self.model_name / track_name / "guitar.wav"

        if not guitar_file.exists():
            # Попробуем альтернативные пути
            possible_files = list(self.output_dir.glob(f"**/{track_name}/*guitar*.wav"))
            if possible_files:
                guitar_file = possible_files[0]
            else:
                raise FileNotFoundError(
                    f"Файл гитары не найден в {self.output_dir}. Проверьте вывод Demucs."
                )

        print(f"[+] Выделенная гитарная дорожка сохранена в: {guitar_file}")
        return str(guitar_file)
