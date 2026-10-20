"""
Главный скрипт ML-пайплайна (Standard E):
Audio -> Demucs (htdemucs_6s) -> Transcriber (Librosa pYIN) -> Fretboard Mapper -> Tab.

Примеры использования:
1. Тест на встроенном рифе:
   python pipeline.py --demo nothing_else_matters

2. Анализ полноценной песни (Demucs отделит гитару от ударных и вокала):
   python pipeline.py --input song.mp3

3. Анализ отдельной гитарной дорожки (пропустить Demucs):
   python pipeline.py --input guitar_solo.wav --skip-demucs
"""

import argparse
import sys
from pathlib import Path

# Поддержка UTF-8 вывода в Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fretboard_mapper import GuitarFretboard
from tab_formatter import TabFormatter
from transcriber import AudioTranscriber


def run_pipeline(
    input_audio: str = None,
    output_tab: str = "output_tab.txt",
    output_json: str = None,
    skip_demucs: bool = False,
    demo_riff: str = None,
):
    print("=" * 60)
    print("[AI GUITAR TAB TRANSCRIBER: ML PIPELINE (Standard E)]")
    print("=" * 60)

    # 1. Получение нот
    if demo_riff:
        print(f"\n[1/3] Режим демонстрации: партия '{demo_riff}'...")
        notes = AudioTranscriber.get_demo_riff(demo_riff)
        track_title = f"Demo: {demo_riff}"
    else:
        if not input_audio:
            print("[-] Ошибка: укажите --input <файл.mp3> или выберите --demo <название_рифа>")
            sys.exit(1)

        input_path = Path(input_audio)
        if not input_path.exists():
            print(f"[-] Файл не найден: {input_audio}")
            sys.exit(1)

        track_title = input_path.stem

        # Шаг 1: Разделение через Demucs
        if skip_demucs:
            print(f"\n[1/3] Пропуск Demucs: используем напрямую {input_path.name}")
            guitar_audio_path = str(input_path)
        else:
            print(f"\n[1/3] Выделение дорожки гитары через Demucs (htdemucs_6s)...")
            from demucs_runner import DemucsRunner
            demucs = DemucsRunner(model_name="htdemucs_6s", output_dir="separated")
            guitar_audio_path = demucs.separate_audio(str(input_path))

        # Шаг 2: Распознавание нот
        print(f"\n[2/3] Транскрибация нот (Librosa pYIN, диапазон Standard E)...")
        transcriber = AudioTranscriber(min_freq=80.0, max_freq=1350.0)
        notes = transcriber.transcribe(guitar_audio_path)

    if not notes:
        print("[-] Ноты не обнаружены в аудиозаписи.")
        sys.exit(0)

    print(f"[+] Распознано нотных событий: {len(notes)}")

    # Шаг 3: Оптимизация аппликатуры на грифе гитары
    print(f"\n[3/3] Оптимизация аппликатуры грифа под Standard E (Dynamic Programming)...")
    fretboard = GuitarFretboard()
    mapped_notes = fretboard.optimize_sequence(notes)

    # Шаг 4: Форматирование табулатуры
    ascii_tab = TabFormatter.to_ascii(mapped_notes)
    json_data = TabFormatter.to_json(mapped_notes, title=track_title)

    # Вывод результата
    print("\n" + "=" * 60)
    print(f"РЕЗУЛЬТАТ ТРАНСКРИБАЦИИ: {track_title}")
    print("=" * 60)
    print(ascii_tab)
    print("=" * 60)

    # Сохранение в файл
    if output_tab:
        out_tab_path = Path(output_tab)
        out_tab_path.parent.mkdir(parents=True, exist_ok=True)
        out_tab_path.write_text(ascii_tab, encoding="utf-8")
        print(f"[+] Табулатура сохранена в: {out_tab_path.resolve()}")

    if output_json:
        out_json_path = Path(output_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(json_data, encoding="utf-8")
        print(f"[+] JSON сохранен в: {out_json_path.resolve()}")

    return mapped_notes, ascii_tab


def main():
    parser = argparse.ArgumentParser(description="AI Guitar Tab Transcriber (Standard E)")
    parser.add_argument("--input", "-i", type=str, help="Путь к аудиофайлу (.mp3, .wav)")
    parser.add_argument("--output_tab", "-o", type=str, default="output_tab.txt", help="Файл для сохранения ASCII табов")
    parser.add_argument("--output_json", "-j", type=str, help="Файл для сохранения JSON табов (для frontend)")
    parser.add_argument("--skip-demucs", action="store_true", help="Пропустить Demucs (если аудио уже чистая гитара)")
    parser.add_argument(
        "--demo",
        choices=["nothing_else_matters", "smoke_on_the_water", "pentatonic"],
        help="Запуск на встроенном демонстрационном рифе",
    )

    args = parser.parse_args()

    if not args.input and not args.demo:
        args.demo = "nothing_else_matters"

    run_pipeline(
        input_audio=args.input,
        output_tab=args.output_tab,
        output_json=args.output_json,
        skip_demucs=args.skip_demucs,
        demo_riff=args.demo,
    )


if __name__ == "__main__":
    main()
