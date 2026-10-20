# ML Pipeline: Audio Separation & Transcription (Standard E)

Модули выделения гитарной дорожки из аудиозаписи и генерации гитарных табулатур в строе Standard E (E A D G B E).

## Архитектура
1. `demucs_runner.py`:
   - Выделение партии гитары (`guitar.wav`) с помощью нейросети Meta Demucs (`htdemucs_6s`).
2. `transcriber.py`:
   - Детекция высоты тона нот и моментов атак (Librosa pYIN + Onset Detection) в диапазоне 80 - 1350 Гц.
3. `fretboard_mapper.py`:
   - Алгоритм динамического программирования (Viterbi) для поиска эргономичной аппликатуры на грифе.
4. `tab_formatter.py`:
   - Генерация классической ASCII-табулатуры и JSON для веб-плеера (alphaTab).
5. `pipeline.py`:
   - Главная консольная точка входа.

## Команды запуска
```bash
# Тест на встроенном рифе:
python pipeline.py --demo nothing_else_matters

# Анализ песни (полный трек):
python pipeline.py --input song.mp3

# Анализ только гитары (без барабанов):
python pipeline.py --input solo.wav --skip-demucs
```
