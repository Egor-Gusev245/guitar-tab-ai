# ML Pipeline: Audio Separation & Transcription

В этой папке располагаются модули обработки звука и генерации табулатур.

## Компоненты

1. `demucs_runner.py`:
   - Запуск Meta Demucs (`htdemucs_6s`) для извлечения отдельной гитарной дорожки (`guitar.wav`).
2. `transcriber.py`:
   - Детекция нот с помощью Spotify `basic-pitch` (извлечение нотных событий: pitch, onsets, offsets, velocities).
3. `fretboard_mapper.py`:
   - Алгоритм отображения MIDI-высот нот на 6 струн акустической/электрогитары (Standard E tuning: E2, A2, D3, G3, B3, E4) с оптимизацией расстояния руки по ладам (минимизация перемещений кисти).
4. `exporters.py`:
   - Генерация ASCII табулатуры (текстовый формат) и MusicXML для визуализации в библиотеке `alphaTab` на фронтенде.

## Зависимости ML
```bash
pip install demucs basic-pitch torch torchaudio librosa
```
