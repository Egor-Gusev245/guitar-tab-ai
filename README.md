# 🎸 AI Guitar Tab Transcriber (MVP)

Веб-приложение для автоматического выделения гитарной дорожки из аудиозаписи (с помощью ИИ **Demucs**), распознавания нот и аккордов, генерации интерактивной гитарной табулатуры и удобного Личного кабинета пользователя.

---

## 🌟 Основные возможности (MVP)

1. **Разделение трека (Stem Separation)**:
   - Изоляция гитарной дорожки с помощью нейросети `Demucs` (модель `htdemucs_6s`, отделяющая гитару от баса, ударных и вокала).
2. **Распознавание и транскрибация (Audio-to-Tab)**:
   - Распознавание нот, ритмики и длительностей с помощью `Basic Pitch`.
   - Оптимизация аппликатуры: автоматический расчет наиболее эргономичной позиции на грифе гитары (струна + лад).
3. **Личный кабинет**:
   - Авторизация (вход по никнейму и паролю, регистрация через email).
   - Список сгенерированных табулатур и статус обработки треков.
4. **Интерактивный плеер табулатур**:
   - Отображение и проигрывание табулатуры прямо в браузере (библиотека `alphaTab`).
   - Экспорт в `TXT` (ASCII tab) и `MusicXML` (Guitar Pro, MuseScore).

---

## 🏗 Архитектура проекта

```text
guitar-tab-ai/
├── backend/                  # FastAPI сервис (REST API, Auth, Task Queue)
│   ├── app/
│   │   ├── api/              # Маршруты: /auth, /users, /tabs
│   │   ├── core/             # JWT, настройки, безопасность
│   │   ├── models/           # Модели базы данных (SQLAlchemy)
│   │   ├── schemas/          # Pydantic схемы валидации
│   │   └── main.py           # Точка входа FastAPI
│   └── requirements.txt
├── ml_pipeline/              # Модули ИИ и аудио-обработки
│   ├── demucs_runner.py      # Выделение дорожки гитары
│   ├── transcriber.py        # Распознавание нот
│   └── fretboard_mapper.py   # Построение табулатуры по ладам/струнам
├── frontend/                 # Клиентская часть (React + TypeScript + Vite + Tailwind)
│   ├── src/
│   │   ├── components/       # Интерактивный плеер alphaTab, uploader
│   │   └── pages/            # Dashboard (Личный кабинет), Login, Register
│   └── package.json
├── docker-compose.yml        # Docker сборка для быстрого развертывания
├── .gitignore
└── README.md
```

---

## 🚀 Быстрый старт (Разработка)

### Требования
- Python 3.10+
- Node.js 18+
- Git
- FFmpeg (необходим для обработки аудио в Demucs)

### 1. Клонирование репозитория
```bash
git clone <URL_РЕПОЗИТОРИЯ>
cd guitar-tab-ai
```

### 2. Запуск Backend
```bash
cd backend
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Документация API будет доступна по адресу: `http://localhost:8000/docs`.

### 3. Запуск Frontend
```bash
cd ../frontend
npm install
npm run dev
```
Интерфейс будет доступен по адресу: `http://localhost:5173`.

---

## 🗺 Дорожная карта (Roadmap)

- [x] Инициализация проекта и архитектурный план
- [ ] Модуль аутентификации (JWT, регистрация, вход)
- [ ] Очередь фоновой обработки аудио
- [ ] Интеграция Demucs (`htdemucs_6s`)
- [ ] Конвертер нот в табулатуру (Fretboard mapper)
- [ ] Интерактивный просмотрщик `alphaTab` в Личном кабинете
- [ ] Экспорт в MusicXML / GPX

---

## 📄 Лицензия
MIT License
