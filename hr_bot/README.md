# 🧠 HR Match Analyzer API

**HR Match Analyzer** — это асинхронное FastAPI-приложение для анализа соответствия **резюме и вакансий** с использованием **эмбеддингов**, **DeBERTa**, и **LLM** (Gemini / Ollama).  
Система оценивает релевантность кандидата, выявляет несоответствия и формирует уточняющие вопросы, чтобы улучшить итоговую оценку.

---

## 🚀 Возможности

- 🔍 **Анализ соответствия** кандидата и вакансии по ключевым полям (город, опыт, образование, язык, занятость и др.)
- 🧩 **Выявление “gap’ов”** — несоответствий между резюме и вакансией
- 🤖 **Генерация уточняющих вопросов** с помощью LLM (Gemini / Ollama / Mistral)
- 🗣️ **Интерпретация ответов** кандидата и пересчёт финального скора
- ⚡ **Быстрая работа** благодаря асинхронному FastAPI и кешированию эмбеддингов

---

## 🧱 Структура проекта

```
.
├── api_server.py
├── llm_client.py
├── embeddings.py
├── scoring.py
├── qa_logic.py
├── deberta_analyzer.py
├── main.py
├── requirements.txt
└── docker-compose.yml

```

---

## ⚙️ Установка и запуск (локально)

### Клонировать репозиторий
```bash
git clone https://github.com/username/hr-match-analyzer.git
cd hr-match-analyzer
```

### Создать виртуальное окружение
```bash
python -m venv venv
source venv/bin/activate    # (Linux/Mac)
venv\Scripts\activate       # (Windows)
```

### Установить зависимости
```bash
pip install -r requirements.txt
```

### Указать переменные окружения
Создай файл `.env` в корне проекта:
```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-1.5-flash
GOOGLE_API_KEY=your_google_api_key

EMBEDDINGS_MODE=local
EMBEDDINGS_LOCAL_MODEL=all-MiniLM-L6-v2
OPENAI_API_KEY=your_openai_key   # только если EMBEDDINGS_MODE=openai
```

---

## 🐳 Запуск через Docker

```bash
docker-compose up --build
```

После сборки API будет доступен по адресу:  
👉 **http://localhost:8000**

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)  
ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🧠 Основные эндпоинты

| Метод | Путь | Описание |
|-------|------|-----------|
| `GET` | `/` | Проверка работы сервера |
| `POST` | `/analyze/base` | Базовый анализ и скоринг |
| `POST` | `/analyze/followup` | Генерация уточняющих вопросов |
| `POST` | `/analyze/final` | Пересчёт финального скора после ответов |

---

## 📘 Пример запроса

```json
POST /analyze/base
{
  "vacancy": {
    "city": "Москва",
    "experience": 5,
    "post": "Python Developer",
    "education": "Высшее техническое",
    "languages": ["русский", "английский"],
    "salary": 200000,
    "busyness": "полная"
  },
  "resume": {
    "city": "Казань",
    "experience": 3,
    "post": "Backend Developer",
    "education": "Бакалавр информатики",
    "languages": ["русский"],
    "salary": 250000,
    "busyness": "полная",
    "мотивация": "Хочу развиваться в backend-разработке"
  }
}
```

---

## 📦 Пример docker-compose.yml

```yaml
version: "3.9"

services:
  hr_api:
    build: .
    container_name: hr-match-analyzer
    command: uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - .:/app
```

---

## 🧑‍💻 Авторы

**ATOMIX TEAM**  
💬 Telegram: [@Dexarim](https://t.me/Dexarim)
              [@B_l_a_c_kS_u_n](https://t.me/B_l_a_c_kS_u_n)

---

## 📄 Лицензия

Проект распространяется под лицензией **MIT**.
