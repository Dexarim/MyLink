# Используем Python 3.8
FROM python:3.8-slim

# Устанавливаем системные зависимости
RUN apt-get update && \
    apt-get install -y git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Задаём рабочую директорию
WORKDIR /app

# Копируем и устанавливаем зависимости отдельно (для кэширования pip)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Создаём директорию для Hugging Face кэша
RUN mkdir -p /root/.cache/huggingface

# Указываем порт FastAPI
EXPOSE 8000

# Запускаем сервер
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
