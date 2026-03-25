FROM python:3.11-slim

LABEL maintainer="DocSort AI Team"
LABEL description="DocSort AI — Система автоматической классификации документов"

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем файлы зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY . .

# Создаём необходимые директории
RUN mkdir -p data/inbox data/sorted logs models

# Порты: 8000 (API), 8501 (Streamlit)
EXPOSE 8000 8501

# Переменные окружения
ENV PYTHONPATH=/app/src
ENV DOCSORT_CONFIG=/app/config/settings.yaml

# Запуск API-сервера
CMD ["python", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
