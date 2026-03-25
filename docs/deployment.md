# Развёртывание DocSort AI

## Системные требования

### Минимальные (Вариант A — Правила)
- **ОС**: Linux (Ubuntu 20.04+, Astra Linux, ALT Linux), Windows 10+, macOS 12+
- **CPU**: 2 ядра
- **RAM**: 2 GB
- **Диск**: 500 MB
- **Python**: 3.10+
- **Tesseract OCR**: 4.0+

### Рекомендуемые (Гибридный — A+B+D)
- **CPU**: 4+ ядра
- **RAM**: 8 GB
- **Диск**: 5 GB
- **GPU**: Опционально (для Вариантов C/D)

## Установка

### 1. Из исходного кода

```bash
# Клонирование
git clone https://github.com/your-org/docsort-ai.git
cd docsort-ai

# Создание виртуального окружения
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Установка зависимостей
pip install -r requirements.txt

# Установка Tesseract OCR
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr tesseract-ocr-rus
# Astra Linux:
sudo apt-get install tesseract-ocr tesseract-ocr-rus
# macOS:
# brew install tesseract tesseract-lang

# Создание директорий
mkdir -p data/inbox data/sorted logs models

# Запуск API
PYTHONPATH=src python -m uvicorn api.server:app --host 0.0.0.0 --port 8000

# Запуск UI (в другом терминале)
PYTHONPATH=src python -m streamlit run src/ui/streamlit_app.py
```

### 2. Docker (рекомендуемый)

```bash
# Сборка и запуск всех сервисов
docker compose up -d

# С поддержкой LLM (Вариант D)
docker compose --profile llm up -d

# Загрузка модели LLM
docker exec docsort-ollama ollama pull qwen2.5:3b
```

### 3. Только API

```bash
docker compose up -d api
```

## Конфигурация

### Основной файл: `config/settings.yaml`

Ключевые параметры:

```yaml
# Активный классификатор
classification:
  active_classifier: "hybrid"  # rules, ml, neural, llm, hybrid
  auto_sort_threshold: 0.85    # Порог автосортировки
  cascade_threshold: 0.7       # Порог каскада

# Сборщик документов
collectors:
  folder_watcher:
    enabled: true
    watch_dir: "data/inbox"
  email:
    enabled: false
    host: "imap.example.com"
```

### Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|-------------|
| `DOCSORT_CONFIG` | Путь к файлу конфигурации | `config/settings.yaml` |
| `DOCSORT_LOG_LEVEL` | Уровень логирования | `INFO` |
| `DOCSORT_API_URL` | URL API-сервера | `http://localhost:8000` |

## Обучение ML-модели

```bash
# Подготовка данных: разложите документы по папкам
# data/training/
# ├── счёт-фактура/
# ├── акт/
# ├── договор/
# ├── накладная/
# └── платёжное_поручение/

# Запуск обучения
PYTHONPATH=src python scripts/train_model.py \
  --data-dir data/training \
  --output models/tfidf_svm.pkl

# Генерация тестовых данных
PYTHONPATH=src python scripts/generate_samples.py \
  --output-dir data/training \
  --samples-per-type 100
```

## Мониторинг

### Логи
```bash
# Просмотр логов API
tail -f logs/docsort.log

# Docker логи
docker compose logs -f api
```

### Проверка здоровья
```bash
curl http://localhost:8000/api/v1/health
```

### Статистика
```bash
curl http://localhost:8000/api/v1/stats
```

## Обновление

```bash
git pull
docker compose build
docker compose up -d
```

## Резервное копирование

Важные данные для бэкапа:
- `data/feedback.db` — база обратной связи
- `models/` — обученные модели
- `config/` — конфигурация

```bash
# Создание бэкапа
tar czf docsort-backup-$(date +%Y%m%d).tar.gz \
  data/feedback.db models/ config/
```

## Безопасность

- Система работает полностью офлайн
- Данные не передаются во внешние сервисы
- Все модели запускаются локально
- Соответствие 152-ФЗ «О персональных данных»
- Рекомендуется ограничить сетевой доступ к API файрволом
