# DocSort AI

**Система автоматической классификации и сортировки сканированных документов**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)

DocSort AI — это полностью офлайн система для автоматической классификации и сортировки сканированных документов. Предназначена для российского бизнеса: счета-фактуры, акты, договоры, накладные ТОРГ-12, платёжные поручения.

## Возможности

- **OCR** — распознавание текста из сканов (Tesseract OCR, русский + английский)
- **Каскадная классификация** — от быстрых правил до LLM для сложных случаев
- **Автосортировка** — раскладывает документы по папкам с правильными именами
- **Самообучение** — учится на подтверждениях пользователя
- **Полный офлайн** — данные не покидают контур организации
- **REST API** — интеграция с любыми системами
- **Web UI** — удобный интерфейс на Streamlit

## Типы документов

| Тип | Точность (гибрид) | Ключевые поля |
|-----|-------------------|---------------|
| Счёт-фактура | 95%+ | ИНН, КПП, Сумма |
| Акт выполненных работ | 93%+ | Исполнитель, Заказчик |
| Договор | 90%+ | Предмет договора, Стороны |
| Товарная накладная ТОРГ-12 | 95%+ | Грузоотправитель, Грузополучатель |
| Платёжное поручение | 96%+ | БИК, Банк получателя |

## Архитектура

Система использует каскадный подход — сначала лёгкие и быстрые методы, затем при необходимости более точные:

```
Документ → Предобработка → OCR → Классификация → Сортировка
                                       │
                          ┌─────────────┼─────────────┐
                          ▼             ▼             ▼
                     Правила(A)      ML(B)        LLM(D)
                     <1 сек        1-2 сек       5-30 сек
                     70-85%        85-92%        95-99%
```

### Варианты классификаторов

| Вариант | Метод | Точность | RAM | GPU |
|---------|-------|----------|-----|-----|
| A — Правила | Regex + ключевые слова | 70-85% | 1-2 GB | Нет |
| B — ML | TF-IDF + SVM | 85-92% | 2-4 GB | Нет |
| C — Нейросеть | MobileNet + SBERT | 92-97% | 4-8 GB | Желателен |
| D — LLM | Ollama (Qwen2.5/Phi-3) | 95-99% | 16+ GB | Желателен |
| **Гибрид** | **A → B → D каскад** | **90-97%** | **2-4 GB** | **Нет** |

## Быстрый старт

### Установка

```bash
git clone https://github.com/your-org/docsort-ai.git
cd docsort-ai

# Виртуальное окружение
python -m venv .venv
source .venv/bin/activate

# Зависимости
pip install -r requirements.txt

# Tesseract OCR (Ubuntu/Debian/Astra Linux)
sudo apt-get install tesseract-ocr tesseract-ocr-rus

# Создание директорий
mkdir -p data/inbox data/sorted logs models
```

### Запуск

```bash
# API-сервер
PYTHONPATH=src uvicorn api.server:app --host 0.0.0.0 --port 8000

# Web-интерфейс (в другом терминале)
PYTHONPATH=src streamlit run src/ui/streamlit_app.py

# Наблюдатель за папкой (в другом терминале)
PYTHONPATH=src python -m core.pipeline
```

### Docker (рекомендуемый способ)

```bash
# Запуск всех сервисов
docker compose up -d

# С поддержкой LLM
docker compose --profile llm up -d
docker exec docsort-ollama ollama pull qwen2.5:3b
```

**Сервисы:**
- API: http://localhost:8000
- UI: http://localhost:8501
- API Docs: http://localhost:8000/docs

## Использование

### Классификация через API

```bash
# Классификация одного документа
curl -X POST http://localhost:8000/api/v1/classify \
  -F "file=@document.pdf"

# Ответ:
# {
#   "doc_type": "invoice",
#   "doc_type_name": "Счёт-фактура",
#   "confidence": 0.94,
#   "classifier": "hybrid",
#   "extracted_fields": {
#     "inn": "7707083893",
#     "kpp": "770701001",
#     "amount": "150000.00"
#   }
# }
```

### Пакетная классификация

```bash
curl -X POST http://localhost:8000/api/v1/classify/batch \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf" \
  -F "files=@doc3.pdf"
```

### Обратная связь

```bash
# Исправить классификацию
curl -X POST http://localhost:8000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "abc-123",
    "predicted_type": "contract",
    "correct_type": "act",
    "comment": "Это акт, не договор"
  }'
```

### Автоматическая сортировка

Положите документы в `data/inbox/` — система автоматически:
1. Распознает текст (OCR)
2. Классифицирует документ
3. Переместит в `data/sorted/{тип}/{год-месяц}/{файл}`

### Python SDK

```python
from core.pipeline import DocumentPipeline
from core.config import AppConfig

config = AppConfig.load("config/settings.yaml")
pipeline = DocumentPipeline(config)

result = await pipeline.process_file("path/to/document.pdf")
print(f"Тип: {result.classification.doc_type}")
print(f"Уверенность: {result.classification.confidence:.0%}")
```

## API документация

### Эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/v1/classify` | Классификация документа |
| `POST` | `/api/v1/classify/batch` | Пакетная классификация |
| `GET` | `/api/v1/documents` | Список обработанных документов |
| `GET` | `/api/v1/documents/{id}` | Детали документа |
| `POST` | `/api/v1/feedback` | Отправка обратной связи |
| `GET` | `/api/v1/stats` | Статистика классификации |
| `GET` | `/api/v1/health` | Проверка здоровья сервиса |

Интерактивная документация: http://localhost:8000/docs

## Обучение ML-модели

```bash
# Генерация тестовых данных
PYTHONPATH=src python scripts/generate_samples.py \
  --output-dir data/training \
  --samples-per-type 100

# Обучение модели
PYTHONPATH=src python scripts/train_model.py \
  --data-dir data/training \
  --output models/tfidf_svm.pkl
```

## Конфигурация

Основной файл: `config/settings.yaml`

```yaml
classification:
  active_classifier: "hybrid"    # rules, ml, neural, llm, hybrid
  auto_sort_threshold: 0.85      # Порог автосортировки
  cascade_threshold: 0.7         # Порог каскада

collectors:
  folder_watcher:
    enabled: true
    watch_dir: "data/inbox"
  email:
    enabled: false
    host: "imap.example.com"
```

Правила классификации: `config/classification_rules.yaml`

## Структура проекта

```
docsort-ai/
├── config/                 # Конфигурация
│   ├── settings.yaml       # Основные настройки
│   └── classification_rules.yaml  # Правила классификации
├── src/
│   ├── core/               # Ядро: модели, конфиг, пайплайн
│   ├── collectors/          # Сборщики: email, папка
│   ├── ocr/                # OCR: Tesseract, предобработка
│   ├── classifiers/        # Классификаторы: правила, ML, нейросеть, LLM
│   ├── sorter/             # Сортировка файлов
│   ├── feedback/           # Обратная связь и обучение
│   ├── api/                # REST API (FastAPI)
│   └── ui/                 # Web UI (Streamlit)
├── tests/                  # Тесты
├── scripts/                # Скрипты обучения
├── docs/                   # Документация
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Тестирование

```bash
# Запуск всех тестов
PYTHONPATH=src pytest tests/ -v

# С покрытием
PYTHONPATH=src pytest tests/ -v --cov=src --cov-report=html

# Только тесты классификатора
PYTHONPATH=src pytest tests/test_variant_a.py -v
```

## Совместимые ОС

- Ubuntu 20.04+ / Debian 11+
- Astra Linux SE / CE
- ALT Linux
- Windows 10+ (WSL2 рекомендуется)
- macOS 12+

## Соответствие стандартам

- **152-ФЗ** — О персональных данных (полный офлайн)
- **149-ФЗ** — Об информации и информационных технологиях
- **ГОСТ 34.602-2020** — ТЗ на автоматизированные системы
- **ГОСТ Р 57580.1-2017** — Информационная безопасность

## Лицензия

Этот проект распространяется под лицензией MIT. Подробности в файле [LICENSE](LICENSE).
