"""
Конфигурация приложения DocSort AI.

Загружает настройки из YAML-файла (config/settings.yaml) и предоставляет
типизированный доступ через Pydantic-модели. Реализует паттерн Singleton
для единственного экземпляра конфигурации.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import structlog
import yaml
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger(__name__)

# Корень проекта — два уровня вверх от этого файла (src/core/config.py -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"


# ---------------------------------------------------------------------------
# Секции конфигурации
# ---------------------------------------------------------------------------


class GeneralConfig(BaseModel):
    """Общие настройки приложения."""

    language: str = Field(
        default="rus+eng",
        description="Язык OCR (rus, eng, rus+eng)",
    )
    log_level: str = Field(
        default="INFO",
        description="Уровень логирования: DEBUG, INFO, WARNING, ERROR",
    )
    log_dir: str = Field(
        default="logs",
        description="Директория для файлов логов",
    )
    max_workers: int = Field(
        default=4,
        ge=1,
        description="Максимальное количество потоков обработки",
    )

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        """Проверяет допустимость уровня логирования."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"Недопустимый уровень логирования: {v}. Допустимые: {allowed}")
        return upper


class EmailCollectorConfig(BaseModel):
    """Настройки IMAP-сборщика электронной почты."""

    enabled: bool = False
    host: str = "imap.example.com"
    port: int = Field(default=993, ge=1, le=65535)
    use_ssl: bool = True
    username: str = ""
    password: str = ""
    folder: str = "INBOX"
    poll_interval: int = Field(
        default=60,
        ge=10,
        description="Интервал проверки почты в секундах",
    )
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".tif", ".tiff", ".png", ".jpg", ".jpeg"],
    )


class FolderWatcherConfig(BaseModel):
    """Настройки наблюдателя за папкой."""

    enabled: bool = True
    watch_dir: str = "data/inbox"
    recursive: bool = False
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".tif", ".tiff", ".png", ".jpg", ".jpeg"],
    )


class CollectorsConfig(BaseModel):
    """Настройки всех сборщиков документов."""

    email: EmailCollectorConfig = Field(default_factory=EmailCollectorConfig)
    folder_watcher: FolderWatcherConfig = Field(default_factory=FolderWatcherConfig)


class OCRConfig(BaseModel):
    """Настройки OCR-движка."""

    engine: str = Field(
        default="tesseract",
        description="OCR-движок: tesseract",
    )
    tesseract_cmd: str = Field(
        default="",
        description="Путь к бинарнику Tesseract (пусто = системный PATH)",
    )
    languages: str = Field(
        default="rus+eng",
        description="Языки распознавания",
    )
    dpi: int = Field(
        default=300,
        ge=72,
        le=1200,
        description="DPI для конвертации PDF в изображение",
    )
    psm: int = Field(
        default=6,
        ge=0,
        le=13,
        description="PSM-режим Tesseract",
    )


class PreprocessingConfig(BaseModel):
    """Настройки предобработки изображений."""

    deskew: bool = True
    denoise: bool = True
    binarize: bool = True
    enhance_contrast: bool = True
    target_dpi: int = Field(default=300, ge=72, le=1200)


class RulesClassifierConfig(BaseModel):
    """Настройки классификатора на правилах."""

    enabled: bool = True


class MLClassifierConfig(BaseModel):
    """Настройки ML-классификатора (TF-IDF + SVM)."""

    enabled: bool = True
    model_path: str = "models/tfidf_svm.pkl"
    min_training_samples: int = Field(
        default=50,
        ge=1,
        description="Минимальное количество образцов для обучения",
    )


class NeuralClassifierConfig(BaseModel):
    """Настройки нейросетевого классификатора."""

    enabled: bool = False
    visual_model_path: str = "models/mobilenet_v3.onnx"
    text_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class LLMClassifierConfig(BaseModel):
    """Настройки LLM-классификатора."""

    enabled: bool = False
    ollama_url: str = "http://localhost:11434"
    model: str = "qwen2.5:3b"
    timeout: int = Field(
        default=120,
        ge=1,
        description="Таймаут запроса в секундах",
    )


class ClassificationConfig(BaseModel):
    """Настройки классификации документов."""

    active_classifier: str = Field(
        default="hybrid",
        description="Активный классификатор: rules, ml, neural, llm, hybrid",
    )
    auto_sort_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Порог уверенности для автоматической сортировки",
    )
    cascade_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Порог для передачи на следующий уровень каскада",
    )
    rules: RulesClassifierConfig = Field(default_factory=RulesClassifierConfig)
    ml: MLClassifierConfig = Field(default_factory=MLClassifierConfig)
    neural: NeuralClassifierConfig = Field(default_factory=NeuralClassifierConfig)
    llm: LLMClassifierConfig = Field(default_factory=LLMClassifierConfig)

    @field_validator("active_classifier")
    @classmethod
    def _validate_classifier(cls, v: str) -> str:
        """Проверяет допустимость имени классификатора."""
        allowed = {"rules", "ml", "neural", "llm", "hybrid"}
        if v not in allowed:
            raise ValueError(f"Недопустимый классификатор: {v}. Допустимые: {allowed}")
        return v


class SortingConfig(BaseModel):
    """Настройки сортировки файлов."""

    output_dir: str = "data/sorted"
    filename_template: str = "{date}_{doc_type}_{original_name}"
    create_type_dirs: bool = True
    create_date_dirs: bool = True


class FeedbackConfig(BaseModel):
    """Настройки обратной связи и переобучения."""

    db_path: str = "data/feedback.db"
    retrain_threshold: int = Field(
        default=20,
        ge=1,
        description="Количество подтверждений до автоматического переобучения",
    )
    store_history: bool = True


class APIConfig(BaseModel):
    """Настройки FastAPI-сервера."""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    max_upload_size: int = Field(
        default=52_428_800,
        ge=1,
        description="Максимальный размер загружаемого файла в байтах",
    )


class UIConfig(BaseModel):
    """Настройки Streamlit UI."""

    port: int = Field(default=8501, ge=1, le=65535)


# ---------------------------------------------------------------------------
# Корневая модель конфигурации
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    """
    Корневая модель конфигурации DocSort AI.

    Загружается из YAML-файла config/settings.yaml.
    Объединяет все секции конфигурации в единую иерархию.
    """

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    collectors: CollectorsConfig = Field(default_factory=CollectorsConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    sorting: SortingConfig = Field(default_factory=SortingConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> AppConfig:
        """
        Загружает конфигурацию из YAML-файла.

        Args:
            path: Путь к YAML-файлу. Если не указан, используется
                  config/settings.yaml относительно корня проекта.

        Returns:
            Экземпляр AppConfig с загруженными настройками.

        Raises:
            FileNotFoundError: Если файл конфигурации не найден.
            yaml.YAMLError: Если файл содержит некорректный YAML.
        """
        config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

        if not config_path.exists():
            logger.warning(
                "config_file_not_found",
                path=str(config_path),
                message="Используются настройки по умолчанию",
            )
            return cls()

        logger.info("config_loading", path=str(config_path))

        with open(config_path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        config = cls.model_validate(raw)

        logger.info(
            "config_loaded",
            path=str(config_path),
            active_classifier=config.classification.active_classifier,
            log_level=config.general.log_level,
        )

        return config


# ---------------------------------------------------------------------------
# Singleton-доступ к конфигурации
# ---------------------------------------------------------------------------

_config_instance: AppConfig | None = None
_config_lock = threading.Lock()


def get_config(path: str | Path | None = None, *, reload: bool = False) -> AppConfig:
    """
    Возвращает глобальный экземпляр конфигурации (Singleton).

    При первом вызове загружает конфигурацию из YAML-файла.
    Последующие вызовы возвращают тот же экземпляр, если не указан ``reload=True``.

    Args:
        path: Путь к YAML-файлу конфигурации. Используется только при первой
              загрузке или при ``reload=True``.
        reload: Принудительно перезагрузить конфигурацию.

    Returns:
        Экземпляр AppConfig.
    """
    global _config_instance

    if _config_instance is not None and not reload:
        return _config_instance

    with _config_lock:
        # Повторная проверка под блокировкой (double-checked locking)
        if _config_instance is not None and not reload:
            return _config_instance

        _config_instance = AppConfig.from_yaml(path)
        return _config_instance


def reset_config() -> None:
    """
    Сбрасывает глобальный экземпляр конфигурации.

    Используется в тестах для изоляции между тест-кейсами.
    """
    global _config_instance

    with _config_lock:
        _config_instance = None
        logger.debug("config_reset")
