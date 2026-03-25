#!/usr/bin/env python3
"""Скрипт обучения ML-модели классификации документов.

Загружает тренировочные данные из структурированной директории (одна папка
на каждый тип документа), выполняет OCR-распознавание, обучает модель
TF-IDF + SVM и сохраняет её для последующего использования в пайплайне.

Структура директории тренировочных данных::

    training_data/
    ├── invoice/
    │   ├── doc_001.pdf
    │   └── doc_002.png
    ├── act/
    │   ├── doc_001.pdf
    │   └── doc_002.txt
    ├── contract/
    │   └── ...
    ├── waybill/
    │   └── ...
    └── payment_order/
        └── ...

Пример запуска::

    python scripts/train_model.py --data-dir data/training --output models/tfidf_svm.pkl
    python scripts/train_model.py --data-dir data/training --output models/tfidf_svm.pkl --verbose
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# ------------------------------------------------------------------
# Типы документов
# ------------------------------------------------------------------

DOCUMENT_TYPES: tuple[str, ...] = (
    "invoice",
    "act",
    "contract",
    "waybill",
    "payment_order",
)

#: Расширения файлов, поддерживаемые для обучения.
SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".txt"}

logger = logging.getLogger("train_model")


# ------------------------------------------------------------------
# Загрузка данных
# ------------------------------------------------------------------


def extract_text_from_file(file_path: Path) -> str:
    """Извлекает текст из файла документа.

    Для текстовых файлов (.txt) — читает содержимое напрямую.
    Для PDF — извлекает текст через PyMuPDF.
    Для изображений — выполняет OCR через Tesseract.

    Args:
        file_path: Путь к файлу документа.

    Returns:
        Извлечённый текст документа.

    Raises:
        ValueError: Если формат файла не поддерживается.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8")

    if suffix == ".pdf":
        return _extract_text_from_pdf(file_path)

    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return _extract_text_from_image(file_path)

    raise ValueError(f"Формат файла не поддерживается: {suffix}")


def _extract_text_from_pdf(file_path: Path) -> str:
    """Извлекает текст из PDF-файла с использованием PyMuPDF.

    Args:
        file_path: Путь к PDF-файлу.

    Returns:
        Текст из всех страниц PDF.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning(
            "PyMuPDF (fitz) не установлен. Установите: pip install pymupdf"
        )
        return ""

    text_parts: list[str] = []
    try:
        with fitz.open(str(file_path)) as doc:
            for page in doc:
                text_parts.append(page.get_text())
    except Exception as exc:
        logger.error("Ошибка извлечения текста из PDF %s: %s", file_path, exc)
        return ""

    return "\n".join(text_parts)


def _extract_text_from_image(file_path: Path) -> str:
    """Выполняет OCR-распознавание текста из изображения.

    Args:
        file_path: Путь к файлу изображения.

    Returns:
        Распознанный текст.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        logger.warning(
            "pytesseract и/или Pillow не установлены. "
            "Установите: pip install pytesseract Pillow"
        )
        return ""

    try:
        image = Image.open(file_path)
        text: str = pytesseract.image_to_string(image, lang="rus+eng")
        return text
    except Exception as exc:
        logger.error("Ошибка OCR для %s: %s", file_path, exc)
        return ""


def load_training_data(data_dir: Path) -> tuple[list[str], list[str]]:
    """Загружает тренировочные данные из структурированной директории.

    Каждая поддиректория соответствует типу документа. Файлы внутри
    поддиректории используются как образцы данного типа.

    Args:
        data_dir: Корневая директория с тренировочными данными.

    Returns:
        Кортеж (тексты, метки) — списки одинаковой длины.

    Raises:
        FileNotFoundError: Если директория не существует.
        ValueError: Если данных недостаточно для обучения.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Директория не найдена: {data_dir}")

    texts: list[str] = []
    labels: list[str] = []
    stats: dict[str, int] = {}

    for doc_type in DOCUMENT_TYPES:
        type_dir = data_dir / doc_type
        if not type_dir.exists():
            logger.warning("Директория для типа '%s' не найдена: %s", doc_type, type_dir)
            continue

        count = 0
        for file_path in sorted(type_dir.iterdir()):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                logger.debug("Пропущен файл с неподдерживаемым расширением: %s", file_path)
                continue

            logger.info("Обработка: %s", file_path)
            text = extract_text_from_file(file_path)

            if not text.strip():
                logger.warning("Пустой текст после обработки: %s", file_path)
                continue

            texts.append(text)
            labels.append(doc_type.upper())
            count += 1

        stats[doc_type] = count
        logger.info("Тип '%s': загружено %d документов", doc_type, count)

    logger.info("Общая статистика: %s", stats)
    logger.info("Всего образцов: %d", len(texts))

    if len(texts) < 2:
        raise ValueError(
            f"Недостаточно данных для обучения: найдено {len(texts)} образцов, "
            f"необходимо минимум 2"
        )

    unique_labels = set(labels)
    if len(unique_labels) < 2:
        raise ValueError(
            f"Недостаточно классов для обучения: найдено {len(unique_labels)}, "
            f"необходимо минимум 2"
        )

    return texts, labels


# ------------------------------------------------------------------
# Обучение модели
# ------------------------------------------------------------------


def build_pipeline() -> Pipeline:
    """Создаёт пайплайн TF-IDF + SVM для классификации документов.

    Returns:
        Необученный пайплайн sklearn.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            sublinear_tf=True,
            strip_accents="unicode",
            min_df=1,
            max_df=0.95,
        )),
        ("svm", LinearSVC(
            max_iter=20000,
            C=1.0,
            class_weight="balanced",
            loss="squared_hinge",
            dual="auto",
        )),
    ])


def train_model(
    texts: list[str],
    labels: list[str],
    test_size: float = 0.2,
) -> tuple[Pipeline, dict[str, Any]]:
    """Обучает модель классификации и возвращает метрики.

    Args:
        texts: Список текстов документов.
        labels: Список меток типов документов.
        test_size: Доля тестовой выборки (от 0.0 до 1.0).

    Returns:
        Кортеж (обученный пайплайн, словарь с метриками).
    """
    pipeline = build_pipeline()

    metrics: dict[str, Any] = {
        "n_samples": len(texts),
        "n_classes": len(set(labels)),
        "classes": sorted(set(labels)),
    }

    # Если данных достаточно для разделения, выполняем train/test split
    if len(texts) >= 10 and len(set(labels)) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels,
            test_size=test_size,
            random_state=42,
            stratify=labels,
        )

        pipeline.fit(X_train, y_train)

        # Оценка на тестовой выборке
        y_pred = pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics["classification_report"] = report
        metrics["test_accuracy"] = report["accuracy"]  # type: ignore[index]
        metrics["train_size"] = len(X_train)
        metrics["test_size"] = len(X_test)

        logger.info("Точность на тестовой выборке: %.4f", metrics["test_accuracy"])

        # Кросс-валидация (если достаточно данных)
        if len(texts) >= 20:
            n_splits = min(5, len(set(labels)))
            cv_scores = cross_val_score(
                build_pipeline(), texts, labels,
                cv=n_splits,
                scoring="accuracy",
            )
            metrics["cv_mean_accuracy"] = float(np.mean(cv_scores))
            metrics["cv_std_accuracy"] = float(np.std(cv_scores))
            logger.info(
                "Кросс-валидация: %.4f (+/- %.4f)",
                metrics["cv_mean_accuracy"],
                metrics["cv_std_accuracy"],
            )

        # Переобучаем на полных данных
        pipeline.fit(texts, labels)
    else:
        # Недостаточно данных для split, обучаем на всём
        logger.warning(
            "Недостаточно данных для train/test split (%d образцов). "
            "Обучение на полном наборе данных.",
            len(texts),
        )
        pipeline.fit(texts, labels)
        metrics["test_accuracy"] = None

    return pipeline, metrics


def save_model(pipeline: Pipeline, output_path: Path) -> None:
    """Сохраняет обученную модель в файл.

    Args:
        pipeline: Обученный пайплайн sklearn.
        output_path: Путь для сохранения модели.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = output_path.stat().st_size
    logger.info(
        "Модель сохранена: %s (%.2f КБ)",
        output_path,
        file_size / 1024,
    )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы командной строки.

    Args:
        argv: Список аргументов. Если None, используется sys.argv.

    Returns:
        Разобранные аргументы.
    """
    parser = argparse.ArgumentParser(
        description="Обучение ML-модели классификации документов DocSort AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s --data-dir data/training --output models/tfidf_svm.pkl
  %(prog)s --data-dir data/training --output models/model.pkl --test-size 0.3
  %(prog)s --data-dir data/training --output models/model.pkl --verbose
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Директория с тренировочными данными (одна поддиректория на тип документа)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/tfidf_svm.pkl"),
        help="Путь для сохранения обученной модели (по умолчанию: models/tfidf_svm.pkl)",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Доля тестовой выборки (по умолчанию: 0.2)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Подробный вывод (уровень DEBUG)",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Главная функция скрипта обучения.

    Args:
        argv: Аргументы командной строки. Если None, используется sys.argv.

    Returns:
        Код возврата: 0 — успех, 1 — ошибка.
    """
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("DocSort AI — Обучение модели классификации")
    logger.info("=" * 60)
    logger.info("Директория данных: %s", args.data_dir)
    logger.info("Выходной файл модели: %s", args.output)
    logger.info("Доля тестовой выборки: %.1f%%", args.test_size * 100)

    try:
        # Загрузка данных
        start_time = time.time()
        logger.info("Загрузка тренировочных данных...")
        texts, labels = load_training_data(args.data_dir)
        load_time = time.time() - start_time
        logger.info("Данные загружены за %.2f сек.", load_time)

        # Обучение модели
        logger.info("Обучение модели TF-IDF + SVM...")
        train_start = time.time()
        pipeline, metrics = train_model(texts, labels, test_size=args.test_size)
        train_time = time.time() - train_start
        logger.info("Модель обучена за %.2f сек.", train_time)

        # Вывод метрик
        logger.info("-" * 40)
        logger.info("Метрики обучения:")
        logger.info("  Количество образцов: %d", metrics["n_samples"])
        logger.info("  Количество классов: %d", metrics["n_classes"])
        logger.info("  Классы: %s", ", ".join(metrics["classes"]))

        if metrics.get("test_accuracy") is not None:
            logger.info("  Точность на тесте: %.4f", metrics["test_accuracy"])

        if "cv_mean_accuracy" in metrics:
            logger.info(
                "  Кросс-валидация: %.4f (+/- %.4f)",
                metrics["cv_mean_accuracy"],
                metrics["cv_std_accuracy"],
            )

        if "classification_report" in metrics:
            report = metrics["classification_report"]
            logger.info("-" * 40)
            logger.info("Отчёт классификации:")
            for class_name, class_metrics in report.items():
                if isinstance(class_metrics, dict):
                    logger.info(
                        "  %s: precision=%.3f recall=%.3f f1=%.3f",
                        class_name,
                        class_metrics.get("precision", 0),
                        class_metrics.get("recall", 0),
                        class_metrics.get("f1-score", 0),
                    )

        # Сохранение модели
        save_model(pipeline, args.output)

        total_time = time.time() - start_time
        logger.info("-" * 40)
        logger.info("Обучение завершено успешно за %.2f сек.", total_time)
        return 0

    except FileNotFoundError as exc:
        logger.error("Ошибка: %s", exc)
        return 1
    except ValueError as exc:
        logger.error("Ошибка данных: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Неожиданная ошибка: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
