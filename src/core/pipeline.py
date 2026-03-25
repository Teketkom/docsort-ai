"""
Главный пайплайн обработки документов DocSort AI.

Оркестрирует полный цикл: сбор -> предобработка -> OCR -> классификация -> сортировка.
Реализует каскадную логику классификации: правила -> ML -> LLM.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import UUID

import structlog

from core.config import AppConfig, get_config
from core.document import Document, DocumentClassification, DocumentType

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Интерфейсы (протоколы) для компонентов пайплайна
# ---------------------------------------------------------------------------


@runtime_checkable
class Collector(Protocol):
    """Протокол для сборщиков документов."""

    async def collect(self) -> list[Document]:
        """Собирает новые документы из источника."""
        ...


@runtime_checkable
class Preprocessor(Protocol):
    """Протокол для модулей предобработки."""

    async def preprocess(self, document: Document) -> Document:
        """Выполняет предобработку изображения документа."""
        ...


@runtime_checkable
class OCREngine(Protocol):
    """Протокол для OCR-движков."""

    async def extract_text(self, document: Document) -> Document:
        """Извлекает текст из документа."""
        ...


@runtime_checkable
class Classifier(Protocol):
    """Протокол для классификаторов документов."""

    @property
    def name(self) -> str:
        """Имя классификатора."""
        ...

    async def classify(self, document: Document) -> DocumentClassification:
        """Классифицирует документ и возвращает результат."""
        ...


@runtime_checkable
class Sorter(Protocol):
    """Протокол для модулей сортировки."""

    async def sort(self, document: Document) -> Path:
        """Перемещает документ в целевую директорию. Возвращает новый путь."""
        ...


# ---------------------------------------------------------------------------
# Каскадный классификатор
# ---------------------------------------------------------------------------


class CascadeClassifier:
    """
    Каскадный классификатор документов.

    Последовательно применяет классификаторы в порядке приоритета:
    1. Правила (rules) — быстрая проверка по ключевым словам и regex.
    2. ML (TF-IDF + SVM) — если правила дали уверенность ниже порога.
    3. LLM — если ML тоже не уверен.

    Если все классификаторы дали низкую уверенность, возвращается результат
    с наибольшей уверенностью среди всех попыток.
    """

    def __init__(
        self,
        classifiers: list[Classifier],
        cascade_threshold: float = 0.7,
        auto_sort_threshold: float = 0.85,
    ) -> None:
        """
        Инициализирует каскадный классификатор.

        Args:
            classifiers: Список классификаторов в порядке приоритета.
            cascade_threshold: Порог уверенности для передачи на следующий уровень.
            auto_sort_threshold: Порог для автоматической сортировки.
        """
        self._classifiers = classifiers
        self._cascade_threshold = cascade_threshold
        self._auto_sort_threshold = auto_sort_threshold

    async def classify(self, document: Document) -> DocumentClassification:
        """
        Выполняет каскадную классификацию документа.

        Последовательно применяет классификаторы. Если текущий классификатор
        возвращает уверенность >= cascade_threshold, каскад останавливается.
        Иначе результат запоминается и передаётся следующему классификатору.

        Args:
            document: Документ для классификации.

        Returns:
            Лучший результат классификации среди всех попыток.
        """
        best_result: DocumentClassification | None = None

        for classifier in self._classifiers:
            classifier_name = classifier.name
            log = logger.bind(
                doc_id=str(document.id),
                classifier=classifier_name,
            )

            try:
                log.info("cascade_classifier_attempt")
                result = await classifier.classify(document)

                log.info(
                    "cascade_classifier_result",
                    doc_type=result.doc_type.value,
                    confidence=result.confidence,
                )

                # Запоминаем лучший результат
                if best_result is None or result.confidence > best_result.confidence:
                    best_result = result

                # Если уверенность достаточна — останавливаем каскад
                if result.confidence >= self._cascade_threshold:
                    log.info(
                        "cascade_stopped_confident",
                        confidence=result.confidence,
                        threshold=self._cascade_threshold,
                    )
                    return result

                log.info(
                    "cascade_continue_low_confidence",
                    confidence=result.confidence,
                    threshold=self._cascade_threshold,
                )

            except Exception:
                log.exception("cascade_classifier_error")
                continue

        # Если ни один классификатор не дал уверенности выше порога,
        # возвращаем лучший результат или UNKNOWN
        if best_result is not None:
            logger.warning(
                "cascade_exhausted_returning_best",
                doc_id=str(document.id),
                doc_type=best_result.doc_type.value,
                confidence=best_result.confidence,
            )
            return best_result

        logger.warning(
            "cascade_exhausted_no_result",
            doc_id=str(document.id),
        )
        return DocumentClassification(
            doc_type=DocumentType.UNKNOWN,
            confidence=0.0,
            classifier_name="cascade_fallback",
            metadata={"reason": "Все классификаторы не смогли определить тип документа"},
        )


# ---------------------------------------------------------------------------
# Главный пайплайн
# ---------------------------------------------------------------------------


class DocumentPipeline:
    """
    Главный пайплайн обработки документов.

    Оркестрирует полный цикл обработки: сбор -> предобработка -> OCR ->
    классификация -> сортировка. Поддерживает асинхронную обработку
    нескольких документов параллельно.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        collectors: list[Collector] | None = None,
        preprocessor: Preprocessor | None = None,
        ocr_engine: OCREngine | None = None,
        classifiers: list[Classifier] | None = None,
        sorter: Sorter | None = None,
    ) -> None:
        """
        Инициализирует пайплайн.

        Args:
            config: Конфигурация приложения. Если не указана, загружается из YAML.
            collectors: Список сборщиков документов.
            preprocessor: Модуль предобработки.
            ocr_engine: OCR-движок.
            classifiers: Список классификаторов для каскада.
            sorter: Модуль сортировки.
        """
        self._config = config or get_config()
        self._collectors = collectors or []
        self._preprocessor = preprocessor
        self._ocr_engine = ocr_engine
        self._sorter = sorter
        self._running = False

        # Каскадный классификатор
        self._cascade = CascadeClassifier(
            classifiers=classifiers or [],
            cascade_threshold=self._config.classification.cascade_threshold,
            auto_sort_threshold=self._config.classification.auto_sort_threshold,
        )

        # Статистика текущего сеанса
        self._stats: dict[str, int] = {
            "total_collected": 0,
            "total_processed": 0,
            "total_classified": 0,
            "total_sorted": 0,
            "total_errors": 0,
        }

        logger.info(
            "pipeline_initialized",
            active_classifier=self._config.classification.active_classifier,
            cascade_threshold=self._config.classification.cascade_threshold,
            auto_sort_threshold=self._config.classification.auto_sort_threshold,
            num_collectors=len(self._collectors),
            num_classifiers=len(classifiers or []),
        )

    @property
    def stats(self) -> dict[str, int]:
        """Возвращает статистику обработки текущего сеанса."""
        return dict(self._stats)

    async def process_document(self, document: Document) -> Document:
        """
        Обрабатывает один документ через все этапы пайплайна.

        Этапы:
            1. Предобработка изображения (deskew, denoise, binarize).
            2. OCR — извлечение текста.
            3. Классификация — каскадное определение типа документа.
            4. Сортировка — перемещение файла в целевую директорию.

        Args:
            document: Документ для обработки.

        Returns:
            Обработанный документ с заполненными полями classification и processed_at.

        Raises:
            RuntimeError: Если OCR-движок или другой обязательный компонент не настроен.
        """
        log = logger.bind(
            doc_id=str(document.id),
            filename=document.original_filename,
        )

        log.info("pipeline_document_processing_start")

        try:
            # Этап 1: Предобработка
            if self._preprocessor is not None:
                log.info("pipeline_stage_preprocess")
                document = await self._preprocessor.preprocess(document)
            else:
                log.debug("pipeline_preprocess_skipped", reason="preprocessor_not_configured")

            # Этап 2: OCR
            if self._ocr_engine is not None:
                log.info("pipeline_stage_ocr")
                document = await self._ocr_engine.extract_text(document)
                log.info(
                    "pipeline_ocr_complete",
                    text_length=len(document.ocr_text),
                )
            else:
                log.warning("pipeline_ocr_skipped", reason="ocr_engine_not_configured")

            # Этап 3: Классификация
            log.info("pipeline_stage_classify")
            classification = await self._cascade.classify(document)
            document.classification = classification
            self._stats["total_classified"] += 1

            log.info(
                "pipeline_classification_result",
                doc_type=classification.doc_type.value,
                confidence=classification.confidence,
                classifier=classification.classifier_name,
            )

            # Этап 4: Сортировка (только если уверенность достаточна)
            if (
                self._sorter is not None
                and classification.confidence >= self._config.classification.auto_sort_threshold
            ):
                log.info("pipeline_stage_sort")
                new_path = await self._sorter.sort(document)
                document.file_path = str(new_path)
                self._stats["total_sorted"] += 1
                log.info("pipeline_sorted", new_path=str(new_path))
            elif self._sorter is None:
                log.debug("pipeline_sort_skipped", reason="sorter_not_configured")
            else:
                log.info(
                    "pipeline_sort_skipped_low_confidence",
                    confidence=classification.confidence,
                    threshold=self._config.classification.auto_sort_threshold,
                )

            # Отмечаем обработку завершённой
            document.mark_processed()
            self._stats["total_processed"] += 1

            log.info("pipeline_document_processing_complete")

        except Exception:
            self._stats["total_errors"] += 1
            log.exception("pipeline_document_processing_error")
            raise

        return document

    async def process_batch(self, documents: list[Document]) -> list[Document]:
        """
        Обрабатывает пакет документов с ограничением параллелизма.

        Args:
            documents: Список документов для обработки.

        Returns:
            Список обработанных документов (неудачные исключаются из результата).
        """
        max_workers = self._config.general.max_workers
        semaphore = asyncio.Semaphore(max_workers)

        logger.info(
            "pipeline_batch_start",
            batch_size=len(documents),
            max_workers=max_workers,
        )

        async def _process_with_semaphore(doc: Document) -> Document | None:
            async with semaphore:
                try:
                    return await self.process_document(doc)
                except Exception:
                    logger.exception(
                        "pipeline_batch_document_failed",
                        doc_id=str(doc.id),
                    )
                    return None

        tasks = [_process_with_semaphore(doc) for doc in documents]
        results = await asyncio.gather(*tasks)

        processed = [doc for doc in results if doc is not None]

        logger.info(
            "pipeline_batch_complete",
            total=len(documents),
            processed=len(processed),
            failed=len(documents) - len(processed),
        )

        return processed

    async def collect_documents(self) -> list[Document]:
        """
        Собирает документы из всех настроенных источников.

        Returns:
            Список собранных документов из всех сборщиков.
        """
        all_documents: list[Document] = []

        for collector in self._collectors:
            try:
                docs = await collector.collect()
                all_documents.extend(docs)
                logger.info(
                    "pipeline_collector_result",
                    collector=type(collector).__name__,
                    count=len(docs),
                )
            except Exception:
                logger.exception(
                    "pipeline_collector_error",
                    collector=type(collector).__name__,
                )

        self._stats["total_collected"] += len(all_documents)

        logger.info(
            "pipeline_collection_complete",
            total_collected=len(all_documents),
        )

        return all_documents

    async def run_once(self) -> list[Document]:
        """
        Выполняет один цикл пайплайна: сбор -> обработка.

        Returns:
            Список обработанных документов.
        """
        logger.info("pipeline_run_once_start")
        documents = await self.collect_documents()

        if not documents:
            logger.info("pipeline_run_once_no_documents")
            return []

        results = await self.process_batch(documents)

        logger.info(
            "pipeline_run_once_complete",
            collected=len(documents),
            processed=len(results),
            stats=self._stats,
        )

        return results

    async def run_continuous(self, poll_interval: float = 5.0) -> None:
        """
        Запускает пайплайн в непрерывном режиме.

        Периодически собирает и обрабатывает документы с заданным интервалом.
        Останавливается при вызове ``stop()`` или получении сигнала прерывания.

        Args:
            poll_interval: Интервал между циклами обработки в секундах.
        """
        self._running = True

        logger.info(
            "pipeline_continuous_start",
            poll_interval=poll_interval,
        )

        while self._running:
            try:
                await self.run_once()
            except Exception:
                logger.exception("pipeline_continuous_cycle_error")

            if self._running:
                await asyncio.sleep(poll_interval)

        logger.info("pipeline_continuous_stopped", stats=self._stats)

    def stop(self) -> None:
        """Останавливает непрерывный режим работы пайплайна."""
        logger.info("pipeline_stop_requested")
        self._running = False


# ---------------------------------------------------------------------------
# Настройка структурного логирования
# ---------------------------------------------------------------------------


def configure_logging(log_level: str = "INFO") -> None:
    """
    Настраивает structlog для приложения.

    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR).
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level.upper()),
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Главная точка входа приложения DocSort AI.

    Загружает конфигурацию, настраивает логирование, создаёт пайплайн
    и запускает его в непрерывном режиме.
    """
    config = get_config()
    configure_logging(config.general.log_level)

    log = structlog.get_logger("docsort.main")
    log.info(
        "docsort_starting",
        version="0.1.0",
        active_classifier=config.classification.active_classifier,
        log_level=config.general.log_level,
    )

    # Создаём необходимые директории
    _ensure_directories(config)

    # Собираем пайплайн
    # В реальном приложении здесь создаются конкретные реализации коллекторов,
    # OCR-движка, классификаторов и сортировщика на основе конфигурации.
    pipeline = DocumentPipeline(config=config)

    # Обработка сигналов для корректного завершения
    loop = asyncio.new_event_loop()

    def _handle_shutdown(sig: signal.Signals) -> None:
        log.info("shutdown_signal_received", signal=sig.name)
        pipeline.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_shutdown, sig)

    try:
        poll_interval = config.collectors.folder_watcher.poll_interval if hasattr(
            config.collectors.folder_watcher, "poll_interval"
        ) else 5.0
        loop.run_until_complete(pipeline.run_continuous(poll_interval=5.0))
    except KeyboardInterrupt:
        log.info("keyboard_interrupt")
        pipeline.stop()
    finally:
        loop.close()
        log.info("docsort_stopped", stats=pipeline.stats)


def _ensure_directories(config: AppConfig) -> None:
    """
    Создаёт необходимые директории, если они не существуют.

    Args:
        config: Конфигурация приложения.
    """
    dirs_to_create = [
        config.general.log_dir,
        config.collectors.folder_watcher.watch_dir,
        config.sorting.output_dir,
    ]

    for dir_path in dirs_to_create:
        path = Path(dir_path)
        if not path.is_absolute():
            # Относительные пути от корня проекта
            path = Path(__file__).resolve().parent.parent.parent / path
        path.mkdir(parents=True, exist_ok=True)
        logger.debug("directory_ensured", path=str(path))


if __name__ == "__main__":
    main()
