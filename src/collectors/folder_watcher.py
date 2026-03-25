"""Сборщик документов на основе мониторинга файловой системы."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Sequence

import structlog
from pydantic import BaseModel, Field
from watchdog.events import FileCreatedEvent, FileMovedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .base import BaseCollector

logger = structlog.get_logger(__name__)


# ------------------------------------------------------------------
# Pydantic-модель конфигурации
# ------------------------------------------------------------------


class FolderWatcherConfig(BaseModel):
    """Конфигурация наблюдателя за папкой.

    Параметры соответствуют секции ``collectors.folder_watcher``
    в ``config/settings.yaml``.
    """

    watch_dir: str = Field(
        default="data/inbox",
        description="Директория для наблюдения за новыми файлами.",
    )
    recursive: bool = Field(
        default=False,
        description="Наблюдать за вложенными директориями рекурсивно.",
    )
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".tif", ".tiff", ".png", ".jpg", ".jpeg"],
        description="Допустимые расширения файлов.",
    )


# ------------------------------------------------------------------
# watchdog-обработчик событий
# ------------------------------------------------------------------


class _FileEventHandler(FileSystemEventHandler):
    """Обработчик событий файловой системы от ``watchdog``.

    При создании или перемещении файла с допустимым расширением кладёт
    путь в ``asyncio.Queue``, привязанную к конкретному event-loop'у.
    """

    def __init__(
        self,
        queue: asyncio.Queue[Path],
        loop: asyncio.AbstractEventLoop,
        allowed_extensions: Sequence[str] | None,
    ) -> None:
        """Инициализация обработчика.

        Args:
            queue: Очередь для передачи путей в асинхронный код.
            loop: Event-loop, в который нужно планировать ``put``.
            allowed_extensions: Допустимые расширения (lowercase) или
                ``None`` для приёма всех файлов.
        """
        super().__init__()
        self._queue = queue
        self._loop = loop
        self._allowed_extensions = (
            tuple(ext.lower() for ext in allowed_extensions)
            if allowed_extensions
            else None
        )
        self._log = structlog.get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------

    def _is_accepted(self, path: str) -> bool:
        """Проверяет, допустимо ли расширение файла."""
        if self._allowed_extensions is None:
            return True
        return Path(path).suffix.lower() in self._allowed_extensions

    def _enqueue(self, path: str) -> None:
        """Помещает путь в очередь из потока ``watchdog``."""
        resolved = Path(path).resolve()
        self._log.debug("file_detected", path=str(resolved))
        self._loop.call_soon_threadsafe(self._queue.put_nowait, resolved)

    # ------------------------------------------------------------------
    # Обработка событий watchdog
    # ------------------------------------------------------------------

    def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
        """Обрабатывает создание нового файла."""
        if event.is_directory:
            return
        if self._is_accepted(event.src_path):
            self._enqueue(event.src_path)

    def on_moved(self, event: FileMovedEvent) -> None:  # type: ignore[override]
        """Обрабатывает перемещение файла в отслеживаемую директорию."""
        if event.is_directory:
            return
        if self._is_accepted(event.dest_path):
            self._enqueue(event.dest_path)


# ------------------------------------------------------------------
# Реализация сборщика
# ------------------------------------------------------------------


class FolderWatcher(BaseCollector):
    """Сборщик документов, отслеживающий появление файлов в директории.

    Использует библиотеку ``watchdog`` для мониторинга файловой системы
    и ``asyncio.Queue`` для передачи обнаруженных путей в асинхронный
    конвейер обработки.
    """

    def __init__(self, config: FolderWatcherConfig) -> None:
        """Инициализация наблюдателя.

        Args:
            config: Экземпляр ``FolderWatcherConfig`` с параметрами
                наблюдения.
        """
        super().__init__(allowed_extensions=config.allowed_extensions)
        self._config = config
        self._watch_dir = Path(config.watch_dir)
        self._queue: asyncio.Queue[Path] = asyncio.Queue()
        self._observer: Observer | None = None

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Запускает наблюдатель за файловой системой.

        Создаёт целевую директорию (если не существует), инициализирует
        ``watchdog.Observer`` и начинает мониторинг.
        """
        if self._running:
            self._log.warning("watcher_already_running")
            return

        self._watch_dir.mkdir(parents=True, exist_ok=True)

        loop = asyncio.get_running_loop()
        handler = _FileEventHandler(
            queue=self._queue,
            loop=loop,
            allowed_extensions=self._config.allowed_extensions,
        )

        self._observer = Observer()
        self._observer.schedule(
            handler,
            str(self._watch_dir),
            recursive=self._config.recursive,
        )
        self._observer.daemon = True
        self._observer.start()
        self._running = True

        self._log.info(
            "watcher_started",
            watch_dir=str(self._watch_dir.resolve()),
            recursive=self._config.recursive,
            extensions=self._allowed_extensions,
        )

    async def stop(self) -> None:
        """Останавливает наблюдатель и освобождает ресурсы."""
        if not self._running:
            return

        self._running = False

        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None

        self._log.info("watcher_stopped")

    # ------------------------------------------------------------------
    # Основной генератор
    # ------------------------------------------------------------------

    async def collect(self) -> AsyncGenerator[Path, None]:
        """Асинхронный генератор путей к обнаруженным файлам.

        Yields:
            ``Path`` — абсолютный путь к новому файлу в отслеживаемой
            директории.
        """
        while self._running or not self._queue.empty():
            try:
                path = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                self._log.info("file_collected", path=str(path))
                yield path
            except asyncio.TimeoutError:
                continue
