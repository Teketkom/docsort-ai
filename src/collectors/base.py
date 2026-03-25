"""Базовый класс сборщика документов."""

from __future__ import annotations

import abc
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Sequence

import structlog

logger = structlog.get_logger(__name__)


class BaseCollector(abc.ABC):
    """Абстрактный базовый класс для всех сборщиков документов.

    Определяет общий интерфейс: запуск, остановка и асинхронная генерация
    путей к найденным файлам.  Конкретные реализации должны переопределить
    методы ``collect``, ``start`` и ``stop``.
    """

    # ------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------

    def __init__(
        self,
        allowed_extensions: Sequence[str] | None = None,
    ) -> None:
        """Инициализация базового сборщика.

        Args:
            allowed_extensions: Допустимые расширения файлов (например,
                ``[".pdf", ".png"]``).  Если ``None`` — принимаются все файлы.
        """
        self._allowed_extensions: tuple[str, ...] | None = (
            tuple(ext.lower() for ext in allowed_extensions)
            if allowed_extensions
            else None
        )
        self._running: bool = False
        self._log = logger.bind(collector=self.__class__.__name__)

    # ------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------

    @property
    def supported_extensions(self) -> tuple[str, ...] | None:
        """Возвращает кортеж допустимых расширений или ``None``, если
        ограничений нет."""
        return self._allowed_extensions

    @property
    def is_running(self) -> bool:
        """Возвращает ``True``, если сборщик запущен."""
        return self._running

    # ------------------------------------------------------------------
    # Публичные методы
    # ------------------------------------------------------------------

    def _is_supported_file(self, path: Path | str) -> bool:
        """Проверяет, поддерживается ли файл по расширению.

        Args:
            path: Путь к файлу (``Path`` или строка).

        Returns:
            ``True``, если расширение входит в список допустимых или если
            список не задан (принимаются все файлы).
        """
        if self._allowed_extensions is None:
            return True

        ext = Path(path).suffix.lower()
        return ext in self._allowed_extensions

    # ------------------------------------------------------------------
    # Абстрактный интерфейс
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def collect(self) -> AsyncGenerator[Path, None]:
        """Асинхронный генератор, отдающий пути к найденным документам.

        Yields:
            ``Path`` — абсолютный путь к очередному найденному файлу.
        """
        ...  # pragma: no cover
        # Необходим для корректной типизации AsyncGenerator:
        yield  # type: ignore[misc]

    @abc.abstractmethod
    async def start(self) -> None:
        """Запускает сборщик (подключение, инициализация ресурсов)."""
        ...

    @abc.abstractmethod
    async def stop(self) -> None:
        """Останавливает сборщик и освобождает ресурсы."""
        ...

    # ------------------------------------------------------------------
    # Магические методы
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"running={self._running} "
            f"extensions={self._allowed_extensions}>"
        )
