"""Пакет сборщиков документов для DocSort AI.

Содержит реализации источников документов:

- :class:`BaseCollector` — абстрактный базовый класс.
- :class:`EmailCollector` — сборщик вложений из электронной почты (IMAP).
- :class:`FolderWatcher` — наблюдатель за файловой системой (watchdog).

Конфигурационные модели:

- :class:`EmailCollectorConfig` — параметры IMAP-подключения.
- :class:`FolderWatcherConfig` — параметры мониторинга директории.
"""

from .base import BaseCollector
from .email_collector import EmailCollector, EmailCollectorConfig
from .folder_watcher import FolderWatcher, FolderWatcherConfig

__all__ = [
    "BaseCollector",
    "EmailCollector",
    "EmailCollectorConfig",
    "FolderWatcher",
    "FolderWatcherConfig",
]
