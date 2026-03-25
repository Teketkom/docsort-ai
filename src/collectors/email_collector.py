"""Сборщик документов из электронной почты по протоколу IMAP."""

from __future__ import annotations

import asyncio
import email
import email.policy
import imaplib
from collections.abc import AsyncGenerator
from email.message import EmailMessage
from pathlib import Path

import structlog
from pydantic import BaseModel, Field, SecretStr

from .base import BaseCollector

logger = structlog.get_logger(__name__)


# ------------------------------------------------------------------
# Pydantic-модель конфигурации
# ------------------------------------------------------------------


class EmailCollectorConfig(BaseModel):
    """Конфигурация IMAP-сборщика электронной почты.

    Все параметры соответствуют секции ``collectors.email``
    в ``config/settings.yaml``.
    """

    host: str = Field(
        ...,
        description="Адрес IMAP-сервера (например, imap.example.com).",
    )
    port: int = Field(
        default=993,
        ge=1,
        le=65535,
        description="Порт IMAP-сервера.",
    )
    use_ssl: bool = Field(
        default=True,
        description="Использовать SSL/TLS при подключении.",
    )
    username: str = Field(
        ...,
        min_length=1,
        description="Имя пользователя (логин).",
    )
    password: SecretStr = Field(
        ...,
        description="Пароль учётной записи.",
    )
    folder: str = Field(
        default="INBOX",
        description="Почтовая папка для мониторинга.",
    )
    poll_interval: float = Field(
        default=60.0,
        gt=0,
        description="Интервал опроса почтового ящика (секунды).",
    )
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".tif", ".tiff", ".png", ".jpg", ".jpeg"],
        description="Допустимые расширения вложений.",
    )
    save_dir: str = Field(
        default="data/inbox",
        description="Директория для сохранения вложений.",
    )


# ------------------------------------------------------------------
# Реализация сборщика
# ------------------------------------------------------------------


class EmailCollector(BaseCollector):
    """Сборщик документов из почтового ящика по протоколу IMAP.

    Подключается к указанному IMAP-серверу, периодически проверяет
    наличие непрочитанных писем, извлекает вложения с допустимыми
    расширениями и сохраняет их в локальную директорию.
    """

    def __init__(self, config: EmailCollectorConfig) -> None:
        """Инициализация IMAP-сборщика.

        Args:
            config: Экземпляр ``EmailCollectorConfig`` с параметрами
                подключения и фильтрации.
        """
        super().__init__(allowed_extensions=config.allowed_extensions)
        self._config = config
        self._save_dir = Path(config.save_dir)
        self._connection: imaplib.IMAP4 | imaplib.IMAP4_SSL | None = None
        self._queue: asyncio.Queue[Path] = asyncio.Queue()
        self._poll_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Запускает сборщик: создаёт директорию, подключается к IMAP,
        запускает фоновый цикл опроса."""
        if self._running:
            self._log.warning("collector_already_running")
            return

        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._connect()
        self._running = True
        self._poll_task = asyncio.create_task(
            self._poll_loop(),
            name="email-collector-poll",
        )
        self._log.info(
            "collector_started",
            host=self._config.host,
            folder=self._config.folder,
        )

    async def stop(self) -> None:
        """Останавливает фоновый цикл опроса и закрывает IMAP-соединение."""
        if not self._running:
            return

        self._running = False

        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        self._disconnect()
        self._log.info("collector_stopped")

    # ------------------------------------------------------------------
    # Основной генератор
    # ------------------------------------------------------------------

    async def collect(self) -> AsyncGenerator[Path, None]:
        """Асинхронный генератор путей к загруженным вложениям.

        Yields:
            ``Path`` — абсолютный путь к сохранённому файлу вложения.
        """
        while self._running or not self._queue.empty():
            try:
                path = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                yield path
            except asyncio.TimeoutError:
                continue

    # ------------------------------------------------------------------
    # IMAP-подключение
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Устанавливает соединение с IMAP-сервером."""
        try:
            if self._config.use_ssl:
                self._connection = imaplib.IMAP4_SSL(
                    host=self._config.host,
                    port=self._config.port,
                )
            else:
                self._connection = imaplib.IMAP4(
                    host=self._config.host,
                    port=self._config.port,
                )

            self._connection.login(
                self._config.username,
                self._config.password.get_secret_value(),
            )
            self._log.info(
                "imap_connected",
                host=self._config.host,
                port=self._config.port,
                ssl=self._config.use_ssl,
            )
        except imaplib.IMAP4.error as exc:
            self._log.error("imap_connection_failed", error=str(exc))
            raise

    def _disconnect(self) -> None:
        """Закрывает IMAP-соединение."""
        if self._connection is None:
            return
        try:
            self._connection.close()
        except imaplib.IMAP4.error:
            pass
        try:
            self._connection.logout()
        except imaplib.IMAP4.error:
            pass
        self._connection = None
        self._log.debug("imap_disconnected")

    # ------------------------------------------------------------------
    # Фоновый цикл опроса
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Цикл периодического опроса почтового ящика."""
        while self._running:
            try:
                await asyncio.to_thread(self._fetch_unread)
            except imaplib.IMAP4.error as exc:
                self._log.error("imap_fetch_error", error=str(exc))
                # Попытка переподключения
                try:
                    self._disconnect()
                    self._connect()
                except Exception as reconnect_exc:
                    self._log.error(
                        "imap_reconnect_failed",
                        error=str(reconnect_exc),
                    )
            except Exception as exc:
                self._log.error("poll_unexpected_error", error=str(exc))

            await asyncio.sleep(self._config.poll_interval)

    # ------------------------------------------------------------------
    # Извлечение непрочитанных писем
    # ------------------------------------------------------------------

    def _fetch_unread(self) -> None:
        """Находит непрочитанные письма и сохраняет допустимые вложения.

        Выполняется в отдельном потоке (``asyncio.to_thread``), так как
        ``imaplib`` — блокирующая библиотека.
        """
        if self._connection is None:
            self._log.warning("imap_not_connected")
            return

        self._connection.select(self._config.folder)
        status, data = self._connection.search(None, "UNSEEN")
        if status != "OK" or not data or not data[0]:
            return

        message_ids: list[bytes] = data[0].split()
        self._log.info("unread_messages_found", count=len(message_ids))

        for msg_id in message_ids:
            try:
                self._process_message(msg_id)
            except Exception as exc:
                self._log.error(
                    "message_processing_failed",
                    msg_id=msg_id.decode(),
                    error=str(exc),
                )

    def _process_message(self, msg_id: bytes) -> None:
        """Обрабатывает одно письмо: извлекает и сохраняет вложения.

        Args:
            msg_id: Идентификатор сообщения на IMAP-сервере.
        """
        if self._connection is None:
            return

        status, msg_data = self._connection.fetch(msg_id, "(RFC822)")
        if status != "OK" or not msg_data or not msg_data[0]:
            return

        raw_email = msg_data[0]
        if not isinstance(raw_email, tuple):
            return

        msg: EmailMessage = email.message_from_bytes(
            raw_email[1],
            policy=email.policy.default,
        )  # type: ignore[assignment]

        subject = msg.get("Subject", "<без темы>")
        saved_count = 0

        for part in msg.walk():
            content_disposition = str(part.get("Content-Disposition", ""))
            if "attachment" not in content_disposition:
                continue

            filename: str | None = part.get_filename()
            if filename is None:
                continue

            if not self._is_supported_file(filename):
                self._log.debug(
                    "attachment_skipped_extension",
                    filename=filename,
                )
                continue

            payload = part.get_payload(decode=True)
            if payload is None:
                continue

            saved_path = self._save_attachment(filename, payload)
            if saved_path is not None:
                self._queue.put_nowait(saved_path)
                saved_count += 1

        if saved_count > 0:
            self._log.info(
                "attachments_saved",
                subject=subject,
                count=saved_count,
            )

        # Помечаем письмо как прочитанное
        self._connection.store(msg_id, "+FLAGS", "\\Seen")

    # ------------------------------------------------------------------
    # Сохранение вложений
    # ------------------------------------------------------------------

    def _save_attachment(self, filename: str, payload: bytes) -> Path | None:
        """Сохраняет вложение на диск.

        Если файл с таким именем уже существует, к имени добавляется
        числовой суффикс.

        Args:
            filename: Оригинальное имя файла вложения.
            payload: Содержимое вложения (байты).

        Returns:
            ``Path`` к сохранённому файлу или ``None`` при ошибке записи.
        """
        target = self._save_dir / filename
        target = self._deduplicate_path(target)

        try:
            target.write_bytes(payload)
            self._log.debug("attachment_written", path=str(target))
            return target.resolve()
        except OSError as exc:
            self._log.error(
                "attachment_write_failed",
                path=str(target),
                error=str(exc),
            )
            return None

    @staticmethod
    def _deduplicate_path(path: Path) -> Path:
        """Возвращает уникальный путь, добавляя суффикс ``_N`` при коллизии.

        Args:
            path: Желаемый путь к файлу.

        Returns:
            Путь, гарантированно не занятый существующим файлом.
        """
        if not path.exists():
            return path

        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        counter = 1

        while True:
            candidate = parent / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1
