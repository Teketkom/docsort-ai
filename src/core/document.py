"""
Модели данных для документов.

Определяет основные Pydantic-модели: Document, DocumentClassification,
а также перечисление типов документов DocumentType.
"""

from __future__ import annotations

import mimetypes
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger(__name__)


class DocumentType(str, Enum):
    """Типы документов, поддерживаемые системой классификации."""

    INVOICE = "invoice"
    ACT = "act"
    CONTRACT = "contract"
    WAYBILL = "waybill"
    PAYMENT_ORDER = "payment_order"
    UNKNOWN = "unknown"

    @property
    def display_name(self) -> str:
        """Возвращает человекочитаемое название типа документа на русском языке."""
        names: dict[str, str] = {
            "invoice": "Счёт-фактура",
            "act": "Акт выполненных работ",
            "contract": "Договор",
            "waybill": "Товарная накладная ТОРГ-12",
            "payment_order": "Платёжное поручение",
            "unknown": "Неопознанный документ",
        }
        return names.get(self.value, self.value)


class DocumentClassification(BaseModel):
    """
    Результат классификации документа.

    Содержит тип документа, уровень уверенности классификатора,
    имя использованного классификатора и дополнительные метаданные.
    """

    doc_type: DocumentType = Field(
        description="Определённый тип документа",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Уровень уверенности классификатора (от 0.0 до 1.0)",
    )
    classifier_name: str = Field(
        description="Имя классификатора, выполнившего классификацию (rules / ml / neural / llm)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Дополнительные метаданные классификации",
    )

    @field_validator("confidence")
    @classmethod
    def _round_confidence(cls, v: float) -> float:
        """Округляет уверенность до четырёх знаков после запятой."""
        return round(v, 4)

    @property
    def is_confident(self) -> bool:
        """Проверяет, достаточно ли высока уверенность для автосортировки (порог 0.85)."""
        return self.confidence >= 0.85


class Document(BaseModel):
    """
    Основная модель документа в системе DocSort AI.

    Представляет документ на всех этапах пайплайна обработки:
    от сбора до классификации и сортировки.
    """

    id: UUID = Field(
        default_factory=uuid4,
        description="Уникальный идентификатор документа",
    )
    file_path: str = Field(
        description="Путь к файлу документа на диске",
    )
    original_filename: str = Field(
        description="Оригинальное имя файла",
    )
    file_size: int = Field(
        ge=0,
        description="Размер файла в байтах",
    )
    page_count: int = Field(
        default=0,
        ge=0,
        description="Количество страниц в документе",
    )
    mime_type: str = Field(
        default="application/octet-stream",
        description="MIME-тип файла",
    )
    ocr_text: str = Field(
        default="",
        description="Текст, извлечённый OCR-движком",
    )
    extracted_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Поля, извлечённые из текста документа (ИНН, КПП, суммы и т.д.)",
    )
    classification: DocumentClassification | None = Field(
        default=None,
        description="Результат классификации документа",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Дата и время создания записи",
    )
    processed_at: datetime | None = Field(
        default=None,
        description="Дата и время завершения обработки",
    )

    @classmethod
    def from_file(cls, file_path: str | Path) -> Document:
        """
        Создаёт экземпляр Document из пути к файлу.

        Автоматически определяет имя файла, размер и MIME-тип.
        Количество страниц устанавливается на этапе предобработки.

        Args:
            file_path: Путь к файлу документа.

        Returns:
            Новый экземпляр Document.

        Raises:
            FileNotFoundError: Если файл не найден по указанному пути.
            PermissionError: Если нет прав на чтение файла.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")

        if not path.is_file():
            raise ValueError(f"Путь не является файлом: {path}")

        stat = path.stat()
        mime_type, _ = mimetypes.guess_type(str(path))

        doc = cls(
            file_path=str(path.resolve()),
            original_filename=path.name,
            file_size=stat.st_size,
            mime_type=mime_type or "application/octet-stream",
        )

        logger.info(
            "document_created_from_file",
            doc_id=str(doc.id),
            file_path=str(path),
            file_size=stat.st_size,
            mime_type=doc.mime_type,
        )

        return doc

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        filename: str,
        file_path: str,
    ) -> Document:
        """
        Создаёт экземпляр Document из байтов (например, из вложения письма).

        Args:
            data: Содержимое файла.
            filename: Оригинальное имя файла.
            file_path: Путь, по которому файл будет сохранён.

        Returns:
            Новый экземпляр Document.
        """
        mime_type, _ = mimetypes.guess_type(filename)

        doc = cls(
            file_path=file_path,
            original_filename=filename,
            file_size=len(data),
            mime_type=mime_type or "application/octet-stream",
        )

        logger.info(
            "document_created_from_bytes",
            doc_id=str(doc.id),
            filename=filename,
            file_size=len(data),
        )

        return doc

    def mark_processed(self) -> None:
        """Отмечает документ как обработанный, устанавливая время обработки."""
        self.processed_at = datetime.now(timezone.utc)
        logger.info(
            "document_marked_processed",
            doc_id=str(self.id),
            processed_at=self.processed_at.isoformat(),
        )

    @property
    def is_classified(self) -> bool:
        """Проверяет, был ли документ классифицирован."""
        return self.classification is not None

    @property
    def doc_type(self) -> DocumentType:
        """Возвращает тип документа или UNKNOWN, если классификация не выполнена."""
        if self.classification is None:
            return DocumentType.UNKNOWN
        return self.classification.doc_type

    @property
    def extension(self) -> str:
        """Возвращает расширение файла в нижнем регистре (включая точку)."""
        return os.path.splitext(self.original_filename)[1].lower()

    def __repr__(self) -> str:
        """Строковое представление документа."""
        classified = (
            f", type={self.classification.doc_type.value}, "
            f"confidence={self.classification.confidence:.2f}"
            if self.classification
            else ""
        )
        return (
            f"Document(id={self.id!s:.8}, "
            f"file={self.original_filename!r}{classified})"
        )
