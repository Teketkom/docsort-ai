"""Менеджер обратной связи и обучения."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class FeedbackEntry(BaseModel):
    """Модель записи обратной связи.

    Attributes:
        id: Уникальный идентификатор записи.
        doc_id: Идентификатор документа.
        predicted_type: Предсказанный тип документа.
        correct_type: Правильный тип документа (указанный пользователем).
        confidence: Уверенность классификатора (0.0 - 1.0).
        user_comment: Комментарий пользователя.
        created_at: Дата и время создания записи.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    predicted_type: str
    correct_type: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    user_comment: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class FeedbackManager:
    """Менеджер обратной связи для системы классификации документов.

    Хранит обратную связь от пользователей, отслеживает точность
    классификации и определяет необходимость переобучения модели.

    Attributes:
        db_path: Путь к файлу базы данных SQLite.
        retrain_threshold: Порог количества исправлений для переобучения.
        _initialized: Флаг инициализации таблиц.
    """

    CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            predicted_type TEXT NOT NULL,
            correct_type TEXT NOT NULL,
            confidence REAL DEFAULT 0.0,
            user_comment TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )
    """

    CREATE_INDEX_SQL = """
        CREATE INDEX IF NOT EXISTS idx_feedback_doc_id ON feedback(doc_id)
    """

    def __init__(
        self,
        db_path: Path | str = "feedback.db",
        retrain_threshold: int = 50,
    ) -> None:
        """Инициализация менеджера обратной связи.

        Args:
            db_path: Путь к файлу базы данных SQLite.
            retrain_threshold: Минимальное количество исправлений
                для запуска переобучения.
        """
        self.db_path = Path(db_path)
        self.retrain_threshold = retrain_threshold
        self._initialized = False

        logger.info(
            "feedback_manager_initialized",
            db_path=str(self.db_path),
            retrain_threshold=self.retrain_threshold,
        )

    async def _ensure_tables(self) -> None:
        """Создание таблиц при первом обращении к базе данных."""
        if self._initialized:
            return

        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute(self.CREATE_TABLE_SQL)
                await db.execute(self.CREATE_INDEX_SQL)
                await db.commit()
            self._initialized = True
            logger.info("database_tables_created", db_path=str(self.db_path))
        except aiosqlite.Error as exc:
            logger.error(
                "database_initialization_failed",
                db_path=str(self.db_path),
                error=str(exc),
            )
            raise

    async def record_feedback(
        self,
        doc_id: str,
        predicted_type: str,
        correct_type: str,
        user_comment: str = "",
        confidence: float = 0.0,
    ) -> FeedbackEntry:
        """Сохранение обратной связи от пользователя.

        Args:
            doc_id: Идентификатор документа.
            predicted_type: Предсказанный тип документа.
            correct_type: Правильный тип документа.
            user_comment: Комментарий пользователя.
            confidence: Уверенность классификатора.

        Returns:
            Созданная запись обратной связи.

        Raises:
            aiosqlite.Error: При ошибке записи в базу данных.
        """
        await self._ensure_tables()

        entry = FeedbackEntry(
            doc_id=doc_id,
            predicted_type=predicted_type,
            correct_type=correct_type,
            confidence=confidence,
            user_comment=user_comment,
        )

        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute(
                    """
                    INSERT INTO feedback
                        (id, doc_id, predicted_type, correct_type,
                         confidence, user_comment, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.id,
                        entry.doc_id,
                        entry.predicted_type,
                        entry.correct_type,
                        entry.confidence,
                        entry.user_comment,
                        entry.created_at.isoformat(),
                    ),
                )
                await db.commit()

            is_correction = predicted_type != correct_type
            logger.info(
                "feedback_recorded",
                doc_id=doc_id,
                predicted_type=predicted_type,
                correct_type=correct_type,
                is_correction=is_correction,
            )
            return entry

        except aiosqlite.Error as exc:
            logger.error(
                "feedback_record_failed",
                doc_id=doc_id,
                error=str(exc),
            )
            raise

    async def get_corrections(self) -> list[FeedbackEntry]:
        """Получение всех исправлений для переобучения.

        Возвращает записи, где предсказанный тип отличается
        от правильного.

        Returns:
            Список записей с исправлениями.
        """
        await self._ensure_tables()

        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    """
                    SELECT id, doc_id, predicted_type, correct_type,
                           confidence, user_comment, created_at
                    FROM feedback
                    WHERE predicted_type != correct_type
                    ORDER BY created_at DESC
                    """
                )
                rows = await cursor.fetchall()

            corrections = [
                FeedbackEntry(
                    id=row["id"],
                    doc_id=row["doc_id"],
                    predicted_type=row["predicted_type"],
                    correct_type=row["correct_type"],
                    confidence=row["confidence"],
                    user_comment=row["user_comment"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in rows
            ]

            logger.info("corrections_retrieved", count=len(corrections))
            return corrections

        except aiosqlite.Error as exc:
            logger.error("corrections_retrieval_failed", error=str(exc))
            raise

    async def get_statistics(self) -> dict:
        """Получение статистики точности классификации.

        Returns:
            Словарь со статистикой:
                - total_feedback: Общее количество отзывов.
                - total_corrections: Количество исправлений.
                - accuracy: Точность классификации (0.0 - 1.0).
                - corrections_by_type: Исправления по типам документов.
                - recent_accuracy: Точность за последние 100 записей.
        """
        await self._ensure_tables()

        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM feedback")
                row = await cursor.fetchone()
                total_feedback: int = row[0] if row else 0

                cursor = await db.execute(
                    """
                    SELECT COUNT(*) FROM feedback
                    WHERE predicted_type != correct_type
                    """
                )
                row = await cursor.fetchone()
                total_corrections: int = row[0] if row else 0

                accuracy = 0.0
                if total_feedback > 0:
                    accuracy = (
                        (total_feedback - total_corrections) / total_feedback
                    )

                cursor = await db.execute(
                    """
                    SELECT predicted_type, correct_type, COUNT(*) as cnt
                    FROM feedback
                    WHERE predicted_type != correct_type
                    GROUP BY predicted_type, correct_type
                    ORDER BY cnt DESC
                    """
                )
                correction_rows = await cursor.fetchall()
                corrections_by_type = [
                    {
                        "predicted_type": r[0],
                        "correct_type": r[1],
                        "count": r[2],
                    }
                    for r in correction_rows
                ]

                cursor = await db.execute(
                    """
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN predicted_type != correct_type
                               THEN 1 ELSE 0 END) as errors
                    FROM (
                        SELECT predicted_type, correct_type
                        FROM feedback
                        ORDER BY created_at DESC
                        LIMIT 100
                    )
                    """
                )
                recent_row = await cursor.fetchone()
                recent_accuracy = 0.0
                if recent_row and recent_row[0] and recent_row[0] > 0:
                    recent_errors = recent_row[1] or 0
                    recent_accuracy = (
                        (recent_row[0] - recent_errors) / recent_row[0]
                    )

            stats = {
                "total_feedback": total_feedback,
                "total_corrections": total_corrections,
                "accuracy": round(accuracy, 4),
                "recent_accuracy": round(recent_accuracy, 4),
                "corrections_by_type": corrections_by_type,
            }

            logger.info("statistics_retrieved", **stats)
            return stats

        except aiosqlite.Error as exc:
            logger.error("statistics_retrieval_failed", error=str(exc))
            raise

    async def should_retrain(self) -> bool:
        """Проверка необходимости переобучения модели.

        Переобучение рекомендуется, если количество накопленных
        исправлений превышает заданный порог.

        Returns:
            True, если переобучение рекомендуется.
        """
        await self._ensure_tables()

        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                cursor = await db.execute(
                    """
                    SELECT COUNT(*) FROM feedback
                    WHERE predicted_type != correct_type
                    """
                )
                row = await cursor.fetchone()
                correction_count: int = row[0] if row else 0

            should = correction_count >= self.retrain_threshold

            logger.info(
                "retrain_check",
                correction_count=correction_count,
                threshold=self.retrain_threshold,
                should_retrain=should,
            )
            return should

        except aiosqlite.Error as exc:
            logger.error("retrain_check_failed", error=str(exc))
            raise
