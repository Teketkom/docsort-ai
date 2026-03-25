"""REST API сервер DocSort AI."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pydantic модели запросов и ответов
# ---------------------------------------------------------------------------


class ClassificationResult(BaseModel):
    """Результат классификации документа.

    Attributes:
        doc_id: Уникальный идентификатор документа.
        filename: Имя исходного файла.
        doc_type: Определённый тип документа.
        confidence: Уверенность классификатора (0.0 - 1.0).
        metadata: Дополнительные метаданные.
        classified_at: Время классификации.
    """

    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    doc_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)
    classified_at: datetime = Field(default_factory=datetime.now)


class BatchClassificationRequest(BaseModel):
    """Запрос пакетной классификации.

    Attributes:
        file_ids: Список идентификаторов файлов для обработки.
    """

    file_ids: list[str] = Field(default_factory=list)


class BatchClassificationResponse(BaseModel):
    """Ответ пакетной классификации.

    Attributes:
        results: Список результатов классификации.
        total: Общее количество обработанных файлов.
        errors: Список ошибок.
    """

    results: list[ClassificationResult] = Field(default_factory=list)
    total: int = 0
    errors: list[dict] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    """Информация о документе.

    Attributes:
        doc_id: Уникальный идентификатор.
        filename: Имя файла.
        doc_type: Тип документа.
        confidence: Уверенность классификации.
        file_path: Путь к файлу.
        created_at: Время обработки.
    """

    doc_id: str
    filename: str
    doc_type: str
    confidence: float
    file_path: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class DocumentListResponse(BaseModel):
    """Ответ со списком документов.

    Attributes:
        documents: Список документов.
        total: Общее количество.
        page: Номер страницы.
        page_size: Размер страницы.
    """

    documents: list[DocumentInfo] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 20


class FeedbackRequest(BaseModel):
    """Запрос обратной связи.

    Attributes:
        doc_id: Идентификатор документа.
        predicted_type: Предсказанный тип.
        correct_type: Правильный тип.
        user_comment: Комментарий пользователя.
    """

    doc_id: str
    predicted_type: str
    correct_type: str
    user_comment: str = ""


class FeedbackResponse(BaseModel):
    """Ответ на обратную связь.

    Attributes:
        id: Идентификатор записи обратной связи.
        doc_id: Идентификатор документа.
        status: Статус обработки.
    """

    id: str
    doc_id: str
    status: str = "recorded"


class StatsResponse(BaseModel):
    """Статистика классификации.

    Attributes:
        total_documents: Общее количество документов.
        total_feedback: Количество отзывов.
        total_corrections: Количество исправлений.
        accuracy: Общая точность.
        recent_accuracy: Точность за последние записи.
        corrections_by_type: Исправления по типам.
        documents_by_type: Документы по типам.
    """

    total_documents: int = 0
    total_feedback: int = 0
    total_corrections: int = 0
    accuracy: float = 0.0
    recent_accuracy: float = 0.0
    corrections_by_type: list[dict] = Field(default_factory=list)
    documents_by_type: dict[str, int] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Ответ проверки здоровья сервиса.

    Attributes:
        status: Статус сервиса.
        version: Версия API.
        timestamp: Время проверки.
    """

    status: str = "ok"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Стандартный ответ об ошибке.

    Attributes:
        detail: Описание ошибки.
        error_code: Код ошибки.
    """

    detail: str
    error_code: str = "unknown_error"


# ---------------------------------------------------------------------------
# In-memory хранилище (заглушка для демонстрации)
# ---------------------------------------------------------------------------


class _DocumentStore:
    """Внутреннее хранилище документов (in-memory).

    В продакшене заменяется на реальную базу данных.
    """

    def __init__(self) -> None:
        self.documents: dict[str, DocumentInfo] = {}

    def add(self, doc: DocumentInfo) -> None:
        """Добавить документ."""
        self.documents[doc.doc_id] = doc

    def get(self, doc_id: str) -> Optional[DocumentInfo]:
        """Получить документ по ID."""
        return self.documents.get(doc_id)

    def list_all(
        self, page: int = 1, page_size: int = 20
    ) -> tuple[list[DocumentInfo], int]:
        """Получить постраничный список документов."""
        all_docs = sorted(
            self.documents.values(),
            key=lambda d: d.created_at,
            reverse=True,
        )
        total = len(all_docs)
        start = (page - 1) * page_size
        end = start + page_size
        return all_docs[start:end], total

    def count_by_type(self) -> dict[str, int]:
        """Подсчёт документов по типам."""
        counts: dict[str, int] = {}
        for doc in self.documents.values():
            counts[doc.doc_type] = counts.get(doc.doc_type, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Фабрика приложения
# ---------------------------------------------------------------------------


def create_app(
    upload_dir: Path | str = "/tmp/docsort_uploads",
    allowed_origins: Optional[list[str]] = None,
) -> FastAPI:
    """Создание и настройка FastAPI приложения.

    Args:
        upload_dir: Директория для загруженных файлов.
        allowed_origins: Список разрешённых источников CORS.

    Returns:
        Настроенное FastAPI приложение.
    """
    app = FastAPI(
        title="DocSort AI API",
        description="REST API для классификации документов",
        version="1.0.0",
    )

    if allowed_origins is None:
        allowed_origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    upload_path = Path(upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)

    store = _DocumentStore()

    # ------------------------------------------------------------------
    # Эндпоинты
    # ------------------------------------------------------------------

    @app.post(
        "/api/v1/classify",
        response_model=ClassificationResult,
        status_code=status.HTTP_200_OK,
        responses={
            400: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    async def classify_document(
        file: UploadFile = File(...),
    ) -> ClassificationResult:
        """Классификация загруженного документа.

        Принимает файл, сохраняет его и возвращает результат
        классификации.
        """
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Имя файла отсутствует.",
            )

        doc_id = str(uuid.uuid4())
        safe_filename = file.filename.replace("/", "_").replace("\\", "_")
        file_dest = upload_path / f"{doc_id}_{safe_filename}"

        try:
            content = await file.read()
            if not content:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Загруженный файл пуст.",
                )
            file_dest.write_bytes(content)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "file_upload_failed",
                filename=file.filename,
                error=str(exc),
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Ошибка сохранения файла.",
            )

        # Заглушка классификации — в продакшене вызывается модель
        result = ClassificationResult(
            doc_id=doc_id,
            filename=safe_filename,
            doc_type="unknown",
            confidence=0.0,
            metadata={"file_size": len(content)},
        )

        store.add(
            DocumentInfo(
                doc_id=doc_id,
                filename=safe_filename,
                doc_type=result.doc_type,
                confidence=result.confidence,
                file_path=str(file_dest),
                created_at=result.classified_at,
            )
        )

        logger.info(
            "document_classified",
            doc_id=doc_id,
            filename=safe_filename,
            doc_type=result.doc_type,
            confidence=result.confidence,
        )
        return result

    @app.post(
        "/api/v1/classify/batch",
        response_model=BatchClassificationResponse,
        status_code=status.HTTP_200_OK,
        responses={
            400: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    async def classify_batch(
        files: list[UploadFile] = File(...),
    ) -> BatchClassificationResponse:
        """Пакетная классификация нескольких документов.

        Принимает список файлов и возвращает результаты
        классификации для каждого.
        """
        results: list[ClassificationResult] = []
        errors: list[dict] = []

        for file in files:
            try:
                if not file.filename:
                    errors.append(
                        {"filename": "unknown", "error": "Имя файла отсутствует"}
                    )
                    continue

                doc_id = str(uuid.uuid4())
                safe_filename = file.filename.replace("/", "_").replace("\\", "_")
                file_dest = upload_path / f"{doc_id}_{safe_filename}"

                content = await file.read()
                if not content:
                    errors.append(
                        {"filename": safe_filename, "error": "Файл пуст"}
                    )
                    continue

                file_dest.write_bytes(content)

                result = ClassificationResult(
                    doc_id=doc_id,
                    filename=safe_filename,
                    doc_type="unknown",
                    confidence=0.0,
                    metadata={"file_size": len(content)},
                )
                results.append(result)

                store.add(
                    DocumentInfo(
                        doc_id=doc_id,
                        filename=safe_filename,
                        doc_type=result.doc_type,
                        confidence=result.confidence,
                        file_path=str(file_dest),
                        created_at=result.classified_at,
                    )
                )

            except Exception as exc:
                fname = file.filename or "unknown"
                logger.error(
                    "batch_file_processing_failed",
                    filename=fname,
                    error=str(exc),
                )
                errors.append({"filename": fname, "error": str(exc)})

        logger.info(
            "batch_classification_complete",
            total=len(results),
            errors=len(errors),
        )
        return BatchClassificationResponse(
            results=results,
            total=len(results),
            errors=errors,
        )

    @app.get(
        "/api/v1/documents",
        response_model=DocumentListResponse,
        status_code=status.HTTP_200_OK,
    )
    async def list_documents(
        page: int = 1,
        page_size: int = 20,
    ) -> DocumentListResponse:
        """Получение списка обработанных документов.

        Args:
            page: Номер страницы (начиная с 1).
            page_size: Количество документов на странице.
        """
        if page < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Номер страницы должен быть >= 1.",
            )
        if page_size < 1 or page_size > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Размер страницы должен быть от 1 до 100.",
            )

        documents, total = store.list_all(page=page, page_size=page_size)

        return DocumentListResponse(
            documents=documents,
            total=total,
            page=page,
            page_size=page_size,
        )

    @app.get(
        "/api/v1/documents/{doc_id}",
        response_model=DocumentInfo,
        status_code=status.HTTP_200_OK,
        responses={404: {"model": ErrorResponse}},
    )
    async def get_document(doc_id: str) -> DocumentInfo:
        """Получение информации о конкретном документе.

        Args:
            doc_id: Уникальный идентификатор документа.
        """
        doc = store.get(doc_id)
        if doc is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Документ не найден: {doc_id}",
            )
        return doc

    @app.post(
        "/api/v1/feedback",
        response_model=FeedbackResponse,
        status_code=status.HTTP_201_CREATED,
        responses={
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
        },
    )
    async def submit_feedback(
        request: FeedbackRequest,
    ) -> FeedbackResponse:
        """Отправка обратной связи / исправления классификации.

        Позволяет пользователю указать правильный тип документа,
        если классификатор ошибся.
        """
        doc = store.get(request.doc_id)
        if doc is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Документ не найден: {request.doc_id}",
            )

        if not request.correct_type.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Поле correct_type не может быть пустым.",
            )

        feedback_id = str(uuid.uuid4())

        logger.info(
            "feedback_received",
            feedback_id=feedback_id,
            doc_id=request.doc_id,
            predicted_type=request.predicted_type,
            correct_type=request.correct_type,
        )

        return FeedbackResponse(
            id=feedback_id,
            doc_id=request.doc_id,
            status="recorded",
        )

    @app.get(
        "/api/v1/stats",
        response_model=StatsResponse,
        status_code=status.HTTP_200_OK,
    )
    async def get_stats() -> StatsResponse:
        """Получение статистики классификации."""
        documents_by_type = store.count_by_type()
        total_documents = sum(documents_by_type.values())

        return StatsResponse(
            total_documents=total_documents,
            documents_by_type=documents_by_type,
        )

    @app.get(
        "/api/v1/health",
        response_model=HealthResponse,
        status_code=status.HTTP_200_OK,
    )
    async def health_check() -> HealthResponse:
        """Проверка работоспособности сервиса."""
        return HealthResponse(
            status="ok",
            version="1.0.0",
        )

    logger.info(
        "api_server_configured",
        upload_dir=str(upload_path),
    )
    return app


# Экземпляр приложения для запуска через uvicorn
app = create_app()
