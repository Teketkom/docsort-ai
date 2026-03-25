"""LLM-классификатор документов через Ollama (Вариант D).

Отправляет OCR-текст документа на локальный сервер Ollama, который
запускает языковую модель для классификации. Результат парсится из
JSON-ответа модели.

Особенности:
    - Работает с локальным Ollama API (http://localhost:11434).
    - Настраиваемые модель, таймаут и температура.
    - Грациозная обработка недоступности Ollama.
    - Ограничение длины текста для экономии контекстного окна.
"""

from __future__ import annotations

import json
import re
from typing import Any

import structlog

from classifiers.base import (
    DOCUMENT_TYPE_LABELS,
    DOCUMENT_TYPES,
    BaseClassifier,
    ClassificationResult,
)

logger = structlog.get_logger(__name__)

# ------------------------------------------------------------------
# Константы
# ------------------------------------------------------------------

_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_MODEL = "qwen2.5:3b"
_DEFAULT_TIMEOUT = 120.0
_DEFAULT_TEMPERATURE = 0.1
_MAX_TEXT_LENGTH = 3000

# ------------------------------------------------------------------
# Системный промпт
# ------------------------------------------------------------------

_SYSTEM_PROMPT = """\
Ты — экспертная система классификации российских бухгалтерских документов.

Твоя задача — определить тип документа по тексту, полученному через OCR.

Допустимые типы документов:
- INVOICE — Счёт-фактура
- ACT — Акт выполненных работ
- CONTRACT — Договор
- WAYBILL — Товарная накладная ТОРГ-12
- PAYMENT_ORDER — Платёжное поручение
- UNKNOWN — Неопознанный документ

Ответь СТРОГО в формате JSON (без markdown-блоков, без пояснений):
{
  "doc_type": "ТИП_ДОКУМЕНТА",
  "confidence": 0.95,
  "reason": "Краткое обоснование на русском языке"
}

Правила:
- confidence — число от 0.0 до 1.0, отражающее уверенность.
- Если текст нечитаемый или не относится ни к одному типу — используй UNKNOWN.
- Обращай внимание на ключевые слова: ИНН, КПП, БИК, ТОРГ-12, счёт-фактура и т.д.
"""


class LLMClassifier(BaseClassifier):
    """Классификатор документов на основе LLM через Ollama.

    Отправляет OCR-текст документа в локальный Ollama API, парсит
    JSON-ответ модели и возвращает ``ClassificationResult``.
    """

    # ------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------

    def __init__(
        self,
        ollama_url: str = _DEFAULT_OLLAMA_URL,
        model: str = _DEFAULT_MODEL,
        timeout: float = _DEFAULT_TIMEOUT,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_text_length: int = _MAX_TEXT_LENGTH,
    ) -> None:
        """Инициализация LLM-классификатора.

        Args:
            ollama_url: Базовый URL Ollama API.
            model: Имя модели Ollama (например, ``qwen2.5:3b``).
            timeout: Таймаут HTTP-запроса в секундах.
            temperature: Температура генерации (0.0 — детерминировано,
                1.0 — максимально случайно).
            max_text_length: Максимальная длина текста, отправляемого
                в модель (символов).
        """
        super().__init__()
        self._ollama_url = ollama_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._temperature = temperature
        self._max_text_length = max_text_length
        self._httpx_available = self._check_httpx()

    # ------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Имя классификатора."""
        return "LLMClassifier"

    @property
    def model_name(self) -> str:
        """Имя используемой LLM-модели."""
        return self._model

    # ------------------------------------------------------------------
    # Проверки
    # ------------------------------------------------------------------

    @staticmethod
    def _check_httpx() -> bool:
        """Проверить доступность httpx."""
        try:
            import httpx  # noqa: F401

            return True
        except ImportError:
            return False

    def _is_ollama_available(self) -> bool:
        """Проверить доступность Ollama API.

        Returns:
            ``True``, если сервер отвечает.
        """
        if not self._httpx_available:
            return False

        import httpx

        try:
            response = httpx.get(
                f"{self._ollama_url}/api/tags",
                timeout=5.0,
            )
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, OSError):
            return False

    # ------------------------------------------------------------------
    # Классификация
    # ------------------------------------------------------------------

    def classify(self, text: str, metadata: dict[str, Any]) -> ClassificationResult:
        """Классифицировать документ с помощью LLM через Ollama.

        Args:
            text: OCR-текст документа.
            metadata: Метаданные документа.

        Returns:
            ``ClassificationResult`` с типом документа и уверенностью.
        """
        if not self._httpx_available:
            self._log.error("httpx не установлен, LLM-классификация невозможна")
            return self._error_result("httpx_not_installed")

        if not text.strip():
            self._log.debug("Пустой текст, возвращаем UNKNOWN")
            return ClassificationResult(
                doc_type="UNKNOWN",
                confidence=0.0,
                classifier_name=self.name,
                details={"reason": "empty_text"},
            )

        import httpx

        # Обрезать текст при необходимости
        truncated = text[: self._max_text_length]
        if len(text) > self._max_text_length:
            truncated += "\n[...текст обрезан...]"

        filename = metadata.get("filename", "неизвестно")
        user_prompt = (
            f"Имя файла: {filename}\n\n"
            f"Текст документа (OCR):\n{truncated}"
        )

        # Запрос к Ollama API
        payload = {
            "model": self._model,
            "prompt": user_prompt,
            "system": _SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": self._temperature,
            },
        }

        try:
            self._log.debug(
                "Отправка запроса в Ollama",
                model=self._model,
                text_length=len(truncated),
            )

            response = httpx.post(
                f"{self._ollama_url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()

            response_data = response.json()
            raw_response = response_data.get("response", "")

            return self._parse_llm_response(raw_response)

        except httpx.ConnectError:
            self._log.warning(
                "Ollama недоступен",
                url=self._ollama_url,
            )
            return self._error_result("ollama_unavailable")

        except httpx.TimeoutException:
            self._log.warning(
                "Таймаут запроса к Ollama",
                timeout=self._timeout,
                model=self._model,
            )
            return self._error_result("timeout")

        except httpx.HTTPStatusError as exc:
            self._log.error(
                "HTTP-ошибка Ollama",
                status_code=exc.response.status_code,
                detail=exc.response.text[:500],
            )
            return self._error_result(
                f"http_error_{exc.response.status_code}"
            )

        except Exception as exc:
            self._log.error(
                "Непредвиденная ошибка LLM-классификации",
                error=str(exc),
            )
            return self._error_result(f"unexpected_error: {exc}")

    # ------------------------------------------------------------------
    # Парсинг ответа
    # ------------------------------------------------------------------

    def _parse_llm_response(self, raw: str) -> ClassificationResult:
        """Распарсить JSON-ответ LLM.

        Поддерживает случаи, когда модель оборачивает JSON в
        markdown-блоки (```json ... ```) или добавляет текст до/после JSON.

        Args:
            raw: Сырой текстовый ответ модели.

        Returns:
            ``ClassificationResult`` на основе распарсенного ответа.
        """
        # Попытка 1: прямой парсинг
        parsed = self._try_parse_json(raw)

        # Попытка 2: извлечь JSON из markdown-блока
        if parsed is None:
            json_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```",
                raw,
                re.DOTALL,
            )
            if json_match:
                parsed = self._try_parse_json(json_match.group(1))

        # Попытка 3: найти первый JSON-объект в тексте
        if parsed is None:
            json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
            if json_match:
                parsed = self._try_parse_json(json_match.group(0))

        if parsed is None:
            self._log.warning(
                "Не удалось распарсить ответ LLM",
                raw_response=raw[:500],
            )
            return ClassificationResult(
                doc_type="UNKNOWN",
                confidence=0.0,
                classifier_name=self.name,
                details={
                    "error": "parse_failed",
                    "raw_response": raw[:500],
                },
            )

        # Извлечь doc_type и confidence
        doc_type = str(parsed.get("doc_type", "UNKNOWN")).upper().strip()
        confidence = parsed.get("confidence", 0.0)
        reason = parsed.get("reason", "")

        # Валидация doc_type
        if doc_type not in DOCUMENT_TYPES:
            # Попробовать сопоставить по русскому названию
            doc_type = self._match_russian_label(doc_type)

        # Валидация confidence
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(confidence, 1.0))
        except (TypeError, ValueError):
            confidence = 0.0

        self._log.info(
            "LLM-классификация завершена",
            doc_type=doc_type,
            confidence=round(confidence, 3),
            model=self._model,
        )

        return ClassificationResult(
            doc_type=doc_type,
            confidence=round(confidence, 4),
            classifier_name=self.name,
            details={
                "model": self._model,
                "reason": reason,
                "method": "llm",
            },
        )

    @staticmethod
    def _try_parse_json(text: str) -> dict[str, Any] | None:
        """Попытка распарсить строку как JSON.

        Args:
            text: Строка для парсинга.

        Returns:
            Словарь или ``None``, если парсинг не удался.
        """
        try:
            result = json.loads(text.strip())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    @staticmethod
    def _match_russian_label(text: str) -> str:
        """Сопоставить русское название типа с константой.

        Args:
            text: Текст для сопоставления (может быть русским названием).

        Returns:
            Код типа документа или ``UNKNOWN``.
        """
        text_lower = text.lower()

        # Обратный маппинг: русское название → код
        label_to_type: dict[str, str] = {
            label.lower(): code
            for code, label in DOCUMENT_TYPE_LABELS.items()
        }

        # Точное совпадение
        if text_lower in label_to_type:
            return label_to_type[text_lower]

        # Частичное совпадение
        for label, code in label_to_type.items():
            if label in text_lower or text_lower in label:
                return code

        return "UNKNOWN"

    # ------------------------------------------------------------------
    # Утилиты
    # ------------------------------------------------------------------

    def _error_result(self, error: str) -> ClassificationResult:
        """Создать результат-ошибку.

        Args:
            error: Описание ошибки.

        Returns:
            ``ClassificationResult`` с типом ``UNKNOWN`` и нулевой
            уверенностью.
        """
        return ClassificationResult(
            doc_type="UNKNOWN",
            confidence=0.0,
            classifier_name=self.name,
            details={"error": error},
        )
