"""Классификатор на основе правил и регулярных выражений (Вариант A).

Быстрый и лёгкий классификатор, не требующий внешних зависимостей кроме
PyYAML. Загружает набор правил из ``config/classification_rules.yaml``,
проверяет паттерны в имени файла, ключевые слова в OCR-тексте и
регулярные выражения, вычисляет взвешенный балл для каждого типа
документа и возвращает тип с наибольшим баллом.

Дополнительно извлекает из текста стандартные реквизиты (ИНН, КПП,
ОГРН, БИК, дата, сумма, номер документа).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import structlog
import yaml

from classifiers.base import (
    DOCUMENT_TYPES,
    BaseClassifier,
    ClassificationResult,
)

logger = structlog.get_logger(__name__)

# ------------------------------------------------------------------
# Веса для разных источников совпадений
# ------------------------------------------------------------------

_FILENAME_MATCH_WEIGHT: float = 0.30
_KEYWORD_MATCH_WEIGHT: float = 0.20
_PATTERN_MATCH_WEIGHT: float = 0.25
_REQUIRED_FIELD_WEIGHT: float = 0.25

# ------------------------------------------------------------------
# Путь к конфигурации по умолчанию
# ------------------------------------------------------------------

_DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "config" / "classification_rules.yaml"


class RulesClassifier(BaseClassifier):
    """Классификатор документов на основе правил и регулярных выражений.

    Алгоритм:
        1. Проверяет паттерны в имени файла (``filename_patterns``).
        2. Ищет ключевые слова в OCR-тексте (``keywords``).
        3. Применяет регулярные выражения к тексту (``patterns``).
        4. Проверяет наличие обязательных полей (``required_fields``).
        5. Вычисляет взвешенный балл для каждого типа документа.
        6. Извлекает стандартные реквизиты через ``field_patterns``.
        7. Возвращает тип с наивысшим баллом или ``UNKNOWN``.
    """

    # ------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------

    def __init__(
        self,
        rules_path: str | Path | None = None,
        min_confidence: float = 0.15,
    ) -> None:
        """Инициализация классификатора на основе правил.

        Args:
            rules_path: Путь к YAML-файлу с правилами классификации.
                Если ``None`` — используется путь по умолчанию
                (``config/classification_rules.yaml``).
            min_confidence: Минимальный порог уверенности. Если балл ниже —
                возвращается ``UNKNOWN``.
        """
        super().__init__()
        self._rules_path = Path(rules_path) if rules_path else _DEFAULT_RULES_PATH
        self._min_confidence = min_confidence
        self._rules: dict[str, Any] = {}
        self._field_patterns: dict[str, dict[str, str]] = {}
        self._compiled_patterns: dict[str, list[re.Pattern[str]]] = {}
        self._compiled_filename_patterns: dict[str, list[re.Pattern[str]]] = {}
        self._compiled_field_patterns: dict[str, re.Pattern[str]] = {}

        self._load_rules()

    # ------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Имя классификатора."""
        return "RulesClassifier"

    # ------------------------------------------------------------------
    # Загрузка правил
    # ------------------------------------------------------------------

    def _load_rules(self) -> None:
        """Загрузить и скомпилировать правила из YAML-файла."""
        try:
            with open(self._rules_path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except FileNotFoundError:
            self._log.error(
                "Файл правил не найден",
                path=str(self._rules_path),
            )
            return
        except yaml.YAMLError as exc:
            self._log.error(
                "Ошибка парсинга YAML",
                path=str(self._rules_path),
                error=str(exc),
            )
            return

        self._rules = data.get("document_types", {})
        self._field_patterns = data.get("field_patterns", {})

        # Предкомпиляция регулярных выражений для типов документов
        for doc_key, rule in self._rules.items():
            # Паттерны в тексте
            compiled: list[re.Pattern[str]] = []
            for pat_str in rule.get("patterns", []):
                try:
                    compiled.append(re.compile(pat_str))
                except re.error as exc:
                    self._log.warning(
                        "Невалидный regex в правилах",
                        doc_type=doc_key,
                        pattern=pat_str,
                        error=str(exc),
                    )
            self._compiled_patterns[doc_key] = compiled

            # Паттерны в имени файла
            fn_compiled: list[re.Pattern[str]] = []
            for pat_str in rule.get("filename_patterns", []):
                try:
                    fn_compiled.append(re.compile(pat_str))
                except re.error as exc:
                    self._log.warning(
                        "Невалидный filename-regex в правилах",
                        doc_type=doc_key,
                        pattern=pat_str,
                        error=str(exc),
                    )
            self._compiled_filename_patterns[doc_key] = fn_compiled

        # Предкомпиляция паттернов для извлечения полей
        for field_key, field_info in self._field_patterns.items():
            pat_str = field_info.get("pattern", "")
            try:
                self._compiled_field_patterns[field_key] = re.compile(pat_str)
            except re.error as exc:
                self._log.warning(
                    "Невалидный field-regex",
                    field=field_key,
                    pattern=pat_str,
                    error=str(exc),
                )

        self._log.info(
            "Правила загружены",
            doc_types=len(self._rules),
            field_patterns=len(self._compiled_field_patterns),
        )

    # ------------------------------------------------------------------
    # Классификация
    # ------------------------------------------------------------------

    def classify(self, text: str, metadata: dict[str, Any]) -> ClassificationResult:
        """Классифицировать документ по правилам и regex-паттернам.

        Args:
            text: OCR-текст документа.
            metadata: Метаданные (ожидаются ключи ``filename``, ``file_size``
                и т.д.).

        Returns:
            ``ClassificationResult`` с типом документа и уверенностью.
        """
        if not self._rules:
            self._log.warning("Правила не загружены, возвращаем UNKNOWN")
            return ClassificationResult(
                doc_type="UNKNOWN",
                confidence=0.0,
                classifier_name=self.name,
                details={"error": "rules_not_loaded"},
            )

        filename = metadata.get("filename", "")
        text_lower = text.lower()

        scores: dict[str, float] = {}

        for doc_key, rule in self._rules.items():
            if doc_key == "unknown":
                continue

            score = self._score_document(
                doc_key=doc_key,
                rule=rule,
                text=text,
                text_lower=text_lower,
                filename=filename,
            )
            if score > 0.0:
                scores[doc_key] = score

        # Извлечение реквизитов
        extracted_fields = self._extract_fields(text)

        # Выбрать тип с наивысшим баллом
        if not scores:
            self._log.debug("Ни один тип не набрал баллов", filename=filename)
            return ClassificationResult(
                doc_type="UNKNOWN",
                confidence=0.0,
                classifier_name=self.name,
                details={"extracted_fields": extracted_fields, "scores": {}},
            )

        best_key = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_key]

        # Нормализация: ограничить уверенность диапазоном [0, 1]
        confidence = min(best_score, 1.0)

        # Маппинг ключей YAML → константы DOCUMENT_TYPES
        type_map: dict[str, str] = {
            "invoice": "INVOICE",
            "act": "ACT",
            "contract": "CONTRACT",
            "waybill": "WAYBILL",
            "payment_order": "PAYMENT_ORDER",
        }
        doc_type = type_map.get(best_key, "UNKNOWN")

        if confidence < self._min_confidence:
            doc_type = "UNKNOWN"
            confidence = 0.0

        self._log.info(
            "Классификация завершена",
            doc_type=doc_type,
            confidence=round(confidence, 3),
            filename=filename,
        )

        return ClassificationResult(
            doc_type=doc_type,
            confidence=round(confidence, 4),
            classifier_name=self.name,
            details={
                "scores": {
                    type_map.get(k, k): round(v, 4) for k, v in scores.items()
                },
                "extracted_fields": extracted_fields,
            },
        )

    # ------------------------------------------------------------------
    # Подсчёт баллов
    # ------------------------------------------------------------------

    def _score_document(
        self,
        doc_key: str,
        rule: dict[str, Any],
        text: str,
        text_lower: str,
        filename: str,
    ) -> float:
        """Вычислить взвешенный балл для одного типа документа.

        Args:
            doc_key: Ключ типа документа из YAML (например, ``invoice``).
            rule: Словарь правил для данного типа.
            text: Оригинальный OCR-текст.
            text_lower: Текст в нижнем регистре.
            filename: Имя файла документа.

        Returns:
            Взвешенный балл (float) от 0.0 до ~1.0.
        """
        weight = rule.get("weight", 1.0)
        component_scores: dict[str, float] = {}

        # 1. Паттерны в имени файла
        fn_patterns = self._compiled_filename_patterns.get(doc_key, [])
        if fn_patterns:
            fn_matches = sum(
                1 for pat in fn_patterns if pat.search(filename)
            )
            component_scores["filename"] = (
                min(fn_matches / max(len(fn_patterns), 1), 1.0)
                * _FILENAME_MATCH_WEIGHT
            )

        # 2. Ключевые слова в тексте
        keywords = rule.get("keywords", [])
        if keywords:
            kw_matches = sum(
                1 for kw in keywords if kw.lower() in text_lower
            )
            component_scores["keywords"] = (
                min(kw_matches / max(len(keywords), 1), 1.0)
                * _KEYWORD_MATCH_WEIGHT
            )

        # 3. Regex-паттерны в тексте
        text_patterns = self._compiled_patterns.get(doc_key, [])
        if text_patterns:
            pat_matches = sum(
                1 for pat in text_patterns if pat.search(text)
            )
            component_scores["patterns"] = (
                min(pat_matches / max(len(text_patterns), 1), 1.0)
                * _PATTERN_MATCH_WEIGHT
            )

        # 4. Обязательные поля
        required_fields = rule.get("required_fields", [])
        if required_fields:
            field_matches = sum(
                1
                for rf in required_fields
                if rf.lower() in text_lower
            )
            component_scores["required_fields"] = (
                min(field_matches / max(len(required_fields), 1), 1.0)
                * _REQUIRED_FIELD_WEIGHT
            )

        total = sum(component_scores.values()) * weight
        return total

    # ------------------------------------------------------------------
    # Извлечение полей
    # ------------------------------------------------------------------

    def _extract_fields(self, text: str) -> dict[str, str | list[str]]:
        """Извлечь стандартные реквизиты из текста документа.

        Использует скомпилированные паттерны из секции ``field_patterns``
        конфигурации.

        Args:
            text: OCR-текст документа.

        Returns:
            Словарь извлечённых полей. Для полей, встречающихся
            несколько раз (например, ИНН продавца и покупателя),
            возвращается список значений.
        """
        result: dict[str, str | list[str]] = {}

        for field_key, pattern in self._compiled_field_patterns.items():
            matches = pattern.findall(text)
            if not matches:
                continue

            # findall может вернуть кортежи при нескольких группах
            cleaned: list[str] = []
            for m in matches:
                if isinstance(m, tuple):
                    # Берём первую непустую группу
                    val = next((g for g in m if g), "")
                else:
                    val = m
                val = val.strip()
                if val and val not in cleaned:
                    cleaned.append(val)

            if len(cleaned) == 1:
                result[field_key] = cleaned[0]
            elif cleaned:
                result[field_key] = cleaned

        return result
