"""Гибридный каскадный классификатор документов.

Объединяет несколько классификаторов в каскадную цепочку, где каждый
следующий уровень вызывается только при недостаточной уверенности
предыдущего. Порядок по умолчанию:

    1. **RulesClassifier** — быстрый, бесплатный, не требует моделей.
    2. **MLClassifier** — средняя скорость, требует обученной модели.
    3. **LLMClassifier** — медленный, требует запущенного Ollama.

Каждый классификатор в каскаде может быть включён или отключён через
конфигурацию. Порог каскада (``cascade_threshold``) определяет,
когда результат считается достаточно уверенным.
"""

from __future__ import annotations

from typing import Any

import structlog

from classifiers.base import BaseClassifier, ClassificationResult

logger = structlog.get_logger(__name__)


class HybridClassifier(BaseClassifier):
    """Гибридный каскадный классификатор документов.

    Последовательно применяет классификаторы, пока не будет достигнута
    достаточная уверенность или не будут исчерпаны все варианты.
    Возвращает результат с наивысшей уверенностью среди всех
    сработавших классификаторов.
    """

    # ------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------

    def __init__(
        self,
        cascade_threshold: float = 0.7,
        enable_rules: bool = True,
        enable_ml: bool = True,
        enable_llm: bool = False,
        rules_kwargs: dict[str, Any] | None = None,
        ml_kwargs: dict[str, Any] | None = None,
        llm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Инициализация гибридного каскадного классификатора.

        Args:
            cascade_threshold: Порог уверенности для каскада. Если
                классификатор возвращает уверенность >= порога —
                каскад останавливается.
            enable_rules: Включить RulesClassifier (Вариант A).
            enable_ml: Включить MLClassifier (Вариант B).
            enable_llm: Включить LLMClassifier (Вариант D).
            rules_kwargs: Дополнительные аргументы для RulesClassifier.
            ml_kwargs: Дополнительные аргументы для MLClassifier.
            llm_kwargs: Дополнительные аргументы для LLMClassifier.
        """
        super().__init__()

        self._cascade_threshold = cascade_threshold
        self._enable_rules = enable_rules
        self._enable_ml = enable_ml
        self._enable_llm = enable_llm

        self._rules_kwargs = rules_kwargs or {}
        self._ml_kwargs = ml_kwargs or {}
        self._llm_kwargs = llm_kwargs or {}

        # Ленивая инициализация классификаторов
        self._classifiers: list[BaseClassifier] = []
        self._initialized = False

    # ------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Имя классификатора."""
        return "HybridClassifier"

    @property
    def cascade_threshold(self) -> float:
        """Текущий порог каскада."""
        return self._cascade_threshold

    @cascade_threshold.setter
    def cascade_threshold(self, value: float) -> None:
        """Установить порог каскада.

        Args:
            value: Новый порог (от 0.0 до 1.0).
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Порог каскада должен быть в [0.0, 1.0], получено: {value}"
            )
        self._cascade_threshold = value

    @property
    def enabled_classifiers(self) -> list[str]:
        """Список имён включённых классификаторов."""
        self._ensure_initialized()
        return [clf.name for clf in self._classifiers]

    # ------------------------------------------------------------------
    # Инициализация
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        """Ленивая инициализация классификаторов каскада.

        Каждый классификатор создаётся при первом обращении, чтобы
        избежать ненужных импортов и загрузки моделей.
        """
        if self._initialized:
            return

        self._classifiers = []

        # Уровень 1: RulesClassifier (быстрый, бесплатный)
        if self._enable_rules:
            try:
                from classifiers.variant_a_rules import RulesClassifier

                clf = RulesClassifier(**self._rules_kwargs)
                self._classifiers.append(clf)
                self._log.info("RulesClassifier добавлен в каскад")
            except Exception as exc:
                self._log.error(
                    "Ошибка инициализации RulesClassifier",
                    error=str(exc),
                )

        # Уровень 2: MLClassifier (средняя скорость)
        if self._enable_ml:
            try:
                from classifiers.variant_b_ml import MLClassifier

                clf = MLClassifier(**self._ml_kwargs)
                self._classifiers.append(clf)
                self._log.info("MLClassifier добавлен в каскад")
            except Exception as exc:
                self._log.error(
                    "Ошибка инициализации MLClassifier",
                    error=str(exc),
                )

        # Уровень 3: LLMClassifier (медленный, мощный)
        if self._enable_llm:
            try:
                from classifiers.variant_d_llm import LLMClassifier

                clf = LLMClassifier(**self._llm_kwargs)
                self._classifiers.append(clf)
                self._log.info("LLMClassifier добавлен в каскад")
            except Exception as exc:
                self._log.error(
                    "Ошибка инициализации LLMClassifier",
                    error=str(exc),
                )

        self._initialized = True

        self._log.info(
            "Каскад инициализирован",
            classifiers=[clf.name for clf in self._classifiers],
            threshold=self._cascade_threshold,
        )

    # ------------------------------------------------------------------
    # Классификация
    # ------------------------------------------------------------------

    def classify(self, text: str, metadata: dict[str, Any]) -> ClassificationResult:
        """Классифицировать документ каскадом классификаторов.

        Логика каскада:
            1. Вызвать первый классификатор (RulesClassifier).
            2. Если ``confidence >= cascade_threshold`` — вернуть результат.
            3. Иначе — вызвать следующий классификатор.
            4. После прохода всех — вернуть результат с наивысшей
               уверенностью.

        Args:
            text: OCR-текст документа.
            metadata: Метаданные документа.

        Returns:
            ``ClassificationResult`` с наилучшим результатом.
        """
        self._ensure_initialized()

        if not self._classifiers:
            self._log.warning("Каскад пуст, нет доступных классификаторов")
            return ClassificationResult(
                doc_type="UNKNOWN",
                confidence=0.0,
                classifier_name=self.name,
                details={"error": "no_classifiers_available"},
            )

        all_results: list[ClassificationResult] = []
        cascade_log: list[dict[str, Any]] = []

        for clf in self._classifiers:
            try:
                self._log.debug(
                    "Каскад: вызов классификатора",
                    classifier=clf.name,
                )

                result = clf.classify(text, metadata)
                all_results.append(result)

                cascade_log.append({
                    "classifier": clf.name,
                    "doc_type": result.doc_type,
                    "confidence": result.confidence,
                })

                self._log.debug(
                    "Каскад: результат получен",
                    classifier=clf.name,
                    doc_type=result.doc_type,
                    confidence=round(result.confidence, 3),
                )

                # Если уверенность достаточна — прервать каскад
                if result.confidence >= self._cascade_threshold:
                    self._log.info(
                        "Каскад: порог достигнут, остановка",
                        classifier=clf.name,
                        doc_type=result.doc_type,
                        confidence=round(result.confidence, 3),
                        threshold=self._cascade_threshold,
                    )
                    return ClassificationResult(
                        doc_type=result.doc_type,
                        confidence=result.confidence,
                        classifier_name=self.name,
                        details={
                            "winning_classifier": clf.name,
                            "cascade_log": cascade_log,
                            "original_details": result.details,
                        },
                    )

            except Exception as exc:
                self._log.error(
                    "Каскад: ошибка классификатора",
                    classifier=clf.name,
                    error=str(exc),
                )
                cascade_log.append({
                    "classifier": clf.name,
                    "error": str(exc),
                })

        # Ни один классификатор не достиг порога — выбрать лучший
        return self._select_best_result(all_results, cascade_log)

    # ------------------------------------------------------------------
    # Выбор лучшего результата
    # ------------------------------------------------------------------

    def _select_best_result(
        self,
        results: list[ClassificationResult],
        cascade_log: list[dict[str, Any]],
    ) -> ClassificationResult:
        """Выбрать лучший результат из всех полученных.

        Критерий: наивысшая уверенность. При равной уверенности
        предпочитается результат, отличный от ``UNKNOWN``.

        Args:
            results: Список результатов от всех классификаторов.
            cascade_log: Журнал каскада для включения в детали.

        Returns:
            Лучший ``ClassificationResult`` или ``UNKNOWN``.
        """
        if not results:
            return ClassificationResult(
                doc_type="UNKNOWN",
                confidence=0.0,
                classifier_name=self.name,
                details={
                    "error": "all_classifiers_failed",
                    "cascade_log": cascade_log,
                },
            )

        # Отсортировать: сначала по уверенности (убывание),
        # затем не-UNKNOWN предпочтительнее
        sorted_results = sorted(
            results,
            key=lambda r: (
                r.confidence,
                0 if r.doc_type == "UNKNOWN" else 1,
            ),
            reverse=True,
        )

        best = sorted_results[0]

        self._log.info(
            "Каскад завершён: выбран лучший результат",
            doc_type=best.doc_type,
            confidence=round(best.confidence, 3),
            winning_classifier=best.classifier_name,
            total_classifiers=len(results),
        )

        return ClassificationResult(
            doc_type=best.doc_type,
            confidence=best.confidence,
            classifier_name=self.name,
            details={
                "winning_classifier": best.classifier_name,
                "cascade_log": cascade_log,
                "below_threshold": True,
                "original_details": best.details,
            },
        )
