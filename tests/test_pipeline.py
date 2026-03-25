"""Тесты пайплайна обработки документов DocSort AI.

Проверяет инициализацию пайплайна, каскадную логику классификации
(правила -> ML -> LLM), автоматическую сортировку по порогу уверенности
и эскалацию при низкой уверенности.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from classifiers.base import BaseClassifier, ClassificationResult


# ------------------------------------------------------------------
# Заглушки классификаторов для тестирования пайплайна
# ------------------------------------------------------------------


class StubRulesClassifier(BaseClassifier):
    """Заглушка правилового классификатора с настраиваемым результатом."""

    def __init__(
        self,
        doc_type: str = "UNKNOWN",
        confidence: float = 0.0,
    ) -> None:
        """Инициализация заглушки правилового классификатора.

        Args:
            doc_type: Тип документа, который будет возвращаться.
            confidence: Уверенность, которая будет возвращаться.
        """
        super().__init__()
        self._doc_type = doc_type
        self._confidence = confidence
        self.call_count: int = 0

    @property
    def name(self) -> str:
        """Имя классификатора."""
        return "rules"

    def classify(self, text: str, metadata: dict[str, Any] | None = None) -> ClassificationResult:
        """Возвращает предустановленный результат классификации."""
        self.call_count += 1
        return ClassificationResult(
            doc_type=self._doc_type,
            confidence=self._confidence,
            classifier_name=self.name,
        )


class StubMLClassifier(BaseClassifier):
    """Заглушка ML-классификатора с настраиваемым результатом."""

    def __init__(
        self,
        doc_type: str = "UNKNOWN",
        confidence: float = 0.0,
    ) -> None:
        """Инициализация заглушки ML-классификатора.

        Args:
            doc_type: Тип документа, который будет возвращаться.
            confidence: Уверенность, которая будет возвращаться.
        """
        super().__init__()
        self._doc_type = doc_type
        self._confidence = confidence
        self.call_count: int = 0

    @property
    def name(self) -> str:
        """Имя классификатора."""
        return "ml"

    def classify(self, text: str, metadata: dict[str, Any] | None = None) -> ClassificationResult:
        """Возвращает предустановленный результат классификации."""
        self.call_count += 1
        return ClassificationResult(
            doc_type=self._doc_type,
            confidence=self._confidence,
            classifier_name=self.name,
        )


class StubLLMClassifier(BaseClassifier):
    """Заглушка LLM-классификатора с настраиваемым результатом."""

    def __init__(
        self,
        doc_type: str = "INVOICE",
        confidence: float = 0.9,
    ) -> None:
        """Инициализация заглушки LLM-классификатора.

        Args:
            doc_type: Тип документа, который будет возвращаться.
            confidence: Уверенность, которая будет возвращаться.
        """
        super().__init__()
        self._doc_type = doc_type
        self._confidence = confidence
        self.call_count: int = 0

    @property
    def name(self) -> str:
        """Имя классификатора."""
        return "llm"

    def classify(self, text: str, metadata: dict[str, Any] | None = None) -> ClassificationResult:
        """Возвращает предустановленный результат классификации."""
        self.call_count += 1
        return ClassificationResult(
            doc_type=self._doc_type,
            confidence=self._confidence,
            classifier_name=self.name,
        )


# ------------------------------------------------------------------
# Конфигурация пайплайна
# ------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Конфигурация пайплайна классификации документов.

    Attributes:
        auto_sort_threshold: Порог уверенности для автоматической сортировки.
        cascade_threshold: Порог для передачи на следующий уровень каскада.
        rules_enabled: Включён ли правиловый классификатор.
        ml_enabled: Включён ли ML-классификатор.
        llm_enabled: Включён ли LLM-классификатор.
    """

    auto_sort_threshold: float = 0.85
    cascade_threshold: float = 0.7
    rules_enabled: bool = True
    ml_enabled: bool = True
    llm_enabled: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Создаёт конфигурацию из словаря.

        Args:
            data: Словарь с параметрами конфигурации.

        Returns:
            Экземпляр PipelineConfig.
        """
        classification = data.get("classification", {})
        return cls(
            auto_sort_threshold=classification.get("auto_sort_threshold", 0.85),
            cascade_threshold=classification.get("cascade_threshold", 0.7),
            rules_enabled=classification.get("rules", {}).get("enabled", True),
            ml_enabled=classification.get("ml", {}).get("enabled", True),
            llm_enabled=classification.get("llm", {}).get("enabled", False),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Загружает конфигурацию из YAML-файла.

        Args:
            path: Путь к файлу настроек.

        Returns:
            Экземпляр PipelineConfig.
        """
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


# ------------------------------------------------------------------
# Пайплайн классификации
# ------------------------------------------------------------------


@dataclass
class ClassificationPipeline:
    """Каскадный пайплайн классификации документов.

    Реализует каскадную логику: если первый классификатор (правила) не даёт
    достаточной уверенности, запрос передаётся ML-классификатору, а затем
    при необходимости — LLM-классификатору.

    Attributes:
        config: Конфигурация пайплайна.
        classifiers: Упорядоченный список классификаторов в порядке каскада.
    """

    config: PipelineConfig
    classifiers: list[BaseClassifier] = field(default_factory=list)

    def add_classifier(self, classifier: BaseClassifier) -> None:
        """Добавляет классификатор в каскад.

        Args:
            classifier: Экземпляр классификатора.
        """
        self.classifiers.append(classifier)

    def classify(self, text: str, metadata: dict[str, Any] | None = None) -> ClassificationResult:
        """Выполняет каскадную классификацию документа.

        Последовательно вызывает классификаторы из каскада. Если уверенность
        результата >= cascade_threshold, классификация считается завершённой.
        Иначе запрос передаётся следующему классификатору.

        Args:
            text: OCR-текст документа.
            metadata: Метаданные документа.

        Returns:
            Лучший результат классификации из каскада.
        """
        metadata = metadata or {}
        best_result: ClassificationResult | None = None

        for classifier in self.classifiers:
            result = classifier.classify(text, metadata)

            # Запоминаем лучший результат
            if best_result is None or result.confidence > best_result.confidence:
                best_result = result

            # Если уверенность достаточна, прекращаем каскад
            if result.confidence >= self.config.cascade_threshold:
                return result

        # Возвращаем лучший найденный результат
        if best_result is not None:
            return best_result

        return ClassificationResult(
            doc_type="UNKNOWN",
            confidence=0.0,
            classifier_name="pipeline",
            details={"reason": "no_classifiers_available"},
        )

    def should_auto_sort(self, result: ClassificationResult) -> bool:
        """Проверяет, можно ли автоматически отсортировать документ.

        Args:
            result: Результат классификации.

        Returns:
            True, если уверенность >= auto_sort_threshold и тип != UNKNOWN.
        """
        return (
            result.confidence >= self.config.auto_sort_threshold
            and result.doc_type != "UNKNOWN"
        )

    def needs_escalation(self, result: ClassificationResult) -> bool:
        """Проверяет, нужна ли эскалация (ручная проверка).

        Args:
            result: Результат классификации.

        Returns:
            True, если уверенность < cascade_threshold или тип UNKNOWN.
        """
        return (
            result.confidence < self.config.cascade_threshold
            or result.doc_type == "UNKNOWN"
        )


# ------------------------------------------------------------------
# Фикстуры
# ------------------------------------------------------------------


@pytest.fixture()
def pipeline_config() -> PipelineConfig:
    """Создаёт конфигурацию пайплайна с настройками по умолчанию."""
    return PipelineConfig(
        auto_sort_threshold=0.85,
        cascade_threshold=0.7,
        rules_enabled=True,
        ml_enabled=True,
        llm_enabled=True,
    )


@pytest.fixture()
def pipeline(pipeline_config: PipelineConfig) -> ClassificationPipeline:
    """Создаёт пустой пайплайн для тестирования."""
    return ClassificationPipeline(config=pipeline_config)


# ------------------------------------------------------------------
# Тесты инициализации
# ------------------------------------------------------------------


class TestPipelineInitialization:
    """Тесты инициализации пайплайна."""

    def test_pipeline_initialization(self, pipeline_config: PipelineConfig) -> None:
        """Пайплайн должен корректно создаваться с конфигурацией."""
        pipeline = ClassificationPipeline(config=pipeline_config)

        assert pipeline.config.auto_sort_threshold == 0.85
        assert pipeline.config.cascade_threshold == 0.7
        assert pipeline.config.rules_enabled is True
        assert pipeline.config.ml_enabled is True
        assert pipeline.classifiers == []

    def test_pipeline_from_yaml(self, tmp_config: Path) -> None:
        """Пайплайн должен корректно загружать конфигурацию из YAML."""
        config = PipelineConfig.from_yaml(tmp_config)
        pipeline = ClassificationPipeline(config=config)

        assert pipeline.config.auto_sort_threshold == 0.85
        assert pipeline.config.cascade_threshold == 0.7
        assert pipeline.config.rules_enabled is True
        assert pipeline.config.ml_enabled is True
        assert pipeline.config.llm_enabled is False

    def test_add_classifiers(self, pipeline: ClassificationPipeline) -> None:
        """Классификаторы должны корректно добавляться в каскад."""
        rules = StubRulesClassifier()
        ml = StubMLClassifier()
        llm = StubLLMClassifier()

        pipeline.add_classifier(rules)
        pipeline.add_classifier(ml)
        pipeline.add_classifier(llm)

        assert len(pipeline.classifiers) == 3
        assert pipeline.classifiers[0].name == "rules"
        assert pipeline.classifiers[1].name == "ml"
        assert pipeline.classifiers[2].name == "llm"

    def test_config_from_dict(self) -> None:
        """Конфигурация должна создаваться из словаря."""
        data = {
            "classification": {
                "auto_sort_threshold": 0.9,
                "cascade_threshold": 0.75,
                "rules": {"enabled": True},
                "ml": {"enabled": False},
                "llm": {"enabled": True},
            }
        }

        config = PipelineConfig.from_dict(data)

        assert config.auto_sort_threshold == 0.9
        assert config.cascade_threshold == 0.75
        assert config.rules_enabled is True
        assert config.ml_enabled is False
        assert config.llm_enabled is True


# ------------------------------------------------------------------
# Тесты каскадной логики
# ------------------------------------------------------------------


class TestCascadeLogic:
    """Тесты каскадной логики классификации."""

    def test_cascade_stops_on_high_confidence(
        self, pipeline: ClassificationPipeline
    ) -> None:
        """Каскад должен остановиться, когда правила дают высокую уверенность.

        Если первый классификатор (правила) возвращает уверенность выше
        cascade_threshold, ML и LLM не должны вызываться.
        """
        rules = StubRulesClassifier(doc_type="INVOICE", confidence=0.95)
        ml = StubMLClassifier(doc_type="INVOICE", confidence=0.8)
        llm = StubLLMClassifier(doc_type="INVOICE", confidence=0.9)

        pipeline.add_classifier(rules)
        pipeline.add_classifier(ml)
        pipeline.add_classifier(llm)

        result = pipeline.classify("текст документа")

        assert result.doc_type == "INVOICE"
        assert result.confidence == 0.95
        assert result.classifier_name == "rules"
        assert rules.call_count == 1
        assert ml.call_count == 0
        assert llm.call_count == 0

    def test_cascade_rules_to_ml(self, pipeline: ClassificationPipeline) -> None:
        """При низкой уверенности правил каскад должен перейти к ML.

        Если правила дают уверенность ниже cascade_threshold,
        вызывается ML-классификатор.
        """
        rules = StubRulesClassifier(doc_type="INVOICE", confidence=0.4)
        ml = StubMLClassifier(doc_type="INVOICE", confidence=0.85)
        llm = StubLLMClassifier(doc_type="INVOICE", confidence=0.9)

        pipeline.add_classifier(rules)
        pipeline.add_classifier(ml)
        pipeline.add_classifier(llm)

        result = pipeline.classify("текст документа")

        assert result.doc_type == "INVOICE"
        assert result.confidence == 0.85
        assert result.classifier_name == "ml"
        assert rules.call_count == 1
        assert ml.call_count == 1
        assert llm.call_count == 0

    def test_cascade_rules_to_ml_to_llm(
        self, pipeline: ClassificationPipeline
    ) -> None:
        """Полный каскад: правила -> ML -> LLM.

        Если ни правила, ни ML не дают достаточной уверенности,
        вызывается LLM-классификатор.
        """
        rules = StubRulesClassifier(doc_type="UNKNOWN", confidence=0.2)
        ml = StubMLClassifier(doc_type="CONTRACT", confidence=0.5)
        llm = StubLLMClassifier(doc_type="CONTRACT", confidence=0.88)

        pipeline.add_classifier(rules)
        pipeline.add_classifier(ml)
        pipeline.add_classifier(llm)

        result = pipeline.classify("текст сложного документа")

        assert result.doc_type == "CONTRACT"
        assert result.confidence == 0.88
        assert result.classifier_name == "llm"
        assert rules.call_count == 1
        assert ml.call_count == 1
        assert llm.call_count == 1

    def test_cascade_returns_best_result_on_all_low(
        self, pipeline: ClassificationPipeline
    ) -> None:
        """Если все классификаторы дают низкую уверенность, возвращается лучший.

        Каскад должен выбрать результат с максимальной уверенностью
        среди всех классификаторов.
        """
        rules = StubRulesClassifier(doc_type="UNKNOWN", confidence=0.1)
        ml = StubMLClassifier(doc_type="ACT", confidence=0.45)
        llm = StubLLMClassifier(doc_type="ACT", confidence=0.6)

        pipeline.add_classifier(rules)
        pipeline.add_classifier(ml)
        pipeline.add_classifier(llm)

        result = pipeline.classify("неопознанный документ")

        assert result.confidence == 0.6
        assert result.doc_type == "ACT"

    def test_empty_pipeline(self, pipeline: ClassificationPipeline) -> None:
        """Пайплайн без классификаторов должен возвращать UNKNOWN."""
        result = pipeline.classify("любой текст")

        assert result.doc_type == "UNKNOWN"
        assert result.confidence == 0.0


# ------------------------------------------------------------------
# Тесты автосортировки
# ------------------------------------------------------------------


class TestAutoSortThreshold:
    """Тесты порога автоматической сортировки."""

    def test_auto_sort_above_threshold(
        self, pipeline: ClassificationPipeline
    ) -> None:
        """Документ с уверенностью выше порога должен автосортироваться."""
        result = ClassificationResult(
            doc_type="INVOICE",
            confidence=0.92,
            classifier_name="rules",
        )

        assert pipeline.should_auto_sort(result) is True

    def test_auto_sort_at_threshold(
        self, pipeline: ClassificationPipeline
    ) -> None:
        """Документ с уверенностью ровно на пороге должен автосортироваться."""
        result = ClassificationResult(
            doc_type="INVOICE",
            confidence=0.85,
            classifier_name="rules",
        )

        assert pipeline.should_auto_sort(result) is True

    def test_auto_sort_below_threshold(
        self, pipeline: ClassificationPipeline
    ) -> None:
        """Документ с уверенностью ниже порога не должен автосортироваться."""
        result = ClassificationResult(
            doc_type="INVOICE",
            confidence=0.75,
            classifier_name="rules",
        )

        assert pipeline.should_auto_sort(result) is False

    def test_auto_sort_unknown_type(
        self, pipeline: ClassificationPipeline
    ) -> None:
        """Документ с типом UNKNOWN не должен автосортироваться даже при высокой уверенности."""
        result = ClassificationResult(
            doc_type="UNKNOWN",
            confidence=0.95,
            classifier_name="rules",
        )

        assert pipeline.should_auto_sort(result) is False

    def test_auto_sort_pipeline_integration(
        self, pipeline: ClassificationPipeline
    ) -> None:
        """Интеграционный тест: классификация + решение об автосортировке."""
        rules = StubRulesClassifier(doc_type="WAYBILL", confidence=0.92)
        pipeline.add_classifier(rules)

        result = pipeline.classify("ТОРГ-12 товарная накладная")

        assert pipeline.should_auto_sort(result) is True
        assert result.doc_type == "WAYBILL"


# ------------------------------------------------------------------
# Тесты эскалации при низкой уверенности
# ------------------------------------------------------------------


class TestLowConfidenceEscalation:
    """Тесты эскалации (передачи на ручную проверку) при низкой уверенности."""

    def test_escalation_on_low_confidence(
        self, pipeline: ClassificationPipeline
    ) -> None:
        """Низкая уверенность должна вызывать эскалацию."""
        result = ClassificationResult(
            doc_type="INVOICE",
            confidence=0.3,
            classifier_name="rules",
        )

        assert pipeline.needs_escalation(result) is True

    def test_no_escalation_on_high_confidence(
        self, pipeline: ClassificationPipeline
    ) -> None:
        """Высокая уверенность не должна вызывать эскалацию."""
        result = ClassificationResult(
            doc_type="INVOICE",
            confidence=0.95,
            classifier_name="ml",
        )

        assert pipeline.needs_escalation(result) is False

    def test_escalation_on_unknown_type(
        self, pipeline: ClassificationPipeline
    ) -> None:
        """Тип UNKNOWN всегда должен вызывать эскалацию."""
        result = ClassificationResult(
            doc_type="UNKNOWN",
            confidence=0.0,
            classifier_name="rules",
        )

        assert pipeline.needs_escalation(result) is True

    def test_low_confidence_triggers_next_classifier(
        self, pipeline: ClassificationPipeline
    ) -> None:
        """Низкая уверенность первого классификатора должна запускать следующий.

        Правила дают низкую уверенность -> вызывается ML.
        ML даёт достаточную уверенность -> результат принимается.
        """
        rules = StubRulesClassifier(doc_type="ACT", confidence=0.3)
        ml = StubMLClassifier(doc_type="ACT", confidence=0.88)

        pipeline.add_classifier(rules)
        pipeline.add_classifier(ml)

        result = pipeline.classify("Акт выполненных работ")

        # ML дал достаточную уверенность
        assert result.classifier_name == "ml"
        assert result.confidence == 0.88
        assert not pipeline.needs_escalation(result)

        # Оба классификатора были вызваны
        assert rules.call_count == 1
        assert ml.call_count == 1

    def test_all_classifiers_low_triggers_escalation(
        self, pipeline: ClassificationPipeline
    ) -> None:
        """Если все классификаторы дали низкую уверенность, нужна эскалация."""
        rules = StubRulesClassifier(doc_type="UNKNOWN", confidence=0.1)
        ml = StubMLClassifier(doc_type="CONTRACT", confidence=0.4)
        llm = StubLLMClassifier(doc_type="CONTRACT", confidence=0.55)

        pipeline.add_classifier(rules)
        pipeline.add_classifier(ml)
        pipeline.add_classifier(llm)

        result = pipeline.classify("непонятный документ")

        # Лучший результат всё ещё ниже порога эскалации
        assert pipeline.needs_escalation(result) is True
        assert rules.call_count == 1
        assert ml.call_count == 1
        assert llm.call_count == 1

    def test_escalation_boundary(self, pipeline: ClassificationPipeline) -> None:
        """Граничный случай: уверенность ровно на cascade_threshold."""
        result_at_threshold = ClassificationResult(
            doc_type="INVOICE",
            confidence=0.7,
            classifier_name="ml",
        )
        # cascade_threshold = 0.7, уверенность == 0.7 -> не нужна эскалация
        assert pipeline.needs_escalation(result_at_threshold) is False

        result_just_below = ClassificationResult(
            doc_type="INVOICE",
            confidence=0.69,
            classifier_name="ml",
        )
        # 0.69 < 0.7 -> нужна эскалация
        assert pipeline.needs_escalation(result_just_below) is True
