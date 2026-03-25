"""Пакет классификаторов документов DocSort AI.

Содержит реализации четырёх вариантов классификации и гибридный
каскадный классификатор:

    - **Вариант A** (``RulesClassifier``): правила и регулярные выражения.
    - **Вариант B** (``MLClassifier``): TF-IDF + SVM / RandomForest.
    - **Вариант C** (``NeuralClassifier``): мультимодальная нейросеть
      (MobileNetV3 + SBERT + метаданные).
    - **Вариант D** (``LLMClassifier``): LLM через Ollama API.
    - **Гибрид** (``HybridClassifier``): каскад классификаторов с
      настраиваемыми порогами.

Все классификаторы наследуют ``BaseClassifier`` и возвращают
``ClassificationResult``.
"""

from classifiers.base import (
    DOCUMENT_TYPE_LABELS,
    DOCUMENT_TYPES,
    BaseClassifier,
    ClassificationResult,
)
from classifiers.hybrid import HybridClassifier
from classifiers.variant_a_rules import RulesClassifier
from classifiers.variant_b_ml import MLClassifier
from classifiers.variant_c_neural import NeuralClassifier
from classifiers.variant_d_llm import LLMClassifier

__all__ = [
    # Базовые классы и типы
    "BaseClassifier",
    "ClassificationResult",
    "DOCUMENT_TYPES",
    "DOCUMENT_TYPE_LABELS",
    # Классификаторы
    "RulesClassifier",
    "MLClassifier",
    "NeuralClassifier",
    "LLMClassifier",
    "HybridClassifier",
]
