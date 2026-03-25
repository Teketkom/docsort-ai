"""ML-классификатор на основе TF-IDF + SVM (Вариант B).

Использует scikit-learn для построения пайплайна
«TF-IDF-векторизация → SVM-классификация». При недоступности SVM
автоматически переключается на RandomForest. Модель сохраняется
и загружается через joblib.

Если модель не обучена — делегирует классификацию ``RulesClassifier``
в качестве фолбэка.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import structlog

from classifiers.base import (
    DOCUMENT_TYPES,
    BaseClassifier,
    ClassificationResult,
)

logger = structlog.get_logger(__name__)

# ------------------------------------------------------------------
# Ленивый импорт тяжёлых зависимостей
# ------------------------------------------------------------------

_sklearn_available: bool | None = None
_joblib_available: bool | None = None


def _check_sklearn() -> bool:
    """Проверить доступность scikit-learn."""
    global _sklearn_available  # noqa: PLW0603
    if _sklearn_available is None:
        try:
            import sklearn  # noqa: F401

            _sklearn_available = True
        except ImportError:
            _sklearn_available = False
    return _sklearn_available


def _check_joblib() -> bool:
    """Проверить доступность joblib."""
    global _joblib_available  # noqa: PLW0603
    if _joblib_available is None:
        try:
            import joblib  # noqa: F401

            _joblib_available = True
        except ImportError:
            _joblib_available = False
    return _joblib_available


# ------------------------------------------------------------------
# Пути по умолчанию
# ------------------------------------------------------------------

_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "tfidf_svm.pkl"
)


class MLClassifier(BaseClassifier):
    """Классификатор документов на основе TF-IDF + SVM.

    Обучается на корпусе размеченных документов. Если модель не обучена
    или scikit-learn недоступен — автоматически возвращает ``UNKNOWN``
    либо делегирует ``RulesClassifier``.

    Attributes:
        model_path: Путь к файлу сохранённой модели.
        is_trained: Флаг, обучена ли модель.
    """

    # ------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_path: str | Path | None = None,
        fallback_to_rules: bool = True,
        min_training_samples: int = 50,
    ) -> None:
        """Инициализация ML-классификатора.

        Args:
            model_path: Путь для сохранения/загрузки модели. Если ``None`` —
                используется путь по умолчанию.
            fallback_to_rules: Использовать ``RulesClassifier`` при
                отсутствии обученной модели.
            min_training_samples: Минимальное количество образцов для
                обучения. Если данных меньше — обучение отклоняется.
        """
        super().__init__()
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._fallback_to_rules = fallback_to_rules
        self._min_training_samples = min_training_samples
        self._pipeline: Any = None
        self._is_trained = False
        self._label_classes: list[str] = []
        self._rules_fallback: BaseClassifier | None = None

        # Попробовать загрузить существующую модель
        if self._model_path.exists():
            self.load_model(self._model_path)

    # ------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Имя классификатора."""
        return "MLClassifier"

    @property
    def is_trained(self) -> bool:
        """Обучена ли модель."""
        return self._is_trained

    # ------------------------------------------------------------------
    # Обучение
    # ------------------------------------------------------------------

    def train(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
    ) -> dict[str, Any]:
        """Обучить модель на корпусе размеченных документов.

        Args:
            texts: Список OCR-текстов документов.
            labels: Список меток (типов документов из ``DOCUMENT_TYPES``).

        Returns:
            Словарь с метриками обучения (accuracy, количество образцов,
            тип модели).

        Raises:
            RuntimeError: Если scikit-learn недоступен.
            ValueError: Если данных недостаточно или метки невалидны.
        """
        if not _check_sklearn():
            raise RuntimeError(
                "scikit-learn не установлен. "
                "Установите: pip install scikit-learn"
            )

        # Валидация
        if len(texts) != len(labels):
            raise ValueError(
                f"Количество текстов ({len(texts)}) не совпадает "
                f"с количеством меток ({len(labels)})"
            )

        if len(texts) < self._min_training_samples:
            raise ValueError(
                f"Недостаточно образцов для обучения: {len(texts)} < "
                f"{self._min_training_samples}. Нужно минимум "
                f"{self._min_training_samples}."
            )

        # Проверка валидности меток
        invalid_labels = set(labels) - set(DOCUMENT_TYPES)
        if invalid_labels:
            raise ValueError(
                f"Невалидные метки: {invalid_labels}. "
                f"Допустимые: {DOCUMENT_TYPES}"
            )

        self._log.info(
            "Начало обучения ML-модели",
            samples=len(texts),
            unique_labels=len(set(labels)),
        )

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC

        # Попробовать SVM, при ошибке — RandomForest
        model_type = "LinearSVC"
        try:
            pipeline = Pipeline([
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=10_000,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        strip_accents="unicode",
                    ),
                ),
                (
                    "clf",
                    LinearSVC(
                        C=1.0,
                        max_iter=5000,
                        class_weight="balanced",
                    ),
                ),
            ])
            pipeline.fit(texts, labels)
        except Exception as exc:
            self._log.warning(
                "SVM не удалось обучить, переход на RandomForest",
                error=str(exc),
            )
            model_type = "RandomForest"
            pipeline = Pipeline([
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=10_000,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        strip_accents="unicode",
                    ),
                ),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=100,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ])
            pipeline.fit(texts, labels)

        # Кросс-валидация
        try:
            cv_scores = cross_val_score(
                pipeline,
                texts,
                labels,
                cv=min(5, len(set(labels))),
                scoring="accuracy",
            )
            mean_accuracy = float(cv_scores.mean())
        except Exception as exc:
            self._log.warning(
                "Ошибка кросс-валидации",
                error=str(exc),
            )
            mean_accuracy = -1.0

        self._pipeline = pipeline
        self._is_trained = True
        self._label_classes = sorted(set(labels))

        metrics = {
            "model_type": model_type,
            "samples": len(texts),
            "unique_labels": len(set(labels)),
            "cv_accuracy": round(mean_accuracy, 4),
        }

        self._log.info("Модель обучена", **metrics)
        return metrics

    # ------------------------------------------------------------------
    # Классификация
    # ------------------------------------------------------------------

    def classify(self, text: str, metadata: dict[str, Any]) -> ClassificationResult:
        """Классифицировать документ с помощью ML-модели.

        Если модель не обучена — делегирует ``RulesClassifier`` (при
        ``fallback_to_rules=True``) или возвращает ``UNKNOWN``.

        Args:
            text: OCR-текст документа.
            metadata: Метаданные документа.

        Returns:
            ``ClassificationResult`` с типом документа и уверенностью.
        """
        if not self._is_trained or self._pipeline is None:
            return self._handle_no_model(text, metadata)

        if not text.strip():
            self._log.debug("Пустой текст, возвращаем UNKNOWN")
            return ClassificationResult(
                doc_type="UNKNOWN",
                confidence=0.0,
                classifier_name=self.name,
                details={"reason": "empty_text"},
            )

        try:
            predicted_label = self._pipeline.predict([text])[0]

            # Попробовать получить вероятности
            confidence = self._get_confidence(text, predicted_label)

            if predicted_label not in DOCUMENT_TYPES:
                self._log.warning(
                    "Модель вернула неизвестный тип",
                    predicted=predicted_label,
                )
                predicted_label = "UNKNOWN"
                confidence = 0.0

            self._log.info(
                "ML-классификация завершена",
                doc_type=predicted_label,
                confidence=round(confidence, 3),
            )

            return ClassificationResult(
                doc_type=predicted_label,
                confidence=round(confidence, 4),
                classifier_name=self.name,
                details={"method": "ml_prediction"},
            )
        except Exception as exc:
            self._log.error(
                "Ошибка ML-классификации",
                error=str(exc),
            )
            return self._handle_no_model(text, metadata)

    def _get_confidence(self, text: str, predicted_label: str) -> float:
        """Получить уверенность предсказания.

        Для моделей с ``predict_proba`` (RandomForest) — возвращает
        максимальную вероятность. Для SVM — использует ``decision_function``
        с сигмоидным преобразованием.

        Args:
            text: OCR-текст документа.
            predicted_label: Предсказанная метка.

        Returns:
            Уверенность от 0.0 до 1.0.
        """
        import numpy as np

        clf = self._pipeline.named_steps["clf"]

        # Модели с predict_proba (RandomForest)
        if hasattr(clf, "predict_proba"):
            try:
                proba = self._pipeline.predict_proba([text])[0]
                return float(np.max(proba))
            except Exception:
                pass

        # LinearSVC — decision_function + sigmoid
        if hasattr(clf, "decision_function"):
            try:
                decision = self._pipeline.decision_function([text])[0]
                if isinstance(decision, np.ndarray):
                    # Мультикласс: выбрать максимальное значение
                    max_val = float(np.max(decision))
                else:
                    max_val = float(decision)
                # Сигмоидное преобразование
                confidence = 1.0 / (1.0 + np.exp(-max_val))
                return float(confidence)
            except Exception:
                pass

        # Фолбэк
        return 0.5

    def _handle_no_model(
        self,
        text: str,
        metadata: dict[str, Any],
    ) -> ClassificationResult:
        """Обработка случая, когда модель не обучена.

        Args:
            text: OCR-текст документа.
            metadata: Метаданные документа.

        Returns:
            Результат от ``RulesClassifier`` или ``UNKNOWN``.
        """
        if self._fallback_to_rules:
            self._log.info("Модель не обучена, фолбэк на RulesClassifier")
            if self._rules_fallback is None:
                from classifiers.variant_a_rules import RulesClassifier

                self._rules_fallback = RulesClassifier()
            result = self._rules_fallback.classify(text, metadata)
            return ClassificationResult(
                doc_type=result.doc_type,
                confidence=result.confidence,
                classifier_name=self.name,
                details={
                    "fallback": "RulesClassifier",
                    "original_details": result.details,
                },
            )

        return ClassificationResult(
            doc_type="UNKNOWN",
            confidence=0.0,
            classifier_name=self.name,
            details={"reason": "model_not_trained"},
        )

    # ------------------------------------------------------------------
    # Сохранение / загрузка модели
    # ------------------------------------------------------------------

    def save_model(self, path: str | Path | None = None) -> Path:
        """Сохранить обученную модель на диск.

        Args:
            path: Путь для сохранения. Если ``None`` — используется
                ``self._model_path``.

        Returns:
            Путь к сохранённому файлу.

        Raises:
            RuntimeError: Если модель не обучена или joblib недоступен.
        """
        if not self._is_trained or self._pipeline is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите train().")

        if not _check_joblib():
            raise RuntimeError(
                "joblib не установлен. Установите: pip install joblib"
            )

        import joblib

        save_path = Path(path) if path else self._model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "pipeline": self._pipeline,
            "label_classes": self._label_classes,
        }
        joblib.dump(model_data, save_path)

        self._log.info("Модель сохранена", path=str(save_path))
        return save_path

    def load_model(self, path: str | Path | None = None) -> None:
        """Загрузить модель с диска.

        Args:
            path: Путь к файлу модели. Если ``None`` — используется
                ``self._model_path``.

        Raises:
            FileNotFoundError: Если файл модели не найден.
        """
        if not _check_joblib():
            self._log.error("joblib не установлен, загрузка невозможна")
            return

        import joblib

        load_path = Path(path) if path else self._model_path

        if not load_path.exists():
            self._log.warning("Файл модели не найден", path=str(load_path))
            return

        try:
            model_data = joblib.load(load_path)
            self._pipeline = model_data["pipeline"]
            self._label_classes = model_data.get("label_classes", [])
            self._is_trained = True
            self._log.info(
                "Модель загружена",
                path=str(load_path),
                labels=self._label_classes,
            )
        except Exception as exc:
            self._log.error(
                "Ошибка загрузки модели",
                path=str(load_path),
                error=str(exc),
            )
            self._is_trained = False
