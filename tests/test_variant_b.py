"""Тесты для Варианта B — ML-классификатор (TF-IDF + SVM).

Проверяет обучение модели на тренировочных данных, сохранение и загрузку
модели, поведение без обученной модели и корректность диапазона уверенности.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from classifiers.base import BaseClassifier, ClassificationResult


# ------------------------------------------------------------------
# Реализация ML-классификатора для тестирования
# ------------------------------------------------------------------


class MLClassifier(BaseClassifier):
    """ML-классификатор документов на основе TF-IDF + SVM.

    Использует TF-IDF-векторизацию текста и линейный SVM для классификации.
    Модель можно обучить, сохранить и загрузить из файла.
    """

    def __init__(self, model_path: str | None = None) -> None:
        """Инициализация ML-классификатора.

        Args:
            model_path: Путь к файлу сохранённой модели (.pkl).
                        Если указан, модель загружается автоматически.
        """
        super().__init__()
        self._model: Pipeline | None = None
        self._model_path: str | None = model_path
        self._is_trained: bool = False

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    @property
    def name(self) -> str:
        """Имя классификатора."""
        return "ml"

    @property
    def is_trained(self) -> bool:
        """Проверяет, обучена ли модель."""
        return self._is_trained

    def train(self, texts: list[str], labels: list[str]) -> dict[str, Any]:
        """Обучает модель TF-IDF + SVM на предоставленных данных.

        Args:
            texts: Список текстов документов для обучения.
            labels: Список меток типов документов (соответствует texts).

        Returns:
            Словарь с метриками обучения (количество классов, образцов и т.д.).

        Raises:
            ValueError: Если количество текстов не совпадает с количеством меток
                        или данных недостаточно.
        """
        if len(texts) != len(labels):
            raise ValueError(
                f"Количество текстов ({len(texts)}) не совпадает "
                f"с количеством меток ({len(labels)})"
            )

        if len(texts) < 2:
            raise ValueError("Для обучения необходимо минимум 2 образца")

        unique_labels = set(labels)
        if len(unique_labels) < 2:
            raise ValueError("Для обучения необходимо минимум 2 различных класса")

        self._model = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                strip_accents="unicode",
            )),
            ("svm", LinearSVC(
                max_iter=10000,
                C=1.0,
                class_weight="balanced",
            )),
        ])

        self._model.fit(texts, labels)
        self._is_trained = True

        self._log.info(
            "model_trained",
            n_samples=len(texts),
            n_classes=len(unique_labels),
            classes=sorted(unique_labels),
        )

        return {
            "n_samples": len(texts),
            "n_classes": len(unique_labels),
            "classes": sorted(unique_labels),
        }

    def classify(self, text: str, metadata: dict[str, Any] | None = None) -> ClassificationResult:
        """Классифицирует документ с помощью обученной ML-модели.

        Args:
            text: OCR-текст документа.
            metadata: Метаданные документа (не используются в ML-классификаторе).

        Returns:
            Результат классификации. Если модель не обучена, возвращает UNKNOWN
            с нулевой уверенностью.
        """
        if not self._is_trained or self._model is None:
            return ClassificationResult(
                doc_type="UNKNOWN",
                confidence=0.0,
                classifier_name=self.name,
                details={"reason": "model_not_trained"},
            )

        prediction = self._model.predict([text])[0]

        # Вычисляем уверенность через decision_function
        svm: LinearSVC = self._model.named_steps["svm"]
        decision = svm.decision_function(
            self._model.named_steps["tfidf"].transform([text])
        )

        # Нормализуем decision_function в диапазон [0, 1] через сигмоиду
        import numpy as np

        if decision.ndim == 1:
            # Бинарная классификация
            raw_confidence = float(1.0 / (1.0 + np.exp(-abs(decision[0]))))
        else:
            # Мультиклассовая классификация
            max_decision = float(np.max(decision))
            raw_confidence = float(1.0 / (1.0 + np.exp(-max_decision)))

        confidence = round(min(max(raw_confidence, 0.0), 1.0), 4)

        return ClassificationResult(
            doc_type=prediction,
            confidence=confidence,
            classifier_name=self.name,
            details={
                "predicted_class": prediction,
                "raw_confidence": raw_confidence,
            },
        )

    def save_model(self, path: str) -> None:
        """Сохраняет обученную модель в файл.

        Args:
            path: Путь для сохранения модели (.pkl).

        Raises:
            RuntimeError: Если модель не обучена.
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("Модель не обучена. Сначала вызовите метод train().")

        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, "wb") as f:
            pickle.dump(self._model, f)

        self._log.info("model_saved", path=str(model_path))

    def load_model(self, path: str) -> None:
        """Загружает обученную модель из файла.

        Args:
            path: Путь к файлу модели (.pkl).

        Raises:
            FileNotFoundError: Если файл модели не найден.
        """
        model_path = Path(path)

        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {path}")

        with open(model_path, "rb") as f:
            self._model = pickle.load(f)  # noqa: S301

        self._is_trained = True
        self._log.info("model_loaded", path=str(model_path))


# ------------------------------------------------------------------
# Фикстуры
# ------------------------------------------------------------------


@pytest.fixture()
def ml_classifier() -> MLClassifier:
    """Создаёт необученный экземпляр MLClassifier."""
    return MLClassifier()


@pytest.fixture()
def training_data() -> tuple[list[str], list[str]]:
    """Возвращает тренировочные данные: тексты и метки документов.

    Генерирует по несколько образцов каждого типа для обучения модели.
    """
    texts: list[str] = []
    labels: list[str] = []

    # Счета-фактуры
    invoice_samples = [
        "Счёт-фактура № 100 от 01.01.2026 ИНН 7707123456 КПП 770701001 "
        "Итого к оплате 150000.00 руб. НДС 20% 30000.00",
        "СЧЁТ-ФАКТУРА № 201 продавец ООО Ромашка ИНН 5001234567 "
        "покупатель АО Прогресс итого к оплате 500000.00",
        "Счет-фактура номер 305 от 15 марта 2026 ИНН продавца 7720456789 "
        "КПП 772001001 сумма НДС итого к оплате",
        "Счёт-фактура № 412 ИНН 7743567890 КПП 774301001 "
        "наименование товара модуль обработки данных к оплате 250000.00",
    ]
    texts.extend(invoice_samples)
    labels.extend(["INVOICE"] * len(invoice_samples))

    # Акты
    act_samples = [
        "Акт выполненных работ № 50 исполнитель ООО СофтДев заказчик ПАО Сбербанк "
        "ИНН 7707083893 разработка модуля аналитики итого 350000.00",
        "АКТ оказанных услуг № 88 исполнитель заказчик "
        "тестирование отладка документация итого 555000.00",
        "Акт приёмки выполненных работ исполнитель ООО Ромашка "
        "заказчик АО Прогресс работы выполнены в срок",
        "Акт № 123 выполненных работ оказанных услуг исполнитель "
        "заказчик претензий по качеству не имеет итого",
    ]
    texts.extend(act_samples)
    labels.extend(["ACT"] * len(act_samples))

    # Договоры
    contract_samples = [
        "Договор № 2026-У/042 на оказание услуг предмет договора "
        "стороны договорились реквизиты сторон подписи сторон",
        "ДОГОВОР поставки № 34/2026 настоящим договором стороны "
        "предмет договора обязанности сторон реквизиты",
        "Договор оказания услуг предмет договора исполнитель обязуется "
        "заказчик оплачивает реквизиты сторон подписи",
        "Контракт соглашение договор № 789 предмет настоящего договора "
        "права и обязанности сторон реквизиты подписи",
    ]
    texts.extend(contract_samples)
    labels.extend(["CONTRACT"] * len(contract_samples))

    # Накладные ТОРГ-12
    waybill_samples = [
        "ТОРГ-12 товарная накладная № 256 грузоотправитель ООО Склад "
        "грузополучатель АО РитейлМарт наименование товара единица измерения",
        "Товарная накладная ТОРГ-12 грузоотправитель грузополучатель "
        "наименование товара бумага А4 картридж единица измерения штука",
        "Накладная ТОРГ 12 № 300 отпуск разрешил грузоотправитель "
        "грузополучатель наименование товара количество сумма",
        "ТОРГ-12 товарная накладная грузоотправитель ООО Складские Технологии "
        "грузополучатель поставщик плательщик единица измерения",
    ]
    texts.extend(waybill_samples)
    labels.extend(["WAYBILL"] * len(waybill_samples))

    # Платёжные поручения
    payment_samples = [
        "Платёжное поручение № 1042 банк получателя АО Альфа-Банк "
        "БИК 044525593 корр счёт 30101810200000000593 р/с назначение платежа",
        "ПЛАТЕЖНОЕ ПОРУЧЕНИЕ банк плательщика ПАО ВТБ БИК 044525745 "
        "расчётный счёт корреспондентский счёт назначение платежа",
        "Платёжное поручение № 500 БИК банк получателя банк плательщика "
        "р/с корр. счёт сумма назначение платежа НДС",
        "Платежное поручение № 777 банк получателя БИК 044525225 "
        "корр счет расчетный счет назначение платежа оплата по счету",
    ]
    texts.extend(payment_samples)
    labels.extend(["PAYMENT_ORDER"] * len(payment_samples))

    return texts, labels


# ------------------------------------------------------------------
# Тесты
# ------------------------------------------------------------------


class TestMLClassifier:
    """Тесты ML-классификатора (TF-IDF + SVM)."""

    def test_train_and_classify(
        self,
        ml_classifier: MLClassifier,
        training_data: tuple[list[str], list[str]],
    ) -> None:
        """Обучает модель на тренировочных данных и проверяет классификацию.

        Модель должна корректно классифицировать текст, похожий на
        тренировочные образцы.
        """
        texts, labels = training_data
        metrics = ml_classifier.train(texts, labels)

        assert metrics["n_samples"] == len(texts)
        assert metrics["n_classes"] == 5
        assert ml_classifier.is_trained is True

        # Классифицируем текст, похожий на счёт-фактуру
        invoice_text = (
            "Счёт-фактура № 999 ИНН 7707777777 КПП 770701001 "
            "итого к оплате 999000.00 руб. НДС 20%"
        )
        result = ml_classifier.classify(invoice_text)

        assert result.doc_type == "INVOICE"
        assert result.confidence > 0.0
        assert result.classifier_name == "ml"

    def test_save_and_load_model(
        self,
        ml_classifier: MLClassifier,
        training_data: tuple[list[str], list[str]],
        tmp_path: Path,
    ) -> None:
        """Обученная модель должна сохраняться и загружаться из файла.

        После загрузки модель должна выдавать те же результаты.
        """
        texts, labels = training_data
        ml_classifier.train(texts, labels)

        model_path = str(tmp_path / "test_model.pkl")
        ml_classifier.save_model(model_path)

        assert Path(model_path).exists()
        assert Path(model_path).stat().st_size > 0

        # Загружаем модель в новый классификатор
        loaded_classifier = MLClassifier(model_path=model_path)

        assert loaded_classifier.is_trained is True

        # Проверяем что загруженная модель работает
        test_text = (
            "Акт выполненных работ № 100 исполнитель заказчик "
            "работы выполнены в полном объёме итого 100000.00"
        )
        original_result = ml_classifier.classify(test_text)
        loaded_result = loaded_classifier.classify(test_text)

        assert original_result.doc_type == loaded_result.doc_type

    def test_classify_without_model(self, ml_classifier: MLClassifier) -> None:
        """Классификация без обученной модели должна возвращать UNKNOWN.

        Необученный классификатор не должен падать с ошибкой, а должен
        корректно возвращать результат с типом UNKNOWN и нулевой уверенностью.
        """
        assert ml_classifier.is_trained is False

        result = ml_classifier.classify("Любой текст документа для проверки")

        assert result.doc_type == "UNKNOWN"
        assert result.confidence == 0.0
        assert result.classifier_name == "ml"
        assert result.details.get("reason") == "model_not_trained"

    def test_confidence_range(
        self,
        ml_classifier: MLClassifier,
        training_data: tuple[list[str], list[str]],
    ) -> None:
        """Уверенность классификации должна быть в диапазоне от 0.0 до 1.0.

        Проверяет на различных текстах, что значение confidence всегда
        находится в корректном диапазоне.
        """
        texts, labels = training_data
        ml_classifier.train(texts, labels)

        test_texts = [
            "Счёт-фактура № 500 ИНН 7707123456 КПП 770701001 итого к оплате",
            "Акт выполненных работ исполнитель заказчик итого",
            "Договор № 100 предмет договора стороны реквизиты",
            "ТОРГ-12 товарная накладная грузоотправитель грузополучатель",
            "Платёжное поручение БИК банк получателя р/с",
            "Просто какой-то произвольный текст без документов",
            "",
            "Один",
        ]

        for text in test_texts:
            result = ml_classifier.classify(text)
            assert 0.0 <= result.confidence <= 1.0, (
                f"Уверенность {result.confidence} для текста '{text[:50]}...' "
                f"вне диапазона [0.0, 1.0]"
            )

    def test_train_validation_errors(self, ml_classifier: MLClassifier) -> None:
        """Обучение с некорректными данными должно вызывать ValueError."""
        # Разная длина текстов и меток
        with pytest.raises(ValueError, match="не совпадает"):
            ml_classifier.train(["текст1", "текст2"], ["INVOICE"])

        # Слишком мало образцов
        with pytest.raises(ValueError, match="минимум 2 образца"):
            ml_classifier.train(["текст1"], ["INVOICE"])

        # Один класс
        with pytest.raises(ValueError, match="минимум 2 различных класса"):
            ml_classifier.train(
                ["текст1", "текст2"],
                ["INVOICE", "INVOICE"],
            )

    def test_save_without_training(self, ml_classifier: MLClassifier, tmp_path: Path) -> None:
        """Попытка сохранить необученную модель должна вызывать RuntimeError."""
        with pytest.raises(RuntimeError, match="не обучена"):
            ml_classifier.save_model(str(tmp_path / "empty.pkl"))

    def test_load_nonexistent_model(self, ml_classifier: MLClassifier) -> None:
        """Попытка загрузить несуществующую модель должна вызывать FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ml_classifier.load_model("/nonexistent/path/model.pkl")

    def test_classify_all_types(
        self,
        ml_classifier: MLClassifier,
        training_data: tuple[list[str], list[str]],
    ) -> None:
        """Обученная модель должна корректно классифицировать все типы документов."""
        texts, labels = training_data
        ml_classifier.train(texts, labels)

        type_texts = {
            "INVOICE": "Счёт-фактура № 800 ИНН КПП итого к оплате НДС 20%",
            "ACT": "Акт выполненных работ оказанных услуг исполнитель заказчик итого",
            "CONTRACT": "Договор № 500 предмет договора стороны договорились реквизиты подписи",
            "WAYBILL": "ТОРГ-12 товарная накладная грузоотправитель грузополучатель единица измерения",
            "PAYMENT_ORDER": "Платёжное поручение банк получателя БИК корр счёт р/с назначение",
        }

        for expected_type, text in type_texts.items():
            result = ml_classifier.classify(text)
            assert result.doc_type == expected_type, (
                f"Ожидался тип {expected_type}, получен {result.doc_type} "
                f"для текста: '{text[:60]}...'"
            )
