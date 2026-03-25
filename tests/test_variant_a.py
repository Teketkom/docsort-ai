"""Тесты для Варианта A — классификатор на основе правил (regex + ключевые слова).

Проверяет корректность определения типов документов по OCR-тексту,
извлечение реквизитов и анализ имён файлов.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
import yaml

from classifiers.base import BaseClassifier, ClassificationResult, DOCUMENT_TYPES


# ------------------------------------------------------------------
# Реализация правилового классификатора для тестирования
# ------------------------------------------------------------------


class RulesClassifier(BaseClassifier):
    """Классификатор документов на основе регулярных выражений и ключевых слов.

    Загружает правила из YAML-конфигурации и последовательно проверяет
    текст документа на совпадение с паттернами каждого типа.
    """

    def __init__(self, rules_path: str | None = None) -> None:
        """Инициализация классификатора правил.

        Args:
            rules_path: Путь к файлу classification_rules.yaml.
                        Если не указан, используются встроенные правила.
        """
        super().__init__()
        self._rules: dict[str, dict[str, Any]] = {}
        self._field_patterns: dict[str, dict[str, str]] = {}

        if rules_path:
            self._load_rules_from_file(rules_path)
        else:
            self._load_default_rules()

    def _load_rules_from_file(self, path: str) -> None:
        """Загружает правила классификации из YAML-файла.

        Args:
            path: Путь к файлу правил.
        """
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self._rules = data.get("document_types", {})
        self._field_patterns = data.get("field_patterns", {})

    def _load_default_rules(self) -> None:
        """Загружает встроенные правила классификации."""
        self._rules = {
            "invoice": {
                "patterns": [
                    r"(?i)сч[её]т[\s-]*фактур[аы]",
                    r"(?i)счёт\s*№\s*\d+",
                    r"(?i)счет\s*№\s*\d+",
                    r"(?i)к\s+оплате",
                    r"(?i)итого\s+к\s+оплате",
                ],
                "keywords": ["счёт-фактура", "счет-фактура", "invoice"],
                "weight": 1.0,
                "filename_patterns": [r"(?i)сч[её]т", r"(?i)invoice", r"(?i)sf[-_]"],
            },
            "act": {
                "patterns": [
                    r"(?i)акт\s+(выполненных|оказанных)",
                    r"(?i)акт\s+при[её]мки",
                    r"(?i)акт\s*№\s*\d+",
                    r"(?i)исполнитель.*заказчик",
                    r"(?i)заказчик.*исполнитель",
                    r"(?i)претензий\s+.*не\s+имеет",
                    r"(?i)работы\s+выполнены\s+полностью",
                ],
                "keywords": ["акт выполненных работ", "акт оказанных услуг", "акт приёмки"],
                "weight": 1.0,
                "filename_patterns": [r"(?i)\bакт", r"(?i)(?<![a-z])act(?![a-z])"],
            },
            "contract": {
                "patterns": [
                    r"(?i)договор\s*№\s*\d+",
                    r"(?i)предмет\s+договора",
                    r"(?i)стороны\s+договорились",
                    r"(?i)настоящ(ий|им)\s+договор",
                    r"(?i)реквизиты\s+сторон",
                    r"(?i)подписи\s+сторон",
                ],
                "keywords": ["договор", "контракт", "соглашение"],
                "weight": 0.9,
                "filename_patterns": [r"(?i)договор", r"(?i)contract", r"(?i)dogovor"],
            },
            "waybill": {
                "patterns": [
                    r"(?i)торг[\s-]*12",
                    r"(?i)товарная\s+накладная",
                    r"(?i)грузоотправитель",
                    r"(?i)грузополучатель",
                    r"(?i)наименование\s+товара",
                    r"(?i)единица\s+измерения",
                ],
                "keywords": ["ТОРГ-12", "товарная накладная", "накладная"],
                "weight": 1.0,
                "filename_patterns": [r"(?i)торг", r"(?i)накладн", r"(?i)torg"],
            },
            "payment_order": {
                "patterns": [
                    r"(?i)плат[её]жное\s+поручение",
                    r"(?i)банк\s+получателя",
                    r"(?i)банк\s+плательщика",
                    r"(?i)\bБИК\b",
                    r"(?i)р/с|р\.с\.|расч[её]тный\s+сч[её]т",
                    r"(?i)корр?\.?\s*сч[её]т",
                ],
                "keywords": ["платёжное поручение", "платежное поручение"],
                "weight": 1.0,
                "filename_patterns": [r"(?i)плат", r"(?i)payment", r"(?i)п\.?п"],
            },
        }

        self._field_patterns = {
            "inn": {
                "pattern": r"(?i)инн\s*:?\s*(\d{10,12})",
                "description": "ИНН (10 или 12 цифр)",
            },
            "kpp": {
                "pattern": r"(?i)кпп\s*:?\s*(\d{9})",
                "description": "КПП (9 цифр)",
            },
            "bik": {
                "pattern": r"(?i)бик\s*:?\s*(\d{9})",
                "description": "БИК (9 цифр)",
            },
            "amount": {
                "pattern": r"(?i)(?:итого|сумма|всего)[\s:]*([\d\s]+[,\.]\d{2})",
                "description": "Сумма документа",
            },
            "doc_number": {
                "pattern": r"(?i)№\s*(\S+)",
                "description": "Номер документа",
            },
        }

    @property
    def name(self) -> str:
        """Имя классификатора."""
        return "rules"

    def classify(self, text: str, metadata: dict[str, Any] | None = None) -> ClassificationResult:
        """Классифицирует документ по тексту с использованием правил.

        Подсчитывает количество совпадений с паттернами и ключевыми словами
        для каждого типа документа. Тип с наибольшим количеством совпадений
        считается результатом. Уверенность рассчитывается как отношение
        найденных совпадений к общему числу паттернов.

        Args:
            text: OCR-текст документа.
            metadata: Метаданные (имя файла, путь и т.д.).

        Returns:
            Результат классификации с типом, уверенностью и деталями.
        """
        metadata = metadata or {}
        scores: dict[str, float] = {}
        match_details: dict[str, list[str]] = {}

        for doc_type, rules in self._rules.items():
            patterns: list[str] = rules.get("patterns", [])
            keywords: list[str] = rules.get("keywords", [])
            weight: float = rules.get("weight", 1.0)

            matched_patterns: list[str] = []
            score = 0.0

            # Проверка regex-паттернов
            for pattern in patterns:
                if re.search(pattern, text):
                    score += 1.0
                    matched_patterns.append(pattern)

            # Проверка ключевых слов
            text_lower = text.lower()
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 0.5
                    matched_patterns.append(f"keyword:{keyword}")

            total_checks = len(patterns) + len(keywords) * 0.5
            if total_checks > 0:
                scores[doc_type] = (score / total_checks) * weight
            else:
                scores[doc_type] = 0.0

            match_details[doc_type] = matched_patterns

        # Анализ имени файла (бонус к уверенности)
        filename = metadata.get("filename", "")
        if filename:
            for doc_type, rules in self._rules.items():
                for fp in rules.get("filename_patterns", []):
                    if re.search(fp, filename):
                        scores[doc_type] = scores.get(doc_type, 0.0) + 0.15
                        match_details.setdefault(doc_type, []).append(f"filename:{fp}")

        if not scores or max(scores.values()) == 0.0:
            return ClassificationResult(
                doc_type="UNKNOWN",
                confidence=0.0,
                classifier_name=self.name,
                details={"match_details": match_details},
            )

        best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
        confidence = min(scores[best_type], 1.0)

        # Извлечение полей
        extracted_fields = self.extract_fields(text)

        return ClassificationResult(
            doc_type=best_type.upper(),
            confidence=round(confidence, 4),
            classifier_name=self.name,
            details={
                "scores": {k: round(v, 4) for k, v in scores.items()},
                "match_details": match_details,
                "extracted_fields": extracted_fields,
            },
        )

    def extract_fields(self, text: str) -> dict[str, list[str]]:
        """Извлекает структурированные поля из текста документа.

        Args:
            text: OCR-текст документа.

        Returns:
            Словарь: имя поля -> список найденных значений.
        """
        fields: dict[str, list[str]] = {}
        for field_name, field_info in self._field_patterns.items():
            pattern = field_info["pattern"]
            matches = re.findall(pattern, text)
            if matches:
                fields[field_name] = matches
        return fields

    def classify_by_filename(self, filename: str) -> ClassificationResult:
        """Определяет тип документа по имени файла.

        Args:
            filename: Имя файла документа.

        Returns:
            Результат классификации на основе имени файла.
        """
        for doc_type, rules in self._rules.items():
            for pattern in rules.get("filename_patterns", []):
                if re.search(pattern, filename):
                    return ClassificationResult(
                        doc_type=doc_type.upper(),
                        confidence=0.6,
                        classifier_name=f"{self.name}_filename",
                        details={"matched_pattern": pattern, "filename": filename},
                    )

        return ClassificationResult(
            doc_type="UNKNOWN",
            confidence=0.0,
            classifier_name=f"{self.name}_filename",
            details={"filename": filename},
        )


# ------------------------------------------------------------------
# Фикстура классификатора
# ------------------------------------------------------------------


@pytest.fixture()
def rules_classifier() -> RulesClassifier:
    """Создаёт экземпляр RulesClassifier с правилами по умолчанию."""
    return RulesClassifier()


# ------------------------------------------------------------------
# Тесты классификации по тексту
# ------------------------------------------------------------------


class TestRulesClassification:
    """Тесты классификации документов по OCR-тексту."""

    def test_classify_invoice(
        self, rules_classifier: RulesClassifier, sample_invoice_text: str
    ) -> None:
        """Текст счёт-фактуры должен классифицироваться как INVOICE."""
        result = rules_classifier.classify(sample_invoice_text)

        assert result.doc_type == "INVOICE"
        assert result.confidence > 0.0
        assert result.classifier_name == "rules"

    def test_classify_act(
        self, rules_classifier: RulesClassifier, sample_act_text: str
    ) -> None:
        """Текст акта выполненных работ должен классифицироваться как ACT."""
        result = rules_classifier.classify(sample_act_text)

        assert result.doc_type == "ACT"
        assert result.confidence > 0.0
        assert result.classifier_name == "rules"

    def test_classify_contract(
        self, rules_classifier: RulesClassifier, sample_contract_text: str
    ) -> None:
        """Текст договора должен классифицироваться как CONTRACT."""
        result = rules_classifier.classify(sample_contract_text)

        assert result.doc_type == "CONTRACT"
        assert result.confidence > 0.0
        assert result.classifier_name == "rules"

    def test_classify_waybill(
        self, rules_classifier: RulesClassifier, sample_waybill_text: str
    ) -> None:
        """Текст товарной накладной ТОРГ-12 должен классифицироваться как WAYBILL."""
        result = rules_classifier.classify(sample_waybill_text)

        assert result.doc_type == "WAYBILL"
        assert result.confidence > 0.0
        assert result.classifier_name == "rules"

    def test_classify_payment_order(
        self, rules_classifier: RulesClassifier, sample_payment_order_text: str
    ) -> None:
        """Текст платёжного поручения должен классифицироваться как PAYMENT_ORDER."""
        result = rules_classifier.classify(sample_payment_order_text)

        assert result.doc_type == "PAYMENT_ORDER"
        assert result.confidence > 0.0
        assert result.classifier_name == "rules"

    def test_classify_unknown(self, rules_classifier: RulesClassifier) -> None:
        """Произвольный текст без признаков документов должен вернуть UNKNOWN."""
        random_text = (
            "Привет! Сегодня хорошая погода. Пойдём гулять в парк. "
            "Купим мороженое и посидим на лавочке."
        )
        result = rules_classifier.classify(random_text)

        assert result.doc_type == "UNKNOWN"
        assert result.confidence == 0.0


# ------------------------------------------------------------------
# Тесты извлечения полей
# ------------------------------------------------------------------


class TestFieldExtraction:
    """Тесты извлечения структурированных полей из текста."""

    def test_extract_inn(
        self, rules_classifier: RulesClassifier, sample_invoice_text: str
    ) -> None:
        """Должен корректно извлекать ИНН (10 или 12 цифр) из текста."""
        fields = rules_classifier.extract_fields(sample_invoice_text)

        assert "inn" in fields
        assert len(fields["inn"]) >= 1

        # Проверяем что все ИНН содержат 10 или 12 цифр
        for inn_value in fields["inn"]:
            assert len(inn_value) in (10, 12), (
                f"ИНН должен содержать 10 или 12 цифр, получено: {inn_value}"
            )
            assert inn_value.isdigit(), f"ИНН должен состоять из цифр: {inn_value}"

    def test_extract_kpp(
        self, rules_classifier: RulesClassifier, sample_invoice_text: str
    ) -> None:
        """Должен корректно извлекать КПП (9 цифр) из текста."""
        fields = rules_classifier.extract_fields(sample_invoice_text)

        assert "kpp" in fields
        assert len(fields["kpp"]) >= 1

        for kpp_value in fields["kpp"]:
            assert len(kpp_value) == 9
            assert kpp_value.isdigit()

    def test_extract_bik_from_payment_order(
        self, rules_classifier: RulesClassifier, sample_payment_order_text: str
    ) -> None:
        """Должен извлекать БИК из текста платёжного поручения."""
        fields = rules_classifier.extract_fields(sample_payment_order_text)

        assert "bik" in fields
        assert len(fields["bik"]) >= 1

        for bik_value in fields["bik"]:
            assert len(bik_value) == 9
            assert bik_value.isdigit()

    def test_extract_doc_number(
        self, rules_classifier: RulesClassifier, sample_invoice_text: str
    ) -> None:
        """Должен извлекать номер документа из текста."""
        fields = rules_classifier.extract_fields(sample_invoice_text)

        assert "doc_number" in fields
        assert len(fields["doc_number"]) >= 1


# ------------------------------------------------------------------
# Тесты анализа имён файлов
# ------------------------------------------------------------------


class TestFilenameAnalysis:
    """Тесты определения типа документа по имени файла."""

    @pytest.mark.parametrize(
        ("filename", "expected_type"),
        [
            ("Счёт-фактура_147.pdf", "INVOICE"),
            ("счет_от_15_03_2026.pdf", "INVOICE"),
            ("invoice_2026_03.pdf", "INVOICE"),
            ("sf-147.pdf", "INVOICE"),
            ("Акт_выполненных_работ.pdf", "ACT"),
            ("act_83.pdf", "ACT"),
            ("Договор_042.pdf", "CONTRACT"),
            ("contract_2026.pdf", "CONTRACT"),
            ("dogovor_postavki.pdf", "CONTRACT"),
            ("ТОРГ-12_256.pdf", "WAYBILL"),
            ("torg12_mart.pdf", "WAYBILL"),
            ("накладная_256.pdf", "WAYBILL"),
            ("платежное_поручение_1042.pdf", "PAYMENT_ORDER"),
            ("payment_order.pdf", "PAYMENT_ORDER"),
            ("пп_1042.pdf", "PAYMENT_ORDER"),
        ],
    )
    def test_filename_analysis(
        self,
        rules_classifier: RulesClassifier,
        filename: str,
        expected_type: str,
    ) -> None:
        """Должен определять тип документа по характерным паттернам в имени файла."""
        result = rules_classifier.classify_by_filename(filename)

        assert result.doc_type == expected_type, (
            f"Файл '{filename}': ожидался тип {expected_type}, "
            f"получен {result.doc_type}"
        )
        assert result.confidence > 0.0

    def test_filename_unknown(self, rules_classifier: RulesClassifier) -> None:
        """Неизвестное имя файла должно возвращать UNKNOWN."""
        result = rules_classifier.classify_by_filename("random_file_12345.pdf")

        assert result.doc_type == "UNKNOWN"
        assert result.confidence == 0.0

    def test_filename_boosts_classification(
        self, rules_classifier: RulesClassifier, sample_invoice_text: str
    ) -> None:
        """Имя файла должно повышать уверенность классификации."""
        result_no_filename = rules_classifier.classify(sample_invoice_text)
        result_with_filename = rules_classifier.classify(
            sample_invoice_text,
            metadata={"filename": "Счёт-фактура_147.pdf"},
        )

        assert result_with_filename.confidence >= result_no_filename.confidence


# ------------------------------------------------------------------
# Тесты результата классификации
# ------------------------------------------------------------------


class TestClassificationResult:
    """Тесты модели ClassificationResult."""

    def test_valid_result(self) -> None:
        """Валидный результат должен создаваться без ошибок."""
        result = ClassificationResult(
            doc_type="INVOICE",
            confidence=0.95,
            classifier_name="rules",
        )
        assert result.doc_type == "INVOICE"
        assert result.confidence == 0.95
        assert result.classifier_name == "rules"

    def test_invalid_doc_type(self) -> None:
        """Невалидный тип документа должен вызывать ValueError."""
        with pytest.raises(ValueError, match="Неизвестный тип документа"):
            ClassificationResult(
                doc_type="INVALID_TYPE",
                confidence=0.5,
                classifier_name="rules",
            )

    def test_confidence_out_of_range(self) -> None:
        """Уверенность за пределами [0, 1] должна вызывать ValueError."""
        with pytest.raises(ValueError, match="диапазоне"):
            ClassificationResult(
                doc_type="INVOICE",
                confidence=1.5,
                classifier_name="rules",
            )

        with pytest.raises(ValueError, match="диапазоне"):
            ClassificationResult(
                doc_type="INVOICE",
                confidence=-0.1,
                classifier_name="rules",
            )

    def test_all_document_types_recognized(self) -> None:
        """Все определённые типы документов должны быть валидными."""
        for doc_type in DOCUMENT_TYPES:
            result = ClassificationResult(
                doc_type=doc_type,
                confidence=0.5,
                classifier_name="test",
            )
            assert result.doc_type == doc_type
