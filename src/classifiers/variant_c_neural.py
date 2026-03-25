"""Нейросетевой классификатор документов (Вариант C).

Комбинирует три ветви признаков:
    1. **Визуальная**: изображение документа обрабатывается через
       MobileNetV3 (ONNX Runtime) и даёт вектор размерности [512].
    2. **Текстовая**: OCR-текст кодируется через SBERT
       (sentence-transformers) и даёт вектор размерности [384].
    3. **Метаданные**: имя файла, количество страниц и размер файла
       преобразуются в вектор размерности [16].

Три вектора конкатенируются и подаются на финальный классификатор
(полносвязный слой или LightGBM/LogisticRegression). Если какой-либо
компонент недоступен (нет модели, нет ONNX Runtime и т.д.) — он
пропускается, и классификация продолжается на доступных признаках.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
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


def _try_import_onnxruntime() -> Any:
    """Попытка импортировать onnxruntime."""
    try:
        import onnxruntime

        return onnxruntime
    except ImportError:
        return None


def _try_import_sentence_transformers() -> Any:
    """Попытка импортировать sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer
    except ImportError:
        return None


def _try_import_pil() -> Any:
    """Попытка импортировать PIL/Pillow."""
    try:
        from PIL import Image

        return Image
    except ImportError:
        return None


# ------------------------------------------------------------------
# Размерности векторов
# ------------------------------------------------------------------

_VISUAL_DIM = 512
_TEXT_DIM = 384
_META_DIM = 16
_TOTAL_DIM = _VISUAL_DIM + _TEXT_DIM + _META_DIM

# ------------------------------------------------------------------
# Пути по умолчанию
# ------------------------------------------------------------------

_DEFAULT_VISUAL_MODEL = (
    Path(__file__).resolve().parents[2] / "models" / "mobilenet_v3.onnx"
)
_DEFAULT_TEXT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_DEFAULT_HEAD_MODEL = (
    Path(__file__).resolve().parents[2] / "models" / "neural_head.pkl"
)


class NeuralClassifier(BaseClassifier):
    """Нейросетевой мультимодальный классификатор документов.

    Объединяет визуальные, текстовые и мета-признаки для классификации.
    Грациозно обрабатывает отсутствие моделей или зависимостей, отключая
    недоступные ветви и продолжая работу на оставшихся.
    """

    # ------------------------------------------------------------------
    # Конструктор
    # ------------------------------------------------------------------

    def __init__(
        self,
        visual_model_path: str | Path | None = None,
        text_model_name: str | None = None,
        head_model_path: str | Path | None = None,
    ) -> None:
        """Инициализация нейросетевого классификатора.

        Args:
            visual_model_path: Путь к ONNX-модели MobileNetV3 для
                извлечения визуальных признаков. Если ``None`` —
                используется путь по умолчанию.
            text_model_name: Имя или путь к SBERT-модели для текстовых
                эмбеддингов. Если ``None`` — используется модель по
                умолчанию.
            head_model_path: Путь к обученному классификационному
                «головному» слою. Если ``None`` — используется путь
                по умолчанию.
        """
        super().__init__()

        self._visual_model_path = (
            Path(visual_model_path) if visual_model_path else _DEFAULT_VISUAL_MODEL
        )
        self._text_model_name = text_model_name or _DEFAULT_TEXT_MODEL
        self._head_model_path = (
            Path(head_model_path) if head_model_path else _DEFAULT_HEAD_MODEL
        )

        # Ленивые ссылки на загруженные модели
        self._onnx_session: Any = None
        self._sbert_model: Any = None
        self._head_model: Any = None
        self._head_classes: list[str] = []

        # Флаги доступности
        self._visual_available = False
        self._text_available = False
        self._head_available = False

        self._init_models()

    # ------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Имя классификатора."""
        return "NeuralClassifier"

    @property
    def available_pipelines(self) -> list[str]:
        """Список доступных ветвей признаков."""
        pipes: list[str] = []
        if self._visual_available:
            pipes.append("visual")
        if self._text_available:
            pipes.append("text")
        # Метаданные всегда доступны
        pipes.append("metadata")
        return pipes

    # ------------------------------------------------------------------
    # Инициализация моделей
    # ------------------------------------------------------------------

    def _init_models(self) -> None:
        """Инициализировать доступные модели.

        Каждая ветвь инициализируется независимо: при ошибке одной
        ветви остальные продолжают работать.
        """
        self._init_visual_model()
        self._init_text_model()
        self._init_head_model()

        self._log.info(
            "Нейросетевой классификатор инициализирован",
            visual=self._visual_available,
            text=self._text_available,
            head=self._head_available,
            pipelines=self.available_pipelines,
        )

    def _init_visual_model(self) -> None:
        """Загрузить ONNX-модель MobileNetV3."""
        ort = _try_import_onnxruntime()
        if ort is None:
            self._log.warning(
                "onnxruntime не установлен, визуальная ветвь отключена"
            )
            return

        if not self._visual_model_path.exists():
            self._log.warning(
                "ONNX-модель не найдена, визуальная ветвь отключена",
                path=str(self._visual_model_path),
            )
            return

        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self._onnx_session = ort.InferenceSession(
                str(self._visual_model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            self._visual_available = True
            self._log.info("ONNX-модель загружена", path=str(self._visual_model_path))
        except Exception as exc:
            self._log.error(
                "Ошибка загрузки ONNX-модели",
                error=str(exc),
            )

    def _init_text_model(self) -> None:
        """Загрузить SBERT-модель для текстовых эмбеддингов."""
        SentenceTransformer = _try_import_sentence_transformers()
        if SentenceTransformer is None:
            self._log.warning(
                "sentence-transformers не установлен, текстовая ветвь отключена"
            )
            return

        try:
            self._sbert_model = SentenceTransformer(self._text_model_name)
            self._text_available = True
            self._log.info(
                "SBERT-модель загружена",
                model=self._text_model_name,
            )
        except Exception as exc:
            self._log.error(
                "Ошибка загрузки SBERT-модели",
                model=self._text_model_name,
                error=str(exc),
            )

    def _init_head_model(self) -> None:
        """Загрузить обученный классификационный слой."""
        if not self._head_model_path.exists():
            self._log.info(
                "Головная модель не найдена, будет использована эвристика",
                path=str(self._head_model_path),
            )
            return

        try:
            import joblib

            data = joblib.load(self._head_model_path)
            self._head_model = data["model"]
            self._head_classes = data.get("classes", list(DOCUMENT_TYPES))
            self._head_available = True
            self._log.info("Головная модель загружена")
        except ImportError:
            self._log.warning("joblib не установлен, головная модель недоступна")
        except Exception as exc:
            self._log.error(
                "Ошибка загрузки головной модели",
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Извлечение признаков
    # ------------------------------------------------------------------

    def _extract_visual_features(
        self,
        image_path: str | Path | None,
    ) -> np.ndarray:
        """Извлечь визуальные признаки из изображения документа.

        Args:
            image_path: Путь к изображению. Если ``None`` или модель
                недоступна — возвращается нулевой вектор.

        Returns:
            Вектор признаков размерности ``[_VISUAL_DIM]``.
        """
        zeros = np.zeros(_VISUAL_DIM, dtype=np.float32)

        if not self._visual_available or image_path is None:
            return zeros

        Image = _try_import_pil()
        if Image is None:
            return zeros

        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize((224, 224))

            # Нормализация: ImageNet-стиль
            arr = np.array(img, dtype=np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            arr = (arr - mean) / std

            # NCHW-формат для ONNX
            arr = arr.transpose(2, 0, 1)
            arr = np.expand_dims(arr, axis=0)

            input_name = self._onnx_session.get_inputs()[0].name
            output = self._onnx_session.run(None, {input_name: arr})

            features = output[0].flatten()

            # Обрезать или дополнить до нужной размерности
            if len(features) >= _VISUAL_DIM:
                features = features[:_VISUAL_DIM]
            else:
                features = np.pad(
                    features, (0, _VISUAL_DIM - len(features))
                )

            return features.astype(np.float32)
        except Exception as exc:
            self._log.warning(
                "Ошибка извлечения визуальных признаков",
                error=str(exc),
            )
            return zeros

    def _extract_text_features(self, text: str) -> np.ndarray:
        """Извлечь текстовые признаки через SBERT.

        Args:
            text: OCR-текст документа.

        Returns:
            Вектор эмбеддинга размерности ``[_TEXT_DIM]``.
        """
        zeros = np.zeros(_TEXT_DIM, dtype=np.float32)

        if not self._text_available or not text.strip():
            return zeros

        try:
            embedding = self._sbert_model.encode(
                text[:2048],  # Ограничение длины текста
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            features = np.array(embedding, dtype=np.float32).flatten()

            # Обрезать или дополнить до нужной размерности
            if len(features) >= _TEXT_DIM:
                features = features[:_TEXT_DIM]
            else:
                features = np.pad(
                    features, (0, _TEXT_DIM - len(features))
                )

            return features
        except Exception as exc:
            self._log.warning(
                "Ошибка извлечения текстовых признаков",
                error=str(exc),
            )
            return zeros

    def _extract_metadata_features(
        self,
        metadata: dict[str, Any],
    ) -> np.ndarray:
        """Извлечь признаки из метаданных документа.

        Кодирует имя файла (через простой хеш), количество страниц
        и размер файла в вектор фиксированной размерности.

        Args:
            metadata: Метаданные документа.

        Returns:
            Вектор признаков размерности ``[_META_DIM]``.
        """
        features = np.zeros(_META_DIM, dtype=np.float32)

        filename = metadata.get("filename", "")
        page_count = metadata.get("page_count", 1)
        file_size = metadata.get("file_size", 0)

        # Простая кодировка имени файла через символьные n-граммы
        if filename:
            fn_lower = filename.lower()
            # Хешируем по парам символов в первые 8 компонент
            for i, ch in enumerate(fn_lower[:8]):
                features[i] = float(ord(ch) % 256) / 256.0

        # Количество страниц (нормализовано)
        features[8] = min(float(page_count) / 100.0, 1.0)

        # Размер файла (нормализовано, в МБ)
        features[9] = min(float(file_size) / (50 * 1024 * 1024), 1.0)

        # Расширение файла (one-hot для частых расширений)
        ext_map = {".pdf": 10, ".png": 11, ".jpg": 12, ".tif": 13, ".jpeg": 14}
        ext = Path(filename).suffix.lower() if filename else ""
        if ext in ext_map:
            features[ext_map[ext]] = 1.0

        # Длина имени файла (нормализовано)
        features[15] = min(float(len(filename)) / 200.0, 1.0)

        return features

    # ------------------------------------------------------------------
    # Классификация
    # ------------------------------------------------------------------

    def classify(self, text: str, metadata: dict[str, Any]) -> ClassificationResult:
        """Классифицировать документ с помощью мультимодальной нейросети.

        Args:
            text: OCR-текст документа.
            metadata: Метаданные документа (``filename``, ``page_count``,
                ``file_size``, ``image_path``).

        Returns:
            ``ClassificationResult`` с типом документа и уверенностью.
        """
        image_path = metadata.get("image_path")

        # Извлечение признаков из всех доступных ветвей
        visual_features = self._extract_visual_features(image_path)
        text_features = self._extract_text_features(text)
        meta_features = self._extract_metadata_features(metadata)

        # Конкатенация
        combined = np.concatenate([
            visual_features,
            text_features,
            meta_features,
        ])

        active_pipelines = self.available_pipelines

        # Классификация
        if self._head_available and self._head_model is not None:
            return self._classify_with_head(combined, active_pipelines)

        # Фолбэк: эвристика на текстовых признаках
        return self._classify_heuristic(
            text=text,
            text_features=text_features,
            active_pipelines=active_pipelines,
        )

    def _classify_with_head(
        self,
        features: np.ndarray,
        active_pipelines: list[str],
    ) -> ClassificationResult:
        """Классификация через обученный головной слой.

        Args:
            features: Объединённый вектор признаков.
            active_pipelines: Активные ветви.

        Returns:
            Результат классификации.
        """
        try:
            features_2d = features.reshape(1, -1)
            predicted = self._head_model.predict(features_2d)[0]

            # Попробовать получить вероятности
            confidence = 0.5
            if hasattr(self._head_model, "predict_proba"):
                proba = self._head_model.predict_proba(features_2d)[0]
                confidence = float(np.max(proba))

            if predicted not in DOCUMENT_TYPES:
                predicted = "UNKNOWN"
                confidence = 0.0

            self._log.info(
                "Нейросетевая классификация завершена",
                doc_type=predicted,
                confidence=round(confidence, 3),
                pipelines=active_pipelines,
            )

            return ClassificationResult(
                doc_type=predicted,
                confidence=round(confidence, 4),
                classifier_name=self.name,
                details={
                    "method": "neural_head",
                    "active_pipelines": active_pipelines,
                },
            )
        except Exception as exc:
            self._log.error(
                "Ошибка классификации через головной слой",
                error=str(exc),
            )
            return ClassificationResult(
                doc_type="UNKNOWN",
                confidence=0.0,
                classifier_name=self.name,
                details={"error": str(exc)},
            )

    def _classify_heuristic(
        self,
        text: str,
        text_features: np.ndarray,
        active_pipelines: list[str],
    ) -> ClassificationResult:
        """Эвристическая классификация при отсутствии головного слоя.

        Использует косинусное сходство текстовых эмбеддингов с эталонными
        описаниями типов документов (если SBERT доступен) или возвращает
        ``UNKNOWN``.

        Args:
            text: OCR-текст документа.
            text_features: SBERT-эмбеддинг текста.
            active_pipelines: Активные ветви.

        Returns:
            Результат классификации.
        """
        if not self._text_available or not text.strip():
            self._log.debug(
                "Недостаточно данных для эвристической классификации"
            )
            return ClassificationResult(
                doc_type="UNKNOWN",
                confidence=0.0,
                classifier_name=self.name,
                details={
                    "method": "heuristic_fallback",
                    "reason": "no_text_model_or_empty_text",
                },
            )

        # Эталонные описания типов документов
        type_descriptions: dict[str, str] = {
            "INVOICE": "счёт-фактура оплата товары услуги НДС итого сумма",
            "ACT": "акт выполненных работ оказанных услуг приёмка исполнитель заказчик",
            "CONTRACT": "договор предмет стороны реквизиты подписи обязательства",
            "WAYBILL": "товарная накладная ТОРГ-12 грузоотправитель грузополучатель наименование товара",
            "PAYMENT_ORDER": "платёжное поручение банк получатель плательщик БИК расчётный счёт",
        }

        try:
            desc_texts = list(type_descriptions.values())
            desc_labels = list(type_descriptions.keys())

            desc_embeddings = self._sbert_model.encode(
                desc_texts,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

            # Нормализация текстового вектора
            text_norm = text_features / (
                np.linalg.norm(text_features) + 1e-8
            )

            # Косинусное сходство
            similarities = np.dot(desc_embeddings, text_norm)
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])

            # Масштабирование: [-1, 1] → [0, 1]
            confidence = max(0.0, min((best_score + 1.0) / 2.0, 1.0))

            doc_type = desc_labels[best_idx]

            self._log.info(
                "Эвристическая классификация завершена",
                doc_type=doc_type,
                confidence=round(confidence, 3),
                pipelines=active_pipelines,
            )

            return ClassificationResult(
                doc_type=doc_type,
                confidence=round(confidence, 4),
                classifier_name=self.name,
                details={
                    "method": "sbert_heuristic",
                    "active_pipelines": active_pipelines,
                    "similarity_scores": {
                        label: round(float(sim), 4)
                        for label, sim in zip(desc_labels, similarities)
                    },
                },
            )
        except Exception as exc:
            self._log.error(
                "Ошибка эвристической классификации",
                error=str(exc),
            )
            return ClassificationResult(
                doc_type="UNKNOWN",
                confidence=0.0,
                classifier_name=self.name,
                details={"error": str(exc)},
            )
