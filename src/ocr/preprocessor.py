"""Предобработка изображений для OCR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import cv2
import numpy as np
import structlog

logger: Final = structlog.get_logger(__name__)


@dataclass
class PreprocessorConfig:
    """Конфигурация шагов предобработки.

    Каждый флаг включает или отключает соответствующий этап
    конвейера предобработки изображения.
    """

    deskew_enabled: bool = True
    denoise_enabled: bool = True
    binarize_enabled: bool = True
    enhance_contrast_enabled: bool = True
    resize_enabled: bool = False
    target_dpi: int = 300
    current_dpi: int = 200
    denoise_strength: int = 10
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)


class ImagePreprocessor:
    """Предобработчик изображений для улучшения качества OCR.

    Выполняет конвейер преобразований: выравнивание, удаление шума,
    бинаризация, усиление контраста и нормализация DPI.

    Пример использования::

        config = PreprocessorConfig(denoise_enabled=True, binarize_enabled=True)
        preprocessor = ImagePreprocessor(config)
        result = preprocessor.preprocess(image)
    """

    def __init__(self, config: PreprocessorConfig | None = None) -> None:
        """Инициализация предобработчика.

        Args:
            config: Конфигурация шагов предобработки.
                    Если не указана, используются значения по умолчанию.
        """
        self._config = config or PreprocessorConfig()
        self._log = logger.bind(component="image_preprocessor")

    @property
    def config(self) -> PreprocessorConfig:
        """Текущая конфигурация предобработчика."""
        return self._config

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Полный конвейер предобработки изображения.

        Последовательно применяет включённые этапы обработки:
        выравнивание -> удаление шума -> усиление контраста ->
        бинаризация -> нормализация DPI.

        Args:
            image: Входное изображение в формате numpy-массива (BGR или grayscale).

        Returns:
            Обработанное изображение в формате numpy-массива (grayscale).

        Raises:
            ValueError: Если изображение пустое или имеет некорректный формат.
        """
        if image is None or image.size == 0:
            raise ValueError("Получено пустое изображение для предобработки.")

        self._log.info(
            "Запуск конвейера предобработки",
            shape=image.shape,
            dtype=str(image.dtype),
        )

        result = self._ensure_grayscale(image)

        steps: list[tuple[str, bool, callable]] = [
            ("deskew", self._config.deskew_enabled, self.deskew),
            ("denoise", self._config.denoise_enabled, self.denoise),
            ("enhance_contrast", self._config.enhance_contrast_enabled, self.enhance_contrast),
            ("binarize", self._config.binarize_enabled, self.binarize),
        ]

        for step_name, enabled, step_fn in steps:
            if not enabled:
                self._log.debug("Шаг пропущен", step=step_name)
                continue
            try:
                self._log.debug("Выполнение шага", step=step_name)
                result = step_fn(result)
            except cv2.error as exc:
                self._log.error(
                    "Ошибка OpenCV на шаге предобработки",
                    step=step_name,
                    error=str(exc),
                )
                raise

        if self._config.resize_enabled:
            try:
                result = self.resize_to_dpi(
                    result,
                    current_dpi=self._config.current_dpi,
                    target_dpi=self._config.target_dpi,
                )
            except cv2.error as exc:
                self._log.error(
                    "Ошибка OpenCV при нормализации DPI",
                    error=str(exc),
                )
                raise

        self._log.info(
            "Конвейер предобработки завершён",
            output_shape=result.shape,
        )
        return result

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """Автокоррекция наклона (выравнивание) изображения.

        Определяет угол наклона текста по минимальному ограничивающему
        прямоугольнику контуров и поворачивает изображение.

        Args:
            image: Grayscale-изображение.

        Returns:
            Выровненное изображение.
        """
        coords = np.column_stack(np.where(image > 0))

        if coords.shape[0] < 5:
            self._log.warning("Недостаточно точек для определения угла наклона")
            return image

        angle = cv2.minAreaRect(coords)[-1]

        # minAreaRect возвращает угол в диапазоне [-90, 0).
        # Нормализуем до малых отклонений.
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.5:
            self._log.debug("Наклон минимален, коррекция не требуется", angle=angle)
            return image

        self._log.debug("Коррекция наклона", angle=round(angle, 2))

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Удаление шума с изображения.

        Использует алгоритм Non-Local Means Denoising из OpenCV.

        Args:
            image: Grayscale-изображение.

        Returns:
            Изображение с уменьшенным шумом.
        """
        self._log.debug(
            "Удаление шума",
            strength=self._config.denoise_strength,
        )
        denoised = cv2.fastNlMeansDenoising(
            image,
            h=self._config.denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )
        return denoised

    def binarize(self, image: np.ndarray) -> np.ndarray:
        """Бинаризация изображения методом Оцу.

        Автоматически определяет оптимальный порог для разделения
        пикселей на чёрные и белые.

        Args:
            image: Grayscale-изображение.

        Returns:
            Бинаризованное изображение.
        """
        _, binary = cv2.threshold(
            image,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        self._log.debug("Бинаризация выполнена (метод Оцу)")
        return binary

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Усиление контраста методом CLAHE.

        Адаптивная эквализация гистограммы с ограничением контраста
        (Contrast Limited Adaptive Histogram Equalization).

        Args:
            image: Grayscale-изображение.

        Returns:
            Изображение с улучшенным контрастом.
        """
        clahe = cv2.createCLAHE(
            clipLimit=self._config.clahe_clip_limit,
            tileGridSize=self._config.clahe_tile_grid_size,
        )
        enhanced = clahe.apply(image)
        self._log.debug(
            "Контраст улучшен (CLAHE)",
            clip_limit=self._config.clahe_clip_limit,
        )
        return enhanced

    def resize_to_dpi(
        self,
        image: np.ndarray,
        current_dpi: int | None = None,
        target_dpi: int | None = None,
    ) -> np.ndarray:
        """Нормализация разрешения изображения к целевому DPI.

        Масштабирует изображение пропорционально разнице между текущим
        и целевым DPI.

        Args:
            image: Входное изображение.
            current_dpi: Текущее разрешение изображения (точек на дюйм).
                         По умолчанию берётся из конфигурации.
            target_dpi: Целевое разрешение (точек на дюйм).
                        По умолчанию берётся из конфигурации.

        Returns:
            Масштабированное изображение.

        Raises:
            ValueError: Если current_dpi или target_dpi <= 0.
        """
        current = current_dpi or self._config.current_dpi
        target = target_dpi or self._config.target_dpi

        if current <= 0 or target <= 0:
            raise ValueError(
                f"DPI должен быть положительным: current={current}, target={target}"
            )

        if current == target:
            self._log.debug("DPI совпадает, масштабирование не требуется")
            return image

        scale_factor = target / current
        self._log.debug(
            "Масштабирование DPI",
            current_dpi=current,
            target_dpi=target,
            scale_factor=round(scale_factor, 3),
        )

        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)
        interpolation = cv2.INTER_CUBIC if scale_factor > 1 else cv2.INTER_AREA
        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=interpolation,
        )
        return resized

    @staticmethod
    def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
        """Преобразование изображения в grayscale, если необходимо.

        Args:
            image: Входное изображение (BGR или grayscale).

        Returns:
            Grayscale-изображение.
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return image
