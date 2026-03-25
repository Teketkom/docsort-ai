"""Пакет OCR для DocSort AI.

Модули оптического распознавания символов: предобработка изображений
и извлечение текста из документов (изображения, PDF).
"""

from ocr.preprocessor import ImagePreprocessor
from ocr.tesseract_engine import TesseractEngine

__all__ = [
    "ImagePreprocessor",
    "TesseractEngine",
]
