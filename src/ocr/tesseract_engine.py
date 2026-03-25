"""Модуль OCR на основе Tesseract."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import fitz  # PyMuPDF
import numpy as np
import pytesseract
import structlog
from PIL import Image

from ocr.preprocessor import ImagePreprocessor, PreprocessorConfig

logger: Final = structlog.get_logger(__name__)

# Поддерживаемые расширения файлов.
_IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp",
})
_PDF_EXTENSIONS: Final[frozenset[str]] = frozenset({".pdf"})


class TesseractNotFoundError(RuntimeError):
    """Исполняемый файл Tesseract не найден в системе."""


class UnsupportedFileTypeError(ValueError):
    """Неподдерживаемый тип файла для OCR."""


class OCRProcessingError(RuntimeError):
    """Общая ошибка обработки OCR."""


@dataclass
class TesseractConfig:
    """Конфигурация движка Tesseract OCR.

    Attributes:
        tesseract_cmd: Путь к исполняемому файлу Tesseract.
                       Если None, используется системный PATH.
        languages: Языки распознавания (например, ``"rus+eng"``).
        psm: Режим сегментации страницы Tesseract (Page Segmentation Mode).
             По умолчанию 3 — полностью автоматическая сегментация.
        dpi: Разрешение изображения для Tesseract.
        preprocessor_config: Конфигурация предобработчика изображений.
                             Если None, предобработка использует параметры по умолчанию.
        pdf_dpi: Разрешение рендеринга страниц PDF в изображения.
    """

    tesseract_cmd: str | None = None
    languages: str = "rus+eng"
    psm: int = 3
    dpi: int = 300
    preprocessor_config: PreprocessorConfig | None = None
    pdf_dpi: int = 300


class TesseractEngine:
    """Движок OCR на основе Tesseract для извлечения текста из документов.

    Поддерживает обработку изображений и PDF-файлов. PDF-страницы
    рендерятся в изображения через PyMuPDF, затем распознаются Tesseract.

    Перед распознаванием каждое изображение проходит через конвейер
    предобработки (:class:`~ocr.preprocessor.ImagePreprocessor`).

    Пример использования::

        engine = TesseractEngine()
        text = engine.extract_text(Path("document.pdf"))
    """

    def __init__(self, config: TesseractConfig | None = None) -> None:
        """Инициализация движка Tesseract OCR.

        Args:
            config: Конфигурация движка. Если не указана,
                    используются значения по умолчанию.

        Raises:
            TesseractNotFoundError: Если исполняемый файл Tesseract не найден.
        """
        self._config = config or TesseractConfig()
        self._log = logger.bind(component="tesseract_engine")
        self._preprocessor = ImagePreprocessor(self._config.preprocessor_config)

        self._configure_tesseract_cmd()
        self._validate_tesseract()

    def _configure_tesseract_cmd(self) -> None:
        """Настройка пути к исполняемому файлу Tesseract."""
        if self._config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self._config.tesseract_cmd
            self._log.info(
                "Задан путь к Tesseract",
                path=self._config.tesseract_cmd,
            )

    def _validate_tesseract(self) -> None:
        """Проверка доступности Tesseract в системе.

        Raises:
            TesseractNotFoundError: Если Tesseract недоступен.
        """
        cmd = self._config.tesseract_cmd or "tesseract"
        if not shutil.which(cmd):
            msg = (
                f"Исполняемый файл Tesseract не найден: '{cmd}'. "
                "Убедитесь, что Tesseract установлен и доступен в PATH."
            )
            self._log.error("Tesseract не найден", cmd=cmd)
            raise TesseractNotFoundError(msg)

        self._log.info("Tesseract доступен", cmd=cmd)

    def extract_text(self, file_path: Path) -> str:
        """Извлечение текста из файла (изображение или PDF).

        Автоматически определяет тип файла по расширению и вызывает
        соответствующий метод обработки.

        Args:
            file_path: Путь к файлу для распознавания.

        Returns:
            Извлечённый текст.

        Raises:
            FileNotFoundError: Если файл не существует.
            UnsupportedFileTypeError: Если формат файла не поддерживается.
            OCRProcessingError: При ошибке обработки.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            msg = f"Файл не найден: {file_path}"
            self._log.error("Файл не найден", path=str(file_path))
            raise FileNotFoundError(msg)

        suffix = file_path.suffix.lower()
        self._log.info(
            "Начало извлечения текста",
            path=str(file_path),
            extension=suffix,
        )

        if suffix in _PDF_EXTENSIONS:
            return self.extract_from_pdf(file_path)

        if suffix in _IMAGE_EXTENSIONS:
            return self.extract_from_image(file_path)

        msg = (
            f"Неподдерживаемый формат файла: '{suffix}'. "
            f"Поддерживаемые форматы: "
            f"{sorted(_IMAGE_EXTENSIONS | _PDF_EXTENSIONS)}"
        )
        self._log.error("Неподдерживаемый формат", extension=suffix)
        raise UnsupportedFileTypeError(msg)

    def extract_from_pdf(self, pdf_path: Path) -> str:
        """Извлечение текста из PDF-документа.

        Каждая страница PDF рендерится в изображение через PyMuPDF (fitz),
        проходит предобработку и распознаётся Tesseract. Результаты
        объединяются с разделением по страницам.

        Args:
            pdf_path: Путь к PDF-файлу.

        Returns:
            Объединённый текст всех страниц.

        Raises:
            OCRProcessingError: При ошибке открытия или обработки PDF.
        """
        self._log.info("Обработка PDF", path=str(pdf_path))

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            msg = f"Ошибка открытия PDF: {pdf_path} — {exc}"
            self._log.error("Ошибка открытия PDF", path=str(pdf_path), error=str(exc))
            raise OCRProcessingError(msg) from exc

        page_texts: list[str] = []
        total_pages = len(doc)
        self._log.info("PDF открыт", total_pages=total_pages)

        try:
            for page_num in range(total_pages):
                self._log.debug(
                    "Обработка страницы PDF",
                    page=page_num + 1,
                    total=total_pages,
                )

                try:
                    page = doc.load_page(page_num)
                    zoom = self._config.pdf_dpi / 72.0
                    matrix = fitz.Matrix(zoom, zoom)
                    pixmap = page.get_pixmap(matrix=matrix)

                    # Преобразование pixmap в numpy-массив.
                    image_data = np.frombuffer(pixmap.samples, dtype=np.uint8)
                    if pixmap.n == 1:
                        image = image_data.reshape(pixmap.h, pixmap.w)
                    elif pixmap.n == 3:
                        image = image_data.reshape(pixmap.h, pixmap.w, 3)
                    elif pixmap.n == 4:
                        image = image_data.reshape(pixmap.h, pixmap.w, 4)
                        image = image[:, :, :3]  # Отбрасываем альфа-канал.
                    else:
                        self._log.warning(
                            "Неожиданное количество каналов в pixmap",
                            channels=pixmap.n,
                            page=page_num + 1,
                        )
                        image = image_data.reshape(pixmap.h, pixmap.w, pixmap.n)

                    text = self._run_tesseract(image)
                    page_texts.append(text)

                except Exception as exc:
                    self._log.error(
                        "Ошибка обработки страницы PDF",
                        page=page_num + 1,
                        error=str(exc),
                    )
                    page_texts.append("")
        finally:
            doc.close()

        result = "\n\n".join(page_texts)
        self._log.info(
            "PDF обработан",
            total_pages=total_pages,
            total_chars=len(result),
        )
        return result

    def extract_from_image(self, image_path: Path) -> str:
        """Извлечение текста из файла изображения.

        Загружает изображение через Pillow, конвертирует в numpy-массив,
        применяет предобработку и запускает распознавание.

        Args:
            image_path: Путь к файлу изображения.

        Returns:
            Распознанный текст.

        Raises:
            OCRProcessingError: При ошибке загрузки или обработки изображения.
        """
        self._log.info("Обработка изображения", path=str(image_path))

        try:
            pil_image = Image.open(image_path)
        except Exception as exc:
            msg = f"Ошибка загрузки изображения: {image_path} — {exc}"
            self._log.error(
                "Ошибка загрузки изображения",
                path=str(image_path),
                error=str(exc),
            )
            raise OCRProcessingError(msg) from exc

        try:
            # Конвертируем в RGB, если изображение в другом режиме.
            if pil_image.mode not in ("RGB", "L"):
                pil_image = pil_image.convert("RGB")

            image_array = np.array(pil_image)
            text = self._run_tesseract(image_array)

            self._log.info(
                "Изображение обработано",
                path=str(image_path),
                chars_extracted=len(text),
            )
            return text

        except OCRProcessingError:
            raise
        except Exception as exc:
            msg = f"Ошибка OCR для изображения: {image_path} — {exc}"
            self._log.error(
                "Ошибка OCR для изображения",
                path=str(image_path),
                error=str(exc),
            )
            raise OCRProcessingError(msg) from exc

    def _run_tesseract(self, image: np.ndarray) -> str:
        """Запуск распознавания Tesseract на изображении.

        Применяет предобработку через :class:`ImagePreprocessor`,
        затем передаёт изображение в pytesseract.

        Args:
            image: Изображение в формате numpy-массива.

        Returns:
            Распознанный текст.

        Raises:
            OCRProcessingError: При ошибке распознавания.
        """
        try:
            preprocessed = self._preprocessor.preprocess(image)
        except Exception as exc:
            self._log.error("Ошибка предобработки", error=str(exc))
            raise OCRProcessingError(
                f"Ошибка предобработки изображения: {exc}"
            ) from exc

        custom_config = (
            f"--psm {self._config.psm} "
            f"--dpi {self._config.dpi}"
        )

        self._log.debug(
            "Запуск Tesseract",
            languages=self._config.languages,
            psm=self._config.psm,
            dpi=self._config.dpi,
        )

        try:
            text: str = pytesseract.image_to_string(
                preprocessed,
                lang=self._config.languages,
                config=custom_config,
            )
        except pytesseract.TesseractNotFoundError as exc:
            msg = f"Tesseract не найден при попытке распознавания: {exc}"
            self._log.error("Tesseract не найден при распознавании", error=str(exc))
            raise TesseractNotFoundError(msg) from exc
        except pytesseract.TesseractError as exc:
            msg = f"Ошибка Tesseract: {exc}"
            self._log.error("Ошибка Tesseract", error=str(exc))
            raise OCRProcessingError(msg) from exc

        return text.strip()
