"""Модуль сортировки файлов по категориям."""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


class FileSorter:
    """Сортировщик файлов по категориям документов.

    Перемещает или копирует файлы в структурированную директорию
    на основе типа документа и метаданных.

    Attributes:
        output_dir: Корневая директория для отсортированных файлов.
        filename_template: Шаблон имени файла.
        create_type_dirs: Создавать поддиректории по типу документа.
        create_date_dirs: Создавать поддиректории по дате (YYYY-MM).
        copy_mode: Копировать файлы вместо перемещения.
    """

    def __init__(
        self,
        output_dir: Path | str,
        filename_template: str = "{date}_{doc_type}_{original_name}",
        create_type_dirs: bool = True,
        create_date_dirs: bool = True,
        copy_mode: bool = False,
    ) -> None:
        """Инициализация сортировщика файлов.

        Args:
            output_dir: Корневая директория для отсортированных файлов.
            filename_template: Шаблон имени файла с плейсхолдерами
                {date}, {doc_type}, {original_name}.
            create_type_dirs: Создавать поддиректории по типу документа.
            create_date_dirs: Создавать поддиректории по дате (YYYY-MM).
            copy_mode: Копировать файлы вместо перемещения.
        """
        self.output_dir = Path(output_dir)
        self.filename_template = filename_template
        self.create_type_dirs = create_type_dirs
        self.create_date_dirs = create_date_dirs
        self.copy_mode = copy_mode

        logger.info(
            "file_sorter_initialized",
            output_dir=str(self.output_dir),
            filename_template=self.filename_template,
            create_type_dirs=self.create_type_dirs,
            create_date_dirs=self.create_date_dirs,
            copy_mode=self.copy_mode,
        )

    def sort(
        self,
        file_path: Path | str,
        doc_type: str,
        metadata: Optional[dict] = None,
    ) -> Path:
        """Сортировка файла в целевую директорию.

        Перемещает или копирует файл в структурированную директорию
        на основе типа документа и метаданных.

        Args:
            file_path: Путь к исходному файлу.
            doc_type: Тип документа (например, 'invoice', 'contract').
            metadata: Дополнительные метаданные документа.

        Returns:
            Путь к файлу в целевой директории.

        Raises:
            FileNotFoundError: Если исходный файл не найден.
            PermissionError: Если нет прав на запись в целевую директорию.
            OSError: При ошибке файловой операции.
        """
        file_path = Path(file_path)
        metadata = metadata or {}

        if not file_path.exists():
            logger.error("source_file_not_found", file_path=str(file_path))
            raise FileNotFoundError(f"Исходный файл не найден: {file_path}")

        if not file_path.is_file():
            logger.error("path_is_not_a_file", file_path=str(file_path))
            raise ValueError(f"Путь не является файлом: {file_path}")

        target_path = self._build_target_path(file_path, doc_type, metadata)

        target_path.parent.mkdir(parents=True, exist_ok=True)

        target_path = self._resolve_duplicate(target_path)

        try:
            if self.copy_mode:
                shutil.copy2(str(file_path), str(target_path))
                logger.info(
                    "file_copied",
                    source=str(file_path),
                    target=str(target_path),
                    doc_type=doc_type,
                )
            else:
                shutil.move(str(file_path), str(target_path))
                logger.info(
                    "file_moved",
                    source=str(file_path),
                    target=str(target_path),
                    doc_type=doc_type,
                )
        except PermissionError:
            logger.error(
                "permission_denied",
                source=str(file_path),
                target=str(target_path),
            )
            raise
        except OSError as exc:
            logger.error(
                "file_operation_failed",
                source=str(file_path),
                target=str(target_path),
                error=str(exc),
            )
            raise

        return target_path

    def _build_target_path(
        self,
        file_path: Path,
        doc_type: str,
        metadata: dict,
    ) -> Path:
        """Построение целевого пути на основе шаблона.

        Формирует путь вида: output_dir/doc_type/YYYY-MM/filename

        Args:
            file_path: Путь к исходному файлу.
            doc_type: Тип документа.
            metadata: Метаданные документа.

        Returns:
            Целевой путь для файла.
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        date_dir = now.strftime("%Y-%m")
        sanitized_type = self._sanitize_filename(doc_type)

        original_name = file_path.stem
        extension = file_path.suffix

        template_vars = {
            "date": date_str,
            "doc_type": sanitized_type,
            "original_name": self._sanitize_filename(original_name),
            **{k: self._sanitize_filename(str(v)) for k, v in metadata.items()},
        }

        try:
            new_filename = self.filename_template.format(**template_vars)
        except KeyError as exc:
            logger.warning(
                "template_key_missing",
                template=self.filename_template,
                missing_key=str(exc),
            )
            new_filename = f"{date_str}_{sanitized_type}_{original_name}"

        new_filename = self._sanitize_filename(new_filename) + extension

        target_dir = self.output_dir

        if self.create_type_dirs:
            target_dir = target_dir / sanitized_type

        if self.create_date_dirs:
            target_dir = target_dir / date_dir

        return target_dir / new_filename

    def _resolve_duplicate(self, target_path: Path) -> Path:
        """Разрешение конфликтов имён файлов.

        Если файл с таким именем уже существует, добавляет
        числовой суффикс (_1, _2 и т.д.).

        Args:
            target_path: Желаемый путь файла.

        Returns:
            Уникальный путь файла.
        """
        if not target_path.exists():
            return target_path

        stem = target_path.stem
        suffix = target_path.suffix
        parent = target_path.parent
        counter = 1

        while True:
            new_name = f"{stem}_{counter}{suffix}"
            candidate = parent / new_name
            if not candidate.exists():
                logger.info(
                    "duplicate_resolved",
                    original=str(target_path),
                    resolved=str(candidate),
                    counter=counter,
                )
                return candidate
            counter += 1

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Очистка имени файла от недопустимых символов.

        Заменяет пробелы на подчёркивания, удаляет специальные символы,
        ограничивает длину.

        Args:
            name: Исходное имя.

        Returns:
            Очищенное имя файла.
        """
        name = name.strip()
        name = name.replace(" ", "_")
        name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name)
        name = re.sub(r"_+", "_", name)
        name = name.strip("_.")

        max_length = 200
        if len(name) > max_length:
            name = name[:max_length]

        if not name:
            name = "unnamed"

        return name
