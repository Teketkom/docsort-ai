"""
Ядро системы DocSort AI.

Содержит основные модели данных, конфигурацию и пайплайн обработки документов.
"""

from core.config import get_config
from core.document import Document, DocumentClassification, DocumentType

__all__ = [
    "Document",
    "DocumentClassification",
    "DocumentType",
    "get_config",
]
