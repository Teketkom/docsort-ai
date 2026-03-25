"""Установочный скрипт для DocSort AI."""

from setuptools import setup, find_packages

setup(
    name="docsort-ai",
    version="0.1.0",
    description="Система автоматической классификации и сортировки сканированных документов",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="DocSort AI Team",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pydantic>=2.0",
        "pyyaml>=6.0",
        "pytesseract>=0.3.10",
        "Pillow>=10.0",
        "pymupdf>=1.23.0",
        "scikit-learn>=1.3",
        "watchdog>=3.0",
        "fastapi>=0.104",
        "uvicorn>=0.24",
        "structlog>=23.2",
    ],
    extras_require={
        "dev": ["pytest>=7.4", "pytest-asyncio>=0.21", "pytest-cov>=4.1"],
        "neural": ["onnxruntime>=1.16", "sentence-transformers>=2.2"],
        "llm": ["httpx>=0.25"],
        "ui": ["streamlit>=1.28"],
    },
    entry_points={
        "console_scripts": [
            "docsort=core.pipeline:main",
        ],
    },
)
