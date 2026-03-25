"""Веб-интерфейс DocSort AI на Streamlit."""

from __future__ import annotations

from typing import Any, Optional

import httpx
import streamlit as st
import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

DEFAULT_API_URL = "http://localhost:8000"
API_TIMEOUT = 30.0


def _get_api_url() -> str:
    """Получение URL API из настроек сессии.

    Returns:
        Базовый URL API-сервера.
    """
    return st.session_state.get("api_url", DEFAULT_API_URL)


# ---------------------------------------------------------------------------
# HTTP-клиент
# ---------------------------------------------------------------------------


class ApiClient:
    """HTTP-клиент для взаимодействия с FastAPI бэкендом.

    Attributes:
        base_url: Базовый URL API-сервера.
        timeout: Таймаут запросов в секундах.
    """

    def __init__(self, base_url: str, timeout: float = API_TIMEOUT) -> None:
        """Инициализация API-клиента.

        Args:
            base_url: Базовый URL API-сервера.
            timeout: Таймаут запросов в секундах.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _url(self, path: str) -> str:
        """Формирование полного URL.

        Args:
            path: Относительный путь эндпоинта.

        Returns:
            Полный URL.
        """
        return f"{self.base_url}{path}"

    def classify_file(
        self,
        file_content: bytes,
        filename: str,
    ) -> Optional[dict[str, Any]]:
        """Классификация файла через API.

        Args:
            file_content: Содержимое файла.
            filename: Имя файла.

        Returns:
            Результат классификации или None при ошибке.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self._url("/api/v1/classify"),
                    files={"file": (filename, file_content)},
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "classify_api_error",
                status_code=exc.response.status_code,
                detail=exc.response.text,
            )
            st.error(f"Ошибка API: {exc.response.status_code}")
            return None
        except httpx.RequestError as exc:
            logger.error("classify_request_error", error=str(exc))
            st.error(f"Ошибка подключения к серверу: {exc}")
            return None

    def classify_batch(
        self,
        files: list[tuple[str, bytes]],
    ) -> Optional[dict[str, Any]]:
        """Пакетная классификация файлов.

        Args:
            files: Список кортежей (имя_файла, содержимое).

        Returns:
            Результаты классификации или None при ошибке.
        """
        try:
            upload_files = [
                ("files", (name, content)) for name, content in files
            ]
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self._url("/api/v1/classify/batch"),
                    files=upload_files,
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "batch_classify_api_error",
                status_code=exc.response.status_code,
            )
            st.error(f"Ошибка API: {exc.response.status_code}")
            return None
        except httpx.RequestError as exc:
            logger.error("batch_classify_request_error", error=str(exc))
            st.error(f"Ошибка подключения к серверу: {exc}")
            return None

    def get_documents(
        self, page: int = 1, page_size: int = 20
    ) -> Optional[dict[str, Any]]:
        """Получение списка документов.

        Args:
            page: Номер страницы.
            page_size: Размер страницы.

        Returns:
            Список документов или None при ошибке.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(
                    self._url("/api/v1/documents"),
                    params={"page": page, "page_size": page_size},
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "documents_api_error",
                status_code=exc.response.status_code,
            )
            st.error(f"Ошибка API: {exc.response.status_code}")
            return None
        except httpx.RequestError as exc:
            logger.error("documents_request_error", error=str(exc))
            st.error(f"Ошибка подключения к серверу: {exc}")
            return None

    def get_document(self, doc_id: str) -> Optional[dict[str, Any]]:
        """Получение информации о документе.

        Args:
            doc_id: Идентификатор документа.

        Returns:
            Информация о документе или None при ошибке.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(
                    self._url(f"/api/v1/documents/{doc_id}"),
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "document_api_error",
                status_code=exc.response.status_code,
                doc_id=doc_id,
            )
            st.error(f"Документ не найден: {doc_id}")
            return None
        except httpx.RequestError as exc:
            logger.error("document_request_error", error=str(exc))
            st.error(f"Ошибка подключения к серверу: {exc}")
            return None

    def submit_feedback(
        self,
        doc_id: str,
        predicted_type: str,
        correct_type: str,
        user_comment: str = "",
    ) -> Optional[dict[str, Any]]:
        """Отправка обратной связи.

        Args:
            doc_id: Идентификатор документа.
            predicted_type: Предсказанный тип.
            correct_type: Правильный тип.
            user_comment: Комментарий пользователя.

        Returns:
            Ответ сервера или None при ошибке.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self._url("/api/v1/feedback"),
                    json={
                        "doc_id": doc_id,
                        "predicted_type": predicted_type,
                        "correct_type": correct_type,
                        "user_comment": user_comment,
                    },
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "feedback_api_error",
                status_code=exc.response.status_code,
            )
            st.error(f"Ошибка отправки отзыва: {exc.response.status_code}")
            return None
        except httpx.RequestError as exc:
            logger.error("feedback_request_error", error=str(exc))
            st.error(f"Ошибка подключения к серверу: {exc}")
            return None

    def get_stats(self) -> Optional[dict[str, Any]]:
        """Получение статистики.

        Returns:
            Статистика или None при ошибке.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(self._url("/api/v1/stats"))
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "stats_api_error",
                status_code=exc.response.status_code,
            )
            st.error(f"Ошибка получения статистики: {exc.response.status_code}")
            return None
        except httpx.RequestError as exc:
            logger.error("stats_request_error", error=str(exc))
            st.error(f"Ошибка подключения к серверу: {exc}")
            return None

    def health_check(self) -> Optional[dict[str, Any]]:
        """Проверка доступности сервера.

        Returns:
            Ответ проверки здоровья или None при ошибке.
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(self._url("/api/v1/health"))
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError):
            return None


# ---------------------------------------------------------------------------
# Страницы приложения
# ---------------------------------------------------------------------------


def _page_main(client: ApiClient) -> None:
    """Главная страница: загрузка и классификация файлов.

    Args:
        client: Экземпляр API-клиента.
    """
    st.header("Классификация документов")
    st.write("Загрузите файл для автоматической классификации.")

    uploaded_files = st.file_uploader(
        "Перетащите файлы сюда или нажмите для выбора",
        accept_multiple_files=True,
        type=["pdf", "docx", "doc", "txt", "png", "jpg", "jpeg", "tiff"],
        help="Поддерживаемые форматы: PDF, DOCX, DOC, TXT, PNG, JPG, TIFF",
    )

    if uploaded_files and st.button("Классифицировать", type="primary"):
        if len(uploaded_files) == 1:
            file = uploaded_files[0]
            with st.spinner("Классификация..."):
                result = client.classify_file(
                    file_content=file.getvalue(),
                    filename=file.name,
                )

            if result:
                _display_classification_result(result)
        else:
            files_data = [
                (f.name, f.getvalue()) for f in uploaded_files
            ]
            with st.spinner(f"Классификация {len(files_data)} файлов..."):
                result = client.classify_batch(files_data)

            if result:
                _display_batch_results(result)


def _display_classification_result(result: dict[str, Any]) -> None:
    """Отображение результата классификации.

    Args:
        result: Данные результата классификации.
    """
    st.success("Классификация завершена!")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Тип документа", result.get("doc_type", "Неизвестно"))
    with col2:
        confidence = result.get("confidence", 0.0)
        st.metric("Уверенность", f"{confidence:.1%}")
    with col3:
        st.metric("ID документа", result.get("doc_id", "")[:8] + "...")

    # Индикатор уверенности
    confidence = result.get("confidence", 0.0)
    st.progress(confidence, text=f"Уверенность: {confidence:.1%}")

    if confidence < 0.5:
        st.warning(
            "Низкая уверенность классификации. "
            "Рекомендуется проверить результат вручную."
        )

    with st.expander("Подробности"):
        st.json(result)


def _display_batch_results(result: dict[str, Any]) -> None:
    """Отображение результатов пакетной классификации.

    Args:
        result: Данные результатов пакетной классификации.
    """
    results = result.get("results", [])
    errors = result.get("errors", [])

    st.success(
        f"Обработано: {result.get('total', 0)} файлов. "
        f"Ошибок: {len(errors)}."
    )

    if results:
        table_data = [
            {
                "Файл": r.get("filename", ""),
                "Тип": r.get("doc_type", ""),
                "Уверенность": f"{r.get('confidence', 0.0):.1%}",
            }
            for r in results
        ]
        st.table(table_data)

    if errors:
        st.error("Ошибки при обработке:")
        for err in errors:
            st.write(f"- {err.get('filename', '?')}: {err.get('error', '?')}")


def _page_history(client: ApiClient) -> None:
    """Страница истории: таблица обработанных документов.

    Args:
        client: Экземпляр API-клиента.
    """
    st.header("История документов")

    col1, col2 = st.columns([1, 4])
    with col1:
        page_size = st.selectbox(
            "На странице",
            options=[10, 20, 50],
            index=1,
        )
    with col2:
        page = st.number_input("Страница", min_value=1, value=1, step=1)

    data = client.get_documents(page=page, page_size=page_size)

    if data is None:
        st.info("Не удалось загрузить данные. Проверьте подключение к серверу.")
        return

    documents = data.get("documents", [])
    total = data.get("total", 0)

    st.write(f"Всего документов: **{total}**")

    if not documents:
        st.info("Документы не найдены.")
        return

    table_data = [
        {
            "ID": doc.get("doc_id", "")[:8] + "...",
            "Файл": doc.get("filename", ""),
            "Тип": doc.get("doc_type", ""),
            "Уверенность": f"{doc.get('confidence', 0.0):.1%}",
            "Дата": doc.get("created_at", ""),
        }
        for doc in documents
    ]
    st.table(table_data)

    # Детали документа
    st.subheader("Просмотр деталей")
    doc_id_input = st.text_input(
        "Введите ID документа для просмотра подробностей"
    )
    if doc_id_input and st.button("Показать"):
        doc_detail = client.get_document(doc_id_input)
        if doc_detail:
            st.json(doc_detail)


def _page_feedback(client: ApiClient) -> None:
    """Страница обратной связи: исправление классификации.

    Args:
        client: Экземпляр API-клиента.
    """
    st.header("Обратная связь")
    st.write(
        "Если классификатор ошибся, укажите правильный тип документа."
    )

    with st.form("feedback_form"):
        doc_id = st.text_input(
            "ID документа",
            help="Идентификатор документа из результатов классификации.",
        )
        predicted_type = st.text_input(
            "Предсказанный тип",
            help="Тип, определённый классификатором.",
        )

        doc_types = [
            "invoice",
            "contract",
            "act",
            "report",
            "letter",
            "order",
            "receipt",
            "other",
        ]
        correct_type = st.selectbox(
            "Правильный тип",
            options=doc_types,
            help="Выберите правильный тип документа.",
        )
        custom_type = st.text_input(
            "Или укажите свой тип",
            help="Если нужного типа нет в списке.",
        )

        user_comment = st.text_area(
            "Комментарий",
            placeholder="Опишите, почему классификация неверна...",
        )

        submitted = st.form_submit_button(
            "Отправить отзыв", type="primary"
        )

    if submitted:
        if not doc_id.strip():
            st.error("Укажите ID документа.")
            return

        final_type = custom_type.strip() if custom_type.strip() else correct_type

        with st.spinner("Отправка..."):
            response = client.submit_feedback(
                doc_id=doc_id.strip(),
                predicted_type=predicted_type.strip(),
                correct_type=final_type,
                user_comment=user_comment,
            )

        if response:
            st.success("Отзыв успешно отправлен! Спасибо за обратную связь.")
            logger.info(
                "feedback_submitted_via_ui",
                doc_id=doc_id,
                correct_type=final_type,
            )


def _page_stats(client: ApiClient) -> None:
    """Страница статистики: графики и метрики.

    Args:
        client: Экземпляр API-клиента.
    """
    st.header("Статистика классификации")

    stats = client.get_stats()
    if stats is None:
        st.info("Не удалось загрузить статистику.")
        return

    # Основные метрики
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего документов", stats.get("total_documents", 0))
    with col2:
        st.metric("Всего отзывов", stats.get("total_feedback", 0))
    with col3:
        st.metric("Исправлений", stats.get("total_corrections", 0))
    with col4:
        accuracy = stats.get("accuracy", 0.0)
        st.metric("Точность", f"{accuracy:.1%}")

    # Точность
    st.subheader("Точность классификации")
    recent_accuracy = stats.get("recent_accuracy", 0.0)
    overall_accuracy = stats.get("accuracy", 0.0)

    col_a, col_b = st.columns(2)
    with col_a:
        st.progress(
            overall_accuracy,
            text=f"Общая точность: {overall_accuracy:.1%}",
        )
    with col_b:
        st.progress(
            recent_accuracy,
            text=f"Недавняя точность: {recent_accuracy:.1%}",
        )

    # Распределение по типам
    documents_by_type = stats.get("documents_by_type", {})
    if documents_by_type:
        st.subheader("Распределение по типам документов")
        chart_data = {
            "Тип документа": list(documents_by_type.keys()),
            "Количество": list(documents_by_type.values()),
        }
        st.bar_chart(
            data=chart_data,
            x="Тип документа",
            y="Количество",
        )

    # Частые ошибки
    corrections_by_type = stats.get("corrections_by_type", [])
    if corrections_by_type:
        st.subheader("Частые ошибки классификации")
        error_table = [
            {
                "Предсказано": c.get("predicted_type", ""),
                "Правильно": c.get("correct_type", ""),
                "Количество": c.get("count", 0),
            }
            for c in corrections_by_type
        ]
        st.table(error_table)


def _page_settings(client: ApiClient) -> None:
    """Страница настроек: конфигурация классификатора.

    Args:
        client: Экземпляр API-клиента.
    """
    st.header("Настройки")

    # Подключение к серверу
    st.subheader("Подключение к серверу")
    api_url = st.text_input(
        "URL API-сервера",
        value=st.session_state.get("api_url", DEFAULT_API_URL),
        help="Базовый URL бэкенда DocSort AI.",
    )

    if st.button("Проверить подключение"):
        test_client = ApiClient(api_url)
        health = test_client.health_check()
        if health:
            st.success(
                f"Сервер доступен. Версия: {health.get('version', '?')}"
            )
            st.session_state["api_url"] = api_url
        else:
            st.error("Сервер недоступен. Проверьте URL и доступность сервера.")

    st.divider()

    # Настройки классификации
    st.subheader("Параметры классификации")

    confidence_threshold = st.slider(
        "Порог уверенности",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get("confidence_threshold", 0.5),
        step=0.05,
        help="Документы с уверенностью ниже порога помечаются для проверки.",
    )
    st.session_state["confidence_threshold"] = confidence_threshold

    st.divider()

    # Настройки сортировки
    st.subheader("Параметры сортировки")

    output_dir = st.text_input(
        "Директория для отсортированных файлов",
        value=st.session_state.get("output_dir", "./sorted_documents"),
        help="Путь к директории, куда будут сортироваться файлы.",
    )
    st.session_state["output_dir"] = output_dir

    create_type_dirs = st.checkbox(
        "Создавать поддиректории по типу документа",
        value=st.session_state.get("create_type_dirs", True),
    )
    st.session_state["create_type_dirs"] = create_type_dirs

    create_date_dirs = st.checkbox(
        "Создавать поддиректории по дате",
        value=st.session_state.get("create_date_dirs", True),
    )
    st.session_state["create_date_dirs"] = create_date_dirs

    copy_mode = st.checkbox(
        "Копировать файлы (вместо перемещения)",
        value=st.session_state.get("copy_mode", False),
    )
    st.session_state["copy_mode"] = copy_mode

    st.divider()

    # Настройки переобучения
    st.subheader("Параметры переобучения")

    retrain_threshold = st.number_input(
        "Порог исправлений для переобучения",
        min_value=1,
        max_value=1000,
        value=st.session_state.get("retrain_threshold", 50),
        step=10,
        help="Количество исправлений, после которого рекомендуется переобучение.",
    )
    st.session_state["retrain_threshold"] = retrain_threshold

    if st.button("Сохранить настройки", type="primary"):
        st.success("Настройки сохранены.")
        logger.info(
            "settings_saved",
            api_url=api_url,
            confidence_threshold=confidence_threshold,
            output_dir=output_dir,
        )


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------


def main() -> None:
    """Точка входа Streamlit-приложения DocSort AI."""
    st.set_page_config(
        page_title="DocSort AI",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("DocSort AI")
    st.caption("Система автоматической классификации документов")

    # Навигация
    pages = {
        "Классификация": _page_main,
        "История": _page_history,
        "Обратная связь": _page_feedback,
        "Статистика": _page_stats,
        "Настройки": _page_settings,
    }

    selected_page = st.sidebar.radio(
        "Навигация",
        options=list(pages.keys()),
        index=0,
    )

    # Статус сервера в сайдбаре
    st.sidebar.divider()
    api_url = _get_api_url()
    client = ApiClient(api_url)

    health = client.health_check()
    if health:
        st.sidebar.success(f"Сервер: подключён (v{health.get('version', '?')})")
    else:
        st.sidebar.error("Сервер: недоступен")
        st.sidebar.caption(f"URL: {api_url}")

    # Рендер выбранной страницы
    page_func = pages.get(selected_page, _page_main)
    page_func(client)


if __name__ == "__main__":
    main()
