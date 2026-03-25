#!/usr/bin/env python3
"""Генератор тестовых документов для обучения и тестирования.

Создаёт текстовые файлы, имитирующие различные типы российских деловых
документов после OCR-распознавания. Генерируемые образцы предназначены
для обучения и тестирования модели классификации.

Поддерживаемые типы документов:
  - invoice (Счёт-фактура)
  - act (Акт выполненных работ)
  - contract (Договор)
  - waybill (Товарная накладная ТОРГ-12)
  - payment_order (Платёжное поручение)

Пример запуска::

    python scripts/generate_samples.py --output-dir data/training --samples-per-type 20
    python scripts/generate_samples.py --output-dir data/training --samples-per-type 50 --verbose
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Callable

logger = logging.getLogger("generate_samples")

# ------------------------------------------------------------------
# Словари для генерации реалистичных данных
# ------------------------------------------------------------------

COMPANY_NAMES: list[str] = [
    'ООО «Технологии Будущего»',
    'АО «Прогресс»',
    'ПАО «Сбербанк»',
    'ООО «СофтДев Лаб»',
    'ООО «Цифровые Решения»',
    'ПАО «РосНефть»',
    'АО «РитейлМарт»',
    'ООО «Складские Технологии»',
    'ПАО «Газпром»',
    'ООО «АльфаСтрой»',
    'АО «МегаТрейд»',
    'ООО «ИнфоСервис»',
    'ПАО «ВТБ»',
    'ООО «Ромашка»',
    'АО «Вектор Плюс»',
    'ООО «ТехноГрупп»',
    'ПАО «МТС»',
    'ООО «Дельта Консалтинг»',
    'АО «Северсталь»',
    'ООО «Горизонт»',
]

CITIES: list[str] = [
    "г. Москва",
    "г. Санкт-Петербург",
    "г. Новосибирск",
    "г. Екатеринбург",
    "г. Нижний Новгород",
    "г. Казань",
    "г. Самара",
    "г. Челябинск",
    "г. Ростов-на-Дону",
    "г. Краснодар",
]

STREETS: list[str] = [
    "ул. Ленина",
    "ул. Пушкина",
    "Невский проспект",
    "ул. Бутырская",
    "ул. Вавилова",
    "проспект Мира",
    "ул. Тверская",
    "ул. Арбат",
    "Дербеневская наб.",
    "ул. Гагарина",
]

PRODUCT_NAMES: list[str] = [
    "Модуль обработки данных",
    "Лицензия ПО (годовая)",
    "Техническая поддержка",
    "Бумага А4, 80 г/м²",
    "Картридж HP CF226A",
    "Папка-регистратор",
    "Скрепки канцелярские",
    "Монитор Dell 27\"",
    "Клавиатура беспроводная",
    "Мышь оптическая",
    "Сервер HP ProLiant",
    "Кабель сетевой Cat6",
    "Принтер лазерный HP",
    "Тонер для принтера",
    "USB-накопитель 64 ГБ",
]

SERVICE_NAMES: list[str] = [
    "Разработка модуля аналитики",
    "Тестирование и отладка",
    "Подготовка технической документации",
    "Настройка серверного оборудования",
    "Внедрение системы документооборота",
    "Аудит информационной безопасности",
    "Разработка веб-приложения",
    "Интеграция платёжной системы",
    "Обучение персонала",
    "Консультационные услуги по ИТ",
    "Миграция данных",
    "Техническое обслуживание сети",
]

BANK_NAMES: list[str] = [
    ("ПАО «ВТБ»", "044525745"),
    ("АО «Альфа-Банк»", "044525593"),
    ("ПАО «Газпромбанк»", "044525823"),
    ("ПАО «Сбербанк»", "044525225"),
    ("АО «Райффайзенбанк»", "044525700"),
    ("ПАО «Промсвязьбанк»", "044525555"),
    ("АО «Россельхозбанк»", "044525111"),
    ("ПАО «Московский Кредитный Банк»", "044525659"),
]

UNITS: list[str] = ["шт.", "упаковка", "пачка", "мес.", "час", "комп.", "компл."]


# ------------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------------


def _random_inn_10() -> str:
    """Генерирует случайный 10-значный ИНН."""
    return f"77{random.randint(0, 99):02d}{random.randint(100000, 999999)}"


def _random_inn_12() -> str:
    """Генерирует случайный 12-значный ИНН."""
    return f"77{random.randint(0, 99):02d}{random.randint(10000000, 99999999)}"


def _random_kpp() -> str:
    """Генерирует случайный КПП (9 цифр)."""
    return f"77{random.randint(0, 99):02d}01001"


def _random_ogrn() -> str:
    """Генерирует случайный ОГРН (13 цифр)."""
    return f"1{random.randint(0, 9):01d}{random.randint(10000000000, 99999999999)}"


def _random_account() -> str:
    """Генерирует случайный номер расчётного счёта (20 цифр)."""
    return f"40702810{random.randint(100000000000, 999999999999)}"


def _random_corr_account() -> str:
    """Генерирует случайный номер корреспондентского счёта."""
    return f"30101810{random.randint(100000000000, 999999999999)}"


def _random_date() -> str:
    """Генерирует случайную дату в формате ДД.ММ.ГГГГ."""
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(2024, 2026)
    return f"{day:02d}.{month:02d}.{year}"


def _random_date_text() -> str:
    """Генерирует дату в текстовом формате для договоров."""
    months = [
        "января", "февраля", "марта", "апреля", "мая", "июня",
        "июля", "августа", "сентября", "октября", "ноября", "декабря",
    ]
    day = random.randint(1, 28)
    month = random.choice(months)
    year = random.randint(2024, 2026)
    return f"«{day:02d}» {month} {year} г."


def _random_amount() -> tuple[float, float, float]:
    """Генерирует случайную сумму с НДС.

    Returns:
        Кортеж (сумма без НДС, НДС, сумма с НДС).
    """
    base = round(random.uniform(10000, 5000000), 2)
    nds = round(base * 0.2, 2)
    total = round(base + nds, 2)
    return base, nds, total


def _random_bank() -> tuple[str, str]:
    """Возвращает случайный банк (название, БИК)."""
    return random.choice(BANK_NAMES)


# ------------------------------------------------------------------
# Генераторы документов
# ------------------------------------------------------------------


def generate_invoice(doc_number: int) -> str:
    """Генерирует текст счёт-фактуры.

    Args:
        doc_number: Порядковый номер документа.

    Returns:
        Текст документа, имитирующий OCR-распознавание счёт-фактуры.
    """
    seller = random.choice(COMPANY_NAMES)
    buyer = random.choice([c for c in COMPANY_NAMES if c != seller])
    city_seller = random.choice(CITIES)
    city_buyer = random.choice(CITIES)
    street_seller = random.choice(STREETS)
    street_buyer = random.choice(STREETS)
    date = _random_date()
    num = random.randint(1, 9999)

    lines: list[str] = [
        f"СЧЁТ-ФАКТУРА № {num} от {date}",
        "",
        f"Продавец: {seller}",
        f"Адрес: {city_seller}, {street_seller}, д. {random.randint(1, 100)}",
        f"ИНН: {_random_inn_10()}",
        f"КПП: {_random_kpp()}",
        f"ОГРН: {_random_ogrn()}",
        "",
        f"Покупатель: {buyer}",
        f"Адрес: {city_buyer}, {street_buyer}, д. {random.randint(1, 100)}",
        f"ИНН: {_random_inn_10()}",
        f"КПП: {_random_kpp()}",
        "",
        "№ | Наименование товара | Кол-во | Ед.изм. | Цена | Сумма",
    ]

    total_base = 0.0
    n_items = random.randint(1, 6)
    for i in range(1, n_items + 1):
        product = random.choice(PRODUCT_NAMES)
        qty = random.randint(1, 100)
        unit = random.choice(UNITS)
        price = round(random.uniform(500, 200000), 2)
        amount = round(qty * price, 2)
        total_base += amount
        lines.append(f"{i} | {product} | {qty} | {unit} | {price:.2f} | {amount:.2f}")

    nds = round(total_base * 0.2, 2)
    total = round(total_base + nds, 2)

    lines.extend([
        "",
        f"Итого без НДС: {total_base:.2f} руб.",
        f"НДС (20%): {nds:.2f} руб.",
        f"Итого к оплате: {total:.2f} руб.",
        "",
        "Руководитель: _________________ / Иванов И.И. /",
        "Главный бухгалтер: _________________ / Петрова А.С. /",
    ])

    return "\n".join(lines)


def generate_act(doc_number: int) -> str:
    """Генерирует текст акта выполненных работ.

    Args:
        doc_number: Порядковый номер документа.

    Returns:
        Текст документа, имитирующий OCR-распознавание акта.
    """
    executor = random.choice(COMPANY_NAMES)
    customer = random.choice([c for c in COMPANY_NAMES if c != executor])
    city = random.choice(CITIES)
    date = _random_date()
    num = random.randint(1, 999)
    contract_num = f"2026-К/{random.randint(100, 999)}"
    contract_date = _random_date()

    variant = random.choice(["выполненных работ", "оказанных услуг"])

    lines: list[str] = [
        f"АКТ № {num}",
        f"{variant}",
        "",
        f"{city}    {date}",
        "",
        f"Исполнитель: {executor}",
        f"ИНН: {_random_inn_10()}  КПП: {_random_kpp()}",
        "",
        f"Заказчик: {customer}",
        f"ИНН: {_random_inn_10()}  КПП: {_random_kpp()}",
        "",
        f"Основание: Договор № {contract_num} от {contract_date}",
        "",
        "№ | Наименование работ / услуг | Кол-во | Ед. | Стоимость",
    ]

    total_base = 0.0
    n_items = random.randint(1, 5)
    for i in range(1, n_items + 1):
        service = random.choice(SERVICE_NAMES)
        qty = random.randint(1, 200)
        unit = random.choice(["шт.", "час", "комп."])
        price = round(random.uniform(5000, 500000), 2)
        amount = round(qty * price, 2)
        total_base += amount
        lines.append(f"{i} | {service} | {qty} | {unit} | {amount:.2f}")

    nds = round(total_base * 0.2, 2)
    total = round(total_base + nds, 2)

    lines.extend([
        "",
        f"Итого: {total_base:.2f} руб.",
        f"НДС (20%): {nds:.2f} руб.",
        f"Всего к оплате: {total:.2f} руб.",
        "",
        "Вышеперечисленные работы выполнены полностью и в срок.",
        "Заказчик претензий по объёму, качеству и срокам оказания услуг не имеет.",
        "",
        "Исполнитель: _________________ / Сидоров К.М. /",
        "Заказчик:    _________________ / Козлов Д.А. /",
    ])

    return "\n".join(lines)


def generate_contract(doc_number: int) -> str:
    """Генерирует текст договора.

    Args:
        doc_number: Порядковый номер документа.

    Returns:
        Текст документа, имитирующий OCR-распознавание договора.
    """
    executor = random.choice(COMPANY_NAMES)
    customer = random.choice([c for c in COMPANY_NAMES if c != executor])
    city = random.choice(CITIES)
    date_text = _random_date_text()
    num = f"2026-У/{random.randint(1, 999):03d}"
    bank_exec, bik_exec = _random_bank()
    bank_cust, bik_cust = _random_bank()

    subjects = [
        "оказать услуги по внедрению и настройке системы электронного документооборота",
        "выполнить работы по разработке программного обеспечения",
        "оказать консультационные услуги по информационной безопасности",
        "выполнить работы по модернизации ИТ-инфраструктуры",
        "оказать услуги по обслуживанию серверного оборудования",
        "осуществить поставку офисного оборудования и расходных материалов",
    ]

    base, nds, total = _random_amount()

    lines: list[str] = [
        f"ДОГОВОР № {num}",
        "на оказание услуг",
        "",
        f"{city}    {date_text}",
        "",
        f"{executor}, именуемое в дальнейшем «Исполнитель»,",
        "в лице генерального директора, действующего на основании Устава,",
        "с одной стороны, и",
        f"{customer}, именуемое в дальнейшем «Заказчик»,",
        "в лице директора, действующего на основании доверенности,",
        "с другой стороны, заключили настоящий договор о нижеследующем:",
        "",
        "1. ПРЕДМЕТ ДОГОВОРА",
        f"1.1. Исполнитель обязуется {random.choice(subjects)}.",
        f"1.2. Стороны договорились о стоимости услуг в размере {total:.2f} руб.,",
        "     включая НДС 20%.",
        "",
        "2. ПРАВА И ОБЯЗАННОСТИ СТОРОН",
        "2.1. Исполнитель обязуется:",
        f"     а) выполнить работы в срок до {_random_date()};",
        "     б) предоставить отчёт о выполненных работах.",
        "2.2. Заказчик обязуется:",
        "     а) обеспечить доступ к необходимой информации;",
        f"     б) произвести оплату в размере {total:.2f} руб.",
        "",
        "3. РЕКВИЗИТЫ СТОРОН",
        "",
        "Исполнитель:                        Заказчик:",
        f"{executor}              {customer}",
        f"ИНН {_random_inn_10()}                      ИНН {_random_inn_10()}",
        f"КПП {_random_kpp()}                       КПП {_random_kpp()}",
        f"р/с {_random_account()}            р/с {_random_account()}",
        f"в {bank_exec}                  в {bank_cust}",
        f"БИК {bik_exec}                       БИК {bik_cust}",
        "",
        "ПОДПИСИ СТОРОН:",
        "",
        "Исполнитель: _________________ ",
        "Заказчик:    _________________ ",
    ]

    return "\n".join(lines)


def generate_waybill(doc_number: int) -> str:
    """Генерирует текст товарной накладной ТОРГ-12.

    Args:
        doc_number: Порядковый номер документа.

    Returns:
        Текст документа, имитирующий OCR-распознавание накладной.
    """
    sender = random.choice(COMPANY_NAMES)
    receiver = random.choice([c for c in COMPANY_NAMES if c != sender])
    date = _random_date()
    num = random.randint(1, 9999)

    lines: list[str] = [
        "Унифицированная форма № ТОРГ-12",
        "Утверждена постановлением Госкомстата России от 25.12.98 № 132",
        "",
        f"ТОВАРНАЯ НАКЛАДНАЯ № {num} от {date}",
        "",
        f"Грузоотправитель: {sender}",
        f"ИНН: {_random_inn_10()}  КПП: {_random_kpp()}",
        f"Адрес: {random.choice(CITIES)}, {random.choice(STREETS)}, д. {random.randint(1, 100)}",
        "",
        f"Грузополучатель: {receiver}",
        f"ИНН: {_random_inn_10()}  КПП: {_random_kpp()}",
        f"Адрес: {random.choice(CITIES)}, {random.choice(STREETS)}, д. {random.randint(1, 100)}",
        "",
        f"Поставщик: {sender}",
        f"Плательщик: {receiver}",
        f"Основание: Договор поставки № {random.randint(1, 500)}/2026 от {_random_date()}",
        "",
        "№ | Наименование товара | Единица измерения | Кол-во | Цена | Сумма",
    ]

    total_base = 0.0
    n_items = random.randint(2, 8)
    for i in range(1, n_items + 1):
        product = random.choice(PRODUCT_NAMES)
        qty = random.randint(1, 1000)
        unit = random.choice(UNITS)
        price = round(random.uniform(50, 50000), 2)
        amount = round(qty * price, 2)
        total_base += amount
        lines.append(f"{i} | {product} | {unit} | {qty} | {price:.2f} | {amount:.2f}")

    nds = round(total_base * 0.2, 2)
    total = round(total_base + nds, 2)

    lines.extend([
        "",
        f"Итого: {total_base:.2f} руб.",
        f"НДС (20%): {nds:.2f} руб.",
        f"Всего с НДС: {total:.2f} руб.",
        "",
        "Отпуск разрешил: _________________ ",
        "Главный бухгалтер: _________________ ",
        "Отпуск груза произвёл: _________________ ",
        "Груз получил: _________________ ",
    ])

    return "\n".join(lines)


def generate_payment_order(doc_number: int) -> str:
    """Генерирует текст платёжного поручения.

    Args:
        doc_number: Порядковый номер документа.

    Returns:
        Текст документа, имитирующий OCR-распознавание платёжного поручения.
    """
    payer = random.choice(COMPANY_NAMES)
    payee = random.choice([c for c in COMPANY_NAMES if c != payer])
    bank_payer, bik_payer = _random_bank()
    bank_payee, bik_payee = _random_bank()
    date = _random_date()
    num = random.randint(1, 9999)
    _, _, total = _random_amount()
    invoice_num = random.randint(1, 999)
    invoice_date = _random_date()
    nds = round(total * 20 / 120, 2)

    amount_int = int(total)
    kopecks = int(round((total - amount_int) * 100))

    lines: list[str] = [
        f"ПЛАТЁЖНОЕ ПОРУЧЕНИЕ № {num}",
        f"Дата: {date}    Вид платежа: электронно",
        "",
        f"Сумма: {total:.2f}",
        "",
        f"Плательщик: {payer}",
        f"ИНН: {_random_inn_10()}  КПП: {_random_kpp()}",
        f"Сч. №: {_random_account()}",
        "",
        f"Банк плательщика: {bank_payer}",
        f"БИК: {bik_payer}",
        f"Корр. счёт: {_random_corr_account()}",
        "",
        f"Получатель: {payee}",
        f"ИНН: {_random_inn_10()}  КПП: {_random_kpp()}",
        f"Сч. №: {_random_account()}",
        "",
        f"Банк получателя: {bank_payee}",
        f"БИК: {bik_payee}",
        f"Корр. счёт: {_random_corr_account()}",
        "",
        "Вид оп.: 01    Срок плат.:    Наз. пл.:",
        "Очер. плат.: 5     Код:",
        "",
        f"Назначение платежа: Оплата по счёт-фактуре № {invoice_num} от {invoice_date}.",
        f"В т.ч. НДС (20%) {nds:.2f} руб.",
        "",
        "М.П.      Подписи: _________________ / _________________",
    ]

    return "\n".join(lines)


# ------------------------------------------------------------------
# Основная логика генерации
# ------------------------------------------------------------------

#: Маппинг типов документов на генераторы.
GENERATORS: dict[str, Callable[[int], str]] = {
    "invoice": generate_invoice,
    "act": generate_act,
    "contract": generate_contract,
    "waybill": generate_waybill,
    "payment_order": generate_payment_order,
}


def generate_samples(
    output_dir: Path,
    samples_per_type: int = 20,
    seed: int | None = None,
) -> dict[str, int]:
    """Генерирует тестовые образцы всех типов документов.

    Создаёт структурированную директорию с поддиректориями для каждого
    типа документа. В каждой поддиректории создаётся указанное количество
    текстовых файлов.

    Args:
        output_dir: Корневая директория для сохранения образцов.
        samples_per_type: Количество образцов каждого типа.
        seed: Seed для генератора случайных чисел (для воспроизводимости).

    Returns:
        Словарь: тип документа -> количество сгенерированных файлов.
    """
    if seed is not None:
        random.seed(seed)

    stats: dict[str, int] = {}

    for doc_type, generator in GENERATORS.items():
        type_dir = output_dir / doc_type
        type_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for i in range(1, samples_per_type + 1):
            text = generator(i)
            filename = f"{doc_type}_{i:04d}.txt"
            file_path = type_dir / filename

            file_path.write_text(text, encoding="utf-8")
            count += 1
            logger.debug("Создан: %s", file_path)

        stats[doc_type] = count
        logger.info("Тип '%s': сгенерировано %d образцов", doc_type, count)

    total = sum(stats.values())
    logger.info("Всего сгенерировано: %d образцов", total)

    return stats


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разбирает аргументы командной строки.

    Args:
        argv: Список аргументов. Если None, используется sys.argv.

    Returns:
        Разобранные аргументы.
    """
    parser = argparse.ArgumentParser(
        description="Генератор тестовых документов для DocSort AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s --output-dir data/training --samples-per-type 20
  %(prog)s --output-dir data/training --samples-per-type 50 --seed 42
  %(prog)s --output-dir data/training --samples-per-type 100 --verbose
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Директория для сохранения сгенерированных образцов",
    )

    parser.add_argument(
        "--samples-per-type",
        type=int,
        default=20,
        help="Количество образцов каждого типа (по умолчанию: 20)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed для генератора случайных чисел (для воспроизводимости)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Подробный вывод (уровень DEBUG)",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Главная функция генератора тестовых документов.

    Args:
        argv: Аргументы командной строки. Если None, используется sys.argv.

    Returns:
        Код возврата: 0 — успех, 1 — ошибка.
    """
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("DocSort AI — Генератор тестовых документов")
    logger.info("=" * 60)
    logger.info("Директория вывода: %s", args.output_dir)
    logger.info("Образцов на тип: %d", args.samples_per_type)
    if args.seed is not None:
        logger.info("Seed: %d", args.seed)

    try:
        stats = generate_samples(
            output_dir=args.output_dir,
            samples_per_type=args.samples_per_type,
            seed=args.seed,
        )

        logger.info("-" * 40)
        logger.info("Результат генерации:")
        for doc_type, count in sorted(stats.items()):
            logger.info("  %s: %d файлов", doc_type, count)
        logger.info("  Всего: %d файлов", sum(stats.values()))
        logger.info("-" * 40)
        logger.info("Генерация завершена успешно")

        return 0

    except Exception as exc:
        logger.exception("Ошибка генерации: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
