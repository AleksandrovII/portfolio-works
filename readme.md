# Portfolio Works

Инструменты для анализа инвестиционного портфеля через T-Invest API.

## Быстрый старт

```bash
# Минимум — только T-Invest
export INVEST_TOKEN='ваш_токен'

# Полная настройка — T-Invest + Финам
export INVEST_TOKEN='ваш_токен_тинвест'
export FINAM_TOKEN='ваш_токен_финам'
```

---

## Структура проекта

```
portfolio-works/
├── portfolio_works_library.py   # Общая библиотека (API-клиент, базовые функции)
├── call_api.py                  # Шаг 1: скачать портфель → data/portfolio.csv
├── data/
│   ├── portfolio.csv            # Снимок портфеля (генерируется автоматически)
│   └── deals.csv                # История сделок и движений (генерируется автоматически)
├── Instruments/
│   ├── import_deals.py          # Импорт сделок и движений по счёту
│   ├── portfolio_visual.py      # Визуализация структуры портфеля
│   └── Markovitz_optimizer/
│       └── mark.py              # Оптимизация по Марковицу
└── Results/                     # Сюда сохраняются графики
```

---

## Скрипты

### 1. `call_api.py` — снимок портфеля

Подключается к T-Invest API и сохраняет текущие позиции в `data/portfolio.csv`.
Запускать перед любым из скриптов в `Instruments/`.

```bash
/opt/miniconda3/bin/python call_api.py
```

Что сохраняется в `portfolio.csv`:

| Колонка | Описание |
|---|---|
| `figi` | Идентификатор инструмента |
| `ticker` | Тикер |
| `name` | Название |
| `account` | Брокерский счёт |
| `instrument_type` | Тип: share / bond / etf / precious_metal / currency |
| `currency` | Валюта инструмента |
| `quantity` | Количество в портфеле |
| `avg_price` | Средняя цена покупки |
| `cur_price` | Текущая цена |
| `rub_value` | Стоимость в рублях |
| `weight_pct` | Доля в портфеле, % |

---

### 2. `Instruments/import_deals.py` — история сделок

Импортирует все исполненные операции по счетам и сохраняет в `data/deals.csv`.
При повторном запуске дубли пропускаются — данные накапливаются.

```bash
# Последние 3 месяца (по умолчанию)
/opt/miniconda3/bin/python Instruments/import_deals.py

# Последние N месяцев
/opt/miniconda3/bin/python Instruments/import_deals.py 6
/opt/miniconda3/bin/python Instruments/import_deals.py 12
```

Что сохраняется в `deals.csv`:

| `type` | `direction` | Описание |
|---|---|---|
| `trade` | `buy` / `sell` | Покупки и продажи |
| `deposit` | — | Пополнения счёта |
| `withdrawal` | — | Снятия со счёта |
| `dividend` | — | Дивиденды |
| `coupon` | — | Купонный доход |
| `tax` | — | Налоги |

Основные колонки: `operation_id`, `date`, `account`, `type`, `direction`, `ticker`, `quantity`, `price`, `price_currency`, `payment`, `payment_currency`, `commission`.

---

### 3. `Instruments/portfolio_visual.py` — визуализация портфеля

Читает `data/portfolio.csv` и строит три диаграммы. Токен не нужен.
Результат сохраняется в `Results/portfolio_visual.png`.

```bash
/opt/miniconda3/bin/python Instruments/portfolio_visual.py
```

Диаграммы:
- **Donut** — распределение по типам активов
- **Горизонтальные бары** — топ-15 позиций по стоимости
- **Stacked bars** — разбивка по брокерским счетам

---

### 4. `Instruments/Markovitz_optimizer/mark.py` — оптимизация Марковица

Читает `data/portfolio.csv`, подтягивает историю цен из T-Invest API и
запускает оптимизацию. Результат сохраняется в `Results/markowitz.png`.

```bash
# 365-дневное окно (по умолчанию)
/opt/miniconda3/bin/python Instruments/Markovitz_optimizer/mark.py

# Своё окно и фильтры
/opt/miniconda3/bin/python Instruments/Markovitz_optimizer/mark.py 730
/opt/miniconda3/bin/python Instruments/Markovitz_optimizer/mark.py 365 --top 15
/opt/miniconda3/bin/python Instruments/Markovitz_optimizer/mark.py 365 --spearman
```

Что выводится:
- Консольный отчёт: Sharpe, волатильность, risk score, оценка диверсификации
- График: эффективная граница, сравнение весов, матрица корреляций, спидометр риска

---

## Типичный рабочий процесс

```bash
# 1. Обновить снимок портфеля
/opt/miniconda3/bin/python call_api.py

# 2. Посмотреть структуру
/opt/miniconda3/bin/python Instruments/portfolio_visual.py

# 3. Запустить анализ Марковица
/opt/miniconda3/bin/python Instruments/Markovitz_optimizer/mark.py

# 4. Импортировать свежие сделки
/opt/miniconda3/bin/python Instruments/import_deals.py 3
```

---

## Переменные окружения

| Переменная | Обязательна | Описание |
|---|---|---|
| `INVEST_TOKEN` | Да | Токен T-Invest API |
| `FINAM_TOKEN` | Нет | Токен Финам Trade API |
| `FINAM_CLIENT_ID` | Нет | Клиентский код Финам (5–6 цифр) |

---

## Как получить токен T-Invest

1. Зайти на [tbank.ru](https://www.tbank.ru) → Инвестиции → Ещё → Настройки
2. Раздел «Торговля роботами» → «Получить токен»
3. Выбрать тип: **полный доступ** (для чтения портфеля)
4. Скопировать и сохранить токен — он показывается только один раз

---

## Как получить токен Финам

### Шаг 1 — создать токен

1. Зайти в личный кабинет [finam.ru](https://finam.ru) → кнопка **«Войти»**
2. Перейти: **Профиль** → **Безопасность** → раздел **«Торговый API»**
3. Нажать **«Создать токен»**, выбрать срок действия
4. Скопировать токен — после закрытия окна он не будет виден снова

### Шаг 2 — узнать клиентский код

Клиентский код (`FINAM_CLIENT_ID`) — это 5–6-значный номер, который виден:
- В шапке личного кабинета (обычно рядом с именем)
- В разделе **Профиль** → **Личные данные**

### Шаг 3 — настроить окружение

```bash
export FINAM_TOKEN='eyJhbGciOi...'   # токен из шага 1
export FINAM_CLIENT_ID='123456'       # ваш клиентский код
```

> **Важно:** позиции Финам попадают в `portfolio.csv` и отображаются
> в графиках, но **не участвуют в анализе Марковица и корреляций** —
> для этого нужен FIGI, которого нет в Finam API.
> В колонке `account` они будут отмечены как `Finam 123456`.

---

## Python

Используется Conda base: `/opt/miniconda3/bin/python` (Python 3.13).
