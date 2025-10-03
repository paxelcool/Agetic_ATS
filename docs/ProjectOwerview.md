# 0) Общая идея и сценарии

* **Сценарий A («Агрессивный, но рациональный»)** — интрадей-контур: пробой диапазона/ключевого уровня **только** в сторону тренда (EMA-фильтр), подтверждение **объёмом (RVOL ≥ порога)**, стопы по ATR/уровню, частичный фикс + трейлинг.
* **Сценарий B (переход по мере роста депозита)** — добавляем среднесрочный мульти-портфель (**Donchian-20/55** и/или **MA-тайминг 100–200** на D1) с **volatility-targeting**.
* Оркестрация — **LangGraph** (состояние стратегии, «долговременная» память); публикация как REST — **LangServe**; наблюдаемость/оценка — **LangSmith**; знания и пост-мортемы — **Vector-RAG (ChromaDB)**; причинные связи сигнал→сделка→риск→режим — **Graph-RAG (Memgraph)**. ([langchain-ai.github.io][2])

---

# 1) Структура репозитория

```
ats/
├─ src/                        # основной код приложения
│  ├─ main.py                  # точка входа и CLI интерфейс
│  ├─ config.py                # настройки через Pydantic
│  ├─ agents/                  # LangGraph агенты
│  │  ├─ __init__.py
│  │  ├─ trading_graph.py       # основной граф агентов
│  │  ├─ signal_agent.py        # агент анализа сигналов
│  │  ├─ risk_agent.py          # агент управления рисками
│  │  ├─ execution_agent.py     # агент исполнения сделок
│  │  └─ monitoring_agent.py    # агент мониторинга
│  ├─ tools/                   # инструменты агентов
│  │  ├─ __init__.py
│  │  ├─ market_data.py         # получение данных из MT5
│  │  ├─ technical_indicators.py # расчет индикаторов
│  │  ├─ risk_management.py     # расчет позиций и рисков
│  │  └─ order_execution.py     # исполнение ордеров в MT5
│  ├─ database/                # код для работы с базами данных
│  │  ├─ __init__.py
│  │  ├─ models.py             # модели данных (Pydantic)
│  │  ├─ storage.py            # операции с базами данных
│  │  ├─ vector_store.py       # интеграция с ChromaDB
│  │  ├─ graph_store.py        # интеграция с Memgraph
│  │  └─ init_db.py           # инициализация баз данных
│  ├─ synchronizer/             # модуль синхронизации данных
│  │  ├─ __init__.py
│  │  ├─ sync_service.py       # основной сервис синхронизации
│  │  ├─ clients/              # клиенты внешних сервисов
│  │  │  └─ mt5_client.py      # клиент MetaTrader5
│  │  ├─ processors/           # обработка данных
│  │  │  └─ data_processor.py  # валидация и трансформация
│  │  ├─ services/             # специализированные сервисы
│  │  │  └─ embedding_service.py # создание эмбеддингов
│  │  └─ utils/                # утилиты
│  │     └─ error_handler.py   # обработка ошибок
│  └─ ui/                      # пользовательские интерфейсы
│     ├─ __init__.py
│     └─ cli.py                # командная строка
│     └─ streamlit_app.py      # веб-интерфейс (опционально)
├─ atl/
│  ├─ agents/
│  │  ├─ data_agent.py
│  │  ├─ signal_agent_a.py     # Intraday (Сценарий A)
│  │  ├─ signal_agent_b.py     # Swing/Trend (Сценарий B)
│  │  ├─ risk_agent.py
│  │  ├─ exec_agent.py
│  │  ├─ governance_agent.py
│  │  └─ explain_agent.py
│  ├─ tools/                   # инструменты ReAct/LCEL
│  │  ├─ features.py           # EMA/ATR/RVOL/Donchian/Range
│  │  ├─ risk.py               # расчёт лота, лимиты
│  │  ├─ execution.py          # обёртка над синхронизатором
│  │  ├─ memory.py             # Vector/Graph retrievers
│  │  └─ utils.py
│  ├─ prompts/
│  │  ├─ react_intraday.md
│  │  ├─ react_swing.md
│  │  └─ system_guards.md
│  └─ graphs/
│     ├─ intraday_graph.py     # LangGraph: IDLE→SETUP→ENTER→MANAGE→EXIT
│     └─ swing_graph.py
├─ storage/
│  ├─ chroma_db/               # векторная база данных (файлы)
│  │  └── chroma.sqlite3       # база данных ChromaDB
│  └─ sqlite/                  # реляционная база данных (файлы)
│     └── trades.db            # база данных SQLite
├─ configs/
│  ├─ settings.example.yaml
│  └─ logging.yaml
├─ tests/
│  ├─ unit/
│  ├─ integration/
│  └─ backtests/
├─ docker/
│  ├─ docker-compose.yml       # ChromaDB, Memgraph, API
│  └─ Dockerfile.api
├─ requirements.txt
├─ pyproject.toml              # зависимости через Poetry (PEP 621)
└─ README.md
```

> Модуль-синхронизатор обеспечивает получение и хранение данных о котировках и сделках в локальной базе данных SQLite, а также интеграцию с векторной (ChromaDB) и графовой (Memgraph) базами данных для анализа и поиска.

---

# 2) Зависимости и конфиг

**pyproject.toml (управление зависимостями через Poetry)**

```toml
[tool.poetry]
name = "agetic-ats"
version = "0.1.0"
description = "Automated Trading System with AI agents"
authors = ["Your Name <your.email@example.com>"]

[tool.poetry.dependencies]
python = "^3.12"

# MetaTrader5 для синхронизации с торговым терминалом
MetaTrader5 = "*"

# Core LangChain ecosystem (локальные модели через Ollama)
langchain = "*"
langchain-ollama = "*"  # Вместо langchain-openai для локальных моделей
langgraph = "*"

# Data validation and settings
pydantic = "*"
pydantic-settings = "*"

# Vector database для семантического поиска (устанавливается через pip)
chromadb = "*"

# Graph database для анализа связей
memgraph-python = "*"

# Data processing для анализа рыночных данных
pandas = "*"
numpy = "*"
scipy = "*"  # Для математических вычислений

# Интерфейсы пользователя
streamlit = "*"  # Для веб-панели мониторинга (опционально)
click = "*"      # Для CLI интерфейса
rich = "*"       # Для красивого вывода в терминале

# Локальные альтернативы вместо платных сервисов:
# - Ollama для LLM вместо OpenAI (бесплатно, локально)
# - ChromaDB вместо векторных сервисов (бесплатно, локально)
# - Memgraph вместо графовых сервисов (бесплатно, локально)

# Async support (встроено в Python 3.12+)
# asyncio уже включен в стандартную библиотеку

## Зачем нужны эти библиотеки? Подробное объяснение

### 🤖 LangChain экосистема (langchain, langchain-ollama, langgraph)
**Назначение:** Фреймворк для создания приложений на базе больших языковых моделей

**Почему нужен:**
- **Стандартизированный интерфейс** к разным LLM (OpenAI, Ollama, локальные модели)
- **Готовые инструменты** для агентов, цепочек и памяти
- **Интеграция с базами данных** (векторными и графовыми)

**Cloud модели через Ollama:**
```python
# Вместо OpenAI API (платно)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(api_key="your-key")  # Требует API ключ

# Используем Ollama с cloud моделями (бесплатно через облако)
from langchain_ollama import ChatOllama
llm = ChatOllama(model="gpt-oss:120b-cloud")  # Работает локально, но использует облако
```

**Как это работает:**
1. **Устанавливаете Ollama** локально (бесплатно)
2. **Авторизуетесь** через `ollama signin`
3. **Загружаете cloud модель** - `ollama pull gpt-oss:120b-cloud`
4. **Используете как локальную** - модель работает через ваш компьютер, но вычисления в облаке

### 🖥️ Интерфейсы пользователя (CLI + опционально Streamlit)

**Архитектура: приложение → MT5 терминал**

```
┌─────────────────────────────────────┐
│           Ваше приложение           │
│        ┌─────────────────┐          │
│        │   LangGraph     │          │
│        │   Агенты        │          │
│        └─────────────────┘          │
│              │                      │
│        ┌─────────────────┐          │
│        │ Синхронизатор   │◄─────────┤ MetaTrader5
│        │ (SQLite + API)  │          │ Терминал
│        └─────────────────┘          │
└─────────────────────────────────────┘
```

**Интерфейсы для управления системой:**

#### 💻 CLI (Command Line Interface)
**Назначение:** Базовый интерфейс для мониторинга и управления

```python
# Основной CLI интерфейс
class ATSCLI:
    def __init__(self):
        self.intraday_app = None
        self.swing_app = None

    async def run(self):
        """Основной цикл CLI"""
        while True:
            command = input("ATS> ")

            if command == "status":
                await self.show_status()
            elif command.startswith("signal"):
                await self.get_signal(command)
            elif command == "agents":
                await self.show_agents_status()
            elif command == "logs":
                await self.show_recent_logs()
            elif command == "quit":
                break

    async def show_status(self):
        """Показать статус системы"""
        print("📊 Статус торговой системы:")
        print(f"Баланс: {await self.get_balance()}")
        print(f"Открытые позиции: {await self.get_positions()}")
        print(f"Активные агенты: {len(self.running_agents)}")
```

#### 📊 Streamlit (опциональная веб-панель)
**Назначение:** Графический интерфейс мониторинга (если понадобится)

```python
# streamlit_app.py
import streamlit as st

async def main():
    st.title("🎯 ATS - Automated Trading System")

    # Статус системы
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Баланс", f"${await get_balance()}")
    with col2:
        st.metric("Открытые позиции", len(await get_positions()))
    with col3:
        st.metric("Активность агентов", "🟢 Активны" if agents_running else "🔴 Остановлены")

    # Графики производительности
    st.line_chart(await get_performance_data())

    # Логи последних действий
    st.subheader("Последние действия агентов")
    for log in await get_recent_logs():
        st.write(f"[{log.timestamp}] {log.agent}: {log.action}")
```

**Почему такая простая архитектура:**
- **Прямое управление** - приложение само управляет MT5
- **Внутренний мониторинг** - логи и статус изнутри системы
- **Без лишней сложности** - нет нужды в REST API для внешних клиентов
- **Легкость отладки** - прямой доступ к состоянию агентов

### 📊 LangSmith (опционально, платный)
**Назначение:** Мониторинг и отладка LLM приложений

**Почему ОПЦИОНАЛЬНО:**
- **Без него проект работает** - просто отключаем трассировку
- **Полезен для продакшена** - мониторинг стоимости, качества ответов
- **Локальная альтернатива:** встроенное логирование в файл

**Настройка (если доступен):**
```python
# Включить трассировку
settings.langchain_tracing = True
settings.langchain_api_key = "your-key"
```

### 💰 MetaTrader5
**Назначение:** Синхронизация с торговым терминалом MT5

**Почему нужен:**
- **Получение котировок** в реальном времени
- **Управление ордерами** (открытие, закрытие, модификация)
- **Доступ к истории** сделок и позиций

```python
import MetaTrader5 as mt5

# Инициализация
mt5.initialize()

# Получение котировок
symbol = "EURUSD"
tick = mt5.symbol_info_tick(symbol)
print(f"Цена: {tick.bid}")

# Открытие ордера
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": 0.1,
    "type": mt5.ORDER_TYPE_BUY,
    "price": tick.ask,
    "deviation": 20,
    "magic": 123456,
    "comment": "ATS signal"
}
result = mt5.order_send(request)
```

## Полная локальная настройка (без платных сервисов)

### Шаг 1: Установка и авторизация Ollama
```bash
# Установка Ollama (если не установлен)
curl -fsSL https://ollama.ai/install.sh | sh

# Авторизация для доступа к cloud моделям
ollama signin

# Проверка доступных моделей
ollama ls
```

### Шаг 2: Загрузка cloud модели
```bash
# Загрузка cloud модели (работает локально, но использует облако)
ollama pull gpt-oss:120b-cloud

# Проверка что модель появилась в списке
ollama ls
# NAME                      SIZE
# gpt-oss:120b-cloud        -
```

### Шаг 3: Запуск баз данных
```bash
# Векторная база данных (ChromaDB) - запускается автоматически в Python коде
# pip install chromadb

# Графовая база данных (Memgraph) - через Docker
docker run -p 7687:7687 memgraph/memgraph:latest
```

### Шаг 4: Запуск приложения
```bash
# Основное приложение (автоматически использует cloud модель)
python -m src.main
```

### Шаг 5: Конфигурация
```python
# Настройки - модель работает локально через ollama клиент
settings = TradingSettings(
    ollama_base_url="http://localhost:11434",  # локальный ollama клиент
    ollama_model="gpt-oss:120b-cloud",         # cloud модель
    langchain_tracing=False
)
```

Теперь ваш проект полностью автономен и работает локально без каких-либо платных сервисов!

### 📦 Что упростили:

**Убрали:**
- Docker для ChromaDB (теперь просто `pip install chromadb`)
- Сложные команды запуска контейнеров

**Оставили только необходимое:**
- MetaTrader5 для связи с терминалом
- Ollama для моделей (локально)
- ChromaDB как Python модуль
- Memgraph через Docker (это оправдано для графовой БД)

### 🚀 Как запускать систему

**Вариант 1: Через CLI (основной способ)**
```bash
# Запуск основного приложения с CLI интерфейсом
python -m src.main

# В CLI будет доступно:
# - status - показать статус системы
# - signal EURUSD M5 - получить торговый сигнал
# - agents - показать активность агентов
# - logs - последние действия системы
```

**Вариант 2: Через Streamlit веб-интерфейс (опционально)**
```bash
# Установка Streamlit
pip install streamlit

# Запуск веб-панели мониторинга
streamlit run src/ui/streamlit_app.py

# Доступно по адресу: http://localhost:8501
```

**Вариант 3: Прямой вызов агентов**
```python
# Программное использование
from src.main import ATS

ats = ATS()
await ats.initialize()

# Получить сигнал
signal = await ats.get_signal("EURUSD", "M5")
print(signal)
```

### 📋 Что осталось в зависимостях

Мы убрали все ненужные веб-фреймворки и оставили только необходимое:

**Оставили:**
- **MetaTrader5** - связь с торговым терминалом
- **LangChain экосистема** - для агентов
- **Базы данных** - для хранения и анализа
- **Streamlit + CLI библиотеки** - для интерфейса

**Убрали:**
- **FastAPI, uvicorn** - не нужны без внешних клиентов
- **LangServe** - не нужен отдельный веб-сервер
- **LangSmith** - платный мониторинг не обязателен

[tool.poetry.group.dev.dependencies]
pytest = "*"
pytest-asyncio = "*"
black = "*"
isort = "*"
mypy = "*"
ruff = "*"

[tool.poetry.group.test.dependencies]
pytest = "*"
pytest-asyncio = "*"
pytest-cov = "*"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

# Команды для разработки
# poetry install          # установить зависимости
# poetry run pytest       # запуск тестов
# poetry run black .      # форматирование кода
# poetry run ollama serve  # запуск локальной модели
```

**Конфигурация через Pydantic-модели**

Для типобезопасной конфигурации используем Pydantic-модели вместо .env файлов:

`configs/settings.py`

```python
from pydantic import BaseSettings, Field
from typing import Optional

class TradingSettings(BaseSettings):
    """Основные настройки торговой системы"""

    # Ollama cloud модели (работают локально через ollama клиент)
    ollama_base_url: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field("gpt-oss:120b-cloud", env="OLLAMA_MODEL")  # cloud модель через ollama

    # Опционально: платные сервисы (если доступны)
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")  # Опционально для fallback
    langchain_api_key: Optional[str] = Field(None, env="LANGCHAIN_API_KEY")
    langchain_project: str = Field("ats", env="LANGCHAIN_PROJECT")
    langchain_tracing: bool = Field(False, env="LANGCHAIN_TRACING_V2")  # Отключено по умолчанию

    # Базы данных
    chromadb_host: str = Field("localhost", env="CHROMADB_HOST")
    chromadb_port: int = Field(8000, env="CHROMADB_PORT")
    chromadb_persist_dir: str = Field("./chroma_db", env="CHROMADB_PERSIST_DIR")

    memgraph_uri: str = Field("bolt://localhost:7687", env="MEMGRAPH_URI")
    memgraph_user: str = Field("memgraph", env="MEMGRAPH_USER")
    memgraph_password: str = Field("memgraph", env="MEMGRAPH_PASSWORD")

    # Синхронизатор (локальная SQLite БД)
    sync_db_path: str = Field("./data/trades.db", env="SYNC_DB_PATH")
    sync_interval: int = Field(60, env="SYNC_INTERVAL")  # секунды

    # Параметры торговли
    account_risk_per_trade: float = Field(0.01, env="ACCOUNT_RISK_PER_TRADE")
    daily_drawdown_limit: float = Field(0.025, env="DAILY_DRAWDOWN_LIMIT")

    # Стратегии
    scenario_a_instruments: list[str] = Field(
        default=["XAUUSD", "US100", "EURUSD"],
        env="SCENARIO_A_INSTRUMENTS"
    )
    scenario_b_instruments: list[str] = Field(
        default=["XAUUSD", "US100", "EURUSD", "GBPUSD", "USDJPY", "Brent", "Gold", "Meta", "TSLA", "Gas", "Oil"],
        env="SCENARIO_B_INSTRUMENTS"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False

# Глобальный экземпляр настроек
settings = TradingSettings()
```

**Использование в коде:**

```python
from configs.settings import settings
from langchain_ollama import ChatOllama

# Доступ к настройкам через экземпляр
print(settings.ollama_model)  # "llama2:7b"
print(settings.chromadb_host)

# Создание модели через Ollama (автоматически использует cloud)
llm = ChatOllama(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url
)

# Использование в агентах
response = llm.invoke("Проанализируй этот рыночный сигнал...")
```

---

# 3) Базы данных

## 3.1 Архитектура хранения данных: три уровня

Проект использует **многоуровневую архитектуру хранения**, где каждый тип базы данных решает свою задачу:

### Уровень 1: SQLite (локальная реляционная БД)
**Зачем нужна:** Модуль-синхронизатор использует SQLite для **оперативного хранения сырых данных**

```python
# apps/synchronizer/database.py
import sqlite3
from datetime import datetime

class QuoteStorage:
    def __init__(self, db_path="./data/quotes.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quotes (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    price REAL NOT NULL,
                    volume INTEGER,
                    UNIQUE(symbol, timestamp)
                )
            """)

    def store_quote(self, symbol: str, timestamp: int, price: float, volume: int = None):
        """Быстрое сохранение котировки"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO quotes (symbol, timestamp, price, volume)
                VALUES (?, ?, ?, ?)
            """, (symbol, timestamp, price, volume))
```

**Почему SQLite:**
- **Скорость записи:** Оперативное сохранение тысяч котировок в секунду
- **Надежность:** Транзакции, ACID-свойства
- **Простота:** Не требует отдельного сервера, работает как файл
- **Запросы:** Мгновенные SELECT-запросы для агрегации данных

### Уровень 2: ChromaDB (векторное хранилище)
**Зачем нужна:** Семантический поиск и анализ паттернов

```python
# atl/tools/memory.py
def embed_and_store_quote(collection, quote_data):
    text = f"Symbol: {quote_data['symbol']}, Price: {quote_data['price']}, Volume: {quote_data['volume']}"
    embedding = embedding_model.embed_query(text)  # Преобразуем текст в вектор

    collection.add(
        embeddings=[embedding],
        documents=[text],
        metadatas=[quote_data]
    )
```

**Что делает:**
- **Преобразует текстовые описания** рыночных ситуаций в математические векторы
- **Находит похожие сценарии** по смыслу (например, "резкий рост после новости" ≈ "прорыв уровня сопротивления")
- **Создает эмбеддинги** для обучения моделей на исторических данных

### Уровень 3: Memgraph (графовая БД)
**Зачем нужна:** Анализ причинно-следственных связей

```cypher
// Связи между событиями
(:Signal {id: "sig_001"}) -[:TRIGGERED]-> (:Trade {id: "trade_001"})
(:Trade) -[:HAD_EVENT]-> (:RiskEvent {type: "stop_loss", amount: -50})
```

**Что анализирует:**
- **Цепочки событий:** сигнал → сделка → результат → коррекция стратегии
- **Зависимости:** какие комбинации условий приводят к убыткам/прибыли
- **Паттерны:** визуализация связей между инструментами, временем, волатильностью

## Почему не всё в ChromaDB?

Потому что **каждый тип данных требует своего инструмента:**

| Задача | Лучший инструмент | Почему |
|--------|------------------|---------|
| **Сохранить котировку быстро** | SQLite | Мгновенная запись, транзакции |
| **Найти похожий паттерн** | ChromaDB | Семантический поиск по векторам |
| **Проанализировать цепочку событий** | Memgraph | Графовые запросы связей |

**Пример рабочего процесса:**
1. **Синхронизатор** → сохраняет котировки в SQLite (быстро, надежно)
2. **Агент анализа** → берет данные из SQLite, создает эмбеддинги, сохраняет в ChromaDB
3. **Графовый анализ** → строит связи в Memgraph для поиска закономерностей

**Инициализация ChromaDB** (простая установка через pip)

```python
import chromadb

# Инициализация ChromaDB (автоматически создает локальную базу данных)
client = chromadb.PersistentClient(path="./chroma_db")

# Создание коллекций для разных типов данных
collections = {
    "quotes": "Хранение исторических котировок с метаданными",
    "trades": "История сделок с результатами и контекстом",
    "signals": "Сигналы с объяснениями и исходами",
    "playbooks": "Стратегические плейбуки и правила",
    "postmortems": "Анализ ошибок и неудачных сделок"
}

for name, description in collections.items():
    try:
        collection = client.get_or_create_collection(
            name=name,
            metadata={"description": description}
        )
        print(f"Коллекция '{name}' создана или уже существует")
    except Exception as e:
        print(f"Ошибка создания коллекции '{name}': {e}")

# ChromaDB готова к работе - никаких Docker контейнеров!
```

## 3.2 Memgraph (граф: инструмент↔сигнал↔сделка↔риск↔режим)

**Cypher миграция** `storage/memgraph/init_memgraph.cypher`

```cypher
CREATE CONSTRAINT ON (t:Trade) ASSERT t.id IS UNIQUE;
CREATE CONSTRAINT ON (s:Signal) ASSERT s.id IS UNIQUE;
CREATE INDEX ON :Instrument(symbol);
CREATE INDEX ON :Regime(name);
```

**Базовая онтология**

```
(:Instrument {symbol})
  <-[:ON]- (:Signal {id,type,tf,rv,ema_slope})
  -[:TRIGGERED]-> (:Trade {id,side,entry,sl,tp,risk_r,opened_at,closed_at})
  -[:HAD_EVENT]-> (:RiskEvent {type,amount,ts})
  -[:IN_REGIME]-> (:Regime {name,vol,session})
```

> Для графовых запросов и Graph-RAG используем `Memgraph` с Cypher-запросами для анализа причинно-следственных связей между сигналами, сделками и рыночными режимами. ([memgraph.com][4])

---

# 4) Pydantic-схемы (строго типизированный обмен)

`apps/api/schemas.py`

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional, List

class FeatureRequest(BaseModel):
    symbol: str
    timeframe: Literal["M1","M5","M15","H1","D1"]
    lookback: int = 500

class SignalDecision(BaseModel):
    action: Literal["enter","skip","exit","manage"]
    side: Optional[Literal["buy","sell"]] = None
    reason: str
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    size: Optional[float] = None
    attachments: dict = Field(default_factory=dict)

class OrderRequest(BaseModel):
    symbol: str
    side: Literal["buy","sell"]
    volume: float
    entry_type: Literal["market","limit","stop"] = "market"
    sl: Optional[float] = None
    tp: Optional[float] = None
    comment: str = "ATS"
```

> Для строгих JSON-выходов у LLM используем **structured outputs** (`with_structured_output()`/Pydantic), это рекомендуемый способ получать валидный JSON напрямую из модели. ([python.langchain.com][5])

---

# 5) Инструменты (tools) для агентов

`atl/tools/features.py`

```python
import numpy as np
import pandas as pd

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def atr(df: pd.DataFrame, period: int=14) -> pd.Series:
    h_l = df['High'] - df['Low']
    h_c = (df['High'] - df['Close'].shift()).abs()
    l_c = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def rvol(volume: pd.Series, window: int=20) -> pd.Series:
    return volume / volume.rolling(window).mean()

def donchian(df: pd.DataFrame, period: int=20):
    return df['High'].rolling(period).max(), df['Low'].rolling(period).min()

def opening_range(df_m1: pd.DataFrame, minutes: int=30):
    first = df_m1.iloc[:minutes]
    return first['High'].max(), first['Low'].min()
```

`atl/tools/risk.py`

```python
def position_size(balance, risk_pct, sl_points, pip_value):
    risk_money = balance * risk_pct
    # объём в лотах с округлением под брокера
    return max(0.01, round(risk_money/(sl_points*pip_value), 2))
```

`atl/tools/execution.py`

```python
import asyncio, logging
from apps.synchronizer.sync_service import SyncService  # модуль синхронизации
log = logging.getLogger("Exec")

class ExecClient:
    def __init__(self, db_path="./data/trades.db"):
        self.sync_service = SyncService(db_path)

    async def start(self): await self.sync_service.initialize()
    async def stop(self): await self.sync_service.close()

    async def place_order(self, symbol, side, volume, sl=None, tp=None, comment="ATS"):
        # Сохранение сделки в локальную базу данных через синхронизатор
        trade_data = {
            "symbol": symbol,
            "side": side,
            "volume": volume,
            "sl": sl,
            "tp": tp,
            "comment": comment,
            "timestamp": asyncio.get_event_loop().time(),
            "status": "pending"
        }

        # Синхронизируем сделку с базой данных
        result = await self.sync_service.sync_trade(trade_data)
        if not result.get("success"):
            raise RuntimeError(f"trade sync failed: {result}")
        return result["trade_id"]
```

> Модуль синхронизации обеспечивает надежное хранение данных о сделках в локальной базе данных SQLite с последующей синхронизацией в векторную и графовую базы данных.

`atl/tools/memory.py`

```python
import chromadb
from langchain_openai import OpenAIEmbeddings
from memgraph import Memgraph

def make_chromadb_client(persist_dir="./chroma_db"):
    """Создание клиента ChromaDB для векторного поиска"""
    return chromadb.PersistentClient(path=persist_dir)

def make_memgraph_driver(uri, user, pwd):
    """Создание драйвера Memgraph для графовых запросов"""
    return Memgraph(uri=uri, user=user, password=pwd)

def create_chroma_collection(client, collection_name, description=""):
    """Создание коллекции в ChromaDB"""
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"description": description}
    )

def embed_and_store_quote(collection, quote_data, embedding_model=None):
    """Сохранение котировки в векторную базу данных"""
    if embedding_model is None:
        embedding_model = OpenAIEmbeddings()

    # Создаем текстовое представление котировки
    text = f"Symbol: {quote_data['symbol']}, Price: {quote_data['price']}, Volume: {quote_data['volume']}"

    # Получаем эмбеддинг
    embedding = embedding_model.embed_query(text)

    # Сохраняем в ChromaDB
    collection.add(
        embeddings=[embedding],
        documents=[text],
        metadatas=[quote_data],
        ids=[f"quote_{quote_data['symbol']}_{quote_data['timestamp']}"]
    )

def query_similar_quotes(collection, query_text, n_results=5):
    """Поиск похожих котировок"""
    return collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
```

> Интеграции ChromaDB и Memgraph обеспечивают эффективное хранение и поиск данных о котировках и сделках с поддержкой семантического поиска и графового анализа. ChromaDB используется для векторного поиска, а Memgraph для анализа связей между торговыми событиями.

---

# 6) Реактивные агенты (ReAct) и оркестрация (LangGraph)

## 6.1 Промпты (шаблоны)

`atl/prompts/react_intraday.md`

```
Ты — Торговый Агент Intraday. Принимай решения ТОЛЬКО по правилам.
Фильтр тренда: входы в сторону наклона EMA(50) и если цена выше/ниже EMA(200) соответственно.
Триггер: пробой Opening Range/ключевого уровня, НО только при RVOL >= {rvol_th}.
Риск: риск на сделку ≤ {risk_pct} баланса; стоп = max(ATR*k, за уровень).
Выход: частичный фикс при R={partial_r}, трейлинг = EMA(20)/фракталы.
Верни СТРУКТУРИРОВАННЫЙ JSON по схеме SignalDecision.
```

`atl/prompts/react_swing.md`

```
Ты — Торговый Агент Swing/Trend.
Вход: пробой Donchian-{d1}/{d2} или цена над EMA(200) (лонг)/ниже EMA(200) (шорт).
Диверсификация: до N инструментов с vol-targeting.
Риск ≤ {risk_pct}, стоп по ATR*k, выход по обратному пробою/закрытию ниже EMA.
Верни JSON по схеме SignalDecision.
```

## 6.2 Граф состояний (LangGraph)

`atl/graphs/intraday_graph.py`

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from atl.agents.signal_agent_a import plan_enter, plan_manage, should_exit

class IState(TypedDict):
    symbol: str
    tf: str
    features: dict
    decision: Optional[dict]
    position: Optional[dict]

g = StateGraph(IState)
g.add_node("SETUP", plan_enter)    # сбор фич, проверка фильтров, решение "enter/skip"
g.add_node("MANAGE", plan_manage)  # трейлинг/частичный фикс
g.add_node("EXIT", should_exit)    # выход и логирование

g.set_entry_point("SETUP")
g.add_edge("SETUP", "MANAGE", condition=lambda s: s["decision"]["action"]=="enter")
g.add_edge("SETUP", END,         condition=lambda s: s["decision"]["action"]=="skip")
g.add_edge("MANAGE", "EXIT",     condition=lambda s: s["decision"]["action"]=="exit")
g.add_edge("EXIT", END)

intraday_app = g.compile()
```

> LangGraph предназначен для длительных, **состоянием управляемых** агентов; LangChain-агенты «поверх» LangGraph — вы получаете контроль, сохранность, human-in-the-loop при необходимости. ([langchain-ai.github.io][6])

## 6.3 Агенты-исполнители (фрейм LCEL)

`atl/agents/signal_agent_a.py` (упрощённо)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from atl.tools.features import opening_range
from atl.tools.risk import position_size
from atl.tools.execution import ExecClient
from apps.api.schemas import SignalDecision

class DecisionSchema(BaseModel):
    action: str; side: str|None; reason: str
    entry: float|None; sl: float|None; tp: float|None; size: float|None

prompt = ChatPromptTemplate.from_template(open("atl/prompts/react_intraday.md","r",encoding="utf8").read())
model = ChatOllama(model="gpt-oss:120b-cloud")  # cloud модель через Ollama
structured = model.with_structured_output(DecisionSchema)  # строгая схема ответа
# рекомендованный способ структурированных выходов в LangChain :contentReference[oaicite:10]{index=10}

async def plan_enter(state):
    # тут вы собираете фичи (EMA, ATR, RVOL, OpeningRange) и формируете контекст
    # затем вызываете structured LLM для принятия решения
    ...
    decision: DecisionSchema = structured.invoke({...})
    return {"decision": decision.model_dump()}
```

> Для параллельного сбора фич/проверок используйте **LCEL RunnableParallel** — это базовый примитив композиции (рядом с RunnableSequence). ([python.langchain.com][7])

---

# 7) LangServe API (публикация цепочек/графов как REST)

`apps/api/main.py`

```python
from fastapi import FastAPI
from langserve import add_routes
from atl.graphs.intraday_graph import intraday_app
from atl.graphs.swing_graph import swing_app

app = FastAPI(title="ATS API")

# Публикуем как Runnable /invoke, /stream, /batch
add_routes(app, intraday_app, path="/signal/intraday")
add_routes(app, swing_app, path="/signal/swing")

# Пример публикации отдельного раннабла (инструмента)
# add_routes(app, some_runnable, path="/tool/risk-size")
```

> LangServe «нативно» выкладывает **runnables/цепочки** как REST, с автоматической документацией и стримингом; интегрирован с FastAPI/Pydantic. ([python.langchain.com][8])

Запуск:

```bash
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
```

---

# 8) Наблюдаемость и оценка (LangSmith)

* Включите переменные окружения `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT=ats` — получаете детальные трейс-логи, датасеты для регрессионных проверок, мониторинг стоимости токенов/задержек. ([docs.langchain.com][9])

---

# 9) Graph-RAG и Vector-RAG: как используем

* **Vector-RAG (ChromaDB)** — плейбуки/постмортемы/разбор инцидентов; быстрый семантический поиск кейсов для ExplainabilityAgent и для правок правил. Хранение котировок и сделок в векторном виде позволяет анализировать исторические паттерны и принимать более обоснованные решения.
* **Graph-RAG (Memgraph)** — вопросы к причинным цепочкам:
  *«какие комбинации (RVOL≥2 & время=EU session) давали лучший Sharpe на XAUUSD?»*,
  *«какие паттерны приводили к просадкам>2% подряд?»*
 Реализация — прямые Cypher-запросы к Memgraph для анализа связей между сигналами, сделками и рыночными режимами.

---

# 10) Переход A → B (логика переключения)

* Включите `SignalAgent-B` при депозите > **$3–5k** или после **N** недель стабильной статистики; веса портфеля — через **volatility targeting** (держим целевую волу).
* Оркестрацией занимается GovernanceAgent: включает контуры/меняет пороги при устойчивых изменениях режима.

---

# 11) Docker-compose для инфраструктуры

`docker/docker-compose.yml`

```yaml
services:
  chromadb:
    image: chromadb/chroma:latest
    ports: ["8000:8000"]
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - CHROMA_SERVER_HOST=0.0.0.0
      - CHROMA_SERVER_HTTP_PORT=8000

  memgraph:
    image: memgraph/memgraph:latest
    ports: ["7687:7687"]
    environment:
      - MEMGRAPH_CONFIG="--bolt-port=7687"
    volumes:
      - memgraph_data:/var/lib/memgraph

  api:
    build: ../
    command: uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
    env_file: ../.env
    depends_on: [chromadb, memgraph]
    ports: ["8000:8000"]

volumes:
  chroma_data:
  memgraph_data:
```

---

# 12) Тесты и бэктесты

* **Unit**: парсинг фичей (EMA/ATR/RVOL), расчёт риск-сайзинга, выходы LCEL-цепочек (валидный JSON по Pydantic-схемам).
* **Integration**: «сухой» прогон графов LangGraph с фиктивным ExecAgent; проверка идемпотентности.
* **Backtests**: по A — XAUUSD/US100/EURUSD (2–3 года минуток), по B — мульти-символ D1 (5–10 лет).
* **Метрики входа в прод**: Sharpe > 1, MAR > 0.5, maxDD < 20%, соблюдение лимитов риска.
* **LangSmith Datasets**: фиксируем типовые рыночные сценарии → regression-оценка после изменений кода. ([docs.langchain.com][9])

---

# 13) Каркас кода агентов (минимально рабочий)

`atl/agents/exec_agent.py`

```python
import asyncio
from atl.tools.execution import ExecClient

class ExecutionAgent:
    def __init__(self, host="127.0.0.1", port=5000):
        self.cli = ExecClient(host, port)
    async def start(self): await self.cli.start()
    async def stop(self): await self.cli.stop()

    async def market_order(self, symbol, side, volume, sl=None, tp=None, comment="ATS"):
        return await self.cli.place_order(symbol, side, volume, sl, tp, comment)
```

`atl/agents/risk_agent.py`

```python
class RiskAgent:
    def __init__(self, balance_fn, daily_dd_limit=0.025, per_trade=0.01):
        self.balance_fn = balance_fn
        self.dd = 0; self.dd_limit = daily_dd_limit; self.per_trade=per_trade

    def can_trade(self) -> bool: return self.dd < self.dd_limit
    def alloc(self, sl_points, pip_value): 
        return max(0.01, round(self.balance_fn()*self.per_trade/(sl_points*pip_value),2))
```

`atl/agents/signal_agent_b.py` (Donchian/MA-тайминг для D1) — аналогично A, но со своими фичами.

---

# 14) Визуализация в MT5

Для наглядности SignalAgent-ы рисуют на графике уровни входа/стопа/тейков и подписи: используем ваши команды (см. `main_app.py` демо). Пример:

```python
await bridge.call({"cmd":"chart.ensure","symbol":symbol,"timeframe":tf})
await bridge.call({"cmd":"object.create","chart_id":chart_id,"type":"OBJ_HLINE","name":f"ENTRY_{id}","p1":entry})
await bridge.call({"cmd":"object.set","chart_id":chart_id,"name":f"ENTRY_{id}","prop":"color","value":"blue"})
await bridge.call({"cmd":"chart.redraw","chart_id":chart_id})
```



---

# 15) Шаблон CI (минимум)

* **lint / typecheck** (ruff+mypy), **tests**, build Docker, upload coverage.
* Smoke-тест API (`/signal/intraday:invoke`) и ретриверов Vector/Graph.

---

# 16) Чек-лист задач (по приоритету)

**Этап 1 — MVP (A-контур)**

1. [ ] Реализовать модуль-синхронизатор: модели данных, база SQLite, интеграция с ChromaDB/Memgraph.
2. [ ] Реализовать фичи: EMA(50/200), ATR(14), RVOL(20), OpeningRange(30).
3. [ ] Инструменты: риск-сайзинг, ExecAgent (интеграция с синхронизатором).
4. [ ] ReAct-агент Intraday + LangGraph-граф состояний; structured outputs (Pydantic). ([python.langchain.com][5])
5. [ ] LangServe: `/signal/intraday` (invoke/stream), `/trade/order`. ([python.langchain.com][8])
6. [ ] Vector-RAG (ChromaDB): коллекции котировок, сделок, сигналов, плейбуков.
7. [ ] Graph-RAG (Memgraph): онтология сигнал→сделка→риск→режим.
8. [ ] LangSmith: трассировка/оценка. ([docs.langchain.com][9])
9. [ ] Бэктест 24–36 мес. XAUUSD/US100/EURUSD; валидация порогов RVOL/ATR.

**Этап 2 — v1 (переход к B-контуру)**

1. [ ] Donchian-20/55 + MA-тайминг(100–200) на D1, PortfolioAllocator с vol-target.
2. [ ] Graph-RAG (Memgraph): онтология сигнал→сделка→риск→режим, Q&A-цепочка.
3. [ ] Политики GovernanceAgent: включение/отключение контуров/порогов.
4. [ ] Docker-compose: chromadb+memgraph+api.

**Этап 3 — v2**

1. [ ] Long-term memory (LangGraph), ExplainabilityAgent (отчёты). ([langchain-ai.github.io][6])
2. [ ] Подготовка витрины/копитрейдинга, SLO на задержки/стоимость.

---

# 17) Почему именно так (ключевые факты из доков)

* **LangGraph** — «stateful orchestration» для долгоживущих, управляемых агентов; агенты LangChain **строятся поверх LangGraph** (надёжность, долговечность, persistence, human-in-the-loop). ([langchain-ai.github.io][2])
* **LangServe** — быстрый способ «как есть» опубликовать **Runnable/Chain** как REST (FastAPI+Pydantic, стриминг ответов, автодоки). ([python.langchain.com][8])
* **Structured outputs** — рекомендовано использовать `.with_structured_output()`/Pydantic для **надёжного JSON**; связывать **tools → потом schema** (важный порядок). ([python.langchain.com][5])
* **ChromaDB** — эффективная векторная база данных для хранения и поиска котировок, сделок и сигналов с семантическим поиском.
* **Memgraph** — высокопроизводительная графовая база данных для анализа связей между торговыми событиями и рыночными режимами.

---

## Что от вас нужно прямо сейчас

1. Подтвердить пороги по умолчанию (RVOL=1.8, ATR-k=1.5/3.0, риск A=1%, дневной лимит=2.5%, переход к B при $3–5k).
2. Подтвердить настройки синхронизатора (источники данных MT5, частота синхронизации, параметры хранения).
3. Подтвердить настройки для cloud модели Ollama (gpt-oss:120b-cloud как основная модель).
4. Определить инструменты для мониторинга (CLI интерфейс или Streamlit веб-панель).

Если ок — я дополнительно приложу **готовые файлы** `settings.yaml`, `Dockerfile.api`, `docker-compose.yml`, а также **инициализацию** ChromaDB/Memgraph и скрипты настройки модуля-синхронизатора — они соответствуют указанным выше схемам и интеграциям.

[1]: https://python.langchain.com/docs/tutorials/?utm_source=chatgpt.com "Tutorials | 🦜️🔗 LangChain"
[2]: https://langchain-ai.github.io/langgraph/?utm_source=chatgpt.com "LangGraph - GitHub Pages"
[3]: https://docs.trychroma.com/ "ChromaDB Documentation"
[4]: https://memgraph.com/docs/ "Memgraph Documentation"
[5]: https://python.langchain.com/docs/concepts/structured_outputs/?utm_source=chatgpt.com "Structured outputs | 🦜️🔗 LangChain"
[6]: https://langchain-ai.github.io/langgraph/concepts/why-langgraph/?utm_source=chatgpt.com "Learn LangGraph basics - Overview"
[7]: https://python.langchain.com/docs/concepts/lcel/?utm_source=chatgpt.com "LangChain Expression Language (LCEL)"
[8]: https://python.langchain.com/docs/langserve/?utm_source=chatgpt.com "🦜️🏓 LangServe | 🦜️🔗 LangChain"
[9]: https://docs.langchain.com/langsmith?utm_source=chatgpt.com "Get started with LangSmith - Docs by LangChain"
[10]: https://python.langchain.com/docs/tutorials/graph/?utm_source=chatgpt.com "Build a Question Answering application over a Graph ..."


Medical References:
1. None — DOI: file-C5eCTQgeQeAtzGMdMX96XL
2. None — DOI: file-KDBfXzqtJQ7S7pvVfLMj6m