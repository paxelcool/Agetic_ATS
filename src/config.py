"""
Конфигурация торговой системы ATS.

Настройки через Pydantic-модели для типобезопасной конфигурации.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class TradingSettings(BaseSettings):
    """
    Основные настройки торговой системы.
    """

    # Ollama cloud модели (работают локально через ollama клиент)
    ollama_base_url: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(
        "gpt-oss:120b-cloud", env="OLLAMA_MODEL"
    )  # cloud модель через ollama

    # Опционально: платные сервисы (если доступны)
    openai_api_key: Optional[str] = Field(
        None, env="OPENAI_API_KEY"
    )  # Опционально для fallback
    langchain_api_key: Optional[str] = Field(None, env="LANGCHAIN_API_KEY")
    langchain_project: str = Field("ats", env="LANGCHAIN_PROJECT")
    langchain_tracing: bool = Field(
        False, env="LANGCHAIN_TRACING_V2"
    )  # Отключено по умолчанию

    # Базы данных
    chromadb_host: str = Field("localhost", env="CHROMADB_HOST")
    chromadb_port: int = Field(8000, env="CHROMADB_PORT")
    chromadb_persist_dir: str = Field("./storage/chroma_db", env="CHROMADB_PERSIST_DIR")

    memgraph_uri: str = Field("bolt://localhost:7687", env="MEMGRAPH_URI")
    memgraph_user: str = Field("memgraph", env="MEMGRAPH_USER")
    memgraph_password: str = Field("memgraph", env="MEMGRAPH_PASSWORD")

    # Синхронизатор (локальная SQLite БД)
    sync_db_path: str = Field("./storage/sqlite/trades.db", env="SYNC_DB_PATH")
    sync_interval: int = Field(60, env="SYNC_INTERVAL")  # секунды

    # Параметры торговли
    account_risk_per_trade: float = Field(0.01, env="ACCOUNT_RISK_PER_TRADE")
    daily_drawdown_limit: float = Field(0.025, env="DAILY_DRAWDOWN_LIMIT")

    # Стратегии
    scenario_a_instruments: list[str] = Field(
        default=["XAUUSD", "US100", "EURUSD"], env="SCENARIO_A_INSTRUMENTS"
    )
    scenario_b_instruments: list[str] = Field(
        default=[
            "XAUUSD",
            "US100",
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "Brent",
            "Gold",
            "Meta",
            "TSLA",
            "Gas",
            "Oil",
        ],
        env="SCENARIO_B_INSTRUMENTS",
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


# Глобальный экземпляр настроек
settings = TradingSettings()
