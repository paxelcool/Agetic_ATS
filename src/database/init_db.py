"""
Инициализация базы данных SQLite для ATS.

Создает таблицы для хранения котировок, сделок, сигналов и индикаторов.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Менеджер базы данных SQLite для ATS.

    Обеспечивает создание таблиц и базовые операции с БД.
    """

    def __init__(self, db_path: str = "./data/trades.db"):
        """
        Инициализация менеджера базы данных.

        Args:
            db_path: Путь к файлу базы данных
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def initialize_database(self) -> bool:
        """
        Инициализирует базу данных и создает таблицы.

        Returns:
            bool: True если инициализация успешна, False иначе
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                self._create_tables(conn)
                logger.info(f"База данных инициализирована: {self.db_path}")
                return True

        except Exception as e:
            logger.error(f"Ошибка инициализации базы данных: {e}")
            return False

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """
        Создает таблицы в базе данных.

        Args:
            conn: Соединение с базой данных
        """
        # Таблица котировок
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS quotes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                bid REAL NOT NULL,
                ask REAL NOT NULL,
                volume INTEGER,
                spread REAL GENERATED ALWAYS AS (ask - bid) VIRTUAL,
                mid_price REAL GENERATED ALWAYS AS ((ask + bid) / 2) VIRTUAL,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                UNIQUE(symbol, timestamp)
            )
        """
        )

        # Таблица сделок
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'open', 'closed', 'cancelled')),
                opened_at INTEGER,
                closed_at INTEGER,
                pnl REAL,
                pnl_points REAL,
                commission REAL DEFAULT 0,
                magic_number INTEGER,
                comment TEXT,
                signal_id TEXT,
                risk_amount REAL,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                updated_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """
        )

        # Таблица сигналов
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                action TEXT NOT NULL CHECK (action IN ('enter', 'exit', 'skip', 'manage')),
                side TEXT CHECK (side IN ('buy', 'sell')),
                confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                quantity REAL,
                risk_amount REAL,
                reason TEXT NOT NULL,
                indicators TEXT, -- JSON строка с индикаторами
                market_regime TEXT,
                related_signals TEXT, -- JSON массив связанных сигналов
                tags TEXT, -- JSON массив тегов
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                expires_at INTEGER
            )
        """
        )

        # Таблица технических индикаторов
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                ema_20 REAL,
                ema_50 REAL,
                ema_200 REAL,
                sma_20 REAL,
                rsi REAL,
                stoch_k REAL,
                stoch_d REAL,
                atr REAL,
                atr_percent REAL,
                volume INTEGER,
                rvol REAL,
                donchian_upper REAL,
                donchian_lower REAL,
                opening_range_high REAL,
                opening_range_low REAL,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                UNIQUE(symbol, timeframe, timestamp)
            )
        """
        )

        # Таблица настроек синхронизации
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT UNIQUE NOT NULL,
                last_sync_timestamp INTEGER,
                last_record_id INTEGER,
                status TEXT DEFAULT 'active' CHECK (status IN ('active', 'paused', 'error')),
                updated_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """
        )

        # Таблица логов синхронизации
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                operation TEXT NOT NULL CHECK (operation IN ('insert', 'update', 'delete', 'sync')),
                record_count INTEGER DEFAULT 0,
                timestamp INTEGER DEFAULT (strftime('%s', 'now')),
                details TEXT
            )
        """
        )

        # Индексы для оптимизации запросов
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_quotes_symbol_timestamp
            ON quotes(symbol, timestamp)
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_quotes_timestamp
            ON quotes(timestamp)
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_trades_symbol
            ON trades(symbol)
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_trades_status
            ON trades(status)
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_trades_opened_at
            ON trades(opened_at)
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_signals_symbol
            ON signals(symbol)
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_signals_created_at
            ON signals(created_at)
        """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_timeframe
            ON technical_indicators(symbol, timeframe, timestamp)
        """
        )

        # Включаем foreign keys
        conn.execute("PRAGMA foreign_keys = ON")

        logger.info("Все таблицы и индексы созданы успешно")

    def get_connection(self) -> sqlite3.Connection:
        """
        Получает соединение с базой данных.

        Returns:
            sqlite3.Connection: Соединение с БД
        """
        return sqlite3.connect(self.db_path)

    def check_database_health(self) -> Dict[str, Any]:
        """
        Проверяет здоровье базы данных.

        Returns:
            Dict[str, Any]: Результаты проверки
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Проверяем таблицы
                cursor.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """
                )
                tables = [row[0] for row in cursor.fetchall()]

                # Получаем статистику по таблицам
                stats = {}
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    stats[table] = count

                # Проверяем последнюю синхронизацию
                cursor.execute(
                    """
                    SELECT table_name, last_sync_timestamp, status
                    FROM sync_state
                """
                )
                sync_info = cursor.fetchall()

                return {
                    "status": "healthy",
                    "tables": tables,
                    "record_counts": stats,
                    "sync_state": {
                        table: {"last_sync": timestamp, "status": status}
                        for table, timestamp, status in sync_info
                    },
                }

        except Exception as e:
            logger.error(f"Ошибка проверки здоровья БД: {e}")
            return {"status": "error", "error": str(e)}

    def reset_database(self) -> bool:
        """
        Сбрасывает базу данных (удаляет все данные).

        Returns:
            bool: True если сброс успешен, False иначе
        """
        try:
            # Удаляем файл базы данных
            if self.db_path.exists():
                self.db_path.unlink()

            # Создаем заново
            return self.initialize_database()

        except Exception as e:
            logger.error(f"Ошибка сброса базы данных: {e}")
            return False


def create_database_manager(db_path: str = "./data/trades.db") -> DatabaseManager:
    """
    Создает экземпляр менеджера базы данных.

    Args:
        db_path: Путь к файлу базы данных

    Returns:
        DatabaseManager: Экземпляр менеджера
    """
    return DatabaseManager(db_path)


# Глобальный экземпляр для использования в приложении
db_manager = create_database_manager()
