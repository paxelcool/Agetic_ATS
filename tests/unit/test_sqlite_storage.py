#!/usr/bin/env python3
"""
Модульные тесты для SQLite хранилища.

Тестирует операции с базой данных SQLite.
"""

import os
import tempfile
from datetime import datetime

import pytest

from src.database.models import (
    MarketRegime,
    OrderSide,
    Quote,
    Signal,
    SignalAction,
    TechnicalIndicators,
    Timeframe,
    Trade,
)
from src.database.init_db import create_database_manager


@pytest.fixture
def temp_db():
    """Создает временную базу данных для тестов."""
    import os
    import shutil

    temp_dir = tempfile.mkdtemp(prefix="sqlite_test_")
    db_path = os.path.join(temp_dir, "test.db")

    yield db_path

    # Очистка после теста
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        # Игнорируем ошибки очистки в Windows
        pass


@pytest.fixture
def db_manager(temp_db):
    """Создает менеджер базы данных для тестов."""
    return create_database_manager(temp_db)


@pytest.fixture
def initialized_db(db_manager):
    """Инициализирует базу данных и возвращает менеджер."""
    assert db_manager.initialize_database()
    return db_manager


@pytest.fixture
def quote_storage(initialized_db):
    """Создает хранилище котировок."""
    from storage.sqlite.storage import QuoteStorage

    return QuoteStorage(initialized_db)


@pytest.fixture
def trade_storage(initialized_db):
    """Создает хранилище сделок."""
    from storage.sqlite.storage import TradeStorage

    return TradeStorage(initialized_db)


@pytest.fixture
def signal_storage(initialized_db):
    """Создает хранилище сигналов."""
    from storage.sqlite.storage import SignalStorage

    return SignalStorage(initialized_db)


@pytest.fixture
def indicators_storage(initialized_db):
    """Создает хранилище индикаторов."""
    from storage.sqlite.storage import TechnicalIndicatorsStorage

    return TechnicalIndicatorsStorage(initialized_db)


class TestDatabaseManager:
    """Тесты менеджера базы данных."""

    def test_database_creation(self, db_manager):
        """Тест создания базы данных."""
        assert db_manager.initialize_database()

        # Проверяем, что файл создан
        assert os.path.exists(db_manager.db_path)

    def test_database_health_check(self, initialized_db):
        """Тест проверки здоровья базы данных."""
        health = initialized_db.check_database_health()

        assert health["status"] == "healthy"
        assert "tables" in health
        assert "record_counts" in health

        # Должны быть созданы таблицы
        expected_tables = {
            "quotes",
            "trades",
            "signals",
            "technical_indicators",
            "sync_state",
            "sync_logs",
        }
        assert set(health["tables"]) == expected_tables

    def test_database_reset(self, db_manager):
        """Тест сброса базы данных."""
        # Создаем базу данных
        assert db_manager.initialize_database()

        # Сбрасываем
        assert db_manager.reset_database()

        # Проверяем, что данные очищены
        health = db_manager.check_database_health()
        for table, count in health["record_counts"].items():
            assert count == 0


class TestQuoteStorage:
    """Тесты хранилища котировок."""

    def test_store_and_retrieve_quote(self, quote_storage):
        """Тест сохранения и получения котировки."""
        quote = Quote(
            symbol="EURUSD",
            timestamp=1640995200,
            bid=1.1234,
            ask=1.1236,
            volume=1000000,
        )

        # Сохраняем
        assert quote_storage.store_quote(quote)

        # Получаем
        quotes = quote_storage.get_quotes("EURUSD", limit=1)
        assert len(quotes) == 1

        retrieved_quote = quotes[0]
        assert retrieved_quote.symbol == quote.symbol
        assert retrieved_quote.bid == quote.bid
        assert retrieved_quote.ask == quote.ask
        assert retrieved_quote.volume == quote.volume

    def test_get_latest_quote(self, quote_storage):
        """Тест получения последней котировки."""
        # Сохраняем несколько котировок
        quotes = [
            Quote(symbol="EURUSD", timestamp=1640995200, bid=1.1234, ask=1.1236),
            Quote(symbol="EURUSD", timestamp=1640995260, bid=1.1235, ask=1.1237),
            Quote(symbol="EURUSD", timestamp=1640995320, bid=1.1236, ask=1.1238),
        ]

        for quote in quotes:
            assert quote_storage.store_quote(quote)

        # Получаем последнюю
        latest = quote_storage.get_latest_quote("EURUSD")
        assert latest is not None
        assert latest.timestamp == 1640995320
        assert latest.bid == 1.1236

    def test_get_quotes_with_filters(self, quote_storage):
        """Тест получения котировок с фильтрами."""
        # Сохраняем котировки разных символов
        quotes = [
            Quote(symbol="EURUSD", timestamp=1640995200, bid=1.1234, ask=1.1236),
            Quote(symbol="GBPUSD", timestamp=1640995200, bid=1.2345, ask=1.2347),
            Quote(symbol="EURUSD", timestamp=1640995260, bid=1.1235, ask=1.1237),
        ]

        for quote in quotes:
            assert quote_storage.store_quote(quote)

        # Тестируем фильтр по символу
        eur_quotes = quote_storage.get_quotes("EURUSD")
        assert len(eur_quotes) == 2

        # Тестируем лимит
        limited_quotes = quote_storage.get_quotes("EURUSD", limit=1)
        assert len(limited_quotes) == 1


class TestTradeStorage:
    """Тесты хранилища сделок."""

    def test_store_and_retrieve_trade(self, trade_storage):
        """Тест сохранения и получения сделки."""
        trade = Trade(
            symbol="EURUSD",
            side=OrderSide.BUY,
            entry_price=1.1234,
            quantity=0.1,
            stop_loss=1.1200,
            take_profit=1.1300,
            status="open",
            opened_at=datetime.utcnow(),
            comment="Test trade",
        )

        # Сохраняем
        assert trade_storage.store_trade(trade)

        # Получаем
        trades = trade_storage.get_trades("EURUSD", limit=1)
        assert len(trades) == 1

        retrieved_trade = trades[0]
        assert retrieved_trade.symbol == trade.symbol
        assert retrieved_trade.side == trade.side
        assert retrieved_trade.entry_price == trade.entry_price
        assert retrieved_trade.status == trade.status

    def test_trade_filters(self, trade_storage):
        """Тест фильтров для сделок."""
        # Создаем сделки разных статусов
        trades = [
            Trade(
                symbol="EURUSD",
                side=OrderSide.BUY,
                entry_price=1.1234,
                quantity=0.1,
                status="open",
            ),
            Trade(
                symbol="EURUSD",
                side=OrderSide.SELL,
                entry_price=1.1235,
                quantity=0.1,
                status="closed",
            ),
            Trade(
                symbol="GBPUSD",
                side=OrderSide.BUY,
                entry_price=1.2345,
                quantity=0.1,
                status="open",
            ),
        ]

        for trade in trades:
            assert trade_storage.store_trade(trade)

        # Тестируем фильтр по статусу
        open_trades = trade_storage.get_trades(status="open")
        assert len(open_trades) >= 1  # Должна быть как минимум одна открытая сделка

        # Тестируем фильтр по символу
        eur_trades = trade_storage.get_trades("EURUSD")
        assert len(eur_trades) == 2


class TestSignalStorage:
    """Тесты хранилища сигналов."""

    def test_store_and_retrieve_signal(self, signal_storage):
        """Тест сохранения и получения сигнала."""
        signal = Signal(
            symbol="EURUSD",
            timeframe=Timeframe.M5,
            action=SignalAction.ENTER,
            side=OrderSide.BUY,
            confidence=0.75,
            entry_price=1.1234,
            stop_loss=1.1200,
            take_profit=1.1300,
            reason="Технический пробой уровня сопротивления с подтверждением объема",
            indicators={"ema_20": 1.1230, "rsi": 65},
            market_regime=MarketRegime.TRENDING,
            tags=["breakout", "momentum"],
        )

        # Сохраняем
        assert signal_storage.store_signal(signal)

        # Получаем
        signals = signal_storage.get_signals("EURUSD", limit=1)
        assert len(signals) == 1

        retrieved_signal = signals[0]
        assert retrieved_signal.symbol == signal.symbol
        assert retrieved_signal.action == signal.action
        assert retrieved_signal.side == signal.side
        assert retrieved_signal.confidence == signal.confidence
        assert retrieved_signal.reason == signal.reason
        assert retrieved_signal.indicators == signal.indicators
        assert retrieved_signal.market_regime == signal.market_regime
        assert retrieved_signal.tags == signal.tags


class TestTechnicalIndicatorsStorage:
    """Тесты хранилища индикаторов."""

    def test_store_and_retrieve_indicators(self, indicators_storage):
        """Тест сохранения и получения индикаторов."""
        indicators = TechnicalIndicators(
            symbol="EURUSD",
            timeframe=Timeframe.H1,
            timestamp=1640995200,
            ema_20=1.1234,
            ema_50=1.1250,
            rsi=65.5,
            atr=0.0012,
            volume=1500000,
            rvol=1.2,
        )

        # Сохраняем
        assert indicators_storage.store_indicators(indicators)

        # Получаем
        retrieved_indicators = indicators_storage.get_indicators(
            "EURUSD", Timeframe.H1, limit=1
        )
        assert len(retrieved_indicators) == 1

        retrieved = retrieved_indicators[0]
        assert retrieved.symbol == indicators.symbol
        assert retrieved.timeframe == indicators.timeframe
        assert retrieved.ema_20 == indicators.ema_20
        assert retrieved.ema_50 == indicators.ema_50
        assert retrieved.rsi == indicators.rsi
        assert retrieved.atr == indicators.atr
        assert retrieved.volume == indicators.volume
        assert retrieved.rvol == indicators.rvol


if __name__ == "__main__":
    pytest.main([__file__])
