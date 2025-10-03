#!/usr/bin/env python3
"""
Модульные тесты для моделей данных.

Тестирует валидацию и корректность Pydantic-моделей.
"""

import pytest
from datetime import datetime, timedelta

from src.database.models import (
    Quote,
    Trade,
    Signal,
    TechnicalIndicators,
    FeatureRequest,
    SignalDecision,
    OrderRequest,
    Timeframe,
    OrderSide,
    SignalAction,
    MarketRegime,
)


class TestQuote:
    """Тесты модели Quote."""

    def test_valid_quote_creation(self):
        """Тест создания корректной котировки."""
        quote = Quote(
            symbol="EURUSD",
            timestamp=1640995200,  # 2022-01-01 00:00:00 UTC
            bid=1.1234,
            ask=1.1236,
            volume=1000000
        )

        assert quote.symbol == "EURUSD"
        assert quote.bid == 1.1234
        assert quote.ask == 1.1236
        assert abs(quote.spread - 0.0002) < 1e-10  # Используем приблизительное сравнение для float
        assert quote.mid_price == 1.1235
        assert quote.volume == 1000000

    def test_quote_validation_bid_ask_order(self):
        """Тест валидации порядка цен bid/ask."""
        with pytest.raises(ValueError, match="ask цена должна быть >= bid цены"):
            Quote(
                symbol="EURUSD",
                timestamp=1640995200,
                bid=1.1236,  # bid > ask - ошибка
                ask=1.1234
            )

    def test_quote_spread_calculation(self):
        """Тест автоматического расчета спреда."""
        quote = Quote(
            symbol="EURUSD",
            timestamp=1640995200,
            bid=1.1234,
            ask=1.1236
        )

        assert abs(quote.spread - 0.0002) < 1e-10  # Используем приблизительное сравнение для float
        assert quote.mid_price == 1.1235


class TestTrade:
    """Тесты модели Trade."""

    def test_valid_trade_creation(self):
        """Тест создания корректной сделки."""
        trade = Trade(
            symbol="EURUSD",
            side=OrderSide.BUY,
            entry_price=1.1234,
            quantity=0.1,
            stop_loss=1.1200,
            take_profit=1.1300
        )

        assert trade.symbol == "EURUSD"
        assert trade.side == OrderSide.BUY
        assert trade.entry_price == 1.1234
        assert trade.quantity == 0.1
        assert trade.stop_loss == 1.1200
        assert trade.take_profit == 1.1300

    def test_trade_tp_validation_buy(self):
        """Тест валидации TP для BUY сделок."""
        with pytest.raises(ValueError, match="Для BUY сделок TP должен быть > цены входа"):
            Trade(
                symbol="EURUSD",
                side=OrderSide.BUY,
                entry_price=1.1234,
                take_profit=1.1200  # TP < entry для BUY - ошибка
            )

    def test_trade_tp_validation_sell(self):
        """Тест валидации TP для SELL сделок."""
        with pytest.raises(ValueError, match="Для SELL сделок TP должен быть < цены входа"):
            Trade(
                symbol="EURUSD",
                side=OrderSide.SELL,
                entry_price=1.1234,
                take_profit=1.1300  # TP > entry для SELL - ошибка
            )


class TestSignal:
    """Тесты модели Signal."""

    def test_valid_signal_creation(self):
        """Тест создания корректного сигнала."""
        signal = Signal(
            symbol="EURUSD",
            timeframe=Timeframe.M5,
            action=SignalAction.ENTER,
            side=OrderSide.BUY,
            confidence=0.75,
            entry_price=1.1234,
            stop_loss=1.1200,
            take_profit=1.1300,
            quantity=0.1,
            risk_amount=100.0,
            reason="Технический пробой уровня сопротивления с подтверждением объема"
        )

        assert signal.symbol == "EURUSD"
        assert signal.timeframe == Timeframe.M5
        assert signal.action == SignalAction.ENTER
        assert signal.side == OrderSide.BUY
        assert signal.confidence == 0.75
        assert signal.reason == "Технический пробой уровня сопротивления с подтверждением объема"

    def test_signal_confidence_bounds(self):
        """Тест границ уверенности сигнала."""
        with pytest.raises(ValueError):
            Signal(
                symbol="EURUSD",
                timeframe=Timeframe.M5,
                action=SignalAction.ENTER,
                confidence=1.5  # > 1 - ошибка
            )

        with pytest.raises(ValueError):
            Signal(
                symbol="EURUSD",
                timeframe=Timeframe.M5,
                action=SignalAction.ENTER,
                confidence=-0.1  # < 0 - ошибка
            )


class TestTechnicalIndicators:
    """Тесты модели TechnicalIndicators."""

    def test_valid_indicators_creation(self):
        """Тест создания индикаторов."""
        indicators = TechnicalIndicators(
            symbol="EURUSD",
            timeframe=Timeframe.H1,
            timestamp=1640995200,
            ema_20=1.1234,
            ema_50=1.1250,
            rsi=65.5,
            atr=0.0012,
            volume=1500000,
            rvol=1.2
        )

        assert indicators.symbol == "EURUSD"
        assert indicators.timeframe == Timeframe.H1
        assert indicators.ema_20 == 1.1234
        assert indicators.ema_50 == 1.1250
        assert indicators.rsi == 65.5
        assert indicators.atr == 0.0012
        assert indicators.volume == 1500000
        assert indicators.rvol == 1.2


class TestEnums:
    """Тесты перечислений."""

    def test_timeframe_enum(self):
        """Тест перечисления таймфреймов."""
        assert Timeframe.M1 == "M1"
        assert Timeframe.H1 == "H1"
        assert Timeframe.D1 == "D1"

    def test_order_side_enum(self):
        """Тест перечисления направлений ордеров."""
        assert OrderSide.BUY == "buy"
        assert OrderSide.SELL == "sell"

    def test_signal_action_enum(self):
        """Тест перечисления действий сигналов."""
        assert SignalAction.ENTER == "enter"
        assert SignalAction.EXIT == "exit"
        assert SignalAction.SKIP == "skip"

    def test_market_regime_enum(self):
        """Тест перечисления рыночных режимов."""
        assert MarketRegime.TRENDING == "trending"
        assert MarketRegime.RANGING == "ranging"
        assert MarketRegime.VOLATILE == "volatile"


class TestFeatureRequest:
    """Тесты модели FeatureRequest."""

    def test_valid_request_creation(self):
        """Тест создания корректного запроса."""
        request = FeatureRequest(
            symbol="EURUSD",
            timeframe=Timeframe.M5,
            lookback=100
        )

        assert request.symbol == "EURUSD"
        assert request.timeframe == Timeframe.M5
        assert request.lookback == 100
        assert "ema_20" in request.indicators


class TestSignalDecision:
    """Тесты модели SignalDecision."""

    def test_valid_decision_creation(self):
        """Тест создания корректного решения."""
        decision = SignalDecision(
            action=SignalAction.ENTER,
            side=OrderSide.BUY,
            reason="Сильный бычий сигнал на пробое уровня",
            entry=1.1234,
            stop_loss=1.1200,
            take_profit=1.1300,
            quantity=0.1,
            confidence=0.8
        )

        assert decision.action == SignalAction.ENTER
        assert decision.side == OrderSide.BUY
        assert decision.confidence == 0.8
        assert decision.entry == 1.1234


class TestOrderRequest:
    """Тесты модели OrderRequest."""

    def test_valid_order_request(self):
        """Тест создания корректного запроса ордера."""
        order = OrderRequest(
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=0.1,
            order_type="market",
            stop_loss=1.1200,
            take_profit=1.1300,
            comment="ATS Signal #123"
        )

        assert order.symbol == "EURUSD"
        assert order.side == OrderSide.BUY
        assert order.quantity == 0.1
        assert order.comment == "ATS Signal #123"

    def test_limit_order_requires_price(self):
        """Тест требования цены для лимитных ордеров."""
        with pytest.raises(ValueError, match="Цена обязательна для лимитных"):
            from src.database.models import OrderType
            OrderRequest(
                symbol="EURUSD",
                side=OrderSide.BUY,
                quantity=0.1,
                order_type=OrderType.LIMIT
                # price отсутствует - ошибка
            )


if __name__ == "__main__":
    pytest.main([__file__])
