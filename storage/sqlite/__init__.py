"""Совместимость импортов для модулей SQLite хранилища."""

from src.database.storage import (
    QuoteStorage,
    TradeStorage,
    SignalStorage,
    TechnicalIndicatorsStorage,
    initialize_storages,
    quote_storage,
    trade_storage,
    signal_storage,
    indicators_storage,
)

__all__ = [
    "QuoteStorage",
    "TradeStorage",
    "SignalStorage",
    "TechnicalIndicatorsStorage",
    "initialize_storages",
    "quote_storage",
    "trade_storage",
    "signal_storage",
    "indicators_storage",
]
