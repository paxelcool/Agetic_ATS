"""
Операции хранения данных в SQLite для ATS.

Предоставляет функции для работы с котировками, сделками и сигналами.
"""

import json
import logging
from datetime import datetime
from typing import List, Optional

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

logger = logging.getLogger(__name__)


class QuoteStorage:
    """
    Хранилище котировок в SQLite.
    """

    def __init__(self, db_manager):
        """
        Инициализация хранилища котировок.

        Args:
            db_manager: Экземпляр DatabaseManager
        """
        self.db_manager = db_manager

    def store_quote(self, quote: Quote) -> bool:
        """
        Сохраняет котировку в базу данных.

        Args:
            quote: Объект котировки

        Returns:
            bool: True если сохранение успешно, False иначе
        """
        try:
            with self.db_manager.get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO quotes (symbol, timestamp, bid, ask, volume)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (quote.symbol, quote.timestamp, quote.bid, quote.ask, quote.volume),
                )

                # Логируем операцию
                self._log_sync_operation("quotes", "insert", 1)

                return True

        except Exception as e:
            logger.error(f"Ошибка сохранения котировки: {e}")
            return False

    def get_quotes(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Quote]:
        """
        Получает котировки из базы данных.

        Args:
            symbol: Символ инструмента
            start_time: Начальное время (timestamp)
            end_time: Конечное время (timestamp)
            limit: Максимальное количество записей

        Returns:
            List[Quote]: Список котировок
        """
        try:
            with self.db_manager.get_connection() as conn:
                query = (
                    "SELECT symbol, timestamp, bid, ask, volume FROM quotes "
                    "WHERE symbol = ?"
                )
                params: List[Any] = [symbol]

                if start_time is not None:
                    query += " AND timestamp >= ?"
                    params.append(start_time)

                if end_time is not None:
                    query += " AND timestamp <= ?"
                    params.append(end_time)

                query += " ORDER BY timestamp DESC"

                if limit is not None:
                    query += " LIMIT ?"
                    params.append(limit)

                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

                quotes = []
                for row in rows:
                    quotes.append(
                        Quote(
                            symbol=row[0],
                            timestamp=row[1],
                            bid=row[2],
                            ask=row[3],
                            volume=row[4],
                        )
                    )

                return quotes

        except Exception as e:
            logger.error(f"Ошибка получения котировок: {e}")
            return []

    def get_latest_quote(self, symbol: str) -> Optional[Quote]:
        """
        Получает последнюю котировку для символа.

        Args:
            symbol: Символ инструмента

        Returns:
            Optional[Quote]: Последняя котировка или None
        """
        quotes = self.get_quotes(symbol, limit=1)
        return quotes[0] if quotes else None

    def _log_sync_operation(
        self, table: str, operation: str, count: int, details: str = None
    ):
        """Логирует операцию синхронизации."""
        try:
            with self.db_manager.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO sync_logs (table_name, operation, record_count, details)
                    VALUES (?, ?, ?, ?)
                """,
                    (table, operation, count, details),
                )
        except Exception as e:
            logger.error(f"Ошибка логирования операции синхронизации: {e}")


class TradeStorage:
    """
    Хранилище сделок в SQLite.
    """

    def __init__(self, db_manager):
        """
        Инициализация хранилища сделок.

        Args:
            db_manager: Экземпляр DatabaseManager
        """
        self.db_manager = db_manager

    def store_trade(self, trade: Trade) -> bool:
        """
        Сохраняет сделку в базу данных.

        Args:
            trade: Объект сделки

        Returns:
            bool: True если сохранение успешно, False иначе
        """
        try:
            with self.db_manager.get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO trades
                    (trade_id, symbol, side, entry_price, quantity, stop_loss, take_profit,
                     status, opened_at, closed_at, pnl, pnl_points, commission, magic_number,
                     comment, signal_id, risk_amount, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%s', 'now'))
                """,
                    (
                        trade.id
                        or f"trade_{int(datetime.now().timestamp())}_{trade.symbol}",
                        trade.symbol,
                        str(trade.side),
                        trade.entry_price,
                        trade.quantity,
                        trade.stop_loss,
                        trade.take_profit,
                        trade.status,
                        trade.opened_at.timestamp() if trade.opened_at else None,
                        trade.closed_at.timestamp() if trade.closed_at else None,
                        trade.pnl,
                        trade.pnl_points,
                        trade.commission,
                        trade.magic_number,
                        trade.comment,
                        trade.signal_id,
                        trade.risk_amount,
                    ),
                )

                return True

        except Exception as e:
            logger.error(f"Ошибка сохранения сделки: {e}")
            return False

    def get_trades(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Trade]:
        """
        Получает сделки из базы данных.

        Args:
            symbol: Фильтр по символу (опционально)
            status: Фильтр по статусу (опционально)
            limit: Максимальное количество записей

        Returns:
            List[Trade]: Список сделок
        """
        try:
            with self.db_manager.get_connection() as conn:
                query = """
                    SELECT trade_id, symbol, side, entry_price, quantity, stop_loss, take_profit,
                           status, opened_at, closed_at, pnl, pnl_points, commission, magic_number,
                           comment, signal_id, risk_amount
                    FROM trades WHERE 1=1
                """
                params = []

                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)

                if status:
                    query += " AND status = ?"
                    params.append(status)

                query += " ORDER BY created_at DESC"

                if limit:
                    query += " LIMIT ?"
                    params.append(limit)

                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

                trades = []
                for row in rows:
                    trades.append(
                        Trade(
                            id=row[0],
                            symbol=row[1],
                            side=row[2] if isinstance(row[2], str) else OrderSide.BUY.value,
                            entry_price=row[3],
                            quantity=row[4],
                            stop_loss=row[5],
                            take_profit=row[6],
                            status=row[7],
                            opened_at=(
                                datetime.fromtimestamp(row[8]) if row[8] else None
                            ),
                            closed_at=(
                                datetime.fromtimestamp(row[9]) if row[9] else None
                            ),
                            pnl=row[10],
                            pnl_points=row[11],
                            commission=row[12],
                            magic_number=row[13],
                            comment=row[14],
                            signal_id=row[15],
                            risk_amount=row[16],
                        )
                    )

                return trades

        except Exception as e:
            logger.error(f"Ошибка получения сделок: {e}")
            return []


class SignalStorage:
    """
    Хранилище сигналов в SQLite.
    """

    def __init__(self, db_manager):
        """
        Инициализация хранилища сигналов.

        Args:
            db_manager: Экземпляр DatabaseManager
        """
        self.db_manager = db_manager

    def store_signal(self, signal: Signal) -> bool:
        """
        Сохраняет сигнал в базу данных.

        Args:
            signal: Объект сигнала

        Returns:
            bool: True если сохранение успешно, False иначе
        """
        try:
            with self.db_manager.get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO signals
                    (signal_id, symbol, timeframe, action, side, confidence, entry_price,
                     stop_loss, take_profit, quantity, risk_amount, reason, indicators,
                     market_regime, related_signals, tags, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        signal.id
                        or f"signal_{int(datetime.now().timestamp())}_{signal.symbol}",
                        signal.symbol,
                        str(signal.timeframe),
                        str(signal.action),
                        signal.side.value if signal.side else None,
                        signal.confidence,
                        signal.entry_price,
                        signal.stop_loss,
                        signal.take_profit,
                        signal.quantity,
                        signal.risk_amount,
                        signal.reason,
                        json.dumps(signal.indicators) if signal.indicators else None,
                        signal.market_regime.value if signal.market_regime else None,
                        (
                            json.dumps(signal.related_signals)
                            if signal.related_signals
                            else None
                        ),
                        json.dumps(signal.tags) if signal.tags else None,
                        signal.expires_at.timestamp() if signal.expires_at else None,
                    ),
                )

                return True

        except Exception as e:
            logger.error(f"Ошибка сохранения сигнала: {e}")
            return False

    def get_signals(
        self,
        symbol: Optional[str] = None,
        action: Optional[SignalAction] = None,
        limit: Optional[int] = None,
    ) -> List[Signal]:
        """
        Получает сигналы из базы данных.

        Args:
            symbol: Фильтр по символу (опционально)
            action: Фильтр по действию (опционально)
            limit: Максимальное количество записей

        Returns:
            List[Signal]: Список сигналов
        """
        try:
            with self.db_manager.get_connection() as conn:
                query = """
                    SELECT signal_id, symbol, timeframe, action, side, confidence, entry_price,
                           stop_loss, take_profit, quantity, risk_amount, reason, indicators,
                           market_regime, related_signals, tags, created_at, expires_at
                    FROM signals WHERE 1=1
                """
                params = []

                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)

                if action:
                    query += " AND action = ?"
                    params.append(action.value)

                query += " ORDER BY created_at DESC"

                if limit:
                    query += " LIMIT ?"
                    params.append(limit)

                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

                signals = []
                for row in rows:
                    signals.append(
                        Signal(
                            id=row[0],
                            symbol=row[1],
                        timeframe=row[2] if isinstance(row[2], str) else Timeframe.M1.value,
                        action=row[3] if isinstance(row[3], str) else SignalAction.ENTER.value,
                        side=row[4] if isinstance(row[4], str) else None,
                            confidence=row[5],
                            entry_price=row[6],
                            stop_loss=row[7],
                            take_profit=row[8],
                            quantity=row[9],
                            risk_amount=row[10],
                            reason=row[11],
                            indicators=json.loads(row[12]) if row[12] else {},
                            market_regime=MarketRegime(row[13]) if row[13] else None,
                            related_signals=json.loads(row[14]) if row[14] else [],
                            tags=json.loads(row[15]) if row[15] else [],
                            created_at=(
                                datetime.fromtimestamp(row[16])
                                if row[16]
                                else datetime.utcnow()
                            ),
                            expires_at=(
                                datetime.fromtimestamp(row[17]) if row[17] else None
                            ),
                        )
                    )

                return signals

        except Exception as e:
            logger.error(f"Ошибка получения сигналов: {e}")
            return []


class TechnicalIndicatorsStorage:
    """
    Хранилище технических индикаторов в SQLite.
    """

    def __init__(self, db_manager):
        """
        Инициализация хранилища индикаторов.

        Args:
            db_manager: Экземпляр DatabaseManager
        """
        self.db_manager = db_manager

    def store_indicators(self, indicators: TechnicalIndicators) -> bool:
        """
        Сохраняет технические индикаторы в базу данных.

        Args:
            indicators: Объект индикаторов

        Returns:
            bool: True если сохранение успешно, False иначе
        """
        try:
            with self.db_manager.get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO technical_indicators
                    (symbol, timeframe, timestamp, ema_20, ema_50, ema_200, sma_20, rsi,
                     stoch_k, stoch_d, atr, atr_percent, volume, rvol, donchian_upper,
                     donchian_lower, opening_range_high, opening_range_low)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        indicators.symbol,
                        str(indicators.timeframe),
                        indicators.timestamp,
                        indicators.ema_20,
                        indicators.ema_50,
                        indicators.ema_200,
                        indicators.sma_20,
                        indicators.rsi,
                        indicators.stoch_k,
                        indicators.stoch_d,
                        indicators.atr,
                        indicators.atr_percent,
                        indicators.volume,
                        indicators.rvol,
                        indicators.donchian_upper,
                        indicators.donchian_lower,
                        indicators.opening_range_high,
                        indicators.opening_range_low,
                    ),
                )

                return True

        except Exception as e:
            logger.error(f"Ошибка сохранения индикаторов: {e}")
            return False

    def get_indicators(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[TechnicalIndicators]:
        """
        Получает технические индикаторы из базы данных.

        Args:
            symbol: Символ инструмента
            timeframe: Таймфрейм
            start_time: Начальное время (timestamp)
            end_time: Конечное время (timestamp)
            limit: Максимальное количество записей

        Returns:
            List[TechnicalIndicators]: Список индикаторов
        """
        try:
            with self.db_manager.get_connection() as conn:
                query = """
                    SELECT symbol, timeframe, timestamp, ema_20, ema_50, ema_200, sma_20, rsi,
                           stoch_k, stoch_d, atr, atr_percent, volume, rvol, donchian_upper,
                           donchian_lower, opening_range_high, opening_range_low
                    FROM technical_indicators
                    WHERE symbol = ? AND timeframe = ?
                """
                params = [symbol, timeframe.value]

                if start_time is not None:
                    query += " AND timestamp >= ?"
                    params.append(start_time)

                if end_time is not None:
                    query += " AND timestamp <= ?"
                    params.append(end_time)

                query += " ORDER BY timestamp DESC"

                if limit is not None:
                    query += " LIMIT ?"
                    params.append(limit)

                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

                indicators_list = []
                for row in rows:
                    indicators_list.append(
                        TechnicalIndicators(
                            symbol=row[0],
                            timeframe=row[1] if isinstance(row[1], str) else Timeframe.M1.value,
                            timestamp=row[2],
                            ema_20=row[3],
                            ema_50=row[4],
                            ema_200=row[5],
                            sma_20=row[6],
                            rsi=row[7],
                            stoch_k=row[8],
                            stoch_d=row[9],
                            atr=row[10],
                            atr_percent=row[11],
                            volume=row[12],
                            rvol=row[13],
                            donchian_upper=row[14],
                            donchian_lower=row[15],
                            opening_range_high=row[16],
                            opening_range_low=row[17],
                        )
                    )

                return indicators_list

        except Exception as e:
            logger.error(f"Ошибка получения индикаторов: {e}")
            return []


# Глобальные экземпляры хранилищ
quote_storage = None
trade_storage = None
signal_storage = None
indicators_storage = None


def initialize_storages(db_manager) -> None:
    """
    Инициализирует глобальные экземпляры хранилищ.

    Args:
        db_manager: Экземпляр DatabaseManager
    """
    global quote_storage, trade_storage, signal_storage, indicators_storage

    quote_storage = QuoteStorage(db_manager)
    trade_storage = TradeStorage(db_manager)
    signal_storage = SignalStorage(db_manager)
    indicators_storage = TechnicalIndicatorsStorage(db_manager)
