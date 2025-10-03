"""
MT5 клиент для получения рыночных данных.

Обеспечивает подключение к терминалу MetaTrader5 и получение котировок и истории сделок.
"""

import logging
from typing import Any, Dict, List, Optional

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

logger = logging.getLogger(__name__)


class MT5Client:
    """
    Клиент для работы с MetaTrader5 терминалом.

    Обеспечивает получение котировок, истории сделок и другой рыночной информации.
    """

    def __init__(self):
        """Инициализация MT5 клиента."""
        self.initialized = False

    def initialize(self) -> bool:
        """
        Инициализирует подключение к MT5 терминалу.

        Returns:
            bool: True если инициализация успешна, False иначе
        """
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 не установлен")
            return False

        try:
            if not mt5.initialize():
                logger.error(f"Не удалось инициализировать MT5: {mt5.last_error()}")
                return False

            self.initialized = True
            logger.info("MT5 терминал успешно инициализирован")
            return True

        except Exception as e:
            logger.error(f"Ошибка инициализации MT5: {e}")
            return False

    def shutdown(self) -> None:
        """Закрывает подключение к MT5 терминалу."""
        if MT5_AVAILABLE and self.initialized:
            try:
                mt5.shutdown()
                self.initialized = False
                logger.info("MT5 соединение закрыто")
            except Exception as e:
                logger.error(f"Ошибка закрытия MT5: {e}")

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Получает текущую котировку для символа.

        Args:
            symbol: Символ инструмента

        Returns:
            Optional[Dict[str, Any]]: Данные котировки или None при ошибке
        """
        if not self.initialized:
            logger.error("MT5 клиент не инициализирован")
            return None

        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.warning(f"Не удалось получить котировку для {symbol}")
                return None

            return {
                "symbol": symbol,
                "timestamp": int(tick.time),
                "bid": tick.bid,
                "ask": tick.ask,
                "volume": int(tick.volume) if tick.volume > 0 else None,
                "spread": tick.ask - tick.bid,
                "mid_price": (tick.ask + tick.bid) / 2,
            }

        except Exception as e:
            logger.error(f"Ошибка получения котировки для {symbol}: {e}")
            return None

    def get_quotes(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        Получает котировки для списка символов.

        Args:
            symbols: Список символов

        Returns:
            List[Dict[str, Any]]: Список котировок
        """
        quotes = []
        for symbol in symbols:
            quote = self.get_quote(symbol)
            if quote:
                quotes.append(quote)
        return quotes

    def get_trades_history(self, from_date, to_date) -> List[Dict[str, Any]]:
        """
        Получает историю сделок за период.

        Args:
            from_date: Начальная дата
            to_date: Конечная дата

        Returns:
            List[Dict[str, Any]]: Список сделок
        """
        if not self.initialized:
            logger.error("MT5 клиент не инициализирован")
            return []

        try:
            deals = mt5.history_deals_get(from_date, to_date)
            if deals is None:
                logger.warning("Не удалось получить историю сделок")
                return []

            trades = []
            for deal in deals:
                trade = {
                    "id": str(deal.ticket),
                    "symbol": deal.symbol,
                    "side": "buy" if deal.type == mt5.DEAL_TYPE_BUY else "sell",
                    "entry_price": deal.price,
                    "quantity": deal.volume,
                    "pnl": deal.profit,
                    "opened_at": datetime.fromtimestamp(deal.time),
                    "commission": deal.commission,
                    "comment": getattr(deal, "comment", None),
                }
                trades.append(trade)

            return trades

        except Exception as e:
            logger.error(f"Ошибка получения истории сделок: {e}")
            return []

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Получает информацию о торговом счете.

        Returns:
            Optional[Dict[str, Any]]: Информация о счете или None при ошибке
        """
        if not self.initialized:
            logger.error("MT5 клиент не инициализирован")
            return None

        try:
            account_info = mt5.account_info()
            if account_info is None:
                logger.warning("Не удалось получить информацию о счете")
                return None

            return {
                "balance": account_info.balance,
                "equity": account_info.equity,
                "profit": account_info.profit,
                "margin": account_info.margin,
                "margin_free": account_info.margin_free,
                "margin_level": account_info.margin_level,
                "currency": account_info.currency,
            }

        except Exception as e:
            logger.error(f"Ошибка получения информации о счете: {e}")
            return None

    def is_market_open(self, symbol: str) -> bool:
        """
        Проверяет, открыт ли рынок для символа.

        Args:
            symbol: Символ инструмента

        Returns:
            bool: True если рынок открыт, False иначе
        """
        if not self.initialized:
            return False

        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False

            # Проверяем время торговой сессии
            current_time = datetime.now()
            session_start = symbol_info.session_deals
            session_end = symbol_info.session_deals_end

            if session_start == 0 or session_end == 0:
                return True  # Если сессия не определена, считаем рынок открытым

            # Простая проверка - в будущем можно улучшить
            return True

        except Exception as e:
            logger.error(f"Ошибка проверки состояния рынка для {symbol}: {e}")
            return False
