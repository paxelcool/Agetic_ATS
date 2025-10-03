"""
Процессор данных для синхронизации.

Обрабатывает и валидирует рыночные данные перед сохранением в базы данных.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.database.models import Quote, Trade

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Процессор для обработки и валидации рыночных данных.

    Обеспечивает очистку, валидацию и трансформацию данных перед сохранением.
    """

    def __init__(self):
        """Инициализация процессора данных."""

    def process_quote(self, raw_quote: Dict[str, Any]) -> Optional[Quote]:
        """
        Обрабатывает сырые данные котировки.

        Args:
            raw_quote: Сырые данные котировки

        Returns:
            Optional[Quote]: Обработанная котировка или None при ошибке
        """
        try:
            # Валидация обязательных полей
            required_fields = ["symbol", "timestamp", "bid", "ask"]
            for field in required_fields:
                if field not in raw_quote:
                    logger.warning(f"Отсутствует обязательное поле в котировке: {field}")
                    return None

            # Проверка типов данных
            if not isinstance(raw_quote["bid"], (int, float)) or raw_quote["bid"] <= 0:
                logger.warning(f"Неверное значение bid: {raw_quote['bid']}")
                return None

            if not isinstance(raw_quote["ask"], (int, float)) or raw_quote["ask"] <= 0:
                logger.warning(f"Неверное значение ask: {raw_quote['ask']}")
                return None

            if raw_quote["bid"] >= raw_quote["ask"]:
                logger.warning(f"Неверный порядок цен: bid >= ask")
                return None

            # Создаем объект котировки
            quote = Quote(
                symbol=raw_quote["symbol"],
                timestamp=raw_quote["timestamp"],
                bid=float(raw_quote["bid"]),
                ask=float(raw_quote["ask"]),
                volume=int(raw_quote.get("volume", 0)) if raw_quote.get("volume") else None,
            )

            return quote

        except Exception as e:
            logger.error(f"Ошибка обработки котировки: {e}")
            return None

    def process_quotes_batch(self, raw_quotes: List[Dict[str, Any]]) -> List[Quote]:
        """
        Обрабатывает пакет котировок.

        Args:
            raw_quotes: Список сырых котировок

        Returns:
            List[Quote]: Список обработанных котировок
        """
        processed_quotes = []

        for raw_quote in raw_quotes:
            processed_quote = self.process_quote(raw_quote)
            if processed_quote:
                processed_quotes.append(processed_quote)

        logger.info(f"Обработано котировок: {len(processed_quotes)} из {len(raw_quotes)}")
        return processed_quotes

    def process_trade(self, raw_trade: Dict[str, Any]) -> Optional[Trade]:
        """
        Обрабатывает сырые данные сделки.

        Args:
            raw_trade: Сырые данные сделки

        Returns:
            Optional[Trade]: Обработанная сделка или None при ошибке
        """
        try:
            # Валидация обязательных полей
            required_fields = ["id", "symbol", "side", "entry_price", "quantity"]
            for field in required_fields:
                if field not in raw_trade:
                    logger.warning(f"Отсутствует обязательное поле в сделке: {field}")
                    return None

            # Проверка типов данных
            if not isinstance(raw_trade["entry_price"], (int, float)) or raw_trade["entry_price"] <= 0:
                logger.warning(f"Неверная цена входа: {raw_trade['entry_price']}")
                return None

            if not isinstance(raw_trade["quantity"], (int, float)) or raw_trade["quantity"] <= 0:
                logger.warning(f"Неверное количество: {raw_trade['quantity']}")
                return None

            # Создаем объект сделки
            trade = Trade(
                id=raw_trade["id"],
                symbol=raw_trade["symbol"],
                side=raw_trade["side"],
                entry_price=float(raw_trade["entry_price"]),
                quantity=float(raw_trade["quantity"]),
                stop_loss=raw_trade.get("stop_loss"),
                take_profit=raw_trade.get("take_profit"),
                status=raw_trade.get("status", "closed"),
                opened_at=raw_trade.get("opened_at"),
                closed_at=raw_trade.get("closed_at"),
                pnl=raw_trade.get("pnl"),
                commission=raw_trade.get("commission", 0.0),
                comment=raw_trade.get("comment"),
            )

            return trade

        except Exception as e:
            logger.error(f"Ошибка обработки сделки: {e}")
            return None

    def process_trades_batch(self, raw_trades: List[Dict[str, Any]]) -> List[Trade]:
        """
        Обрабатывает пакет сделок.

        Args:
            raw_trades: Список сырых сделок

        Returns:
            List[Trade]: Список обработанных сделок
        """
        processed_trades = []

        for raw_trade in raw_trades:
            processed_trade = self.process_trade(raw_trade)
            if processed_trade:
                processed_trades.append(processed_trade)

        logger.info(f"Обработано сделок: {len(processed_trades)} из {len(raw_trades)}")
        return processed_trades

    def filter_duplicate_quotes(self, quotes: List[Quote]) -> List[Quote]:
        """
        Фильтрует дублирующиеся котировки.

        Args:
            quotes: Список котировок

        Returns:
            List[Quote]: Отфильтрованный список котировок
        """
        seen = set()
        filtered_quotes = []

        for quote in quotes:
            key = (quote.symbol, quote.timestamp)
            if key not in seen:
                seen.add(key)
                filtered_quotes.append(quote)

        if len(filtered_quotes) < len(quotes):
            logger.info(f"Отфильтровано дубликатов котировок: {len(quotes) - len(filtered_quotes)}")

        return filtered_quotes

    def filter_recent_quotes(self, quotes: List[Quote], max_age_seconds: int = 3600) -> List[Quote]:
        """
        Фильтрует котировки по времени (оставляет только свежие).

        Args:
            quotes: Список котировок
            max_age_seconds: Максимальный возраст котировки в секундах

        Returns:
            List[Quote]: Отфильтрованный список котировок
        """
        current_time = int(datetime.now().timestamp())
        filtered_quotes = []

        for quote in quotes:
            if current_time - quote.timestamp <= max_age_seconds:
                filtered_quotes.append(quote)

        if len(filtered_quotes) < len(quotes):
            logger.info(f"Отфильтровано старых котировок: {len(quotes) - len(filtered_quotes)}")

        return filtered_quotes

    def validate_market_hours(self, quotes: List[Quote], symbol: str) -> List[Quote]:
        """
        Валидирует котировки на предмет рыночных часов.

        Args:
            quotes: Список котировок
            symbol: Символ инструмента

        Returns:
            List[Quote]: Провалидированный список котировок
        """
        # В будущем можно добавить проверку рыночных часов для разных инструментов
        # Пока просто возвращаем все котировки

        valid_quotes = []

        for quote in quotes:
            # Простая валидация - цена должна быть в разумных пределах
            if 0.0001 <= quote.bid <= 100000 and 0.0001 <= quote.ask <= 100000:
                valid_quotes.append(quote)
            else:
                logger.warning(f"Котировка с подозрительными ценами: {quote.symbol} {quote.bid}/{quote.ask}")

        return valid_quotes

    def calculate_quote_metrics(self, quotes: List[Quote]) -> Dict[str, Any]:
        """
        Рассчитывает метрики для котировок.

        Args:
            quotes: Список котировок

        Returns:
            Dict[str, Any]: Рассчитанные метрики
        """
        if not quotes:
            return {}

        metrics = {
            "total_quotes": len(quotes),
            "symbols": list(set(q.symbol for q in quotes)),
            "time_range": {
                "start": min(q.timestamp for q in quotes),
                "end": max(q.timestamp for q in quotes),
            },
            "price_ranges": {},
        }

        # Рассчитываем диапазоны цен для каждого символа
        for symbol in metrics["symbols"]:
            symbol_quotes = [q for q in quotes if q.symbol == symbol]
            prices = [q.bid for q in symbol_quotes] + [q.ask for q in symbol_quotes]

            if prices:
                metrics["price_ranges"][symbol] = {
                    "min": min(prices),
                    "max": max(prices),
                    "avg": sum(prices) / len(prices),
                }

        return metrics
