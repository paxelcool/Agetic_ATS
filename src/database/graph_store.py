"""
Графовое хранилище на базе Memgraph для ATS.

Предоставляет функции для построения и анализа графов связей между торговыми событиями.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import memgraph

    MEMGRAPH_AVAILABLE = True
except ImportError:
    MEMGRAPH_AVAILABLE = False
    logger.warning("Memgraph не установлен. Графовые операции будут недоступны.")


class GraphStore:
    """
    Графовое хранилище на базе Memgraph.

    Обеспечивает анализ причинно-следственных связей между торговыми событиями.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "memgraph",
        password: str = "memgraph",
    ):
        """
        Инициализация графового хранилища.

        Args:
            uri: URI подключения к Memgraph
            user: Имя пользователя
            password: Пароль
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

        if MEMGRAPH_AVAILABLE:
            try:
                self.driver = memgraph.connect(uri, user, password)
                logger.info(f"Подключение к Memgraph установлено: {uri}")
            except Exception as e:
                logger.error(f"Ошибка подключения к Memgraph: {e}")
                logger.warning("Memgraph не установлен. Графовые операции будут недоступны.")
                self.driver = None

    def initialize_graph(self) -> bool:
        """
        Инициализирует граф с базовой онтологией.

        Returns:
            bool: True если инициализация успешна, False иначе
        """
        if not self.driver:
            logger.error("Memgraph не доступен")
            return False

        try:
            # Создаем ограничения уникальности
            constraints = [
                "CREATE CONSTRAINT ON (t:Trade) ASSERT t.id IS UNIQUE",
                "CREATE CONSTRAINT ON (s:Signal) ASSERT s.id IS UNIQUE",
                "CREATE CONSTRAINT ON (q:Quote) ASSERT q.id IS UNIQUE",
                "CREATE CONSTRAINT ON (i:Instrument) ASSERT i.symbol IS UNIQUE",
                "CREATE CONSTRAINT ON (r:Regime) ASSERT r.name IS UNIQUE",
            ]

            for constraint in constraints:
                try:
                    self.driver.execute(constraint)
                except Exception as e:
                    logger.warning(f"Не удалось создать ограничение: {e}")

            # Создаем индексы
            indexes = [
                "CREATE INDEX ON :Instrument(symbol)",
                "CREATE INDEX ON :Regime(name)",
                "CREATE INDEX ON :Trade(symbol)",
                "CREATE INDEX ON :Signal(symbol)",
            ]

            for index in indexes:
                try:
                    self.driver.execute(index)
                except Exception as e:
                    logger.warning(f"Не удалось создать индекс: {e}")

            logger.info("Граф Memgraph инициализирован успешно")
            return True

        except Exception as e:
            logger.error(f"Ошибка инициализации графа: {e}")
            return False

    def store_instrument(
        self, symbol: str, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Сохраняет инструмент в граф.

        Args:
            symbol: Символ инструмента
            metadata: Дополнительные метаданные

        Returns:
            bool: True если сохранение успешно, False иначе
        """
        if not self.driver:
            return False

        try:
            query = """
                MERGE (i:Instrument {symbol: $symbol})
                SET i += $metadata
                RETURN i
            """

            self.driver.execute(query, symbol=symbol, metadata=metadata or {})
            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения инструмента: {e}")
            return False

    def store_quote(self, quote_data: Dict[str, Any]) -> bool:
        """
        Сохраняет котировку в граф.

        Args:
            quote_data: Данные котировки

        Returns:
            bool: True если сохранение успешно, False иначе
        """
        if not self.driver:
            return False

        try:
            # Создаем узел котировки
            quote_query = """
                MATCH (i:Instrument {symbol: $symbol})
                CREATE (q:Quote {
                    id: $id,
                    timestamp: $timestamp,
                    bid: $bid,
                    ask: $ask,
                    volume: $volume
                })
                CREATE (i)-[:HAS_QUOTE]->(q)
                RETURN q
            """

            self.driver.execute(quote_query, **quote_data)
            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения котировки в граф: {e}")
            return False

    def store_trade(self, trade_data: Dict[str, Any]) -> bool:
        """
        Сохраняет сделку в граф.

        Args:
            trade_data: Данные сделки

        Returns:
            bool: True если сохранение успешно, False иначе
        """
        if not self.driver:
            return False

        try:
            # Создаем узел сделки
            trade_query = """
                MATCH (i:Instrument {symbol: $symbol})
                CREATE (t:Trade {
                    id: $id,
                    side: $side,
                    entry_price: $entry_price,
                    quantity: $quantity,
                    status: $status,
                    pnl: $pnl,
                    opened_at: $opened_at
                })
                CREATE (i)-[:HAS_TRADE]->(t)
                RETURN t
            """

            self.driver.execute(trade_query, **trade_data)
            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения сделки в граф: {e}")
            return False

    def store_signal(self, signal_data: Dict[str, Any]) -> bool:
        """
        Сохраняет сигнал в граф.

        Args:
            signal_data: Данные сигнала

        Returns:
            bool: True если сохранение успешно, False иначе
        """
        if not self.driver:
            return False

        try:
            # Создаем узел сигнала
            signal_query = """
                MATCH (i:Instrument {symbol: $symbol})
                CREATE (s:Signal {
                    id: $id,
                    action: $action,
                    side: $side,
                    confidence: $confidence,
                    reason: $reason,
                    created_at: $created_at
                })
                CREATE (i)-[:HAS_SIGNAL]->(s)
                RETURN s
            """

            self.driver.execute(signal_query, **signal_data)
            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения сигнала в граф: {e}")
            return False

    def create_trade_signal_relationship(self, trade_id: str, signal_id: str) -> bool:
        """
        Создает связь между сделкой и сигналом.

        Args:
            trade_id: ID сделки
            signal_id: ID сигнала

        Returns:
            bool: True если связь создана, False иначе
        """
        if not self.driver:
            return False

        try:
            query = """
                MATCH (t:Trade {id: $trade_id})
                MATCH (s:Signal {id: $signal_id})
                CREATE (s)-[:TRIGGERED]->(t)
                RETURN t, s
            """

            self.driver.execute(query, trade_id=trade_id, signal_id=signal_id)
            return True

        except Exception as e:
            logger.error(f"Ошибка создания связи сигнал-сделка: {e}")
            return False

    def store_market_regime(
        self,
        regime_name: str,
        timestamp: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Сохраняет рыночный режим.

        Args:
            regime_name: Название режима
            timestamp: Временная метка
            metadata: Дополнительные метаданные

        Returns:
            bool: True если сохранение успешно, False иначе
        """
        if not self.driver:
            return False

        try:
            query = """
                MERGE (r:Regime {name: $regime_name})
                SET r.last_seen = $timestamp
                SET r += $metadata
                RETURN r
            """

            self.driver.execute(
                query,
                regime_name=regime_name,
                timestamp=timestamp,
                metadata=metadata or {},
            )
            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения рыночного режима: {e}")
            return False

    def link_trade_to_regime(self, trade_id: str, regime_name: str) -> bool:
        """
        Связывает сделку с рыночным режимом.

        Args:
            trade_id: ID сделки
            regime_name: Название режима

        Returns:
            bool: True если связь создана, False иначе
        """
        if not self.driver:
            return False

        try:
            query = """
                MATCH (t:Trade {id: $trade_id})
                MATCH (r:Regime {name: $regime_name})
                CREATE (t)-[:IN_REGIME]->(r)
                RETURN t, r
            """

            self.driver.execute(query, trade_id=trade_id, regime_name=regime_name)
            return True

        except Exception as e:
            logger.error(f"Ошибка связи сделка-режим: {e}")
            return False

    def get_signal_trade_chain(self, signal_id: str) -> List[Dict[str, Any]]:
        """
        Получает цепочку сигнал -> сделка для анализа.

        Args:
            signal_id: ID сигнала

        Returns:
            List[Dict[str, Any]]: Цепочка событий
        """
        if not self.driver:
            return []

        try:
            query = """
                MATCH (s:Signal {id: $signal_id})-[:TRIGGERED]->(t:Trade)
                OPTIONAL MATCH (t)-[:IN_REGIME]->(r:Regime)
                RETURN s, t, r
            """

            results = self.driver.execute(query, signal_id=signal_id)

            chain = []
            for record in results:
                signal_node = dict(record["s"])
                trade_node = dict(record["t"])
                regime_node = dict(record["r"]) if record["r"] else None

                chain.append(
                    {"signal": signal_node, "trade": trade_node, "regime": regime_node}
                )

            return chain

        except Exception as e:
            logger.error(f"Ошибка получения цепочки сигнал-сделка: {e}")
            return []

    def get_instruments_in_regime(self, regime_name: str) -> List[Dict[str, Any]]:
        """
        Получает инструменты, торговавшиеся в определенном режиме.

        Args:
            regime_name: Название режима

        Returns:
            List[Dict[str, Any]]: Инструменты с метаданными
        """
        if not self.driver:
            return []

        try:
            query = """
                MATCH (i:Instrument)-[:HAS_TRADE]->(t:Trade)-[:IN_REGIME]->(r:Regime {name: $regime_name})
                RETURN i, COUNT(t) as trade_count
            """

            results = self.driver.execute(query, regime_name=regime_name)

            instruments = []
            for record in results:
                instrument = dict(record["i"])
                instrument["trade_count"] = record["trade_count"]
                instruments.append(instrument)

            return instruments

        except Exception as e:
            logger.error(f"Ошибка получения инструментов в режиме: {e}")
            return []

    def analyze_signal_effectiveness(self) -> Dict[str, Any]:
        """
        Анализирует эффективность сигналов.

        Returns:
            Dict[str, Any]: Результаты анализа
        """
        if not self.driver:
            return {}

        try:
            # Анализируем успешность сигналов
            query = """
                MATCH (s:Signal)-[:TRIGGERED]->(t:Trade)
                WHERE t.pnl IS NOT NULL
                RETURN
                    s.action as action,
                    AVG(t.pnl) as avg_pnl,
                    COUNT(t) as total_trades,
                    SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    AVG(s.confidence) as avg_confidence
            """

            results = self.driver.execute(query)

            analysis = {}
            for record in results:
                action = record["action"]
                analysis[action] = {
                    "avg_pnl": record["avg_pnl"],
                    "total_trades": record["total_trades"],
                    "winning_trades": record["winning_trades"],
                    "win_rate": (
                        record["winning_trades"] / record["total_trades"]
                        if record["total_trades"] > 0
                        else 0
                    ),
                    "avg_confidence": record["avg_confidence"],
                }

            return analysis

        except Exception as e:
            logger.error(f"Ошибка анализа эффективности сигналов: {e}")
            return {}

    def get_regime_performance(self) -> Dict[str, Any]:
        """
        Получает статистику производительности по режимам.

        Returns:
            Dict[str, Any]: Статистика по режимам
        """
        if not self.driver:
            return {}

        try:
            query = """
                MATCH (r:Regime)<-[:IN_REGIME]-(t:Trade)
                WHERE t.pnl IS NOT NULL
                RETURN
                    r.name as regime,
                    COUNT(t) as trade_count,
                    AVG(t.pnl) as avg_pnl,
                    SUM(t.pnl) as total_pnl,
                    AVG(t.entry_price * t.quantity) as avg_position_size
            """

            results = self.driver.execute(query)

            performance = {}
            for record in results:
                regime = record["regime"]
                performance[regime] = {
                    "trade_count": record["trade_count"],
                    "avg_pnl": record["avg_pnl"],
                    "total_pnl": record["total_pnl"],
                    "avg_position_size": record["avg_position_size"],
                }

            return performance

        except Exception as e:
            logger.error(f"Ошибка получения статистики режимов: {e}")
            return {}

    def close(self) -> None:
        """Закрывает соединение с графовой базой данных."""
        if self.driver:
            try:
                self.driver.close()
                logger.info("Соединение с Memgraph закрыто")
            except Exception as e:
                logger.error(f"Ошибка закрытия соединения: {e}")


def create_graph_store(
    uri: str = "bolt://localhost:7687",
    user: str = "memgraph",
    password: str = "memgraph",
) -> GraphStore:
    """
    Создает экземпляр графового хранилища.

    Args:
        uri: URI подключения к Memgraph
        user: Имя пользователя
        password: Пароль

    Returns:
        GraphStore: Экземпляр графового хранилища
    """
    return GraphStore(uri, user, password)


# Глобальный экземпляр для использования в приложении
graph_store = None


def initialize_graph_store(
    uri: str = "bolt://localhost:7687",
    user: str = "memgraph",
    password: str = "memgraph",
) -> None:
    """
    Инициализирует глобальный экземпляр графового хранилища.

    Args:
        uri: URI подключения к Memgraph
        user: Имя пользователя
        password: Пароль
    """
    global graph_store
    graph_store = create_graph_store(uri, user, password)
