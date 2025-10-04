"""
Основной сервис синхронизации данных для ATS.

Координирует получение данных из MT5 и их хранение в различных базах данных.

Содержит следующие компоненты:
- MT5 клиент для получения данных
- Обработчик данных для валидации и трансформации
- Сервис эмбеддингов для векторного хранения
- Обработчик ошибок для надежной работы
"""

import asyncio
import importlib
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - среда может не содержать MetaTrader5
    mt5 = None

from src.config import settings
graph_store_module = importlib.import_module("src.database.graph_store")
storage_module = importlib.import_module("src.database.storage")
vector_store_module = importlib.import_module("src.database.vector_store")
from src.database.init_db import create_database_manager
from src.database.models import Quote, Trade

# Импортируем компоненты синхронизации
from .clients.mt5_client import MT5Client
from .processors.data_processor import DataProcessor
from .services.embedding_service import EmbeddingService
from .utils.error_handler import ErrorHandler, RetryConfig

logger = logging.getLogger(__name__)


class SyncService:
    """
    Основной сервис синхронизации данных.

    Управляет получением данных из MT5 и их распределением по различным хранилищам.
    """

    def __init__(self):
        """
        Инициализация сервиса синхронизации.
        """
        self.db_manager = create_database_manager(settings.sync_db_path)
        self.is_running = False
        self.last_sync_time: Optional[datetime] = None
        self.sync_errors: List[Dict[str, Any]] = []

        # Инициализируем компоненты синхронизации
        self.mt5_client = MT5Client()
        self.data_processor = DataProcessor()
        self.embedding_service = EmbeddingService()
        self.error_handler = ErrorHandler(RetryConfig(max_attempts=3, base_delay=1.0))

        # Инициализируем хранилища
        self._initialize_storages()

    def _initialize_storages(self) -> None:
        """
        Инициализирует все типы хранилищ.
        """
        try:
            # SQLite
            if self.db_manager.initialize_database():
                storage_module.initialize_storages(self.db_manager)
                logger.info("SQLite хранилище инициализировано")
            else:
                logger.error("Не удалось инициализировать SQLite хранилище")

            # ChromaDB
            vector_store_module.initialize_vector_store(settings.chromadb_persist_dir)
            vector_store = vector_store_module.vector_store
            if vector_store and vector_store.initialize_collections():
                logger.info("ChromaDB хранилище инициализировано")
            else:
                logger.error("Не удалось инициализировать ChromaDB хранилище")

            # Memgraph
            graph_store_module.initialize_graph_store(
                settings.memgraph_uri,
                settings.memgraph_user,
                settings.memgraph_password,
            )
            graph_store = graph_store_module.graph_store
            if graph_store and graph_store.driver and graph_store.initialize_graph():
                logger.info("Memgraph хранилище инициализировано")
            else:
                logger.warning(
                    "Memgraph хранилище недоступно или не инициализировано"
                )

        except Exception as e:
            logger.error(f"Ошибка инициализации хранилищ: {e}")

    async def initialize(self) -> bool:
        """
        Асинхронная инициализация сервиса синхронизации.

        Returns:
            bool: True если инициализация успешна, False иначе
        """
        try:
            logger.info("Инициализация сервиса синхронизации...")

            # Инициализируем MT5 клиент
            if not self.mt5_client.initialize():
                logger.error("Не удалось инициализировать MT5 клиент")
                return False

            # Инициализируем базы данных
            self._initialize_storages()

            # Проверяем здоровье всех систем
            health_status = await self.check_system_health()

            # Система готова к работе, даже если некоторые компоненты имеют предупреждения
            if health_status["status"] in ["healthy", "warning"]:
                logger.info("Системы синхронизации готовы к работе")
                return True
            else:
                logger.error(f"Критические проблемы со здоровьем системы: {health_status}")
                return False

        except Exception as e:
            logger.error(f"Ошибка инициализации сервиса синхронизации: {e}")
            return False

    async def check_system_health(self) -> Dict[str, Any]:
        """
        Проверяет здоровье всех компонентов системы.

        Returns:
            Dict[str, Any]: Статус здоровья системы
        """
        health = {"status": "healthy", "components": {}}

        try:
            # Проверяем MT5 клиент
            if self.mt5_client.initialized:
                # Проверяем подключение к терминалу
                try:
                    account_info = self.mt5_client.get_account_info()
                    health["components"]["mt5"] = {
                        "status": "ok",
                        "connected": True,
                        "account": bool(account_info),
                    }
                except Exception as e:
                    health["components"]["mt5"] = {
                        "status": "error",
                        "error": str(e),
                    }
            else:
                health["components"]["mt5"] = {
                    "status": "disconnected",
                    "error": "MT5 клиент не инициализирован",
                }

            # Проверяем базы данных
            try:
                db_health = self.db_manager.check_database_health()
                health["components"]["sqlite"] = db_health
            except Exception as e:
                health["components"]["sqlite"] = {"status": "error", "error": str(e)}

            # Проверяем ChromaDB
            try:
                vector_store = vector_store_module.vector_store

                if vector_store:
                    chroma_stats = vector_store.get_collection_stats()
                    health["components"]["chromadb"] = {
                        "status": "ok",
                        "collections": chroma_stats,
                    }
                else:
                    health["components"]["chromadb"] = {
                        "status": "unavailable",
                        "error": "Векторное хранилище не инициализировано",
                    }
            except Exception as e:
                health["components"]["chromadb"] = {"status": "error", "error": str(e)}

            # Проверяем Memgraph
            try:
                graph_store = graph_store_module.graph_store

                if graph_store and graph_store.driver:
                    health["components"]["memgraph"] = {
                        "status": "ok",
                        "connected": True,
                    }
                else:
                    health["components"]["memgraph"] = {
                        "status": "disconnected",
                        "connected": False,
                    }
            except Exception as e:
                health["components"]["memgraph"] = {"status": "error", "error": str(e)}

            # Определяем общий статус
            component_statuses = [
                comp["status"] for comp in health["components"].values()
            ]
            if "error" in component_statuses:
                health["status"] = "critical"
            elif (
                "unavailable" in component_statuses
                or "disconnected" in component_statuses
            ):
                health["status"] = "warning"
            else:
                health["status"] = "healthy"

        except Exception as e:
            health["status"] = "error"
            health["error"] = str(e)

        return health

    async def sync_quotes(
        self, symbols: List[str], timeframe: str = "M1"
    ) -> Dict[str, Any]:
        """
        Синхронизирует котировки для указанных символов.

        Args:
            symbols: Список символов для синхронизации
            timeframe: Таймфрейм котировок

        Returns:
            Dict[str, Any]: Результат синхронизации
        """
        if not self.mt5_client.initialized:
            return {"success": False, "error": "MT5 клиент не инициализирован"}

        result = {
            "success": True,
            "synced_quotes": 0,
            "errors": [],
            "symbols_processed": [],
        }

        try:
            # Получаем котировки через MT5 клиент
            raw_quotes = self.mt5_client.get_quotes(symbols)

            if not raw_quotes:
                result["errors"].append("Не удалось получить котировки ни для одного символа")
                return result

            # Обрабатываем котировки через процессор данных
            processed_quotes = self.data_processor.process_quotes_batch(raw_quotes)

            # Фильтруем дубликаты и старые котировки
            filtered_quotes = self.data_processor.filter_duplicate_quotes(processed_quotes)
            filtered_quotes = self.data_processor.filter_recent_quotes(filtered_quotes)

            # Сохраняем в SQLite
            quote_storage = storage_module.quote_storage

            for quote in filtered_quotes:
                try:
                    if quote_storage and quote_storage.store_quote(quote):
                        result["synced_quotes"] += 1
                        result["symbols_processed"].append(quote.symbol)

                        # Создаем и сохраняем эмбеддинг в ChromaDB
                        await self._store_quote_embedding(quote.model_dump())

                        # Сохраняем данные в графовое хранилище
                        self._store_quote_in_graph(quote)

                        logger.debug(f"Котировка {quote.symbol} синхронизирована")
                    else:
                        result["errors"].append(
                            f"Ошибка сохранения котировки {quote.symbol} в SQLite"
                        )

                except Exception as e:
                    error_msg = f"Ошибка сохранения котировки {quote.symbol}: {e}"
                    logger.error(error_msg)
                    result["errors"].append(error_msg)

        except Exception as e:
            error_msg = f"Критическая ошибка синхронизации котировок: {e}"
            logger.error(error_msg)
            result["success"] = False
            result["errors"].append(error_msg)

        return result

    async def sync_trades(self) -> Dict[str, Any]:
        """
        Синхронизирует историю сделок из MT5.

        Returns:
            Dict[str, Any]: Результат синхронизации
        """
        if not self.mt5_client.initialized:
            return {"success": False, "error": "MT5 клиент не инициализирован"}

        result = {"success": True, "synced_trades": 0, "errors": []}

        try:
            # Получаем историю сделок за последние 24 часа
            from_date = datetime.now() - timedelta(days=1)
            to_date = datetime.now()

            raw_trades = self.mt5_client.get_trades_history(from_date, to_date)

            if not raw_trades:
                result["errors"].append("Не удалось получить историю сделок")
                return result

            # Обрабатываем сделки через процессор данных
            processed_trades = self.data_processor.process_trades_batch(raw_trades)

            # Сохраняем в SQLite
            trade_storage = storage_module.trade_storage

            for trade in processed_trades:
                try:
                    trade_id = trade.id or f"trade_{int(datetime.now().timestamp())}_{trade.symbol}"
                    if trade.id != trade_id:
                        trade.id = trade_id

                    if trade_storage and trade_storage.store_trade(trade):
                        result["synced_trades"] += 1

                        # Создаем и сохраняем эмбеддинг в ChromaDB
                        await self._store_trade_embedding(trade.model_dump())

                        # Сохраняем данные в графовое хранилище
                        self._store_trade_in_graph(trade)

                        logger.debug(f"Сделка {trade.id} синхронизирована")
                    else:
                        result["errors"].append(
                            f"Ошибка сохранения сделки {trade.id}"
                        )

                except Exception as e:
                    error_msg = f"Ошибка обработки сделки {trade.id}: {e}"
                    logger.error(error_msg)
                    result["errors"].append(error_msg)

        except Exception as e:
            error_msg = f"Критическая ошибка синхронизации сделок: {e}"
            logger.error(error_msg)
            result["success"] = False
            result["errors"].append(error_msg)

        return result

    def _get_mt5_timeframe(self, timeframe: str) -> int:
        """
        Преобразует строковый таймфрейм в константу MT5.

        Args:
            timeframe: Строковый таймфрейм (M1, M5, H1 и т.д.)

        Returns:
            int: Константа MT5 для таймфрейма
        """
        if mt5 is None:
            fallback_map = {
                "M1": 1,
                "M5": 5,
                "M15": 15,
                "H1": 60,
                "H4": 240,
                "D1": 1440,
            }
            logger.warning(
                "MetaTrader5 недоступен, используются резервные значения таймфреймов"
            )
            return fallback_map.get(timeframe, fallback_map["M1"])

        timeframe_map = {
            "M1": getattr(mt5, "TIMEFRAME_M1"),
            "M5": getattr(mt5, "TIMEFRAME_M5"),
            "M15": getattr(mt5, "TIMEFRAME_M15"),
            "H1": getattr(mt5, "TIMEFRAME_H1"),
            "H4": getattr(mt5, "TIMEFRAME_H4"),
            "D1": getattr(mt5, "TIMEFRAME_D1"),
        }

        default_timeframe = getattr(mt5, "TIMEFRAME_M1")
        return timeframe_map.get(timeframe, default_timeframe)

    async def _store_quote_embedding(self, quote_data: Dict[str, Any]) -> None:
        """
        Сохраняет эмбеддинг котировки в векторную базу данных.

        Args:
            quote_data: Данные котировки
        """
        try:
            vector_store = vector_store_module.vector_store

            if vector_store:
                # Создаем эмбеддинг через сервис эмбеддингов
                embedding = self.embedding_service.create_quote_embedding(quote_data)

                # Сохраняем в ChromaDB
                vector_store.store_quote_embedding(
                    quote_data["symbol"],
                    quote_data["timestamp"],
                    embedding,
                    quote_data,
                )

        except Exception as e:
            logger.error(f"Ошибка сохранения эмбеддинга котировки: {e}")

    async def _store_trade_embedding(self, trade_data: Dict[str, Any]) -> None:
        """
        Сохраняет эмбеддинг сделки в векторную базу данных.

        Args:
            trade_data: Данные сделки
        """
        try:
            vector_store = vector_store_module.vector_store

            if vector_store:
                # Создаем эмбеддинг через сервис эмбеддингов
                embedding = self.embedding_service.create_trade_embedding(trade_data)

                # Сохраняем в ChromaDB
                vector_store.store_trade_embedding(
                    trade_data.get("id", f"trade_{int(time.time())}"),
                    embedding,
                    trade_data,
                )

        except Exception as e:
            logger.error(f"Ошибка сохранения эмбеддинга сделки: {e}")

    def _store_quote_in_graph(self, quote: Quote) -> None:
        """Сохраняет котировку в Memgraph, если доступен GraphStore."""

        graph_store = graph_store_module.graph_store
        if not graph_store or not graph_store.driver:
            return

        try:
            graph_store.store_instrument(
                quote.symbol,
                {"last_quote_timestamp": quote.timestamp},
            )
            graph_store.store_quote(
                {
                    "id": f"quote_{quote.symbol}_{quote.timestamp}",
                    "symbol": quote.symbol,
                    "timestamp": quote.timestamp,
                    "bid": quote.bid,
                    "ask": quote.ask,
                    "volume": quote.volume or 0,
                }
            )
        except Exception as e:
            logger.error(f"Ошибка сохранения котировки в граф: {e}")

    def _store_trade_in_graph(self, trade: Trade) -> None:
        """Сохраняет сделку в Memgraph, если доступен GraphStore."""

        graph_store = graph_store_module.graph_store
        if not graph_store or not graph_store.driver:
            return

        try:
            opened_at_dt = (
                trade.opened_at if isinstance(trade.opened_at, datetime) else None
            )
            last_trade_ts = (
                int(opened_at_dt.timestamp()) if opened_at_dt else int(time.time())
            )

            graph_store.store_instrument(
                trade.symbol,
                {"last_trade_timestamp": last_trade_ts},
            )

            trade_data = {
                "id": trade.id
                or f"trade_{int(datetime.now().timestamp())}_{trade.symbol}",
                "symbol": trade.symbol,
                "side": getattr(trade.side, "value", trade.side),
                "entry_price": trade.entry_price,
                "quantity": trade.quantity,
                "status": trade.status,
                "pnl": trade.pnl if trade.pnl is not None else 0.0,
                "opened_at": (
                    opened_at_dt.isoformat()
                    if opened_at_dt
                    else (
                        trade.opened_at
                        if isinstance(trade.opened_at, str)
                        else datetime.utcnow().isoformat()
                    )
                ),
            }

            graph_store.store_trade(trade_data)

            if getattr(trade, "signal_id", None):
                graph_store.create_trade_signal_relationship(
                    trade_data["id"], trade.signal_id
                )
        except Exception as e:
            logger.error(f"Ошибка сохранения сделки в граф: {e}")

    async def start_continuous_sync(
        self, symbols: List[str], interval: int = 60
    ) -> None:
        """
        Запускает непрерывную синхронизацию данных.

        Args:
            symbols: Список символов для синхронизации
            interval: Интервал синхронизации в секундах
        """
        self.is_running = True
        logger.info(f"Запуск непрерывной синхронизации для символов: {symbols}")

        try:
            while self.is_running:
                try:
                    # Синхронизируем котировки
                    quotes_result = await self.sync_quotes(symbols)
                    if quotes_result["success"]:
                        logger.info(
                            f"Синхронизировано котировок: {quotes_result['synced_quotes']}"
                        )
                    else:
                        logger.error(
                            f"Ошибка синхронизации котировок: {quotes_result['errors']}"
                        )

                    # Синхронизируем сделки
                    trades_result = await self.sync_trades()
                    if trades_result["success"]:
                        logger.info(
                            f"Синхронизировано сделок: {trades_result['synced_trades']}"
                        )
                    else:
                        logger.error(
                            f"Ошибка синхронизации сделок: {trades_result['errors']}"
                        )

                    self.last_sync_time = datetime.now()

                    # Ждем следующий интервал
                    await asyncio.sleep(interval)

                except asyncio.CancelledError:
                    logger.info("Синхронизация отменена пользователем")
                    break
                except Exception as e:
                    logger.error(f"Ошибка в цикле синхронизации: {e}")
                    await asyncio.sleep(interval)  # Продолжаем даже при ошибке

        finally:
            self.is_running = False
            logger.info("Непрерывная синхронизация остановлена")

    def stop_sync(self) -> None:
        """
        Останавливает непрерывную синхронизацию.
        """
        if self.is_running:
            logger.info("Запрос на остановку синхронизации")
            self.is_running = False

    async def get_sync_status(self) -> Dict[str, Any]:
        """
        Получает текущий статус синхронизации.

        Returns:
            Dict[str, Any]: Статус синхронизации
        """
        return {
            "is_running": self.is_running,
            "last_sync_time": (
                self.last_sync_time.isoformat() if self.last_sync_time else None
            ),
            "sync_errors": len(self.sync_errors),
            "system_health": await self.check_system_health(),
        }

    async def close(self) -> None:
        """
        Закрывает сервис синхронизации и освобождает ресурсы.
        """
        logger.info("Закрытие сервиса синхронизации...")

        # Останавливаем синхронизацию
        self.stop_sync()

        # Закрываем MT5 клиент
        try:
            self.mt5_client.shutdown()
            logger.info("MT5 клиент закрыт")
        except Exception as e:
            logger.error(f"Ошибка закрытия MT5 клиента: {e}")

        # Закрываем графовое хранилище
        try:
            graph_store = graph_store_module.graph_store

            if graph_store:
                graph_store.close()
                logger.info("Memgraph соединение закрыто")
        except Exception as e:
            logger.error(f"Ошибка закрытия Memgraph: {e}")

        logger.info("Сервис синхронизации закрыт")


def create_sync_service() -> SyncService:
    """
    Создает экземпляр сервиса синхронизации.

    Returns:
        SyncService: Экземпляр сервиса синхронизации
    """
    return SyncService()


# Глобальный экземпляр для использования в приложении
sync_service = None


def initialize_sync_service() -> None:
    """
    Инициализирует глобальный экземпляр сервиса синхронизации.
    """
    global sync_service
    sync_service = create_sync_service()
