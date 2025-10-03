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
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.config import settings
from src.database.vector_store import initialize_vector_store
from src.database.graph_store import initialize_graph_store
from src.database.init_db import create_database_manager
from src.database.storage import initialize_storages

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
                initialize_storages(self.db_manager)
                logger.info("SQLite хранилище инициализировано")
            else:
                logger.error("Не удалось инициализировать SQLite хранилище")

            # ChromaDB
            initialize_vector_store(settings.chromadb_persist_dir)
            from storage.chromadb.vector_store import vector_store
            if vector_store and vector_store.initialize_collections():
                logger.info("ChromaDB хранилище инициализировано")
            else:
                logger.error("Не удалось инициализировать ChromaDB хранилище")

            # Memgraph
            initialize_graph_store(
                settings.memgraph_uri,
                settings.memgraph_user,
                settings.memgraph_password,
            )
            logger.info("Memgraph хранилище инициализировано")

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
            from src.database.vector_store import vector_store

            if vector_store:
                chroma_stats = vector_store.get_collection_stats()
                health["components"]["chromadb"] = {
                    "status": "ok",
                    "collections": chroma_stats,
                }
                else:
                    health["components"]["chromadb"] = {
                        "status": "error",
                        "error": "Векторное хранилище не инициализировано",
                    }
            except Exception as e:
                health["components"]["chromadb"] = {"status": "error", "error": str(e)}

            # Проверяем Memgraph
            try:
                from storage.memgraph.graph_store import graph_store

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
            from src.database.storage import quote_storage

            for quote in filtered_quotes:
                try:
                    if quote_storage and quote_storage.store_quote(quote):
                        result["synced_quotes"] += 1
                        result["symbols_processed"].append(quote.symbol)

                        # Создаем и сохраняем эмбеддинг в ChromaDB
                        await self._store_quote_embedding(quote.model_dump())

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
            from src.database.storage import trade_storage

            for trade in processed_trades:
                try:
                    if trade_storage and trade_storage.store_trade(trade):
                        result["synced_trades"] += 1

                        # Создаем и сохраняем эмбеддинг в ChromaDB
                        await self._store_trade_embedding(trade.model_dump())

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
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

        return timeframe_map.get(timeframe, mt5.TIMEFRAME_M1)

    async def _store_quote_embedding(self, quote_data: Dict[str, Any]) -> None:
        """
        Сохраняет эмбеддинг котировки в векторную базу данных.

        Args:
            quote_data: Данные котировки
        """
        try:
            from src.database.vector_store import vector_store

            if vector_store:
                # Создаем эмбеддинг через сервис эмбеддингов
                embedding = self.embedding_service.create_quote_embedding(quote_data)

                # Создаем текстовое представление для полнотекстового поиска
                text = f"Quote: {quote_data['symbol']} at {quote_data['timestamp']}, bid={quote_data['bid']}, ask={quote_data['ask']}"

                # Сохраняем в ChromaDB
                vector_store.store_quote_embedding(
                    quote_data["symbol"],
                    quote_data["timestamp"],
                    embedding,
                    quote_data
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
            from src.database.vector_store import vector_store

            if vector_store:
                # Создаем эмбеддинг через сервис эмбеддингов
                embedding = self.embedding_service.create_trade_embedding(trade_data)

                # Создаем текстовое представление для полнотекстового поиска
                text = f"Trade: {trade_data.get('symbol', 'UNKNOWN')} {trade_data.get('side', 'UNKNOWN')} at {trade_data.get('entry_price', 0)}"

                # Сохраняем в ChromaDB
                vector_store.store_trade_embedding(
                    trade_data.get("id", f"trade_{int(time.time())}"),
                    embedding,
                    trade_data
                )

        except Exception as e:
            logger.error(f"Ошибка сохранения эмбеддинга сделки: {e}")

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
            from storage.memgraph.graph_store import graph_store

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
