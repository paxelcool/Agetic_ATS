"""
Векторное хранилище на базе ChromaDB для ATS.

Предоставляет функции для хранения и поиска котировок, сделок и сигналов в векторном пространстве.
"""

import logging
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Векторное хранилище на базе ChromaDB.

    Обеспечивает семантический поиск и хранение эмбеддингов данных.
    """

    def __init__(
        self, persist_dir: str = "./chroma_db", collection_name: str = "ats_data"
    ):
        """
        Инициализация векторного хранилища.

        Args:
            persist_dir: Директория для хранения данных
            collection_name: Название коллекции по умолчанию
        """
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(
            path=persist_dir, settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = collection_name

    def initialize_collections(self) -> bool:
        """
        Инициализирует коллекции для разных типов данных.

        Returns:
            bool: True если инициализация успешна, False иначе
        """
        try:
            # Коллекция для котировок
            self.client.get_or_create_collection(
                name="quotes",
                metadata={
                    "description": "Хранение исторических котировок с метаданными"
                },
            )

            # Коллекция для сделок
            self.client.get_or_create_collection(
                name="trades",
                metadata={"description": "История сделок с результатами и контекстом"},
            )

            # Коллекция для сигналов
            self.client.get_or_create_collection(
                name="signals",
                metadata={"description": "Сигналы с объяснениями и исходами"},
            )

            # Коллекция для технических индикаторов
            self.client.get_or_create_collection(
                name="indicators",
                metadata={"description": "Технические индикаторы и паттерны"},
            )

            logger.info("Коллекции ChromaDB инициализированы успешно")
            return True

        except Exception as e:
            logger.error(f"Ошибка инициализации коллекций ChromaDB: {e}")
            return False

    def store_quote_embedding(
        self,
        symbol: str,
        timestamp: int,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Сохраняет эмбеддинг котировки в векторную базу данных.

        Args:
            symbol: Символ инструмента
            timestamp: Временная метка
            embedding: Векторное представление котировки
            metadata: Дополнительные метаданные

        Returns:
            bool: True если сохранение успешно, False иначе
        """
        try:
            collection = self.client.get_or_create_collection(name="quotes")

            # Создаем текстовое представление котировки
            text = f"Quote: {symbol} at {timestamp}"

            # Обеспечиваем непустые метаданные
            quote_metadata = metadata or {}
            quote_metadata.update({
                "symbol": symbol,
                "timestamp": timestamp,
                "type": "quote"
            })

            # Сохраняем в ChromaDB
            collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[quote_metadata],
                ids=[f"quote_{symbol}_{timestamp}"],
            )

            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения эмбеддинга котировки: {e}")
            return False

    def store_trade_embedding(
        self,
        trade_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Сохраняет эмбеддинг сделки в векторную базу данных.

        Args:
            trade_id: Уникальный идентификатор сделки
            embedding: Векторное представление сделки
            metadata: Дополнительные метаданные

        Returns:
            bool: True если сохранение успешно, False иначе
        """
        try:
            collection = self.client.get_or_create_collection(name="trades")

            # Создаем текстовое представление сделки
            text = f"Trade: {trade_id}"

            # Обеспечиваем непустые метаданные
            trade_metadata = metadata or {}
            trade_metadata.update({
                "trade_id": trade_id,
                "type": "trade"
            })

            # Сохраняем в ChromaDB
            collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[trade_metadata],
                ids=[f"trade_{trade_id}"],
            )

            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения эмбеддинга сделки: {e}")
            return False

    def store_signal_embedding(
        self,
        signal_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Сохраняет эмбеддинг сигнала в векторную базу данных.

        Args:
            signal_id: Уникальный идентификатор сигнала
            embedding: Векторное представление сигнала
            metadata: Дополнительные метаданные

        Returns:
            bool: True если сохранение успешно, False иначе
        """
        try:
            collection = self.client.get_or_create_collection(name="signals")

            # Создаем текстовое представление сигнала
            text = f"Signal: {signal_id}"

            # Обеспечиваем непустые метаданные
            signal_metadata = metadata or {}
            signal_metadata.update({
                "signal_id": signal_id,
                "type": "signal"
            })

            # Сохраняем в ChromaDB
            collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[signal_metadata],
                ids=[f"signal_{signal_id}"],
            )

            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения эмбеддинга сигнала: {e}")
            return False

    def search_similar_quotes(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ищет похожие котировки по векторному сходству.

        Args:
            query_embedding: Вектор запроса
            n_results: Количество результатов
            symbol: Фильтр по символу (опционально)

        Returns:
            List[Dict[str, Any]]: Список похожих котировок с метаданными
        """
        try:
            collection = self.client.get_or_create_collection(name="quotes")

            where_clause = {"symbol": symbol} if symbol else None

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
            )

            # Преобразуем результаты в удобный формат
            similar_quotes = []
            for i, (doc, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                similar_quotes.append(
                    {
                        "id": results["ids"][0][i],
                        "document": doc,
                        "metadata": metadata,
                        "distance": distance,
                    }
                )

            return similar_quotes

        except Exception as e:
            logger.error(f"Ошибка поиска похожих котировок: {e}")
            return []

    def search_similar_trades(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ищет похожие сделки по векторному сходству.

        Args:
            query_embedding: Вектор запроса
            n_results: Количество результатов
            symbol: Фильтр по символу (опционально)

        Returns:
            List[Dict[str, Any]]: Список похожих сделок с метаданными
        """
        try:
            collection = self.client.get_or_create_collection(name="trades")

            where_clause = {"symbol": symbol} if symbol else None

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
            )

            # Преобразуем результаты в удобный формат
            similar_trades = []
            for i, (doc, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                similar_trades.append(
                    {
                        "id": results["ids"][0][i],
                        "document": doc,
                        "metadata": metadata,
                        "distance": distance,
                    }
                )

            return similar_trades

        except Exception as e:
            logger.error(f"Ошибка поиска похожих сделок: {e}")
            return []

    def search_similar_signals(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ищет похожие сигналы по векторному сходству.

        Args:
            query_embedding: Вектор запроса
            n_results: Количество результатов
            symbol: Фильтр по символу (опционально)

        Returns:
            List[Dict[str, Any]]: Список похожих сигналов с метаданными
        """
        try:
            collection = self.client.get_or_create_collection(name="signals")

            where_clause = {"symbol": symbol} if symbol else None

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
            )

            # Преобразуем результаты в удобный формат
            similar_signals = []
            for i, (doc, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                similar_signals.append(
                    {
                        "id": results["ids"][0][i],
                        "document": doc,
                        "metadata": metadata,
                        "distance": distance,
                    }
                )

            return similar_signals

        except Exception as e:
            logger.error(f"Ошибка поиска похожих сигналов: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Получает статистику по всем коллекциям.

        Returns:
            Dict[str, Dict[str, Any]]: Статистика по коллекциям
        """
        try:
            stats = {}

            for collection_name in ["quotes", "trades", "signals", "indicators"]:
                try:
                    collection = self.client.get_collection(collection_name)
                    count = collection.count()
                    stats[collection_name] = {"count": count, "status": "active"}
                except Exception as e:
                    stats[collection_name] = {
                        "count": 0,
                        "status": "error",
                        "error": str(e),
                    }

            return stats

        except Exception as e:
            logger.error(f"Ошибка получения статистики коллекций: {e}")
            return {}

    def clear_collection(self, collection_name: str) -> bool:
        """
        Очищает коллекцию.

        Args:
            collection_name: Название коллекции

        Returns:
            bool: True если очистка успешна, False иначе
        """
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Коллекция {collection_name} очищена")
            return True

        except Exception as e:
            logger.error(f"Ошибка очистки коллекции {collection_name}: {e}")
            return False

    def reset_database(self) -> bool:
        """
        Сбрасывает всю базу данных.

        Returns:
            bool: True если сброс успешен, False иначе
        """
        try:
            # Удаляем все коллекции
            for collection_name in ["quotes", "trades", "signals", "indicators"]:
                try:
                    self.client.delete_collection(collection_name)
                except Exception:
                    pass  # Коллекция может не существовать

            # Создаем заново
            return self.initialize_collections()

        except Exception as e:
            logger.error(f"Ошибка сброса базы данных: {e}")
            return False


def create_vector_store(persist_dir: str = "./chroma_db") -> VectorStore:
    """
    Создает экземпляр векторного хранилища.

    Args:
        persist_dir: Директория для хранения данных

    Returns:
        VectorStore: Экземпляр векторного хранилища
    """
    return VectorStore(persist_dir)


# Глобальный экземпляр для использования в приложении
vector_store = None


def initialize_vector_store(persist_dir: str = "./chroma_db") -> None:
    """
    Инициализирует глобальный экземпляр векторного хранилища.

    Args:
        persist_dir: Директория для хранения данных
    """
    global vector_store
    vector_store = create_vector_store(persist_dir)
