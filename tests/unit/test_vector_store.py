#!/usr/bin/env python3
"""
Модульные тесты для векторного хранилища ChromaDB.

Тестирует операции с векторной базой данных.
"""

import tempfile
from unittest.mock import Mock

import pytest

from src.database.vector_store import create_vector_store


@pytest.fixture
def temp_chroma_dir():
    """Создает временную директорию для ChromaDB."""
    import os
    import shutil

    temp_dir = tempfile.mkdtemp(prefix="chroma_test_")
    yield temp_dir

    # Очищаем директорию вручную из-за проблем с Windows
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        # Игнорируем ошибки очистки в Windows
        pass


@pytest.fixture
def vector_store(temp_chroma_dir):
    """Создает векторное хранилище для тестов."""
    return create_vector_store(temp_chroma_dir)


class TestVectorStore:
    """Тесты векторного хранилища."""

    def test_vector_store_creation(self, vector_store):
        """Тест создания векторного хранилища."""
        assert vector_store is not None
        assert vector_store.collection_name == "ats_data"
        # persist_dir может быть любым в тесте, главное что он не None
        assert vector_store.persist_dir is not None

    def test_initialize_collections(self, vector_store):
        """Тест инициализации коллекций."""
        assert vector_store.initialize_collections()

        # Проверяем, что коллекции созданы
        stats = vector_store.get_collection_stats()
        expected_collections = {"quotes", "trades", "signals", "indicators"}

        assert set(stats.keys()) == expected_collections
        for collection_name in expected_collections:
            assert stats[collection_name]["status"] == "active"
            assert stats[collection_name]["count"] == 0

    def test_store_quote_embedding(self, vector_store):
        """Тест сохранения эмбеддинга котировки."""
        # Инициализируем коллекции
        assert vector_store.initialize_collections()

        # Тестовые данные
        symbol = "EURUSD"
        timestamp = 1640995200
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        metadata = {"price": 1.1234, "volume": 1000000}

        # Сохраняем эмбеддинг
        assert vector_store.store_quote_embedding(
            symbol, timestamp, embedding, metadata
        )

        # Проверяем статистику
        stats = vector_store.get_collection_stats()
        assert stats["quotes"]["count"] == 1

    def test_store_trade_embedding(self, vector_store):
        """Тест сохранения эмбеддинга сделки."""
        # Инициализируем коллекции
        assert vector_store.initialize_collections()

        # Тестовые данные
        trade_id = "trade_001"
        embedding = [0.5, 0.4, 0.3, 0.2, 0.1]
        metadata = {"symbol": "EURUSD", "pnl": 50.0}

        # Сохраняем эмбеддинг
        assert vector_store.store_trade_embedding(trade_id, embedding, metadata)

        # Проверяем статистику
        stats = vector_store.get_collection_stats()
        assert stats["trades"]["count"] == 1

    def test_store_signal_embedding(self, vector_store):
        """Тест сохранения эмбеддинга сигнала."""
        # Инициализируем коллекции
        assert vector_store.initialize_collections()

        # Тестовые данные
        signal_id = "signal_001"
        embedding = [0.3, 0.6, 0.2, 0.8, 0.1]
        metadata = {"symbol": "EURUSD", "confidence": 0.75}

        # Сохраняем эмбеддинг
        assert vector_store.store_signal_embedding(signal_id, embedding, metadata)

        # Проверяем статистику
        stats = vector_store.get_collection_stats()
        assert stats["signals"]["count"] == 1

    def test_search_similar_quotes(self, vector_store):
        """Тест поиска похожих котировок."""
        # Инициализируем коллекции
        assert vector_store.initialize_collections()

        # Добавляем тестовые данные
        embeddings_data = [
            ("EURUSD", 1640995200, [0.1, 0.2, 0.3, 0.4, 0.5]),
            ("EURUSD", 1640995260, [0.15, 0.25, 0.35, 0.45, 0.55]),
            ("GBPUSD", 1640995200, [0.8, 0.7, 0.6, 0.5, 0.4]),
        ]

        for symbol, timestamp, embedding in embeddings_data:
            vector_store.store_quote_embedding(symbol, timestamp, embedding)

        # Ищем похожие котировки
        query_embedding = [0.12, 0.22, 0.32, 0.42, 0.52]
        similar_quotes = vector_store.search_similar_quotes(
            query_embedding, n_results=2
        )

        assert len(similar_quotes) == 2
        # Первая котировка должна быть наиболее похожей
        assert "EURUSD" in similar_quotes[0]["id"]

    def test_clear_collection(self, vector_store):
        """Тест очистки коллекции."""
        # Инициализируем коллекции
        assert vector_store.initialize_collections()

        # Добавляем данные (теперь метаданные обязательны)
        vector_store.store_quote_embedding("EURUSD", 1640995200, [0.1, 0.2, 0.3])
        vector_store.store_trade_embedding("trade_001", [0.5, 0.4, 0.3])

        # Проверяем, что данные есть
        stats = vector_store.get_collection_stats()
        assert stats["quotes"]["count"] == 1
        assert stats["trades"]["count"] == 1

        # Очищаем коллекцию котировок
        assert vector_store.clear_collection("quotes")

        # Проверяем, что коллекция очищена
        stats = vector_store.get_collection_stats()
        assert stats["quotes"]["count"] == 0
        assert stats["trades"]["count"] == 1  # Другие коллекции не затронуты

    def test_reset_database(self, vector_store):
        """Тест сброса базы данных."""
        # Инициализируем коллекции
        assert vector_store.initialize_collections()

        # Добавляем данные
        vector_store.store_quote_embedding("EURUSD", 1640995200, [0.1, 0.2, 0.3])

        # Сбрасываем базу данных
        assert vector_store.reset_database()

        # Проверяем, что все коллекции пустые
        stats = vector_store.get_collection_stats()
        for collection_name in ["quotes", "trades", "signals", "indicators"]:
            assert stats[collection_name]["count"] == 0


class TestGraphStore:
    """Тесты графового хранилища."""

    def test_graph_store_creation(self):
        """Тест создания графового хранилища."""
        from storage.memgraph.graph_store import GraphStore

        # Создаем хранилище без подключения к Memgraph (для теста)
        store = GraphStore()

        assert store.uri == "bolt://localhost:7687"
        assert store.user == "memgraph"
        assert store.password == "memgraph"
        assert store.driver is None  # Нет подключения

    def test_graph_store_with_mock_driver(self):
        """Тест графового хранилища с мок-драйвером."""
        from storage.memgraph.graph_store import GraphStore

        # Создаем мок-драйвер
        mock_driver = Mock()
        mock_driver.execute.return_value = []
        mock_driver.close.return_value = None

        store = GraphStore()
        store.driver = mock_driver

        # Тестируем инициализацию графа
        assert store.initialize_graph()

        # Проверяем, что были вызваны методы создания ограничений и индексов
        assert mock_driver.execute.call_count >= 5  # Минимум 5 операций

        # Тестируем сохранение инструмента
        assert store.store_instrument("EURUSD", {"sector": "forex"})

        # Тестируем сохранение котировки
        quote_data = {
            "id": "quote_001",
            "symbol": "EURUSD",
            "timestamp": 1640995200,
            "bid": 1.1234,
            "ask": 1.1236,
            "volume": 1000000,
        }
        assert store.store_quote(quote_data)

        # Тестируем сохранение сделки
        trade_data = {
            "id": "trade_001",
            "symbol": "EURUSD",
            "side": "buy",
            "entry_price": 1.1234,
            "quantity": 0.1,
            "status": "open",
            "pnl": None,
            "opened_at": 1640995200,
        }
        assert store.store_trade(trade_data)

        # Тестируем создание связи
        assert store.create_trade_signal_relationship("trade_001", "signal_001")

        # Тестируем закрытие соединения
        store.close()
        mock_driver.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
