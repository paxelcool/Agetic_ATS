"""
Сервис эмбеддингов для синхронизации.

Преобразует рыночные данные в векторные представления для хранения в ChromaDB.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Сервис для создания эмбеддингов рыночных данных.

    Преобразует котировки, сделки и другие данные в векторные представления.
    """

    def __init__(self):
        """Инициализация сервиса эмбеддингов."""

    def create_quote_embedding(self, quote_data: Dict[str, Any]) -> List[float]:
        """
        Создает векторное представление котировки.

        Args:
            quote_data: Данные котировки

        Returns:
            List[float]: Векторное представление
        """
        try:
            # Нормализуем цену для создания эмбеддинга
            bid = quote_data.get("bid", 0.0)
            ask = quote_data.get("ask", 0.0)
            volume = quote_data.get("volume", 0)
            timestamp = quote_data.get("timestamp", 0)

            # Создаем простой вектор на основе цены и объема
            # В будущем можно использовать более сложные модели эмбеддингов
            embedding = [
                float(bid) / 10000,  # Нормализованная цена bid
                float(ask) / 10000,  # Нормализованная цена ask
                float(ask - bid) / 100,  # Нормализованный спред
                float(volume) / 1000000,  # Нормализованный объем
                float(timestamp) / 1000000000,  # Нормализованное время
            ]

            return embedding

        except Exception as e:
            logger.error(f"Ошибка создания эмбеддинга котировки: {e}")
            return [0.0, 0.0, 0.0, 0.0, 0.0]

    def create_trade_embedding(self, trade_data: Dict[str, Any]) -> List[float]:
        """
        Создает векторное представление сделки.

        Args:
            trade_data: Данные сделки

        Returns:
            List[float]: Векторное представление
        """
        try:
            # Нормализуем данные сделки
            entry_price = trade_data.get("entry_price", 0.0)
            quantity = trade_data.get("quantity", 0.0)
            pnl = trade_data.get("pnl", 0.0)
            side_buy = 1.0 if trade_data.get("side") == "buy" else 0.0

            # Создаем вектор на основе параметров сделки
            embedding = [
                float(entry_price) / 10000,  # Нормализованная цена входа
                float(quantity) / 10,  # Нормализованный объем
                float(pnl) / 1000,  # Нормализованный P&L
                side_buy,  # Направление сделки (buy/sell)
                0.1,  # Заглушка для будущих метрик
            ]

            return embedding

        except Exception as e:
            logger.error(f"Ошибка создания эмбеддинга сделки: {e}")
            return [0.0, 0.0, 0.0, 0.0, 0.0]

    def create_signal_embedding(self, signal_data: Dict[str, Any]) -> List[float]:
        """
        Создает векторное представление сигнала.

        Args:
            signal_data: Данные сигнала

        Returns:
            List[float]: Векторное представление
        """
        try:
            # Нормализуем данные сигнала
            confidence = signal_data.get("confidence", 0.5)
            entry_price = signal_data.get("entry_price", 0.0)
            side_buy = 1.0 if signal_data.get("side") == "buy" else 0.0

            # Создаем вектор на основе параметров сигнала
            embedding = [
                confidence,  # Уверенность сигнала
                float(entry_price) / 10000 if entry_price > 0 else 0.0,  # Нормализованная цена входа
                side_buy,  # Направление сигнала
                0.1,  # Заглушка для будущих метрик
                0.1,  # Заглушка для будущих метрик
            ]

            return embedding

        except Exception as e:
            logger.error(f"Ошибка создания эмбеддинга сигнала: {e}")
            return [0.5, 0.0, 0.0, 0.0, 0.0]

    def batch_create_embeddings(
        self, data_list: List[Dict[str, Any]], data_type: str = "quote"
    ) -> List[List[float]]:
        """
        Создает эмбеддинги для пакета данных.

        Args:
            data_list: Список данных
            data_type: Тип данных (quote, trade, signal)

        Returns:
            List[List[float]]: Список векторных представлений
        """
        embeddings = []

        for data in data_list:
            if data_type == "quote":
                embedding = self.create_quote_embedding(data)
            elif data_type == "trade":
                embedding = self.create_trade_embedding(data)
            elif data_type == "signal":
                embedding = self.create_signal_embedding(data)
            else:
                logger.warning(f"Неизвестный тип данных: {data_type}")
                embedding = [0.0] * 5

            embeddings.append(embedding)

        return embeddings

    def create_text_embedding(self, text: str) -> List[float]:
        """
        Создает простое векторное представление текста.

        Args:
            text: Текст для эмбеддинга

        Returns:
            List[float]: Векторное представление текста
        """
        # Простая хэш-функция для создания вектора из текста
        # В будущем можно использовать настоящие модели эмбеддингов

        import hashlib

        hash_obj = hashlib.md5(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()

        # Преобразуем хэш в вектор из 5 элементов
        embedding = []
        for i in range(5):
            # Берем байты хэша и нормализуем в диапазон [0, 1]
            byte_value = hash_bytes[i % len(hash_bytes)]
            normalized_value = byte_value / 255.0
            embedding.append(normalized_value)

        return embedding
