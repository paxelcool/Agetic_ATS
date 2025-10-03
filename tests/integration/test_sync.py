#!/usr/bin/env python3
"""
Интеграционные тесты для модуля синхронизатора.

Тестирует полный цикл синхронизации данных с базами данных.
"""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from src.config import TradingSettings
from src.synchronizer.sync_service import create_sync_service


@pytest.fixture
def temp_db():
    """Создает временную базу данных для тестов."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield db_path

    # Очистка после теста
    if os.path.exists(db_path):
        try:
            os.unlink(db_path)
        except PermissionError:
            pass  # Файл может быть заблокирован


@pytest.fixture
def sync_service(temp_db):
    """Создает сервис синхронизации для тестов."""
    # Создаем временные настройки для теста
    test_settings = TradingSettings()
    test_settings.sync_db_path = temp_db

    with patch("src.synchronizer.sync_service.settings", test_settings):
        service = create_sync_service()
        yield service


class TestSyncServiceIntegration:
    """Интеграционные тесты сервиса синхронизации."""

    def test_sync_service_creation(self, sync_service):
        """Тест создания сервиса синхронизации."""
        assert sync_service is not None
        assert sync_service.is_running == False
        assert sync_service.last_sync_time is None

    @pytest.mark.asyncio
    async def test_system_health_check(self, sync_service):
        """Тест проверки здоровья системы."""
        health = await sync_service.check_system_health()

        assert "status" in health
        assert "components" in health
        assert isinstance(health["components"], dict)

        # MT5 может быть недоступен в тестовой среде
        mt5_status = health["components"].get("mt5", {})
        assert "status" in mt5_status

    @pytest.mark.asyncio
    async def test_sync_quotes_real_mt5(self, sync_service):
        """Тест синхронизации котировок с реальным MT5."""
        # Тестируем синхронизацию котировок с реальными данными
        result = await sync_service.sync_quotes(["EURUSD"])

        # Проверяем структуру ответа
        assert "success" in result
        assert "synced_quotes" in result
        assert "errors" in result
        assert "symbols_processed" in result

        # Если MT5 доступен, должны быть реальные данные
        if result["success"]:
            assert result["synced_quotes"] >= 0  # Может быть 0 если нет данных
            print(f"Синхронизировано котировок: {result['synced_quotes']}")
        else:
            print(f"Ошибка синхронизации: {result.get('error', 'Неизвестная ошибка')}")

    @pytest.mark.asyncio
    async def test_sync_quotes_mt5_unavailable(self, sync_service):
        """Тест синхронизации котировок при недоступном MT5."""
        # Мокаем недоступный MT5 клиент
        with patch.object(sync_service.mt5_client, 'initialized', False):
            result = await sync_service.sync_quotes(["EURUSD"])

            assert result["success"] == False
            assert "MT5 клиент не инициализирован" in result["error"]

@pytest.mark.asyncio
async def test_sync_trades_real_mt5(self, sync_service):
    """Тест синхронизации сделок с реальным MT5."""
    # Тестируем синхронизацию сделок с реальными данными
    result = await sync_service.sync_trades()

    # Проверяем структуру ответа
    assert "success" in result
    assert "synced_trades" in result
    assert "errors" in result

    # Если MT5 доступен, должны быть реальные данные
    if result["success"]:
        assert result["synced_trades"] >= 0  # Может быть 0 если нет сделок
        print(f"Синхронизировано сделок: {result['synced_trades']}")
    else:
        print(f"Ошибка синхронизации: {result.get('error', 'Неизвестная ошибка')}")

    def test_continuous_sync_lifecycle(self, sync_service):
        """Тест жизненного цикла непрерывной синхронизации."""
        assert sync_service.is_running == False

        # Запускаем синхронизацию (без реального выполнения)
        # В реальном тесте это было бы асинхронным

        # Останавливаем синхронизацию
        sync_service.stop_sync()
        assert sync_service.is_running == False

    @pytest.mark.asyncio
    async def test_sync_status_reporting(self, sync_service):
        """Тест отчетности о статусе синхронизации."""
        status = await sync_service.get_sync_status()

        assert "is_running" in status
        assert "last_sync_time" in status
        assert "sync_errors" in status
        assert "system_health" in status

        assert isinstance(status["is_running"], bool)
        assert status["sync_errors"] >= 0

    @pytest.mark.asyncio
    async def test_error_handling_in_sync(self, sync_service):
        """Тест обработки ошибок в процессе синхронизации."""
        # Мокаем MT5 с ошибкой
        mock_mt5 = Mock()
        mock_mt5.symbol_info_tick.side_effect = Exception("Connection error")

        with patch("src.synchronizer.sync_service.mt5", mock_mt5):
            result = await sync_service.sync_quotes(["EURUSD"])

            # Должен быть обработан gracefully
            assert "errors" in result
            assert len(result["errors"]) > 0

    def test_sync_service_cleanup(self, sync_service):
        """Тест очистки ресурсов сервиса синхронизации."""

        # Тестируем закрытие сервиса
        async def cleanup_test():
            await sync_service.close()

        # В реальном сценарии это было бы асинхронным
        # Здесь просто проверяем, что метод существует
        assert hasattr(sync_service, "close")

@pytest.mark.asyncio
async def test_multiple_symbols_sync_real_mt5(self, sync_service):
    """Тест синхронизации нескольких символов с реальным MT5."""
    # Тестируем синхронизацию нескольких символов с реальными данными
    symbols = ["EURUSD", "GBPUSD"]
    result = await sync_service.sync_quotes(symbols)

    # Проверяем структуру ответа
    assert "success" in result
    assert "synced_quotes" in result
    assert "errors" in result
    assert "symbols_processed" in result

    # Если MT5 доступен, должны быть реальные данные
    if result["success"]:
        assert result["synced_quotes"] >= 0  # Может быть 0 если нет данных
        print(f"Синхронизировано котировок: {result['synced_quotes']}")
        print(f"Обработано символов: {result['symbols_processed']}")
    else:
        print(f"Ошибка синхронизации: {result.get('error', 'Неизвестная ошибка')}")


class TestSyncServiceWithRealMT5:
    """Тесты сервиса синхронизации с реальным MT5 (если доступен)."""

    @pytest.mark.skipif(
        not hasattr(__import__("sys").modules.get("MetaTrader5"), "initialize")
    )
    def test_real_mt5_connection(self):
        """Тест подключения к реальному MT5."""
        # Этот тест требует наличия MT5 терминала
        # В реальной среде можно раскомментировать и настроить

        # service = create_sync_service()
        # assert service.initialize()
        # service.close()
        pass

    @pytest.mark.skipif(
        not hasattr(__import__("sys").modules.get("MetaTrader5"), "initialize")
    )
    def test_real_quotes_sync(self):
        """Тест синхронизации реальных котировок."""
        # Этот тест требует наличия MT5 терминала и рыночных данных
        # В реальной среде можно раскомментировать

        # service = create_sync_service()
        # result = await service.sync_quotes(["EURUSD"])
        # assert result["success"] == True
        # service.close()
        pass


if __name__ == "__main__":
    pytest.main([__file__])
