#!/usr/bin/env python3
"""
Тест интеграции с MetaTrader5.

Проверяет инициализацию терминала и получение котировок.
"""

import sys
import time
from datetime import datetime, timedelta

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError as e:
    print(f"❌ MetaTrader5 не установлен: {e}")
    MT5_AVAILABLE = False


def test_mt5_initialization():
    """Тестируем инициализацию MT5"""
    if not MT5_AVAILABLE:
        return False

    print("[INIT] Инициализируем MetaTrader5...")
    try:
        if not mt5.initialize():
            print(f"[ERROR] Не удалось инициализировать MT5: {mt5.last_error()}")
            return False

        print("[SUCCESS] MetaTrader5 успешно инициализирован")
        return True

    except Exception as e:
        print(f"[ERROR] Ошибка при инициализации MT5: {e}")
        return False


def test_mt5_version():
    """Получаем версию MT5"""
    if not MT5_AVAILABLE:
        return False

    try:
        version = mt5.version()
        print(f"[VERSION] Версия MetaTrader5: {version}")
        return True
    except Exception as e:
        print(f"[ERROR] Не удалось получить версию MT5: {e}")
        return False


def test_account_info():
    """Получаем информацию о счете"""
    if not MT5_AVAILABLE:
        return False

    try:
        account_info = mt5.account_info()
        if account_info is None:
            print("[WARNING] Не удалось получить информацию о счете (возможно, терминал не запущен)")
            return False

        print("[ACCOUNT] Информация о счете:")
        print(f"   Баланс: {account_info.balance}")
        print(f"   Свободные средства: {account_info.equity}")
        print(f"   Валюта: {account_info.currency}")
        return True

    except Exception as e:
        print(f"[ERROR] Ошибка при получении информации о счете: {e}")
        return False


def test_symbol_info(symbol="EURUSD"):
    """Получаем информацию о символе"""
    if not MT5_AVAILABLE:
        return False

    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"[WARNING] Символ {symbol} не найден")
            return False

        print(f"[SYMBOL] Информация о символе {symbol}:")
        print(f"   Цена Ask: {symbol_info.ask}")
        print(f"   Цена Bid: {symbol_info.bid}")
        print(f"   Спред: {symbol_info.spread}")
        print(f"   Объем: {symbol_info.volume_min} - {symbol_info.volume_max}")
        return True

    except Exception as e:
        print(f"[ERROR] Ошибка при получении информации о символе: {e}")
        return False


def test_tick_data(symbol="EURUSD"):
    """Получаем тиковые данные"""
    if not MT5_AVAILABLE:
        return False

    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"[WARNING] Не удалось получить тиковые данные для {symbol}")
            return False

        print(f"[TICK] Тиковые данные для {symbol}:")
        print(f"   Время: {datetime.fromtimestamp(tick.time)}")
        print(f"   Цена Bid: {tick.bid}")
        print(f"   Цена Ask: {tick.ask}")
        print(f"   Последняя цена: {tick.last}")
        print(f"   Объем: {tick.volume}")
        return True

    except Exception as e:
        print(f"[ERROR] Ошибка при получении тиковых данных: {e}")
        return False


def test_historical_data(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M1, count=100):
    """Получаем исторические данные"""
    if not MT5_AVAILABLE:
        return False

    try:
        # Получаем данные за последние count баров
        utc_from = datetime.now() - timedelta(hours=count//60 + 1)
        rates = mt5.copy_rates_from(symbol, timeframe, utc_from, count)

        if rates is None or len(rates) == 0:
            print(f"[WARNING] Не удалось получить исторические данные для {symbol}")
            return False

        print(f"[HISTORY] Исторические данные для {symbol} ({len(rates)} баров):")
        print(f"   Первый бар: {datetime.fromtimestamp(rates[0][0])} - Open: {rates[0][1]}, Close: {rates[0][4]}")
        print(f"   Последний бар: {datetime.fromtimestamp(rates[-1][0])} - Open: {rates[-1][1]}, Close: {rates[-1][4]}")

        return True

    except Exception as e:
        print(f"[ERROR] Ошибка при получении исторических данных: {e}")
        return False


def main():
    """Основная функция тестирования"""
    print("TESTING: Тестирование интеграции с MetaTrader5")
    print("=" * 50)

    tests = [
        ("Инициализация MT5", test_mt5_initialization),
        ("Версия MT5", test_mt5_version),
        ("Информация о счете", test_account_info),
        ("Информация о символе EURUSD", lambda: test_symbol_info("EURUSD")),
        ("Тиковые данные EURUSD", lambda: test_tick_data("EURUSD")),
        ("Исторические данные EURUSD", lambda: test_historical_data("EURUSD")),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] Критическая ошибка в тесте '{test_name}': {e}")
            results.append((test_name, False))

    # Подводим итоги
    print("\n" + "=" * 50)
    print("[RESULTS] РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "[PASS] Пройден" if result else "[FAIL] Провал"
        print(f"{status}: {test_name}")
        if result:
            passed += 1

    print(f"\n[SUMMARY] Общий результат: {passed}/{total} тестов пройдено")

    if passed == total:
        print("[SUCCESS] Все тесты прошли успешно! MT5 готов к работе.")
    elif passed >= total // 2:
        print("[WARNING] Некоторые тесты провалились. Проверьте настройки MT5 терминала.")
    else:
        print("[ERROR] Большинство тестов провалились. Требуется настройка MT5.")

    # Закрываем соединение
    if MT5_AVAILABLE:
        try:
            mt5.shutdown()
            print("\n[SHUTDOWN] Соединение с MT5 закрыто.")
        except:
            pass

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
