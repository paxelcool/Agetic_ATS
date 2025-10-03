"""
Обработчик ошибок для синхронизации.

Обеспечивает надежную обработку ошибок и повторные попытки операций.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryConfig:
    """
    Конфигурация для повторных попыток.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
    ):
        """
        Инициализация конфигурации повторных попыток.

        Args:
            max_attempts: Максимальное количество попыток
            base_delay: Базовая задержка в секундах
            max_delay: Максимальная задержка в секундах
            backoff_factor: Коэффициент экспоненциального роста задержки
            jitter: Использовать ли случайную задержку
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter


class SyncError(Exception):
    """
    Базовый класс для ошибок синхронизации.
    """

    def __init__(self, message: str, error_code: str = "SYNC_ERROR"):
        """
        Инициализация ошибки синхронизации.

        Args:
            message: Сообщение об ошибке
            error_code: Код ошибки
        """
        super().__init__(message)
        self.error_code = error_code


class ConnectionError(SyncError):
    """
    Ошибка подключения к внешнему сервису.
    """

    def __init__(self, service: str, message: str = "Ошибка подключения"):
        super().__init__(message, "CONNECTION_ERROR")
        self.service = service


class DataValidationError(SyncError):
    """
    Ошибка валидации данных.
    """

    def __init__(self, message: str = "Ошибка валидации данных"):
        super().__init__(message, "VALIDATION_ERROR")


class RateLimitError(SyncError):
    """
    Ошибка превышения лимита запросов.
    """

    def __init__(self, message: str = "Превышен лимит запросов"):
        super().__init__(message, "RATE_LIMIT_ERROR")


class ErrorHandler:
    """
    Обработчик ошибок для операций синхронизации.

    Обеспечивает повторные попытки, логирование и graceful degradation.
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Инициализация обработчика ошибок.

        Args:
            config: Конфигурация повторных попыток
        """
        self.config = config or RetryConfig()
        self.error_stats = {
            "total_attempts": 0,
            "successful_attempts": 0,
            "failed_attempts": 0,
            "errors_by_type": {},
        }

    async def execute_with_retry(
        self,
        operation: Callable[[], T],
        operation_name: str = "unknown_operation",
        custom_config: Optional[RetryConfig] = None,
    ) -> T:
        """
        Выполняет операцию с повторными попытками.

        Args:
            operation: Функция для выполнения
            operation_name: Название операции для логирования
            custom_config: Кастомная конфигурация повторных попыток

        Returns:
            T: Результат операции

        Raises:
            SyncError: Если все попытки исчерпаны
        """
        config = custom_config or self.config
        last_exception = None

        for attempt in range(1, config.max_attempts + 1):
            try:
                self.error_stats["total_attempts"] += 1

                logger.debug(f"Попытка {attempt}/{config.max_attempts}: {operation_name}")
                result = await operation() if asyncio.iscoroutinefunction(operation) else operation()

                self.error_stats["successful_attempts"] += 1
                logger.debug(f"Операция выполнена успешно: {operation_name}")
                return result

            except Exception as e:
                last_exception = e
                self.error_stats["failed_attempts"] += 1

                # Обновляем статистику ошибок
                error_type = type(e).__name__
                self.error_stats["errors_by_type"][error_type] = (
                    self.error_stats["errors_by_type"].get(error_type, 0) + 1
                )

                if attempt == config.max_attempts:
                    logger.error(
                        f"Все попытки исчерпаны для операции '{operation_name}': {e}"
                    )
                    break

                # Вычисляем задержку перед следующей попыткой
                delay = min(
                    config.base_delay * (config.backoff_factor ** (attempt - 1)),
                    config.max_delay,
                )

                if config.jitter:
                    # Добавляем случайную задержку (±25%)
                    import random
                    jitter_range = delay * 0.25
                    delay += random.uniform(-jitter_range, jitter_range)

                logger.warning(
                    f"Попытка {attempt} провалилась для '{operation_name}': {e}. "
                    f"Повтор через {delay:.1f} сек."
                )

                await asyncio.sleep(delay)

        # Все попытки исчерпаны
        raise SyncError(
            f"Операция '{operation_name}' провалилась после {config.max_attempts} попыток. "
            f"Последняя ошибка: {last_exception}",
            "MAX_RETRIES_EXCEEDED",
        )

    def log_error(self, error: Exception, context: str = "") -> None:
        """
        Логирует ошибку с контекстом.

        Args:
            error: Исключение для логирования
            context: Дополнительный контекст
        """
        error_type = type(error).__name__

        if isinstance(error, SyncError):
            logger.error(f"[{error.error_code}] {error} - Контекст: {context}")
        else:
            logger.error(f"[{error_type}] {error} - Контекст: {context}")

    def get_error_stats(self) -> Dict[str, Any]:
        """
        Получает статистику ошибок.

        Returns:
            Dict[str, Any]: Статистика ошибок
        """
        success_rate = (
            self.error_stats["successful_attempts"] / self.error_stats["total_attempts"]
            if self.error_stats["total_attempts"] > 0
            else 0
        )

        return {
            **self.error_stats,
            "success_rate": success_rate,
        }

    def reset_stats(self) -> None:
        """Сбрасывает статистику ошибок."""
        self.error_stats = {
            "total_attempts": 0,
            "successful_attempts": 0,
            "failed_attempts": 0,
            "errors_by_type": {},
        }
