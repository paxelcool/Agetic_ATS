"""
Основная точка входа приложения ATS (Automated Trading System).

Предоставляет CLI интерфейс для управления системой и запуска компонентов.
"""

import asyncio
import logging
import signal
import sys
from typing import List

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.synchronizer.sync_service import SyncService

# Создаем экземпляр сервиса синхронизации напрямую
sync_service = SyncService()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ats.log"), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)
console = Console()


class ATSCLI:
    """
    CLI интерфейс для управления ATS.

    Предоставляет команды для запуска синхронизации, мониторинга и управления системой.
    """

    def __init__(self):
        """Инициализация CLI интерфейса."""
        self.shutdown_requested = False

    def setup_signal_handlers(self) -> None:
        """Настраивает обработчики сигналов для корректного завершения."""

        def signal_handler(signum, frame):
            console.print("\n[yellow]Получен сигнал завершения...[/yellow]")
            self.shutdown_requested = True
            if sync_service:
                sync_service.stop_sync()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def initialize_system(self) -> bool:
        """
        Инициализирует всю систему ATS.

        Returns:
            bool: True если инициализация успешна, False иначе
        """
        try:
            console.print("[bold blue]Инициализация ATS...[/bold blue]")

            # Сервис синхронизации уже создан выше

            if not await sync_service.initialize():
                console.print("[bold red][ERR] Ошибка инициализации системы[/bold red]")
                return False

            # Проверяем здоровье системы
            health = await sync_service.check_system_health()

            if health["status"] == "healthy":
                console.print("[bold green]Система готова к работе[/bold green]")
                self._display_system_status(health)
                return True
            else:
                console.print(
                    "[bold yellow]Система инициализирована с предупреждениями[/bold yellow]"
                )
                self._display_system_status(health)
                return True

        except Exception as e:
            console.print(
                f"[bold red]Критическая ошибка инициализации: {e}[/bold red]"
            )
            return False

    def _display_system_status(self, health: dict) -> None:
        """Отображает статус системы в красивом формате."""
        table = Table(title="Статус системы")
        table.add_column("Компонент", style="cyan")
        table.add_column("Статус", style="green")
        table.add_column("Детали", style="white")

        for component, status in health["components"].items():
            status_icon = {
                "ok": "[OK]",
                "unavailable": "[WARN]",
                "disconnected": "[DISC]",
                "error": "[ERR]",
            }.get(status["status"], "[?]")

            details = str(status.get("version", status.get("error", "OK")))
            table.add_row(component, f"{status_icon} {status['status']}", details)

        console.print(table)

    async def run_sync_command(
        self, symbols: List[str], continuous: bool = False, interval: int = 60
    ) -> None:
        """
        Запускает синхронизацию данных.

        Args:
            symbols: Список символов для синхронизации
            continuous: Флаг непрерывной синхронизации
            interval: Интервал синхронизации в секундах
        """
        if not sync_service:
            console.print(
                "[bold red][ERR] Сервис синхронизации не инициализирован[/bold red]"
            )
            return

        console.print(
            f"[bold blue]Синхронизация данных для: {', '.join(symbols)}[/bold blue]"
        )

        if continuous:
            console.print(
                f"[yellow]Непрерывная синхронизация с интервалом {interval} сек.[/yellow]"
            )

            # Запускаем непрерывную синхронизацию
            await sync_service.start_continuous_sync(symbols, interval)

            # Ожидаем завершения
            while not self.shutdown_requested and sync_service.is_running:
                await asyncio.sleep(1)

        else:
            # Однократная синхронизация
            quotes_result = await sync_service.sync_quotes(symbols)
            trades_result = await sync_service.sync_trades()

            # Отображаем результаты
            self._display_sync_results(quotes_result, trades_result)

    def _display_sync_results(self, quotes_result: dict, trades_result: dict) -> None:
        """Отображает результаты синхронизации."""
        # Результаты котировок
        quotes_table = Table(title="Синхронизация котировок")
        quotes_table.add_column("Метрика", style="cyan")
        quotes_table.add_column("Значение", style="white")

        quotes_table.add_row(
            "Статус", "[OK] Успешно" if quotes_result["success"] else "[ERR] Ошибка"
        )
        quotes_table.add_row("Синхронизировано", str(quotes_result["synced_quotes"]))
        quotes_table.add_row(
            "Обработано символов", str(len(quotes_result["symbols_processed"]))
        )

        if quotes_result["errors"]:
            quotes_table.add_row("Ошибок", str(len(quotes_result["errors"])))

        console.print(quotes_table)

        # Результаты сделок
        trades_table = Table(title="Синхронизация сделок")
        trades_table.add_column("Метрика", style="cyan")
        trades_table.add_column("Значение", style="white")

        trades_table.add_row(
            "Статус", "[OK] Успешно" if trades_result["success"] else "[ERR] Ошибка"
        )
        trades_table.add_row("Синхронизировано", str(trades_result["synced_trades"]))

        if trades_result["errors"]:
            trades_table.add_row("Ошибок", str(len(trades_result["errors"])))

        console.print(trades_table)

    async def show_status_command(self) -> None:
        """
        Отображает текущий статус системы.
        """
        if not sync_service:
            console.print(
                "[bold red][ERR] Сервис синхронизации не инициализирован[/bold red]"
            )
            return

        status = await sync_service.get_sync_status()

        # Основной статус
        status_panel = Panel(
            f"[bold]Статус синхронизации:[/bold] {'🟢 Запущена' if status['is_running'] else '🔴 Остановлена'}\n"
            f"[bold]Последняя синхронизация:[/bold] {status['last_sync_time'] or 'Никогда'}\n"
            f"[bold]Ошибок синхронизации:[/bold] {status['sync_errors']}",
            title="Синхронизация",
            border_style="blue",
        )
        console.print(status_panel)

        # Детальный статус здоровья
        health = status["system_health"]
        health_panel = Panel(
            f"[bold]Общий статус:[/bold] {health['status']}\n"
            f"[bold]Компонентов:[/bold] {len(health['components'])}",
            title="Здоровье системы",
            border_style="green" if health["status"] == "healthy" else "yellow",
        )
        console.print(health_panel)

    async def run_interactive_mode(self) -> None:
        """
        Запускает интерактивный режим CLI.
        """
        console.print("[bold green]Добро пожаловать в ATS CLI![/bold green]")
        console.print("[yellow]Доступные команды:[/yellow]")
        console.print(
            "  [cyan]sync[/cyan] <символы> [--continuous] [--interval N] - синхронизация данных"
        )
        console.print("  [cyan]status[/cyan] - показать статус системы")
        console.print("  [cyan]health[/cyan] - проверить здоровье системы")
        console.print("  [cyan]quit[/cyan] или [cyan]exit[/cyan] - выход")

        while not self.shutdown_requested:
            try:
                command = await self._get_input_async("ATS> ")

                if command.lower() in ["quit", "exit", "q"]:
                    break
                elif command.lower() == "status":
                    await self.show_status_command()
                elif command.lower() == "health":
                    health = await sync_service.check_system_health()
                    self._display_system_status(health)
                elif command.startswith("sync"):
                    await self._parse_sync_command(command)
                else:
                    console.print(
                        "[yellow]Неизвестная команда. Введите 'help' для справки.[/yellow]"
                    )

            except KeyboardInterrupt:
                console.print("\n[yellow]Используйте 'quit' для выхода[/yellow]")
            except Exception as e:
                console.print(f"[red]Ошибка: {e}[/red]")

    async def _get_input_async(self, prompt: str) -> str:
        """Асинхронный ввод команды."""
        loop = asyncio.get_event_loop()

        # В Windows может потребоваться другой подход для асинхронного ввода
        return await loop.run_in_executor(None, input, prompt)

    async def _parse_sync_command(self, command: str) -> None:
        """Парсит команду синхронизации."""
        parts = command.split()
        if len(parts) < 2:
            console.print(
                "[red]Использование: sync <символы> [--continuous] [--interval N][/red]"
            )
            return

        symbols = parts[1].split(",") if "," in parts[1] else [parts[1]]
        continuous = "--continuous" in command
        interval = 60  # значение по умолчанию

        # Ищем параметр interval
        for i, part in enumerate(parts):
            if part == "--interval" and i + 1 < len(parts):
                try:
                    interval = int(parts[i + 1])
                    break
                except ValueError:
                    console.print("[red]Неверное значение интервала[/red]")
                    return

        await self.run_sync_command(symbols, continuous, interval)

    async def cleanup(self) -> None:
        """Очищает ресурсы перед завершением."""
        if sync_service:
            await sync_service.close()


@click.group()
@click.option("--debug", is_flag=True, help="Включить отладочное логирование")
def cli(debug):
    """ATS - Automated Trading System CLI."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Отладочное логирование включено")


@cli.command()
@click.argument("symbols", nargs=-1)
@click.option("--continuous", "-c", is_flag=True, help="Непрерывная синхронизация")
@click.option(
    "--interval", "-i", default=60, type=int, help="Интервал синхронизации (секунды)"
)
@click.option("--once", "-o", is_flag=True, help="Однократная синхронизация")
def sync(symbols, continuous, interval, once):
    """Запустить синхронизацию данных."""
    if not symbols and not once:
        console.print("[red]Не указаны символы для синхронизации. Используйте --once для однократной синхронизации всех инструментов.[/red]")
        return

    ats_cli = ATSCLI()
    ats_cli.setup_signal_handlers()

    async def run_sync():
        if not await ats_cli.initialize_system():
            return

        # Если указан флаг --once, синхронизируем все доступные инструменты однократно
        if once:
            all_symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US100"]  # Стандартные инструменты
            console.print(f"[blue]Однократная синхронизация для инструментов: {', '.join(all_symbols)}[/blue]")
            await ats_cli.run_sync_command(all_symbols, continuous=False, interval=interval)
        else:
            # Непрерывная синхронизация указанных символов
            await ats_cli.run_sync_command(list(symbols), continuous, interval)

    try:
        asyncio.run(run_sync())
    except KeyboardInterrupt:
        console.print("\n[yellow]Синхронизация прервана пользователем[/yellow]")
    finally:
        asyncio.run(ats_cli.cleanup())


@cli.command()
def status():
    """Показать статус системы."""
    ats_cli = ATSCLI()

    async def run_status():
        if not await ats_cli.initialize_system():
            return

        await ats_cli.show_status_command()

    asyncio.run(run_status())


@cli.command()
def interactive():
    """Запустить интерактивный режим."""
    ats_cli = ATSCLI()
    ats_cli.setup_signal_handlers()

    async def run_interactive():
        if not await ats_cli.initialize_system():
            return

        await ats_cli.run_interactive_mode()

    try:
        asyncio.run(run_interactive())
    except KeyboardInterrupt:
        console.print("\n[yellow]Интерактивный режим завершен[/yellow]")
    finally:
        asyncio.run(ats_cli.cleanup())


@cli.command()
@click.option("--continuous", "-c", is_flag=True, help="Запустить непрерывную синхронизацию после инициализации")
@click.option("--symbols", "-s", multiple=True, help="Символы для синхронизации")
def init(continuous, symbols):
    """Инициализировать систему и проверить здоровье."""
    ats_cli = ATSCLI()

    async def run_init():
        if await ats_cli.initialize_system():
            console.print(
                "[bold green]Система успешно инициализирована![/bold green]"
            )

            # Если запрошена непрерывная синхронизация, запускаем её
            if continuous:
                sync_symbols = list(symbols) if symbols else ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US100"]
                console.print(f"[blue]🚀 Запуск непрерывной синхронизации для: {', '.join(sync_symbols)}[/blue]")
                await ats_cli.run_sync_command(sync_symbols, continuous=True, interval=60)
        else:
            console.print("[bold red][ERR] Ошибка инициализации системы[/bold red]")
            sys.exit(1)

    try:
        asyncio.run(run_init())
    except KeyboardInterrupt:
        console.print("\n[yellow]Инициализация прервана пользователем[/yellow]")
    finally:
        asyncio.run(ats_cli.cleanup())


if __name__ == "__main__":
    cli()
