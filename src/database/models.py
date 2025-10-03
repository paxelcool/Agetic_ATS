"""
Модели данных для торговой системы ATS.

Определяет Pydantic-модели для всех основных сущностей системы.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class Timeframe(str, Enum):
    """Перечисление доступных таймфреймов."""

    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"


class OrderType(str, Enum):
    """Типы ордеров."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderSide(str, Enum):
    """Направление ордера."""

    BUY = "buy"
    SELL = "sell"


class SignalAction(str, Enum):
    """Действия торгового сигнала."""

    ENTER = "enter"
    EXIT = "exit"
    SKIP = "skip"
    MANAGE = "manage"


class MarketRegime(str, Enum):
    """Рыночные режимы."""

    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    QUIET = "quiet"


class Quote(BaseModel):
    """
    Модель котировки финансового инструмента.

    Представляет текущую рыночную котировку с ценами и объемами.
    """

    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Символ инструмента (например, EURUSD)",
    )
    timestamp: int = Field(..., ge=0, description="Unix timestamp котировки")
    bid: float = Field(..., ge=0, description="Цена Bid")
    ask: float = Field(..., ge=0, description="Цена Ask")
    volume: Optional[int] = Field(None, ge=0, description="Объем торгов")

    # Вычисляемые поля
    spread: Optional[float] = Field(None, ge=0, description="Спред (ask - bid)")
    mid_price: Optional[float] = Field(
        None, ge=0, description="Средняя цена ((ask + bid) / 2)"
    )

    @validator("spread", always=True)
    def calculate_spread(cls, v, values):
        """Вычисляем спред из bid и ask."""
        if v is None and "bid" in values and "ask" in values:
            return float(values["ask"]) - float(values["bid"])
        return v

    @validator("mid_price", always=True)
    def calculate_mid_price(cls, v, values):
        """Вычисляем среднюю цену из bid и ask."""
        if v is None and "bid" in values and "ask" in values:
            return (float(values["ask"]) + float(values["bid"])) / 2
        return v

    @validator("ask")
    def ask_greater_than_bid(cls, v, values):
        """Проверяем, что ask >= bid."""
        if "bid" in values and v < values["bid"]:
            raise ValueError("ask цена должна быть >= bid цены")
        return v

    class Config:
        """Конфигурация модели."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class Trade(BaseModel):
    """
    Модель торговой сделки.

    Представляет информацию о выполненной или планируемой сделке.
    """

    id: Optional[str] = Field(None, description="Уникальный идентификатор сделки")
    symbol: str = Field(
        ..., min_length=1, max_length=20, description="Символ инструмента"
    )
    side: OrderSide = Field(..., description="Направление сделки (BUY/SELL)")
    entry_price: float = Field(..., ge=0, description="Цена входа в сделку")
    quantity: float = Field(..., gt=0, description="Количество контрактов/лотов")
    stop_loss: Optional[float] = Field(None, ge=0, description="Уровень стоп-лосса")
    take_profit: Optional[float] = Field(None, ge=0, description="Уровень тейк-профита")

    # Временные метки
    opened_at: Optional[datetime] = Field(None, description="Время открытия сделки")
    closed_at: Optional[datetime] = Field(None, description="Время закрытия сделки")

    # Финансовые результаты
    pnl: Optional[float] = Field(None, description="Прибыль/убыток в валюте счета")
    pnl_points: Optional[float] = Field(None, description="Прибыль/убыток в пунктах")
    commission: Optional[float] = Field(None, ge=0, description="Комиссия брокера")

    # Статус и контекст
    status: str = Field(
        "pending", description="Статус сделки (pending, open, closed, cancelled)"
    )
    magic_number: Optional[int] = Field(
        None, description="Magic number для идентификации"
    )
    comment: Optional[str] = Field(
        None, max_length=255, description="Комментарий к сделке"
    )

    # Связанные данные
    signal_id: Optional[str] = Field(None, description="ID сигнала, вызвавшего сделку")
    risk_amount: Optional[float] = Field(
        None, ge=0, description="Сумма риска на сделку"
    )

    @validator("take_profit")
    def tp_greater_than_entry(cls, v, values):
        """Проверяем, что TP >= entry_price для BUY и <= entry_price для SELL."""
        if v is not None and "entry_price" in values and "side" in values:
            if values["side"] == OrderSide.BUY and v <= values["entry_price"]:
                raise ValueError("Для BUY сделок TP должен быть > цены входа")
            elif values["side"] == OrderSide.SELL and v >= values["entry_price"]:
                raise ValueError("Для SELL сделок TP должен быть < цены входа")
        return v

    class Config:
        """Конфигурация модели."""

        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class Signal(BaseModel):
    """
    Модель торгового сигнала.

    Представляет торговый сигнал с обоснованием и параметрами исполнения.
    """

    id: Optional[str] = Field(None, description="Уникальный идентификатор сигнала")
    symbol: str = Field(
        ..., min_length=1, max_length=20, description="Символ инструмента"
    )
    timeframe: Timeframe = Field(..., description="Таймфрейм анализа")

    # Основные параметры сигнала
    action: SignalAction = Field(..., description="Рекомендуемое действие")
    side: Optional[OrderSide] = Field(None, description="Рекомендуемое направление")
    confidence: float = Field(..., ge=0, le=1, description="Уверенность сигнала (0-1)")

    # Ценовые уровни
    entry_price: Optional[float] = Field(
        None, ge=0, description="Рекомендуемая цена входа"
    )
    stop_loss: Optional[float] = Field(
        None, ge=0, description="Рекомендуемый стоп-лосс"
    )
    take_profit: Optional[float] = Field(
        None, ge=0, description="Рекомендуемый тейк-профит"
    )

    # Размеры позиции
    quantity: Optional[float] = Field(
        None, gt=0, description="Рекомендуемый размер позиции"
    )
    risk_amount: Optional[float] = Field(
        None, ge=0, description="Рекомендуемая сумма риска"
    )

    # Временные метки
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Время создания сигнала"
    )
    expires_at: Optional[datetime] = Field(None, description="Время истечения сигнала")

    # Обоснование и контекст
    reason: str = Field(..., min_length=10, description="Подробное обоснование сигнала")
    indicators: Dict[str, Any] = Field(
        default_factory=dict, description="Значения индикаторов"
    )
    market_regime: Optional[MarketRegime] = Field(
        None, description="Текущий рыночный режим"
    )

    # Связанные данные
    related_signals: List[str] = Field(
        default_factory=list, description="Связанные сигналы"
    )
    tags: List[str] = Field(default_factory=list, description="Теги для категоризации")

    class Config:
        """Конфигурация модели."""

        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class TechnicalIndicators(BaseModel):
    """
    Модель технических индикаторов.

    Содержит рассчитанные значения индикаторов для анализа рынка.
    """

    symbol: str = Field(..., description="Символ инструмента")
    timeframe: Timeframe = Field(..., description="Таймфрейм")
    timestamp: int = Field(..., description="Время расчета")

    # Трендовые индикаторы
    ema_20: Optional[float] = Field(None, description="EMA 20 периодов")
    ema_50: Optional[float] = Field(None, description="EMA 50 периодов")
    ema_200: Optional[float] = Field(None, description="EMA 200 периодов")
    sma_20: Optional[float] = Field(None, description="SMA 20 периодов")

    # Осцилляторы
    rsi: Optional[float] = Field(None, ge=0, le=100, description="RSI (0-100)")
    stoch_k: Optional[float] = Field(None, ge=0, le=100, description="Стохастик %K")
    stoch_d: Optional[float] = Field(None, ge=0, le=100, description="Стохастик %D")

    # Волатильность
    atr: Optional[float] = Field(None, ge=0, description="ATR (Average True Range)")
    atr_percent: Optional[float] = Field(
        None, ge=0, description="ATR в процентах от цены"
    )

    # Объемы
    volume: Optional[int] = Field(None, ge=0, description="Объем торгов")
    rvol: Optional[float] = Field(None, ge=0, description="Относительный объем (RVOL)")

    # Диапазоны
    donchian_upper: Optional[float] = Field(None, description="Верхняя линия Дончиана")
    donchian_lower: Optional[float] = Field(None, description="Нижняя линия Дончиана")
    opening_range_high: Optional[float] = Field(
        None, description="Максимум Opening Range"
    )
    opening_range_low: Optional[float] = Field(
        None, description="Минимум Opening Range"
    )

    class Config:
        """Конфигурация модели."""

        use_enum_values = True


class FeatureRequest(BaseModel):
    """
    Модель запроса на расчет индикаторов.

    Используется для запроса технических индикаторов для анализа.
    """

    symbol: str = Field(
        ..., min_length=1, max_length=20, description="Символ инструмента"
    )
    timeframe: Timeframe = Field(..., description="Таймфрейм анализа")
    lookback: int = Field(
        500, ge=1, le=10000, description="Количество баров для анализа"
    )
    indicators: List[str] = Field(
        default_factory=lambda: ["ema_20", "ema_50", "rsi", "atr", "rvol"],
        description="Список запрашиваемых индикаторов",
    )

    class Config:
        """Конфигурация модели."""

        use_enum_values = True


class SignalDecision(BaseModel):
    """
    Модель решения торгового агента.

    Стандартизированный формат ответа от торговых агентов.
    """

    action: SignalAction = Field(..., description="Рекомендуемое действие")
    side: Optional[OrderSide] = Field(None, description="Направление сделки")
    reason: str = Field(..., min_length=10, description="Обоснование решения")

    # Ценовые уровни
    entry: Optional[float] = Field(None, ge=0, description="Цена входа")
    stop_loss: Optional[float] = Field(None, ge=0, description="Стоп-лосс")
    take_profit: Optional[float] = Field(None, ge=0, description="Тейк-профит")

    # Параметры позиции
    quantity: Optional[float] = Field(None, gt=0, description="Размер позиции")
    risk_percent: Optional[float] = Field(None, ge=0, le=1, description="Процент риска")

    # Дополнительная информация
    confidence: float = Field(0.5, ge=0, le=1, description="Уверенность решения")
    attachments: Dict[str, Any] = Field(
        default_factory=dict, description="Дополнительные данные"
    )

    class Config:
        """Конфигурация модели."""

        use_enum_values = True


class OrderRequest(BaseModel):
    """
    Модель запроса на исполнение ордера.

    Используется для создания ордеров через торговый терминал.
    """

    symbol: str = Field(
        ..., min_length=1, max_length=20, description="Символ инструмента"
    )
    side: OrderSide = Field(..., description="Направление ордера")
    quantity: float = Field(..., gt=0, description="Количество")
    order_type: OrderType = Field(OrderType.MARKET, description="Тип ордера")

    # Ценовые параметры
    price: Optional[float] = Field(None, ge=0, description="Цена для лимитных ордеров")
    stop_loss: Optional[float] = Field(None, ge=0, description="Стоп-лосс")
    take_profit: Optional[float] = Field(None, ge=0, description="Тейк-профит")

    # Дополнительные параметры
    comment: str = Field("ATS", description="Комментарий к ордеру")
    magic_number: Optional[int] = Field(None, description="Magic number")
    expiration: Optional[datetime] = Field(None, description="Время истечения ордера")

    @validator("price", always=True)
    def price_required_for_limit_orders(cls, v, values):
        """Цена обязательна для лимитных ордеров."""
        order_type = values.get("order_type")
        if order_type in [OrderType.LIMIT, OrderType.STOP] and v is None:
            raise ValueError("Цена обязательна для лимитных и стоп-ордеров")
        return v

    class Config:
        """Конфигурация модели."""

        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}
