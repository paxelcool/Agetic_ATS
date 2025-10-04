"""Utility functions for computing technical indicators used by ATS agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DonchianChannels:
    """Container for Donchian channel values."""

    upper: pd.Series
    lower: pd.Series
    middle: pd.Series


def ema(series: pd.Series, period: int, adjust: bool = False) -> pd.Series:
    """Return an exponential moving average."""

    if period <= 0:
        raise ValueError("period must be a positive integer")

    cleaned = series.astype(float)
    result = cleaned.ewm(span=period, adjust=adjust, min_periods=1).mean()
    return result


def atr(
    data: Mapping[str, Iterable[float]] | pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """Average True Range (ATR)."""

    if period <= 0:
        raise ValueError("period must be a positive integer")

    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        frame = pd.DataFrame(data)

    required = {"high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        missing_fields = ", ".join(sorted(missing))
        raise KeyError(f"ATR requires columns: {missing_fields}")

    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)

    previous_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)

    atr_series = true_range.rolling(window=period, min_periods=1).mean()
    return atr_series


def relative_volume(volume: Iterable[float] | pd.Series, window: int = 20) -> pd.Series:
    """Relative volume (RVOL)."""

    if window <= 0:
        raise ValueError("window must be a positive integer")

    volume_series = pd.Series(volume, dtype=float)
    rolling_mean = volume_series.rolling(window=window, min_periods=1).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rvol = volume_series / rolling_mean
    rvol = rvol.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return rvol


def donchian_channels(
    data: Mapping[str, Iterable[float]] | pd.DataFrame,
    period: int = 20,
) -> DonchianChannels:
    """Compute Donchian channel values."""

    if period <= 0:
        raise ValueError("period must be a positive integer")

    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        frame = pd.DataFrame(data)

    required = {"high", "low"}
    missing = required.difference(frame.columns)
    if missing:
        missing_fields = ", ".join(sorted(missing))
        raise KeyError(f"Donchian channels require columns: {missing_fields}")

    high = frame["high"].astype(float)
    low = frame["low"].astype(float)

    upper = high.rolling(window=period, min_periods=1).max()
    lower = low.rolling(window=period, min_periods=1).min()
    middle = (upper + lower) / 2.0

    return DonchianChannels(upper=upper, lower=lower, middle=middle)


def enrich_with_indicators(
    frame: pd.DataFrame,
    ema_periods: Optional[tuple[int, int]] = (50, 200),
    atr_period: int = 14,
    rvol_window: int = 20,
    donchian_period: int = 20,
) -> pd.DataFrame:
    """Return a copy of ``frame`` enriched with default indicators."""

    if not {"close", "high", "low", "volume"}.issubset(frame.columns):
        missing = sorted({"close", "high", "low", "volume"}.difference(frame.columns))
        raise KeyError(
            "Cannot enrich frame without required columns: " + ", ".join(missing)
        )

    result = frame.copy()

    if ema_periods:
        fast_period, slow_period = ema_periods
        result[f"ema_{fast_period}"] = ema(result["close"], fast_period)
        result[f"ema_{slow_period}"] = ema(result["close"], slow_period)

    result["atr"] = atr(result[["high", "low", "close"]], period=atr_period)
    result["rvol"] = relative_volume(result["volume"], window=rvol_window)

    channels = donchian_channels(result[["high", "low"]], period=donchian_period)
    result["donchian_upper"] = channels.upper
    result["donchian_lower"] = channels.lower
    result["donchian_mid"] = channels.middle

    return result


__all__ = [
    "DonchianChannels",
    "ema",
    "atr",
    "relative_volume",
    "donchian_channels",
    "enrich_with_indicators",
]
