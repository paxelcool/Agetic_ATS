"""Intraday (scenario A) signal agent implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.database.models import (
    MarketRegime,
    OrderSide,
    Signal,
    SignalAction,
    Timeframe,
)

from ..tools.risk import RiskManager
from .base import BasePromptAgent, SupportsLLM, load_prompt_config


def _clamp(value: float, *, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(value, high))


def _ensure_reason(text: str) -> str:
    cleaned = text.strip()
    if len(cleaned) < 10:
        return (cleaned + " decision.").ljust(10, ".")
    return cleaned


@dataclass
class AgentThresholds:
    rvol_threshold: float = 1.8
    risk_percent: float = 1.0
    atr_multiplier: float = 1.5
    reward_ratio: float = 2.0
    partial_r: float = 1.0
    atr_limit: float = 8.0
    drawdown_limit: float = 6.0


class IntradaySignalAgent(BasePromptAgent):
    """Generate signals for intraday trading based on scenario A rules."""

    def __init__(
        self,
        *,
        thresholds: AgentThresholds | None = None,
        llm: Optional[SupportsLLM] = None,
        risk_manager: Optional[RiskManager] = None,
    ) -> None:
        self.thresholds = thresholds or AgentThresholds()
        self.risk_manager = risk_manager
        config = load_prompt_config("react_intraday.md")
        super().__init__(
            config.path,
            llm=llm,
            default_context={
                "rvol_threshold": self.thresholds.rvol_threshold,
                "risk_percent": self.thresholds.risk_percent,
                "atr_multiplier": self.thresholds.atr_multiplier,
                "reward_ratio": self.thresholds.reward_ratio,
                "partial_r": self.thresholds.partial_r,
                "atr_limit": self.thresholds.atr_limit,
                "drawdown_limit": self.thresholds.drawdown_limit,
            },
        )

    def _parse_decision(
        self,
        raw: str,
        *,
        symbol: str,
        timeframe: Timeframe,
        features: Dict[str, Any],
        market_regime: Optional[str],
    ) -> Signal:
        data = json.loads(raw or "{}")
        action_value = data.get("action", SignalAction.SKIP.value)
        try:
            action = SignalAction(action_value)
        except ValueError:
            action = SignalAction.SKIP
        side_value = data.get("side")
        try:
            side = OrderSide(side_value) if side_value else None
        except ValueError:
            side = None
        entry = data.get("entry") or data.get("entry_price")
        stop_loss = data.get("sl") or data.get("stop_loss")
        take_profit = data.get("tp") or data.get("take_profit")
        quantity = data.get("size") or data.get("quantity")
        reason_text = data.get("reason") or "Intraday decision"
        confidence = data.get("confidence")
        attachments = data.get("attachments") or {}
        if confidence is None and isinstance(attachments, dict):
            confidence = attachments.get("confidence", 0.5)

        if isinstance(attachments, dict):
            reason_text = attachments.get("reason", reason_text)
            tags = attachments.get("tags", [])
            risk_amount = attachments.get("risk_amount")
        else:
            tags = []
            risk_amount = None

        market_regime_enum = None
        if market_regime:
            try:
                market_regime_enum = MarketRegime(market_regime)
            except ValueError:
                market_regime_enum = None

        try:
            confidence_value = float(confidence) if confidence is not None else 0.5
        except (TypeError, ValueError):
            confidence_value = 0.5

        signal = Signal(
            symbol=symbol,
            timeframe=timeframe,
            action=action,
            side=side,
            confidence=_clamp(confidence_value),
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            risk_amount=risk_amount,
            reason=_ensure_reason(reason_text),
            indicators=features,
            market_regime=market_regime_enum,
            tags=list(tags),
        )
        return signal

    def _fallback_signal(
        self,
        *,
        symbol: str,
        timeframe: Timeframe,
        features: Dict[str, Any],
        market_regime: Optional[str],
    ) -> Signal:
        close_price = features.get("close")
        ema_fast = features.get("ema_50") or features.get("ema_fast")
        ema_slow = features.get("ema_200") or features.get("ema_slow")
        rvol = features.get("rvol")
        atr_value = features.get("atr")
        donchian_upper = features.get("donchian_upper")
        donchian_lower = features.get("donchian_lower")

        action = SignalAction.SKIP
        side: Optional[OrderSide] = None
        entry_price = close_price
        stop_loss = None
        take_profit = None
        quantity = None
        reason = "Недостаточно условий для открытия позиции"
        confidence = 0.35

        if (
            close_price is not None
            and ema_fast is not None
            and ema_slow is not None
            and rvol is not None
            and atr_value is not None
            and rvol >= self.thresholds.rvol_threshold
        ):
            trend_up = ema_fast > ema_slow
            breakout_up = donchian_upper is not None and close_price >= donchian_upper
            breakout_down = donchian_lower is not None and close_price <= donchian_lower

            if trend_up and breakout_up:
                action = SignalAction.ENTER
                side = OrderSide.BUY
            elif (not trend_up) and breakout_down:
                action = SignalAction.ENTER
                side = OrderSide.SELL

            if action == SignalAction.ENTER and side is not None:
                confidence = 0.68
                reason = (
                    "Fallback правила: подтвержден тренд и пробой диапазона с высоким объемом"
                )
                if self.risk_manager is not None and close_price is not None:
                    assessment = self.risk_manager.assess_trade(
                        side=side,
                        entry_price=close_price,
                        atr_value=atr_value,
                    )
                    stop_loss = assessment.stop_loss
                    take_profit = assessment.take_profit
                    quantity = assessment.position_size
                else:
                    direction = 1 if side == OrderSide.BUY else -1
                    stop_loss = close_price - direction * atr_value * self.thresholds.atr_multiplier
                    take_profit = close_price + direction * (
                        abs(close_price - stop_loss) * self.thresholds.reward_ratio
                    )

        market_regime_enum = None
        if market_regime:
            try:
                market_regime_enum = MarketRegime(market_regime)
            except ValueError:
                market_regime_enum = None

        return Signal(
            symbol=symbol,
            timeframe=timeframe,
            action=action,
            side=side,
            confidence=_clamp(confidence),
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            reason=_ensure_reason(reason),
            indicators=features,
            market_regime=market_regime_enum,
            tags=["fallback"] if action != SignalAction.SKIP else ["skip"],
        )

    def generate_signal(
        self,
        *,
        symbol: str,
        timeframe: Timeframe | str,
        features: Dict[str, Any],
        market_regime: Optional[str] = None,
        current_position: Optional[Dict[str, Any]] = None,
        account_info: Optional[Dict[str, Any]] = None,
    ) -> Signal:
        timeframe_enum = Timeframe(timeframe) if isinstance(timeframe, str) else timeframe
        context = {
            "symbol": symbol,
            "timeframe": timeframe_enum.value,
            "features": json.dumps(features),
            "current_position": json.dumps(current_position or {}),
            "market_regime": market_regime or "unknown",
            "account_info": json.dumps(account_info or {}),
        }

        if self.llm is None:
            return self._fallback_signal(
                symbol=symbol,
                timeframe=timeframe_enum,
                features=features,
                market_regime=market_regime,
            )

        try:
            raw = self.invoke(**context)
            return self._parse_decision(
                raw,
                symbol=symbol,
                timeframe=timeframe_enum,
                features=features,
                market_regime=market_regime,
            )
        except Exception:
            return self._fallback_signal(
                symbol=symbol,
                timeframe=timeframe_enum,
                features=features,
                market_regime=market_regime,
            )


__all__ = ["IntradaySignalAgent", "AgentThresholds"]
