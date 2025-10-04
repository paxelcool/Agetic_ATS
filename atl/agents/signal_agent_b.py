"""Swing/trend (scenario B) signal agent."""

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
from .signal_agent_a import _clamp, _ensure_reason


@dataclass
class SwingThresholds:
    donchian_high: int = 55
    donchian_low: int = 20
    rvol_threshold: float = 1.5
    risk_percent: float = 0.8
    atr_multiplier: float = 2.0
    max_positions: int = 6
    max_correlation: float = 0.65
    rebalance_period: str = "weekly"


class SwingSignalAgent(BasePromptAgent):
    """Generate swing-trading signals respecting portfolio constraints."""

    def __init__(
        self,
        *,
        thresholds: SwingThresholds | None = None,
        llm: Optional[SupportsLLM] = None,
        risk_manager: Optional[RiskManager] = None,
    ) -> None:
        self.thresholds = thresholds or SwingThresholds()
        self.risk_manager = risk_manager
        config = load_prompt_config("react_swing.md")
        super().__init__(
            config.path,
            llm=llm,
            default_context={
                "donchian_high": self.thresholds.donchian_high,
                "donchian_low": self.thresholds.donchian_low,
                "rvol_threshold": self.thresholds.rvol_threshold,
                "risk_percent": self.thresholds.risk_percent,
                "atr_multiplier": self.thresholds.atr_multiplier,
                "max_positions": self.thresholds.max_positions,
                "max_correlation": self.thresholds.max_correlation,
                "rebalance_period": self.thresholds.rebalance_period,
            },
        )

    def _market_regime(self, value: Optional[str]) -> Optional[MarketRegime]:
        if not value:
            return None
        try:
            return MarketRegime(value)
        except ValueError:
            return None

    def _parse_decision(
        self,
        raw: str,
        *,
        symbol: str,
        timeframe: Timeframe,
        indicators: Dict[str, Any],
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

        reason = data.get("reason") or "Swing decision"
        confidence = data.get("confidence")
        attachments = data.get("attachments") or {}
        if confidence is None and isinstance(attachments, dict):
            confidence = attachments.get("confidence")

        try:
            confidence_value = float(confidence) if confidence is not None else 0.55
        except (TypeError, ValueError):
            confidence_value = 0.55

        tags = []
        risk_amount = None
        if isinstance(attachments, dict):
            reason = attachments.get("reason", reason)
            tags = attachments.get("tags", [])
            risk_amount = attachments.get("risk_amount")

        return Signal(
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
            reason=_ensure_reason(reason),
            indicators=indicators,
            market_regime=self._market_regime(market_regime),
            tags=list(tags),
        )

    def _fallback_signal(
        self,
        *,
        symbol: str,
        timeframe: Timeframe,
        indicators: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        market_regime: Optional[str],
    ) -> Signal:
        close_price = indicators.get("close")
        ema200 = indicators.get("ema_200")
        donchian_high = indicators.get("donchian_upper")
        donchian_low = indicators.get("donchian_lower")
        atr_value = indicators.get("atr")
        rvol = indicators.get("rvol")

        current_positions = portfolio_state.get("positions", [])
        positions_limit_reached = (
            isinstance(current_positions, list)
            and len(current_positions) >= self.thresholds.max_positions
        )

        action = SignalAction.SKIP
        side: Optional[OrderSide] = None
        confidence = 0.45
        reason = "Портфельные условия не позволяют открыть сделку"
        stop_loss = None
        take_profit = None
        quantity = None

        if positions_limit_reached:
            reason = "Достигнут лимит количества позиций"
        elif (
            close_price is not None
            and ema200 is not None
            and atr_value is not None
            and rvol is not None
            and rvol >= self.thresholds.rvol_threshold
        ):
            if donchian_high is not None and close_price >= donchian_high and close_price >= ema200:
                action = SignalAction.ENTER
                side = OrderSide.BUY
            elif donchian_low is not None and close_price <= donchian_low and close_price <= ema200:
                action = SignalAction.ENTER
                side = OrderSide.SELL

            if action == SignalAction.ENTER and side is not None:
                confidence = 0.62
                reason = (
                    "Fallback правила: подтвержден пробой Donchian с учетом тренда и объема"
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
                        abs(close_price - stop_loss) * self.thresholds.atr_multiplier
                    )

        return Signal(
            symbol=symbol,
            timeframe=timeframe,
            action=action,
            side=side,
            confidence=_clamp(confidence),
            entry_price=close_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            reason=_ensure_reason(reason),
            indicators=indicators,
            market_regime=self._market_regime(market_regime),
            tags=["fallback"] if action != SignalAction.SKIP else ["skip"],
        )

    def generate_signal(
        self,
        *,
        symbol: str,
        timeframe: Timeframe | str,
        indicators: Dict[str, Any],
        portfolio_state: Optional[Dict[str, Any]] = None,
        market_regime: Optional[str] = None,
        volatility_data: Optional[Dict[str, Any]] = None,
        correlation_matrix: Optional[Dict[str, Any]] = None,
    ) -> Signal:
        timeframe_enum = Timeframe(timeframe) if isinstance(timeframe, str) else timeframe
        portfolio_state = portfolio_state or {}
        context = {
            "symbol": symbol,
            "timeframe": timeframe_enum.value,
            "portfolio_state": json.dumps(portfolio_state),
            "market_regime": market_regime or "unknown",
            "volatility_data": json.dumps(volatility_data or {}),
            "correlation_matrix": json.dumps(correlation_matrix or {}),
        }

        if self.llm is None:
            return self._fallback_signal(
                symbol=symbol,
                timeframe=timeframe_enum,
                indicators=indicators,
                portfolio_state=portfolio_state,
                market_regime=market_regime,
            )

        try:
            raw = self.invoke(**context)
            return self._parse_decision(
                raw,
                symbol=symbol,
                timeframe=timeframe_enum,
                indicators=indicators,
                market_regime=market_regime,
            )
        except Exception:
            return self._fallback_signal(
                symbol=symbol,
                timeframe=timeframe_enum,
                indicators=indicators,
                portfolio_state=portfolio_state,
                market_regime=market_regime,
            )


__all__ = ["SwingSignalAgent", "SwingThresholds"]
