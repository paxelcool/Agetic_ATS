"""Risk management helpers for ATS agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from src.database.models import OrderSide, Signal


@dataclass
class RiskParameters:
    """Static risk parameters for the trading account."""

    account_balance: float
    risk_per_trade: float = 0.01
    contract_size: float = 1.0
    min_position: float = 0.01
    max_position: Optional[float] = None
    reward_risk: float = 2.0
    atr_multiplier: float = 1.5

    def risk_amount(self) -> float:
        """Return the monetary risk permitted per trade."""

        if self.account_balance <= 0:
            raise ValueError("account_balance must be positive")
        if not 0 < self.risk_per_trade <= 1:
            raise ValueError("risk_per_trade must be in (0, 1]")
        return self.account_balance * self.risk_per_trade


@dataclass
class PositionSizingResult:
    """Result of a position sizing computation."""

    quantity: float
    risk_amount: float
    per_unit_risk: float


@dataclass
class RiskAssessment:
    """Comprehensive risk profile for a trade idea."""

    stop_loss: float
    take_profit: float
    position_size: float
    risk_amount: float
    reward_ratio: float
    warnings: Sequence[str] = field(default_factory=tuple)


class RiskManager:
    """Encapsulates position sizing and stop management rules."""

    def __init__(self, parameters: RiskParameters) -> None:
        self.parameters = parameters

    @staticmethod
    def _direction(side: OrderSide) -> int:
        return 1 if side == OrderSide.BUY else -1

    def suggest_stop_loss(
        self,
        side: OrderSide,
        entry_price: float,
        atr_value: Optional[float] = None,
        atr_multiplier: Optional[float] = None,
    ) -> float:
        """Suggest a stop loss price using ATR or a fixed offset."""

        if entry_price <= 0:
            raise ValueError("entry_price must be positive")

        multiplier = atr_multiplier or self.parameters.atr_multiplier
        if atr_value is None or atr_value <= 0:
            distance = entry_price * 0.01
        else:
            distance = atr_value * multiplier

        direction = self._direction(side)
        stop_loss = entry_price - direction * distance
        return max(stop_loss, 0.0)

    def suggest_take_profit(
        self,
        side: OrderSide,
        entry_price: float,
        stop_loss: float,
        reward_risk: Optional[float] = None,
    ) -> float:
        """Return a take profit price from the reward/risk ratio."""

        ratio = reward_risk or self.parameters.reward_risk
        if ratio <= 0:
            raise ValueError("reward/risk ratio must be positive")

        risk_per_unit = abs(entry_price - stop_loss)
        direction = self._direction(side)
        take_profit = entry_price + direction * risk_per_unit * ratio
        return max(take_profit, 0.0)

    def _position_size(
        self,
        side: OrderSide,
        entry_price: float,
        stop_loss: float,
    ) -> PositionSizingResult:
        risk_amount = self.parameters.risk_amount()
        distance = abs(entry_price - stop_loss)
        if distance <= 0:
            raise ValueError("stop loss must differ from entry price")

        per_unit_risk = distance * self.parameters.contract_size
        quantity = risk_amount / per_unit_risk

        quantity = max(quantity, self.parameters.min_position)
        if self.parameters.max_position is not None:
            quantity = min(quantity, self.parameters.max_position)

        return PositionSizingResult(
            quantity=round(quantity, 6),
            risk_amount=risk_amount,
            per_unit_risk=per_unit_risk,
        )

    @staticmethod
    def reward_ratio(
        side: OrderSide,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> float:
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        if risk == 0:
            return 0.0
        return reward / risk

    def assess_trade(
        self,
        side: OrderSide,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        atr_value: Optional[float] = None,
    ) -> RiskAssessment:
        """Evaluate trade risk and return a :class:`RiskAssessment`."""

        sl_price = (
            stop_loss
            if stop_loss is not None
            else self.suggest_stop_loss(side, entry_price, atr_value=atr_value)
        )
        tp_price = (
            take_profit
            if take_profit is not None
            else self.suggest_take_profit(side, entry_price, sl_price)
        )

        sizing = self._position_size(side, entry_price, sl_price)
        reward_ratio = self.reward_ratio(side, entry_price, sl_price, tp_price)

        warnings: list[str] = []
        if reward_ratio < 1.0:
            warnings.append("Reward-to-risk ratio below 1.0")
        if sizing.quantity <= self.parameters.min_position:
            warnings.append("Position size at minimum threshold")

        return RiskAssessment(
            stop_loss=round(sl_price, 6),
            take_profit=round(tp_price, 6),
            position_size=sizing.quantity,
            risk_amount=round(sizing.risk_amount, 2),
            reward_ratio=round(reward_ratio, 2),
            warnings=tuple(warnings),
        )

    def ensure_can_trade(self, current_drawdown: float, max_drawdown: float) -> bool:
        """Check drawdown limits."""

        if current_drawdown < 0:
            return True
        return current_drawdown <= max_drawdown

    def score_portfolio_risk(
        self,
        exposures: Iterable[float],
        correlation_risk: float,
        drawdown_risk: float,
    ) -> dict[str, float]:
        """Return simple portfolio risk metrics."""

        total_exposure = float(sum(abs(e) for e in exposures))
        concentration = max((abs(e) for e in exposures), default=0.0)
        return {
            "total_exposure": round(total_exposure, 4),
            "concentration_risk": round(concentration, 4),
            "correlation_risk": round(float(correlation_risk), 4),
            "drawdown_risk": round(float(drawdown_risk), 4),
        }


def signal_to_assessment(
    signal: Signal,
    risk_manager: RiskManager,
    atr_value: Optional[float] = None,
) -> RiskAssessment:
    """Convenience helper converting a signal to a risk assessment."""

    if signal.side is None or signal.entry_price is None:
        raise ValueError("signal must define side and entry_price")

    return risk_manager.assess_trade(
        side=signal.side,
        entry_price=signal.entry_price,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        atr_value=atr_value,
    )


__all__ = [
    "RiskParameters",
    "PositionSizingResult",
    "RiskAssessment",
    "RiskManager",
    "signal_to_assessment",
]
