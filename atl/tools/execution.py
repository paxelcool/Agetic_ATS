"""Helpers that translate signals into executable trade plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.database.models import OrderSide, OrderType, Quote, Signal, Trade

from .risk import RiskAssessment, RiskManager


@dataclass
class ExecutionPlan:
    """Result of preparing a trade for execution."""

    trade: Trade
    assessment: RiskAssessment
    order_type: OrderType = OrderType.MARKET
    slippage: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)


class ExecutionPlanner:
    """Convert :class:`Signal` instances into :class:`Trade` objects."""

    def __init__(
        self,
        risk_manager: RiskManager,
        default_order_type: OrderType = OrderType.MARKET,
        default_slippage: float = 0.0,
    ) -> None:
        self.risk_manager = risk_manager
        self.default_order_type = default_order_type
        self.default_slippage = default_slippage

    @staticmethod
    def _price_from_quote(signal: Signal, quote: Optional[Quote]) -> float:
        if quote is None:
            if signal.entry_price is None:
                raise ValueError("entry price must be provided via signal or quote")
            return signal.entry_price

        if signal.side == OrderSide.SELL:
            return quote.bid or quote.mid_price or quote.ask
        return quote.ask or quote.mid_price or quote.bid

    def create_plan(
        self,
        signal: Signal,
        quote: Optional[Quote] = None,
        atr_value: Optional[float] = None,
        assessment: Optional[RiskAssessment] = None,
    ) -> ExecutionPlan:
        """Create an execution plan from a signal."""

        if signal.side is None:
            raise ValueError("Signal must include trade side")

        entry_price = signal.entry_price or self._price_from_quote(signal, quote)
        risk_assessment = assessment or self.risk_manager.assess_trade(
            side=signal.side,
            entry_price=entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            atr_value=atr_value,
        )

        trade = Trade(
            symbol=signal.symbol,
            side=signal.side,
            entry_price=entry_price,
            quantity=risk_assessment.position_size,
            stop_loss=risk_assessment.stop_loss,
            take_profit=risk_assessment.take_profit,
            status="pending",
            comment=signal.reason,
            signal_id=signal.id,
            risk_amount=risk_assessment.risk_amount,
        )

        metadata = {
            "generated_at": datetime.utcnow().isoformat(),
            "source_action": signal.action.value,
        }

        return ExecutionPlan(
            trade=trade,
            assessment=risk_assessment,
            order_type=self.default_order_type,
            slippage=self.default_slippage,
            metadata=metadata,
        )


__all__ = ["ExecutionPlanner", "ExecutionPlan"]
