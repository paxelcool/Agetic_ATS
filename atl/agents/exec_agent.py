"""Execution agent translating signals into actionable orders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.database.models import OrderRequest, OrderType, Quote, Signal, Trade

from ..tools.execution import ExecutionPlan, ExecutionPlanner
from .base import BasePromptAgent, SupportsLLM, load_prompt_config


@dataclass
class ExecutionReport:
    """Structured execution output used for logging and monitoring."""

    execution_status: str
    order_id: Optional[str]
    executed_price: Optional[float]
    executed_volume: Optional[float]
    sl_price: Optional[float]
    tp_price: Optional[float]
    execution_time: datetime
    errors: List[str]
    warnings: List[str]
    trade: Trade


class ExecutionAgent(BasePromptAgent):
    """Prepare trades for execution and produce monitoring reports."""

    def __init__(
        self,
        planner: ExecutionPlanner,
        *,
        llm: Optional[SupportsLLM] = None,
    ) -> None:
        self.planner = planner
        config = load_prompt_config("order_execution.md")
        super().__init__(config.path, llm=llm)

    def build_plan(
        self,
        signal: Signal,
        *,
        quote: Optional[Quote] = None,
        atr_value: Optional[float] = None,
    ) -> ExecutionPlan:
        return self.planner.create_plan(signal, quote=quote, atr_value=atr_value)

    def to_order_request(self, plan: ExecutionPlan) -> OrderRequest:
        """Convert an execution plan into an :class:`OrderRequest`."""

        trade = plan.trade
        return OrderRequest(
            symbol=trade.symbol,
            side=trade.side,
            quantity=trade.quantity,
            order_type=plan.order_type,
            price=trade.entry_price if plan.order_type != OrderType.MARKET else None,
            stop_loss=trade.stop_loss,
            take_profit=trade.take_profit,
            comment=trade.comment or "ATS",
        )

    def _fallback_report(self, plan: ExecutionPlan) -> ExecutionReport:
        trade = plan.trade
        return ExecutionReport(
            execution_status="pending",
            order_id=None,
            executed_price=None,
            executed_volume=None,
            sl_price=trade.stop_loss,
            tp_price=trade.take_profit,
            execution_time=datetime.utcnow(),
            errors=[],
            warnings=[],
            trade=trade,
        )

    def report(
        self,
        plan: ExecutionPlan,
        *,
        account_info: Optional[Dict[str, Any]] = None,
        market_data: Optional[Dict[str, Any]] = None,
        risk_limits: Optional[Dict[str, Any]] = None,
        execution_history: Optional[List[Dict[str, Any]]] = None,
    ) -> ExecutionReport:
        if self.llm is None:
            return self._fallback_report(plan)

        context = {
            "signal": json.dumps(plan.trade.dict(exclude_none=True)),
            "account_info": json.dumps(account_info or {}),
            "market_data": json.dumps(market_data or {}),
            "risk_limits": json.dumps(risk_limits or {}),
            "execution_history": json.dumps(execution_history or []),
        }

        try:
            raw = self.invoke(**context)
            data = json.loads(raw or "{}")
        except Exception:
            return self._fallback_report(plan)

        trade = plan.trade
        status = str(data.get("execution_status", "pending"))
        order_id = data.get("order_id")
        executed_price = data.get("executed_price")
        executed_volume = data.get("executed_volume")
        sl_price = data.get("sl_price", trade.stop_loss)
        tp_price = data.get("tp_price", trade.take_profit)
        execution_time_value = data.get("execution_time" )

        try:
            execution_time = (
                datetime.fromisoformat(execution_time_value)
                if isinstance(execution_time_value, str)
                else datetime.utcnow()
            )
        except ValueError:
            execution_time = datetime.utcnow()

        return ExecutionReport(
            execution_status=status,
            order_id=order_id,
            executed_price=executed_price,
            executed_volume=executed_volume,
            sl_price=sl_price,
            tp_price=tp_price,
            execution_time=execution_time,
            errors=list(data.get("errors", [])),
            warnings=list(data.get("warnings", [])),
            trade=trade,
        )


__all__ = ["ExecutionAgent", "ExecutionReport"]
