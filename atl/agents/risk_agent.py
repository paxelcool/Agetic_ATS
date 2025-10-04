"""Risk management agent that coordinates position sizing limits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..tools.risk import RiskAssessment, RiskManager, RiskParameters
from .base import BasePromptAgent, SupportsLLM, load_prompt_config


@dataclass
class RiskReport:
    """Structured risk analysis returned by :class:`RiskAgent`."""

    can_trade: bool
    max_position_size: float
    risk_adjustment_factor: float
    violations: List[str]
    recommendations: List[str]
    portfolio_risk: Dict[str, float]


class RiskAgent(BasePromptAgent):
    """Validate trade limits and provide portfolio guidance."""

    def __init__(
        self,
        parameters: RiskParameters,
        *,
        llm: Optional[SupportsLLM] = None,
    ) -> None:
        self.parameters = parameters
        self.manager = RiskManager(parameters)
        config = load_prompt_config("risk_management.md")
        super().__init__(
            config.path,
            llm=llm,
            default_context={
                "risk_per_trade": parameters.risk_per_trade * 100,
                "daily_drawdown_limit": 2.0,
                "weekly_drawdown_limit": 6.0,
                "max_positions": 6,
            },
        )

    def _fallback_report(self, account_state: Dict[str, Any]) -> RiskReport:
        balance = float(account_state.get("account_balance", self.parameters.account_balance))
        positions = account_state.get("current_positions", [])
        drawdown_history = account_state.get("drawdown_history", [])
        correlation_data = account_state.get("correlation_data", {})

        exposures = []
        if isinstance(positions, list):
            for pos in positions:
                exposure = pos.get("exposure") if isinstance(pos, dict) else None
                if exposure is None:
                    qty = float(pos.get("quantity", 0)) if isinstance(pos, dict) else 0.0
                    price = float(pos.get("price", 0)) if isinstance(pos, dict) else 0.0
                    exposure = qty * price
                exposures.append(float(exposure))

        correlation_risk = float(correlation_data.get("max", 0.0)) if isinstance(correlation_data, dict) else 0.0
        drawdown_risk = float(drawdown_history[-1]) if drawdown_history else 0.0

        can_trade = self.manager.ensure_can_trade(drawdown_risk, max_drawdown=6.0)
        portfolio_risk = self.manager.score_portfolio_risk(
            exposures,
            correlation_risk=correlation_risk,
            drawdown_risk=drawdown_risk,
        )

        violations: List[str] = []
        if not can_trade:
            violations.append("Drawdown limit reached")
        if correlation_risk > 0.65:
            violations.append("High correlation between positions")

        recommendations: List[str] = []
        if correlation_risk > 0.65:
            recommendations.append("Reduce exposure to correlated instruments")
        if drawdown_risk > 4.0:
            recommendations.append("Pause trading until drawdown recovers")

        risk_adjustment = 1.0
        if correlation_risk > 0.65 or drawdown_risk > 4.0:
            risk_adjustment = 0.5

        max_position_size = self.parameters.max_position or (
            balance * self.parameters.risk_per_trade
        )

        return RiskReport(
            can_trade=can_trade,
            max_position_size=float(max_position_size),
            risk_adjustment_factor=risk_adjustment,
            violations=violations,
            recommendations=recommendations,
            portfolio_risk=portfolio_risk,
        )

    def analyze_account(self, account_state: Dict[str, Any]) -> RiskReport:
        """Return a :class:`RiskReport` either from LLM or fallback logic."""

        if self.llm is None:
            return self._fallback_report(account_state)

        context = {
            "account_balance": account_state.get("account_balance", self.parameters.account_balance),
            "current_positions": json.dumps(account_state.get("current_positions", [])),
            "open_trades": json.dumps(account_state.get("open_trades", [])),
            "market_volatility": json.dumps(account_state.get("market_volatility", {})),
            "correlation_data": json.dumps(account_state.get("correlation_data", {})),
            "drawdown_history": json.dumps(account_state.get("drawdown_history", [])),
        }

        try:
            raw = self.invoke(**context)
            data = json.loads(raw or "{}")
        except Exception:
            return self._fallback_report(account_state)

        return RiskReport(
            can_trade=bool(data.get("can_trade", True)),
            max_position_size=float(data.get("max_position_size", 0.0)),
            risk_adjustment_factor=float(data.get("risk_adjustment_factor", 1.0)),
            violations=list(data.get("violations", [])),
            recommendations=list(data.get("recommendations", [])),
            portfolio_risk=dict(data.get("portfolio_risk", {})),
        )

    def assess_signal(
        self,
        signal: Any,
        *,
        atr_value: Optional[float] = None,
    ) -> RiskAssessment:
        """Build a risk assessment using the internal manager."""

        if signal.side is None or signal.entry_price is None:
            raise ValueError("Signal must define side and entry price for risk assessment")

        return self.manager.assess_trade(
            side=signal.side,
            entry_price=signal.entry_price,
            stop_loss=getattr(signal, "stop_loss", None),
            take_profit=getattr(signal, "take_profit", None),
            atr_value=atr_value,
        )


__all__ = ["RiskAgent", "RiskReport"]
