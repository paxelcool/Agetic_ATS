"""Governance agent orchestrating scenario selection and risk adjustments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .base import BasePromptAgent, SupportsLLM, load_prompt_config


@dataclass
class GovernanceDecision:
    """Decision returned by :class:`GovernanceAgent`."""

    scenario: str
    reason: str
    adjustments: Dict[str, float] = field(default_factory=dict)
    recommended_timeframe: Optional[str] = None


class GovernanceAgent(BasePromptAgent):
    """Choose between trading scenarios and adapt high level parameters."""

    default_balance_threshold: float = 5000.0
    caution_threshold: float = 3000.0

    def __init__(
        self,
        *,
        llm: Optional[SupportsLLM] = None,
        default_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        config = load_prompt_config("governance.md")
        context = {
            "balance_threshold": self.default_balance_threshold,
            "caution_threshold": self.caution_threshold,
        }
        if default_context:
            context.update(default_context)
        super().__init__(config.path, llm=llm, default_context=context)

    def _fallback_decision(
        self,
        account_state: Dict[str, Any],
        *,
        preferred: Optional[str] = None,
    ) -> GovernanceDecision:
        balance = float(account_state.get("account_balance", 0.0) or 0.0)
        open_positions = account_state.get("open_positions") or []
        realized_drawdown = float(account_state.get("realized_drawdown", 0.0) or 0.0)

        if balance >= self.default_balance_threshold:
            scenario = "swing"
            reason = "Account balance exceeds swing threshold"
        elif balance >= self.caution_threshold:
            scenario = preferred or "intraday"
            reason = "Balance in caution zone, maintain preferred scenario"
        else:
            scenario = "intraday"
            reason = "Balance below swing threshold"

        if preferred and preferred != scenario:
            reason += f" (override recommendation: {preferred})"

        adjustments: Dict[str, float] = {}
        if realized_drawdown > 4.0:
            adjustments["risk_per_trade"] = 0.5 / 100
        elif balance >= self.default_balance_threshold:
            adjustments["risk_per_trade"] = 0.8 / 100
        else:
            adjustments["risk_per_trade"] = 1.0 / 100

        if isinstance(open_positions, list) and len(open_positions) >= 5:
            adjustments["rvol_threshold"] = 2.0

        recommended_timeframe = account_state.get("preferred_timeframe")
        if isinstance(recommended_timeframe, str):
            recommended_timeframe = recommended_timeframe.upper()

        return GovernanceDecision(
            scenario=scenario,
            reason=reason,
            adjustments=adjustments,
            recommended_timeframe=recommended_timeframe,
        )

    def decide(
        self,
        account_state: Dict[str, Any],
        *,
        preferred: Optional[str] = None,
    ) -> GovernanceDecision:
        """Return a governance decision using either LLM or fallback logic."""

        if self.llm is None:
            return self._fallback_decision(account_state, preferred=preferred)

        context = {
            "account_balance": account_state.get("account_balance", 0.0),
            "realized_drawdown": account_state.get("realized_drawdown", 0.0),
            "open_positions": account_state.get("open_positions", []),
            "preferred": preferred or "",
            "preferred_timeframe": account_state.get("preferred_timeframe", ""),
        }

        try:
            raw = self.invoke(**context)
        except Exception:
            return self._fallback_decision(account_state, preferred=preferred)

        if not raw:
            return self._fallback_decision(account_state, preferred=preferred)

        try:
            data = self._parse_raw_response(raw)
        except Exception:
            return self._fallback_decision(account_state, preferred=preferred)

        scenario = str(data.get("scenario") or preferred or "intraday").lower()
        reason = str(data.get("reason") or "LLM decision")
        adjustments = {
            key: float(value)
            for key, value in (data.get("adjustments") or {}).items()
            if isinstance(value, (int, float))
        }
        recommended_timeframe = data.get("recommended_timeframe")
        if isinstance(recommended_timeframe, str):
            recommended_timeframe = recommended_timeframe.upper()

        return GovernanceDecision(
            scenario=scenario,
            reason=reason,
            adjustments=adjustments,
            recommended_timeframe=recommended_timeframe,
        )

    @staticmethod
    def _parse_raw_response(raw: str) -> Dict[str, Any]:
        import json

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}


__all__ = ["GovernanceAgent", "GovernanceDecision"]
