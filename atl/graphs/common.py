"""Shared utilities for LangGraph orchestrations."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Optional, TypedDict

import numpy as np
import pandas as pd
from langchain_core.runnables import RunnableLambda, RunnableParallel

from src.database.models import Quote, Signal, SignalAction, Timeframe

from ..agents.exec_agent import ExecutionAgent, ExecutionReport
from ..agents.governance_agent import GovernanceAgent, GovernanceDecision
from ..agents.risk_agent import RiskAgent, RiskReport
from ..tools.execution import ExecutionPlan
from ..tools.features import enrich_with_indicators
from ..tools.risk import RiskAssessment, RiskManager, RiskParameters


class TradeGraphState(TypedDict, total=False):
    """State shared between graph nodes."""

    symbol: str
    timeframe: Timeframe
    market_data: Any
    market_frame: Any
    features: Dict[str, Any]
    stats: Dict[str, float]
    signal: Any
    risk_assessment: Any
    risk_report: Any
    execution_plan: Any
    execution_report: Any
    latest_quote: Any
    account_state: Dict[str, Any]
    logs: list[str]
    scenario: str
    governance: Any
    agents: Any
    result: Dict[str, Any]


@dataclass
class AgentBundle:
    """Container with instantiated agents for a graph run."""

    signal: Any
    risk: RiskAgent
    execution: ExecutionAgent
    risk_parameters: RiskParameters
    risk_manager: RiskManager


@dataclass
class GraphDependencies:
    """Dependency factories required by both graphs."""

    governance_agent: GovernanceAgent
    build_risk_parameters: Callable[[Dict[str, Any], Dict[str, float]], RiskParameters]
    make_signal_agent: Callable[[RiskManager, Dict[str, float]], Any]
    make_risk_agent: Callable[[RiskParameters], RiskAgent]
    make_execution_agent: Callable[[RiskManager], ExecutionAgent]


def ensure_timeframe(value: Any, default: Timeframe = Timeframe.M15) -> Timeframe:
    """Convert external input into :class:`Timeframe`."""

    if isinstance(value, Timeframe):
        return value
    if isinstance(value, str):
        try:
            return Timeframe(value)
        except ValueError:
            upper = value.upper()
            try:
                return Timeframe(upper)
            except ValueError:
                return default
    return default


def prepare_quote(data: Any) -> Optional[Quote]:
    if data is None:
        return None
    if isinstance(data, Quote):
        return data
    if isinstance(data, dict):
        try:
            return Quote.model_validate(data)
        except Exception:
            return None
    return None


def prepare_market_frame(data: Any) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, list):
        frame = pd.DataFrame(data)
    elif isinstance(data, dict):
        frame = pd.DataFrame(data)
    else:
        frame = pd.DataFrame()
    return frame


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if pd.isna(value):  # type: ignore[arg-type]
        return None
    return value


def compute_feature_set(frame: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, float]]:
    """Compute indicators and descriptive statistics in parallel."""

    def _enrich(data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data
        return enrich_with_indicators(data)

    def _statistics(data: pd.DataFrame) -> Dict[str, float]:
        if data.empty:
            return {}
        tail = data.tail(30)
        stats: Dict[str, float] = {}
        for column in {"atr", "rvol"} & set(tail.columns):
            series = tail[column].astype(float)
            stats[f"{column}_mean"] = float(series.mean())
            stats[f"{column}_recent"] = float(series.iloc[-1])
        return stats

    pipeline = RunnableParallel(
        enriched=RunnableLambda(lambda data: _enrich(data)),
        stats=RunnableLambda(lambda data: _statistics(data)),
    )
    result = pipeline.invoke(frame)
    enriched = result["enriched"].copy() if isinstance(result["enriched"], pd.DataFrame) else frame
    stats = dict(result.get("stats") or {})
    return enriched, {key: float(value) for key, value in stats.items()}


def latest_feature_row(frame: pd.DataFrame) -> Dict[str, Any]:
    if frame.empty:
        return {}
    row = frame.iloc[-1].to_dict()
    return {key: _sanitize_value(value) for key, value in row.items()}


def apply_signal_defaults(signal: Signal, features: Dict[str, Any]) -> Signal:
    if signal.entry_price is None and "close" in features:
        try:
            signal.entry_price = float(features["close"])
        except (TypeError, ValueError):
            pass
    if signal.side and signal.stop_loss is None and "atr" in features:
        atr_value = features.get("atr")
        try:
            atr_float = float(atr_value)
        except (TypeError, ValueError):
            atr_float = None
        if atr_float and signal.entry_price:
            direction = 1 if signal.side == signal.side.BUY else -1
            signal.stop_loss = signal.entry_price - direction * atr_float
    return signal


def serialize_plan(plan: Optional[ExecutionPlan]) -> Optional[Dict[str, Any]]:
    if plan is None:
        return None
    trade = plan.trade.dict() if hasattr(plan.trade, "dict") else {}
    metadata = dict(plan.metadata) if isinstance(plan.metadata, dict) else {}
    return {
        "trade": trade,
        "assessment": asdict(plan.assessment) if isinstance(plan.assessment, RiskAssessment) else {},
        "order_type": plan.order_type.value,
        "slippage": float(plan.slippage),
        "metadata": metadata,
    }


def serialize_execution_report(report: Optional[ExecutionReport]) -> Optional[Dict[str, Any]]:
    if report is None:
        return None
    payload = asdict(report)
    if hasattr(report.trade, "dict"):
        payload["trade"] = report.trade.dict()
    if isinstance(payload.get("errors"), tuple):
        payload["errors"] = list(payload["errors"])
    if isinstance(payload.get("warnings"), tuple):
        payload["warnings"] = list(payload["warnings"])
    return payload


def serialize_risk_assessment(assessment: Optional[RiskAssessment]) -> Optional[Dict[str, Any]]:
    if assessment is None:
        return None
    return asdict(assessment)


def serialize_risk_report(report: Optional[RiskReport]) -> Optional[Dict[str, Any]]:
    if report is None:
        return None
    payload = asdict(report)
    payload["violations"] = list(payload.get("violations", []))
    payload["recommendations"] = list(payload.get("recommendations", []))
    payload["portfolio_risk"] = dict(payload.get("portfolio_risk", {}))
    return payload


def serialize_governance(decision: Optional[GovernanceDecision]) -> Optional[Dict[str, Any]]:
    if decision is None:
        return None
    payload = asdict(decision)
    payload["adjustments"] = dict(payload.get("adjustments", {}))
    return payload


def serialize_signal(signal: Optional[Signal]) -> Optional[Dict[str, Any]]:
    if signal is None:
        return None
    if hasattr(signal, "model_dump"):
        return signal.model_dump(exclude_none=True)
    if hasattr(signal, "dict"):
        return signal.dict(exclude_none=True)
    return None


def serialize_quote(quote: Optional[Quote]) -> Optional[Dict[str, Any]]:
    if quote is None:
        return None
    if hasattr(quote, "model_dump"):
        return quote.model_dump(exclude_none=True)
    if hasattr(quote, "dict"):
        return quote.dict(exclude_none=True)
    return None


def signal_requires_management(signal: Optional[Signal]) -> bool:
    if signal is None:
        return False
    try:
        action = signal.action
    except AttributeError:
        return False
    if isinstance(action, SignalAction):
        return action == SignalAction.ENTER
    try:
        return SignalAction(action) == SignalAction.ENTER
    except Exception:
        return False


def merge_adjustments(*dicts: Dict[str, float]) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    for item in dicts:
        for key, value in item.items():
            try:
                merged[key] = float(value)
            except (TypeError, ValueError):
                continue
    return merged


__all__ = [
    "AgentBundle",
    "GraphDependencies",
    "TradeGraphState",
    "apply_signal_defaults",
    "compute_feature_set",
    "ensure_timeframe",
    "latest_feature_row",
    "merge_adjustments",
    "prepare_market_frame",
    "prepare_quote",
    "serialize_execution_report",
    "serialize_governance",
    "serialize_plan",
    "serialize_risk_assessment",
    "serialize_risk_report",
    "serialize_signal",
    "serialize_quote",
    "signal_requires_management",
]
