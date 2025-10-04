"""LangGraph orchestration for scenario B (swing/trend)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph

from src.config import settings
from src.database.models import Timeframe

from ..agents.exec_agent import ExecutionAgent
from ..agents.governance_agent import GovernanceAgent
from ..agents.risk_agent import RiskAgent
from ..agents.signal_agent_b import SwingSignalAgent, SwingThresholds
from ..tools.execution import ExecutionPlanner
from ..tools.risk import RiskManager, RiskParameters
from .common import (
    AgentBundle,
    GraphDependencies,
    TradeGraphState,
    apply_signal_defaults,
    compute_feature_set,
    ensure_timeframe,
    latest_feature_row,
    merge_adjustments,
    prepare_market_frame,
    prepare_quote,
    serialize_execution_report,
    serialize_governance,
    serialize_plan,
    serialize_quote,
    serialize_risk_assessment,
    serialize_risk_report,
    serialize_signal,
    signal_requires_management,
)


@dataclass
class SwingGraphConfig:
    default_balance: float = 15_000.0
    contract_size: float = 1.0
    min_position: float = 0.1
    max_position: float = 8.0
    base_risk_per_trade: float = max(settings.account_risk_per_trade * 0.8, 0.005)


def _apply_swing_adjustments(agent: SwingSignalAgent, adjustments: Dict[str, float]) -> None:
    thresholds = agent.thresholds
    if "donchian_high" in adjustments:
        thresholds.donchian_high = max(20, int(adjustments["donchian_high"]))
    if "donchian_low" in adjustments:
        thresholds.donchian_low = max(10, int(adjustments["donchian_low"]))
    if "rvol_threshold" in adjustments:
        thresholds.rvol_threshold = max(1.0, float(adjustments["rvol_threshold"]))
    if "atr_multiplier" in adjustments:
        thresholds.atr_multiplier = max(1.0, float(adjustments["atr_multiplier"]))
    if "risk_per_trade" in adjustments:
        thresholds.risk_percent = max(0.2, float(adjustments["risk_per_trade"]) * 100)


def _derive_swing_adjustments(stats: Dict[str, float], bundle: AgentBundle) -> Dict[str, float]:
    adjustments: Dict[str, float] = {}
    atr_recent = stats.get("atr_recent")
    atr_mean = stats.get("atr_mean")
    rvol_recent = stats.get("rvol_recent")

    if atr_recent and atr_mean:
        try:
            ratio = max(0.5, min(2.5, float(atr_recent) / float(atr_mean)))
            adjustments["atr_multiplier"] = round(max(1.0, min(4.0, bundle.signal.thresholds.atr_multiplier * ratio)), 2)
            adjustments["risk_per_trade"] = round(
                max(0.003, min(0.03, bundle.risk_parameters.risk_per_trade / max(ratio, 0.75))),
                4,
            )
            adjustments["donchian_high"] = max(30, int(bundle.signal.thresholds.donchian_high * ratio))
            adjustments["donchian_low"] = max(15, int(bundle.signal.thresholds.donchian_low * ratio))
        except (TypeError, ValueError):
            pass

    if rvol_recent:
        try:
            recent = float(rvol_recent)
            adjustments["rvol_threshold"] = round(max(1.0, min(3.0, (recent + 1.5) / 2)), 2)
        except (TypeError, ValueError):
            pass

    return adjustments


def _update_risk(bundle: AgentBundle, adjustments: Dict[str, float]) -> None:
    if "risk_per_trade" in adjustments:
        bundle.risk_parameters.risk_per_trade = max(0.002, float(adjustments["risk_per_trade"]))
    if "max_position" in adjustments:
        bundle.risk_parameters.max_position = float(adjustments["max_position"])


def default_swing_dependencies(
    governance: Optional[GovernanceAgent] = None,
    *,
    config: Optional[SwingGraphConfig] = None,
) -> GraphDependencies:
    cfg = config or SwingGraphConfig()
    governance_agent = governance or GovernanceAgent()

    def build_risk_parameters(account_state: Dict[str, Any], adjustments: Dict[str, float]) -> RiskParameters:
        balance = float(account_state.get("account_balance", cfg.default_balance) or cfg.default_balance)
        risk_per_trade = float(
            adjustments.get("risk_per_trade", account_state.get("risk_per_trade", cfg.base_risk_per_trade))
        )
        max_position = adjustments.get("max_position") or account_state.get("max_position") or cfg.max_position
        params = RiskParameters(
            account_balance=balance,
            risk_per_trade=risk_per_trade,
            contract_size=cfg.contract_size,
            min_position=cfg.min_position,
            max_position=max_position,
            reward_risk=2.5,
            atr_multiplier=2.0,
        )
        return params

    def make_signal_agent(risk_manager: RiskManager, adjustments: Dict[str, float]) -> SwingSignalAgent:
        thresholds = SwingThresholds()
        _apply_swing_adjustments_placeholder = {
            key: float(value)
            for key, value in adjustments.items()
            if key
            in {"donchian_high", "donchian_low", "rvol_threshold", "atr_multiplier", "risk_per_trade"}
        }
        for key, value in _apply_swing_adjustments_placeholder.items():
            if key == "donchian_high":
                thresholds.donchian_high = max(20, int(value))
            elif key == "donchian_low":
                thresholds.donchian_low = max(10, int(value))
            elif key == "rvol_threshold":
                thresholds.rvol_threshold = max(1.0, value)
            elif key == "atr_multiplier":
                thresholds.atr_multiplier = max(1.0, value)
            elif key == "risk_per_trade":
                thresholds.risk_percent = max(0.2, value * 100)
        return SwingSignalAgent(thresholds=thresholds, risk_manager=risk_manager)

    def make_risk_agent(params: RiskParameters) -> RiskAgent:
        return RiskAgent(params)

    def make_execution_agent(risk_manager: RiskManager) -> ExecutionAgent:
        return ExecutionAgent(ExecutionPlanner(risk_manager))

    return GraphDependencies(
        governance_agent=governance_agent,
        build_risk_parameters=build_risk_parameters,
        make_signal_agent=make_signal_agent,
        make_risk_agent=make_risk_agent,
        make_execution_agent=make_execution_agent,
    )


def build_swing_graph(dependencies: Optional[GraphDependencies] = None) -> Any:
    deps = dependencies or default_swing_dependencies()
    graph = StateGraph(TradeGraphState)

    def idle(state: TradeGraphState) -> TradeGraphState:
        logs = list(state.get("logs", []))
        account_state = state.get("account_state") or {}
        decision = deps.governance_agent.decide(account_state, preferred="swing")
        default_tf = ensure_timeframe(decision.recommended_timeframe or Timeframe.H4, default=Timeframe.H4)
        timeframe = ensure_timeframe(state.get("timeframe"), default=default_tf)
        adjustments = dict(decision.adjustments)
        risk_parameters = deps.build_risk_parameters(account_state, adjustments)
        risk_manager = RiskManager(risk_parameters)
        signal_agent = deps.make_signal_agent(risk_manager, adjustments)
        risk_agent = deps.make_risk_agent(risk_parameters)
        execution_agent = deps.make_execution_agent(risk_manager)
        bundle = AgentBundle(
            signal=signal_agent,
            risk=risk_agent,
            execution=execution_agent,
            risk_parameters=risk_parameters,
            risk_manager=risk_manager,
        )
        logs.append(f"Governance scenario: {decision.scenario}")
        frame = prepare_market_frame(state.get("market_data"))
        quote = prepare_quote(state.get("latest_quote"))
        return {
            "logs": logs,
            "governance": decision,
            "agents": bundle,
            "scenario": decision.scenario,
            "market_frame": frame,
            "timeframe": timeframe,
            "latest_quote": quote,
            "adjustments": adjustments,
        }

    def setup(state: TradeGraphState) -> TradeGraphState:
        logs = list(state.get("logs", []))
        frame = state.get("market_frame")
        if frame is None or frame.empty:
            logs.append("Недостаточно рыночных данных для swing-графа")
            return {"logs": logs}
        enriched, stats = compute_feature_set(frame)
        features = latest_feature_row(enriched)
        bundle: AgentBundle = state["agents"]
        stat_adjustments = _derive_swing_adjustments(stats, bundle)
        combined_adjustments = merge_adjustments(state.get("adjustments", {}), stat_adjustments)
        _apply_swing_adjustments(bundle.signal, combined_adjustments)
        _update_risk(bundle, combined_adjustments)
        logs.append("Swing-индикаторы обновлены")
        return {
            "logs": logs,
            "market_frame": enriched,
            "features": features,
            "stats": stats,
            "adjustments": combined_adjustments,
        }

    def enter(state: TradeGraphState) -> TradeGraphState:
        logs = list(state.get("logs", []))
        bundle: AgentBundle = state["agents"]
        features = state.get("features") or {}
        signal = bundle.signal.generate_signal(
            symbol=state.get("symbol", "UNKNOWN"),
            timeframe=state.get("timeframe", Timeframe.H4),
            indicators=features,
            portfolio_state=(state.get("account_state") or {}).get("portfolio_state", {}),
            market_regime=(state.get("account_state") or {}).get("market_regime"),
        )
        signal = apply_signal_defaults(signal, features)
        action_value = getattr(signal.action, "value", signal.action)
        logs.append(f"Swing сигнал: {action_value}")
        return {"logs": logs, "signal": signal}

    def manage(state: TradeGraphState) -> TradeGraphState:
        logs = list(state.get("logs", []))
        signal = state.get("signal")
        if not signal_requires_management(signal):
            logs.append("Позиция не требует управления")
            return {"logs": logs}
        bundle: AgentBundle = state["agents"]
        features = state.get("features") or {}
        atr_value = features.get("atr")
        risk_assessment = bundle.risk.assess_signal(signal, atr_value=atr_value)
        risk_report = bundle.risk.analyze_account(state.get("account_state") or {})
        logs.append("Оценка swing-рисков выполнена")
        return {
            "logs": logs,
            "risk_assessment": risk_assessment,
            "risk_report": risk_report,
        }

    def exit_state(state: TradeGraphState) -> TradeGraphState:
        logs = list(state.get("logs", []))
        signal = state.get("signal")
        if not signal_requires_management(signal):
            logs.append("Исполнение swing сигнала не требуется")
            return {"logs": logs}
        bundle: AgentBundle = state["agents"]
        features = state.get("features") or {}
        plan = bundle.execution.build_plan(
            signal,
            quote=state.get("latest_quote"),
            atr_value=features.get("atr"),
        )
        report = bundle.execution.report(
            plan,
            account_info=state.get("account_state"),
            market_data={"features": features, "stats": state.get("stats", {})},
            risk_limits=serialize_risk_report(state.get("risk_report")) or {},
            execution_history=(state.get("account_state") or {}).get("execution_history"),
        )
        logs.append("План swing-исполнения готов")
        return {
            "logs": logs,
            "execution_plan": plan,
            "execution_report": report,
        }

    def summarize(state: TradeGraphState) -> TradeGraphState:
        result = {
            "symbol": state.get("symbol"),
            "timeframe": state.get("timeframe").value if isinstance(state.get("timeframe"), Timeframe) else state.get("timeframe"),
            "scenario": state.get("scenario"),
            "governance": serialize_governance(state.get("governance")),
            "features": state.get("features") or {},
            "stats": state.get("stats") or {},
            "adjustments": state.get("adjustments") or {},
            "latest_quote": serialize_quote(state.get("latest_quote")),
            "signal": serialize_signal(state.get("signal")),
            "risk_assessment": serialize_risk_assessment(state.get("risk_assessment")),
            "risk_report": serialize_risk_report(state.get("risk_report")),
            "execution_plan": serialize_plan(state.get("execution_plan")),
            "execution_report": serialize_execution_report(state.get("execution_report")),
            "logs": state.get("logs", []),
        }
        return {
            "result": result,
            "signal": result["signal"],
            "risk_assessment": result["risk_assessment"],
            "risk_report": result["risk_report"],
            "execution_plan": result["execution_plan"],
            "execution_report": result["execution_report"],
            "governance": result["governance"],
            "features": result["features"],
            "stats": result["stats"],
            "latest_quote": result["latest_quote"],
            "market_frame": None,
            "agents": None,
        }

    def route_after_enter(state: TradeGraphState) -> str:
        signal = state.get("signal")
        if signal_requires_management(signal):
            return "manage"
        return "skip"

    graph.add_node("IDLE", idle)
    graph.add_node("SETUP", setup)
    graph.add_node("ENTER", enter)
    graph.add_node("MANAGE", manage)
    graph.add_node("EXIT", exit_state)
    graph.add_node("SUMMARY", summarize)

    graph.set_entry_point("IDLE")
    graph.add_edge("IDLE", "SETUP")
    graph.add_edge("SETUP", "ENTER")
    graph.add_conditional_edges(
        "ENTER",
        route_after_enter,
        {"manage": "MANAGE", "skip": "SUMMARY"},
    )
    graph.add_edge("MANAGE", "EXIT")
    graph.add_edge("EXIT", "SUMMARY")
    graph.add_edge("SUMMARY", END)

    return graph.compile()


__all__ = ["build_swing_graph", "default_swing_dependencies", "SwingGraphConfig"]
