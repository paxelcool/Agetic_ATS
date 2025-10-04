"""
Модуль торговой логики ATS.

Содержит агенты, инструменты и графы состояний для торговых решений.
"""

__version__ = "0.1.0"

from .agents.exec_agent import ExecutionAgent, ExecutionReport
from .agents.risk_agent import RiskAgent, RiskReport
from .agents.signal_agent_a import AgentThresholds, IntradaySignalAgent
from .agents.signal_agent_b import SwingSignalAgent, SwingThresholds
from .tools.execution import ExecutionPlan, ExecutionPlanner
from .tools.features import (
    DonchianChannels,
    atr,
    donchian_channels,
    ema,
    enrich_with_indicators,
    relative_volume,
)
from .tools.risk import PositionSizingResult, RiskAssessment, RiskManager, RiskParameters

__all__ = [
    "__version__",
    "AgentThresholds",
    "IntradaySignalAgent",
    "SwingSignalAgent",
    "SwingThresholds",
    "RiskAgent",
    "RiskReport",
    "RiskParameters",
    "RiskAssessment",
    "RiskManager",
    "PositionSizingResult",
    "ExecutionPlanner",
    "ExecutionPlan",
    "ExecutionAgent",
    "ExecutionReport",
    "DonchianChannels",
    "ema",
    "atr",
    "relative_volume",
    "donchian_channels",
    "enrich_with_indicators",
]
