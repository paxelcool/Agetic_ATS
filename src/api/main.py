"""FastAPI application exposing LangGraph orchestrations via LangServe."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI
from langserve import add_routes
from pydantic import BaseModel, Field

from atl.agents.governance_agent import GovernanceAgent
from atl.graphs import (
    build_intraday_graph,
    build_swing_graph,
    default_intraday_dependencies,
    default_swing_dependencies,
)
from atl.graphs.common import serialize_governance
from src.database.models import Timeframe


class SignalGraphRequest(BaseModel):
    """Payload accepted by graph endpoints."""

    symbol: str = Field(..., description="Trading symbol, e.g. XAUUSD")
    timeframe: Timeframe = Field(Timeframe.M15, description="Requested timeframe")
    market_data: list[Dict[str, Any]] = Field(
        ..., description="Historical bars with open/high/low/close/volume"
    )
    account_state: Dict[str, Any] = Field(default_factory=dict)
    latest_quote: Optional[Dict[str, Any]] = Field(
        default=None, description="Latest quote snapshot"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "XAUUSD",
                "timeframe": "H1",
                "market_data": [
                    {
                        "open": 2000.0,
                        "high": 2005.0,
                        "low": 1995.0,
                        "close": 2002.0,
                        "volume": 1200,
                    }
                ],
                "account_state": {"account_balance": 12000, "market_regime": "trending"},
            }
        }
    }


class AutoSignalResponse(BaseModel):
    scenario: str
    decision: Dict[str, Any]
    result: Dict[str, Any]


def _prepare_graph_input(payload: SignalGraphRequest) -> Dict[str, Any]:
    data = payload.model_dump(mode="json")
    data["timeframe"] = payload.timeframe.value
    return data


def create_app() -> FastAPI:
    """Instantiate FastAPI application with LangServe routes."""

    app = FastAPI(title="Agetic ATS API", version="0.1.0")

    governance = GovernanceAgent()
    intraday_graph = build_intraday_graph(default_intraday_dependencies(governance))
    swing_graph = build_swing_graph(default_swing_dependencies(governance))

    add_routes(app, intraday_graph, path="/signal/intraday")
    add_routes(app, swing_graph, path="/signal/swing")

    @app.post("/signal/auto", response_model=AutoSignalResponse)
    async def auto_signal(payload: SignalGraphRequest) -> AutoSignalResponse:
        decision = governance.decide(payload.account_state)
        graph = intraday_graph if decision.scenario == "intraday" else swing_graph
        graph_input = _prepare_graph_input(payload)
        result_state = await graph.ainvoke(graph_input)
        result_payload = result_state.get("result") or result_state
        return AutoSignalResponse(
            scenario=decision.scenario,
            decision=serialize_governance(decision) or {},
            result=result_payload,
        )

    @app.get("/")
    async def root() -> Dict[str, Any]:
        return {
            "service": "Agetic ATS",
            "endpoints": [
                "/signal/intraday/invoke",
                "/signal/intraday/stream",
                "/signal/swing/invoke",
                "/signal/swing/stream",
                "/signal/auto",
            ],
        }

    return app


app = create_app()
