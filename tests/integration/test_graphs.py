from __future__ import annotations

from typing import Dict, List

import pytest
from fastapi.testclient import TestClient

from atl.agents.governance_agent import GovernanceAgent
from atl.graphs import (
    build_intraday_graph,
    build_swing_graph,
    default_intraday_dependencies,
    default_swing_dependencies,
)
from src.api.main import create_app


def _make_market_data(length: int = 60) -> List[Dict[str, float]]:
    data: List[Dict[str, float]] = []
    price = 2000.0
    for idx in range(length):
        open_price = price + idx * 0.2
        high = open_price + 2.0
        low = open_price - 2.0
        close = open_price + 0.5
        volume = 1000 + idx * 5
        data.append(
            {
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": volume,
            }
        )
    return data


@pytest.mark.parametrize("account_balance", [3500, 8000])
def test_governance_decision_switch(account_balance: float) -> None:
    agent = GovernanceAgent()
    decision = agent.decide({"account_balance": account_balance})
    assert decision.scenario in {"intraday", "swing"}
    if account_balance >= 5000:
        assert decision.scenario == "swing"
    else:
        assert decision.scenario == "intraday"


def test_intraday_graph_runs_with_sample_data() -> None:
    governance = GovernanceAgent()
    graph = build_intraday_graph(default_intraday_dependencies(governance))
    payload = {
        "symbol": "XAUUSD",
        "timeframe": "M15",
        "market_data": _make_market_data(),
        "account_state": {"account_balance": 4000, "market_regime": "trending"},
        "latest_quote": {"symbol": "XAUUSD", "timestamp": 1700000000, "bid": 2001.0, "ask": 2001.5},
    }
    state = graph.invoke(payload)
    assert "result" in state
    result = state["result"]
    assert isinstance(result, dict)
    assert result.get("signal") is not None
    assert result.get("features")


def test_swing_graph_runs_with_sample_data() -> None:
    governance = GovernanceAgent()
    graph = build_swing_graph(default_swing_dependencies(governance))
    payload = {
        "symbol": "US100",
        "timeframe": "H4",
        "market_data": _make_market_data(),
        "account_state": {"account_balance": 15000, "market_regime": "trending"},
        "latest_quote": {"symbol": "US100", "timestamp": 1700000000, "bid": 15000.0, "ask": 15001.0},
    }
    state = graph.invoke(payload)
    assert "result" in state
    result = state["result"]
    assert isinstance(result, dict)
    assert result.get("signal") is not None
    assert result.get("features")


def test_api_auto_endpoint_returns_payload() -> None:
    app = create_app()
    client = TestClient(app)
    payload = {
        "symbol": "XAUUSD",
        "timeframe": "M15",
        "market_data": _make_market_data(),
        "account_state": {"account_balance": 4500},
    }
    response = client.post("/signal/auto", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "scenario" in data
    assert "result" in data
    assert data["result"].get("signal") is not None
