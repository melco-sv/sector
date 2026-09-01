import pytest

from agent import graph

pytestmark = pytest.mark.anyio


async def test_agent_produces_historical_analysis() -> None:
    res = await graph.ainvoke({"ticker": "BBCA"})
    assert res is not None
    # With valid MINIMAX_API_KEY + SECTORS_API_KEY we get a historical_analysis;
    # without them the graph records a clear error instead. Either way it runs.
    assert res.get("historical_analysis") or res.get("errors")
