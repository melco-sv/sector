"""Agent 1: historical data (full LLM / ReAct).

An LLM agent (MiniMax) that calls the ``get_daily_transactions`` tool to fetch
real daily data from Sectors, then writes a short analysis. The LLM decides the
ticker/date arguments and interprets the results; the numbers always come from
the tool, never from the model's imagination.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from langchain.agents import create_agent

from agent.llm import get_llm
from agent.state import State
from agent.tools import get_daily_transactions

SYSTEM_PROMPT = """You are a market-data analyst for the Indonesian Stock Exchange (IDX).

Always use the `get_daily_transactions` tool to obtain real price/volume data.
Never invent or estimate numbers — if the tool returns an error, say so plainly.

Given a ticker, fetch its recent daily data and write a concise analysis (in
Indonesian) covering:
- the price trend and percentage change over the period,
- the period high/low and average volume,
- one or two short, factual observations useful for a later trading decision.

Base every statement strictly on the tool's data."""

# The compiled ReAct agent is cached so the model client is reused across calls.
_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = create_agent(
            get_llm(),
            tools=[get_daily_transactions],
            system_prompt=SYSTEM_PROMPT,
        )
    return _agent


def _strip_thinking(text: str) -> str:
    """Remove MiniMax <think>...</think> blocks from the final answer."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


async def historical_data_agent(state: State) -> Dict[str, Any]:
    """Run the historical-data ReAct agent for ``state.ticker``."""
    ticker = (state.ticker or "").strip().upper()
    if not ticker:
        return {"errors": ['No ticker provided. Set `ticker`, e.g. "BBCA".']}

    task = f"Analyze the recent daily historical data for {ticker}."
    date_range = _describe_range(state.start_date, state.end_date)
    if date_range:
        task += f" {date_range}"

    try:
        result = await _get_agent().ainvoke({"messages": [("user", task)]})
    except Exception as exc:  # noqa: BLE001 - surface any LLM/config error cleanly
        return {"errors": [f"[historical] {type(exc).__name__}: {exc}"]}

    analysis = _strip_thinking(result["messages"][-1].content or "")
    return {"historical_analysis": analysis}


def _describe_range(start: Optional[str], end: Optional[str]) -> str:
    if start and end:
        return f"Use the date range from {start} to {end}."
    if start:
        return f"Start from {start}."
    if end:
        return f"Up to {end}."
    return ""
