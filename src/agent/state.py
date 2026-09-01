"""Shared state and context for the multi-agent trading graph.

Every agent is a full LLM (ReAct) agent. Each reads the shared state, does its
work with tools, and writes back a short natural-language result that later
agents can reason over:

    Agent 1  historical data   -> historical_analysis
    Agent 2  market news        -> news_analysis   (todo)
    Agent 3  trading strategy   -> strategy        (todo)
    Agent 4  decision           -> decision         (todo)
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, List, Optional

from typing_extensions import TypedDict


class Context(TypedDict, total=False):
    """Runtime-configurable context, shared by every agent."""

    # Override the Sectors API base URL (rarely needed; handy for testing).
    sectors_base_url: str


@dataclass
class State:
    """State shared across all agents in the graph."""

    # --- input ---
    ticker: str = "BBCA"
    start_date: Optional[str] = None  # YYYY-MM-DD, optional
    end_date: Optional[str] = None  # YYYY-MM-DD, optional

    # --- each LLM agent writes its analysis here ---
    historical_analysis: Optional[str] = None  # Agent 1
    news_analysis: Optional[str] = None  # Agent 2 (todo)
    strategy: Optional[str] = None  # Agent 3 (todo)
    decision: Optional[str] = None  # Agent 4 (todo)

    # Errors collected from any agent (reducer lets parallel agents each append).
    errors: Annotated[List[str], operator.add] = field(default_factory=list)
