"""Multi-agent trading graph.

Several specialised agents each enrich a shared :class:`~agent.state.State`.
For now only Agent 1 is wired in; Agents 2-4 slot into the chain where marked.

    START -> historical_data -> END

Planned agents:
    1. historical_data   -- daily prices & summary        [implemented]
    2. market_news       -- recent news & filings         [todo]
    3. trading_strategy  -- signals from data + news       [todo]
    4. decision          -- final buy / hold / sell call    [todo]
"""

from __future__ import annotations

from langgraph.graph import StateGraph

from agent.historical_agent import historical_data_agent
from agent.state import Context, State

# Build the graph
builder = StateGraph(State, context_schema=Context)

# --- Agent 1: historical data ---
builder.add_node("historical_data", historical_data_agent)
builder.add_edge("__start__", "historical_data")
builder.add_edge("historical_data", "__end__")

# --- Add the next agents here, e.g. ---
# builder.add_node("market_news", market_news_agent)
# builder.add_edge("historical_data", "market_news")   # runs after Agent 1
# builder.add_edge("market_news", "trading_strategy")
# ... then point the last agent at "__end__" instead of historical_data above.

graph = builder.compile(name="Trading Multi-Agent")
