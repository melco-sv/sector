"""Tools the LLM agents can call.

Each tool wraps a Sectors API endpoint so the model works with real data
instead of guessing numbers. Add more tools here for the other agents
(news, filings, financials, ...).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from langchain_core.tools import tool

from agent.sectors import SectorsError, sectors_get


@tool
async def get_daily_transactions(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[List[Dict[str, Any]], str]:
    """Get daily close price, volume, and market cap for an IDX stock.

    Args:
        ticker: IDX ticker symbol, e.g. "BBCA", "TLKM", "GOTO".
        start_date: optional start date "YYYY-MM-DD" (defaults to ~30 days ago).
        end_date: optional end date "YYYY-MM-DD" (defaults to today).

    Returns a list of {symbol, date, close, volume, market_cap} sorted oldest to
    newest. The API caps the range at the most recent 90 days.
    """
    ticker = (ticker or "").strip().upper()
    params: Dict[str, str] = {}
    if start_date:
        params["start"] = start_date
    if end_date:
        params["end"] = end_date

    try:
        rows = await sectors_get(f"daily/{ticker}/", params=params)
    except SectorsError as exc:
        return f"Error fetching data for {ticker}: {exc}"

    if isinstance(rows, list):
        return sorted(rows, key=lambda r: r.get("date", ""))
    return rows
