"""Thin async client for the Sectors Financial API (v2).

Shared by every agent. The API key is read from the ``SECTORS_API_KEY``
environment variable, which LangGraph loads from ``.env``.

Docs: https://docs.sectors.app
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

# Base URL for the Sectors Financial API (v2). v1 was discontinued 2026-05-11.
SECTORS_BASE_URL = "https://api.sectors.app/v2"


class SectorsError(RuntimeError):
    """Raised when the Sectors API is unreachable or returns a non-200 status."""


async def sectors_get(
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    base_url: str = SECTORS_BASE_URL,
) -> Any:
    """GET a Sectors API endpoint and return the parsed JSON.

    ``path`` is appended to the base URL, e.g. ``"daily/BBCA/"``.
    Raises :class:`SectorsError` on a missing key, network failure, or non-200
    response so callers can surface a clean message.
    """
    api_key = os.environ.get("SECTORS_API_KEY")
    if not api_key:
        raise SectorsError("SECTORS_API_KEY is not set. Add it to your .env file.")

    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    headers = {"Authorization": api_key}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=headers, params=params)
    except httpx.HTTPError as exc:
        raise SectorsError(f"Request to Sectors failed: {exc}") from exc

    if resp.status_code != 200:
        raise SectorsError(
            f"Sectors API returned {resp.status_code}: {resp.text[:300]}"
        )
    return resp.json()
