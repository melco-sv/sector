"""LLM factory: builds the MiniMax chat model shared by every agent.

MiniMax exposes an OpenAI-compatible API, so we use langchain-openai's
``ChatOpenAI`` pointed at the MiniMax base URL.

Environment variables:
    MINIMAX_API_KEY   (required) your MiniMax API key
    MINIMAX_MODEL     (optional) default "MiniMax-M3"
    MINIMAX_BASE_URL  (optional) default "https://api.minimax.io/v1"
                      Use "https://api.minimaxi.com/v1" for Mainland China.
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "MiniMax-M3"
DEFAULT_BASE_URL = "https://api.minimax.io/v1"


def get_llm(*, temperature: float = 0.3) -> ChatOpenAI:
    """Return a MiniMax chat model configured from the environment.

    Raises RuntimeError if MINIMAX_API_KEY is missing so the failure is clear.
    """
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY is not set. Add it to your .env file.")

    return ChatOpenAI(
        model=os.environ.get("MINIMAX_MODEL", DEFAULT_MODEL),
        base_url=os.environ.get("MINIMAX_BASE_URL", DEFAULT_BASE_URL),
        api_key=api_key,
        temperature=temperature,
    )
