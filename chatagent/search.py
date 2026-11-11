"""The agent's single tool: web search.

Pluggable backend. If TAVILY_API_KEY is set we use Tavily (cleaner, ranked
results); otherwise we fall back to DuckDuckGo, which needs no API key so the
agent works the moment you clone it. Results are cached in SQLite (TTL) by the
Memory layer, so repeat queries return instantly.

`web_search` returns a compact, model-friendly string: numbered results with
title, URL and snippet, so the model can cite sources.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from . import config

if TYPE_CHECKING:
    from .memory import Memory

# The tool schema handed to Claude. Exactly one tool — per the brief.
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web for current, factual, or niche information you are not "
        "confident about. Use it for anything time-sensitive (news, prices, "
        "releases, 'latest'), for facts you might get wrong, and to verify "
        "claims. Do NOT use it for questions about this conversation's own "
        "history or for chit-chat. Returns ranked results with titles, URLs and "
        "snippets to cite."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A focused search query, as you'd type into a search engine.",
            }
        },
        "required": ["query"],
    },
}


def _search_tavily(query: str, max_results: int) -> list[dict]:
    from tavily import TavilyClient  # imported lazily so it's optional

    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    resp = client.search(query=query, max_results=max_results)
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
        for r in resp.get("results", [])
    ]


def _search_duckduckgo(query: str, max_results: int) -> list[dict]:
    # The library was renamed from duckduckgo_search to ddgs; support both.
    try:
        from ddgs import DDGS
    except ImportError:  # pragma: no cover - depends on installed version
        from duckduckgo_search import DDGS  # type: ignore

    out: list[dict] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            out.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href") or r.get("url", ""),
                    "snippet": r.get("body", ""),
                }
            )
    return out


def raw_search(query: str, memory: "Memory | None" = None) -> list[dict]:
    """Return a list of {title, url, snippet}, using the cache when possible."""
    if memory is not None:
        cached = memory.cache_get(query, config.SEARCH_CACHE_TTL)
        if cached is not None:
            return cached

    backend = _search_tavily if config.TAVILY_API_KEY else _search_duckduckgo
    try:
        results = backend(query, config.MAX_SEARCH_RESULTS)
    except Exception as exc:  # never crash the chat loop on a search hiccup
        return [{"title": "search error", "url": "", "snippet": f"{type(exc).__name__}: {exc}"}]

    if memory is not None and results:
        memory.cache_set(query, results)
    return results


def web_search(query: str, memory: "Memory | None" = None) -> str:
    """Tool entry point — returns a formatted string for the tool_result."""
    results = raw_search(query, memory)
    if not results:
        return f"No results found for: {query!r}"
    lines = [f"Search results for {query!r}:"]
    for i, r in enumerate(results, 1):
        lines.append(f"\n[{i}] {r['title']}\n{r['url']}\n{r['snippet']}")
    return "\n".join(lines)
