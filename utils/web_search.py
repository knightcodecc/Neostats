"""
utils/web_search.py
Provides real-time web search capability using Tavily API (primary)
and SerpAPI (fallback). Falls back to DuckDuckGo if both are unavailable.
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# ─── Tavily Search ────────────────────────────────────────────────────────────

def search_tavily(query: str, api_key: str, max_results: int = 5) -> List[Dict]:
    """
    Search the web using the Tavily API.

    Returns list of {"title": ..., "url": ..., "content": ...}
    """
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
        )
        results = []
        for r in response.get("results", []):
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "content": r.get("content", ""),
            })
        return results
    except ImportError:
        logger.warning("tavily-python not installed.")
        return []
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return []


# ─── SerpAPI Search ───────────────────────────────────────────────────────────

def search_serpapi(query: str, api_key: str, max_results: int = 5) -> List[Dict]:
    """
    Search the web using SerpAPI (Google results).

    Returns list of {"title": ..., "url": ..., "content": ...}
    """
    try:
        import requests
        params = {
            "q": query,
            "api_key": api_key,
            "engine": "google",
            "num": max_results,
        }
        resp = requests.get("https://serpapi.com/search", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("organic_results", [])[:max_results]:
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("link", ""),
                "content": r.get("snippet", ""),
            })
        return results
    except Exception as e:
        logger.error(f"SerpAPI search error: {e}")
        return []


# ─── DuckDuckGo Fallback ──────────────────────────────────────────────────────

def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict]:
    """
    Fallback web search using DuckDuckGo (no API key required).
    Uses the duckduckgo_search package.
    """
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "content": r.get("body", ""),
                })
        return results
    except ImportError:
        logger.warning("duckduckgo_search not installed.")
        return []
    except Exception as e:
        logger.error(f"DuckDuckGo search error: {e}")
        return []


# ─── Unified Search ───────────────────────────────────────────────────────────

def web_search(
    query: str,
    tavily_key: str = "",
    serpapi_key: str = "",
    max_results: int = 5,
) -> List[Dict]:
    """
    Perform a web search using the best available provider.
    Priority: Tavily → SerpAPI → DuckDuckGo

    Returns list of {"title": ..., "url": ..., "content": ...}
    """
    if tavily_key:
        results = search_tavily(query, tavily_key, max_results)
        if results:
            logger.info(f"Web search via Tavily: {len(results)} results.")
            return results

    if serpapi_key:
        results = search_serpapi(query, serpapi_key, max_results)
        if results:
            logger.info(f"Web search via SerpAPI: {len(results)} results.")
            return results

    # Free fallback
    results = search_duckduckgo(query, max_results)
    logger.info(f"Web search via DuckDuckGo (fallback): {len(results)} results.")
    return results


def format_search_results(results: List[Dict]) -> str:
    """
    Format search results into a context string for the LLM prompt.
    """
    if not results:
        return "No web search results found."

    lines = ["### Live Web Search Results\n"]
    for i, r in enumerate(results, 1):
        title   = r.get("title", "No title")
        url     = r.get("url", "")
        content = r.get("content", "")
        lines.append(f"**[{i}] {title}**")
        if url:
            lines.append(f"🔗 {url}")
        if content:
            lines.append(f"{content[:400]}...")
        lines.append("")

    return "\n".join(lines)


def should_web_search(query: str, threshold_keywords: Optional[List[str]] = None) -> bool:
    """
    Heuristic: decide if the user's query likely requires live web data.
    """
    if threshold_keywords is None:
        threshold_keywords = [
            "latest", "recent", "news", "today", "current", "2024", "2025",
            "update", "new law", "new regulation", "case", "ruling", "verdict",
            "amendment", "bill", "act", "passed", "announced",
        ]
    ql = query.lower()
    return any(kw in ql for kw in threshold_keywords)
