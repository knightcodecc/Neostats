"""
utils/chat_utils.py
Helper functions for building prompts, managing chat history,
and constructing the final message list sent to the LLM.
"""

import logging
from typing import List, Dict, Optional
from config.config import MAX_CHAT_HISTORY, RESPONSE_MODES

logger = logging.getLogger(__name__)


# ─── System Prompts ───────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """You are AI bot ⚖️, an expert AI Legal Document Assistant.

Your role is to:
• Analyze and explain legal documents, contracts, agreements, and clauses in plain language
• Identify potential risks, obligations, and rights in legal texts
• Answer questions about legal concepts, procedures, and terminology
• Summarize lengthy legal documents clearly
• Highlight important deadlines, penalties, and conditions
• Provide general legal information (NOT legal advice — always recommend consulting a licensed attorney for specific situations)

Guidelines:
• Always be accurate, clear, and neutral
• Flag clauses that may be unfavorable to the user
• Use plain English to explain complex legal jargon
• When uncertain, say so and recommend professional legal consultation
• If given document excerpts via RAG context, base your answers primarily on that content
• If given web search results, cite the source and date when relevant
• Never fabricate legal cases, statutes, or facts

⚠️ DISCLAIMER: I provide general legal information only. This is not legal advice. 
Always consult a qualified attorney for advice specific to your situation.
"""


def build_system_prompt(response_mode: str = "Detailed") -> str:
    """Combine base system prompt with the selected response mode instruction."""
    mode_config = RESPONSE_MODES.get(response_mode, RESPONSE_MODES["Detailed"])
    return BASE_SYSTEM_PROMPT + "\n\n**Response Style:** " + mode_config["system_suffix"]


def build_user_message(
    user_query: str,
    rag_context: str = "",
    web_context: str = "",
) -> str:
    """
    Construct the full user message, injecting RAG and/or web context if available.
    """
    parts = []

    if rag_context:
        parts.append(rag_context)
        parts.append("---")

    if web_context:
        parts.append(web_context)
        parts.append("---")

    parts.append(f"**User Question:** {user_query}")

    if rag_context:
        parts.append(
            "\n*Please base your answer on the document excerpts provided above. "
            "If the answer isn't in the documents, say so clearly.*"
        )

    return "\n\n".join(parts)


# ─── History Management ───────────────────────────────────────────────────────

def trim_history(history: List[Dict], max_turns: int = MAX_CHAT_HISTORY) -> List[Dict]:
    """
    Keep only the most recent `max_turns` message pairs to avoid
    exceeding context window limits.
    """
    # Always keep system prompt (index 0)
    if len(history) <= 1:
        return history

    non_system = [m for m in history if m["role"] != "system"]
    system_msgs = [m for m in history if m["role"] == "system"]

    # Keep last max_turns messages
    trimmed = non_system[-max_turns:]
    return system_msgs + trimmed


def build_messages(
    history: List[Dict],
    user_query: str,
    response_mode: str = "Detailed",
    rag_context: str = "",
    web_context: str = "",
) -> List[Dict]:
    """
    Assemble the full message list to send to the LLM.

    Args:
        history:       Previous chat messages (role + content)
        user_query:    The latest user input
        response_mode: "Concise" or "Detailed"
        rag_context:   Retrieved document text
        web_context:   Live web search results text

    Returns:
        List of message dicts ready for the LLM API
    """
    try:
        system_prompt = build_system_prompt(response_mode)
        user_content  = build_user_message(user_query, rag_context, web_context)

        # Trim old history
        trimmed = trim_history(history)

        messages = [{"role": "system", "content": system_prompt}]
        for msg in trimmed:
            if msg["role"] in ("user", "assistant"):
                messages.append(msg)

        messages.append({"role": "user", "content": user_content})
        return messages
    except Exception as e:
        logger.error(f"Error building messages: {e}")
        return [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]


# ─── Display Helpers ──────────────────────────────────────────────────────────

def format_source_badges(rag_used: bool, web_used: bool) -> str:
    """Return emoji badges indicating which sources were used."""
    badges = []
    if rag_used:
        badges.append("📄 Document context used")
    if web_used:
        badges.append("🌐 Live web search used")
    return " · ".join(badges) if badges else ""
