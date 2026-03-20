"""
config.py - Central configuration for AI bot
All API keys and settings are loaded from environment variables.
Never hardcode API keys here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── LLM Provider API Keys ────────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")

# ─── Web Search API Keys ──────────────────────────────────────────────────────
# Uses SerpAPI for live web search
SERPAPI_API_KEY  = os.getenv("SERPAPI_API_KEY", "")
# Alternatively, Tavily Search API
TAVILY_API_KEY   = os.getenv("TAVILY_API_KEY", "")

# ─── Model Settings ───────────────────────────────────────────────────────────
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "groq")

OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
GROQ_MODELS   = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-8b-8192", "gemma2-9b-it"]
GEMINI_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]

DEFAULT_OPENAI_MODEL  = "gpt-4o-mini"
DEFAULT_GROQ_MODEL    = "llama-3.3-70b-versatile"
DEFAULT_GEMINI_MODEL  = "gemini-1.5-flash"

# ─── RAG Settings ─────────────────────────────────────────────────────────────
CHUNK_SIZE           = 800
CHUNK_OVERLAP        = 100
TOP_K_RESULTS        = 4
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # sentence-transformers model (local, free)
VECTOR_STORE_PATH    = "data/vector_store"

# ─── Response Mode Settings ───────────────────────────────────────────────────
RESPONSE_MODES = {
    "Concise": {
        "description": "Short, summarized replies",
        "max_tokens": 300,
        "system_suffix": (
            "Be concise. Respond in 2–4 sentences maximum. "
            "Get straight to the point. Avoid filler words."
        ),
    },
    "Detailed": {
        "description": "Expanded, in-depth responses",
        "max_tokens": 1500,
        "system_suffix": (
            "Be thorough and detailed. Provide in-depth explanations, "
            "examples, and relevant context. Structure your answer clearly "
            "using headings or bullet points where helpful."
        ),
    },
}

# ─── App Settings ─────────────────────────────────────────────────────────────
APP_TITLE       = "AI bot ⚖️"
APP_SUBTITLE    = "Your AI-Powered Legal Document Assistant"
APP_ICON        = "⚖️"
MAX_CHAT_HISTORY = 20