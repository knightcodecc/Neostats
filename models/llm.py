"""
models/llm.py
Handles instantiation and response generation for OpenAI, Groq, and Gemini.
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# ─── OpenAI ───────────────────────────────────────────────────────────────────
def get_openai_response(
    messages: List[Dict],
    api_key: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 1000,
    temperature: float = 0.7,
) -> str:
    """Generate a response using OpenAI's chat completion API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai")
        return "❌ OpenAI package not installed."
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return f"❌ OpenAI error: {str(e)}"


# ─── Groq ─────────────────────────────────────────────────────────────────────
def get_groq_response(
    messages: List[Dict],
    api_key: str,
    model: str = "llama3-70b-8192",
    max_tokens: int = 1000,
    temperature: float = 0.7,
) -> str:
    """Generate a response using Groq's chat completion API."""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except ImportError:
        logger.error("groq package not installed. Run: pip install groq")
        return "❌ Groq package not installed."
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return f"❌ Groq error: {str(e)}"


# ─── Google Gemini ────────────────────────────────────────────────────────────
def get_gemini_response(
    messages: List[Dict],
    api_key: str,
    model: str = "gemini-1.5-flash",
    max_tokens: int = 1000,
    temperature: float = 0.7,
) -> str:
    """Generate a response using Google Gemini API."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        # Convert OpenAI-style messages to Gemini format
        history = []
        system_content = ""
        user_message = ""

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_content = content
            elif role == "user":
                user_message = content
                history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                history.append({"role": "model", "parts": [content]})

        gem_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_content if system_content else None,
            generation_config=genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )

        # Use last user message; history excludes the last user turn
        chat_history = history[:-1] if history and history[-1]["role"] == "user" else history
        chat = gem_model.start_chat(history=chat_history)
        response = chat.send_message(user_message)
        return response.text

    except ImportError:
        logger.error("google-generativeai package not installed.")
        return "❌ google-generativeai package not installed."
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return f"❌ Gemini error: {str(e)}"


# ─── Unified entry point ──────────────────────────────────────────────────────
def generate_response(
    messages: List[Dict],
    provider: str,
    api_key: str,
    model: str,
    max_tokens: int = 1000,
    temperature: float = 0.7,
) -> str:
    """
    Unified function to call the correct LLM provider.

    Args:
        messages:    List of {"role": ..., "content": ...} dicts
        provider:    "openai" | "groq" | "gemini"
        api_key:     The provider's API key
        model:       Model name string
        max_tokens:  Maximum tokens in the response
        temperature: Sampling temperature

    Returns:
        The assistant's reply as a string.
    """
    provider = provider.lower().strip()

    if not api_key:
        return f"❌ No API key provided for **{provider}**. Please add it in the sidebar."

    dispatch = {
        "openai": get_openai_response,
        "groq":   get_groq_response,
        "gemini": get_gemini_response,
    }

    if provider not in dispatch:
        return f"❌ Unknown provider '{provider}'. Choose from: openai, groq, gemini."

    return dispatch[provider](
        messages=messages,
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
