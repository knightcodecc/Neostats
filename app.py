"""
app.py — AI bot ⚖️: AI-Powered Legal Document Assistant
Main Streamlit UI entry point.

Features:
  • RAG: Upload and query legal documents (PDF, DOCX, TXT)
  • Live web search: Real-time legal news and regulations
  • Response modes: Concise vs Detailed
  • Multi-provider LLM support: OpenAI, Groq, Gemini
  • Chat history with source attribution
"""

import logging
import streamlit as st

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Local Imports ────────────────────────────────────────────────────────────
from config.config import (
    APP_TITLE, APP_SUBTITLE, APP_ICON,
    OPENAI_MODELS, GROQ_MODELS, GEMINI_MODELS,
    DEFAULT_OPENAI_MODEL, DEFAULT_GROQ_MODEL, DEFAULT_GEMINI_MODEL,
    RESPONSE_MODES,
)
from models.llm import generate_response
from models.embeddings import load_embedding_model
from utils.rag_utils import SimpleVectorStore, ingest_file, build_rag_context
from utils.web_search import web_search, format_search_results, should_web_search
from utils.chat_utils import build_messages, format_source_badges

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global font */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* Header bar */
.header-bar {
    background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 1.2rem 2rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.header-title { color: #e2b96f; font-size: 2rem; font-weight: 700; margin: 0; }
.header-sub   { color: #a8c6e8; font-size: 0.95rem; margin: 0; }

/* Chat bubbles */
.chat-user {
    background: #1e3a5f;
    color: #e8f4fd;
    border-radius: 18px 18px 4px 18px;
    padding: 0.8rem 1.1rem;
    margin: 0.4rem 0;
    max-width: 80%;
    margin-left: auto;
}
.chat-bot {
    background: #0d1f36;
    color: #d4e9f7;
    border-radius: 18px 18px 18px 4px;
    border-left: 3px solid #e2b96f;
    padding: 0.8rem 1.1rem;
    margin: 0.4rem 0;
    max-width: 85%;
}
.source-badge {
    font-size: 0.72rem;
    color: #7fb3d3;
    margin-top: 0.3rem;
    font-style: italic;
}

/* Sidebar tweaks */
.sidebar-section {
    background: #0d1f36;
    border-radius: 8px;
    padding: 0.8rem;
    margin-bottom: 0.8rem;
}
.sidebar-label { color: #e2b96f; font-weight: 600; font-size: 0.85rem; }

/* Disclaimer */
.disclaimer {
    background: #1a0a0a;
    border-left: 4px solid #c0392b;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-size: 0.8rem;
    color: #e8a0a0;
    margin-top: 0.5rem;
}

/* Upload area */
.upload-hint { color: #7fb3d3; font-size: 0.82rem; margin-top: 0.3rem; }

/* Mode badge */
.mode-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.mode-concise  { background: #0f3460; color: #7fb3d3; }
.mode-detailed { background: #1a3a1a; color: #7fba7f; }

/* Stats bar */
.stats-bar {
    background: #0d1f36;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.8rem;
    color: #7fb3d3;
    margin-bottom: 0.8rem;
    display: flex;
    gap: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ───────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "chat_history":    [],        # [{"role": ..., "content": ..., "badges": ...}]
        "llm_messages":    [],        # raw messages for the LLM
        "vector_store":    SimpleVectorStore(),
        "uploaded_files":  [],        # list of ingested filenames
        "provider":        "groq",
        "model":           DEFAULT_GROQ_MODEL,
        "response_mode":   "Detailed",
        "use_rag":         True,
        "use_web_search":  False,
        "auto_web_search": True,
        "temperature":     0.7,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="header-bar">
    <div>
        <p class="header-title">⚖️ AI bot</p>
        <p class="header-sub">AI-Powered Legal Document Assistant · Understand. Analyze. Clarify.</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
⚠️ <strong>Disclaimer:</strong> AI bot provides general legal information only. 
This is <strong>not legal advice</strong>. Always consult a qualified attorney for advice specific to your situation.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # ── Provider & Model ──
    st.markdown('<p class="sidebar-label">🤖 LLM Provider</p>', unsafe_allow_html=True)
    provider = st.selectbox(
        "Provider",
        ["groq", "openai", "gemini"],
        index=["groq", "openai", "gemini"].index(st.session_state.provider),
        label_visibility="collapsed",
    )
    st.session_state.provider = provider

    model_options = {"groq": GROQ_MODELS, "openai": OPENAI_MODELS, "gemini": GEMINI_MODELS}[provider]
    default_model = {"groq": DEFAULT_GROQ_MODEL, "openai": DEFAULT_OPENAI_MODEL, "gemini": DEFAULT_GEMINI_MODEL}[provider]
    current_model = st.session_state.model if st.session_state.model in model_options else default_model

    st.markdown('<p class="sidebar-label">🧠 Model</p>', unsafe_allow_html=True)
    model = st.selectbox("Model", model_options,
                         index=model_options.index(current_model),
                         label_visibility="collapsed")
    st.session_state.model = model

    # ── API Keys ──
    st.markdown("---")
    st.markdown('<p class="sidebar-label">🔑 API Keys</p>', unsafe_allow_html=True)

    provider_key_map = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
    llm_key = st.text_input(
        f"{provider.upper()} API Key",
        type="password",
        placeholder=f"Enter your {provider.upper()} API key",
        key=f"api_key_{provider}",
    )
    import os
    if not llm_key:
        llm_key = os.getenv(provider_key_map[provider], "")

    # ── Response Mode ──
    st.markdown("---")
    st.markdown('<p class="sidebar-label">📝 Response Mode</p>', unsafe_allow_html=True)
    response_mode = st.radio(
        "Response Mode",
        list(RESPONSE_MODES.keys()),
        index=list(RESPONSE_MODES.keys()).index(st.session_state.response_mode),
        label_visibility="collapsed",
        horizontal=True,
    )
    st.session_state.response_mode = response_mode
    mode_desc = RESPONSE_MODES[response_mode]["description"]
    st.caption(f"ℹ️ {mode_desc}")

    # ── RAG Settings ──
    st.markdown("---")
    st.markdown('<p class="sidebar-label">📄 Document RAG</p>', unsafe_allow_html=True)
    use_rag = st.toggle("Enable RAG", value=st.session_state.use_rag)
    st.session_state.use_rag = use_rag

    if use_rag:
        uploaded = st.file_uploader(
            "Upload legal documents",
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
            help="Upload contracts, agreements, legal documents, etc.",
        )

        if uploaded:
            vs = st.session_state.vector_store
            for uf in uploaded:
                if uf.name not in st.session_state.uploaded_files:
                    with st.spinner(f"Processing {uf.name}…"):
                        try:
                            success, n_chunks = ingest_file(uf.read(), uf.name, vs)
                            if success:
                                st.session_state.uploaded_files.append(uf.name)
                                st.success(f"✅ {uf.name} — {n_chunks} chunks indexed")
                            else:
                                st.error(f"❌ Failed to process {uf.name}")
                        except Exception as e:
                            st.error(f"Error: {e}")
                            logger.error(e)

        if st.session_state.uploaded_files:
            st.markdown("**Indexed files:**")
            for fname in st.session_state.uploaded_files:
                st.markdown(f"📎 `{fname}`")

            if st.button("🗑️ Clear all documents"):
                st.session_state.vector_store = SimpleVectorStore()
                st.session_state.uploaded_files = []
                st.rerun()

    # ── Web Search Settings ──
    st.markdown("---")
    st.markdown('<p class="sidebar-label">🌐 Live Web Search</p>', unsafe_allow_html=True)
    use_web = st.toggle("Enable Web Search", value=st.session_state.use_web_search)
    st.session_state.use_web_search = use_web

    if use_web:
        auto_web = st.checkbox("Auto-detect when to search", value=st.session_state.auto_web_search)
        st.session_state.auto_web_search = auto_web

        tavily_key  = st.text_input("Tavily API Key (optional)", type="password",
                                    placeholder="tvly-...", key="tavily_key")
        serpapi_key = st.text_input("SerpAPI Key (optional)",  type="password",
                                    placeholder="serpapi key", key="serpapi_key")

        if not tavily_key:  tavily_key  = os.getenv("TAVILY_API_KEY", "")
        if not serpapi_key: serpapi_key = os.getenv("SERPAPI_API_KEY", "")

        st.caption("💡 No keys? DuckDuckGo fallback will be used automatically.")
    else:
        tavily_key = serpapi_key = ""

    # ── Temperature ──
    st.markdown("---")
    temperature = st.slider("🌡️ Temperature", 0.0, 1.0,
                            value=st.session_state.temperature, step=0.05)
    st.session_state.temperature = temperature

    # ── Clear Chat ──
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.llm_messages = []
        st.rerun()

    # ── Stats ──
    st.markdown("---")
    vs = st.session_state.vector_store
    st.markdown(f"""
    **📊 Session Stats**  
    💬 Messages: `{len(st.session_state.chat_history)}`  
    📄 Chunks indexed: `{len(vs.chunks)}`  
    📁 Files: `{len(st.session_state.uploaded_files)}`
    """)


# ─── Main Chat Area ───────────────────────────────────────────────────────────

# Display mode badge
mode_class = "mode-concise" if response_mode == "Concise" else "mode-detailed"
st.markdown(
    f'<span class="mode-badge {mode_class}">{"⚡ Concise" if response_mode == "Concise" else "📖 Detailed"} Mode</span>',
    unsafe_allow_html=True,
)

# Chat container
chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; color: #5a7fa0;">
            <div style="font-size:3rem">⚖️</div>
            <h3 style="color:#e2b96f">Welcome to AI bot</h3>
            <p>Upload a legal document and start asking questions, or ask about any legal topic.</p>
            <br>
            <b style="color:#7fb3d3">Try asking:</b><br>
            "Summarize the key obligations in this contract"<br>
            "What are the termination clauses?"<br>
            "Explain what indemnification means"<br>
            "What are the latest GDPR updates?"
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant", avatar="⚖️"):
                st.markdown(msg["content"])
                badges = msg.get("badges", "")
                if badges:
                    st.markdown(f'<div class="source-badge">ℹ️ {badges}</div>', unsafe_allow_html=True)


# ─── Quick Prompt Suggestions ─────────────────────────────────────────────────
if not st.session_state.chat_history:
    st.markdown("**💡 Quick prompts:**")
    cols = st.columns(4)
    quick_prompts = [
        "📋 Summarize this contract",
        "⚠️ Identify risk clauses",
        "📰 Latest legal news",
        "🔍 Explain indemnification",
    ]
    for col, prompt in zip(cols, quick_prompts):
        with col:
            if st.button(prompt, use_container_width=True):
                st.session_state["prefill_prompt"] = prompt.split(" ", 1)[1]
                st.rerun()


# ─── Chat Input ───────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill_prompt", "")
user_input = st.chat_input(
    "Ask about your legal document or any legal question…",
)

if not user_input and prefill:
    user_input = prefill

if user_input:
    if not llm_key:
        st.error(f"⚠️ Please enter your {provider.upper()} API key in the sidebar to continue.")
        st.stop()

    # Show user message immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # ── Determine if RAG / web search should be triggered ──
    rag_context = ""
    web_context = ""
    rag_used    = False
    web_used    = False

    # RAG
    if use_rag and not st.session_state.vector_store.is_empty:
        try:
            results = st.session_state.vector_store.search(user_input)
            if results:
                rag_context = build_rag_context(results)
                rag_used = True
                logger.info(f"RAG: {len(results)} chunks retrieved.")
        except Exception as e:
            logger.error(f"RAG search error: {e}")

    # Web Search
    if use_web:
        run_search = True
        if st.session_state.auto_web_search:
            run_search = should_web_search(user_input)

        if run_search:
            with st.spinner("🌐 Searching the web…"):
                try:
                    results = web_search(
                        query=user_input,
                        tavily_key=tavily_key,
                        serpapi_key=serpapi_key,
                        max_results=5,
                    )
                    if results:
                        web_context = format_search_results(results)
                        web_used = True
                        logger.info(f"Web search: {len(results)} results.")
                except Exception as e:
                    logger.error(f"Web search error: {e}")

    # ── Build messages and call LLM ──
    max_tokens = RESPONSE_MODES[response_mode]["max_tokens"]
    messages   = build_messages(
        history=st.session_state.llm_messages,
        user_query=user_input,
        response_mode=response_mode,
        rag_context=rag_context,
        web_context=web_context,
    )

    with st.chat_message("assistant", avatar="⚖️"):
        with st.spinner("⚖️ AI bot is thinking…"):
            try:
                response = generate_response(
                    messages=messages,
                    provider=provider,
                    api_key=llm_key,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                response = f"❌ Error generating response: {str(e)}"
                logger.error(e)

        st.markdown(response)

        badges = format_source_badges(rag_used, web_used)
        if badges:
            st.markdown(f'<div class="source-badge">ℹ️ {badges}</div>', unsafe_allow_html=True)

    # ── Update session state ──
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response,
        "badges": format_source_badges(rag_used, web_used),
    })

    # Store raw messages for history (without injected context to keep it clean)
    st.session_state.llm_messages.append({"role": "user",      "content": user_input})
    st.session_state.llm_messages.append({"role": "assistant", "content": response})


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#3d5a73; font-size:0.78rem;">'
    'AI bot ⚖️ · Built on Streamlit · Powered by OpenAI / Groq / Gemini · '
    'RAG with SentenceTransformers · NeoStats AI Engineer Challenge'
    '</div>',
    unsafe_allow_html=True,
)
