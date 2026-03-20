# AI bot ⚖️ — AI-Powered Legal Document Assistant

> **NeoStats AI Engineer Case Study Submission**

AI bot is an intelligent chatbot that helps users understand legal documents, contracts, and agreements in plain language. It uses RAG (Retrieval-Augmented Generation) to answer questions based on uploaded documents and live web search to retrieve current legal information.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 📄 **RAG Integration** | Upload PDF, DOCX, or TXT legal documents; ask questions grounded in the content |
| 🌐 **Live Web Search** | Real-time web search using Tavily, SerpAPI, or DuckDuckGo fallback |
| ⚡ **Response Modes** | Switch between **Concise** (2–4 sentences) and **Detailed** (full explanation) |
| 🤖 **Multi-LLM Support** | OpenAI, Groq (Llama 3), and Google Gemini |
| 💬 **Chat Memory** | Maintains conversation history across turns |
| 🔍 **Source Attribution** | Indicates when RAG or web search was used |

---

## 🗂️ Project Structure

```
project/
├── config/
│   ├── __init__.py
│   └── config.py           ← All API keys, settings (loaded from env vars)
│
├── models/
│   ├── __init__.py
│   ├── llm.py              ← OpenAI / Groq / Gemini response generation
│   └── embeddings.py       ← SentenceTransformer embedding models (RAG)
│
├── utils/
│   ├── __init__.py
│   ├── rag_utils.py        ← Document ingestion, chunking, vector store, search
│   ├── web_search.py       ← Tavily / SerpAPI / DuckDuckGo web search
│   └── chat_utils.py       ← Message building, history management, prompts
│
├── data/                   ← Vector store cache (auto-created)
├── .streamlit/
│   ├── config.toml         ← UI theme
│   └── secrets.toml        ← API keys for Streamlit Cloud (do not commit)
├── app.py                  ← Main Streamlit app
├── requirements.txt
├── .env.example            ← Environment variable template
└── README.md
```

---

## ⚙️ Setup

### 1. Clone & Install

```bash
git clone https://github.com/your-username/ai-bot.git
cd ai-bot
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

You need **at least one LLM provider key**:
- **Groq** (free tier available): https://console.groq.com
- **OpenAI**: https://platform.openai.com
- **Gemini**: https://aistudio.google.com

For web search (optional — DuckDuckGo fallback available for free):
- **Tavily**: https://tavily.com
- **SerpAPI**: https://serpapi.com

### 3. Run Locally

```bash
streamlit run app.py
```

---

## ☁️ Streamlit Cloud Deployment

1. Push your code to GitHub (do **not** commit `.env` or `secrets.toml`)
2. Go to https://streamlit.io/cloud → New App
3. Select your repo and set `app.py` as the entry point
4. In **Settings → Secrets**, paste the contents of `.streamlit/secrets.toml` with your real keys
5. Click Deploy!

---

## 💡 Use Case: Legal Document Assistant

**Problem:** Legal documents are dense, jargon-heavy, and difficult for non-lawyers to understand. People often sign contracts without fully understanding their obligations, rights, or risks.

**Solution:** AI bot allows users to:
- Upload contracts, NDAs, lease agreements, employment contracts, etc.
- Ask questions in plain English: *"What happens if I terminate early?"*
- Get instant, grounded answers based on the actual document text (RAG)
- Search for the latest legal news and regulatory updates (web search)
- Choose response depth based on their needs (Concise vs Detailed)

---

## 📋 Tech Stack

- **Framework:** Streamlit
- **LLMs:** OpenAI GPT-4o / Groq Llama 3 / Google Gemini
- **Embeddings:** `sentence-transformers` (all-MiniLM-L6-v2) — runs locally, no API key
- **Vector Store:** Custom numpy-based cosine similarity search
- **PDF Parsing:** pdfplumber
- **Web Search:** Tavily → SerpAPI → DuckDuckGo (cascade)

---

*Built for the NeoStats AI Engineer Case Study Challenge.*
"# Neostats" 
