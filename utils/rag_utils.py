"""
utils/rag_utils.py
Handles document ingestion, chunking, vector store creation, and similarity search
for Retrieval-Augmented Generation (RAG).
"""

import os
import re
import json
import logging
import pickle
from typing import List, Dict, Tuple, Optional
import numpy as np

from models.embeddings import embed_texts, embed_query, cosine_similarity
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS, VECTOR_STORE_PATH

logger = logging.getLogger(__name__)


# ─── Text Extraction ──────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file (bytes)."""
    try:
        import pdfplumber
        import io
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n".join(text_parts)
    except ImportError:
        logger.error("pdfplumber not installed. Run: pip install pdfplumber")
        return ""
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode plain-text file bytes."""
    try:
        return file_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        logger.error(f"TXT extraction error: {e}")
        return ""


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a .docx file."""
    try:
        import docx
        import io
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        logger.error("python-docx not installed. Run: pip install python-docx")
        return ""
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        return ""


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Route to correct extractor based on file extension."""
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in ("txt", "md"):
        return extract_text_from_txt(file_bytes)
    elif ext == "docx":
        return extract_text_from_docx(file_bytes)
    else:
        logger.warning(f"Unsupported file type: {ext}")
        return ""


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks of roughly chunk_size characters.
    Tries to split at sentence boundaries first.
    """
    try:
        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text:
            return []

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, current = [], ""

        for sentence in sentences:
            if len(current) + len(sentence) <= chunk_size:
                current += (" " if current else "") + sentence
            else:
                if current:
                    chunks.append(current.strip())
                # If sentence itself is larger than chunk_size, hard-split
                if len(sentence) > chunk_size:
                    for i in range(0, len(sentence), chunk_size - overlap):
                        chunks.append(sentence[i : i + chunk_size].strip())
                    current = sentence[-(overlap):] if overlap else ""
                else:
                    current = sentence

        if current.strip():
            chunks.append(current.strip())

        return [c for c in chunks if len(c) > 30]  # discard tiny shards
    except Exception as e:
        logger.error(f"Chunking error: {e}")
        return [text]


# ─── Vector Store ─────────────────────────────────────────────────────────────

class SimpleVectorStore:
    """
    A lightweight in-memory vector store backed by numpy arrays.
    Supports saving/loading to disk via pickle.
    """

    def __init__(self):
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict] = []

    def add_documents(self, chunks: List[str], source_name: str = "document") -> bool:
        """Embed chunks and add them to the store."""
        try:
            if not chunks:
                return False
            new_embeddings = embed_texts(chunks)
            if new_embeddings is None:
                logger.error("Embedding failed — vector store not updated.")
                return False

            self.chunks.extend(chunks)
            self.metadata.extend([{"source": source_name}] * len(chunks))

            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])

            logger.info(f"Added {len(chunks)} chunks from '{source_name}' to vector store.")
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False

    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Tuple[str, float, Dict]]:
        """Return top_k most similar chunks to the query."""
        try:
            if self.embeddings is None or len(self.chunks) == 0:
                return []

            q_vec = embed_query(query)
            if q_vec is None:
                return []

            scores = [cosine_similarity(q_vec, self.embeddings[i]) for i in range(len(self.chunks))]
            top_indices = np.argsort(scores)[::-1][:top_k]

            results = [
                (self.chunks[i], scores[i], self.metadata[i])
                for i in top_indices
                if scores[i] > 0.2   # minimum relevance threshold
            ]
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def save(self, path: str = VECTOR_STORE_PATH) -> bool:
        """Persist the vector store to disk."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".pkl", "wb") as f:
                pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings, "metadata": self.metadata}, f)
            return True
        except Exception as e:
            logger.error(f"Save error: {e}")
            return False

    def load(self, path: str = VECTOR_STORE_PATH) -> bool:
        """Load a previously saved vector store from disk."""
        try:
            fpath = path + ".pkl"
            if not os.path.exists(fpath):
                return False
            with open(fpath, "rb") as f:
                data = pickle.load(f)
            self.chunks     = data["chunks"]
            self.embeddings = data["embeddings"]
            self.metadata   = data["metadata"]
            return True
        except Exception as e:
            logger.error(f"Load error: {e}")
            return False

    def clear(self):
        """Reset the vector store."""
        self.chunks = []
        self.embeddings = None
        self.metadata = []

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0


# ─── Context Builder ──────────────────────────────────────────────────────────

def build_rag_context(results: List[Tuple[str, float, Dict]]) -> str:
    """
    Format retrieved chunks into a context string for the LLM prompt.
    """
    if not results:
        return ""

    parts = ["### Relevant Document Excerpts\n"]
    for i, (chunk, score, meta) in enumerate(results, 1):
        source = meta.get("source", "document")
        parts.append(f"**[Excerpt {i}]** *(source: {source}, relevance: {score:.2f})*\n{chunk}\n")

    return "\n".join(parts)


def ingest_file(file_bytes: bytes, filename: str, vector_store: SimpleVectorStore) -> Tuple[bool, int]:
    """
    Full pipeline: extract → chunk → embed → store.

    Returns:
        (success: bool, num_chunks: int)
    """
    try:
        text = extract_text(file_bytes, filename)
        if not text.strip():
            logger.warning(f"No text extracted from '{filename}'.")
            return False, 0

        chunks = chunk_text(text)
        if not chunks:
            return False, 0

        success = vector_store.add_documents(chunks, source_name=filename)
        return success, len(chunks)
    except Exception as e:
        logger.error(f"Ingest error for '{filename}': {e}")
        return False, 0
