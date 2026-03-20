"""
models/embeddings.py
Manages embedding models used for RAG (Retrieval-Augmented Generation).
Uses sentence-transformers locally — no API key required.
"""

import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Cache the model so it's only loaded once per session
_embedding_model = None


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load and cache a sentence-transformer embedding model.

    Args:
        model_name: HuggingFace model name (default: all-MiniLM-L6-v2)

    Returns:
        SentenceTransformer model instance or None on failure
    """
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully.")
        return _embedding_model
    except ImportError:
        logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        return None
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        return None


def embed_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> Optional[np.ndarray]:
    """
    Embed a list of text strings into dense vectors.

    Args:
        texts:       List of text strings to embed
        model_name:  SentenceTransformer model to use

    Returns:
        numpy array of shape (len(texts), embedding_dim) or None on failure
    """
    try:
        model = load_embedding_model(model_name)
        if model is None:
            return None
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings
    except Exception as e:
        logger.error(f"Error embedding texts: {e}")
        return None


def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> Optional[np.ndarray]:
    """
    Embed a single query string.

    Args:
        query:      The query string
        model_name: SentenceTransformer model to use

    Returns:
        1-D numpy array (embedding vector) or None on failure
    """
    try:
        result = embed_texts([query], model_name)
        if result is not None:
            return result[0]
        return None
    except Exception as e:
        logger.error(f"Error embedding query: {e}")
        return None


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    try:
        dot   = np.dot(vec_a, vec_b)
        norm  = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        return float(dot / norm) if norm > 0 else 0.0
    except Exception as e:
        logger.error(f"Cosine similarity error: {e}")
        return 0.0
