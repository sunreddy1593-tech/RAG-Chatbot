"""Embedding model wrapper around the BGE retrieval models (local, no API key).

Phase 3 implementation. The active model is ``config.EMBEDDING_MODEL`` — by
default ``bge-small-en-v1.5`` (384-dim) so the app fits low-resource hosts like
the Streamlit Community Cloud free tier; ``bge-large-en-v1.5`` (1024-dim) can be
selected on a bigger host for higher recall (rebuild the index to match).

BGE is an *asymmetric* retrieval model, so corpus passages and search queries
are encoded differently:

- **Passages** (corpus chunks) are encoded with **no instruction prefix**.
- **Queries** are prefixed with ``config.BGE_QUERY_INSTRUCTION`` before encoding.

All embeddings are **L2-normalised** (unit vectors) so cosine similarity equals
the dot product, which keeps scores bounded and makes the
``config.SIMILARITY_THRESHOLD`` gate meaningful downstream.

The model is loaded locally via ``sentence-transformers`` (no API key, offline
after the first download). If the primary model cannot be loaded, we fall back
to ``config.EMBEDDING_MODEL_FALLBACK`` (bge-small, 384-dim) for low-resource /
offline environments (edge 3.1).
"""

from __future__ import annotations

import config

# Process-wide singleton so the (heavy) model loads at most once.
_EMBEDDER: "BGEEmbedder | None" = None


def _pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001 - torch always present via sentence-transformers
        pass
    return "cpu"


class BGEEmbedder:
    """Reusable wrapper that produces normalised BGE embeddings."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.device = _pick_device(device)
        requested = model_name or config.EMBEDDING_MODEL
        try:
            self.model = SentenceTransformer(requested, device=self.device)
            self.model_name = requested
        except Exception as exc:  # noqa: BLE001 - fall back to the small model
            if requested == config.EMBEDDING_MODEL_FALLBACK:
                raise
            print(
                f"WARNING: could not load '{requested}' ({exc}); "
                f"falling back to '{config.EMBEDDING_MODEL_FALLBACK}'"
            )
            self.model = SentenceTransformer(
                config.EMBEDDING_MODEL_FALLBACK, device=self.device
            )
            self.model_name = config.EMBEDDING_MODEL_FALLBACK

        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed_passages(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        """Encode corpus passages (no instruction prefix), normalised."""
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=batch_size or config.EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Encode a single search query with the BGE retrieval instruction."""
        vector = self.model.encode(
            config.BGE_QUERY_INSTRUCTION + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.tolist()


def get_embedder() -> BGEEmbedder:
    """Return the process-wide embedder singleton (loads the model on first use)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = BGEEmbedder()
    return _EMBEDDER


def embed(texts: list[str]) -> list[list[float]]:
    """Return normalised BGE embeddings for the given passages (backward-compat)."""
    return get_embedder().embed_passages(texts)


def embed_query(query: str) -> list[float]:
    """Return the normalised BGE embedding for a search query."""
    return get_embedder().embed_query(query)
