"""Similarity search over the ChromaDB index (Phase 4).

Turns a user query into the most relevant corpus chunks. The strategy is driven
by empirical probing of the built index (see ImplementationPlan.md §Phase 4):

- **Scheme-scoped metadata filtering** — the 465-chunk shared factsheet otherwise
  dilutes scheme-specific queries (a "Small Cap exit load" query returned five
  wrong-scheme chunks). When a scheme is detected in the query, retrieval is
  filtered to it via Chroma ``where`` (edge 2.8 / 5.2).
- **Out-of-scope detection** — queries about other AMCs / non-scope schemes are
  flagged so the caller can return a scope message instead of a wrong answer
  (edge 5.5).
- **MMR + de-duplication** — near-identical shared-factsheet chunks and Groww
  navigation boilerplate are diversified away with Maximal Marginal Relevance
  (edge 5.4).
- **Recalibrated similarity floor** — BGE cosine scores are compressed into a
  high band, so ``config.SIMILARITY_THRESHOLD`` (~0.55) is a weak "not in corpus"
  signal used alongside scheme/scope detection, not on its own.
- **Input hygiene** — empty queries return nothing; over-long queries are
  truncated before embedding (edge 5.6 / 5.7).
"""

from __future__ import annotations

import re

import numpy as np

import config
from vectorstore.embedder import embed_query
from vectorstore.indexer import get_collection

# Canonical scheme names exactly as stored in the index metadata.
_LARGE = "HDFC Large Cap Fund - Direct Growth"
_MID = "HDFC Mid Cap Fund - Direct Growth"
_SMALL = "HDFC Small Cap Fund - Direct Growth"
_GOLD = "HDFC Gold ETF Fund of Fund - Direct Plan Growth"
_SILVER = "HDFC Silver ETF FoF - Direct Growth"

IN_SCOPE_SCHEMES = (_LARGE, _MID, _SMALL, _GOLD, _SILVER)

# Alias patterns -> canonical scheme. Order matters (checked top to bottom).
_SCHEME_ALIASES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(large[\s-]?cap|top\s?100)\b"), _LARGE),
    (re.compile(r"\b(mid[\s-]?cap|mid[\s-]?cap\s+opportunit)"), _MID),
    (re.compile(r"\b(small[\s-]?cap)\b"), _SMALL),
    (re.compile(r"\b(gold)\b"), _GOLD),
    (re.compile(r"\b(silver)\b"), _SILVER),
]

# Competing AMCs — presence (without an in-scope HDFC scheme) => out of scope.
_OTHER_AMCS = re.compile(
    r"\b(sbi|icici|axis|kotak|nippon|aditya\s*birla|uti|dsp|franklin|"
    r"mirae|tata|quant|parag\s*parikh|ppfas|motilal|edelweiss|bandhan|"
    r"canara|invesco|sundaram|baroda|lic\s*mf|groww\s+mutual)\b",
    re.IGNORECASE,
)


def detect_scheme(query: str) -> str | None:
    """Return the canonical in-scope scheme referenced by the query, or None."""
    q = query.lower()
    matches = {canon for pat, canon in _SCHEME_ALIASES if pat.search(q)}
    # Only filter when the reference is unambiguous (exactly one scheme).
    return next(iter(matches)) if len(matches) == 1 else None


def detect_out_of_scope(query: str) -> bool:
    """True if the query targets another AMC (and not HDFC).

    A competitor AMC name takes precedence over a category alias: e.g.
    "Nippon India Small Cap" must be flagged out-of-scope even though it contains
    "small cap", otherwise it would wrongly be answered with HDFC Small Cap data
    (edge 5.5). Only when the query also names HDFC do we treat it as in-scope
    and let scheme filtering / classification handle it.
    """
    if _OTHER_AMCS.search(query) and "hdfc" not in query.lower():
        return True
    return False


def _clean_query(query: str) -> str:
    q = (query or "").strip()
    if len(q) > config.MAX_QUERY_CHARS:  # edge 5.7
        q = q[: config.MAX_QUERY_CHARS]
    return q


def _mmr(
    query_vec: np.ndarray,
    cand_vecs: np.ndarray,
    cand_sims: np.ndarray,
    k: int,
    lambda_: float,
) -> list[int]:
    """Maximal Marginal Relevance selection; returns indices into candidates."""
    selected: list[int] = []
    remaining = list(range(len(cand_vecs)))
    while remaining and len(selected) < k:
        if not selected:
            best = int(max(remaining, key=lambda i: cand_sims[i]))
        else:
            sel_mat = cand_vecs[selected]  # (s, d), rows are unit vectors
            best, best_score = remaining[0], -1e9
            for i in remaining:
                redundancy = float(np.max(sel_mat @ cand_vecs[i]))
                score = lambda_ * cand_sims[i] - (1 - lambda_) * redundancy
                if score > best_score:
                    best, best_score = i, score
        selected.append(best)
        remaining.remove(best)
    return selected


def retrieve(query: str, top_k: int = config.TOP_K) -> list[dict]:
    """Return the top-K most relevant chunks (with metadata + score) for a query.

    Applies scheme filtering, MMR de-duplication, and the similarity floor.
    Returns ``[]`` for empty queries or when nothing clears the threshold
    (the caller treats an empty result as "not in corpus").
    """
    q = _clean_query(query)
    if not q:  # edge 5.6
        return []

    collection = get_collection()
    query_vec = np.asarray(embed_query(q), dtype=np.float64)

    scheme = detect_scheme(q)
    where = {"scheme": scheme} if scheme else None

    pool = max(config.CANDIDATE_POOL, top_k * 4)
    res = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=pool,
        where=where,
        include=["documents", "metadatas", "distances", "embeddings"],
    )
    # If a scheme filter returned nothing, retry unfiltered (widen).
    if where and not res["ids"][0]:
        res = collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=pool,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

    ids = res["ids"][0]
    if not ids:
        return []

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    sims = np.array([1.0 - d for d in res["distances"][0]])  # cosine similarity
    vecs = np.asarray(res["embeddings"][0], dtype=np.float64)

    # De-duplicate identical / near-identical chunk texts before ranking.
    seen: set[str] = set()
    keep: list[int] = []
    for i, doc in enumerate(docs):
        key = " ".join(doc.split()).lower()[:200]
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)

    ids = [ids[i] for i in keep]
    docs = [docs[i] for i in keep]
    metas = [metas[i] for i in keep]
    sims = sims[keep]
    vecs = vecs[keep]

    order = _mmr(query_vec, vecs, sims, top_k, config.MMR_LAMBDA)

    results: list[dict] = []
    for i in order:
        if sims[i] < config.SIMILARITY_THRESHOLD:
            continue
        results.append(
            {
                "id": ids[i],
                "text": docs[i],
                "score": round(float(sims[i]), 4),
                "metadata": metas[i],
            }
        )
    return results


def assemble_context(chunks: list[dict]) -> dict:
    """Concatenate retrieved chunks and surface their source metadata.

    Returns ``{"context": str, "sources": [...], "last_updated": str|None}`` for
    the LLM prompt and the response formatter (Phase 6) to cite one source and
    stamp the freshness footer.
    """
    blocks: list[str] = []
    sources: list[dict] = []
    for i, c in enumerate(chunks, 1):
        m = c.get("metadata", {})
        blocks.append(
            f"[Source {i}] ({m.get('scheme')} — {m.get('doc_type')})\n{c['text']}"
        )
        sources.append(
            {
                "rank": i,
                "score": c.get("score"),
                "source_url": m.get("source_url"),
                "scheme": m.get("scheme"),
                "doc_type": m.get("doc_type"),
                "last_updated": m.get("last_updated"),
            }
        )

    dates = [s["last_updated"] for s in sources if s.get("last_updated")]
    return {
        "context": "\n\n".join(blocks),
        "sources": sources,
        "last_updated": max(dates) if dates else None,
    }
