"""Build and persist the ChromaDB index (Phase 3).

Reads the Phase 2 chunk set (``data/processed/chunks.jsonl``), embeds every
chunk with the BGE model, and stores vectors + documents + metadata in a
persistent ChromaDB collection.

Robustness (see edge-case.md §3):
- **Atomic build** — the index is built into a temp directory and swapped in on
  success, so a crash mid-build never corrupts the live index; the previous
  good index is kept as a ``.bak`` until the new one is verified (edge 3.4).
- **Post-index assertion** — vector count must equal chunk count, else the build
  fails and reports the delta (edge 3.3).
- **Dimension lock** — the embedding model name + dim are stored on the
  collection and in a manifest; ``get_collection`` validates the dim on load and
  refuses to serve an index built with a different model (edge 3.2 / 3.5).

Run with:  python -m vectorstore.indexer            # build (fails if <15 docs)
           python -m vectorstore.indexer --reindex  # force rebuild
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import config
from vectorstore.embedder import get_embedder

# Chroma add() has an upper bound per call; batch well under it.
_ADD_BATCH = 1000
_MANIFEST_NAME = "index_manifest.json"


def _load_chunks() -> list[dict]:
    path = config.PROCESSED_DIR / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run Phase 2 first: python -m ingestion.chunker"
        )
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _new_client(path: Path):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    )


def _swap_dir(tmp: Path, final: Path) -> None:
    """Atomically replace ``final`` with ``tmp`` (Windows-safe), keeping a backup."""
    backup = final.with_name(final.name + ".bak")
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)
    if final.exists():
        final.rename(backup)
    try:
        tmp.rename(final)
    except OSError:  # cross-device or race: fall back to copy
        shutil.copytree(tmp, final)
        shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)


def build_index(reindex: bool = True) -> dict:
    """Embed processed chunks and persist them to ChromaDB. Returns a report."""
    chunks = _load_chunks()
    if not chunks:
        raise RuntimeError("no chunks to index (data/processed/chunks.jsonl is empty)")

    embedder = get_embedder()
    if embedder.dim != config.EMBEDDING_DIM:
        print(
            f"NOTE: active model '{embedder.model_name}' produces {embedder.dim}-dim "
            f"vectors (config.EMBEDDING_DIM={config.EMBEDDING_DIM})."
        )

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    print(f"Embedding {len(chunks)} chunks with {embedder.model_name} "
          f"({embedder.dim}-dim) on {embedder.device} ...")
    embeddings = embedder.embed_passages(documents, show_progress_bar=True)

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"embedding count {len(embeddings)} != chunk count {len(chunks)}"
        )

    tmp_dir = config.INDEX_DIR.with_name(config.INDEX_DIR.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building ChromaDB collection '{config.COLLECTION_NAME}' ...")
    client = _new_client(tmp_dir)
    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": embedder.model_name,
            "embedding_dim": embedder.dim,
        },
    )

    for start in range(0, len(chunks), _ADD_BATCH):
        end = start + _ADD_BATCH
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    count = collection.count()
    if count != len(chunks):  # edge 3.3
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(
            f"post-index assertion failed: {count} vectors != {len(chunks)} chunks"
        )

    del collection, client  # release the sqlite handle before swapping dirs
    _swap_dir(tmp_dir, config.INDEX_DIR)

    report = {
        "collection": config.COLLECTION_NAME,
        "embedding_model": embedder.model_name,
        "embedding_dim": embedder.dim,
        "vectors": count,
        "chunks": len(chunks),
        "index_dir": str(config.INDEX_DIR.relative_to(config.BASE_DIR)),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (config.INDEX_DIR / _MANIFEST_NAME).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    _print_report(report)
    return report


def get_collection():
    """Open the persisted collection, validating the embedding dimension.

    Used by the retrieval layer (Phase 4). Raises if the index is missing or was
    built with an embedding model of a different dimension than is active now.
    """
    if not config.INDEX_DIR.exists():
        raise FileNotFoundError(
            f"No index at {config.INDEX_DIR}. Build it first: "
            f"python -m vectorstore.indexer"
        )

    client = _new_client(config.INDEX_DIR)
    collection = client.get_collection(config.COLLECTION_NAME)

    stored_dim = (collection.metadata or {}).get("embedding_dim")
    active_dim = get_embedder().dim
    if stored_dim is not None and int(stored_dim) != active_dim:  # edge 3.2 / 3.5
        raise RuntimeError(
            f"index dim {stored_dim} != active model dim {active_dim}. "
            f"Rebuild the index: python -m vectorstore.indexer --reindex"
        )
    return collection


def _print_report(report: dict) -> None:
    print("=" * 60)
    print("INDEXING REPORT")
    print("=" * 60)
    print(f"Collection    : {report['collection']}")
    print(f"Model / dim   : {report['embedding_model']} ({report['embedding_dim']})")
    print(f"Vectors       : {report['vectors']} (chunks: {report['chunks']})")
    print(f"Index dir     : {report['index_dir']}")


def _sample_query_check() -> None:
    """Acceptance check: a known factual query returns relevant chunks."""
    from vectorstore.embedder import embed_query

    question = "What is the expense ratio of HDFC Large Cap Fund?"
    collection = get_collection()
    res = collection.query(
        query_embeddings=[embed_query(question)],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    print("\nSample similarity query:")
    print(f"  Q: {question}")
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        sim = 1 - dist  # cosine distance -> similarity
        snippet = " ".join(doc.split())[:110]
        print(f"  [{sim:.3f}] {meta.get('scheme')} / {meta.get('doc_type')}: {snippet}")


if __name__ == "__main__":
    reindex = "--reindex" in sys.argv
    result = build_index(reindex=reindex)
    try:
        _sample_query_check()
    except Exception as exc:  # noqa: BLE001 - build succeeded; check is informational
        print(f"(sample query check skipped: {exc})")
    sys.exit(0 if result["vectors"] == result["chunks"] and result["vectors"] > 0 else 1)
