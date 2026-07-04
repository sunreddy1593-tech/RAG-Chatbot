"""View and verify the embeddings stored in the ChromaDB index.

Reads the persisted index directly (no embedding model is loaded, so it's fast)
and runs a full health check over every vector, then lets you preview one.

Usage:
    python view_embeddings.py                 # verify all + preview first 3
    python view_embeddings.py --index 42      # preview the 43rd chunk's vector
    python view_embeddings.py --id <chunk_id> # preview a specific chunk by id
    python view_embeddings.py --full          # show the whole vector for previews
    python view_embeddings.py --export emb.npz # save ids + vectors to a .npz file
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import chromadb
from chromadb.config import Settings

import config


def _open_collection():
    client = chromadb.PersistentClient(
        path=str(config.INDEX_DIR), settings=Settings(anonymized_telemetry=False)
    )
    return client.get_collection(config.COLLECTION_NAME)


def _expected_chunk_count() -> int | None:
    path = config.PROCESSED_DIR / "chunks.jsonl"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _load_all(collection):
    data = collection.get(include=["embeddings", "metadatas", "documents"])
    ids = data["ids"]
    embeddings = np.asarray(data["embeddings"], dtype=np.float64)
    return ids, embeddings, data["metadatas"], data["documents"]


def verify(ids, embeddings, metadatas) -> bool:
    """Run health checks over every embedding; return True if all pass."""
    n, dim = embeddings.shape if embeddings.ndim == 2 else (len(embeddings), 0)
    norms = np.linalg.norm(embeddings, axis=1)

    expected = _expected_chunk_count()
    stored_dim = (_open_collection().metadata or {}).get("embedding_dim")

    checks: list[tuple[str, bool, str]] = []

    checks.append((
        "vector count == chunk count",
        expected is None or n == expected,
        f"{n} vectors" + (f" vs {expected} chunks" if expected else " (chunks.jsonl absent)"),
    ))
    checks.append((
        "all vectors have equal dimension",
        embeddings.ndim == 2,
        f"dim = {dim}",
    ))
    checks.append((
        f"dimension matches collection metadata ({stored_dim})",
        stored_dim is None or dim == int(stored_dim),
        f"{dim} == {stored_dim}",
    ))
    checks.append((
        "no NaN / Inf values",
        bool(np.isfinite(embeddings).all()),
        f"{int((~np.isfinite(embeddings)).sum())} bad values",
    ))
    zero_vecs = int((norms == 0).sum())
    checks.append((
        "no all-zero vectors",
        zero_vecs == 0,
        f"{zero_vecs} zero vectors",
    ))
    # BGE embeddings are stored L2-normalised -> norms should be ~1.0.
    norm_ok = bool(np.allclose(norms, 1.0, atol=1e-3))
    checks.append((
        "vectors are L2-normalised (norm ~ 1.0)",
        norm_ok,
        f"norm min/mean/max = {norms.min():.5f} / {norms.mean():.5f} / {norms.max():.5f}",
    ))
    checks.append((
        "ids are unique",
        len(ids) == len(set(ids)),
        f"{len(ids) - len(set(ids))} duplicates",
    ))
    meta_ok = all(
        all(str(m.get(f, "")).strip() for f in
            ("scheme", "doc_type", "source_url", "last_updated", "category"))
        for m in metadatas
    )
    checks.append((
        "every vector carries the 5 required metadata fields",
        meta_ok,
        "all present" if meta_ok else "some missing",
    ))

    print("=" * 68)
    print("EMBEDDING VERIFICATION")
    print("=" * 68)
    all_ok = True
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<48} {detail}")
        all_ok = all_ok and ok
    print("-" * 68)
    print(f"  RESULT: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    print()
    return all_ok


def preview(ids, embeddings, metadatas, documents, *, index=None, chunk_id=None,
            full=False, default_n=3) -> None:
    if chunk_id is not None:
        if chunk_id not in ids:
            print(f"id not found: {chunk_id}")
            return
        picks = [ids.index(chunk_id)]
    elif index is not None:
        picks = [index]
    else:
        picks = list(range(min(default_n, len(ids))))

    for i in picks:
        vec = embeddings[i]
        meta = metadatas[i]
        shown = vec if full else vec[:12]
        tail = "" if full else f" ... (+{len(vec) - len(shown)} more)"
        preview_vals = ", ".join(f"{x:+.4f}" for x in shown)
        print(f"[{i}] id: {ids[i]}")
        print(f"    scheme={meta.get('scheme')} | doc_type={meta.get('doc_type')}")
        print(f"    dim={len(vec)} | L2 norm={np.linalg.norm(vec):.5f}")
        print(f"    text: {' '.join(documents[i].split())[:100]}")
        print(f"    embedding: [{preview_vals}{tail}]")
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description="View / verify stored embeddings")
    ap.add_argument("--index", type=int, help="preview the chunk at this position")
    ap.add_argument("--id", dest="chunk_id", help="preview a specific chunk id")
    ap.add_argument("--full", action="store_true", help="print the full vector")
    ap.add_argument("--export", help="save ids + vectors to this .npz path")
    args = ap.parse_args()

    if not config.INDEX_DIR.exists():
        print(f"No index at {config.INDEX_DIR}. Build it: python -m vectorstore.indexer")
        return 1

    collection = _open_collection()
    ids, embeddings, metadatas, documents = _load_all(collection)

    all_ok = verify(ids, embeddings, metadatas)
    preview(ids, embeddings, metadatas, documents,
            index=args.index, chunk_id=args.chunk_id, full=args.full)

    if args.export:
        np.savez_compressed(args.export, ids=np.array(ids), embeddings=embeddings)
        print(f"Exported {len(ids)} embeddings -> {args.export}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
