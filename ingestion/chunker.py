"""Text splitter, metadata tagger, and Phase 2 corpus processor.

Two responsibilities:

1. ``chunk(text, metadata)`` — split one document's clean text into overlapping
   chunks and tag each with the 5 mandatory metadata fields.
2. ``run()`` — the Phase 2 driver: read ``data/metadata.json``, parse every
   successfully-scraped raw file (via :mod:`ingestion.parser`), chunk it, and
   write the tagged chunks to ``data/processed/`` as JSONL plus a manifest.

Chunking uses LangChain's ``RecursiveCharacterTextSplitter`` with a
**token-accurate length function**: chunk size (~500 tokens) and overlap
(~50 tokens) are measured with the *same* tokenizer as the embedding model
(``config.EMBEDDING_MODEL``), so no chunk ever exceeds BGE's 512-token input
window (which would otherwise silently truncate the tail of ~1 in 6 chunks at
embed time). If the tokenizer cannot be loaded (e.g. fully offline first run),
we fall back to a conservative character approximation.

Design notes (see edge-case.md §2):
- Every chunk must carry all 5 metadata fields; a chunk with a missing/empty
  field is rejected by the validation gate (edge 2.6).
- Empty / garbage chunks (too short, or almost no letters/words) are dropped so
  no boilerplate-only fragments reach the index (edge 2.4 / 2.5).
- Overlap preserves facts that straddle a chunk boundary (edge 2.3).
- Chunks are sized to the embedding model's 512-token window so no content is
  lost to truncation during indexing (edge 3.2-adjacent).
- Identical chunk texts are de-duplicated by content hash (edge 1.7).

Run with:  python -m ingestion.chunker
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from ingestion import parser

# Conservative char/token ratio used ONLY when the embedding tokenizer cannot
# be loaded (offline first run). Finance text with dense numeric tables can run
# as low as ~1.3 chars/token, so we stay well under 4 to avoid overflowing the
# 512-token window when approximating.
_FALLBACK_CHARS_PER_TOKEN = 3

# A chunk shorter than this (after stripping) is considered garbage/boilerplate.
_MIN_CHUNK_CHARS = 80
# Require at least this fraction of letters, else it's a pure separator / symbol
# blob. Kept low on purpose: number-heavy factsheet tables still carry facts.
_MIN_ALPHA_RATIO = 0.20
# Real words (3+ letters). Below this a chunk is a stray table fragment.
_MIN_REAL_WORDS = 5

REQUIRED_METADATA_FIELDS = ("scheme", "doc_type", "source_url", "last_updated", "category")

_SLUG_RE = re.compile(r"[^\w]+")
_WORD_RE = re.compile(r"[A-Za-z]{3,}")

# Lazily-loaded, process-wide cache for the embedding tokenizer.
_TOKENIZER = None
_TOKENIZER_TRIED = False


def _slugify(value: str) -> str:
    return _SLUG_RE.sub("-", value.strip().lower()).strip("-") or "doc"


def _get_tokenizer():
    """Return the embedding model's tokenizer, or None if it can't be loaded.

    Cached across calls. Loads only the tokenizer (not the model weights), so it
    is cheap; used to size chunks in true tokens against the 512-token window.
    """
    global _TOKENIZER, _TOKENIZER_TRIED
    if _TOKENIZER_TRIED:
        return _TOKENIZER
    _TOKENIZER_TRIED = True
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
        # Silence the "sequence longer than 512" warning: the splitter probes
        # oversized candidate strings on purpose while it searches for a cut.
        tok.model_max_length = 1_000_000_000
        _TOKENIZER = tok
    except Exception as exc:  # noqa: BLE001 - offline / missing weights fallback
        print(f"    (tokenizer unavailable, using char approximation: {exc})")
        _TOKENIZER = None
    return _TOKENIZER


def _build_splitter() -> RecursiveCharacterTextSplitter:
    """Token-accurate splitter (BGE tokenizer); char-approx fallback if offline."""
    tok = _get_tokenizer()
    separators = ["\n\n", "\n", ". ", " ", ""]

    if tok is not None:
        return RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=separators,
            length_function=lambda t: len(tok.encode(t, add_special_tokens=False)),
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE * _FALLBACK_CHARS_PER_TOKEN,
        chunk_overlap=config.CHUNK_OVERLAP * _FALLBACK_CHARS_PER_TOKEN,
        separators=separators,
        length_function=len,
    )


def _is_garbage(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < _MIN_CHUNK_CHARS:
        return True
    letters = sum(c.isalpha() for c in stripped)
    if (letters / len(stripped)) < _MIN_ALPHA_RATIO:
        return True
    return len(_WORD_RE.findall(stripped)) < _MIN_REAL_WORDS


def _validate_metadata(metadata: dict) -> None:
    """Raise ValueError if any of the 5 mandatory fields is missing/empty."""
    missing = [
        f for f in REQUIRED_METADATA_FIELDS
        if not str(metadata.get(f, "")).strip()
    ]
    if missing:
        raise ValueError(f"chunk metadata missing required field(s): {missing}")


def chunk(text: str, metadata: dict) -> list[dict]:
    """Split ``text`` into overlapping chunks, each tagged with ``metadata``.

    ``metadata`` must contain the 5 mandatory fields (scheme, doc_type,
    source_url, last_updated, category). Empty / garbage fragments are dropped.
    Returns a list of ``{"id", "text", "metadata"}`` records.
    """
    _validate_metadata(metadata)

    if not text or not text.strip():
        return []

    base_meta = {f: str(metadata[f]).strip() for f in REQUIRED_METADATA_FIELDS}
    source_domain = str(metadata.get("source_domain", "")).strip()

    # A short hash of the (unique) source URL disambiguates documents that share
    # the same scheme + doc_type + domain (e.g. several SEBI guidance pages),
    # guaranteeing globally-unique chunk ids for the vector store.
    url_tag = hashlib.sha1(base_meta["source_url"].encode("utf-8")).hexdigest()[:8]
    id_stem = f"{_slugify(base_meta['scheme'])}__{_slugify(base_meta['doc_type'])}__{url_tag}"

    splitter = _build_splitter()

    records: list[dict] = []
    idx = 0
    for piece in splitter.split_text(text):
        piece = piece.strip()
        if _is_garbage(piece):
            continue
        chunk_meta = dict(base_meta)
        if source_domain:
            chunk_meta["source_domain"] = source_domain
        chunk_meta["chunk_index"] = idx
        records.append(
            {
                "id": f"{id_stem}__{idx:04d}",
                "text": piece,
                "metadata": chunk_meta,
            }
        )
        idx += 1

    return records


# --------------------------------------------------------------------------- #
# Phase 2 orchestration
# --------------------------------------------------------------------------- #
def _resolve_raw_path(rel_or_abs: str) -> Path:
    """Resolve a metadata ``file`` entry (relative to BASE_DIR) to a real path."""
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = config.BASE_DIR / p
    return p


def _chunk_metadata_from_record(record: dict) -> dict:
    """Map a Phase 1 metadata record onto the 5 required chunk fields."""
    return {
        "scheme": record.get("scheme", ""),
        "doc_type": record.get("doc_type", ""),
        "source_url": record.get("url", ""),
        "last_updated": record.get("last_updated") or record.get("scraped_at", ""),
        "category": record.get("category", ""),
        "source_domain": record.get("source_domain", ""),
    }


def run() -> dict:
    """Parse + chunk every OK document in metadata.json into data/processed/."""
    records = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))
    ok_records = [r for r in records if r.get("status") == "ok" and r.get("file")]

    print(f"Loaded {len(records)} metadata records "
          f"({len(ok_records)} with status=ok)\n")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []
    seen_hashes: set[str] = set()
    manifest: list[dict] = []
    skipped: list[dict] = []
    duplicates = 0

    for i, record in enumerate(ok_records, 1):
        path = _resolve_raw_path(record["file"])
        label = f"{record.get('scheme', '?')} [{record.get('doc_type', '?')}]"
        print(f"[{i}/{len(ok_records)}] {label}\n    {path.name}")

        if not path.exists():
            print("    SKIP: raw file not found on disk\n")
            skipped.append({"file": str(path), "reason": "missing file"})
            continue

        try:
            text = parser.extract_text(str(path))
        except Exception as exc:  # noqa: BLE001 - one bad doc shouldn't abort
            print(f"    SKIP: parse error: {exc}\n")
            skipped.append({"file": str(path), "reason": f"parse error: {exc}"})
            continue

        if not text.strip():
            print("    SKIP: no extractable text (scanned/empty)\n")
            skipped.append({"file": str(path), "reason": "empty after parse"})
            continue

        try:
            doc_chunks = chunk(text, _chunk_metadata_from_record(record))
        except ValueError as exc:
            print(f"    SKIP: {exc}\n")
            skipped.append({"file": str(path), "reason": str(exc)})
            continue

        # De-duplicate identical chunk texts across the corpus (edge 1.7).
        unique_chunks: list[dict] = []
        for c in doc_chunks:
            h = hashlib.sha256(c["text"].encode("utf-8")).hexdigest()
            if h in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(h)
            unique_chunks.append(c)

        all_chunks.extend(unique_chunks)
        manifest.append(
            {
                "file": str(path.relative_to(config.BASE_DIR)),
                "scheme": record.get("scheme"),
                "doc_type": record.get("doc_type"),
                "source_url": record.get("url"),
                "text_chars": len(text),
                "chunks": len(unique_chunks),
            }
        )
        print(f"    OK: {len(unique_chunks)} chunks "
              f"({len(text):,} chars)\n")

    # Write outputs: one JSONL of chunks + a manifest/report.
    chunks_path = config.PROCESSED_DIR / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    manifest_path = config.PROCESSED_DIR / "processed_manifest.json"
    tokenization = (
        f"token-accurate ({config.EMBEDDING_MODEL})"
        if _get_tokenizer() is not None
        else f"char-approx ({_FALLBACK_CHARS_PER_TOKEN} chars/token)"
    )
    report = {
        "documents_processed": len(manifest),
        "documents_skipped": len(skipped),
        "total_chunks": len(all_chunks),
        "duplicate_chunks_dropped": duplicates,
        "chunk_size_tokens": config.CHUNK_SIZE,
        "chunk_overlap_tokens": config.CHUNK_OVERLAP,
        "tokenization": tokenization,
        "required_metadata_fields": list(REQUIRED_METADATA_FIELDS),
        "per_document": manifest,
        "skipped": skipped,
    }
    manifest_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _print_report(report, chunks_path, manifest_path)
    return report


def _print_report(report: dict, chunks_path: Path, manifest_path: Path) -> None:
    print("=" * 60)
    print("DOCUMENT PROCESSING REPORT")
    print("=" * 60)
    print(f"Documents processed : {report['documents_processed']}")
    print(f"Documents skipped   : {report['documents_skipped']}")
    print(f"Total chunks        : {report['total_chunks']}")
    print(f"Duplicate chunks    : {report['duplicate_chunks_dropped']} (dropped)")
    print(f"Chunk size / overlap: {report['chunk_size_tokens']} / "
          f"{report['chunk_overlap_tokens']} tokens")
    print(f"Tokenization        : {report['tokenization']}")
    print(f"Chunks written      : {chunks_path.relative_to(config.BASE_DIR)}")
    print(f"Manifest written    : {manifest_path.relative_to(config.BASE_DIR)}")
    for s in report["skipped"]:
        print(f"  - SKIPPED {s['file']}: {s['reason']}")


if __name__ == "__main__":
    result = run()
    # Fail the build if nothing usable was produced.
    sys.exit(0 if result["total_chunks"] > 0 else 1)
