"""Central configuration for the Mutual Fund FAQ Assistant.

All tunable parameters (model names, chunk sizes, retrieval settings, and
filesystem paths) live here so the rest of the codebase imports from a single
source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_FACTSHEETS_DIR = RAW_DIR / "factsheets"
RAW_KIM_DIR = RAW_DIR / "kim"
RAW_SID_DIR = RAW_DIR / "sid"
RAW_FAQ_DIR = RAW_DIR / "faq"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_FILE = DATA_DIR / "metadata.json"
SOURCE_URLS_FILE = DATA_DIR / "source_urls.json"

VECTORSTORE_DIR = BASE_DIR / "vectorstore"
INDEX_DIR = VECTORSTORE_DIR / "index"
# Timestamped snapshots of the previous good index, kept for rollback (Phase 10).
INDEX_BACKUPS_DIR = VECTORSTORE_DIR / "backups"

# --------------------------------------------------------------------------- #
# LLM (Groq)
# --------------------------------------------------------------------------- #
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# Lighter/cheaper model used as the degrade target when the primary model's
# daily quota is exhausted or on sustained 429s.
GROQ_MODEL_FALLBACK = os.getenv("GROQ_MODEL_FALLBACK", "llama3-8b-8192")
LLM_TEMPERATURE = 0.0  # deterministic, facts-only
# A grounded, <=3-sentence answer needs little completion room; keeping this
# small protects the per-minute (TPM) and per-day (TPD) token budgets below.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "256"))

# --------------------------------------------------------------------------- #
# Groq free-tier rate limits (llama-3.3-70b-versatile)
# --------------------------------------------------------------------------- #
# Four independent buckets; the first to run dry throttles the request. These
# are surfaced here so they can be raised for a paid tier without code changes.
# Client-side throttling + 429 backoff in rag/pipeline.py enforce them.
GROQ_RPM = int(os.getenv("GROQ_RPM", "30"))        # requests per minute
GROQ_RPD = int(os.getenv("GROQ_RPD", "1000"))      # requests per day
GROQ_TPM = int(os.getenv("GROQ_TPM", "12000"))     # tokens per minute
GROQ_TPD = int(os.getenv("GROQ_TPD", "100000"))    # tokens per day

# Per-call token budget guards (pre-flight gate before hitting Groq).
# Cap the assembled retrieval context and the number of chunks passed to the
# LLM so prompt + max_tokens stays well under TPM on every call. TPD is the
# binding daily constraint, so keeping calls lean stretches the answer budget.
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "2000"))
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "3"))

# 429 retry policy (exponential backoff + jitter; honour Retry-After header).
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "3"))
GROQ_BACKOFF_BASE_SECONDS = float(os.getenv("GROQ_BACKOFF_BASE_SECONDS", "1.0"))

# Cache normalized-query -> formatted-answer to avoid spending quota on repeats.
RESPONSE_CACHE_ENABLED = os.getenv("RESPONSE_CACHE_ENABLED", "true").lower() == "true"
RESPONSE_CACHE_SIZE = int(os.getenv("RESPONSE_CACHE_SIZE", "256"))

# --------------------------------------------------------------------------- #
# Embeddings (BGE via sentence-transformers)
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
EMBEDDING_MODEL_FALLBACK = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 1024  # bge-large-en-v1.5 -> 1024 dims (bge-small -> 384)
EMBEDDING_MAX_TOKENS = 512  # BGE input window; chunks are sized to fit this
EMBED_BATCH_SIZE = 32  # batched encoding to bound memory (edge 3.6)

# BGE is an asymmetric retrieval model: this instruction is prepended to the
# QUERY only (not corpus passages) at search time to improve recall.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
CHUNK_SIZE = 500  # tokens (approx)
CHUNK_OVERLAP = 50  # tokens

# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
TOP_K = 5
# Recalibrated from probing the live index: BGE cosine scores sit in a
# compressed high band (even off-topic queries score ~0.37), so 0.30 was far
# too low. In-scope facts score ~0.74-0.80; ~0.55 is a sane "not in corpus"
# floor used together with scheme/scope detection (a weak signal on its own).
SIMILARITY_THRESHOLD = 0.55
CANDIDATE_POOL = 20  # candidates fetched before MMR/dedup trims to TOP_K
MMR_LAMBDA = 0.6  # relevance-vs-diversity trade-off for MMR (1.0 = pure relevance)
MAX_QUERY_CHARS = 1200  # over-long queries are truncated before embedding
COLLECTION_NAME = "mf_faq_corpus"

# --------------------------------------------------------------------------- #
# Response constraints (compliance)
# --------------------------------------------------------------------------- #
MAX_SENTENCES = 3
DISCLAIMER = "Facts-only. No investment advice."
EDUCATIONAL_LINK = "https://www.amfiindia.com/investor-corner"

# --------------------------------------------------------------------------- #
# Source allowlist (no third-party / aggregator sites)
# --------------------------------------------------------------------------- #
ALLOWED_DOMAINS = (
    "hdfcfund.com",
    "amfiindia.com",
    "sebi.gov.in",
    "camsonline.com",
    "groww.in",
)


# --------------------------------------------------------------------------- #
# Scheduled data refresh (Phase 10)
# --------------------------------------------------------------------------- #
# The refresh (scrape -> parse -> chunk -> embed -> index) runs on a schedule.
# Cadence is owned by the GitHub Actions workflow cron (.github/workflows/
# refresh.yml, "0 5 * * *" = 05:00 UTC = 10:30 AM IST) so timing has a single
# source of truth; these knobs govern the run's *behaviour*, not its timing.
REFRESH_ENABLED = os.getenv("REFRESH_ENABLED", "true").lower() == "true"
# Minimum OK documents a scrape must yield before the index is rebuilt. A run
# that collects fewer aborts and leaves the previous good index served (edge 1.9).
MIN_CORPUS_DOCS = int(os.getenv("MIN_CORPUS_DOCS", "15"))
# How many timestamped previous-index snapshots to retain in INDEX_BACKUPS_DIR
# (0 disables snapshotting). Corpus is small (~few MB), so a couple is cheap.
INDEX_BACKUP_RETENTION = int(os.getenv("INDEX_BACKUP_RETENTION", "2"))
# Small JSON status file recording the last refresh (timestamp, status, counts)
# so the UI can show "Corpus last refreshed: <date>" (edge 9.6 / 10.6).
LAST_REFRESH_FILE = DATA_DIR / "last_refresh.json"
# Single-flight lock so a manual local run can't overlap another refresh; the
# GitHub Actions `concurrency` group is the CI-level equivalent.
REFRESH_LOCK_FILE = BASE_DIR / ".refresh.lock"
# A lock older than this (seconds) is treated as stale (crashed run) and broken.
REFRESH_LOCK_STALE_SECONDS = int(os.getenv("REFRESH_LOCK_STALE_SECONDS", "10800"))


def validate() -> list[str]:
    """Return a list of configuration problems (empty list == OK)."""
    problems: list[str] = []
    if not GROQ_API_KEY:
        problems.append("GROQ_API_KEY is not set (see .env.example)")
    return problems
