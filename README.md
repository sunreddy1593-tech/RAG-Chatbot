# Mutual Fund FAQ Assistant (Facts-Only RAG)

A lightweight **Retrieval-Augmented Generation (RAG)** assistant that answers
**factual, source-backed** questions about a curated set of **HDFC Mutual Fund**
schemes. It never gives investment advice — every answer is concise (≤3
sentences), cites exactly one official source, and shows a last-updated date.

> **Disclaimer:** Facts-only. No investment advice.

---

## Table of Contents

- [What it does](#what-it-does)
- [Selected AMC & Schemes](#selected-amc--schemes)
- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Quick Start (run the app)](#quick-start-run-the-app)
- [Building the Corpus & Index from Scratch](#building-the-corpus--index-from-scratch)
- [Re-ingestion / Refresh Procedure](#re-ingestion--refresh-procedure)
- [Testing & Evaluation](#testing--evaluation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Compliance & Safety](#compliance--safety)
- [Known Limitations](#known-limitations)
- [Deployment Guide](#deployment-guide)
- [Disclaimer](#disclaimer)

---

## What it does

- Answers **factual** questions (expense ratio, exit load, minimum SIP, benchmark,
  riskometer, fund manager, category, statement downloads, MF basics) grounded in
  official documents.
- **Refuses advisory** questions ("should I invest?", "which is better?", "will it
  give returns?") with a polite, compliant message + an AMFI/SEBI educational link.
- Returns a **scope message** for out-of-scope AMCs/schemes and a **grounded
  fallback** ("not available in the current corpus") when the answer isn't indexed.
- Every factual answer carries **exactly one allowlisted source URL** and a
  **`Last updated from sources: <date>`** footer.

---

## Selected AMC & Schemes

**AMC:** HDFC Mutual Fund

| # | Scheme | Category |
|---|--------|----------|
| 1 | HDFC Large Cap Fund – Direct Growth | Large Cap Equity |
| 2 | HDFC Mid Cap Fund – Direct Growth | Mid Cap Equity |
| 3 | HDFC Small Cap Fund – Direct Growth | Small Cap Equity |
| 4 | HDFC Gold ETF Fund of Fund – Direct Plan Growth | Commodity (Gold) |
| 5 | HDFC Silver ETF FoF – Direct Growth | Commodity (Silver) |

The corpus is drawn only from an **allowlist** of official domains:
`hdfcfund.com`, `amfiindia.com`, `sebi.gov.in`, `camsonline.com`, `groww.in`.

---

## Architecture Overview

Two phases: an **offline** indexing pipeline and an **online** inference pipeline.

```
OFFLINE (build the knowledge base)
  Official sources ─▶ scrape ─▶ parse ─▶ chunk (500/50 tok) ─▶ embed (BGE) ─▶ ChromaDB

ONLINE (answer a query)
  Query ─▶ Intent classifier ─┬─ advisory?      ─▶ refusal (no LLM)
                              ├─ out-of-scope?   ─▶ scope message (no LLM)
                              ├─ cache hit?      ─▶ cached answer (no LLM)
                              └─ retrieve (top-K) ─▶ assemble context ─▶ Groq LLM
                                                     ─▶ formatter (≤3 sent., 1 cite, footer) ─▶ UI
```

- **Embeddings:** `BAAI/bge-large-en-v1.5` (local, 1024-dim, **no API key**;
  falls back to `bge-small-en-v1.5` / 384-dim).
- **Vector store:** ChromaDB (cosine space), persisted to `vectorstore/index/`.
- **LLM:** Groq API — primary `llama-3.3-70b-versatile`, fallback `llama3-8b-8192`.
- **Compliance:** three layers — intent classifier → strict system prompt →
  formatter guard — plus client-side Groq rate-limiting (RPM/RPD/TPM/TPD).
- **UI:** Streamlit chat with a persistent disclaimer banner.

See [`Architecture.md`](./Architecture.md) for the full design and
[`ImplementationPlan.md`](./ImplementationPlan.md) for the phase-wise build plan.

---

## Prerequisites

- **Python 3.10+** (tested on 3.14, `win_amd64`).
- A free **Groq API key** — get one at <https://console.groq.com/keys>.
  (Only required for answer *generation*; embedding/retrieval run fully offline.)
- ~2–3 GB disk for the ML dependencies (`torch`, `transformers`, `sentence-transformers`).

> **Windows note:** `torch` and `pymupdf` load native DLLs that require the
> **Microsoft Visual C++ Redistributable (x64)**. If you hit a `WinError 126` /
> `DLL load failed` on import, install it once from
> <https://aka.ms/vs/17/release/vc_redist.x64.exe> (or
> `winget install Microsoft.VCRedist.2015+.x64`) and restart your shell.
> On Windows you can use the `py` launcher in place of `python` below.

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure secrets
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
# then edit .env and set GROQ_API_KEY=<your key>
```

---

## Quick Start (run the app)

The repository ships with a **pre-built corpus** (`data/`) and a **persisted
index** (`vectorstore/index/`), so once setup is done you can launch straight away:

```bash
streamlit run ui/app.py
```

Then open the local URL Streamlit prints (default <http://localhost:8501>).

- Without a `GROQ_API_KEY`, the app still boots and shows a friendly warning;
  retrieval works but answer generation is disabled.
- Without a built index, the app warns and points you to the build steps below.

---

## Building the Corpus & Index from Scratch

Run the offline pipeline **in order**. Each step is an idempotent, runnable module.

```bash
# 1. Scrape official sources -> data/raw/ + data/metadata.json
python -m ingestion.scraper

# 2. Parse + clean + chunk -> data/processed/chunks.jsonl
python -m ingestion.chunker

# 3. Embed + build the ChromaDB index -> vectorstore/index/
python -m vectorstore.indexer
```

Notes:
- The scraper enforces the domain **allowlist** and targets 15–25 documents
  across the 5 schemes; it exits non-zero if fewer than 15 are downloaded.
- `ingestion/scraper.py` uses `Playwright` for JS-rendered pages (Groww). If you
  scrape from scratch, install its browser once: `playwright install chromium`.
- The indexer builds into a temp dir and **atomically swaps** it in, asserting
  `vector count == chunk count` before publishing (never serves a partial index).

---

## Re-ingestion / Refresh Procedure

Figures (NAV, expense ratio, factsheets) change over time, so the corpus needs
periodic refresh. There are two paths: the **automated scheduler** (recommended)
and a **manual** run.

### Automated daily refresh (Phase 10)

A scheduled **GitHub Actions** workflow ([`.github/workflows/refresh.yml`](./.github/workflows/refresh.yml))
is the scheduler — the CI runner does the work, so no always-on host is needed.
It runs the whole offline pipeline via a single orchestrator, `scheduler/refresh.py`:

```bash
python -m scheduler.refresh            # scrape -> chunk -> reindex (skips if unchanged)
python -m scheduler.refresh --force    # rebuild even if the corpus is unchanged
python -m scheduler.refresh --dry-run  # report what would run; touch nothing
```

- **Cadence:** daily at **`0 5 * * *` (05:00 UTC = 10:30 AM IST)**. GitHub cron is
  **UTC-only**, so the IST time is encoded as its UTC equivalent. `workflow_dispatch`
  also allows a manual/one-off run (with an optional `force_reindex` input).
- **Change detection:** a corpus fingerprint (per-document `content_sha256`) is
  compared before/after the scrape; an unchanged corpus is a fast **no-op** (no
  re-embed) unless `--force`.
- **Minimum-corpus gate:** a scrape yielding fewer than `MIN_CORPUS_DOCS` (15) docs
  aborts **before** touching the index, so the previous good index keeps serving.
- **Single-flight:** the workflow `concurrency` group + a local `.refresh.lock`
  prevent overlapping runs.
- **Freshness:** each run records status + counts to `data/last_refresh.json`,
  which the UI sidebar surfaces as "Corpus last updated: <date>".
- **Publishing:** because the Actions runner is ephemeral, the workflow uploads
  the rebuilt `vectorstore/index/` + `data/` as a **workflow artifact** for the app
  to consume (swap in a data-branch commit / object-storage push for a persistent
  deploy target). Set the `GROQ_API_KEY` repo secret if your run needs it.

### Manual refresh

To refresh by hand without the orchestrator, re-run the offline pipeline and force
a clean rebuild of the index:

```bash
python -m ingestion.scraper          # re-fetch latest official docs
python -m ingestion.chunker          # re-parse + re-chunk
python -m vectorstore.indexer --reindex   # rebuild + atomic swap
```

Either path's `--reindex` build is atomic: the live index under `vectorstore/index/`
keeps serving until the new one is verified, and the previous good index is retained
on failure (the scheduler also keeps a timestamped snapshot under
`vectorstore/backups/`). After a successful refresh, each chunk's `last_updated`
advances, which is what the response footer surfaces.

---

## Testing & Evaluation

Phase 8 ships a compliance test suite and an end-to-end evaluation harness. Both
run standalone (no `pytest` required) and are also `pytest`-collectable.

```bash
# Deterministic compliance checks (offline, no index/LLM needed)
python -m tests.test_compliance

# End-to-end evaluation over the 85-case golden set -> reports/phase8_eval_report.md
python -m tests.evaluate            # OFFLINE (safe): no Groq quota spent
python -m tests.evaluate --live     # LIVE: generates answers (spends Groq quota)
```

- **Offline** mode clears the API key for the run and validates every safety
  gate (advisory leakage, PII echo/persistence, source allowlist, fallback).
- **Live** mode additionally validates citation / footer / sentence compliance on
  generated answers and paces itself to respect the free-tier rate limits.
- Results are written to [`reports/phase8_eval_report.md`](./reports/).

---

## Configuration

All tunables live in [`config.py`](./config.py) and can be overridden via `.env`
(see [`.env.example`](./.env.example)). The most useful ones:

| Variable | Default | Purpose |
|----------|---------|---------|
| `GROQ_API_KEY` | — | Groq key (required only for generation) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Primary LLM |
| `GROQ_MODEL_FALLBACK` | `llama3-8b-8192` | Degrade target on quota/429 |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Local embedding model |
| `LLM_MAX_TOKENS` | `256` | Completion cap (≤3-sentence answers) |
| `MAX_CONTEXT_CHUNKS` / `MAX_CONTEXT_TOKENS` | `3` / `2000` | Per-call context budget |
| `GROQ_RPM` / `GROQ_RPD` / `GROQ_TPM` / `GROQ_TPD` | `30` / `1000` / `12000` / `100000` | Free-tier rate limits (raise for paid tier) |
| `TOP_K` / `SIMILARITY_THRESHOLD` | `5` / `0.55` | Retrieval depth + "not in corpus" floor |

---

## Project Structure

```
RAG ChatBot/
├── config.py                # Central config (models, paths, chunking, retrieval, limits)
├── requirements.txt
├── .env.example
│
├── data/
│   ├── raw/                 # Downloaded PDFs / HTML (factsheets, kim, sid, faq)
│   ├── processed/           # chunks.jsonl + processing manifest
│   ├── metadata.json        # Per-document source registry + scrape dates
│   └── source_urls.json     # Curated URL allowlist
│
├── ingestion/
│   ├── scraper.py           # URL fetcher (requests/BeautifulSoup + Playwright + PDF)
│   ├── parser.py            # PDF/HTML text + table extraction, clean_text
│   └── chunker.py           # Token-accurate splitter + metadata tagging (runnable)
│
├── vectorstore/
│   ├── embedder.py          # BGE embedder (asymmetric passage/query, L2-normalised)
│   ├── indexer.py           # Build + persist ChromaDB (atomic swap, dim lock)
│   ├── retriever.py         # Scheme-scoped search, MMR dedup, scope detection
│   └── index/               # Persisted ChromaDB collection
│
├── rag/
│   ├── classifier.py        # Intent classifier (factual vs advisory, 5 categories)
│   ├── prompts.py           # System prompt + refusal / scope / busy templates
│   ├── pipeline.py          # End-to-end orchestration + response cache
│   ├── formatter.py         # ≤3 sentences, one allowlisted citation, footer, guards
│   ├── llm.py               # Groq client: rate-limit gate, 429 backoff, model degrade
│   └── ratelimit.py         # Sliding-window RPM/TPM/RPD/TPD limiter
│
├── ui/
│   └── app.py               # Streamlit chat interface (persistent disclaimer)
│
├── scheduler/
│   └── refresh.py           # Phase 10 orchestrator (scrape->chunk->reindex, atomic, locked)
│
├── .github/workflows/
│   └── refresh.yml          # Scheduled daily corpus refresh (cron 05:00 UTC = 10:30 IST)
│
├── tests/
│   ├── golden_set.py        # 85-case release-gate set (factual/advisory/OOC/PII/…)
│   ├── pii_scanner.py       # PAN/Aadhaar/account/OTP/email/phone detector
│   ├── test_compliance.py   # Deterministic offline compliance checks
│   └── evaluate.py          # End-to-end eval harness -> reports/
│
├── reports/                 # Generated evaluation report(s)
├── Architecture.md
├── ImplementationPlan.md
└── README.md
```

---

## Compliance & Safety

| Concern | Mechanism |
|---------|-----------|
| No advisory responses | Intent classifier pre-filter → strict system prompt → formatter advisory guard (3 layers) |
| Source grounding | LLM answers **only** from retrieved context; grounded "not in corpus" fallback otherwise |
| One citation per answer | Formatter injects exactly one **allowlisted** URL; model-invented / inline URLs are stripped |
| No PII collection | UI has no login or personal-data fields; queries live only in transient session state |
| No third-party data | Corpus restricted to the domain allowlist (audited in the test suite) |
| Cost / quota safety | Refusals, scope messages, cache hits, and out-of-corpus replies never call Groq |
| Disclaimer visibility | Persistent UI banner; appended to every refusal |
| Freshness | Metadata tracks source dates; shown in every response footer |

---

## Known Limitations

- **Static corpus between refreshes** — NAV / expense ratios can go stale;
  mitigated by the [refresh procedure](#re-ingestion--refresh-procedure) — a daily
  automated GitHub Actions scheduler (Phase 10), plus a manual fallback.
- **Scope** — limited to the 5 HDFC schemes above; other AMCs/schemes get a scope
  message.
- **English only** — sources and queries are English-language.
- **Stateless** — no multi-turn conversation memory (intentional, facts-only).
- **PDF parsing** — complex tables or scanned PDFs may extract imperfectly.
- **LLM hallucination risk** — mitigated by context-only prompting + formatter
  validation, but not eliminated; always verify against the cited source.
- **Free-tier throughput** — a burst of questions can be throttled ("at capacity")
  under Groq free-tier limits; it degrades gracefully rather than erroring.

---

## Deployment Guide

The app is a standard Streamlit application. Two common options:

### Streamlit Community Cloud

1. Push the repo to GitHub.
2. Create a new app on <https://share.streamlit.io> pointing at `ui/app.py`.
3. Add `GROQ_API_KEY` under **App settings → Secrets**.
4. Ensure a built index is available: either commit `vectorstore/index/` (it may
   be `.gitignore`d by default) **or** run the
   [build steps](#building-the-corpus--index-from-scratch) as part of your deploy.
5. To keep the deployed index fresh, enable the Phase 10
   [GitHub Actions scheduler](#automated-daily-refresh-phase-10) and have your
   deploy consume its published index artifact (or point the workflow at a
   data-branch commit / object-storage push your app reads on boot).

### Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t mf-faq-assistant .
docker run -p 8501:8501 -e GROQ_API_KEY=<your key> mf-faq-assistant
```

For an air-gapped deploy, pre-download the BGE weights during the image build so
the first run doesn't need network access.

---

## Disclaimer

> This assistant provides **facts-only** information sourced from official public
> documents (HDFC AMC, AMFI, SEBI, CAMS). It **does not constitute financial or
> investment advice**. For guidance on investment decisions, consult a
> SEBI-registered financial advisor or visit
> <https://www.amfiindia.com/investor-corner>.
>
> **Facts-only. No investment advice.**
