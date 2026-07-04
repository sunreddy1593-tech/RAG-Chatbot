# Architecture: Mutual Fund FAQ Assistant (RAG-Based)

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture Diagram](#2-high-level-architecture-diagram)
3. [Component Breakdown](#3-component-breakdown)
   - [3.1 Data Ingestion Pipeline](#31-data-ingestion-pipeline)
   - [3.2 Document Processing & Chunking](#32-document-processing--chunking)
   - [3.3 Embedding & Vector Store](#33-embedding--vector-store)
   - [3.4 Query Processing & Retrieval](#34-query-processing--retrieval)
   - [3.5 Response Generation (LLM)](#35-response-generation-llm)
   - [3.6 Refusal Handler](#36-refusal-handler)
   - [3.7 Response Formatter](#37-response-formatter)
   - [3.8 User Interface](#38-user-interface)
   - [3.9 Scheduler & Data Refresh](#39-scheduler--data-refresh)
4. [RAG Pipeline – Step-by-Step Flow](#4-rag-pipeline--step-by-step-flow)
5. [Data Sources & Corpus](#5-data-sources--corpus)
6. [Technology Stack](#6-technology-stack)
7. [Directory Structure](#7-directory-structure)
8. [Compliance & Safety Layer](#8-compliance--safety-layer)
9. [Known Limitations](#9-known-limitations)

---

## 1. System Overview

The Mutual Fund FAQ Assistant is a **Retrieval-Augmented Generation (RAG)** system that answers factual queries about HDFC Mutual Fund schemes. It operates in two distinct phases:

- **Offline Phase (Indexing):** Official documents are scraped, chunked, embedded, and stored in a vector database.
- **Online Phase (Inference):** User queries are embedded, matched against stored chunks, and passed to an LLM along with a strict system prompt to generate source-backed, facts-only responses.

The system is designed around three core principles:
1. **Factual accuracy** — all answers grounded in retrieved official documents
2. **Compliance** — no investment advice, no user data collection
3. **Transparency** — every response carries a source citation and a last-updated date

---

## 2. High-Level Architecture Diagram

```
        ┌─────────────────────────────┐
        │  Scheduler (daily cron job) │ ──── triggers on schedule ────┐
        └─────────────────────────────┘                              │
                                                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OFFLINE PHASE (Indexing)                     │
│              (re-run automatically by the Scheduler daily)          │
│                                                                     │
│  Official Sources          Document Processor        Vector Store   │
│  ┌─────────────┐          ┌──────────────────┐      ┌───────────┐  │
│  │ HDFC AMC    │─────────▶│  PDF / HTML      │─────▶│           │  │
│  │ AMFI        │  Scrape  │  Parser          │ Embed│  ChromaDB │  │
│  │ SEBI        │          │  Chunker         │──────│  / FAISS  │  │
│  │ CAMS        │          │  Metadata Tagger │      │           │  │
│  └─────────────┘          └──────────────────┘      └───────────┘  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        ONLINE PHASE (Inference)                     │
│                                                                     │
│  User Interface                                                     │
│  ┌──────────────┐                                                   │
│  │  Chat UI     │──── User Query                                    │
│  │  (Streamlit) │         │                                         │
│  └──────────────┘         ▼                                         │
│                    ┌─────────────────┐                              │
│                    │ Query Classifier │ ──── Advisory? ──▶ Refusal  │
│                    │ (Intent Check)  │           Handler            │
│                    └────────┬────────┘                              │
│                             │ Factual                               │
│                             ▼                                       │
│                    ┌─────────────────┐      ┌───────────┐          │
│                    │ Query Embedder  │─────▶│ Vector DB │          │
│                    │ (same model)    │      │ Retriever │          │
│                    └─────────────────┘      └─────┬─────┘          │
│                                                   │ Top-K Chunks   │
│                                                   ▼                │
│                                          ┌─────────────────┐       │
│                                          │   LLM via Groq  │       │
│                                          │  (llama3-8b /   │       │
│                                          │  mixtral-8x7b)  │       │
│                                          └────────┬────────┘       │
│                                                   │                │
│                                                   ▼                │
│                                          ┌─────────────────┐       │
│                                          │ Response        │       │
│                                          │ Formatter       │       │
│                                          │ + Citation      │       │
│                                          │ + Last Updated  │       │
│                                          └────────┬────────┘       │
│                                                   │                │
│                                          ┌────────▼────────┐       │
│                                          │   Chat UI       │       │
│                                          │   (Response)    │       │
│                                          └─────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Breakdown

### 3.1 Data Ingestion Pipeline

**Purpose:** Collect and store raw content from official public sources.

**Inputs:** URLs from HDFC AMC, AMFI, SEBI, CAMS

**Process:**
- For **PDF documents** (Factsheets, KIM, SID): download and parse using `PyMuPDF` or `pdfplumber`
- For **HTML pages** (FAQ pages, Groww scheme pages): scrape using `BeautifulSoup` or `Playwright` (for JS-rendered pages)
- All raw content is saved locally with metadata (source URL, document type, scheme name, scrape date)

**Output:** Raw text files with associated metadata JSON

**Corpus Sources:**

| Source | Type | URL Base |
|--------|------|----------|
| HDFC AMC | Factsheets, KIM, SID | hdfcfund.com |
| AMFI | Scheme data, NAV, guidance | amfiindia.com |
| SEBI | Investor education, circulars | sebi.gov.in |
| CAMS | Statement download guide | camsonline.com |
| Groww | Scheme overview pages (reference) | groww.in/mutual-funds |

---

### 3.2 Document Processing & Chunking

**Purpose:** Transform raw text into clean, semantically meaningful chunks suitable for embedding.

**Steps:**

1. **Text Cleaning** — strip headers/footers, boilerplate legal text, page numbers
2. **Chunking Strategy** — recursive character text splitter
   - Chunk size: ~500 tokens
   - Overlap: ~50 tokens (to preserve context across chunk boundaries)
3. **Metadata Tagging** — each chunk is tagged with:

```json
{
  "scheme_name": "HDFC Large Cap Fund – Direct Growth",
  "document_type": "Factsheet",
  "source_url": "https://hdfcfund.com/...",
  "last_updated": "2026-06-01",
  "category": "Large Cap Equity"
}
```

**Tool:** `LangChain RecursiveCharacterTextSplitter`

---

### 3.3 Embedding & Vector Store

**Purpose:** Convert text chunks into dense vector representations and store them for similarity search.

**Embedding Model:**
- Primary: `BAAI/bge-large-en-v1.5` (BGE — open-source via HuggingFace `sentence-transformers`)
- Fallback: `BAAI/bge-small-en-v1.5` (lighter, faster, 384-dim)
- All chunks from the corpus are embedded using the same model used at query time
- BGE models are loaded locally — no external API calls required for embedding

**Vector Store:**
- **ChromaDB** (local, lightweight) — preferred for development
- Alternative: **FAISS** (Facebook AI Similarity Search) for in-memory indexing

**Index Schema:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique chunk ID |
| `embedding` | float[] | Dense vector (1024-dim for `bge-large-en-v1.5`) |
| `document` | string | Raw chunk text |
| `metadata` | object | Scheme name, doc type, source URL, date |

---

### 3.4 Query Processing & Retrieval

**Purpose:** Convert a user query into an embedding and retrieve the most relevant document chunks.

**Steps:**

1. **Query Classification (Intent Check)**
   - A lightweight classifier (rule-based or prompted LLM) checks if the query is:
     - `FACTUAL` → proceed with retrieval
     - `ADVISORY` → route to Refusal Handler
   - Advisory patterns to detect:
     - "Should I invest…"
     - "Which fund is better…"
     - "Is this a good fund…"
     - "Will this fund give returns…"

2. **Query Embedding** — embed the cleaned user query using the same model as the corpus

3. **Similarity Search** — retrieve **top-K = 3–5** most similar chunks from the vector store using cosine similarity

4. **Context Assembly** — concatenate retrieved chunks into a single context block passed to the LLM

---

### 3.5 Response Generation (LLM)

**Purpose:** Generate a factual, concise, source-grounded answer using retrieved context.

**LLM:** Groq API (ultra-low latency inference)

| Model | Groq Model ID | Context Window | Notes |
|-------|---------------|----------------|-------|
| Primary | `llama3-8b-8192` | 8192 tokens | Fast, cost-effective, strong instruction following |
| Alternative | `mixtral-8x7b-32768` | 32768 tokens | Larger context, better for long documents |
| Fallback | `llama3-70b-8192` | 8192 tokens | Higher accuracy for complex queries |

**Groq Advantages for this project:**
- Free tier with generous rate limits
- Sub-second response latency
- No data stored by Groq beyond the request
- Drop-in compatible with OpenAI SDK (`groq` Python package)

**System Prompt (strict):**

```
You are a facts-only Mutual Fund FAQ assistant for HDFC Mutual Fund schemes.

Rules:
1. Answer ONLY using the provided context. Do not use prior knowledge.
2. Limit your response to a maximum of 3 sentences.
3. Do not provide investment advice, recommendations, or performance predictions.
4. End every response with exactly one source citation link from the metadata.
5. Append a footer: "Last updated from sources: <date from metadata>"
6. If the context does not contain the answer, say:
   "This information is not available in the current corpus. Please refer to [source link]."
```

**Input to LLM:**
- System prompt (above)
- Retrieved context chunks (with metadata)
- User query

**Output:** A 1–3 sentence factual answer with citation and footer

---

### 3.6 Refusal Handler

**Purpose:** Gracefully decline advisory or out-of-scope queries.

**Trigger Conditions:**
- Query classified as `ADVISORY` by the intent classifier
- Query asks for return predictions, fund comparisons, or personal investment recommendations

**Refusal Response Template:**

```
I'm only able to provide factual information about mutual fund schemes —
such as expense ratios, exit loads, or SIP minimums.

For guidance on investment decisions, please refer to a SEBI-registered
financial advisor or visit: https://www.amfiindia.com/investor-corner

Facts-only. No investment advice.
```

---

### 3.7 Response Formatter

**Purpose:** Enforce a consistent, compliant output structure for every factual response.

**Output Structure:**

```
<Answer — max 3 sentences>

Source: <single URL from retrieved chunk metadata>
Last updated from sources: <date>
```

**Validation checks before displaying:**
- Response does not contain advisory language (e.g., "you should", "I recommend", "better than")
- Exactly one source URL is present
- Footer date is present and formatted correctly

---

### 3.8 User Interface

**Purpose:** Provide a clean, minimal chat interface aligned with compliance requirements.

**Framework:** Streamlit (Python)

**UI Elements:**

| Element | Description |
|---------|-------------|
| Welcome message | Brief intro to what the assistant can answer |
| Example questions | 3 pre-filled clickable prompts |
| Chat input box | Free-text query input |
| Response area | Displays answer, source link, and last-updated footer |
| Disclaimer banner | Persistent: "Facts-only. No investment advice." |

**Example Starter Questions (shown in UI):**
1. "What is the expense ratio of HDFC Large Cap Fund?"
2. "What is the exit load for HDFC Small Cap Fund?"
3. "What is the minimum SIP amount for HDFC Mid Cap Fund?"

---

### 3.9 Scheduler & Data Refresh

**Purpose:** Keep the served corpus current by automatically re-running the **offline (indexing) phase** on a daily schedule, so answers always reflect the latest official documents (NAVs, expense ratios, factsheets). This directly mitigates the "static corpus" limitation (§9) and keeps the `Last updated from sources` footer meaningful.

**Trigger:** A daily cron-style schedule (off-peak, e.g. 02:00 IST). The default implementation is an in-process `APScheduler` `BackgroundScheduler`; equivalent OS-level (`cron` / Windows Task Scheduler) or cloud (`Kubernetes CronJob`, scheduled GitHub Action, managed scheduler) triggers are supported.

**Orchestration (`scheduler/refresh.py` → `run_refresh()`):**

```
Scheduler fires (daily)
       │
       ▼
[1] Scraper    — re-fetch allowlisted sources (robots-aware, backoff on 429)
       │
       ▼
[2] Parser + Chunker — re-parse & re-chunk changed documents
       │
       ▼
[3] Indexer (--reindex) — embed & build a NEW index in a temp dir
       │
       ▼
[4] Atomic swap — publish the new index; keep the previous good one on failure
       │
       ▼
[5] Record last_refresh (timestamp, status, doc/chunk counts)
```

**Key properties:**

| Property | Mechanism |
|----------|-----------|
| Freshness | Daily re-scrape + re-index; updates `last_updated` / `last_refresh` |
| Change detection | Content checksum per document; unchanged docs skip re-embedding (idempotent, edge 1.6) |
| Atomicity | Build to temp dir + pointer swap; never serve a partial index (edge 3.4 / 3.7) |
| No overlap | Single-flight lockfile prevents concurrent refreshes |
| Safety on failure | Retry transient failures; keep last good index; block publish if `< MIN_CORPUS_DOCS` (edge 1.9) |
| Observability | `last_refresh` status file surfaced in the UI ("Corpus last refreshed: …") |

**Configuration:** `REFRESH_ENABLED`, `REFRESH_CRON` (or time + `REFRESH_TIMEZONE`), `MIN_CORPUS_DOCS`, and previous-index retention live in `config.py`.

---

## 4. RAG Pipeline – Step-by-Step Flow

```
User types query
       │
       ▼
[1] Intent Classifier
       │
       ├── ADVISORY ──▶ [2a] Refusal Handler ──▶ Polite refusal + AMFI/SEBI link
       │
       └── FACTUAL
              │
              ▼
       [2b] Query Embedder
              │
              ▼
       [3] Vector Store — cosine similarity search → Top-K chunks
              │
              ▼
       [4] Context Assembly (chunks + metadata)
              │
              ▼
       [5] LLM (system prompt + context + query)
              │
              ▼
       [6] Response Formatter
              │  ├── Answer (≤ 3 sentences)
              │  ├── Source URL
              │  └── Last updated: <date>
              │
              ▼
       [7] UI — display to user
```

---

## 5. Data Sources & Corpus

| # | Document Type | Scheme | Source |
|---|---------------|--------|--------|
| 1 | Factsheet | HDFC Large Cap Fund | hdfcfund.com |
| 2 | Factsheet | HDFC Mid Cap Fund | hdfcfund.com |
| 3 | Factsheet | HDFC Small Cap Fund | hdfcfund.com |
| 4 | Factsheet | HDFC Gold ETF FoF | hdfcfund.com |
| 5 | Factsheet | HDFC Silver ETF FoF | hdfcfund.com |
| 6 | KIM | HDFC Large Cap Fund | hdfcfund.com |
| 7 | KIM | HDFC Mid Cap Fund | hdfcfund.com |
| 8 | KIM | HDFC Small Cap Fund | hdfcfund.com |
| 9 | KIM | HDFC Gold ETF FoF | hdfcfund.com |
| 10 | KIM | HDFC Silver ETF FoF | hdfcfund.com |
| 11 | SID | HDFC Large Cap Fund | hdfcfund.com / SEBI |
| 12 | SID | HDFC Mid Cap Fund | hdfcfund.com / SEBI |
| 13 | SID | HDFC Small Cap Fund | hdfcfund.com / SEBI |
| 14 | AMC FAQ / Help | HDFC AMC General | hdfcfund.com |
| 15 | AMFI Guidance | Investor Corner | amfiindia.com |
| 16 | SEBI Investor Education | Mutual Fund Basics | sebi.gov.in |
| 17 | Statement Download Guide | HDFC / CAMS | camsonline.com |

---

## 6. Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Scraping | `BeautifulSoup`, `Playwright`, `requests` | Fetch HTML & PDF documents |
| PDF Parsing | `PyMuPDF` / `pdfplumber` | Extract text from factsheets, KIM, SID |
| Chunking | `LangChain` `RecursiveCharacterTextSplitter` | Split documents into overlapping chunks |
| Embedding | `BAAI/bge-large-en-v1.5` via `sentence-transformers` | Dense 1024-dim vector representations (local, no API key) |
| Vector Store | `ChromaDB` / `FAISS` | Similarity search over corpus |
| LLM | `llama3-8b-8192` / `mixtral-8x7b-32768` via **Groq API** | Ultra-low latency response generation |
| Orchestration | `LangChain` / `LlamaIndex` | RAG pipeline management |
| Scheduling | `APScheduler` (or `cron` / Task Scheduler / K8s `CronJob`) | Daily automated corpus refresh |
| UI | `Streamlit` | Chat interface |
| Language | Python 3.10+ | Primary implementation language |

---

## 7. Directory Structure

```
RAG ChatBot/
│
├── data/
│   ├── raw/                    # Downloaded PDFs and HTML files
│   │   ├── factsheets/
│   │   ├── kim/
│   │   ├── sid/
│   │   └── faq/
│   ├── processed/              # Cleaned text chunks with metadata
│   └── metadata.json           # Source URL registry with scrape dates
│
├── ingestion/
│   ├── scraper.py              # URL fetcher (HTML + PDF)
│   ├── parser.py               # PDF/HTML text extractor
│   └── chunker.py              # Text splitter and metadata tagger
│
├── vectorstore/
│   ├── embedder.py             # Embedding model wrapper
│   ├── indexer.py              # Build and persist ChromaDB index
│   └── retriever.py            # Similarity search interface
│
├── rag/
│   ├── classifier.py           # Intent classifier (factual vs advisory)
│   ├── pipeline.py             # End-to-end RAG orchestration
│   ├── prompts.py              # System prompts and refusal templates
│   ├── formatter.py            # Response structure enforcement
│   ├── llm.py                  # Groq client (rate-limit gate, backoff, degrade)
│   └── ratelimit.py            # Sliding-window RPM/TPM/RPD/TPD limiter
│
├── scheduler/
│   ├── refresh.py              # run_refresh(): scrape -> parse -> chunk -> reindex
│   └── worker.py               # APScheduler daily trigger (standalone worker)
│
├── ui/
│   └── app.py                  # Streamlit chat interface
│
├── problemstatement.md
├── Architecture.md
├── README.md
└── requirements.txt
```

---

## 8. Compliance & Safety Layer

| Concern | Mechanism |
|---------|-----------|
| No advisory responses | System prompt constraint + intent classifier pre-filter |
| Source grounding | LLM instructed to answer only from retrieved context |
| Single citation per response | Response formatter validates exactly one URL |
| No PII collection | UI has no login, no form fields for personal data |
| No third-party data | Corpus URL allowlist restricts scraping to approved domains |
| Disclaimer visibility | Persistent banner in UI; appended to every refusal response |
| Data freshness | Metadata tracks scrape date; shown in every response footer |

---

## 9. Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|-----------|
| Corpus is static between refreshes | NAV, expense ratio may become outdated | **Scheduler component (§3.9)** re-runs ingestion + indexing **daily** with an atomic index swap; `last_refresh` is surfaced in the UI |
| LLM hallucination risk | May generate plausible but unsourced facts | Strict prompt + context-only instruction + source validation in formatter |
| PDF parsing accuracy | Scanned PDFs or complex tables may parse incorrectly | Use `pdfplumber` with fallback OCR (`pytesseract`) for scanned documents |
| Groww pages are JS-rendered | `requests` + `BeautifulSoup` may miss dynamic content | Use `Playwright` for JS-heavy pages |
| No multi-turn memory | Each query is independent; no conversation context | Acceptable for facts-only FAQ; stateless design is intentional |
| Limited to 5 HDFC schemes | Cannot answer questions on other AMCs or schemes | Clearly communicate scope in UI welcome message |
| Language support | English only | Scope is English-language sources and queries only |

---

> **Disclaimer:** This assistant provides facts-only information from official public sources. It does not constitute financial or investment advice.
