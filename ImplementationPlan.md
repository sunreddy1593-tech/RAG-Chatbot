# Implementation Plan: Mutual Fund FAQ Assistant (RAG-Based)

This document breaks down the build of the **HDFC Mutual Fund Facts-Only FAQ Assistant** into sequential, verifiable phases, derived directly from [`Architecture.md`](./Architecture.md).

## Phase Overview

| Phase | Name | Goal | Key Output |
|-------|------|------|-----------|
| 0 | Project Setup | Scaffold repo, env, dependencies | Runnable skeleton |
| 1 | Corpus Collection | Gather official source URLs & raw docs | `data/raw/` populated |
| 2 | Document Processing | Parse, clean, chunk, tag metadata | `data/processed/` chunks |
| 3 | Embedding & Indexing | Build the vector store | Persisted ChromaDB index |
| 4 | Retrieval Layer | Query embedding + similarity search | Working retriever |
| 5 | Intent Classification & Refusal | Filter advisory queries | Classifier + refusal handler |
| 6 | LLM Response Generation | Grounded, compliant answers | Formatted RAG responses |
| 7 | User Interface | Streamlit chat app styled to the Stitch design system | Deployable UI (5 screen states) |
| 8 | Compliance & Testing | Validate safety & accuracy | Test suite + report |
| 9 | Documentation & Delivery | README, deployment, handoff | Final deliverables |
| 10 | Scheduled Data Refresh | Auto re-run ingestion daily to keep the corpus fresh | GitHub Actions workflow + atomic re-index |

> **Note on Phase 10:** the scheduler is an *operational* component that re-runs the offline pipeline (Phases 1→2→3) via a **scheduled GitHub Actions workflow**. It depends only on those phases and can be built any time after Phase 3; it is numbered last because it is a production/deployment concern rather than a core inference step.

---

## Phase 0 — Project Setup & Scaffolding

**Goal:** Establish a clean, reproducible development environment.

**Tasks:**
- [ ] Initialize git repository and `.gitignore` (exclude `data/raw/`, `.env`, vector index)
- [ ] Create directory structure as defined in Architecture §7
- [ ] Set up Python 3.10+ virtual environment
- [ ] Create `requirements.txt` with pinned dependencies:
  - `langchain`, `langchain-groq`, `chromadb`, `sentence-transformers`, `beautifulsoup4`, `playwright`, `pymupdf`, `pdfplumber`, `streamlit`, `python-dotenv`, `groq`
- [ ] Create `.env.example` for API keys (`GROQ_API_KEY`)
- [ ] Add config module (`config.py`) for model names, chunk sizes, top-K, paths

**Deliverable:** Empty but runnable project skeleton.

**Acceptance Criteria:** `pip install -r requirements.txt` succeeds; folder structure matches architecture.

---

## Phase 1 — Corpus Collection (Data Ingestion)

**Goal:** Collect 15–25 official public URLs and download raw documents.

**Tasks:**
- [ ] Build a curated **URL allowlist** (`data/source_urls.json`) restricted to: `hdfcfund.com`, `amfiindia.com`, `sebi.gov.in`, `camsonline.com`, `groww.in`
- [ ] Implement `ingestion/scraper.py`:
  - HTML fetch via `requests` + `BeautifulSoup`
  - JS-rendered pages (Groww) via `Playwright`
  - PDF download for Factsheets, KIM, SID
- [ ] Save raw files to `data/raw/{factsheets,kim,sid,faq}/`
- [ ] Record metadata per document (source URL, doc type, scheme, scrape date) in `data/metadata.json`

**Deliverable:** Populated `data/raw/` with all corpus documents.

**Acceptance Criteria:** ≥15 documents downloaded across all 5 schemes; every file has a metadata entry.

**Dependencies:** Phase 0

---

## Phase 2 — Document Processing & Chunking

**Goal:** Convert raw documents into clean, metadata-tagged chunks.

**Tasks:**
- [x] Implement `ingestion/parser.py`:
  - PDF text extraction (`PyMuPDF`, fallback `pdfplumber`)
  - `pdfplumber` also recovers tables as pipe-delimited rows so expense-ratio / exit-load grids keep their row/column relationships
  - HTML text extraction (strip `script/style/nav/header/footer/aside/form`; prefer `<main>`/`<article>`)
  - Shared `clean_text` pass: normalise whitespace, drop standalone page-number lines, preserve footnote markers (e.g. `1.25%*`)
- [x] Implement `ingestion/chunker.py`:
  - `RecursiveCharacterTextSplitter` — chunk size ~500 tokens, overlap ~50
  - Attach metadata to each chunk (scheme, doc type, source URL, last_updated, category)
- [x] Save processed chunks to `data/processed/` (JSONL) via a runnable driver (`python -m ingestion.chunker`)

#### Chunking Strategy (as implemented)

Data analysis of the parsed corpus drove two deliberate refinements over the naive ~500-token character split:

1. **Token-accurate sizing (not character approximation).** Chunk size (500) and overlap (50) are measured with the **same tokenizer as the embedding model** (`BAAI/bge-large-en-v1.5`) via a token-counting `length_function`. This matters because a char/token approximation badly underestimates tokens for finance text — dense numeric tables run ≈1.3 chars/token, not ~4 — which left **~18% of chunks over 500 tokens and 109 chunks over BGE's hard 512-token input window (max 1567)**. Those tails would be **silently truncated at embed time** in Phase 3. Token-accurate sizing guarantees every chunk fits the window (verified: max = 500 tokens, 0 over 512). If the tokenizer can't be loaded (fully offline first run), it falls back to a conservative 3 chars/token approximation.
2. **Recursive separators favouring semantic boundaries.** Split priority is `["\n\n", "\n", ". ", " ", ""]` so paragraph/line/sentence boundaries are preferred and the ~50-token overlap preserves facts that straddle a boundary (edge 2.3).

Additional quality gates:
- **Metadata validation gate** — a chunk missing any of the 5 mandatory fields is rejected (edge 2.6). Each chunk also carries `source_domain` + `chunk_index`, and a **globally-unique id** (`scheme__doc_type__<url-hash>__NNNN`) so documents sharing scheme/doc_type/domain (e.g. multiple SEBI guidance pages) never collide.
- **Empty/garbage filtering** — drop chunks under 80 chars, under 20% letters, or with fewer than 5 real words (strips stray table fragments / separators). Number-heavy factsheet tables are intentionally kept since they interleave company names + ratings and carry real facts.
- **Cross-corpus de-duplication** — identical chunk texts are dropped by SHA-256 (edge 1.7).

**Output:** `data/processed/chunks.jsonl` (one `{id, text, metadata}` per line) + `data/processed/processed_manifest.json` (per-document counts + run report).

> **Note / known trade-off:** The shared HDFC monthly factsheet is a 40+ scheme document tagged `scheme: "Shared (HDFC AMC)"` and contributes the majority of chunks. The 5 in-scope schemes are additionally covered by their own scheme pages and per-scheme factsheets; scheme-scoped retrieval filtering is handled in Phase 4 to avoid cross-scheme dilution (edge 2.8 / 5.2).

**Deliverable:** Clean chunk set with full metadata.

**Acceptance Criteria:** Each chunk carries all 5 metadata fields; no empty/garbage chunks; spot-check 10 chunks for readability. *(Met: 669 chunks, 0 missing fields, 0 duplicate ids, all ≤512 tokens.)*

**Dependencies:** Phase 1

---

## Phase 3 — Embedding & Vector Store Indexing

**Goal:** Embed all chunks and build a persistent searchable index.

**Tasks:**
- [x] Implement `vectorstore/embedder.py`:
  - Load `BAAI/bge-large-en-v1.5` locally via `sentence-transformers`
  - Reusable `BGEEmbedder` class + singleton; L2-normalised, asymmetric `embed_passages` (no prefix) / `embed_query` (instruction prefix)
  - Fallback: `BAAI/bge-small-en-v1.5` (384-dim) if the primary model can't load
  - No API key required — runs fully offline; auto-selects CUDA when available
- [x] Implement `vectorstore/indexer.py`:
  - Embed all processed chunks (batched)
  - Store vectors + documents + 7 scalar metadata fields in ChromaDB (`hnsw:space=cosine`), persisted to `vectorstore/index/`
  - Atomic temp-dir build + swap (edge 3.4); post-index assertion vectors == chunks (edge 3.3); dim-locked `get_collection` loader (edge 3.2 / 3.5)
- [x] Add a re-index script for periodic corpus refresh (`python -m vectorstore.indexer --reindex`)

#### Embedding Strategy (informed by the produced chunk set)

The Phase 2 output is **669 chunks**, each already sized in true BGE tokens (min 26 / avg 412 / **max 500**, 0 over the 512-token window). This directly shapes how they are embedded:

1. **Passages embedded as-is; instruction prefix on the query only.** BGE is an *asymmetric* retrieval model. Corpus chunks are encoded with **no prefix**; at query time (Phase 4) the query is prefixed with BGE's retrieval instruction `"Represent this sentence for searching relevant passages:"`. Applying the prefix to the corpus would break this asymmetry and degrade recall — so the instruction lives in the query path, not the indexer.
2. **L2-normalised embeddings + cosine space.** Encode with `normalize_embeddings=True` (unit vectors) and create the Chroma collection with `hnsw:space="cosine"`. This makes similarity scores bounded and comparable, so the `SIMILARITY_THRESHOLD = 0.30` gate (used for the "not in corpus" fallback in Phases 4/6) is meaningful and stable.
3. **No truncation risk.** Because chunks are capped at 500 tokens against the *same* tokenizer BGE uses, every chunk embeds in full — the reason Phase 2 switched from a char approximation to token-accurate sizing.
4. **One model for corpus and query (dimension-locked).** The configured model (`config.EMBEDDING_MODEL`) is used for both indexing and querying; mixing `bge-large` (1024-dim) with `bge-small` (384-dim) is forbidden (edge 3.5). Persist the model name + dim in the index and **validate the dimension on load, rebuilding if it changed** (edge 3.2).
5. **Batched, streamed encoding.** Encode in batches (e.g. `batch_size=32`, `show_progress_bar`) rather than all at once (edge 3.6). At this corpus size the vector matrix is tiny (669 × 1024 float32 ≈ 2.7 MB), and it runs comfortably on CPU; auto-select CUDA when available.
6. **ChromaDB record schema.** Each record stores `id`, the 1024-dim `embedding`, the chunk `document` text, and the 7 chunk metadata fields — `scheme, doc_type, source_url, last_updated, category, source_domain, chunk_index`. All values are scalars (str/int), verified **Chroma-compatible with 0 non-scalar values**, so no metadata flattening is needed. Collection name: `config.COLLECTION_NAME` (`mf_faq_corpus`), persisted to `vectorstore/index/`.
7. **Atomic build + post-index assertion.** Build to a temp directory and swap on success to avoid serving a partial index (edge 3.4); after building, assert **vector count == chunk count (669)** and fail the build on any delta (edge 3.3).
8. **Offline-first.** BGE weights are cached locally on first download; document a pre-download step for air-gapped deploys and emit a clear error if weights are missing (edge 3.1 / 10.7).

> **Distribution note (drives Phase 4, not embedding):** the chunk set is skewed — `Shared (HDFC AMC)` ≈ 465 chunks (69%) and the Small Cap SID ≈ 123, while the other four schemes have only 10–16 chunks each. Embedding treats every chunk equally; **scheme-scoped metadata filtering at query time** (Phase 4) is what prevents the shared factsheet from diluting scheme-specific results (edge 2.8 / 5.2). No per-scheme weighting is applied at embed time.

> **Deployment note (default model switched to bge-small):** the figures above were measured with `bge-large-en-v1.5` (1024-dim). For the Streamlit Community Cloud free tier (RAM-limited), `config.EMBEDDING_MODEL` now **defaults to `bge-small-en-v1.5` (384-dim, ~130 MB)**, and the Phase 10 workflow rebuilds/commits a **384-dim** index to match. The two BGE v1.5 models share the same tokenizer and 512-token window, so the Phase 2 chunk set (669 chunks) is unchanged; only the vector dimension differs. `bge-large` remains available for higher recall on a bigger host — set `EMBEDDING_MODEL` and rebuild, since the dimension is locked to the index (edge 3.2/3.5).

**Deliverable:** Persisted ChromaDB index.

**Acceptance Criteria:** Index loads from disk; count of vectors == count of chunks (669); sample similarity query returns relevant chunks. *(Met: 669/669 vectors, cosine/1024-dim; query "expense ratio of HDFC Large Cap Fund" returns the Large Cap factsheet as top hit.)*

**Dependencies:** Phase 2

---

## Phase 4 — Retrieval Layer

**Goal:** Retrieve the most relevant chunks for a given query.

**Tasks:**
- [x] Implement `vectorstore/retriever.py`:
  - Embed query using the same `BAAI/bge-large-en-v1.5` model as the corpus (via `embed_query`, with BGE instruction prefix)
  - Cosine similarity search, return top-K chunks with metadata
  - Scheme-scoped metadata filtering (`detect_scheme` + Chroma `where`), out-of-scope AMC detection (`detect_out_of_scope`), MMR de-duplication, recalibrated similarity floor, empty/over-long query hygiene
- [x] Add context assembly helper (`assemble_context`: concatenate chunks + carry `source_url/scheme/doc_type/last_updated`)

#### Retrieval Strategy (informed by probing the built index)

Probing the live 669-vector index with representative queries produced concrete, sometimes surprising, results that drive the strategy below:

| Query | Top-1 cosine sim | Top-K schemes returned |
|-------|:---:|-----------------------|
| *expense ratio of HDFC Large Cap Fund* (in-scope) | 0.758 | correct scheme is top-1 |
| *exit load for HDFC Small Cap Fund* (in-scope) | 0.797 | **top-5 all `Shared (HDFC AMC)` — wrong schemes** |
| *minimum SIP for HDFC Mid Cap Fund* (in-scope) | 0.787 | mixed; shared factsheet intrudes |
| *capital of France* (out-of-corpus) | 0.372 | irrelevant macro/table chunks |
| *current price of Tesla stock* (out-of-corpus) | 0.606 | Groww nav boilerplate |
| *expense ratio of SBI Bluechip* (out-of-scope AMC) | 0.666 | HDFC content (wrong) |

Four findings and the resulting strategy:

1. **Scheme-scoped metadata filtering is mandatory, not optional.** Without it, *"exit load for HDFC Small Cap Fund"* returns **five `Shared (HDFC AMC)` chunks about unrelated schemes and the correct scheme does not appear in the top-5** — the 465-chunk shared factsheet dilutes everything (edge 2.8 / 5.2). Restricting the same query with Chroma `where={"scheme": "HDFC Small Cap Fund - Direct Growth"}` returns the correct scheme's expense-ratio/exit-load content. So: **detect the target scheme from the query** (keyword/alias map over the 5 canonical names — "large cap", "small cap", "mid cap", "gold", "silver") and pass it as a metadata filter; widen (or `$or` in the shared AMC + regulatory docs) only if the filtered result set is too small. For general/regulatory questions with no scheme mention, search unfiltered.
2. **The `SIMILARITY_THRESHOLD = 0.30` default is far too low — recalibrate to ≈0.55–0.60.** BGE cosine scores on this corpus sit in a **compressed high band**: even *"capital of France"* scores 0.37 (which would pass a 0.30 gate), while genuine in-scope facts score 0.74–0.80. A single global threshold cannot cleanly separate finance-adjacent out-of-scope queries (*Tesla* 0.61, *SBI Bluechip* 0.67) from in-scope ones. Therefore: raise the "not in corpus" floor to ~0.55–0.60 **and** combine it with (a) scheme/scope detection and (b) the LLM's context-only "not in corpus" fallback (Phase 6). Treat the raw score as a weak signal; **relative ranking + metadata filtering matter more than the absolute number.**
3. **Out-of-scope AMC/scheme detection is needed (edge 5.5).** *"SBI Bluechip"* scored 0.67 and returned HDFC chunks — thresholding alone would let a wrong-AMC answer through. Detect schemes/AMCs outside the 5-scheme scope and return a scope-limitation message instead of retrieving.
4. **Query instruction prefix is marginal here; diversity matters more.** Encoding the query with vs without BGE's instruction prefix changed top-1 similarity negligibly (0.797 vs 0.806) — expected for bge-*v1.5* — so the prefix is retained as canonical but is not decisive. More useful is **de-duplication / MMR** (edge 5.4): near-identical shared-factsheet chunks and Groww navigation boilerplate crowd the top-K, so apply MMR or drop same-text/same-source duplicates to widen context coverage.

**Retrieval settings:** top-K = 5 (`config.TOP_K`); cosine space; scheme filter when detected; recalibrated similarity floor. **Context assembly:** concatenate the surviving top chunks and carry each chunk's `{source_url, scheme, doc_type, last_updated}` so Phase 6 can cite exactly one source and stamp the freshness footer. **Input hygiene:** reject empty/whitespace queries (edge 5.6) and truncate over-long queries before embedding (edge 5.7).

**Deliverable:** Working retriever module.

**Acceptance Criteria:** For a known factual question (e.g., "expense ratio of HDFC Large Cap Fund"), the correct chunk appears in top-K. *(Met: all top-5 now correctly scheme-scoped for Large/Small/Mid Cap queries — the prior cross-scheme leakage is fixed; "capital of France" → 0 results via the 0.55 floor; "SBI Bluechip" → flagged out-of-scope; empty query → 0 results.)*

**Dependencies:** Phase 3

---

## Phase 5 — Intent Classification & Refusal Handling

**Goal:** Separate factual queries from advisory ones and refuse the latter.

**Tasks:**
- [x] Implement `rag/classifier.py`:
  - Rule-based pattern matching ("should I", "which is better", "good fund", "returns")
  - Optional LLM fallback classification for ambiguous queries
- [x] Implement refusal templates in `rag/prompts.py`
- [x] Refusal response includes polite wording + AMFI/SEBI educational link + disclaimer

#### Classification & Refusal Strategy (as implemented)

The classifier is the **first of three compliance layers** (classifier → system prompt → formatter guard) that jointly guarantee no advice leaks (edge 9.1). It runs *before* retrieval/LLM so advisory queries never spend Groq quota (ties into the Phase 6 rate-limit budget). `classify(query)` returns `"FACTUAL"`/`"ADVISORY"`; `classify_intent(query)` returns a richer `Intent(label, category, reason, via)` used to pick a tailored refusal.

1. **Normalize before matching (edge 4.5).** Lower-case, collapse 3+ char runs (`"gooood" → "good"`), and expand common SMS-speak/typos (`"shud i invst" → "should i invest"`) so obfuscated advisory intent can't slip past the rules. *(Verified: "shud i invst…" → ADVISORY.)*
2. **Rule-based patterns across five advisory categories** — `recommendation`, `comparison`, `prediction`, `performance`, `injection`. **Any** match ⇒ ADVISORY, so a mixed *"what's the expense ratio, and should I invest?"* is refused because an advisory intent is present (edge 4.2/4.3/4.4). Performance/return-figure queries are refused too (factsheet link only, edge 6.10/9.5), and prompt-injection phrasings are caught (edge 4.8). Romanized-Hindi cues (`chahiye`, `lena chahiye`, …) refuse advisory intent regardless of language (edge 4.7).
3. **Optional LLM fallback for genuinely ambiguous queries (edge 4.6).** When no full pattern matches but a lone advisory-ish keyword is present (e.g. a stray `buy`/`return`), a tiny Groq call (`max_tokens=1`) adjudicates to avoid over-refusal. It's best-effort/offline-safe — if no key or the call fails, we default to FACTUAL and let the Phase 6 formatter advisory-guard be the safety net. *(Verified: "What is the minimum purchase amount?" → FACTUAL, not over-refused.)*
4. **Refusal always carries the educational link + disclaimer (edge 7.5).** `prompts.refusal_message(category)` prepends a one-line, category-specific lead-in (e.g. *"I can't compare schemes or say which one is better."*) onto the shared `REFUSAL_TEMPLATE`, which always ends with the AMFI/SEBI link and the facts-only disclaimer.

**Deliverable:** Classifier + refusal handler.

**Acceptance Criteria:** Advisory test queries are refused; factual queries pass through; refusal includes an educational link and disclaimer. *(Met: all recommendation/comparison/prediction/performance/injection/obfuscated/Hindi samples → ADVISORY; the three canonical factual questions + "minimum purchase amount" → FACTUAL; every refusal ends with the AMFI link + disclaimer.)*

**Dependencies:** Phase 0 (can be built in parallel with 3–4)

---

## Phase 6 — LLM Response Generation & Formatting

**Goal:** Produce grounded, compliant, source-cited answers using Groq.

**Tasks:**
- [x] Set up Groq client in `rag/llm.py` using the `groq` Python SDK (lazy-initialised, called from `rag/pipeline.py`)
  - Primary model: `llama-3.3-70b-versatile`
  - Fallback model: `llama3-8b-8192` (lighter/cheaper; also the free-tier degrade target when the 70B daily quota is exhausted)
- [x] Define strict system prompt in `rag/prompts.py` (context-only, ≤3 sentences, one citation, footer) + user-prompt builder + busy/error/scope templates
- [x] Implement `rag/pipeline.py` — orchestrate: classify → retrieve → assemble → Groq LLM → format
- [x] Enforce Groq free-tier rate limits for `llama-3.3-70b-versatile` (`rag/ratelimit.py` sliding-window limiter + `rag/llm.py` gate/backoff)
- [x] Implement `rag/formatter.py`:
  - Enforce ≤3 sentences
  - Inject exactly one source URL from metadata
  - Append `"Last updated from sources: <date>"`
  - Validation guard against advisory language
- [x] Handle "not in corpus" fallback response

#### Groq Rate-Limit Handling (free tier — `llama-3.3-70b-versatile`)

The primary model is metered on **four independent buckets** that must all be respected; the first one to run dry throttles the request:

| Limit | Value | Design implication |
|-------|:-----:|--------------------|
| Requests / min (RPM) | **30** | 1 LLM call per user turn → cap concurrent/sequential turns at ≤30/min. |
| Requests / day (RPD) | **1,000** | Hard daily ceiling on answered questions; refusals/out-of-corpus replies that skip the LLM don't count. |
| Tokens / min (TPM) | **12,000** | `prompt + max_tokens` per call must stay well under this; ~12K/min ≈ only a handful of full-context calls per minute. |
| Tokens / day (TPD) | **100,000** | Total prompt+completion budget/day → **the binding constraint**; forces a tight per-call token budget. |

**Strategy to stay within limits:**

1. **Tight per-call token budget.** The dominant cost is retrieved context. Cap the assembled context (e.g. trim to top-3 chunks / ~1,500–2,000 prompt tokens) and keep `LLM_MAX_TOKENS` small (≤256 is enough for a ≤3-sentence answer). Budget ≈ system + context + query + completion ≤ ~2.5K tokens/call, so a call comfortably fits under TPM and yields ~40 calls/day against TPD before degrade.
2. **Pre-count tokens and gate before sending.** Estimate prompt tokens (tokenizer or `len/4` heuristic) and reject/trim any request whose `prompt + max_tokens` would exceed the per-minute TPM headroom, rather than letting Groq 429.
3. **Client-side throttle (token-bucket / sliding window).** Track rolling RPM, TPM, RPD, TPD counters in `rag/pipeline.py`. If a bucket is exhausted, either queue with a short wait (for RPM/TPM which refill each minute) or fail fast with a friendly "capacity reached, try again shortly" message (for RPD/TPD).
4. **Retry with exponential backoff + jitter on HTTP 429.** Honour Groq's `Retry-After` / `x-ratelimit-reset-*` response headers; cap retries (e.g. 3) so a throttled call degrades gracefully instead of hanging the UI.
5. **Avoid spending LLM budget on non-answers.** Route advisory refusals (Phase 5) and out-of-corpus / out-of-scope cases (Phase 4 similarity floor + scope detection) to **templated responses that never call Groq** — this preserves the 1,000 RPD / 100K TPD budget for genuine factual answers.
6. **Response cache.** Cache normalized-query → formatted-answer (in-memory / on-disk) so repeated or example questions are served without a new Groq call, protecting all four buckets.
7. **Graceful degrade + config surface.** On sustained 429 or exhausted 70B daily quota, fall back to `llama3-8b-8192`. Expose all limits (`GROQ_RPM`, `GROQ_RPD`, `GROQ_TPM`, `GROQ_TPD`) and the context/`max_tokens` caps in `config.py` so they can be raised for a paid tier without code changes.

#### Module Layout (as implemented)

Phase 6 is split into four cohesive modules rather than one monolith, so the rate-limit logic and the compliance guards are independently testable:

- **`rag/ratelimit.py`** — a thread-safe sliding-window `RateLimiter` over all four buckets. `check(est_tokens)` is side-effect-free and returns a `RateDecision` (`allowed`, `scope`, `retry_after`, `is_daily`); `reserve()` records an **upper-bound** cost (`prompt + max_tokens`) so counters never undercount. A process-wide singleton (`get_limiter`) means every session shares one free-tier budget.
- **`rag/llm.py`** — `GroqLLM` wraps the SDK: a **pre-flight gate** (`_gate`) that waits on transient per-minute exhaustion but fails fast (`CapacityError`) on daily exhaustion; **429/5xx retry** with `Retry-After`-aware exponential backoff + jitter capped at `GROQ_MAX_RETRIES`; and **model degrade** from the 70B to `llama3-8b-8192`. A missing key / dead backend raises `LLMUnavailable` — it never fabricates an ungrounded answer (edge 6.1/6.3). `estimate_tokens` uses a ~4-chars/token heuristic (no tokenizer dependency).
- **`rag/formatter.py`** — the third compliance layer: `enforce_sentences` (≤3, with a decimal-safe splitter so `1.25%` / `Rs. 500` aren't split), `pick_source` + `is_allowed_url` (inject exactly one **allowlisted** citation, never a model-invented URL — edge 6.6/7.1/9.4), a non-empty date footer (edge 7.2), `contains_advisory` (edge 6.7), and `signals_not_in_corpus` (edge 6.11).
- **`rag/pipeline.py`** — `answer(query)` orchestration with an LRU **response cache** (normalized-query → formatted answer). Only genuine factual questions reach Groq; empty/advisory/out-of-scope/cache-hit/out-of-corpus branches all short-circuit **before** any LLM call, preserving the RPD/TPD budget. The classifier's ambiguous-query LLM fallback (Phase 5) was rerouted through this same metered wrapper so *every* Groq call counts against the limiter.

**Deliverable:** End-to-end RAG pipeline returning formatted answers, with Groq rate-limit guards (throttle + backoff + non-LLM refusals + cache) that keep it inside the free-tier RPM/RPD/TPM/TPD limits.

**Acceptance Criteria:** Responses are ≤3 sentences, carry exactly one citation + date footer, and contain no advisory phrasing; a burst of >30 requests/min or a 429 is handled without a crash (throttled/backed-off/degraded), and refusal/out-of-corpus replies consume no Groq quota. *(Met — offline verification: formatter trims a 4-sentence answer to 3 with a single allowlisted citation + footer and rejects a non-allowlisted URL; the limiter enforces RPM/TPM and fail-fast RPD; the LLM gate raises `CapacityError` on an exhausted daily bucket and `LLMUnavailable` with no key; pipeline advisory/empty/out-of-scope branches return templated replies with zero Groq calls.)*

> **Live-path note:** end-to-end factual answers require a valid `GROQ_API_KEY` and a built vector index (`vectorstore/index/`); those paths were validated structurally but not exercised against the live API to avoid spending quota.

**Dependencies:** Phases 4, 5

---

## Phase 7 — User Interface

**Goal:** Deliver a minimal, compliant chat interface, styled to the **Stitch-generated design system** (`stitch_mutual_fund_faq_assistant/`).

**Tasks:**
- [x] Implement `ui/app.py` (Streamlit):
  - Welcome message
  - 3 example clickable questions
  - Chat input + response display (answer, source link, footer)
  - Persistent disclaimer banner: "Facts-only. No investment advice."
- [x] Wire UI to `rag/pipeline.py`
- [x] **Adopt the Stitch design system** (from `stitch_mutual_fund_faq_assistant/stewardship_interface/DESIGN.md`) — a `.streamlit/config.toml` `[theme]` (Inter base, Trustworthy-Blue `primaryColor`, Soft-Neutral-Gray background, White surfaces) plus a scoped CSS block injected once via `st.markdown(..., unsafe_allow_html=True)`:
  - Palette: Deep Navy `#1E2A54` (authority/user bubble), Trustworthy Blue `#2F6BFF` (primary action/focus), Soft Neutral Gray `#F5F7FA` canvas, White `#FFFFFF` surfaces, Verified Green `#1B8A5A` (source chips), Warning Amber `#B7791F` on `#FEF3C7` (disclaimers/warnings)
  - Typography: **Inter** (loaded via Google Fonts `@import`) across the app, semi-bold tight-tracked headlines, 1.5× body line-height, small tracked-out uppercase labels for metadata
  - Shape/elevation: 8–12px rounded surfaces, pill (`full`) chips, 1px `#E2E8F0` borders on cards, soft `0 4px 12px rgba(30,42,84,0.05)` shadow on chat bubbles
- [x] **Build the 5 Stitch screen states** (see mapping table below) as styled render paths driven by the pipeline response type (`_classify_state` keys off the stable `rag/prompts.py` template leads), not new logic
- [x] Wire the sidebar to the Stitch **Stewardship** layout: MF Assistant identity block, "In scope" scheme list, disclaimer card, "Corpus last updated" line, Clear Chat

#### Stitch Screen → Pipeline State Mapping

The five Google-Stitch screens each correspond to a response branch the pipeline already emits, so the UI only needs to **style** each branch — no new backend behaviour:

| # | Stitch screen (`code.html`) | Pipeline / app trigger | Styling treatment |
|---|-----------------------------|------------------------|-------------------|
| 1 | `1._welcome_state` | first load, empty transcript | Welcome card + scope note + 3 pill suggestion chips over the input |
| 2 | `2._factual_answer` | grounded factual answer | White assistant bubble (navy text), green **Source: `<domain>`** chip + "Last updated from sources: `<date>`" footer |
| 3 | `3._refusal_state` | advisory query refused (Phase 5) | White bubble, 2px amber left-border, "Regulatory Notice", AMFI/SEBI educational link button + disclaimer |
| 4 | `4._out_of_scope_state` | out-of-scope AMC/"not in corpus" (Phase 4/6) | Subtle gray italic bubble, "outside my scope" scope-limitation message |
| 5 | `5._configuration_warning_state` | missing `GROQ_API_KEY` / absent `vectorstore/index/` | Dismissible amber warning banner above the transcript (edge 10.1 / 10.2) |

The user turn is a right-aligned Deep Navy bubble with white text in every screen; the persistent amber disclaimer banner and the 280px sidebar are shared chrome across all five.

#### UI Notes (as implemented)

- **Chat-native layout.** Uses `st.chat_message` + `st.chat_input` with the transcript held in `st.session_state.messages`. The welcome message and the three clickable example buttons show only before the first turn (then collapse to keep the chat clean); a **Clear chat** button in the sidebar resets the session.
- **Single source of truth for output.** The pipeline already returns the fully-formatted string (answer body + one `Source:` link + `Last updated` footer, or a refusal/scope/busy/error template), so the UI just renders it — the styling classifies which of the 5 screen states to apply from the returned text/response type, but does **not** re-format the answer.
- **Persistent disclaimer.** `st.info(DISCLAIMER)` is rendered on every run (top banner) and also pinned in the sidebar, so "Facts-only. No investment advice." is always visible — mapped to the Stitch amber `#FEF3C7` / `#B7791F` disclaimer banner that is fixed across all five screens.
- **Graceful startup + errors.** Missing `GROQ_API_KEY` or an absent `vectorstore/index/` surface as friendly `st.warning` banners rather than failures (edge 10.1 / 10.2) — this is the Stitch **configuration-warning** screen (screen 5); every `pipeline.answer` call is wrapped so an unexpected exception shows a safe message and is logged server-side, never dumped to the user (edge 8.6).
- **No PII surface.** The UI has no login and no personal-data fields; queries live only in transient session state and are never persisted (edge 8.4 / 9.3). `st.markdown` renders with HTML escaping on, so echoed input can't inject markup (edge 8.7). The Stitch mockups' decorative `attach_file` / "Sync across devices" affordances are intentionally **dropped** to preserve the no-upload, stateless, PII-free surface.

#### Styling Approach (Streamlit ≠ static HTML)

The Stitch output is Tailwind-CDN HTML; Streamlit renders its own widget DOM, so the screens are **adapted, not copied**:
- The design tokens (colors/type/spacing from `DESIGN.md`) drive a `.streamlit/config.toml` `[theme]` (base font Inter, `primaryColor` = Trustworthy Blue, background = Soft Neutral Gray) plus a small scoped CSS block injected once at startup for chat-bubble corners, source chips, and the amber banner.
- Chat bubbles use `st.chat_message` containers with per-role CSS (user = navy/right, assistant = white/left with the state-specific border) rather than the raw Tailwind `<div>`s.
- Material Symbols icons from the mockups are substituted with Streamlit-native equivalents (`st.info`/`st.warning` icons, unicode/emoji, or inline SVG) to avoid an external icon-font dependency on Community Cloud.

**Deliverable:** Running Streamlit app that renders all five Stitch screen states with the Stitch design system applied.

**Acceptance Criteria:** `streamlit run ui/app.py` launches; example questions return valid answers; disclaimer always visible; each of the five pipeline branches (welcome / factual / refusal / out-of-scope / config-warning) renders in its corresponding Stitch style. *(Met: the styled app boots headless — Uvicorn serves HTTP 200 and `/_stcore/health` returns `ok`; module imports cleanly; the Stitch theme (`.streamlit/config.toml` + injected CSS) and all five screen states are wired, with `_classify_state` verified to route factual/refusal/scope/not-in-corpus/busy/empty responses to the correct Stitch treatment; disclaimer banner renders every run. Live answer quality depends on the built index + `GROQ_API_KEY`.)*

**Dependencies:** Phase 6, and the Stitch design assets in `stitch_mutual_fund_faq_assistant/`

---

## Phase 8 — Compliance, Testing & Evaluation

**Goal:** Validate accuracy, compliance, and robustness.

**Tasks:**
- [x] Build a test query set (factual + advisory + out-of-corpus) — `tests/golden_set.py` (25 factual + 30 advisory + 5 performance + 10 out-of-corpus/scope + 10 adversarial + 5 PII = **85 cases**, mirrors eval.md's release-gate golden set)
- [x] Verify each Success Criterion from the problem statement (via `tests/test_compliance.py` + `tests/evaluate.py`):
  - Accurate factual retrieval
  - Strict facts-only adherence
  - Consistent valid citations
  - Proper refusal of advisory queries
- [x] PII safety check — `tests/pii_scanner.py` (PAN/Aadhaar/account/OTP/email/phone) + UI "no PII field" audit + no-PII-echo / no-persistence checks
- [x] Domain allowlist check — audit of `data/source_urls.json` + `data/metadata.json` (0 off-allowlist)
- [x] Log evaluation results in a short report — `reports/phase8_eval_report.md` (generated by `python -m tests.evaluate`)

#### Test Suite & Evaluation (as implemented)

Two complementary layers, both runnable without pytest (each has a standalone runner) but also `pytest`-collectable:

1. **`tests/test_compliance.py` — deterministic, offline, no index/LLM.** 17 checks over the safety-critical invariants: advisory recall (advisory/performance/adversarial/PII all classified `ADVISORY`), factual over-refusal ≤10%, refusal completeness (disclaimer + edu link on every refusal), formatter guarantees (≤3 sentences, exactly one **allowlisted** citation, model-invented URLs stripped, non-empty footer), corpus source-domain allowlist audit, PII scanner detection, and the "no PII input field" UI audit. **17/17 pass.**
2. **`tests/evaluate.py` — end-to-end harness over the real pipeline → `reports/phase8_eval_report.md`.** Runs all 85 golden cases through `rag.pipeline.answer` and scores the eval.md release-gate metrics. Defaults to **offline** (clears `GROQ_API_KEY` for the run → **zero quota spent**; advisory/PII/performance refused pre-LLM, factual/out-of-corpus exercise retrieval + graceful degrade); `--live` adds citation/footer/sentence compliance on generated answers. It also runs a **file-system persistence audit** (no new files written → stateless, no PII persisted).

**Findings & fixes (this phase caught two real bugs):** (a) the classifier missed several advisory phrasings ("should my", "which HDFC scheme", "top performing", "going to rise", "how has X performed", "what to buy", "is X good") and initially over-refused three factual phrasings — both were fixed and re-verified; (b) `detect_out_of_scope` pulled "Nippon India **Small Cap**" in-scope via the category alias, which would have answered a competitor query with HDFC data — fixed so a named competitor AMC (without "HDFC") is always out-of-scope (edge 5.5).

**Deliverable:** Test suite + evaluation report.

**Acceptance Criteria:** All success criteria pass; zero advisory leakage; all responses cite an allowlisted domain. *(Met — offline run: all **hard gates = 0** (advisory leakage, advisory-language leakage, PII echo, PII/query persistence, off-allowlist citations); advisory/performance/adversarial/PII (50 cases) all refused; out-of-corpus fallback correct; corpus 100% allowlisted; stateless (no files written). The three answer-quality metrics — citation/footer/sentence on generated text — require `python -m tests.evaluate --live` (spends ~25–30 Groq calls) and are marked SKIP in the offline report.)*

**Dependencies:** Phase 7

---

## Phase 9 — Documentation & Delivery

**Goal:** Finalize deliverables for handoff.

**Tasks:**
- [x] Write `README.md`: setup instructions, selected AMC & schemes, RAG architecture overview, known limitations
- [x] Include disclaimer snippet
- [x] Document re-ingestion/refresh procedure
- [x] (Optional) Deployment guide (Streamlit Community Cloud / Docker)

#### Documentation (as delivered)

The rewritten [`README.md`](./README.md) is a self-contained end-to-end guide:
prerequisites (Python 3.10+, Groq key, Windows VC++ note), venv + `pip install` +
`.env` setup, a **Quick Start** that runs the shipped corpus/index, a full
**build-from-scratch** offline pipeline (`ingestion.scraper` → `ingestion.chunker`
→ `vectorstore.indexer`), the **refresh procedure** (the Phase 10 automated
GitHub Actions scheduler / `python -m scheduler.refresh` **and** the manual
`--reindex` fallback), the **Phase 8 test/eval** commands (offline + `--live`), a
configuration table mirroring `config.py`, an accurate project-structure tree (now
incl. `scheduler/` + `.github/workflows/`), the compliance/safety summary, known
limitations, a **deployment guide** (Streamlit Community Cloud + a Docker
`Dockerfile`/run, with the scheduler wired in to keep the deployed index fresh),
and the facts-only disclaimer snippet. Model names and module layout were corrected
to match the built system (`llama-3.3-70b-versatile` primary / `llama3-8b-8192`
fallback; `rag/llm.py` + `rag/ratelimit.py` + `tests/`).

> **Phase 10 follow-up:** the refresh section, known-limitations note, project tree,
> and deployment guide were updated once Phase 10 landed so the docs describe the
> **implemented** GitHub Actions scheduler rather than a planned one.

**Deliverable:** Complete documentation set.

**Acceptance Criteria:** A new developer can set up and run the project end-to-end using only the README. *(Met: README covers install → configure → (quick start on shipped index | build from scratch) → run UI → test/eval → deploy, with correct entrypoints verified against the codebase.)*

**Dependencies:** Phase 8

---

## Phase 10 — Scheduled Data Refresh (Ingestion Scheduler)

**Goal:** Keep the served corpus current by automatically re-running the offline pipeline (scrape → parse → chunk → embed → index) on a daily cadence, so the assistant always answers from the latest official documents (aligns with the "static corpus" limitation in Architecture §9 and the freshness footer requirement, edge 9.6).

**Tasks:**
- [x] Implement a scheduler orchestrator (`scheduler/refresh.py`) exposing a single `run_refresh()` entrypoint (runnable as `python -m scheduler.refresh`, with `--force` / `--dry-run`) that runs the offline pipeline **in order** and produces a run report:
  - `ingestion/scraper.py` → refresh raw docs (respect allowlist + robots, backoff on 429, edge 1.2/1.5/1.8)
  - `ingestion/parser.py` + `ingestion/chunker.py` → re-parse and re-chunk
  - `vectorstore/indexer.py --reindex` → rebuild the index
- [x] **Scheduling mechanism: a scheduled GitHub Actions workflow** (`.github/workflows/refresh.yml`) — the CI runner is the scheduler, so no always-on worker/host is needed:
  - `on.schedule` **cron trigger** `0 5 * * *` (**05:00 UTC = 10:30 AM IST** daily; note GitHub cron is **UTC-only**, so the IST time is encoded as its UTC equivalent) **plus** `workflow_dispatch` (with a `force_reindex` input) for manual/one-off runs
  - Job steps: `actions/checkout` → `actions/setup-python` (3.12) → `pip install -r requirements.txt` → `playwright install --with-deps chromium` (JS pages) → `python -m scheduler.refresh`
  - `GROQ_API_KEY` injected via **GitHub repo secrets**, never committed; `concurrency: {group, cancel-in-progress: false}` so overlapping scheduled/dispatch runs queue rather than collide (CI-level single-flight)
  - **Persist the rebuilt index** so the live app can consume it — the runner is ephemeral, so on a real change (status `ok`) the workflow **commits `vectorstore/index/` + `data/metadata.json` + `data/last_refresh.json` back to the deployed branch**, which **Streamlit Community Cloud watches and auto-reboots on** (git is the deploy channel; requires `permissions: contents: write`). The rebuilt index is also uploaded as a downloadable **artifact** for inspection. A no-change run commits nothing.
  - `timeout-minutes: 45` guard + failure notification (non-zero exit ⇒ GitHub's standard failed-run notification; status artifact uploaded on failure)
- [x] **Change detection / idempotency** — a corpus fingerprint (sorted per-document `content_sha256` from `data/metadata.json`) is compared before/after the scrape; when unchanged and an index already exists, the chunk+embed+index stages are skipped as a fast no-op unless `--force` (edge 1.6).
- [x] **Atomic hot-swap** — index rebuild goes through `vectorstore.indexer` (builds into a temp dir, swaps on success, keeps a `.bak`); on failure the live index is untouched, and a timestamped snapshot is kept under `vectorstore/backups/` for rollback (edge 3.4 / 3.7).
- [x] **Single-flight concurrency guard** — the workflow `concurrency` group prevents overlapping runs at the CI level; `scheduler/refresh.py` additionally takes an `O_EXCL` lockfile (`.refresh.lock`, auto-breaks a stale lock) so a manual local run can't collide.
- [x] **Failure handling & minimum-corpus gate** — the scraper retries transient failures; a scrape yielding `< MIN_CORPUS_DOCS` OK docs aborts **before** touching the index (previous index stays served), the error is recorded to the status file, and the process exits non-zero so the Actions run is marked failed and notifies (edge 1.9).
- [x] **Freshness surfacing** — every run records `last_refresh` (status + timestamp + doc/chunk/vector counts) to `data/last_refresh.json`, published alongside the index and shown in the Streamlit sidebar ("Corpus last updated: <date>"), reconciled with per-chunk `last_updated` (edge 9.6 / 10.6).
- [x] **Config surface** — added `REFRESH_ENABLED`, `MIN_CORPUS_DOCS`, `INDEX_BACKUP_RETENTION`, `LAST_REFRESH_FILE`, and lock knobs to `config.py`; the cadence stays in the workflow's `on.schedule` cron (single source of truth for timing) with the UTC note documented there. No new runtime dependency (scheduling is the CI runner; `refresh.py` is stdlib-only).

#### Module Layout (as implemented)

- **`scheduler/refresh.py`** — the orchestrator. `run_refresh(force, dry_run)` runs scrape → (change-detect) → snapshot → chunk → reindex and returns a run report; `main()` wraps it with the lock, status-file write, and exit code. Helpers: `_corpus_fingerprint()` (change detection over `content_sha256`), `_single_flight_lock()` (`O_EXCL` lockfile + stale-lock break), `_snapshot_index()` (timestamped backup + retention prune), and `write_status()` / `read_status()` (freshness). Heavy imports (scraper/chunker/indexer → torch/chroma) are deferred so `--help` / `--dry-run` stay instant. Exit code is 0 on `ok`/`no-change`, non-zero on any hard failure.
- **`.github/workflows/refresh.yml`** — the scheduler itself: `schedule` cron `0 5 * * *` (10:30 IST) + `workflow_dispatch`, `concurrency` single-flight, `permissions: contents: write`, checkout → setup-python 3.12 → install → playwright → `python -m scheduler.refresh`. On a real change it **commits the rebuilt index back to the deployed branch** (Streamlit Community Cloud auto-reboots on the new commit), uploads an inspection artifact, and uploads the status file on failure. The commit is gated on status `ok`, so no-change runs push nothing; the workflow triggers only on schedule/dispatch (never on push), so committing back can't re-trigger it.
- **`config.py`** — `REFRESH_ENABLED`, `MIN_CORPUS_DOCS`, `INDEX_BACKUP_RETENTION`, `INDEX_BACKUPS_DIR`, `LAST_REFRESH_FILE`, `REFRESH_LOCK_FILE`, `REFRESH_LOCK_STALE_SECONDS`.
- **`ui/app.py`** — sidebar reads `data/last_refresh.json` and shows "Corpus last updated: <date>".

**Deliverable:** A scheduled GitHub Actions workflow that refreshes the corpus daily (plus manual `workflow_dispatch`), runs `scheduler.refresh` end-to-end, atomically rebuilds the index, publishes it as an artifact for the app to consume, and records a last-refresh status — with change detection, a CI concurrency guard, and secret-based key handling.

**Acceptance Criteria:** The workflow triggers on its cron schedule and via manual dispatch, executes scrape→parse→chunk→index end-to-end on the runner; on success the index is rebuilt atomically, published as an artifact, and `last_refresh`/`last_updated` advance; a failed or empty run exits non-zero and leaves the previously published index intact; overlapping scheduled/dispatch runs do not execute concurrently (blocked by the `concurrency` group). *(Met — offline verification: `--dry-run` computes the corpus fingerprint, confirms the index is present, and reports the min-corpus gate; the fingerprint/status round-trips through `data/last_refresh.json`; the `O_EXCL` lock is acquired, a second acquisition is correctly blocked as an overlap, and the lock is released on exit. The full live scrape→reindex and the artifact publish run on the Actions runner / on demand.)*

**Dependencies:** Phases 1–3 (offline pipeline + `indexer --reindex`). Independent of Phases 4–9; can run in parallel with them once Phase 3 exists.

---

## Dependency Flow

```
Phase 0 ──▶ Phase 1 ──▶ Phase 2 ──▶ Phase 3 ──▶ Phase 4 ──┐
                                    │                     ├──▶ Phase 6 ──▶ Phase 7 ──▶ Phase 8 ──▶ Phase 9
Phase 0 ──▶ Phase 5 ────────────────┼─────────────────────┘
                                    │
                                    └──▶ Phase 10 (GitHub Actions cron) re-runs Phases 1→2→3 daily
```

> Phase 5 (Classifier/Refusal) can be developed in parallel with Phases 3–4 since it does not depend on the vector store.
> Phase 10 (Scheduler) depends only on the offline pipeline (Phases 1–3) and periodically re-invokes it; it is independent of the online phases (4–9).

---

## Suggested Timeline (Indicative)

| Phase | Effort Estimate |
|-------|-----------------|
| 0. Setup | 0.5 day |
| 1. Corpus Collection | 1–2 days |
| 2. Processing & Chunking | 1 day |
| 3. Embedding & Indexing | 0.5 day |
| 4. Retrieval | 0.5 day |
| 5. Classifier & Refusal | 1 day |
| 6. LLM Generation | 1–2 days |
| 7. UI | 1 day |
| 8. Testing & Compliance | 1 day |
| 9. Documentation | 0.5 day |
| 10. Scheduled Data Refresh | 0.5–1 day |
| **Total** | **~9–11 days** |

---

> **Disclaimer:** Facts-only. No investment advice.
