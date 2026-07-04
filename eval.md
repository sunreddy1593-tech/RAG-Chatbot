# Evaluation Plan: Mutual Fund FAQ Assistant

This document defines how each phase of [`ImplementationPlan.md`](./ImplementationPlan.md) is evaluated — the **metrics**, **methods**, **pass thresholds**, and **test data** used to verify that the phase is complete and correct.

## How to Use This Document

- Each phase has an **evaluation table** (metric → method → target).
- A phase is **DONE** only when all its metrics meet their targets.
- The final [End-to-End System Evaluation](#end-to-end-system-evaluation) gates release.

| Status | Meaning |
|--------|---------|
| Pass | Metric meets or exceeds target |
| Warn | Within 10% of target — needs review |
| Fail | Below target — phase blocked |

---

## Phase 0 — Project Setup & Scaffolding

| Metric | Method | Target |
|--------|--------|--------|
| Environment reproducibility | Fresh `pip install -r requirements.txt` in clean venv | Succeeds with zero errors |
| Directory structure correctness | Compare tree against Architecture §7 | 100% match |
| Config loads | Import `config.py`, assert all keys present | No missing config keys |
| Secret hygiene | Confirm `.env` git-ignored, only `.env.example` committed | No secrets in repo |

**Exit criteria:** Project runs a no-op smoke script without import errors.

---

## Phase 1 — Corpus Collection (Data Ingestion)

| Metric | Method | Target |
|--------|--------|--------|
| Corpus size | Count downloaded documents | ≥ 15 (target 15–25) |
| Scheme coverage | Verify docs exist per scheme | All 5 schemes represented |
| Document-type diversity | Check Factsheet / KIM / SID / FAQ present | ≥ 4 distinct types |
| Allowlist compliance | Validate every source domain | 100% from approved domains |
| Metadata completeness | Every file has a `metadata.json` entry | 100% |
| Fetch success rate | Successful / attempted URLs | ≥ 90% (rest logged) |

**Test data:** `data/source_urls.json` (curated allowlist).

**Exit criteria:** Ingestion report shows ≥15 docs, all 5 schemes, zero off-allowlist sources.

---

## Phase 2 — Document Processing & Chunking

| Metric | Method | Target |
|--------|--------|--------|
| Parse success rate | Docs parsed / docs ingested | ≥ 95% |
| Chunk metadata completeness | All 5 fields present per chunk | 100% |
| Empty/garbage chunk rate | Manual + heuristic scan of chunks | < 2% |
| Chunk size adherence | Token length distribution | ~500 tokens, ≤ 5% outliers |
| Fact integrity (spot check) | Manually verify 10 key facts survive chunking | 10/10 intact |
| Table extraction accuracy | Check expense ratio / exit load tables | Values match source |

**Test data:** 10 hand-picked facts (expense ratios, exit loads, SIP minimums, lock-ins).

**Exit criteria:** No chunk missing metadata; key facts verifiably present and correct.

---

## Phase 3 — Embedding & Vector Store Indexing

| Metric | Method | Target |
|--------|--------|--------|
| Index integrity | Vector count == chunk count | Exact match |
| Embedding dimension | Assert vector dim | 1024 (`bge-large-en-v1.5`) |
| Persistence | Reload index from disk, re-query | Identical results |
| Model consistency | Same model id for index + query | Single configured model |
| Indexing throughput | Chunks embedded per minute | Logged (baseline) |
| Smoke retrieval | 5 known queries return relevant chunks | 5/5 relevant in top-K |

**Exit criteria:** Index loads cleanly; dimension and counts verified; smoke queries relevant.

---

## Phase 4 — Retrieval Layer

| Metric | Method | Target |
|--------|--------|--------|
| Hit@K (K=5) | Gold chunk appears in top-K | ≥ 0.90 |
| MRR (Mean Reciprocal Rank) | Rank of first relevant chunk | ≥ 0.80 |
| Scheme precision | Retrieved chunk matches queried scheme | ≥ 0.95 |
| Out-of-corpus detection | Low-similarity queries flagged | ≥ 0.90 correctly flagged |
| Latency (retrieval) | Time per query | < 300 ms |

**Test data:** Labeled query→gold-chunk set (≥ 20 factual queries across 5 schemes).

**Exit criteria:** Hit@5 ≥ 0.90 and scheme precision ≥ 0.95 on the labeled set.

---

## Phase 5 — Intent Classification & Refusal Handling

| Metric | Method | Target |
|--------|--------|--------|
| Advisory recall | Advisory queries correctly refused | ≥ 0.98 (safety-critical) |
| Factual precision | Factual queries not wrongly refused | ≥ 0.90 |
| False-refusal rate | Factual queries refused in error | ≤ 0.10 |
| Refusal completeness | Refusals include disclaimer + edu link | 100% |
| Robustness | Typos/obfuscation/injection variants caught | ≥ 0.95 |

**Test data:** Balanced set — 30 advisory + 30 factual + 10 adversarial (typos, injection, mixed).

**Exit criteria:** Advisory recall ≥ 0.98 with acceptable false-refusal rate; zero advisory leakage.

---

## Phase 6 — LLM Response Generation (Groq) & Formatting

| Metric | Method | Target |
|--------|--------|--------|
| Groundedness / faithfulness | Answer supported by retrieved context (manual or LLM-judge) | ≥ 0.95 |
| Answer accuracy | Answer matches source fact | ≥ 0.90 |
| Sentence-limit adherence | Responses ≤ 3 sentences | 100% |
| Citation presence | Exactly one valid citation | 100% |
| Citation validity | URL from allowlisted domain & relevant | 100% |
| Footer presence | `Last updated from sources: <date>` present | 100% |
| Advisory-language leakage | Scan for "should/recommend/better" | 0 occurrences |
| Fallback correctness | Out-of-corpus → fallback message | 100% |
| Latency (LLM, Groq) | End-to-end generation time | < 2 s p95 |

**Test data:** 25 factual gold-Q&A pairs + 10 out-of-corpus queries.

**Exit criteria:** Groundedness ≥ 0.95, 100% citation + footer compliance, zero advisory leakage.

---

## Phase 7 — User Interface

| Metric | Method | Target |
|--------|--------|--------|
| Launch success | `streamlit run ui/app.py` | Starts without error |
| Example questions work | Click each of the 3 prompts | All return valid answers |
| Disclaimer visibility | Check across viewport sizes | Always visible |
| Error resilience | Trigger backend error | Friendly message, no stack trace |
| Input validation | Empty / very long input | Handled gracefully |
| Output sanitization | Inject markdown/HTML via query | Escaped, no injection |

**Exit criteria:** App runs end-to-end; disclaimer persistent; no raw errors leak to UI.

---

## Phase 8 — Compliance, Testing & Evaluation

| Metric | Method | Target |
|--------|--------|--------|
| Advisory leakage (system-wide) | Full advisory test suite | 0 leaks |
| Citation coverage | All factual answers cited | 100% |
| PII safety | Scan logs/storage after PII inputs | 0 PII persisted |
| Source allowlist | Audit all cited domains | 100% approved |
| No performance/return output | Run performance-style queries | 0 calculations returned |
| Freshness footer | All responses carry date | 100% |
| Success-criteria checklist | Map to problem statement | All criteria pass |

**Test data:** Consolidated regression suite (factual + advisory + adversarial + PII + out-of-corpus).

**Exit criteria:** All critical invariants pass with zero violations.

---

## Phase 9 — Documentation & Delivery

| Metric | Method | Target |
|--------|--------|--------|
| Setup reproducibility | New dev follows README only | Runs end-to-end successfully |
| Completeness | README has setup, AMC/schemes, architecture, limitations | All sections present |
| Disclaimer snippet | Present in README | Included |
| Refresh procedure | Re-ingestion steps documented | Reproducible |

**Exit criteria:** A fresh developer reaches a working app using only the README.

---

## End-to-End System Evaluation

The release gate. Runs after Phase 8 on the full regression suite.

### Golden Test Set

| Category | Count | Purpose |
|----------|-------|---------|
| Factual (in-corpus) | 25 | Accuracy, groundedness, citation |
| Advisory | 30 | Refusal correctness |
| Out-of-corpus | 10 | Fallback behavior |
| Adversarial (typo/injection/mixed) | 10 | Robustness |
| PII-containing | 5 | Privacy safety |

### Release Gate Metrics

| Metric | Target | Type |
|--------|--------|------|
| Factual accuracy | ≥ 0.90 | Quality |
| Groundedness / faithfulness | ≥ 0.95 | Quality |
| Citation compliance (1 valid URL) | 100% | Compliance |
| Footer/date compliance | 100% | Compliance |
| Advisory leakage | 0 | Safety (hard gate) |
| Advisory recall | ≥ 0.98 | Safety |
| PII persistence | 0 | Privacy (hard gate) |
| Off-allowlist citations | 0 | Compliance (hard gate) |
| End-to-end latency (p95) | < 3 s | Performance |

> **Hard gates** (advisory leakage, PII persistence, off-allowlist citations) must be **exactly zero** — any violation blocks release regardless of other scores.

---

## Evaluation Methods Reference

| Method | Description | Used In |
|--------|-------------|---------|
| Exact/fuzzy match | Compare answer value to gold fact | Phases 2, 4, 6 |
| Hit@K / MRR | Ranking quality of retrieval | Phase 4 |
| LLM-as-judge | Model scores groundedness/faithfulness | Phase 6, E2E |
| Rule-based scan | Regex for advisory terms, citations, footer | Phases 5, 6, 8 |
| Manual spot check | Human verification of sampled outputs | Phases 2, 6 |
| Confusion matrix | Classifier precision/recall | Phase 5 |
| PII scanner | Detect PAN/Aadhaar/account/OTP/email/phone in logs | Phases 8, E2E |

---

> **Disclaimer:** Facts-only. No investment advice.
