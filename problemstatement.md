# Problem Statement: Mutual Fund FAQ Assistant (Facts-Only Q&A)

## Overview

The objective of this project is to build a **facts-only FAQ assistant** for mutual fund schemes, using **Groww** as the reference product context. The assistant will answer objective, verifiable queries related to mutual funds by retrieving information exclusively from official public sources, such as AMC (Asset Management Company) websites, AMFI, and SEBI.

The system must strictly avoid providing investment advice, opinions, or recommendations. Every response must include a single, clear source link and adhere to defined constraints around clarity, accuracy, and compliance.

---

## Objective

Design and implement a lightweight **Retrieval-Augmented Generation (RAG)**-based assistant that:

- Answers factual queries about mutual fund schemes
- Uses a curated corpus of official documents
- Provides concise, source-backed responses

### Mutual Fund Schemes in Scope

The following Groww URLs represent the schemes for which the FAQ assistant is being built:

| Scheme | URL |
|--------|-----|
| HDFC Gold ETF Fund of Fund – Direct Plan Growth | https://groww.in/mutual-funds/hdfc-gold-etf-fund-of-fund-direct-plan-growth |
| HDFC Large Cap Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| HDFC Small Cap Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| HDFC Silver ETF FoF – Direct Growth | https://groww.in/mutual-funds/hdfc-silver-etf-fof-direct-growth |
| HDFC Mid Cap Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth |

---

## Target Users

- Retail investors comparing mutual fund schemes
- Customer support and content teams handling repetitive mutual fund queries

---

## Scope of Work

### 1. Corpus Definition

**Selected AMC:** HDFC Mutual Fund

**Selected Schemes (5 schemes across diverse categories):**

| # | Scheme | Category | Groww URL |
|---|--------|----------|-----------|
| 1 | HDFC Gold ETF Fund of Fund – Direct Plan Growth | Commodity (Gold) | https://groww.in/mutual-funds/hdfc-gold-etf-fund-of-fund-direct-plan-growth |
| 2 | HDFC Large Cap Fund – Direct Growth | Large Cap Equity | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| 3 | HDFC Small Cap Fund – Direct Growth | Small Cap Equity | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| 4 | HDFC Silver ETF FoF – Direct Growth | Commodity (Silver) | https://groww.in/mutual-funds/hdfc-silver-etf-fof-direct-growth |
| 5 | HDFC Mid Cap Fund – Direct Growth | Mid Cap Equity | https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth |

**Official Public URLs to Collect (15–25 total):**

| Document Type | Source | Per Scheme |
|---------------|--------|------------|
| Scheme Factsheet (monthly) | HDFC AMC website | 1 per scheme (×5) |
| KIM (Key Information Memorandum) | HDFC AMC website | 1 per scheme (×5) |
| SID (Scheme Information Document) | HDFC AMC / SEBI | 1 per scheme (×5) |
| AMC FAQ / Help pages | hdfcfund.com | Shared (×1–2) |
| AMFI Scheme Data & Guidance | amfiindia.com | Shared (×1–2) |
| SEBI Investor Education pages | sebi.gov.in | Shared (×1–2) |
| Statement & Capital Gains Download Guide | HDFC AMC / CAMS | Shared (×1) |

### 2. FAQ Assistant Requirements

The assistant must answer **facts-only queries**, such as:

- Expense ratio of a scheme
- Exit load details
- Minimum SIP amount
- ELSS lock-in period
- Riskometer classification
- Benchmark index
- Process to download statements or capital gains reports

Each response must:

- Be limited to a **maximum of 3 sentences**
- Include **exactly one citation link**
- Include a footer: `"Last updated from sources: <date>"`

### 3. Refusal Handling

The assistant must refuse non-factual or advisory queries, such as:

- *"Should I invest in this fund?"*
- *"Which fund is better?"*

Refusal responses should:

- Be polite and clearly worded
- Reinforce the facts-only limitation
- Provide a relevant educational link (e.g., AMFI or SEBI resource)

### 4. User Interface (Minimal)

The solution should include a simple interface with:

- A welcome message
- Three example questions
- A visible disclaimer: **"Facts-only. No investment advice."**

---

## Constraints

### Data and Sources

- Use only **official public sources** (AMC, AMFI, SEBI)
- Do not use third-party blogs or aggregator websites

### Privacy and Security

Do not collect, store, or process:

- PAN or Aadhaar numbers
- Account numbers
- OTPs
- Email addresses or phone numbers

### Content Restrictions

- No investment advice or recommendations
- No performance comparisons or return calculations
- For performance-related queries, provide a link to the official factsheet only

### Transparency

- Responses must be short, factual, and verifiable
- Every answer must include a source link and last updated date

---

## Expected Deliverables

| Deliverable | Description |
|-------------|-------------|
| README Document | Setup instructions, selected AMC and schemes, architecture overview (RAG approach), known limitations |
| Disclaimer Snippet | "Facts-only. No investment advice." |

---

## Success Criteria

- Accurate retrieval of factual mutual fund information
- Strict adherence to facts-only responses
- Consistent inclusion of valid source citations
- Proper refusal of advisory queries
- Clean, minimal, and user-friendly interface

---

## Summary

The goal is to build a **trustworthy, transparent, and compliant** mutual fund FAQ assistant that prioritizes accuracy over intelligence. The system should ensure that users receive only verified, source-backed financial information, without any advisory bias or speculative content.

---

> **Disclaimer:** Facts-only. No investment advice.
