"""PDF / HTML text extractor (Phase 2).

Turns a single raw document (downloaded in Phase 1) into clean, plain UTF-8
text ready for chunking.

Design notes (see edge-case.md §2):
- PDFs are extracted with PyMuPDF (fast, layout-aware). If PyMuPDF yields too
  little text (scanned / image-only / odd encoding), we fall back to
  ``pdfplumber``, which also recovers multi-column tables (expense-ratio /
  exit-load grids) as pipe-delimited rows so row/column relationships survive
  into a chunk (edge 2.2).
- HTML is parsed with BeautifulSoup; script/style/nav/header/footer/aside and
  other boilerplate containers are removed before extracting text (edge 2.4).
- A shared ``clean_text`` pass normalises whitespace and drops page-number-only
  lines so downstream chunks stay readable.
- Empty extraction (e.g. a scanned PDF with no text layer and no OCR available)
  returns ``""``; the caller logs and skips it (edge 2.1 / 2.5).
"""

from __future__ import annotations

import re
from pathlib import Path

# Below this many extracted characters a PDF is treated as "thin" and retried
# with pdfplumber before being given up on.
_MIN_PDF_TEXT_CHARS = 200

# Boilerplate container tags / roles to drop from HTML before text extraction.
_HTML_STRIP_TAGS = (
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "button",
    "svg",
    "iframe",
)

_PAGE_NUMBER_RE = re.compile(r"^\s*(page\s*)?\d+(\s*/\s*\d+)?\s*$", re.IGNORECASE)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t\u00a0]{2,}")


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Normalise whitespace and drop obvious boilerplate lines.

    Keeps numeric footnote markers (e.g. ``1.25%*``) intact so their meaning
    survives into the chunk (edge 2.7).
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")

    kept: list[str] = []
    for raw_line in text.split("\n"):
        line = _MULTI_SPACE_RE.sub(" ", raw_line).strip()
        if not line:
            kept.append("")
            continue
        if _PAGE_NUMBER_RE.match(line):  # standalone page number
            continue
        kept.append(line)

    text = "\n".join(kept)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def _extract_pdf_pymupdf(path: Path) -> str:
    import fitz  # PyMuPDF

    parts: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    return "\n".join(parts)


def _extract_pdf_pdfplumber(path: Path) -> str:
    import pdfplumber

    parts: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            parts.append(page_text)
            # Preserve tabular facts (expense ratio / exit load grids) as
            # pipe-delimited rows so column relationships are not lost.
            for table in page.extract_tables() or []:
                for row in table:
                    cells = [(c or "").strip() for c in row]
                    if any(cells):
                        parts.append(" | ".join(cells))
    return "\n".join(parts)


def extract_pdf(path: Path) -> str:
    """Extract text from a PDF, falling back from PyMuPDF to pdfplumber."""
    try:
        text = _extract_pdf_pymupdf(path)
    except Exception:  # noqa: BLE001 - corrupt/edge PDFs shouldn't abort the run
        text = ""

    if len(text.strip()) < _MIN_PDF_TEXT_CHARS:
        try:
            fallback = _extract_pdf_pdfplumber(path)
        except Exception:  # noqa: BLE001
            fallback = ""
        if len(fallback.strip()) > len(text.strip()):
            text = fallback

    return clean_text(text)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def extract_html(path: Path) -> str:
    """Extract readable body text from an HTML file, stripping boilerplate."""
    from bs4 import BeautifulSoup

    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(list(_HTML_STRIP_TAGS)):
        tag.decompose()

    # Prefer the semantic <main> / <article> body when present, else <body>.
    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = root.get_text(separator="\n", strip=True)
    return clean_text(text)


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def extract_text(path: str) -> str:
    """Extract clean text from a PDF or HTML file, dispatching on extension.

    Returns an empty string if nothing extractable is found (caller skips).
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf(p)
    if suffix in (".html", ".htm"):
        return extract_html(p)

    raise ValueError(f"Unsupported file type for parsing: {p.name}")
