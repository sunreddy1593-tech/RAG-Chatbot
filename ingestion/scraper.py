"""URL fetcher for the official-source corpus (Phase 1).

Reads the curated allowlist in ``data/source_urls.json`` and downloads each
document into ``data/raw/{factsheets,kim,sid,faq}/`` while recording provenance
in ``data/metadata.json``.

Design notes (see edge-case.md):
- Allowlist is enforced per-URL; any off-allowlist domain is hard-rejected.
- Network calls retry with exponential backoff and are skipped (logged) on
  final failure, so one bad URL never aborts the whole run.
- HTML is fetched with requests + BeautifulSoup; JS-rendered pages (Groww,
  hdfcfund.com SPA) use Playwright. ``render: auto`` tries static first and
  escalates to Playwright when the static body is too small.
- PDFs are downloaded as bytes and validated by their ``%PDF`` magic header.

Run with:  python -m ingestion.scraper
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

import config

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
MIN_HTML_TEXT_CHARS = 600  # below this, a static fetch is considered "thin"

# doc_type -> raw subdirectory
_SUBDIR_BY_DOCTYPE = {
    "factsheet": config.RAW_FACTSHEETS_DIR,
    "kim": config.RAW_KIM_DIR,
    "sid": config.RAW_SID_DIR,
}
_DEFAULT_SUBDIR = config.RAW_FAQ_DIR  # scheme_page, amc_scheme_page, faq, guidance


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _domain(url: str) -> str:
    return (urlparse(url).hostname or "").lower().lstrip("www.")


def is_allowed(url: str) -> bool:
    """True only if the URL's host matches an allowlisted domain."""
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in config.ALLOWED_DOMAINS)


def _slugify(value: str) -> str:
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[\s_-]+", "-", value) or "doc"


def _target_dir(doc_type: str) -> Path:
    return _SUBDIR_BY_DOCTYPE.get(doc_type, _DEFAULT_SUBDIR)


def _filename(source: dict, ext: str) -> str:
    scheme_part = _slugify(source.get("scheme", "shared"))
    url_tail = _slugify(Path(urlparse(source["url"]).path).stem or _domain(source["url"]))
    return f"{scheme_part}__{source['doc_type']}__{url_tail}.{ext}"


def _backoff_sleep(attempt: int) -> None:
    time.sleep(min(2 ** attempt, 8))


# --------------------------------------------------------------------------- #
# Fetchers
# --------------------------------------------------------------------------- #
def _http_get(url: str, *, stream: bool = False) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
                timeout=REQUEST_TIMEOUT,
                stream=stream,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                last_exc = RuntimeError(f"HTTP {resp.status_code}")
                _backoff_sleep(attempt)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:  # network / timeout / 4xx
            last_exc = exc
            _backoff_sleep(attempt)
    raise RuntimeError(f"GET failed after {MAX_RETRIES} attempts: {last_exc}")


def fetch_static_html(url: str) -> str:
    """Return raw HTML via requests."""
    return _http_get(url).text


def fetch_js_html(url: str) -> str:
    """Return rendered HTML via Playwright (Chromium)."""
    from playwright.sync_api import sync_playwright  # lazy import

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(user_agent=USER_AGENT)
            page.goto(url, wait_until="networkidle", timeout=60_000)
            page.wait_for_timeout(1500)  # let late content settle
            return page.content()
        finally:
            browser.close()


def _html_text_len(html: str) -> int:
    return len(BeautifulSoup(html, "html.parser").get_text(" ", strip=True))


def download_pdf(url: str, dest: Path) -> int:
    """Download a PDF, validating the %PDF header. Returns bytes written."""
    resp = _http_get(url, stream=True)
    content = resp.content
    if not content[:4] == b"%PDF":
        raise RuntimeError("downloaded file is not a valid PDF (missing %PDF header)")
    dest.write_bytes(content)
    return len(content)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _save_html(source: dict, dest_dir: Path) -> tuple[Path, int]:
    render = source.get("render", "auto")
    url = source["url"]

    if render == "js":
        html = fetch_js_html(url)
    elif render == "static":
        html = fetch_static_html(url)
    else:  # auto: try static, escalate to Playwright on thin content OR error
        try:
            html = fetch_static_html(url)
            if _html_text_len(html) < MIN_HTML_TEXT_CHARS:
                html = fetch_js_html(url)
        except Exception:  # noqa: BLE001 - 403/WAF/etc. -> try a real browser
            html = fetch_js_html(url)

    text_len = _html_text_len(html)
    lowered = html.lower()
    if text_len < MIN_HTML_TEXT_CHARS or "access denied" in lowered:
        raise RuntimeError(
            f"content too thin / likely blocked ({text_len} chars)"
        )

    dest = dest_dir / _filename(source, "html")
    dest.write_text(html, encoding="utf-8")
    return dest, text_len


def _process_source(source: dict) -> dict:
    """Download a single source; returns its metadata record."""
    url = source["url"]
    record: dict = {
        "url": url,
        "source_domain": _domain(url),
        "scheme": source.get("scheme", "Shared"),
        "category": source.get("category", ""),
        "doc_type": source["doc_type"],
        "render": source.get("render", "auto"),
        "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_updated": date.today().isoformat(),
        "status": "pending",
    }

    if not is_allowed(url):
        record.update(status="rejected", error="domain not in allowlist")
        return record

    dest_dir = _target_dir(source["doc_type"])
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        if source.get("render") == "pdf":
            dest = dest_dir / _filename(source, "pdf")
            size = download_pdf(url, dest)
            text_len = None
        else:
            dest, text_len = _save_html(source, dest_dir)
            size = dest.stat().st_size

        record.update(
            status="ok",
            file=str(dest.relative_to(config.BASE_DIR)),
            bytes=size,
            text_chars=text_len,
            content_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        )
    except Exception as exc:  # noqa: BLE001 - resilient by design
        record.update(status="failed", error=str(exc))

    return record


def run() -> dict:
    """Execute the full ingestion and write metadata.json. Returns the report."""
    sources_doc = json.loads(config.SOURCE_URLS_FILE.read_text(encoding="utf-8"))
    sources = sources_doc["sources"]

    print(f"Loaded {len(sources)} sources from {config.SOURCE_URLS_FILE.name}\n")

    records: list[dict] = []
    for i, source in enumerate(sources, 1):
        print(f"[{i}/{len(sources)}] {source['doc_type']:16} {source['url']}")
        rec = _process_source(source)
        flag = {"ok": "  OK", "failed": "  FAIL", "rejected": "  REJECT"}.get(
            rec["status"], "  ?"
        )
        detail = rec.get("file") or rec.get("error", "")
        print(f"{flag}: {detail}\n")
        records.append(rec)

    config.METADATA_FILE.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return _report(records)


def _report(records: list[dict]) -> dict:
    ok = [r for r in records if r["status"] == "ok"]
    failed = [r for r in records if r["status"] == "failed"]
    rejected = [r for r in records if r["status"] == "rejected"]

    schemes = sorted({r["scheme"] for r in ok if not r["scheme"].startswith("Shared")})

    print("=" * 60)
    print("INGESTION REPORT")
    print("=" * 60)
    print(f"Downloaded : {len(ok)}")
    print(f"Failed     : {len(failed)}")
    print(f"Rejected   : {len(rejected)} (off-allowlist)")
    print(f"Schemes covered ({len(schemes)}): {', '.join(schemes) or 'none'}")
    print(f"Metadata   : {config.METADATA_FILE.relative_to(config.BASE_DIR)}")

    if len(ok) < 15:
        print("\nWARNING: fewer than 15 documents downloaded (target 15-25).")
    for r in failed:
        print(f"  - FAILED {r['url']}: {r.get('error')}")

    return {
        "downloaded": len(ok),
        "failed": len(failed),
        "rejected": len(rejected),
        "schemes": schemes,
    }


if __name__ == "__main__":
    report = run()
    sys.exit(0 if report["downloaded"] >= 15 else 1)
