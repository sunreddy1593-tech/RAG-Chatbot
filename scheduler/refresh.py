"""Scheduled corpus-refresh orchestrator (Phase 10).

Re-runs the offline pipeline **in order** and atomically republishes the index:

    ingestion.scraper  ->  ingestion.chunker  ->  vectorstore.indexer(--reindex)

It is designed to be driven by the scheduled GitHub Actions workflow
(`.github/workflows/refresh.yml`, cron ``0 5 * * *`` = 05:00 UTC = 10:30 AM IST)
but also runs standalone:

    python -m scheduler.refresh            # refresh; skip rebuild if unchanged
    python -m scheduler.refresh --force    # always rebuild the index
    python -m scheduler.refresh --dry-run  # print what would run, touch nothing

Guarantees (see edge-case.md and ImplementationPlan.md Phase 10):

- **Single-flight** — a lockfile (``config.REFRESH_LOCK_FILE``) prevents a manual
  run from overlapping another; a stale lock (crashed run) is auto-broken. The
  Actions ``concurrency`` group is the CI-level equivalent.
- **Change detection / idempotency** — a corpus fingerprint (sorted per-document
  ``content_sha256`` from ``data/metadata.json``) is compared before/after the
  scrape. If nothing changed and an index already exists, the expensive
  chunk+embed+index stages are skipped (fast no-op) unless ``--force`` (edge 1.6).
- **Minimum-corpus gate** — if the scrape yields fewer than
  ``config.MIN_CORPUS_DOCS`` OK documents, the run aborts *before* touching the
  index, so the previous good index stays served (edge 1.9).
- **Atomic hot-swap** — index rebuild goes through ``vectorstore.indexer`` which
  builds into a temp dir and swaps on success; on failure the live index is
  untouched (edge 3.4 / 3.7). A timestamped snapshot is kept for rollback.
- **Freshness surfacing** — every run (success, no-change, or failure) records a
  status to ``config.LAST_REFRESH_FILE`` for the UI (edge 9.6 / 10.6).
- **Exit code** — 0 on success/no-change; non-zero on any hard failure so the
  Actions run is marked failed and its failure notification fires.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import config


# --------------------------------------------------------------------------- #
# Status file (freshness surfacing)
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_status(status: str, **fields) -> dict:
    """Persist the last-refresh status for the UI. Returns the record written."""
    record = {"status": status, "finished_at": _now_iso(), **fields}
    try:
        config.LAST_REFRESH_FILE.parent.mkdir(parents=True, exist_ok=True)
        config.LAST_REFRESH_FILE.write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:  # status is best-effort; never fail a run over it
        print(f"WARNING: could not write status file: {exc}")
    return record


def read_status() -> dict | None:
    """Return the last-refresh status record, or None if never run."""
    try:
        return json.loads(config.LAST_REFRESH_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# Change detection
# --------------------------------------------------------------------------- #
def _corpus_fingerprint() -> str | None:
    """Fingerprint the OK corpus from metadata.json, or None if absent.

    Combines each successfully-scraped document's ``content_sha256`` (sorted, so
    order is irrelevant) into one SHA-256. Two scrapes with identical document
    bytes produce the same fingerprint -> nothing to re-embed (edge 1.6).
    """
    if not config.METADATA_FILE.exists():
        return None
    try:
        records = json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    hashes = sorted(
        r["content_sha256"]
        for r in records
        if r.get("status") == "ok" and r.get("content_sha256")
    )
    if not hashes:
        return None
    return hashlib.sha256("\n".join(hashes).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Single-flight lock
# --------------------------------------------------------------------------- #
@contextmanager
def _single_flight_lock():
    """Exclusive lockfile guard; breaks a stale lock from a crashed run."""
    lock = config.REFRESH_LOCK_FILE
    if lock.exists():
        age = time.time() - lock.stat().st_mtime
        if age < config.REFRESH_LOCK_STALE_SECONDS:
            raise RuntimeError(
                f"another refresh appears to be running (lock held for {int(age)}s "
                f"at {lock}); aborting to avoid overlap"
            )
        print(f"Breaking stale refresh lock ({int(age)}s old).")
        lock.unlink(missing_ok=True)

    # O_EXCL makes creation atomic: if two runs race, exactly one wins.
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:  # lost the race
        raise RuntimeError("another refresh just acquired the lock") from exc
    try:
        os.write(fd, f"pid={os.getpid()} started={_now_iso()}\n".encode("utf-8"))
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Index snapshot (retention / rollback)
# --------------------------------------------------------------------------- #
def _snapshot_index() -> str | None:
    """Copy the current index to a timestamped backup; prune to retention.

    Returns the backup path (relative) or None if there was nothing to snapshot.
    """
    if config.INDEX_BACKUP_RETENTION <= 0 or not config.INDEX_DIR.exists():
        return None

    config.INDEX_BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = config.INDEX_BACKUPS_DIR / f"index_{stamp}"
    shutil.copytree(config.INDEX_DIR, dest, dirs_exist_ok=True)

    backups = sorted(
        (p for p in config.INDEX_BACKUPS_DIR.glob("index_*") if p.is_dir()),
        key=lambda p: p.name,
    )
    for old in backups[: -config.INDEX_BACKUP_RETENTION]:
        shutil.rmtree(old, ignore_errors=True)

    return str(dest.relative_to(config.BASE_DIR))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_refresh(force: bool = False, dry_run: bool = False) -> dict:
    """Run scrape -> chunk -> index end-to-end. Returns a run report.

    Raises on any hard failure (min-corpus gate, empty chunking, index mismatch)
    so the caller can exit non-zero. The previous index is left intact on failure.
    """
    started = time.monotonic()
    print("=" * 60)
    print(f"CORPUS REFRESH  (force={force}, dry_run={dry_run})  {_now_iso()}")
    print("=" * 60)

    if dry_run:
        fp = _corpus_fingerprint()
        print(f"Would scrape -> chunk -> reindex.")
        print(f"Current corpus fingerprint : {fp or '(none)'}")
        print(f"Index present              : {config.INDEX_DIR.exists()}")
        print(f"Min corpus docs gate       : {config.MIN_CORPUS_DOCS}")
        return {"status": "dry-run", "fingerprint": fp}

    # Deferred so a --dry-run / --help never imports heavy deps (torch, chroma).
    from ingestion import chunker, scraper
    from vectorstore import indexer

    fingerprint_before = _corpus_fingerprint()
    index_exists = config.INDEX_DIR.exists()

    # 1) Scrape ------------------------------------------------------------- #
    print("\n[1/3] Scraping sources ...")
    scrape_report = scraper.run()
    downloaded = scrape_report["downloaded"]

    # Minimum-corpus gate (edge 1.9): abort before touching the served index.
    if downloaded < config.MIN_CORPUS_DOCS:
        raise RuntimeError(
            f"minimum-corpus gate failed: {downloaded} documents scraped "
            f"(< {config.MIN_CORPUS_DOCS}); keeping previous index"
        )

    # 2) Change detection --------------------------------------------------- #
    fingerprint_after = _corpus_fingerprint()
    unchanged = (
        not force
        and index_exists
        and fingerprint_before is not None
        and fingerprint_after == fingerprint_before
    )
    if unchanged:
        elapsed = round(time.monotonic() - started, 1)
        print("\nCorpus unchanged since last refresh — skipping re-index (no-op).")
        return {
            "status": "no-change",
            "documents": downloaded,
            "fingerprint": fingerprint_after,
            "elapsed_seconds": elapsed,
        }

    # 3) Snapshot + chunk + reindex ---------------------------------------- #
    backup = _snapshot_index()
    if backup:
        print(f"\nPrevious index snapshotted to {backup}")

    print("\n[2/3] Parsing + chunking ...")
    chunk_report = chunker.run()
    if chunk_report["total_chunks"] <= 0:
        raise RuntimeError("chunking produced 0 chunks; keeping previous index")

    print("\n[3/3] Rebuilding index (atomic swap) ...")
    index_report = indexer.build_index(reindex=True)
    if index_report["vectors"] != index_report["chunks"] or index_report["vectors"] <= 0:
        raise RuntimeError(
            f"index verification failed: {index_report['vectors']} vectors "
            f"!= {index_report['chunks']} chunks"
        )

    elapsed = round(time.monotonic() - started, 1)
    return {
        "status": "ok",
        "documents": downloaded,
        "chunks": index_report["chunks"],
        "vectors": index_report["vectors"],
        "embedding_model": index_report["embedding_model"],
        "fingerprint": fingerprint_after,
        "index_backup": backup,
        "elapsed_seconds": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scheduler.refresh",
        description="Refresh the corpus and atomically rebuild the vector index.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="rebuild the index even if the corpus is unchanged",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report what would run without scraping or rebuilding",
    )
    args = parser.parse_args(argv)

    if not config.REFRESH_ENABLED:
        print("REFRESH_ENABLED is false — refresh is disabled; nothing to do.")
        write_status("disabled")
        return 0

    try:
        with _single_flight_lock():
            report = run_refresh(force=args.force, dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001 - record + exit non-zero on any failure
        traceback.print_exc()
        write_status("failed", error=str(exc))
        print(f"\nREFRESH FAILED: {exc}")
        return 1

    if report.get("status") != "dry-run":
        write_status(**report)

    print("\n" + "=" * 60)
    print(f"REFRESH {report['status'].upper()}")
    for key in ("documents", "chunks", "vectors", "elapsed_seconds", "index_backup"):
        if key in report and report[key] is not None:
            print(f"  {key:16}: {report[key]}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
