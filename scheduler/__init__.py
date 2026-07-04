"""Scheduled data-refresh package (Phase 10).

Re-runs the offline ingestion pipeline (scrape -> parse -> chunk -> embed ->
index) so the served corpus stays current. The cadence is owned by the
scheduled GitHub Actions workflow (`.github/workflows/refresh.yml`); this
package holds the orchestration that a scheduled or manual run invokes.
"""
