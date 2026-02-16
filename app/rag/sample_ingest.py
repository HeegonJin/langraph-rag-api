"""Shared utility: auto-ingest sample_data/ files on first run.

Used by both FastAPI startup and Streamlit demo to avoid duplicated logic.
"""

import logging
from pathlib import Path

from app.config import SAMPLE_DATA_DIR
from app.rag.ingestion import ingest_file

logger = logging.getLogger(__name__)


def auto_ingest_sample_data() -> None:
    """Ingest files from ``sample_data/`` if not already done.

    Tracks completion via a marker file so the operation is idempotent
    across restarts.
    """
    marker = Path("uploads/.sample_ingested")

    if marker.exists():
        logger.info("Sample data already ingested – skipping")
        return

    if not SAMPLE_DATA_DIR.is_dir():
        logger.info("No sample_data/ directory found – skipping")
        return

    supported = {".pdf", ".txt", ".md", ".csv"}
    files = [f for f in SAMPLE_DATA_DIR.iterdir() if f.suffix.lower() in supported]

    if not files:
        logger.info("No supported files in sample_data/ – skipping")
        return

    total_chunks = 0
    for filepath in files:
        try:
            n = ingest_file(filepath)
            total_chunks += n
            logger.info("Ingested %s → %d chunks", filepath.name, n)
        except Exception:
            logger.exception("Failed to ingest sample file %s", filepath.name)

    # Write marker so we don't re-ingest on next restart
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"Ingested {len(files)} files, {total_chunks} chunks\n")
    logger.info(
        "Sample data ingestion complete: %d files, %d chunks",
        len(files),
        total_chunks,
    )
