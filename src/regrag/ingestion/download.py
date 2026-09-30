"""Download + validate source PDFs. Idempotent: existing files with a matching hash are skipped."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from regrag.logging_utils import get_logger, log
from regrag.schemas import DocumentMeta

logger = get_logger("ingestion.download")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    return isinstance(exc, httpx.TransportError)


@retry(
    retry=retry_if_exception(_transient),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, max=20),
    reraise=True,
)
def _fetch(url: str, timeout_s: int) -> bytes:
    headers = {"User-Agent": "regrag/0.1 (research; +https://ladisasimone.com)"}
    with httpx.Client(follow_redirects=True, timeout=timeout_s, headers=headers) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.content


def download_document(meta: DocumentMeta, raw_dir: Path, timeout_s: int = 60) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"{meta.document_id}.pdf"
    if target.exists() and target.stat().st_size > 0:
        if meta.sha256 and sha256_of(target) != meta.sha256:
            log(logger, "hash_mismatch_redownload", document_id=meta.document_id)
        else:
            return target
    content = _fetch(str(meta.source_url), timeout_s)
    if not content.startswith(b"%PDF"):
        raise ValueError(f"{meta.document_id}: downloaded content is not a PDF")
    target.write_bytes(content)
    meta.download_date = date.today()
    meta.sha256 = sha256_of(target)
    log(logger, "downloaded", document_id=meta.document_id, bytes=len(content))
    return target
