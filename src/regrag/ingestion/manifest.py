"""Document manifest: the single source of truth for which documents are in the corpus."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from regrag.schemas import DocumentMeta


class ManifestError(ValueError):
    pass


def lock_path(manifest: Path) -> Path:
    return manifest.with_name(manifest.stem + ".lock.yaml")


def load_manifest(path: Path) -> list[DocumentMeta]:
    """Load the hand-written manifest and overlay download provenance (sha256, download_date)
    from the machine-written lock file next to it. The manifest itself is never rewritten."""
    raw = yaml.safe_load(path.read_text()) or {}
    lock = {}
    lp = lock_path(path)
    if lp.exists():
        lock = (yaml.safe_load(lp.read_text()) or {}).get("documents", {})
    docs, errors, seen = [], [], set()
    for i, entry in enumerate(raw.get("documents", [])):
        entry = {**entry, **lock.get(entry.get("document_id"), {})}
        try:
            meta = DocumentMeta.model_validate(entry)
        except ValidationError as e:
            errors.append(f"entry {i} ({entry.get('document_id')}): {e.errors()[0]['msg']}")
            continue
        if meta.document_id in seen:
            errors.append(f"duplicate document_id: {meta.document_id}")
            continue
        seen.add(meta.document_id)
        docs.append(meta)
    if errors:
        raise ManifestError("Invalid manifest:\n  " + "\n  ".join(errors))
    return docs


def save_lock(path: Path, docs: list[DocumentMeta]) -> None:
    """Write provenance of downloaded files to <manifest>.lock.yaml (commit it for reproducibility)."""
    data = {"documents": {
        d.document_id: {"sha256": d.sha256, "download_date": d.download_date.isoformat()}
        for d in docs if d.sha256 and d.download_date
    }}
    header = "# Machine-written by `regrag ingest`: provenance of downloaded files. Do not edit.\n"
    lock_path(path).write_text(header + yaml.safe_dump(data, sort_keys=True))
