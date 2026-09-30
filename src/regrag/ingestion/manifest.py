"""Document manifest: the single source of truth for which documents are in the corpus."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from regrag.schemas import DocumentMeta


class ManifestError(ValueError):
    pass


def load_manifest(path: Path) -> list[DocumentMeta]:
    raw = yaml.safe_load(path.read_text()) or {}
    docs, errors, seen = [], [], set()
    for i, entry in enumerate(raw.get("documents", [])):
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


def save_manifest(path: Path, docs: list[DocumentMeta]) -> None:
    data = {"documents": [d.model_dump(mode="json", exclude_none=True) for d in docs]}
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
