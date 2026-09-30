"""Structure-preserving PDF parsing with PyMuPDF.

Headings are detected from font size / boldness relative to the document's body font,
plus numbering patterns common in regulatory texts ("4.2 Enhanced due diligence",
"Title II", "Article 18"). Every page keeps its original page number so each chunk can be
traced back to the page it came from.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from regrag.ingestion.cleaning import clean_text, detect_repeated_lines, remove_lines
from regrag.logging_utils import get_logger, log
from regrag.schemas import DocumentMeta, DocumentPage

logger = get_logger("ingestion.parsing")

HEADING_PATTERN = re.compile(
    r"^(\d+(\.\d+){0,3}\.?\s+[A-Z]|(Title|Chapter|Section|Article|Annex|Recommendation)\s+[IVXLC\d]+)",
)


class ParseError(RuntimeError):
    pass


def _span_stats(doc) -> float:
    sizes: Counter[float] = Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if span["text"].strip():
                        sizes[round(span["size"], 1)] += len(span["text"])
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _page_lines(page, body_size: float) -> tuple[list[str], list[str]]:
    """Return (all lines, heading lines) for a page."""
    lines, headings = [], []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text:
                continue
            size = max(s["size"] for s in line["spans"])
            bold = any("bold" in s["font"].lower() or (s["flags"] & 16) for s in line["spans"])
            short = len(text) < 120
            looks_numbered = bool(HEADING_PATTERN.match(text))
            if short and (size >= body_size + 1.5 or (bold and looks_numbered) or (looks_numbered and size > body_size)):
                headings.append(text)
            lines.append(text)
        lines.append("")  # block separator -> paragraph break
    return lines, headings


def parse_pdf(path: Path, meta: DocumentMeta, header_footer_min_ratio: float = 0.5) -> list[DocumentPage]:
    import fitz  # PyMuPDF

    try:
        doc = fitz.open(path)
    except Exception as e:  # corrupted / not a PDF
        raise ParseError(f"{meta.document_id}: cannot open PDF: {e}") from e

    if doc.needs_pass:
        raise ParseError(f"{meta.document_id}: encrypted PDF")

    body_size = _span_stats(doc)
    raw_pages: list[tuple[list[str], list[str]]] = []
    for i, page in enumerate(doc):
        try:
            raw_pages.append(_page_lines(page, body_size))
        except Exception as e:
            log(logger, "page_parse_failed", document_id=meta.document_id, page=i + 1, error=str(e))
            raw_pages.append(([], []))

    texts = ["\n".join(lines) for lines, _ in raw_pages]
    repeated = detect_repeated_lines(texts, min_ratio=header_footer_min_ratio)

    pages: list[DocumentPage] = []
    current_section: str | None = None
    for idx, ((_lines, headings), text) in enumerate(zip(raw_pages, texts, strict=True)):
        headings = [h for h in headings if h.strip().lower() not in {r for r in repeated}]
        if headings:
            current_section = headings[-1]
        cleaned = clean_text(remove_lines(text, repeated))
        pages.append(
            DocumentPage(
                document_id=meta.document_id,
                page_number=idx + 1,
                text=cleaned,
                section=headings[0] if headings else current_section,
                headings=headings,
                metadata={"title": meta.title, "authority": meta.authority},
            )
        )
    doc.close()
    return pages
