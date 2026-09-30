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

from regrag.ingestion.cleaning import clean_text, detect_repeated_lines, normalize_line, remove_lines
from regrag.logging_utils import get_logger, log
from regrag.schemas import DocumentMeta, DocumentPage

logger = get_logger("ingestion.parsing")

NUMBER_ONLY = re.compile(r"^([A-Z]|\d+(\.\d+){0,3})\.?$")
STRUCTURAL = re.compile(
    r"^(TITLE|Title|CHAPTER|Chapter|SECTION|Section|ARTICLE|Article|ANNEX|Annex|Recommendation|RECOMMENDATION)"
    r"\s+[IVXLC\d]+[a-z]?$"
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


def _is_bold(span) -> bool:
    return bool(span["flags"] & 16) or "bold" in span["font"].lower()


def _classify(text: str, spans: list, body_size: float) -> str:
    """'heading' | 'marker' (numbering/structural label that may prefix a heading) | 'body'."""
    t = " ".join(text.split())
    if not re.search(r"[A-Za-z]", t):
        return "marker" if NUMBER_ONLY.match(t) and all(_is_bold(s) for s in spans if s["text"].strip()) else "body"
    if STRUCTURAL.match(t):
        return "marker"
    if len(t) < 4 or len(t) > 110 or t.endswith((".", ",", ";", ":")):
        return "body"
    size = max(s["size"] for s in spans)
    all_bold = all(_is_bold(s) for s in spans if s["text"].strip())
    if size >= body_size + 2.0 or (all_bold and size >= body_size - 0.5):
        return "heading"
    return "body"


def _page_lines(page, body_size: float) -> tuple[list[str], list[str]]:
    """Return (lines, headings) for a page.

    Heading rules (tuned on EBA / FATF / EUR-Lex layouts):
      * a short line, not ending like a sentence, that is fully bold or clearly larger than body text;
      * numbered paragraphs ("4.38. Firms should ...") are body text even if the number is bold;
      * a numbering-only or structural line ("3.", "Article 18", "SECTION 2") is merged with the
        heading that follows it ("3. Money laundering offence"), or kept alone if none follows.
    Headings are emitted as their own paragraph so they never get glued to body text.
    """
    rows: list[tuple[str, str]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(s["text"] for s in line["spans"]).strip()
            if text:
                rows.append((" ".join(text.split()), _classify(text, line["spans"], body_size)))
        rows.append(("", "sep"))

    lines: list[str] = []
    headings: list[str] = []
    pending: list[str] = []  # markers waiting for a title

    def flush_pending():
        if pending:
            h = " ".join(pending)
            if STRUCTURAL.match(h):  # "Article 14" alone is still a useful section label
                headings.append(h)
            lines.extend(["", h, ""])
            pending.clear()

    for text, kind in rows:
        if kind == "sep":
            if not pending:
                lines.append("")
            continue
        if kind == "marker":
            pending.append(text)
            continue
        if kind == "heading":
            h = " ".join([*pending, text]).rstrip(" *")
            pending.clear()
            if headings and lines and lines[-2:-1] == [headings[-1]] and not STRUCTURAL.match(headings[-1]):
                # multi-line heading: continue the previous one
                headings[-1] = f"{headings[-1]} {h}"
                lines[-2] = headings[-1]
                continue
            headings.append(h)
            lines.extend(["", h, ""])
            continue
        flush_pending()
        lines.append(text)
    flush_pending()
    return lines, headings


def parse_pdf(path: Path, meta: DocumentMeta, header_footer_min_ratio: float = 0.5) -> list[DocumentPage]:
    try:
        import pymupdf as fitz
    except ImportError:  # PyMuPDF < 1.24.3
        import fitz

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
        headings = [h for h in headings if normalize_line(h) not in repeated]
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
