"""Text cleaning helpers. Pure functions -> easy to unit test."""

from __future__ import annotations

import re
from collections import Counter

_PAGE_NUM = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*(/|of)\s*\d{1,4})?\s*$", re.IGNORECASE)
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_SPACE = re.compile(r"[ \t ]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def normalize_line(line: str) -> str:
    """Line key for header/footer detection: digits masked so 'Page 3' == 'Page 7'."""
    return re.sub(r"\d+", "#", line.strip().lower())


def detect_repeated_lines(pages: list[str], min_ratio: float = 0.5, edge_lines: int = 3) -> set[str]:
    """Lines that appear in the first/last `edge_lines` of at least `min_ratio` of pages."""
    if len(pages) < 3:
        return set()
    counter: Counter[str] = Counter()
    for text in pages:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        n = max(1, min(edge_lines, len(lines) // 4))  # short pages: only the very first/last line
        edges = {normalize_line(ln) for ln in lines[:n] + lines[-n:]}
        counter.update(edges)
    threshold = max(2, int(len(pages) * min_ratio))
    return {k for k, c in counter.items() if c >= threshold and k}


def remove_lines(text: str, repeated: set[str]) -> str:
    out = []
    for ln in text.splitlines():
        if normalize_line(ln) in repeated:
            continue
        if _PAGE_NUM.match(ln):
            continue
        out.append(ln)
    return "\n".join(out)


def fix_hyphenation(text: str) -> str:
    """'require-\\nments' -> 'requirements'. Keeps real hyphens inside a line."""
    return _HYPHEN_BREAK.sub(r"\1\2", text)


def unwrap_lines(text: str) -> str:
    """Join soft-wrapped lines inside paragraphs; keep blank lines, bullets and numbered items."""
    lines = text.split("\n")
    out: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            out.append("")
            continue
        starts_block = bool(re.match(r"^([•\-–▪*]|\(?[a-z0-9]{1,3}[.)]|\d+(\.\d+)*\s)", s))
        if out and out[-1] and not starts_block and not out[-1].endswith((".", ":", ";")):
            out[-1] = out[-1] + " " + s
        else:
            out.append(s)
    return "\n".join(out)


def clean_text(text: str) -> str:
    text = text.replace("\r", "")
    text = fix_hyphenation(text)
    text = unwrap_lines(text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()
