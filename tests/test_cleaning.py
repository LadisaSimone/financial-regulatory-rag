from regrag.ingestion.cleaning import (
    clean_text,
    detect_repeated_lines,
    fix_hyphenation,
    remove_lines,
    unwrap_lines,
)


def test_fix_hyphenation():
    assert fix_hyphenation("due dili-\ngence") == "due diligence"
    assert fix_hyphenation("risk-based approach") == "risk-based approach"


def test_repeated_header_footer_removed():
    pages = [f"EBA Public\nBody text {i}\nMore text\nPage {i} of 10" for i in range(1, 8)]
    rep = detect_repeated_lines(pages, 0.5)
    out = remove_lines(pages[2], rep)
    assert "EBA Public" not in out and "Page" not in out and "Body text 3" in out


def test_unwrap_keeps_bullets():
    txt = "Firms should apply\nenhanced measures.\n(a) first item\n(b) second item"
    out = unwrap_lines(txt)
    assert "Firms should apply enhanced measures." in out
    assert "\n(a) first item\n(b) second item" in out


def test_clean_collapses_whitespace():
    assert clean_text("a   b\n\n\n\nc") == "a b\n\nc"
