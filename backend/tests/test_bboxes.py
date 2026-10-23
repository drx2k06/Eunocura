"""Offline tests for exact citation boxes. No PDF, no network, no model.

    cd backend
    python tests/test_bboxes.py

The old coordinate code matched chunks to words by spelling and then padded
the box by 50 pt, so a highlight could sit on the wrong line. These tests pin
down the replacement: a chunk is boxed only by the words at its own character
offsets, repeated words resolve to the right instance, and nothing is padded.

Everything here is stdlib only. PyMuPDF is used only by render_highlight, and
that path is checked for its readable error when the library is missing.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.services.bbox import (  # noqa: E402
    BboxRenderError,
    clamp_rect,
    find_span,
    locate_chunk,
    render_highlight,
    span_to_rect,
    union_rects,
    word_spans,
)

# ---------------------------------------------------------------- fixtures

# A page with the same word on two different lines: "blood" in line 2 and
# line 3. This is exactly the case the word -> position dict got wrong.
LINES = [
    "DISCHARGE SUMMARY 12 Aug 2026",
    "Vitals on admission BP 148/92 mmHg",
    "Repeat blood test after 6 hours.",
    "Patient stable, no blood products given.",
]
PAGE_TEXT = "\n".join(LINES)

PAGE_WIDTH = 612.0
PAGE_HEIGHT = 792.0

LINE_TOP0 = 72.0
LINE_HEIGHT = 24.0
FONT_SIZE = 12.0
LEFT = 72.0
CHAR_WIDTH = 6.0
WORD_GAP = 4.0


def page_words(
    lines,
    page_height=PAGE_HEIGHT,
    top0=LINE_TOP0,
    line_height=LINE_HEIGHT,
    font_size=FONT_SIZE,
    left=LEFT,
    char_width=CHAR_WIDTH,
    word_gap=WORD_GAP,
):
    """Fake pdfplumber words, already converted to bottom-left PDF coords."""
    words = []
    for line_number, line in enumerate(lines):
        top = top0 + line_number * line_height
        bottom = top + font_size
        x = left
        for token in line.split(" "):
            x1 = x + char_width * len(token)
            words.append((token, {
                "x0": x,
                "y0": page_height - bottom,  # bottom edge (smaller Y)
                "x1": x1,
                "y1": page_height - top,     # top edge (larger Y)
            }))
            x = x1 + word_gap
    return words


WORDS = page_words(LINES)
SPANS = word_spans(PAGE_TEXT, WORDS)


def word_box(token, occurrence=0):
    """Rect of the nth occurrence of a word, straight from the fixture."""
    seen = 0
    for text, rect in WORDS:
        if text != token:
            continue
        if seen == occurrence:
            return rect
        seen += 1
    raise AssertionError(f"{token!r} occurrence {occurrence} not in fixture")


failures: list[str] = []
checks = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if ok:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


def skip(name: str, reason: str) -> None:
    print(f"  skip  {name}  ({reason})")


def close(a: float, b: float, tolerance: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tolerance


# ---------------------------------------------------------------- tests

print("\nexact citation boxes\n")

# 1. union of rects
r = union_rects([
    {"x0": 10, "y0": 20, "x1": 30, "y1": 40},
    {"x0": 25, "y0": 5, "x1": 60, "y1": 35},
])
check(
    "union of two boxes is the smallest box around both",
    r is not None and close(r["x0"], 10) and close(r["y0"], 5)
    and close(r["x1"], 60) and close(r["y1"], 40),
    str(r),
)
check("union of no boxes is None", union_rects([]) is None and union_rects(iter([])) is None)

# 2. clamp
r = clamp_rect({"x0": -12, "y0": 780, "x1": 700, "y1": 830}, PAGE_WIDTH, PAGE_HEIGHT)
check(
    "clamp keeps a box inside the page",
    close(r["x0"], 0) and close(r["y0"], 780) and close(r["x1"], PAGE_WIDTH) and close(r["y1"], PAGE_HEIGHT),
    str(r),
)
r = clamp_rect({"x0": 90, "y0": 200, "x1": 30, "y1": 100}, PAGE_WIDTH, PAGE_HEIGHT)
check(
    "clamp orders an inverted box",
    close(r["x0"], 30) and close(r["x1"], 90) and close(r["y0"], 100) and close(r["y1"], 200),
    str(r),
)

# 3. every word is pinned to its own offsets
check("every word on the page gets a span", len(SPANS) == len(WORDS), f"{len(SPANS)} of {len(WORDS)}")
check("the page starts at offset 0", SPANS[0][0] == 0, str(SPANS[0][:2]))
check(
    "spans are in order and every one matches its word text",
    all(PAGE_TEXT[s:e] == token for (s, e, _), (token, _) in zip(SPANS, WORDS)),
)
case_words = [("PATIENT", {"x0": 72, "y0": 700, "x1": 120, "y1": 712})]
check(
    "a word that only matches ignoring case is still placed",
    len(word_spans("Patient stable", case_words)) == 1,
)

# 4. repeated word: the chunk must box the instance it was cut from
first_blood = PAGE_TEXT.find("blood")
second_blood = PAGE_TEXT.find("blood products")
chunk_start = second_blood
chunk_text = PAGE_TEXT[chunk_start:chunk_start + len("blood products given.")]
r = locate_chunk(PAGE_TEXT, SPANS, chunk_text, chunk_start)
check("a chunk spanning the second 'blood' is located", r is not None, str(r))
check(
    "the box sits on the second 'blood' line, not the first",
    r is not None and close(r["y0"], word_box("blood", 1)["y0"]),
    f"{r} vs first {word_box('blood', 0)} second {word_box('blood', 1)}",
)
check(
    "the first 'blood' line is not inside the box",
    r is not None and r["y0"] < word_box("blood", 0)["y0"] and r["y1"] <= word_box("blood", 0)["y0"],
    str(r),
)
check(
    "the box is one line tall (no 50 pt padding into the neighbour)",
    r is not None and close(r["y1"] - r["y0"], FONT_SIZE),
    f"height {r['y1'] - r['y0'] if r else None}",
)
r = locate_chunk(PAGE_TEXT, SPANS, "blood", second_blood)
check(
    "the hint picks the near occurrence of a repeated word",
    r is not None and close(r["y0"], word_box("blood", 1)["y0"]) and close(r["x0"], word_box("blood", 1)["x0"]),
    str(r),
)
check("the repeated word really is at two offsets", first_blood != second_blood)

# 5. overlapping range unions across a line break
start = PAGE_TEXT.find("148/92")
end = PAGE_TEXT.find("hours.") + len("hours.")
r = span_to_rect(SPANS, start, end)
check("two overlapping lines union into one box", r is not None, str(r))
check(
    "the union reaches from the first line up to the second",
    r is not None and close(r["y0"], word_box("Repeat")["y0"]) and close(r["y1"], word_box("Vitals")["y1"]),
    str(r),
)
check(
    "the union spans exactly the two lines it touches",
    r is not None and close(r["y1"] - r["y0"], LINE_HEIGHT + FONT_SIZE),
    f"height {r['y1'] - r['y0'] if r else None}",
)
check(
    "the union starts at the leftmost word of the range",
    r is not None and close(r["x0"], min(word_box("148/92")["x0"], word_box("Repeat")["x0"])),
    f"{r} leftmost {min(word_box('148/92')['x0'], word_box('Repeat')['x0'])}",
)
check("a range with no words under it has no box", span_to_rect(SPANS, len(PAGE_TEXT) + 5, len(PAGE_TEXT) + 20) is None)

# 6. a chunk that cannot be located
check(
    "a chunk that is not on the page returns None",
    locate_chunk(PAGE_TEXT, SPANS, "this sentence is not on the page", 0) is None,
)
check("an empty chunk returns None", locate_chunk(PAGE_TEXT, SPANS, "   ", 0) is None)

# 7. whitespace and newline normalised match
needle = "vitals   on admission\n BP"
hit = find_span(PAGE_TEXT, needle)
check("a re-spaced quote is still found", hit is not None, str(hit))
check(
    "the normalised match maps back to the original offsets",
    hit == (PAGE_TEXT.find("Vitals"), PAGE_TEXT.find("BP") + len("BP")),
    f"{hit} vs {(PAGE_TEXT.find('Vitals'), PAGE_TEXT.find('BP') + len('BP'))}",
)
r = locate_chunk(PAGE_TEXT, SPANS, needle, None)
check(
    "a re-spaced chunk boxes the same words",
    r is not None and close(r["x0"], word_box("Vitals")["x0"])
    and close(r["x1"], word_box("BP")["x1"]) and close(r["y1"] - r["y0"], FONT_SIZE),
    str(r),
)
check("a quote that is not there at all returns None", find_span(PAGE_TEXT, "sepsis protocol") is None)

# 8. rendering reports a missing PyMuPDF in words a human can act on
have_pymupdf = (
    importlib.util.find_spec("pymupdf") is not None
    or importlib.util.find_spec("fitz") is not None
)
if have_pymupdf:
    skip("render_highlight explains a missing PyMuPDF", "PyMuPDF is installed")
    try:
        render_highlight("no-such-file.pdf", 1, [word_box("Vitals")], "never-written.png")
    except BboxRenderError as exc:
        check("render_highlight explains an unreadable PDF", "could not open" in str(exc), str(exc))
    else:
        check("render_highlight explains an unreadable PDF", False, "no error raised")
else:
    try:
        render_highlight("no-such-file.pdf", 1, [word_box("Vitals")], "never-written.png")
    except BboxRenderError as exc:
        check(
            "render_highlight explains a missing PyMuPDF",
            "pip install pymupdf" in str(exc),
            str(exc),
        )
    else:
        check("render_highlight explains a missing PyMuPDF", False, "no error raised")

# ---------------------------------------------------------------- result

print(f"\n  {checks - len(failures)}/{checks} passed\n")
if failures:
    print("  FAIL  " + ", ".join(failures) + "\n")
    sys.exit(1)
print("  PASS  citation boxes are exact without a model\n")
