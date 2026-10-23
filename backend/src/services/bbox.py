"""Exact citation boxes.

A citation is only trustworthy if the highlight lands on the line it quotes.
The first version of the coordinate code in `services/pdf_rag.py` could not do
that: it built a word -> position dict, matched only the first ten words of a
chunk, then grew the box by 50 pt in every direction. Repeated words ("blood",
"patient") therefore attached to whichever instance was seen first, and the
50 pt padding swallowed the neighbouring line too.

This module is the replacement, and it is deliberately geometry only: no
pdfplumber, no PyMuPDF, no app imports at module level, so the whole thing is
testable with the standard library alone. A page is walked left to right once,
every word is pinned to its own character offsets, and a chunk is boxed by the
words that actually fall inside its own character range — no guessing, no
padding, no reuse of a word that belongs to a different line.

Coordinate systems (the other place the old code went wrong)
-----------------------------------------------------------
pdfplumber reports boxes with a top-left origin: `top` grows downwards and
`bottom` is the larger number. The rest of the app — `PDFChunk`, `search()`,
the frontend viewer — speaks PDF space with a bottom-left origin, where `y0` is
the bottom edge and `y1` is the top edge, so `y0 < y1`. Every rect in this
module uses the app convention:

    x0 = left, y0 = bottom, x1 = right, y1 = top      (y0 < y1)

A caller holding pdfplumber words converts once, before calling:

    pdf_y0 = page_height - word["y1"]   # bottom edge (smaller Y)
    pdf_y1 = page_height - word["y0"]   # top edge (larger Y)

`render_highlight` accepts those bottom-left rects and flips them back to
top-left space internally, because PyMuPDF draws in top-left space as well.
"""

from __future__ import annotations

import pathlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Rect = Dict[str, float]  # keys exactly: x0, y0, x1, y1

# A whitespace-normalised match may cover more source characters than the
# needle (collapsed runs, a newline where the quote had a space). Past this
# slack the match is not the same sentence any more, so we reject it.
_SLACK_BASE = 16


class BboxRenderError(RuntimeError):
    """A highlight could not be drawn (usually PyMuPDF is not installed)."""


# ---------------------------------------------------------------- geometry


def union_rects(rects: Iterable[Rect]) -> Optional[Rect]:
    """Smallest rect covering all of `rects`, or None when there are none."""
    result: Optional[Rect] = None
    for rect in rects:
        if not rect:
            continue
        x0, y0, x1, y1 = (float(rect["x0"]), float(rect["y0"]),
                          float(rect["x1"]), float(rect["y1"]))
        if result is None:
            result = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
            continue
        result["x0"] = min(result["x0"], x0)
        result["y0"] = min(result["y0"], y0)
        result["x1"] = max(result["x1"], x1)
        result["y1"] = max(result["y1"], y1)
    return result


def clamp_rect(rect: Rect, page_width: float, page_height: float) -> Rect:
    """Keep a rect inside the page and in x0<=x1, y0<=y1 order."""
    x0 = min(float(rect["x0"]), float(rect["x1"]))
    x1 = max(float(rect["x0"]), float(rect["x1"]))
    y0 = min(float(rect["y0"]), float(rect["y1"]))
    y1 = max(float(rect["y0"]), float(rect["y1"]))
    width = max(0.0, float(page_width))
    height = max(0.0, float(page_height))
    return {
        "x0": min(max(x0, 0.0), width),
        "y0": min(max(y0, 0.0), height),
        "x1": min(max(x1, 0.0), width),
        "y1": min(max(y1, 0.0), height),
    }


def word_spans(
    page_text: str,
    words: Sequence[Tuple[str, Rect]],
) -> List[Tuple[int, int, Rect]]:
    """Pin each word to its character range in `page_text`.

    Words are matched in reading order with a cursor that only moves forward,
    so the second "blood" on a page gets the second "blood" offsets. A word is
    matched case-sensitively first and case-insensitively second; a word we
    cannot place (ligatures, odd glyph mapping, out-of-order layout) is
    skipped rather than forced onto the wrong occurrence.
    """
    spans: List[Tuple[int, int, Rect]] = []
    if not page_text:
        return spans

    lowered = page_text.lower()
    cursor = 0

    for word_text, rect in words:
        needle = (word_text or "").strip()
        if not needle or not rect:
            continue

        start = page_text.find(needle, cursor)
        if start < 0:
            start = lowered.find(needle.lower(), cursor)
        if start < 0:
            continue

        end = start + len(needle)
        spans.append((
            start,
            end,
            {
                "x0": float(rect["x0"]),
                "y0": float(rect["y0"]),
                "x1": float(rect["x1"]),
                "y1": float(rect["y1"]),
            },
        ))
        cursor = end

    return spans


def span_to_rect(
    spans: Sequence[Tuple[int, int, Rect]],
    start: int,
    end: int,
) -> Optional[Rect]:
    """Union of every word that overlaps the character range [start, end)."""
    hits = [rect for (s_start, s_end, rect) in spans if s_end > start and s_start < end]
    return union_rects(hits)


# ---------------------------------------------------------------- location


def _collapse(text: str) -> Tuple[str, List[int]]:
    """Lower-case text with collapsed whitespace, plus char -> original index."""
    chars: List[str] = []
    origin: List[int] = []
    pending_space = False

    for index, char in enumerate(text):
        if char.isspace():
            if chars and not pending_space:
                pending_space = True
                chars.append(" ")
                origin.append(index)
            continue
        pending_space = False
        chars.append(char.lower())
        origin.append(index)

    return "".join(chars), origin


def _nearest(candidates: Sequence[int], hint: Optional[int]) -> int:
    """Index of the candidate nearest `hint` (the first one when no hint)."""
    if hint is None or not candidates:
        return 0
    best = 0
    best_distance = abs(candidates[0] - hint)
    for i in range(1, len(candidates)):
        distance = abs(candidates[i] - hint)
        if distance < best_distance:
            best, best_distance = i, distance
    return best


def find_span(
    page_text: str,
    needle: str,
    hint: Optional[int] = None,
) -> Optional[Tuple[int, int]]:
    """Character range of `needle` in `page_text`, or None.

    Tries an exact substring first, then a search over whitespace-collapsed,
    lower-cased copies (a quote copied out of a PDF rarely keeps the original
    newlines and double spaces). `hint` is the offset the caller already
    expects the text at — the chunk's own start — and is used only to pick
    between otherwise identical candidates.
    """
    needle = (needle or "").strip()
    if not page_text or not needle:
        return None

    starts: List[int] = []
    pos = page_text.find(needle)
    while pos >= 0:
        starts.append(pos)
        pos = page_text.find(needle, pos + 1)
    if starts:
        best = starts[_nearest(starts, hint)]
        return best, best + len(needle)

    normalised, origin = _collapse(page_text)
    flat_needle = _collapse(needle)[0]
    if not flat_needle or not normalised:
        return None

    # Same length budget: whitespace collapsing may stretch the source span,
    # but not past the needle plus a little slack.
    budget = len(needle) + max(_SLACK_BASE, len(needle) // 4)

    hits: List[Tuple[int, int]] = []
    pos = normalised.find(flat_needle)
    while pos >= 0:
        end = pos + len(flat_needle)
        started = origin[pos]
        finished = origin[end] if end < len(origin) else len(page_text)
        if finished - started <= budget:
            hits.append((started, finished))
        pos = normalised.find(flat_needle, pos + 1)

    if not hits:
        return None
    best = hits[_nearest([hit[0] for hit in hits], hint)]
    return best


def locate_chunk(
    page_text: str,
    spans: Sequence[Tuple[int, int, Rect]],
    chunk_text: str,
    hint_start: Optional[int],
) -> Optional[Rect]:
    """Box the words a chunk actually covers, or None if it cannot be placed."""
    needle = (chunk_text or "").strip()
    if not needle:
        return None

    found = find_span(page_text, needle, hint_start)
    if found is None:
        return None
    return span_to_rect(spans, found[0], found[1])


# ---------------------------------------------------------------- rendering


def render_highlight(
    pdf_path: str,
    page_number: int,
    rects: Sequence[Rect],
    out_path: str,
    pad: float = 2.0,
) -> str:
    """Render one page as a PNG with a translucent yellow box per rect.

    `rects` are in the app's bottom-left-origin PDF space; PyMuPDF wants
    top-left, so each box is flipped here. The PDF itself is never modified —
    only the rendered image is written.
    """
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24 name
    except ImportError:
        try:
            import fitz  # older name, same library
        except ImportError as exc:
            raise BboxRenderError(
                "PyMuPDF is not installed. Highlight rendering needs it: pip install pymupdf"
            ) from exc

    boxes = [rect for rect in rects if rect]
    if not boxes:
        raise BboxRenderError("no rectangles to draw")

    try:
        document = fitz.open(pdf_path)
    except Exception as exc:  # PyMuPDF raises its own error types
        raise BboxRenderError(f"could not open {pdf_path}: {exc}") from exc

    try:
        page_count = len(document)
        if not 1 <= int(page_number) <= page_count:
            raise BboxRenderError(f"page {page_number} is outside 1..{page_count}")

        page = document[int(page_number) - 1]
        page_box = page.rect
        page_height = page.rect.height

        for rect in boxes:
            # bottom-left (app) -> top-left (PyMuPDF)
            top = page_height - float(rect["y1"]) - pad
            bottom = page_height - float(rect["y0"]) + pad
            left = float(rect["x0"]) - pad
            right = float(rect["x1"]) + pad
            target = fitz.Rect(
                min(left, right),
                min(top, bottom),
                max(left, right),
                max(top, bottom),
            ) & page_box
            if target.is_empty:
                continue
            page.draw_rect(
                target,
                color=(0.72, 0.52, 0.0),
                fill=(1.0, 0.94, 0.25),
                fill_opacity=0.35,
                width=1.0,
            )

        output = pathlib.Path(out_path)
        if output.parent and not output.parent.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
        page.get_pixmap(dpi=150).save(str(output))
    finally:
        document.close()

    return str(out_path)
