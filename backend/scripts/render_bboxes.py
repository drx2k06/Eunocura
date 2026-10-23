"""Render a PDF page with the citation boxes this app actually stores.

    cd backend
    python scripts/render_bboxes.py --pdf report.pdf --page 2 --out highlighted.png
    python scripts/render_bboxes.py --pdf report.pdf --page 2 --text "Vitals on admission" --out highlighted.png
    python scripts/render_bboxes.py --pdf report.pdf --page 2 --rect 80,120,540,160 --out highlighted.png

Rects are x0,y0,x1,y1 in the same convention the pipeline and the frontend
use: bottom-left origin, y0 is the bottom edge, y1 the top edge, so y0 < y1.
`--rect` can be repeated. `--text` locates a sentence on the page with the same
word-span logic `services/pdf_rag.py` uses, so what you see here is what the
viewer highlights. The resolved JSON is printed before the PNG is written.

This is the looking-at-it tool for the "exact boxes" fix: if a box lands on the
wrong line, this is the fastest way to see it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from src.services.bbox import (  # noqa: E402
    BboxRenderError,
    Rect,
    clamp_rect,
    locate_chunk,
    render_highlight,
    word_spans,
)


def parse_rect(value: str) -> Rect:
    """Parse 'x0,y0,x1,y1' (bottom-left origin, app convention)."""
    parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"expected x0,y0,x1,y1 — got {value!r}")
    try:
        x0, y0, x1, y1 = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not four numbers") from exc
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _load_fitz() -> Any:
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24 name
    except ImportError:
        try:
            import fitz  # older name, same library
        except ImportError as exc:
            raise BboxRenderError(
                "PyMuPDF is not installed. Highlight rendering needs it: pip install pymupdf"
            ) from exc
    return fitz


def page_size(pdf_path: str, page_number: int) -> Tuple[float, float]:
    """Page width/height in points (PyMuPDF, because rendering needs it anyway)."""
    fitz = _load_fitz()
    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        raise BboxRenderError(f"could not open {pdf_path}: {exc}") from exc
    try:
        if not 1 <= page_number <= len(document):
            raise BboxRenderError(f"page {page_number} is outside 1..{len(document)}")
        rect = document[page_number - 1].rect
        return float(rect.width), float(rect.height)
    finally:
        document.close()


def page_words(pdf_path: str, page_number: int) -> Tuple[str, float, float, List[Tuple[str, Rect]]]:
    """Page text and word rects, extracted the way the pipeline extracts them."""
    try:
        import pdfplumber
    except ImportError as exc:
        raise BboxRenderError(
            "locating --text needs pdfplumber: pip install pdfplumber"
        ) from exc

    try:
        document = pdfplumber.open(pdf_path)
    except Exception as exc:
        raise BboxRenderError(f"could not open {pdf_path}: {exc}") from exc

    with document:
        if not 1 <= page_number <= len(document.pages):
            raise BboxRenderError(f"page {page_number} is outside 1..{len(document.pages)}")
        page = document.pages[page_number - 1]
        width, height = float(page.width), float(page.height)
        text = page.extract_text() or ""
        words: List[Tuple[str, Rect]] = []
        for word in page.extract_words(use_text_flow=True, keep_blank_chars=False):
            token = (word.get("text") or "").strip()
            if not token:
                continue
            x0 = float(word.get("x0", 0))
            x1 = float(word.get("x1", x0))
            top = float(word.get("top", word.get("y0", 0)))
            bottom = float(word.get("bottom", word.get("y1", top)))
            # pdfplumber is top-left origin; the app stores bottom-left PDF coords
            words.append((token, {
                "x0": x0,
                "y0": height - bottom,
                "x1": x1,
                "y1": height - top,
            }))
    return text, width, height, words


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw the app's citation boxes on a PDF page and save a PNG.",
    )
    parser.add_argument("--pdf", required=True, help="path to the PDF")
    parser.add_argument("--page", required=True, type=int, help="1-based page number")
    parser.add_argument("--out", default="highlighted.png", help="PNG to write (default: highlighted.png)")
    parser.add_argument(
        "--rect",
        action="append",
        default=[],
        type=parse_rect,
        metavar="x0,y0,x1,y1",
        help="box in bottom-left PDF coords; repeat for more boxes",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        metavar="SENTENCE",
        help="locate this sentence on the page with the app's word spans; repeatable",
    )
    parser.add_argument("--pad", type=float, default=2.0, help="padding in points around each box (default: 2)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.rect and not args.text:
        parser.error('give at least one --rect x0,y0,x1,y1 or --text "some sentence"')

    pdf_path = str(args.pdf)
    if not pathlib.Path(pdf_path).is_file():
        print(f"error: no such PDF: {pdf_path}", file=sys.stderr)
        return 2

    try:
        text = ""
        spans: List[Tuple[int, int, Rect]] = []
        if args.text:
            text, width, height, words = page_words(pdf_path, args.page)
            spans = word_spans(text, words)
        else:
            width, height = page_size(pdf_path, args.page)

        resolved: List[Dict[str, Any]] = []
        for rect in args.rect:
            resolved.append({"source": "--rect", "rect": clamp_rect(rect, width, height)})

        for needle in args.text:
            rect = locate_chunk(text, spans, needle, None)
            if rect is None:
                print(f"error: could not locate {needle!r} on page {args.page}", file=sys.stderr)
                return 3
            resolved.append({"source": f"--text {needle!r}", "rect": clamp_rect(rect, width, height)})

        print(json.dumps({
            "pdf": pdf_path,
            "page": args.page,
            "page_size": {"width": round(width, 3), "height": round(height, 3)},
            "convention": "bottom-left origin: x0=left, y0=bottom, x1=right, y1=top",
            "rects": [
                {"source": item["source"], **{key: round(value, 3) for key, value in item["rect"].items()}}
                for item in resolved
            ],
        }, indent=2))

        out = render_highlight(
            pdf_path,
            args.page,
            [item["rect"] for item in resolved],
            args.out,
            pad=args.pad,
        )
    except BboxRenderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
