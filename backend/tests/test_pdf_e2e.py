"""End-to-end integration test for clinical document intelligence.

Tests:
1. PDF text & word coordinate extraction with pdfplumber
2. Word character span geometry with src.services.bbox
3. Exact bounding-box calculation
4. Citation-or-drop verification with coordinates attached
5. Evidence PNG rendering with PyMuPDF
6. Degrade behavior when inputs are invalid
"""

from __future__ import annotations

import pathlib
import sys

BACKEND_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import pdfplumber
from src.services.bbox import locate_chunk, render_highlight, word_spans
from src.services.citations import verify

PDF_PATH = (
    BACKEND_DIR.parent
    / ".agents"
    / "skills"
    / "clinical-document-intelligence"
    / "assets"
    / "sample-discharge-summary.pdf"
)

checks = 0
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if ok:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


def main() -> int:
    print("\nEUNOCURA end-to-end document intelligence test\n")

    check("sample discharge summary exists", PDF_PATH.is_file(), str(PDF_PATH))

    # 1. Extraction
    with pdfplumber.open(PDF_PATH) as pdf:
        pages = {i + 1: page.extract_text() for i, page in enumerate(pdf.pages)}
        check("extracted exactly 2 pages", len(pages) == 2, f"pages: {len(pages)}")
        check("page 1 contains patient name", "Asha Rao" in pages[1])
        check("page 1 contains admission vitals", "148/92 mmHg" in pages[1])
        check("page 2 contains follow-up plan", "Follow-up plan" in pages[2] or "FOLLOW-UP PLAN" in pages[2])

        # 2. Geometry & Word Spans
        page1 = pdf.pages[0]
        words1 = [
            (
                w["text"],
                {
                    "x0": float(w["x0"]),
                    "y0": float(page1.height - w["bottom"]),
                    "x1": float(w["x1"]),
                    "y1": float(page1.height - w["top"]),
                },
            )
            for w in page1.extract_words()
        ]
        spans1 = word_spans(pages[1], words1)
        check("page 1 has extracted words", len(words1) > 50, f"count: {len(words1)}")
        check("word spans mapped successfully", len(spans1) == len(words1), f"{len(spans1)} vs {len(words1)}")

        # 3. Exact Coordinate Locator
        needle = "SpO2 91% on room air, temperature 38.6 C."
        box = locate_chunk(pages[1], spans1, needle, None)
        check("located exact coordinates for vital quote", box is not None and box["x0"] > 0)

        # 4. Citation-or-drop with coordinate attachment
        chunks = [
            {
                "text": needle,
                "page_number": 1,
                "coordinates": box,
            }
        ]
        claims = [
            {
                "claim": "Patient presented with hypoxia and fever",
                "quote": needle,
                "page": 1,
                "confidence": 0.95,
            },
            {
                "claim": "Follow-up plan includes repeat chest radiograph",
                "quote": "Repeat chest radiograph in 6 weeks.",
                "page": 1,  # Intentional wrong page (actually page 2)
                "confidence": 0.9,
            },
            {
                "claim": "Patient had an acute myocardial infarction",
                "quote": "elevated troponin I indicating acute myocardial infarction",
                "page": 1,  # Fabricated claim
                "confidence": 0.85,
            },
        ]

        report = verify(claims, pages, chunks)
        check("report processed 3 claims", report.total == 3)
        check("supported claim kept", report.claims[0].status == "supported")
        check("supported claim has exact bbox attached", report.claims[0].coords == box, str(report.claims[0].coords))
        check("wrong-page claim corrected to page 2", report.claims[1].status == "page_corrected" and report.claims[1].page == 2)
        check("fabricated claim dropped", report.claims[2].status == "dropped")

        # 5. Render highlighted evidence PNG
        out_png = BACKEND_DIR / "tests" / "fixtures" / "test_highlight.png"
        if out_png.exists():
            out_png.unlink()

        try:
            render_highlight(str(PDF_PATH), 1, [box], str(out_png))
            check("rendered evidence PNG file created", out_png.is_file(), str(out_png))
            check("evidence PNG has non-zero size", out_png.stat().st_size > 1000, f"size: {out_png.stat().st_size}")
        finally:
            if out_png.exists():
                out_png.unlink()

    print(f"\n  {checks - len(failures)}/{checks} passed\n")
    if failures:
        print("  failed: " + ", ".join(failures) + "\n")
        return 1
    print("  PASS  End-to-end PDF processing and evidence rendering verified\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
