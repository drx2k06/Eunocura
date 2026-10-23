"""Generate assets/sample-discharge-summary.pdf — run once, the PDF is committed.

    python assets/make_sample.py

The PDF is a two-page synthetic discharge summary for a patient who does not
exist ("Asha Rao", MRN SYN-2026-0147). It exists so the skill can be run
end-to-end with no real patient data and so the citation verifier has lines
that either exist on a page or do not.

Only PyMuPDF (``fitz``) is needed, and only if you want to regenerate the file.
The committed PDF is the deliverable; this script is kept so a reviewer can see
exactly what is in it and can rebuild it byte-for-byte.

After writing, the script re-opens the PDF and checks that every intended line
is extractable as its own line, because a quote that cannot be extracted cannot
be verified. Exit code 0 means the file is good, 1 means a line was mangled.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "sample-discharge-summary.pdf"

PAGE_WIDTH, PAGE_HEIGHT = 595, 842  # A4 points
MARGIN = 72
MAX_WIDTH = PAGE_WIDTH - 2 * MARGIN
BODY_SIZE = 10.5

# ("style", "text"): style is title | sub | heading | body | gap
PAGES: list[list[tuple[str, str]]] = [
    [
        ("title", "EUNOCURA GENERAL HOSPITAL"),
        ("sub", "Department of Internal Medicine"),
        ("gap", ""),
        ("heading", "DISCHARGE SUMMARY"),
        ("body", "Patient: Asha Rao"),
        ("body", "MRN: SYN-2026-0147"),
        ("body", "Age / Sex: 54 years / Female"),
        ("body", "Admitted: 09 Aug 2026    Discharged: 12 Aug 2026"),
        ("body", "Ward: Medical ICU, Bed 6"),
        ("gap", ""),
        ("heading", "REASON FOR ADMISSION"),
        ("body", "Fever with cough and breathlessness for four days."),
        ("gap", ""),
        ("heading", "VITALS ON ADMISSION"),
        ("body", "BP 148/92 mmHg, HR 104 bpm, SpO2 91% on room air, temperature 38.6 C."),
        ("body", "After 2 L/min oxygen by nasal prongs: SpO2 96%, HR 98 bpm."),
        ("gap", ""),
        ("heading", "EXAMINATION"),
        ("body", "Bilateral basal crepitations, right more than left. No pedal oedema."),
        ("gap", ""),
        ("heading", "INVESTIGATIONS"),
        ("body", "HbA1c 8.4% (raised)."),
        ("body", "Fasting plasma glucose 168 mg/dL."),
        ("body", "Serum creatinine 1.1 mg/dL (within reference range)."),
        ("body", "Total leucocyte count 14,200 per microlitre (raised)."),
        ("body", "Chest radiograph: right lower zone consolidation consistent with pneumonia."),
        ("gap", ""),
        ("heading", "COURSE IN HOSPITAL"),
        ("body", "Started on intravenous ceftriaxone 1 g twice daily on 09 Aug 2026."),
        ("body", "Afebrile from 10 Aug 2026. Oxygen discontinued on 11 Aug 2026."),
    ],
    [
        ("heading", "MEDICATIONS ON DISCHARGE"),
        ("body", "Inj ceftriaxone 1 g intravenous twice daily for 5 days."),
        ("body", "Tab paracetamol 650 mg oral every 8 hours as needed for fever."),
        ("body", "Tab metformin 500 mg oral twice daily."),
        ("gap", ""),
        ("heading", "DIAGNOSIS"),
        ("body", "Right lower zone consolidation consistent with pneumonia."),
        ("body", "Type 2 diabetes mellitus, suboptimally controlled."),
        ("gap", ""),
        ("heading", "FOLLOW-UP PLAN"),
        ("body", "Review in medicine OPD after 1 week with repeat CBC and creatinine."),
        ("body", "Repeat chest radiograph in 6 weeks."),
        ("body", "Return earlier if breathlessness, fever above 38 C, or SpO2 below 92%."),
        ("gap", ""),
        ("heading", "DISCHARGE CONDITION"),
        ("body", "Afebrile for 48 hours, SpO2 96% on room air, ambulating independently."),
        ("gap", ""),
        ("body", "Prepared by: Dr. N. Menon, Internal Medicine"),
        ("body", "Synthetic document for testing. No real patient data."),
    ],
]

STYLES = {
    "title": ("hebo", 15.0, 21.0),
    "sub": ("helv", 9.0, 14.0),
    "heading": ("hebo", 11.0, 20.0),
    "body": ("helv", BODY_SIZE, 15.0),
}


def fit(text: str, fontname: str, size: float) -> float:
    """Shrink the size until the line fits; never wrap, so one line stays one line."""
    import fitz

    while size > 7.0 and fitz.get_text_length(text, fontname=fontname, fontsize=size) > MAX_WIDTH:
        size -= 0.5
    return size


def build() -> list[str]:
    import fitz

    doc = fitz.open()
    expected: list[str] = []

    for lines in PAGES:
        page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        y = MARGIN
        for style, text in lines:
            if style == "gap":
                y += 8
                continue
            fontname, size, leading = STYLES[style]
            size = fit(text, fontname, size)
            page.insert_text((MARGIN, y), text, fontname=fontname, fontsize=size, color=(0, 0, 0))
            expected.append(text)
            y += leading

    doc.set_metadata(
        {
            "title": "Discharge summary (synthetic)",
            "author": "EUNOCURA sample generator",
            "subject": "Synthetic clinical document for the clinical-document-intelligence skill",
            "keywords": "synthetic, no real patient data",
            "creator": "assets/make_sample.py",
            "producer": "PyMuPDF",
        }
    )
    doc.save(str(OUT), deflate=True, garbage=3, pretty=False)
    doc.close()
    return expected


def check(expected: list[str]) -> list[str]:
    """Every intended line must come back as its own extractable line."""
    import fitz

    with fitz.open(str(OUT)) as doc:
        found = {line.strip() for page in doc for line in page.get_text().splitlines()}
    return [line for line in expected if line not in found]


def main() -> int:
    try:
        import fitz  # noqa: F401
    except ImportError:
        print("PyMuPDF is required to regenerate the sample: pip install pymupdf", file=sys.stderr)
        return 1

    expected = build()
    if not OUT.read_bytes().startswith(b"%PDF-"):
        print(f"{OUT} does not start with %PDF-", file=sys.stderr)
        return 1

    missing = check(expected)
    size = OUT.stat().st_size
    if missing:
        print(f"{OUT.name} written ({size} bytes) but {len(missing)} line(s) are not extractable:", file=sys.stderr)
        for line in missing:
            print(f"  {line}", file=sys.stderr)
        return 1

    print(f"{OUT} ({size} bytes, {len(expected)} lines, all extractable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
