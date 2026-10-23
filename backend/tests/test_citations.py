"""Offline tests for citation-or-drop. No API key, no model, no network.

    cd backend
    python tests/test_citations.py

Everything here is the part of the pipeline the model cannot influence, which is
exactly why it must be tested like this: deterministic, fast, and runnable at the
venue with no wifi.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.services.citations import normalize, verify  # noqa: E402

# ---------------------------------------------------------------- fixtures

PAGES = {
    1: """DISCHARGE SUMMARY — 12 Aug 2026
    Patient: 54F. Admitted 09 Aug with fever and dyspnoea.
    Vitals on admission: BP 148/92 mmHg, HR 104 bpm, SpO2 91% on room air, temp 38.6 C.
    Started on intravenous ceftriaxone 1 g twice daily.""",
    2: """INVESTIGATIONS
    HbA1c 8.4% (raised). Fasting glucose 168 mg/dL.
    Chest radiograph shows right lower zone consolidation consistent with pneumonia.
    Serum creatinine 1.1 mg/dL. Total leucocyte count 14,200 per microlitre.""",
    3: """PLAN
    Continue antibiotics for 5 days. Repeat chest radiograph in 6 weeks.
    Review in OPD after 1 week with repeat CBC.""",
}


class FakeChunk:
    """Stand-in for services.pdf_rag.PDFChunk, which pulls in torch and FAISS."""

    def __init__(self, text, page, box):
        self.text = text
        self.page_number = page
        self._box = box

    def to_dict(self):
        return {
            "text": self.text,
            "page_number": self.page_number,
            "coordinates": dict(zip(("x0", "y0", "x1", "y1"), self._box)),
        }


CHUNKS = [
    FakeChunk("Vitals on admission: BP 148/92 mmHg, HR 104 bpm, SpO2 91% on room air, temp 38.6 C.",
              1, (72, 310, 508, 336)),
    FakeChunk("HbA1c 8.4% (raised). Fasting glucose 168 mg/dL.", 2, (72, 120, 402, 146)),
]

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


# ---------------------------------------------------------------- tests

print("\ncitation-or-drop\n")

# 1. happy path
r = verify(
    [{"claim": "The patient was hypoxic on arrival.", "quote": "SpO2 91% on room air", "page": 1,
      "confidence": 0.91}],
    PAGES, CHUNKS,
)
c = r.claims[0]
check("exact line on the cited page is supported", c.status == "supported", c.reason)
check("confidence survives the round trip", c.confidence == 0.91)
check("a highlight box is attached", c.coords is not None and c.coords["y0"] == 310, str(c.coords))

# 2. normalisation: case, curly quotes, en dash, odd spacing
r = verify(
    [{"claim": "There is consolidation.", "quote": "  right   lower zone CONSOLIDATION  ", "page": 2}],
    PAGES, CHUNKS,
)
check("spacing and case do not break a real citation", r.claims[0].status == "supported", r.claims[0].reason)

r = verify([{"claim": "x", "quote": "DISCHARGE SUMMARY — 12 Aug 2026", "page": 1}], PAGES)
check("em dash inside the quote still matches", r.claims[0].status == "supported", r.claims[0].reason)

# 3. the model mis-numbered the page — corrected, not dropped
r = verify([{"claim": "Glucose was high.", "quote": "Fasting glucose 168 mg/dL", "page": 3}], PAGES, CHUNKS)
c = r.claims[0]
check("wrong page number is corrected, not dropped", c.status == "page_corrected", c.status)
check("the corrected page is page 2", c.page == 2, str(c.page))
check("a corrected claim still carries coords", c.coords is not None)
check("the reason names the real page", "page 2" in c.reason, c.reason)

# 4. fabricated quote — dropped
r = verify([{"claim": "The patient has diabetes.", "quote": "diagnosed with type 2 diabetes mellitus", "page": 2}], PAGES)
check("a quote that is not in the document is dropped", r.claims[0].status == "dropped", r.claims[0].status)

# 5. no quote offered
r = verify([{"claim": "Probably sepsis.", "quote": "", "page": 1}], PAGES)
check("a claim with no line at all is dropped", r.claims[0].reason == "no line offered", r.claims[0].reason)

# 6. quote too short to be evidence
r = verify([{"claim": "Fever.", "quote": "fever", "page": 1}], PAGES)
check("a two-word quote is not evidence", r.claims[0].reason == "quote too short to verify", r.claims[0].reason)

# 7. page given as a string (models do this constantly)
r = verify([{"claim": "Creatinine is normal.", "quote": "Serum creatinine 1.1 mg/dL", "page": "2"}], PAGES)
check("page '2' as a string is handled", r.claims[0].status == "supported", r.claims[0].status)

# 8. aggregate report
r = verify(
    [
        {"claim": "a", "quote": "SpO2 91% on room air", "page": 1},
        {"claim": "b", "quote": "Total leucocyte count 14,200", "page": 2},
        {"claim": "c", "quote": "started on vancomycin", "page": 1},
    ],
    PAGES, CHUNKS,
)
d = r.to_dict()
check("coverage counts only surviving claims", round(d["coverage"], 2) == 0.67, str(d["coverage"]))
check("headline reads like a sentence", d["headline"].startswith("2 of 3 claims cite a line"), d["headline"])
check("the dropped claim is named in the report", len(d["claims"]) == 3 and d["dropped"] == 1)

# 9. the empty case does not divide by zero
r = verify([], PAGES)
check("no claims is handled", r.coverage == 0.0 and r.headline == "no claims offered", r.headline)

# 10. normalisation is stable
check("normalize collapses whitespace", normalize("A  b\tc\n") == "a b c")
check("normalize folds curly apostrophes", normalize("patient\u2019s") == "patient's")
check("normalize folds curly double quotes", normalize("\u201chello\u201d") == '"hello"')
check("normalize handles en dash and em dash", normalize("fever \u2013 cough \u2014 dyspnoea") == "fever - cough - dyspnoea")

# 11. edge cases: repeated quotes across multiple pages
pages_multi = {
    1: "Patient admitted on 09 Aug 2026 for observation.",
    2: "No acute distress noted.",
    3: "Summary note: Patient admitted on 09 Aug 2026 for observation.",
}
r = verify([{"claim": "Admission date", "quote": "admitted on 09 Aug 2026", "page": 2}], pages_multi)
check("repeated quote on wrong page is corrected", r.claims[0].status == "page_corrected")
check("reason mentions multiple page occurrences", "appears on 2 pages" in r.claims[0].reason, r.claims[0].reason)

# 12. edge cases: malformed and missing page values
r = verify([{"claim": "Invalid page format", "quote": "SpO2 91% on room air", "page": "page 1"}], PAGES)
check("non-integer page string defaults safely and corrects", r.claims[0].status == "page_corrected" and r.claims[0].page == 1)

r = verify([{"claim": "None page provided", "quote": "SpO2 91% on room air", "page": None}], PAGES)
check("None page locates first occurrence", r.claims[0].status == "page_corrected" and r.claims[0].page == 1)

# 13. empty document and empty text
r = verify([{"claim": "Some claim", "quote": "SpO2 91% on room air", "page": 1}], {})
check("empty pages drops claim gracefully", r.claims[0].status == "dropped")

r = verify([{"claim": "Empty string quote", "quote": "   ", "page": 1}], PAGES)
check("whitespace-only quote is dropped as no line", r.claims[0].reason == "no line offered")

# 14. semantic limitation awareness (documenting that string matching proves quote presence, not clinical truth)
# An erroneous or contradictory claim citing a real verbatim line still matches the quote presence
r = verify([{"claim": "Patient had NO fever at all", "quote": "temp 38.6 C", "page": 1}], PAGES)
check("verifier checks quote presence rather than semantic truth", r.claims[0].status == "supported")

# ---------------------------------------------------------------- result

print(f"\n  {checks - len(failures)}/{checks} passed\n")
if failures:
    print("  failed: " + ", ".join(failures) + "\n")
    sys.exit(1)
print("  PASS  citation-or-drop is solid without a model\n")
