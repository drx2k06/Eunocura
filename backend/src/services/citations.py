"""Citation-or-drop.

The rule this project is built on: **a claim survives only if a line it quotes
actually exists on the page it cites.** No quote, no claim. Wrong page, we fix
the page or drop it. This is the part of the pipeline the model cannot fake,
because it never runs through the model at all — it is plain string work over
the PDF text we already extracted.

The model's only job is to *offer* claims with a quote and a page. This module
decides whether that offer is worth showing to a doctor.

Coordinates come from `services/pdf_rag.PDFChunk`, which already stores
x0/y0/x1/y1 per chunk, so a supported claim can be highlighted in the PDF viewer.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

DEFAULT_MIN_QUOTE = 8

_WS = re.compile(r"\s+")
_DASHES = dict.fromkeys(map(ord, "‐-‒–—―−"), "-")
_QUOTES = {
    ord("\u2018"): "'", ord("\u2019"): "'", ord("\u201a"): "'", ord("\u2032"): "'",
    ord("\u201c"): '"', ord("\u201d"): '"', ord("\u201e"): '"', ord("\u2033"): '"',
}


def normalize(text: str) -> str:
    """Case-, space- and punctuation-tolerant form used for every comparison."""
    t = unicodedata.normalize("NFKC", text or "")
    t = t.translate(_DASHES).translate(_QUOTES)
    t = t.replace("\u00a0", " ").replace("\u2009", " ")
    t = _WS.sub(" ", t)
    return t.strip().lower()


@dataclass
class Claim:
    """One thing the model wants to say, plus where it says it came from."""

    claim: str
    quote: str = ""
    page: Optional[int] = None
    confidence: Optional[float] = None
    kind: str = "finding"

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Claim":
        page = d.get("page", d.get("page_number"))
        try:
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = None
        conf = d.get("confidence")
        try:
            conf = float(conf) if conf is not None else None
        except (TypeError, ValueError):
            conf = None
        return cls(
            claim=str(d.get("claim", d.get("text", ""))).strip(),
            quote=str(d.get("quote", d.get("evidence", "")) or "").strip(),
            page=page,
            confidence=conf,
            kind=str(d.get("kind", "finding")),
        )


@dataclass
class VerifiedClaim:
    claim: str
    status: str  # supported | page_corrected | dropped
    reason: str
    page: Optional[int] = None
    quote: str = ""
    coords: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None
    kind: str = "finding"

    @property
    def kept(self) -> bool:
        return self.status in ("supported", "page_corrected")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "status": self.status,
            "reason": self.reason,
            "page": self.page,
            "quote": self.quote,
            "coords": self.coords,
            "confidence": self.confidence,
            "kind": self.kind,
        }


@dataclass
class CitationReport:
    claims: List[VerifiedClaim] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.claims)

    @property
    def kept(self) -> List[VerifiedClaim]:
        return [c for c in self.claims if c.kept]

    @property
    def dropped(self) -> List[VerifiedClaim]:
        return [c for c in self.claims if not c.kept]

    @property
    def corrected(self) -> List[VerifiedClaim]:
        return [c for c in self.claims if c.status == "page_corrected"]

    @property
    def coverage(self) -> float:
        return len(self.kept) / self.total if self.total else 0.0

    @property
    def headline(self) -> str:
        if not self.total:
            return "no claims offered"
        pct = round(self.coverage * 100)
        line = f"{len(self.kept)} of {self.total} claims cite a line that exists ({pct}%)"
        if self.corrected:
            line += f" · {len(self.corrected)} had the wrong page number, corrected"
        if self.dropped:
            line += f" · {len(self.dropped)} dropped for lack of a line"
        return line

    def to_dict(self) -> Dict[str, Any]:
        return {
            "headline": self.headline,
            "total": self.total,
            "supported": len(self.kept),
            "dropped": len(self.dropped),
            "page_corrected": len(self.corrected),
            "coverage": round(self.coverage, 4),
            "claims": [c.to_dict() for c in self.claims],
        }


# ---------------------------------------------------------------- chunks


def _chunk_page(chunk: Any) -> Optional[int]:
    page = getattr(chunk, "page_number", None)
    if page is None and isinstance(chunk, Mapping):
        page = chunk.get("page_number", chunk.get("page"))
    try:
        return int(page) if page is not None else None
    except (TypeError, ValueError):
        return None


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, Mapping):
        return str(chunk.get("text", ""))
    return str(getattr(chunk, "text", ""))


def _chunk_coords(chunk: Any) -> Optional[Dict[str, float]]:
    if hasattr(chunk, "to_dict"):
        coords = chunk.to_dict().get("coordinates")
        return coords if coords else None
    if isinstance(chunk, Mapping):
        return chunk.get("coordinates") or None
    return None


def _locate(quote_norm: str, page: Optional[int], chunks: Sequence[Any]) -> Optional[Dict[str, float]]:
    """Bounding box of the chunk that contains the quote, if we can find one."""
    if page is None:
        return None
    for chunk in chunks:
        if _chunk_page(chunk) != page:
            continue
        if quote_norm in normalize(_chunk_text(chunk)):
            return _chunk_coords(chunk)
    return None


# ---------------------------------------------------------------- verify


def verify(
    claims: Iterable[Any],
    pages: Mapping[Any, str],
    chunks: Sequence[Any] = (),
    *,
    min_quote: int = DEFAULT_MIN_QUOTE,
) -> CitationReport:
    """Check every offered claim against the document text.

    `pages` is {page_number: page_text}. `chunks` is anything with
    page_number/text/coordinates — used only to attach a highlight box.
    """
    page_index: Dict[int, str] = {}
    for k, v in pages.items():
        try:
            page_index[int(k)] = normalize(v)
        except (TypeError, ValueError):
            continue

    out: List[VerifiedClaim] = []

    for raw in claims:
        c = raw if isinstance(raw, Claim) else Claim.from_dict(raw)
        quote_norm = normalize(c.quote)

        if not c.claim:
            continue
        if not quote_norm:
            out.append(VerifiedClaim(c.claim, "dropped", "no line offered", c.page, c.quote, None, c.confidence, c.kind))
            continue
        if len(quote_norm) < min_quote:
            out.append(VerifiedClaim(c.claim, "dropped", "quote too short to verify", c.page, c.quote, None, c.confidence, c.kind))
            continue

        # 1. does the cited page contain the quote?
        if c.page is not None and c.page in page_index and quote_norm in page_index[c.page]:
            out.append(
                VerifiedClaim(c.claim, "supported", "line found on the cited page", c.page,
                              c.quote, _locate(quote_norm, c.page, chunks), c.confidence, c.kind)
            )
            continue

        # 2. it exists, but the model mis-numbered the page
        hits = sorted(p for p, text in page_index.items() if quote_norm in text)
        if hits:
            page = hits[0]
            reason = (
                f"line found on page {page}, not the cited page"
                if c.page is not None
                else f"line found on page {page}; no page was cited"
            )
            if len(hits) > 1:
                reason += f" (appears on {len(hits)} pages)"
            out.append(
                VerifiedClaim(c.claim, "page_corrected", reason, page, c.quote,
                              _locate(quote_norm, page, chunks), c.confidence, c.kind)
            )
            continue

        # 3. nothing to stand on
        out.append(
            VerifiedClaim(c.claim, "dropped", "this line does not appear anywhere in the document",
                          c.page, c.quote, None, c.confidence, c.kind)
        )

    return CitationReport(out)
