"""Extract clinical facts from one PDF, then verify every claim it makes.

    python scripts/extract_clinical.py --pdf discharge.pdf --json out.json
    python scripts/extract_clinical.py --pdf discharge.pdf --profile med
    python scripts/extract_clinical.py --pdf discharge.pdf --claims saved.json

Two halves, and only the first one needs a model:

1. **Offer** — page text from pdfplumber goes to ``src.llm.router.ask_json``
   once, with the JSON Schema from ``references/extraction-schema.md``. The
   model returns structured facts plus a ``claims`` array. Default profile is
   ``slm`` (``qwen2.5:7b-instruct``); ``--profile med`` uses ``medgemma:4b``.
   No API key is involved: the router talks to Ollama on
   ``http://localhost:11434/v1``. Every call is cached on disk by the router,
   so a repeated run replays the same answer without the server.
2. **Verify** — ``src.services.citations.verify(claims, pages, chunks)``
   checks each claim by plain string comparison: the quote must occur on the
   page it cites. Supported, page-corrected and dropped claims all appear in
   the report. Nothing is repaired by the model and nothing is dropped
   silently; the report is the output.

``--claims`` runs only step 2 against a claims file you already have (a list,
or an object with a ``claims`` list). That makes the deterministic half
demoable with no model, no Ollama, no network.

Chunks from an index built by ``build_index.py`` (``<pdf name>.index/`` next to
the PDF, or ``--index``) are attached so supported claims carry a highlight
box. Without them the report is identical except ``coords`` is ``null``.

Exit codes
----------
0  report printed
1  bad input or an answer we cannot use (missing pdf, no text, no claim list,
   unparseable model output)
2  ``pdfplumber`` is missing; the exact pip line is printed
3  the model layer is unavailable (router not installed, provider switched
   off, Ollama not reachable, or the call failed). The hint is printed and the
   ``--claims`` path is named, because verification needs no model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_index as bi  # noqa: E402 - shares repo-path and stream helpers

SKILL_DIR = bi.SKILL_DIR
SCHEMA_PATH = SKILL_DIR / "references" / "extraction-schema.md"

PDFPLUMBER_MODULES: Tuple[Tuple[str, str], ...] = (("pdfplumber", "pdfplumber"),)

DEFAULT_MAX_CHARS = 48000

PROMPT_TEMPLATE = """\
You are reading a clinical document for a reviewer who will check every line you quote.

RULES
1. quote must be copied verbatim from the page it cites: the same words, digits and
   punctuation that appear there. Do not paraphrase, expand an abbreviation, or join
   two separate lines into one quote.
2. page is the 1-based number in the "=== PAGE n ===" header above that text. Never
   infer a page from the content.
3. Claims are atomic. One fact per claim. "The patient was hypoxic and started on
   ceftriaxone" is two claims, because no single line proves both.
4. Never invent a number and never compute one. No unit conversion, no totals, no
   averages, no "likely" values. Copy what is printed.
5. If a field is not stated, use null, or leave the array empty. An absent fact is a
   correct answer; a guessed fact is a dropped claim.
6. Extract only what this document says. No outside medical knowledge, no diagnosis
   the document does not name.

DOCUMENT
{pages}

Return the JSON object now.
"""


# ------------------------------------------------------------------ inputs


def load_schema() -> Dict[str, Any]:
    """The JSON Schema, read from the marked block in references/extraction-schema.md."""
    if not SCHEMA_PATH.is_file():
        raise FileNotFoundError(
            f"schema file not found: {SCHEMA_PATH}. The skill needs its references/ directory."
        )
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    start = text.find("<!-- schema:start -->")
    end = text.find("<!-- schema:end -->")
    block = text[start:end] if start != -1 and end > start else text
    fence = re.search(r"```json\s*(.*?)```", block, re.DOTALL)
    if not fence:
        raise ValueError(f"no ```json block in {SCHEMA_PATH}")
    return json.loads(fence.group(1))


def read_pages(pdf_path: Path) -> Dict[int, str]:
    """{1-based page number: page text}. Pages without text are kept as ''."""
    import pdfplumber

    pages: Dict[int, str] = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            pages[number] = page.extract_text() or ""
    return pages


def load_chunks(index_dir: Optional[Path]) -> List[Mapping[str, Any]]:
    """Chunk dicts for highlight boxes; empty when no index is around."""
    if index_dir is None:
        return []
    chunks_path = index_dir / "chunks.json"
    if not chunks_path.is_file():
        return []
    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    return list(payload.get("chunks", []))


def document_text(pages: Mapping[int, str], max_chars: int) -> Tuple[str, bool]:
    """Page headers plus text, stopping at max_chars. Returns (text, truncated)."""
    blocks: List[str] = []
    used = 0
    truncated = False
    for number in sorted(pages):
        block = f"\n=== PAGE {number} ===\n{pages[number].strip()}\n"
        if used + len(block) > max_chars:
            truncated = True
            remaining = max_chars - used
            if remaining > 200:
                blocks.append(block[:remaining] + "\n[truncated]\n")
            break
        blocks.append(block)
        used += len(block)
    return "".join(blocks), truncated


def read_claims_file(path: Path) -> List[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if isinstance(payload.get("claims"), list):
            return payload["claims"]
        raise ValueError(
            f"{path} is an object without a 'claims' list "
            f"(keys: {', '.join(sorted(payload)) or 'none'})"
        )
    if isinstance(payload, list):
        return payload
    raise ValueError(f"{path} must contain a list of claims or an object with a 'claims' list")


# -------------------------------------------------------------------- model


def import_router():
    """Import the repo router; returns (ask_json, model_label, health, ModelUnavailable)."""
    backend = bi.repo_backend()
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    with bi.chatter_to_stderr():
        from src.llm.router import ModelUnavailable, ask_json, health, model_label

    return ask_json, model_label, health, ModelUnavailable


def model_layer_failure(detail: str, hint: str = "") -> int:
    print(f"model unavailable: {detail}", file=sys.stderr)
    if hint:
        print(hint, file=sys.stderr)
    print(
        "verification still runs with no model at all:\n"
        "  python scripts/extract_clinical.py --pdf <file.pdf> --claims <claims.json>",
        file=sys.stderr,
    )
    return 3


# -------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract_clinical.py",
        description=(
            "Extract clinical facts from a PDF with a local open-weight model, "
            "then verify every claim's quote against the page it cites."
        ),
        epilog=(
            "exit codes: 0 report printed, 1 bad input or unusable answer, "
            "2 missing pdfplumber, 3 model unavailable (use --claims instead)"
        ),
    )
    parser.add_argument("--pdf", required=True, help="path to the clinical PDF")
    parser.add_argument(
        "--json", default=None, help="also write the full result to this file"
    )
    parser.add_argument(
        "--claims",
        default=None,
        help="skip the model and verify a saved claims file (list, or {claims: [...]})",
    )
    parser.add_argument(
        "--profile",
        choices=("slm", "med"),
        default="slm",
        help="router profile: slm = qwen2.5:7b-instruct (default), med = medgemma:4b",
    )
    parser.add_argument(
        "--index",
        default=None,
        help="index directory from build_index.py, for highlight coordinates "
        "(default: <pdf name>.index next to the PDF, when it exists)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=f"cap on document characters sent to the model (default: {DEFAULT_MAX_CHARS})",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="do not reuse the router's on-disk answer cache",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    bi.utf8_streams()
    args = build_parser().parse_args(argv)

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.is_file():
        print(f"pdf not found: {pdf_path}", file=sys.stderr)
        return 1

    status = bi.dependency_failure(PDFPLUMBER_MODULES)
    if status:
        print(
            "pdfplumber reads the page text; it is needed for the model half and "
            "for the verifier, which compares quotes against page text.",
            file=sys.stderr,
        )
        return status

    # Verify needs the citation module only; import it early so --claims works
    # even when the model layer is not installed at all.
    try:
        backend = bi.repo_backend()
        if str(backend) not in sys.path:
            sys.path.insert(0, str(backend))
        from src.services.citations import verify
    except Exception as exc:  # noqa: BLE001 - repo checkout problem
        print(
            f"cannot import src.services.citations: {exc.__class__.__name__}: {exc}\n"
            "Run this script from a checkout of the EUNOCURA repo, or set "
            "EUNOCURA_BACKEND to its backend/ directory.",
            file=sys.stderr,
        )
        return 1

    try:
        pages = read_pages(pdf_path)
    except Exception as exc:  # noqa: BLE001 - unreadable pdf
        print(f"could not read {pdf_path.name}: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 1

    if not any(text.strip() for text in pages.values()):
        print(
            f"no extractable text in {pdf_path.name}. This skill reads a text layer; "
            "scanned pages need OCR first.",
            file=sys.stderr,
        )
        return 1

    index_dir = (
        Path(args.index).expanduser().resolve()
        if args.index
        else bi.default_index_dir(pdf_path)
    )
    try:
        chunks = load_chunks(index_dir)
    except Exception as exc:  # noqa: BLE001 - a broken index must not stop verification
        print(
            f"warning: ignoring {index_dir} ({exc.__class__.__name__}: {exc}); "
            "claims will verify without highlight boxes",
            file=sys.stderr,
        )
        chunks = []
    if chunks:
        print(f"using {len(chunks)} chunks from {index_dir} for highlight boxes", file=sys.stderr)

    # ------------------------------------------------------------- offer
    extraction: Dict[str, Any] = {}
    label = "deterministic verification only (no model called)"

    if args.claims:
        claims_path = Path(args.claims).expanduser().resolve()
        if not claims_path.is_file():
            print(f"claims file not found: {claims_path}", file=sys.stderr)
            return 1
        try:
            offered = read_claims_file(claims_path)
        except Exception as exc:  # noqa: BLE001 - bad input file
            print(f"could not read {claims_path}: {exc}", file=sys.stderr)
            return 1
        source = str(claims_path)
    else:
        try:
            ask_json, model_label, health, ModelUnavailable = import_router()
        except Exception as exc:  # noqa: BLE001 - router package missing
            return model_layer_failure(
                f"cannot import src.llm.router ({exc.__class__.__name__}: {exc})",
                "install the backend requirements: pip install -r backend/requirements.txt",
            )

        status = health(timeout=2.0)
        if not status.get("reachable"):
            return model_layer_failure(
                f"{status.get('provider')} is not reachable",
                status.get("hint", ""),
            )

        try:
            schema = load_schema()
        except Exception as exc:  # noqa: BLE001 - a broken reference file is a setup bug
            print(f"cannot load the extraction schema: {exc}", file=sys.stderr)
            return 1

        text, truncated = document_text(pages, args.max_chars)
        if truncated:
            print(
                f"warning: document text cut at {args.max_chars} characters; the tail "
                "of the PDF was not shown to the model",
                file=sys.stderr,
            )

        label = model_label(args.profile)
        prompt = PROMPT_TEMPLATE.format(pages=text)
        try:
            extraction = ask_json(
                prompt,
                schema,
                profile=args.profile,
                temperature=0.0,
                cache=not args.no_cache,
            )
        except ModelUnavailable as exc:
            return model_layer_failure(str(exc))
        except Exception as exc:  # noqa: BLE001 - model answered, unusably
            print(f"the model call failed: {exc.__class__.__name__}: {exc}", file=sys.stderr)
            return 1

        offered = extraction.get("claims")
        if offered is None:
            print(
                "the model returned JSON without a 'claims' array; nothing to verify",
                file=sys.stderr,
            )
            return 1
        if not isinstance(offered, list):
            print(
                f"the model returned 'claims' as {type(offered).__name__}, not a list",
                file=sys.stderr,
            )
            return 1
        source = f"{label} on {pdf_path.name}"

    # ------------------------------------------------------------ verify
    report = verify(offered, pages, chunks)

    payload: Dict[str, Any] = {
        "pdf": str(pdf_path),
        "model": label,
        "pages": len(pages),
        "source": source,
        "index": str(index_dir) if chunks else None,
        "extraction": extraction,
    }
    payload.update(report.to_dict())

    print(report.headline, file=sys.stderr)
    text_out = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text_out)

    if args.json:
        out_path = Path(args.json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text_out + "\n", encoding="utf-8")
        print(f"wrote {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
