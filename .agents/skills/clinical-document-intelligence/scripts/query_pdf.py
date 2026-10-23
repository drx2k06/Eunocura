"""Ask one question of one clinical PDF and get the exact chunks back.

    python scripts/query_pdf.py --pdf discharge.pdf --ask "what was the creatinine?" --k 5
    python scripts/query_pdf.py --pdf discharge.pdf --ask "..." --render evidence.png

Retrieval only: nothing here calls a model. The chunks are returned with page
numbers and highlight coordinates so the harness's own model can answer from
them, and so a reviewer can see the lines the answer would be built from.

Where the index comes from
--------------------------
If a directory built by ``build_index.py`` exists next to the PDF
(``discharge.index/``), it is loaded. Otherwise the PDF is chunked and embedded
in memory for this one query, which needs the full chunking stack
(pdfplumber, faiss-cpu, langchain-community, fastapi) and takes longer. Use
``--index`` to point at a specific directory.

Similarity is cosine over unit-length MiniLM vectors: an exact dot product.
Embedding the question uses ``sentence-transformers/all-MiniLM-L6-v2``
directly; that is the model the repo's ``PDFRAGService`` wraps, and both sides
are L2-normalised, so the vectors share one space.

Exit codes
----------
0  results printed
1  bad input (pdf missing, unreadable index, no text in the PDF)
2  a required dependency is missing; the exact pip line is printed. Querying
   always needs ``sentence-transformers`` because the question has to be
   embedded; building an index in memory needs the rest of the stack too.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_index as bi  # noqa: E402 - sibling module, same directory

DEFAULT_ANSWER_MODEL = "ollama:qwen2.5:7b-instruct"

_embedder = None


# ------------------------------------------------------------------- model


def answer_model_label() -> str:
    """The model a harness would answer with. Never required for retrieval."""
    try:
        backend = bi.repo_backend()
        if str(backend) not in sys.path:
            sys.path.insert(0, str(backend))
        with bi.chatter_to_stderr():
            from src.llm.router import model_label

        return model_label("slm")
    except Exception:  # noqa: BLE001 - the label is cosmetic, never fatal
        return DEFAULT_ANSWER_MODEL


def embed_question(text: str):
    """Unit-length query vector from the same MiniLM model as the index."""
    global _embedder
    import numpy as np

    if _embedder is None:
        print(
            "loading sentence-transformers/all-MiniLM-L6-v2 "
            "(first run downloads ~80 MB)",
            file=sys.stderr,
        )
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(bi.EMBEDDING_MODEL)
    vector = _embedder.encode(
        [text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )[0]
    return np.asarray(vector, dtype="float32")


# ------------------------------------------------------------------ render


def render_evidence(pdf_path: Path, result: Dict[str, Any], out_path: Path) -> Optional[str]:
    """Highlight the winning chunk's box with src.services.bbox.

    ``src.services.bbox.render_highlight(pdf_path, page_number, rects,
    out_path, pad=2.0)`` takes rects in PDF coordinates (bottom-left origin),
    which is exactly what ``PDFChunk.to_dict()["coordinates"]`` stores. The
    module is imported here, not at the top, so retrieval still works when the
    renderer is unavailable.
    """
    try:
        backend = bi.repo_backend()
        if str(backend) not in sys.path:
            sys.path.insert(0, str(backend))
        from src.services.bbox import render_highlight
    except Exception as exc:  # noqa: BLE001 - continue without the PNG
        print(
            "render skipped: cannot import src.services.bbox "
            f"({exc.__class__.__name__}: {exc})",
            file=sys.stderr,
        )
        return None

    coords = result.get("coordinates") or {}
    page = result.get("page_number")
    if not coords or page is None:
        print("render skipped: this result carries no coordinates", file=sys.stderr)
        return None

    rect = {
        "x0": float(coords.get("x0", 0.0)),
        "y0": float(coords.get("y0", 0.0)),
        "x1": float(coords.get("x1", 0.0)),
        "y1": float(coords.get("y1", 0.0)),
    }
    try:
        written = render_highlight(str(pdf_path), int(page), [rect], str(out_path), pad=2.0)
    except Exception as exc:  # noqa: BLE001 - continue without the PNG
        print(
            f"render skipped: render_highlight failed ({exc.__class__.__name__}: {exc})",
            file=sys.stderr,
        )
        return None
    return str(written or out_path)


# -------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="query_pdf.py",
        description=(
            "Retrieve the chunks of a clinical PDF that answer a question. "
            "No model is called; page numbers and boxes come back with each chunk."
        ),
        epilog=(
            "exit codes: 0 results printed, 1 bad input, 2 missing dependency"
        ),
    )
    parser.add_argument("--pdf", required=True, help="path to the clinical PDF")
    parser.add_argument("--ask", required=True, help="the question to ask of the document")
    parser.add_argument(
        "--k", type=int, default=5, help="how many chunks to return (default: 5)"
    )
    parser.add_argument(
        "--index",
        default=None,
        help="index directory from build_index.py (default: <pdf name>.index next to the PDF)",
    )
    parser.add_argument(
        "--render",
        default=None,
        help="write a PNG of the top result's box on its page to this path",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    bi.utf8_streams()
    args = build_parser().parse_args(argv)

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.is_file():
        print(f"pdf not found: {pdf_path}", file=sys.stderr)
        return 1

    status = bi.dependency_failure(bi.QUERY_MODULES)
    if status:
        print(
            "the question has to be embedded, so sentence-transformers is "
            "required even when the index already exists.",
            file=sys.stderr,
        )
        return status

    index_dir = (
        Path(args.index).expanduser().resolve()
        if args.index
        else bi.default_index_dir(pdf_path)
    )

    chunks: List[Dict[str, Any]]
    matrix: Any
    source: str

    if (index_dir / "chunks.json").is_file():
        try:
            meta, chunks, matrix = bi.load_index(index_dir)
        except Exception as exc:  # noqa: BLE001 - corrupt index is a bad input
            print(f"could not read {index_dir}: {exc}", file=sys.stderr)
            return 1
        source = str(index_dir)
        indexed_pdf = str(meta.get("pdf", "")).strip()
        if indexed_pdf and Path(indexed_pdf).resolve() != pdf_path:
            print(
                f"warning: {index_dir} was built from {indexed_pdf}, not {pdf_path}",
                file=sys.stderr,
            )
    else:
        missing = bi.missing_dependencies()
        if missing:
            print(
                f"no index at {index_dir}, so the PDF has to be chunked in memory, "
                "which needs the full chunking stack:\n" + bi.install_hint(missing) + "\n"
                f"faster: python scripts/build_index.py --pdf \"{pdf_path}\" --out \"{index_dir}\"",
                file=sys.stderr,
            )
            return 2
        started_chunks = bi.chunk_pdf(pdf_path)
        chunks, service = started_chunks[0], started_chunks[2]
        if not chunks:
            print(f"no extractable text in {pdf_path.name}", file=sys.stderr)
            return 1
        matrix, _ = bi.embed_texts([c["text"] for c in chunks], service)
        source = f"in-memory ({len(chunks)} chunks; no index written)"

    query_vector = embed_question(args.ask)
    results = bi.search(chunks, matrix, query_vector, args.k)

    payload: Dict[str, Any] = {
        "query": args.ask,
        "model": answer_model_label(),
        "results": results,
        "index": {
            "source": source,
            "chunks": len(chunks),
            "similarity": "cosine (unit-length MiniLM vectors, exact dot product)",
        },
    }

    if args.render:
        rendered = render_evidence(pdf_path, results[0], Path(args.render).expanduser().resolve()) if results else None
        if rendered:
            payload["render"] = rendered
            payload["render_note"] = "box of the top result on its own page"

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
