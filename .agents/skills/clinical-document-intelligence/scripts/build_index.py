"""Build a searchable index over one clinical PDF.

    python scripts/build_index.py --pdf discharge.pdf --out discharge.index

What this actually does
-----------------------
1. Chunks the PDF with the application's own service, ``src.services.pdf_rag``
   (``PDFRAGService.process_pdf``). Using the service matters: chunk
   boundaries, page numbers and highlight coordinates are then identical to
   what the EUNOCURA API produces for the same file.
2. Embeds every chunk with that service's
   ``sentence-transformers/all-MiniLM-L6-v2`` embeddings object, L2-normalised.
3. Writes into ``--out``:

   ==============  =========================================================
   chunks.json     text, page_number, coordinates, chunk_index per chunk
   embeddings.npz  float32 matrix, one row per chunk, unit length
   index.faiss     IndexFlatIP over the same rows (only when faiss imports)
   meta.json       pdf path, model, dim, counts, timestamps
   ==============  =========================================================

Honest note on persistence
--------------------------
``PDFRAGService`` keeps its vector store in memory and exposes no save/load
method, so this script does not pretend to persist a service-owned FAISS
store. It persists the embedding matrix itself. Querying is an exact
inner-product search over that matrix, which is numerically what FAISS
``IndexFlatIP`` computes; the binary ``index.faiss`` is written as a
convenience when ``faiss-cpu`` is installed but nothing in this skill needs it
at query time.

Importing ``pdf_rag`` constructs its module-level service, which downloads the
~80 MB MiniLM model on first use. The service logs with emoji to stdout, so
that chatter is redirected to stderr and stdout stays machine-readable JSON.

Exit codes
----------
0  index written
1  bad input (pdf missing/unreadable, no extractable text, --out exists)
2  a required dependency is missing; the exact pip line is printed, never a
   raw traceback
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SKILL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# (import name, pip name). langchain_text_splitters is checked separately
# because pdf_rag falls back to the older langchain.text_splitter module.
REQUIRED_MODULES: Tuple[Tuple[str, str], ...] = (
    ("pdfplumber", "pdfplumber"),
    ("sentence_transformers", "sentence-transformers"),
    ("faiss", "faiss-cpu"),
    ("langchain_community", "langchain-community"),
    ("fastapi", "fastapi"),
    ("numpy", "numpy"),
)

# What --render and querying an existing index need, without the chunking stack.
QUERY_MODULES: Tuple[Tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("sentence_transformers", "sentence-transformers"),
)


# --------------------------------------------------------------------- setup


def utf8_streams() -> None:
    """Windows consoles default to a legacy code page; PDF text is not ASCII."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - never fail because of a stream
            pass


@contextlib.contextmanager
def chatter_to_stderr():
    """Keep third-party logging out of the JSON we print on stdout."""
    with contextlib.redirect_stdout(sys.stderr):
        yield


def repo_backend() -> Path:
    """The ``backend/`` directory that contains ``src.services.pdf_rag``."""
    candidates: List[Path] = []
    override = os.environ.get("EUNOCURA_BACKEND")
    if override:
        candidates.append(Path(override).expanduser())
    candidates.append(REPO_ROOT / "backend")
    env_repo = os.environ.get("EUNOCURA_REPO")
    if env_repo:
        candidates.append(Path(env_repo).expanduser() / "backend")

    for candidate in candidates:
        if (candidate / "src" / "services" / "pdf_rag.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "cannot find backend/src/services/pdf_rag.py. Run this script from a "
        "checkout of the EUNOCURA repo, or set EUNOCURA_BACKEND to its "
        "backend/ directory."
    )


# ------------------------------------------------------------------- deps


def _spec_present(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def missing_dependencies(modules: Sequence[Tuple[str, str]] = REQUIRED_MODULES) -> List[str]:
    """pip names of the modules that are not importable right now."""
    missing = [package for module, package in modules if not _spec_present(module)]
    if modules is REQUIRED_MODULES:
        if not _spec_present("langchain_text_splitters") and not _spec_present("langchain"):
            missing.append("langchain-text-splitters")
    return missing


def install_hint(missing: Sequence[str]) -> str:
    packages = sorted(set(missing))
    joined = " ".join(packages)
    return (
        f"missing: {', '.join(packages)}\n"
        f"install: pip install {joined}\n"
        f"or, from the repo root: pip install -r backend/requirements.txt"
    )


def dependency_failure(modules: Sequence[Tuple[str, str]] = REQUIRED_MODULES) -> int:
    missing = missing_dependencies(modules)
    if not missing:
        return 0
    print(install_hint(missing), file=sys.stderr)
    return 2


# ---------------------------------------------------------------- chunking


class _PathUpload:
    """The bare minimum of ``fastapi.UploadFile`` that ``process_pdf`` uses."""

    def __init__(self, path: Path) -> None:
        self.filename = path.name
        self._path = path

    async def read(self) -> bytes:
        return self._path.read_bytes()


def load_pdf_rag():
    """Import the repo service with its ``backend/`` directory on sys.path."""
    backend = repo_backend()
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    with chatter_to_stderr():
        from src.services import pdf_rag  # noqa: WPS433 - deliberate late import

    return pdf_rag


def _service(pdf_rag_module: Any) -> Any:
    service = getattr(pdf_rag_module, "pdf_rag_service", None)
    if service is None:
        service = pdf_rag_module.PDFRAGService()
    return service


def chunk_pdf(
    pdf_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Any]:
    """Chunk one PDF through ``PDFRAGService``.

    Returns ``(chunks, summary, service)``. The live service is handed back
    because it already holds the embedding model; building a second one would
    download and load MiniLM twice.
    """
    pdf_rag_module = load_pdf_rag()
    service = _service(pdf_rag_module)
    document_id = f"skill:{pdf_path.stem}"

    with chatter_to_stderr():
        summary = asyncio.run(service.process_pdf(_PathUpload(pdf_path), document_id))
        stored = service.chunk_stores[document_id]

    chunks = [chunk.to_dict() for chunk in stored]
    return chunks, dict(summary), service


# --------------------------------------------------------------- embedding


def embed_texts(
    texts: Sequence[str], service: Any = None
) -> Tuple[Any, str]:
    """Embed chunk texts; returns (float32 matrix, how it was produced).

    ``service`` is the live ``PDFRAGService``; its ``embeddings`` object is the
    canonical MiniLM wrapper the API uses. Without a service we call
    sentence-transformers directly with the same model and the same
    normalisation, so both routes land in the same vector space.
    """
    import numpy as np

    texts = list(texts)
    if service is not None:
        vectors = service.embeddings.embed_documents(texts)
        label = f"src.services.pdf_rag.embeddings ({EMBEDDING_MODEL})"
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
        vectors = model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        label = f"sentence-transformers ({EMBEDDING_MODEL})"

    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim == 1:  # a single chunk
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms, label


def embed_query(text: str, service: Any = None):
    """One unit-length query vector, produced the same way as the chunks."""
    import numpy as np

    if service is not None:
        vector = service.embeddings.embed_query(text)
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
        vector = model.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0]
    return np.asarray(vector, dtype="float32")


# ------------------------------------------------------------ persistence


def default_index_dir(pdf_path: Path) -> Path:
    """``discharge.pdf`` -> ``discharge.index/`` next to the file."""
    return Path(str(pdf_path.with_suffix("")) + ".index")


def save_index(
    out_dir: Path,
    pdf_path: Path,
    chunks: Sequence[Dict[str, Any]],
    matrix: Any,
    embedder_label: str,
) -> Dict[str, Any]:
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = out_dir / "chunks.json"
    chunks_path.write_text(
        json.dumps(
            {
                "pdf": str(pdf_path),
                "chunk_count": len(chunks),
                "chunks": list(chunks),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    matrix_path = out_dir / "embeddings.npz"
    np.savez_compressed(matrix_path, embeddings=matrix)

    faiss_written = False
    faiss_note = "not written"
    try:
        import faiss

        index = faiss.IndexFlatIP(int(matrix.shape[1]))
        index.add(np.ascontiguousarray(matrix))
        faiss.write_index(index, str(out_dir / "index.faiss"))
        faiss_written = True
        faiss_note = "index.faiss (IndexFlatIP, cosine after unit normalisation)"
    except Exception as exc:  # noqa: BLE001 - faiss is optional at query time
        faiss_note = f"not written ({exc.__class__.__name__}: {exc})"

    meta = {
        "pdf": str(pdf_path),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "embedding_model": EMBEDDING_MODEL,
        "embedder": embedder_label,
        "dim": int(matrix.shape[1]),
        "chunks": len(chunks),
        "faiss_index": faiss_note,
        "notes": (
            "PDFRAGService keeps its vector store in memory and exposes no "
            "save method, so this index is the persisted embedding matrix. "
            "Querying does an exact inner-product search over it."
        ),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "out": str(out_dir),
        "chunks": len(chunks),
        "dim": int(matrix.shape[1]),
        "files": {
            "chunks": str(chunks_path),
            "embeddings": str(matrix_path),
            "faiss": str(out_dir / "index.faiss") if faiss_written else None,
            "meta": str(out_dir / "meta.json"),
        },
    }


def load_index(index_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Any]:
    """Read ``chunks.json`` + ``embeddings.npz`` written by this script."""
    import numpy as np

    chunks_path = index_dir / "chunks.json"
    matrix_path = index_dir / "embeddings.npz"
    if not chunks_path.is_file() or not matrix_path.is_file():
        raise FileNotFoundError(
            f"{index_dir} does not look like an index built by build_index.py "
            "(expected chunks.json and embeddings.npz)"
        )
    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    matrix = np.load(matrix_path)["embeddings"].astype("float32")
    meta = {}
    meta_path = index_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta, chunks, matrix


def search(
    chunks: Sequence[Dict[str, Any]], matrix: Any, query_vector: Any, k: int
) -> List[Dict[str, Any]]:
    """Top-k by cosine similarity (vectors are already unit length)."""
    import numpy as np

    if not chunks or matrix.shape[0] == 0:
        return []
    scores = matrix @ np.asarray(query_vector, dtype="float32")
    order = np.argsort(-scores)[: max(1, min(k, len(chunks)))]
    results = []
    for position in order:
        chunk = chunks[int(position)]
        results.append(
            {
                "text": chunk.get("text", ""),
                "page_number": chunk.get("page_number"),
                "coordinates": chunk.get("coordinates"),
                "score": round(float(scores[int(position)]), 4),
                "chunk_index": chunk.get("chunk_index"),
            }
        )
    return results


# -------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_index.py",
        description=(
            "Chunk a clinical PDF with the repo's pdf_rag service and persist "
            "an embedding index (chunks.json + embeddings.npz)."
        ),
        epilog=(
            "exit codes: 0 index written, 1 bad input, "
            "2 missing dependency (pip line printed)"
        ),
    )
    parser.add_argument("--pdf", required=True, help="path to the clinical PDF")
    parser.add_argument(
        "--out",
        default=None,
        help="index directory (default: <pdf name>.index next to the PDF)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing non-empty --out directory",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not print the JSON summary on success",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    utf8_streams()
    args = build_parser().parse_args(argv)

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.is_file():
        print(f"pdf not found: {pdf_path}", file=sys.stderr)
        return 1
    if pdf_path.suffix.lower() != ".pdf":
        print(f"not a .pdf file: {pdf_path}", file=sys.stderr)
        return 1

    status = dependency_failure()
    if status:
        return status

    try:
        repo_backend()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else default_index_dir(pdf_path)
    )
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        print(
            f"{out_dir} already exists and is not empty (pass --force to overwrite)",
            file=sys.stderr,
        )
        return 1

    started = time.time()
    try:
        chunks, summary, service = chunk_pdf(pdf_path)
    except ValueError as exc:
        print(
            f"no extractable text in {pdf_path.name}: {exc}\n"
            "This skill reads a text layer. Scanned pages need OCR first; "
            "that is not part of this skill.",
            file=sys.stderr,
        )
        return 1
    except ImportError as exc:
        print(
            f"the repo service could not be imported: {exc}\n"
            + install_hint(missing_dependencies() or ["backend/requirements.txt"]),
            file=sys.stderr,
        )
        return 2
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not chunks:
        print(f"no chunks produced for {pdf_path.name}", file=sys.stderr)
        return 1

    with chatter_to_stderr():
        matrix, embedder_label = embed_texts([c["text"] for c in chunks], service)

    written = save_index(out_dir, pdf_path, chunks, matrix, embedder_label)

    if not args.quiet:
        payload = {
            "pdf": str(pdf_path),
            "out": written["out"],
            "chunks": written["chunks"],
            "dim": written["dim"],
            "pages": summary.get("total_pages"),
            "embedder": embedder_label,
            "files": written["files"],
            "seconds": round(time.time() - started, 2),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
