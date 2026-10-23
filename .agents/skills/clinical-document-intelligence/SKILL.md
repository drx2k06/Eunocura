---
name: clinical-document-intelligence
description: Read a clinical PDF and answer from it with evidence. Chunks a discharge summary, radiology report or lab sheet with the EUNOCURA pdf_rag service, retrieves the exact chunks for a question with page numbers and bounding boxes, extracts structured clinical facts (symptoms, diagnoses, medications, vitals, lab results, recommendations) with a local open-weight model, and keeps a claim only if the line it quotes really exists on the page it cites. That check is citation-or-drop, plain string comparison, never a model. Runs offline with zero API keys on Ollama (qwen2.5:7b-instruct, medgemma:4b). Use when an agent must read a clinical PDF and cite the exact page or box, when a claim needs checking against source text, or when a reviewer must see which extracted claims were dropped and why.
license: MIT
compatibility: Python >= 3.11 with pdfplumber and PyMuPDF. Offline citation verification requires no GPU or network.
---

# Clinical document intelligence

Reads one clinical PDF, returns the lines that answer a question, and refuses to
show a claim it cannot point at. Three one-command entry points, all runnable
from any agent harness (Claude Code, OpenCode, Cursor) with a shell and Python.

Everything here is deterministic except the extraction step. Retrieval is an
embedding search; verification is string comparison; only the structured
extraction asks a model, and the model's output is checked afterwards.

## Zero API keys

The default model server is Ollama at `http://localhost:11434/v1`:

| profile | model | used for |
| --- | --- | --- |
| `slm` (default) | `qwen2.5:7b-instruct` | extraction, the answer an agent writes |
| `med` (`--profile med`) | `medgemma:4b` | medical-profile extraction |

No key, no cloud, no account. Check the server before the first model call:

```bash
curl -s http://localhost:11434/v1/models
# or:  ollama list   /   ollama serve && ollama pull qwen2.5:7b-instruct
```

If the server is down, retrieval and verification still work. Only
`extract_clinical.py` without `--claims` needs it.

## The three entry points

Run from anywhere; the scripts find the repo's `backend/` themselves (override
with `EUNOCURA_BACKEND` if the checkout lives somewhere unusual).

**1. Build an index over one PDF** (optional, makes later steps fast):

```bash
python .agents/skills/clinical-document-intelligence/scripts/build_index.py \
  --pdf discharge.pdf --out discharge.index
```

Writes `chunks.json` (text, page number, coordinates, chunk index),
`embeddings.npz`, `meta.json`, and `index.faiss` when `faiss-cpu` is installed.
Chunking is done by `src.services.pdf_rag.PDFRAGService`, so boundaries and
boxes match the API's.

**2. Ask a question, get the chunks back for the harness's own model to answer:**

```bash
python .agents/skills/clinical-document-intelligence/scripts/query_pdf.py \
  --pdf discharge.pdf --ask "what was the creatinine?" --k 5
```

No model is called. With `--render evidence.png` the top result's box is drawn
on its page using `src.services.bbox.render_highlight` (needs PyMuPDF; if that
module is unavailable the script says so and returns the JSON without a PNG).

**3. Extract structured facts and verify every claim:**

```bash
python .agents/skills/clinical-document-intelligence/scripts/extract_clinical.py \
  --pdf discharge.pdf --json out.json
```

One model call through `src.llm.router.ask_json`, then
`src.services.citations.verify(claims, pages, chunks)`. Claims whose quote is
not on the cited page are reported as dropped. `--profile med` switches to the
medical model.

**Dry run with no model at all** — the deterministic half, demoable offline:

```bash
python .agents/skills/clinical-document-intelligence/scripts/extract_clinical.py \
  --pdf discharge.pdf --claims claims.json
```

where `claims.json` is a list, or `{"claims": [...]}`, of
`{"claim","quote","page","kind","confidence"}` objects.

## Citation-or-drop

The rule the whole skill is arranged around: **a claim survives only if a line
it quotes exists on the page it cites.** No quote, no claim. Wrong page, we fix
the page or drop the claim.

| Situation | Report status |
| --- | --- |
| quote found on the cited page | `supported` |
| quote found on another page | `page_corrected` — page fixed, claim kept |
| quote found nowhere in the document | `dropped` |
| no quote offered | `dropped`, `no line offered` |
| quote under 8 characters after normalisation | `dropped`, `quote too short to verify` |

Comparison normalises case, whitespace, curly quotes and dashes, then does a
substring test. It does not match synonyms, it does not ask a model whether two
sentences mean the same thing, and it never edits a claim to make it pass. A
dropped claim stays dropped and is shown with its reason.

## Output shapes

`query_pdf.py` (stdout, JSON):

```json
{
  "query": "what was the creatinine?",
  "model": "ollama:qwen2.5:7b-instruct",
  "results": [
    {"text": "...", "page_number": 1, "coordinates": {"x0": 72, "y0": 300, "x1": 512, "y1": 326},
     "score": 0.62, "chunk_index": 4}
  ],
  "index": {"source": "discharge.index", "chunks": 9, "similarity": "cosine (...)"}
}
```

`model` is the label of the model that *would* answer; retrieval itself uses no
model. `coordinates` are PDF points, bottom-left origin, ready for
`render_highlight`. `score` is cosine similarity.

`extract_clinical.py` (stdout, JSON) is the citation report with the pipeline
around it: `pdf`, `model`, `pages`, `source`, `index`, `extraction` (the raw
model object) and then the report's own fields — `headline`, `total`,
`supported`, `dropped`, `page_corrected`, `coverage`, and `claims[]` with
`claim`, `status`, `reason`, `page`, `quote`, `coords`, `confidence`, `kind`.
`supported` counts every kept claim, so page-corrected ones are included in it;
`page_corrected` is broken out separately. `--json FILE` writes the same
payload.

## Rendering an evidence PNG

`query_pdf.py --ask "..." --render evidence.png` writes a PNG of the winning
chunk's box on its own page (padding 2 pt). It calls
`src.services.bbox.render_highlight(pdf_path, page_number, rects, out_path,
pad=2.0)` lazily, so a missing renderer never breaks retrieval. Attachment of
`coords` in the extract report comes from the index built by
`build_index.py`; without an index, claims still verify and `coords` is `null`.

## Exit codes

| Code | Meaning | What to do |
| --- | --- | --- |
| 0 | success | — |
| 1 | bad input: missing PDF, no text layer, unreadable index, unusable model JSON | fix the input; scanned PDFs need OCR first |
| 2 | a required dependency is missing | run the printed `pip install ...` line, or `pip install -r backend/requirements.txt` |
| 3 | model layer unavailable (router not installed, `LLM_PROVIDER=none`, Ollama down, call failed) | start Ollama and pull the model, or drop to `--claims` and verify only |

No exit path prints a raw traceback; every failure names the missing thing and
the command that fixes it.

## Progressive disclosure: when to read more

- **`references/extraction-schema.md`** — read it when you need the exact JSON
  Schema sent to the model, when you are changing what gets extracted, or when
  a field you want is missing. The schema block in that file is read verbatim by
  `scripts/extract_clinical.py`; editing it is how the extraction changes.
  It also carries the field-by-field table and the verifier's decision table.
- **`scripts/`** — read a script when you need to know what it really does. Each
  docstring states plainly what is real and what is degraded.
- **`assets/sample-discharge-summary.pdf`** — a two-page synthetic discharge
  summary ("Asha Rao", MRN SYN-2026-0147) with admission vitals, HbA1c,
  creatinine and TLC, a pneumonia diagnosis line, medications and a follow-up
  plan. No real patient data; use it to try the whole workflow.
  `assets/make_sample.py` regenerates it (needs PyMuPDF, run once).

## What is real, what is degraded

- Real: chunking through the application's service, page numbers, highlight
  boxes, cosine retrieval, verbatim-quote verification, and the report itself.
- Real but slow: the first run downloads the ~80 MB `all-MiniLM-L6-v2`
  embedding model and the model server must hold the extraction model in memory.
- Degraded on purpose: `PDFRAGService` keeps its FAISS store in memory and
  exposes no save method, so `build_index.py` persists the embedding matrix
  (`embeddings.npz`) and searches it exactly (inner product over unit vectors =
  what `IndexFlatIP` does). The binary `index.faiss` is a convenience, not a
  requirement.
- Not handled: scanned PDFs with no text layer. This skill reads text; OCR is a
  different problem and is not simulated here.
