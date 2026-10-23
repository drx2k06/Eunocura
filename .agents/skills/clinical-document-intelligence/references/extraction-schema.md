# Extraction schema

Read this file when you need the exact shape of what `extract_clinical.py`
sends to the model and gets back, when you are changing what the extractor
asks for, or when a field you want is missing from the output.

The schema block marked `schema:start` / `schema:end` below is **read verbatim**
by `scripts/extract_clinical.py` and passed to `src.llm.router.ask_json(...)`.
Editing that block is how you change the extraction; there is no second copy to
keep in sync. The markdown around it is for humans.

## Why the shape looks like this

Every extracted item carries `quote` and `page` because those two fields are
what `src.services.citations.verify(claims, pages, chunks)` checks: a claim
survives only if the normalised text of `quote` occurs on the page numbered
`page` (1-based). The structured arrays (`symptoms`, `vitals`, ...) are the
presentation layer; `claims` is the audited layer — one atomic statement per
entry, each with its own quotable line.

Two consequences worth stating plainly:

- A quote must be **copied character for character** from the page text. A
  paraphrase is a dropped claim, not a citation.
- A claim must be **atomic**. "Hypoxic on arrival and started on ceftriaxone"
  cannot be supported by one line, so it cannot be one claim. Split it.

The verifier normalises case, whitespace, curly quotes and dashes before
comparing, so `SpO2 91% on room air` matches `SPO2  91 % on room air` only if
the words and digits are identical. It does not match synonyms, and it never
asks a model whether two sentences mean the same thing.

<!-- schema:start -->
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://eunocura.local/schemas/clinical-extraction.json",
  "title": "EUNOCURA clinical extraction",
  "description": "Facts read from one clinical PDF. Every item quotes the line it came from and the 1-based page that line is on, because the citation verifier checks quote-against-page as plain string work. The claims array is the audited layer: one atomic statement per entry.",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "symptoms",
    "diagnoses",
    "medications",
    "vitals",
    "lab_results",
    "recommendations",
    "claims"
  ],
  "properties": {
    "symptoms": {
      "type": "array",
      "description": "Symptoms the patient reported or the clinician recorded.",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["name", "quote", "page"],
        "properties": {
          "name": { "type": "string", "description": "Symptom as written, e.g. 'breathlessness'." },
          "duration": { "type": ["string", "null"], "description": "As stated, e.g. 'four days'. Null if not stated." },
          "quote": { "type": "string", "minLength": 8, "description": "Verbatim line from the page." },
          "page": { "type": "integer", "minimum": 1, "description": "1-based page the quote is on." }
        }
      }
    },
    "diagnoses": {
      "type": "array",
      "description": "Diagnoses named in the document.",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["name", "status", "quote", "page"],
        "properties": {
          "name": { "type": "string" },
          "status": {
            "type": "string",
            "enum": ["confirmed", "probable", "differential"],
            "description": "Only as the document words it; do not upgrade a probable diagnosis."
          },
          "quote": { "type": "string", "minLength": 8 },
          "page": { "type": "integer", "minimum": 1 }
        }
      }
    },
    "medications": {
      "type": "array",
      "description": "Medications given or prescribed, one entry per drug.",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["name", "quote", "page"],
        "properties": {
          "name": { "type": "string" },
          "dose": { "type": ["string", "null"], "description": "e.g. '1 g'. Null if not stated." },
          "route": { "type": ["string", "null"], "description": "e.g. 'intravenous'. Null if not stated." },
          "frequency": { "type": ["string", "null"], "description": "e.g. 'twice daily'. Null if not stated." },
          "quote": { "type": "string", "minLength": 8 },
          "page": { "type": "integer", "minimum": 1 }
        }
      }
    },
    "vitals": {
      "type": "array",
      "description": "Vital signs with the numbers as written.",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["name", "value", "quote", "page"],
        "properties": {
          "name": { "type": "string", "description": "e.g. 'BP', 'HR', 'SpO2', 'temperature'." },
          "value": { "type": "string", "description": "As written, e.g. '148/92'. Never recomputed." },
          "unit": { "type": ["string", "null"], "description": "e.g. 'mmHg', 'bpm', '%', 'C'." },
          "measured_at": { "type": ["string", "null"], "description": "e.g. 'admission', 'discharge'." },
          "quote": { "type": "string", "minLength": 8 },
          "page": { "type": "integer", "minimum": 1 }
        }
      }
    },
    "lab_results": {
      "type": "array",
      "description": "Investigations with a numeric or named result.",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["name", "value", "quote", "page"],
        "properties": {
          "name": { "type": "string", "description": "e.g. 'HbA1c', 'serum creatinine'." },
          "value": { "type": "string", "description": "As written, e.g. '8.4', '14,200'." },
          "unit": { "type": ["string", "null"], "description": "e.g. '%', 'mg/dL', 'per microlitre'." },
          "reference_range": { "type": ["string", "null"], "description": "Only if the document prints one." },
          "abnormal": { "type": ["boolean", "null"], "description": "True only if the document marks it raised/low/abnormal." },
          "quote": { "type": "string", "minLength": 8 },
          "page": { "type": "integer", "minimum": 1 }
        }
      }
    },
    "recommendations": {
      "type": "array",
      "description": "Plan items: follow-up, repeat tests, advice, red flags.",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["action", "quote", "page"],
        "properties": {
          "action": { "type": "string", "description": "What is to be done, as written." },
          "timeframe": { "type": ["string", "null"], "description": "e.g. 'after 1 week'. Null if not stated." },
          "quote": { "type": "string", "minLength": 8 },
          "page": { "type": "integer", "minimum": 1 }
        }
      }
    },
    "claims": {
      "type": "array",
      "description": "Atomic statements, each backed by one verbatim line on one page. This is the layer citations.verify checks; a claim with a paraphrased or invented quote is dropped.",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["claim", "quote", "page", "kind", "confidence"],
        "properties": {
          "claim": { "type": "string", "minLength": 1, "description": "One fact, stated plainly." },
          "quote": { "type": "string", "minLength": 8, "description": "Verbatim line from the cited page. 8+ characters after normalisation." },
          "page": { "type": "integer", "minimum": 1, "description": "1-based page number." },
          "kind": {
            "type": "string",
            "enum": ["finding", "vital", "lab", "medication", "diagnosis", "symptom", "plan"]
          },
          "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
        }
      }
    }
  }
}
```
<!-- schema:end -->

## Field table

| Field | Type | Required | Why it exists |
| --- | --- | --- | --- |
| `symptoms[].name` | string | yes | Symptom as written, so it can be shown without re-reading the page. |
| `symptoms[].duration` | string \| null | no | Duration only as stated; null when the page does not say. |
| `symptoms[].quote` / `.page` | string / integer | yes | The line and 1-based page that prove the symptom. |
| `diagnoses[].name` | string | yes | The diagnosis text. |
| `diagnoses[].status` | enum `confirmed` \| `probable` \| `differential` | yes | Keeps "probable" from being rendered as "confirmed" downstream. |
| `diagnoses[].quote` / `.page` | string / integer | yes | Evidence for the diagnosis line. |
| `medications[].name` | string | yes | Drug name as written. |
| `medications[].dose` | string \| null | no | Dose as written; null when absent, never inferred from a class. |
| `medications[].route` | string \| null | no | Route as written (intravenous, oral, ...). |
| `medications[].frequency` | string \| null | no | Frequency as written. |
| `medications[].quote` / `.page` | string / integer | yes | Evidence line for the drug. |
| `vitals[].name` | string | yes | Which vital (BP, HR, SpO2, temperature). |
| `vitals[].value` | string | yes | The number as printed. String, so `148/92` and `14,200` survive intact. |
| `vitals[].unit` | string \| null | no | mmHg, bpm, %, C. |
| `vitals[].measured_at` | string \| null | no | Admission vs discharge separates two different BP readings. |
| `vitals[].quote` / `.page` | string / integer | yes | Evidence line for the reading. |
| `lab_results[].name` | string | yes | Investigation name. |
| `lab_results[].value` | string | yes | Result as printed; no unit conversion, no arithmetic. |
| `lab_results[].unit` | string \| null | no | Unit as printed. |
| `lab_results[].reference_range` | string \| null | no | Only when the document prints a range. |
| `lab_results[].abnormal` | boolean \| null | no | True only when the document itself marks it raised/low/abnormal. |
| `lab_results[].quote` / `.page` | string / integer | yes | Evidence line for the result. |
| `recommendations[].action` | string | yes | The plan item. |
| `recommendations[].timeframe` | string \| null | no | "after 1 week", "in 6 weeks". |
| `recommendations[].quote` / `.page` | string / integer | yes | Evidence line for the plan item. |
| `claims[].claim` | string | yes | One atomic statement. Two facts in one claim cannot be verified by one line. |
| `claims[].quote` | string | yes | Verbatim line; the only thing that keeps the claim. |
| `claims[].page` | integer ≥ 1 | yes | Where the line is. Wrong page is corrected by the verifier when the line exists elsewhere. |
| `claims[].kind` | enum | yes | Lets the UI group claims without re-parsing the text. |
| `claims[].confidence` | number 0-1 | yes | The model's own estimate. It never overrides verification; a confident unsupported claim is still dropped. |

## Example of model output

```json
{
  "vitals": [
    {
      "name": "SpO2",
      "value": "91",
      "unit": "%",
      "measured_at": "admission",
      "quote": "BP 148/92 mmHg, HR 104 bpm, SpO2 91% on room air, temperature 38.6 C.",
      "page": 1
    }
  ],
  "claims": [
    {
      "claim": "The patient was hypoxic on room air at admission.",
      "quote": "SpO2 91% on room air",
      "page": 1,
      "kind": "vital",
      "confidence": 0.93
    },
    {
      "claim": "The patient was started on vancomycin.",
      "quote": "started on intravenous vancomycin",
      "page": 1,
      "kind": "medication",
      "confidence": 0.4
    }
  ]
}
```

The first claim is supported: `SpO2 91% on room air` occurs on page 1. The
second is dropped, because that line is not in the document — the model
invented a drug. `extract_clinical.py` shows both, with the second marked
`"status": "dropped"`. Nothing silently repairs it.

## What the verifier does with each claim

| Situation | Result |
| --- | --- |
| Quote found on the cited page | `supported` |
| Quote found on a different page | `page_corrected` (page fixed, claim kept, reason names the real page) |
| Quote found nowhere | `dropped`, reason `this line does not appear anywhere in the document` |
| No quote offered | `dropped`, reason `no line offered` |
| Quote shorter than 8 characters after normalisation | `dropped`, reason `quote too short to verify` |

Coordinates for a highlight box are attached only when `chunks` are supplied
(the index built by `build_index.py`); without them the claim is still verified
and `coords` is `null`.
