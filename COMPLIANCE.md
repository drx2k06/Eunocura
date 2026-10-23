# EUNOCURA — Competition Compliance & Technical Positioning Report

**Evaluator:** Principal Engineering & Architecture Reviewer  
**Date:** October 2026  
**Repository:** https://github.com/sanju-zxt/EUNOCURA  
**License:** MIT (Application Code)  

---

## 1. Verified Official Competition Sources

1. **MLH Open-Source AI Challenge:**  
   `https://www.mlh.com/opensource-ai`  
   *Requirements:* Meaningful incorporation of open-source or open-weight AI, original software application, public repository, reproducible setup, working demonstration.
2. **Hacktoberfest Bengaluru / Hack Day Challenges:**  
   `https://www.mlh.com/events/hacktoberfest-hack-day-bengaluru-x-aws-sbg-rvu/challenges/01a0d987-0c41-12ea-13ac-5b85755dc2f3`  
   *Requirements:* Open innovation, public collaborative repository, open-weight model usage (e.g., Gemma family or local open weights), clear attribution, zero hidden closed-source dependencies.
3. **Agent Skills Specification:**  
   `https://agentskills.io/specification`  
   *Requirements:* Standard directory layout (`skill-name/SKILL.md`), valid YAML frontmatter, progressive disclosure architecture, deterministic execution fallback, documented schemas.
4. **Model Registries & Licenses:**  
   - Qwen2.5-7B-Instruct: Apache-2.0 (OSI)
   - MedGemma-4B-IT: Health AI Developer Foundations / Gemma Terms of Use (Open Weights)
   - all-MiniLM-L6-v2: Apache-2.0 (OSI)

---

## 2. Competition Positioning Analysis

### Best-Supported Category: Dual Positioning
EUNOCURA qualifies with highest defensibility under:
1. **Primary: Open-Source AI Application (Clinical Document Intelligence)**
2. **Secondary: Reusable Agent Skill (`clinical-document-intelligence`)**

> [!IMPORTANT]
> **Model Harness Positioning Distinction:**  
> While the README colloquially uses the phrase *"open-weight clinical document intelligence harness"*, EUNOCURA should **not** claim to be a generic LLM evaluation harness (such as lm-evaluation-harness or inspect-ai). Rather, it is a **specialized clinical extraction harness** built around the deterministic **citation-or-drop** verifier (`verify()`) and exact bounding box geometry (`bbox.py`). Its core innovation is that it treats open-weight models as probabilistic claim proposers and verifies verbatim quotes against PDF text using plain Python algorithms.

---

## 3. Comprehensive Compliance Matrix

| Requirement | Official Source / Standard | Verification Method | Actual Result | Status | Remaining Action |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Meaningful Open-Weight AI** | MLH Open-Source AI Track | Inspected `src/llm/router.py`, `pdf_rag.py` | Local Qwen2.5-7B, MedGemma-4B, MiniLM embeddings via Ollama/Torch | **PASS** | None; zero external cloud keys required. |
| **Model Identification & Licensing** | MLH Open-Source AI Rules | Checked README §2, model cards | Fully disclosed: MIT code, Apache-2.0 for Qwen & MiniLM, Gemma terms for MedGemma | **PASS** | Maintained in README and COMPLIANCE.md. |
| **Public GitHub Repository** | Hackathon Rules | Git remote `origin` | Clean public repository tracking `main` branch | **PASS** | Repository is publicly accessible. |
| **Valid Open-Source License** | OSI / Hackathon Rules | Inspected root `LICENSE` | Valid MIT license present | **PASS** | Fully compliant. |
| **Agent Skill Compliance** | `agentskills.io/specification` | `test_agent_skill.py` (22 checks) | Standard frontmatter, progressive disclosure, 3 entry points verified | **PASS** | Validated offline and with schema. |
| **Original Technical Contribution** | Hackathon Evaluation Rubric | Inspected `citations.py` & `bbox.py` | Deterministic citation verifier and exact character-span word rect union | **PASS** | Tested across 28 citation checks & 29 bbox checks. |
| **Reproducible Setup** | MLH Judging Standards | Clean venv install, `preflight.py` | Reproducible via `requirements.txt`, `npm install`, and PowerShell | **PASS** | Complete documented instructions in README. |
| **Working Software Demo** | MLH Demo Requirements | Runtime probes on localhost | Backend (8000) and Frontend (3000) active and serving HTTP 200 | **PASS** | Live services operational. |
| **Degrade Matrix Truthfulness** | Engineering Integrity | Healthz probe & preflight audit | Server boots in text-only and offline-degrade modes without crashing | **PASS** | Preflight and healthz accurately report status. |
| **Security & Privacy Hardening** | Security Audit | Route audits, token, path checks | Credential projection fixed (`{"hashed_password": 0}`), path traversal secured | **PASS** | Audit fixes applied and verified. |
| **Synthetic Data / No Real PHI** | Clinical Safety / Privacy Rules | Inspected all PDFs and fixtures | All test data uses synthetic identity ("Asha Rao", SYN-2026-0147) | **PASS** | Zero real patient data in codebase. |
| **Event Eligibility & Authorship** | Event Submission Guidelines | Git log inspection | Git history preserved cleanly; author commits intact | **PASS** | Ensure team registration is finalized on MLH portal. |

---

## 4. Defensible Innovation Claim
> **"EUNOCURA's core innovation is citation-or-drop: a deterministic, model-independent verifier that anchors clinical language model claims directly to source PDF geometry. By decoupling claim generation from evidence validation, it guarantees that no hallucinated claim reaches a clinician without verifiable line-level provenance."**

## 5. Non-Complete Features (Disclosed Fixtures)
To maintain complete competition honesty, the following UI modules are fixture-backed and must **not** be presented as fully connected backend workflows:
- **Medications Portal:** Renders dummy schedule rows (labeled with `FixtureNotice`).
- **Diet Plan:** Renders nutritional guidance fixtures (labeled with `FixtureNotice`).
- **Notifications:** Preview fixtures (labeled with `FixtureNotice`).
- **Test Results OCR:** Mock charts; raw image OCR ingestion is not implemented.
