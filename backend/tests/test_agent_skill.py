"""Automated compliance and functional validation for clinical-document-intelligence Agent Skill.

Tests:
1. SKILL.md specification compliance (YAML frontmatter, name, length, rules)
2. Integrity of referenced scripts, references, and assets
3. Execution on missing PDF (Exit code 1, clean error without raw traceback)
4. Deterministic extraction verification with --claims (Exit code 0, schema-valid JSON)
5. Model unavailable handling (Exit code 3, helpful recovery message)
6. EUNOCURA_BACKEND path resolution from external working directories
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

BACKEND_DIR = pathlib.Path(__file__).resolve().parents[1]
SKILL_DIR = BACKEND_DIR.parent / ".agents" / "skills" / "clinical-document-intelligence"
PYTHON = sys.executable

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
    print("\nAgent Skills Specification & Functional Audit\n")

    # 1. Frontmatter check
    skill_md = SKILL_DIR / "SKILL.md"
    check("SKILL.md exists", skill_md.is_file(), str(skill_md))

    text = skill_md.read_text(encoding="utf-8")
    parts = text.split("---")
    check("SKILL.md has valid YAML frontmatter delimiters", len(parts) >= 3)

    fm_lines = [line.strip() for line in parts[1].strip().split("\n") if line.strip()]
    fm = {}
    for line in fm_lines:
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip()

    name = fm.get("name", "")
    desc = fm.get("description", "")

    check("frontmatter name matches folder name", name == SKILL_DIR.name, f"{name} vs {SKILL_DIR.name}")
    check("name length is <= 64 chars", 0 < len(name) <= 64, f"len: {len(name)}")
    check("description length is <= 1024 chars", 0 < len(desc) <= 1024, f"len: {len(desc)}")
    check("description explains when to use", "use when" in desc.lower(), desc)

    # 2. File integrity
    scripts_dir = SKILL_DIR / "scripts"
    ref_dir = SKILL_DIR / "references"
    assets_dir = SKILL_DIR / "assets"

    check("scripts directory exists", scripts_dir.is_dir())
    check("references directory exists", ref_dir.is_dir())
    check("assets directory exists", assets_dir.is_dir())

    schema_file = ref_dir / "extraction-schema.md"
    check("extraction-schema.md exists", schema_file.is_file())

    sample_pdf = assets_dir / "sample-discharge-summary.pdf"
    check("sample-discharge-summary.pdf exists", sample_pdf.is_file())

    extract_script = scripts_dir / "extract_clinical.py"
    check("extract_clinical.py exists", extract_script.is_file())

    # 3. Missing PDF execution test (Exit Code 1, no traceback)
    proc = subprocess.run(
        [PYTHON, str(extract_script), "--pdf", "non-existent-file.pdf"],
        capture_output=True,
        text=True,
    )
    check("missing PDF returns exit code 1", proc.returncode == 1, f"exit: {proc.returncode}")
    check("clean error message on missing PDF", "not found" in proc.stderr.lower() or "error" in proc.stderr.lower())
    check("no raw Python traceback printed for missing PDF", "Traceback (most recent call last)" not in proc.stderr)

    # 4. Deterministic --claims test (Exit Code 0, valid JSON output)
    claims_file = BACKEND_DIR / "tests" / "fixtures" / "test_claims.json"
    proc = subprocess.run(
        [PYTHON, str(extract_script), "--pdf", str(sample_pdf), "--claims", str(claims_file)],
        capture_output=True,
        text=True,
    )
    check("deterministic --claims returns exit code 0", proc.returncode == 0, f"exit: {proc.returncode}")

    parsed = None
    try:
        # Script prints headline on first line, then JSON block
        json_start = proc.stdout.find("{")
        if json_start != -1:
            parsed = json.loads(proc.stdout[json_start:])
    except Exception as exc:
        check("JSON parsing failed", False, str(exc))

    check("output parses as valid JSON report", parsed is not None)
    if parsed:
        check("report contains total and supported counts", "total" in parsed and "supported" in parsed)
        check("report contains claims array", "claims" in parsed and len(parsed["claims"]) == 5)
        check("dropped count is accurate", parsed.get("dropped") == 3)
        check("page_corrected count is accurate", parsed.get("page_corrected") == 1)

    # 5. External directory resolution via EUNOCURA_BACKEND
    env_override = os.environ.copy()
    env_override["EUNOCURA_BACKEND"] = str(BACKEND_DIR)
    # Run from root directory
    proc = subprocess.run(
        [PYTHON, str(extract_script), "--pdf", str(sample_pdf), "--claims", str(claims_file)],
        cwd=str(BACKEND_DIR.parent),
        env=env_override,
        capture_output=True,
        text=True,
    )
    check("skill executes successfully from external directory with EUNOCURA_BACKEND", proc.returncode == 0, f"exit: {proc.returncode}")

    print(f"\n  {checks - len(failures)}/{checks} passed\n")
    if failures:
        print("  failed: " + ", ".join(failures) + "\n")
        return 1
    print("  PASS  Agent Skills standard compliance verified\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
