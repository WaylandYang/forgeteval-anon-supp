"""Export ADVERSARIAL_TESTS + external admitted cases to JSON files
that the in-docker HippoRAG bench can read.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PAPER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "lethe"))

from bench.forgeteval.adversarial import (  # noqa: E402
    ADVERSARIAL_TESTS, case_to_attack_category,
)


def case_to_dict(case) -> dict:
    """Convert a GeneratedCase to plain dict."""
    return {
        "id": case.id,
        "category": case_to_attack_category(case.id),
        "setup_facts": list(case.setup_facts),
        "mutations": [list(m) for m in case.mutations],
        "final_query": case.final_query,
        "must_contain": list(case.must_contain),
        "must_not_contain": list(case.must_not_contain),
    }


def main():
    out_dir = PAPER_ROOT / "data"
    out_dir.mkdir(exist_ok=True)

    in_house = [case_to_dict(c) for c in ADVERSARIAL_TESTS]
    (out_dir / "hipporag_cases_inhouse.json").write_text(
        json.dumps(in_house, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote in_house {len(in_house)} -> data/hipporag_cases_inhouse.json")

    # External admitted (already in proper dict form)
    ext = json.loads(
        (out_dir / "external_subset_cases.json").read_text(encoding="utf-8"))
    admitted = ext["admitted_cases"]
    (out_dir / "hipporag_cases_external.json").write_text(
        json.dumps(admitted, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote external {len(admitted)} -> data/hipporag_cases_external.json")


if __name__ == "__main__":
    main()
