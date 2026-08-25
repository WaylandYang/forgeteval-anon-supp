"""Run A-MEM across all 10 Memora-weekly personas (150 questions).

Output: data/memora_xeval_amem.json (per-question verdicts).

Goal: get A-MEM's recall-axis number to plot as the 5th point on the
Pareto figure in the paper.
"""
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from eval_on_memora import (  # noqa: E402
    load_persona_data, apply_session_to_adapter,
    score_question, build_adapter,
)

PERSONAS = [
    "academic_researcher", "business_executive", "content_writer",
    "creative_designer", "financial_analyst", "management_consultant",
    "marketing_manager", "sales_manager", "software_engineer",
    "startup_founder",
]
TIMESCALE = "weekly"
SYS_NAME = "amem"

OUT = Path(__file__).resolve().parent.parent / "data" / "memora_xeval_amem.json"


def main():
    rows = []
    by_task = {"remembering": [0, 0], "reasoning": [0, 0], "recommending": [0, 0]}
    by_persona = {p: {"pass": 0, "total": 0} for p in PERSONAS}
    t_start = time.time()

    try:
        adapter = build_adapter(SYS_NAME)
    except Exception as e:
        print(f"build_adapter failed: {e}", flush=True)
        return

    for pi, persona in enumerate(PERSONAS, 1):
        print(f"\n=== persona {pi}/{len(PERSONAS)}: {persona} ===", flush=True)
        adapter.reset()
        sessions, qdoc = load_persona_data(persona, TIMESCALE)
        t0 = time.perf_counter()
        for s in sessions:
            apply_session_to_adapter(adapter, s)
        inscribe_s = time.perf_counter() - t0
        print(f"  inscribed {len(sessions)} sessions in {inscribe_s:.0f}s", flush=True)

        for task in ("remembering", "reasoning", "recommending"):
            for q in qdoc["questions"].get(task, []):
                passed = score_question(adapter, q)
                row = {
                    "persona": persona,
                    "task": task,
                    "question_id": q.get("question_id", ""),
                    "pass": passed,
                }
                rows.append(row)
                by_task[task][1] += 1
                by_task[task][0] += int(bool(passed))
                by_persona[persona]["total"] += 1
                by_persona[persona]["pass"] += int(bool(passed))

        # Persona summary
        p_d = by_persona[persona]
        rate = p_d["pass"] / p_d["total"] * 100 if p_d["total"] else 0
        elapsed = time.time() - t_start
        print(f"  {persona}: {p_d['pass']}/{p_d['total']} ({rate:.1f}%)  "
              f"total_elapsed={elapsed:.0f}s",
              flush=True)

        # Save partial after each persona
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump({
                "system": SYS_NAME,
                "timescale": TIMESCALE,
                "by_task": by_task,
                "by_persona": by_persona,
                "rows": rows,
            }, f, indent=2)

    # Final aggregate
    print("\n=== A-MEM Memora-weekly aggregate ===", flush=True)
    tot_pass = sum(d["pass"] for d in by_persona.values())
    tot = sum(d["total"] for d in by_persona.values())
    print(f"Overall: {tot_pass}/{tot} ({tot_pass/tot*100:.1f}%)", flush=True)
    for task, (p, n) in by_task.items():
        print(f"  {task}: {p}/{n} ({p/n*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
