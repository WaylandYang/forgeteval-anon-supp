"""Generate the reproducibility appendix's wall-clock table from the runs.

The Reproducibility Statement promised "exact package versions, seeds,
and per-experiment wall-clock costs" and the appendix it pointed at was
a two-sentence stub. The wall-clock numbers are in the run files
themselves (wall_seconds, 66 of them), so the table is derived rather
than typed.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
P = "openrouter_hook_deepseek_deepseek-v4-flash_"

# display name -> result file; one row per experiment the paper reports
ROWS = [
    ("Deterministic, 385 cases", P + "nollm_v07_probed.json"),
    ("Mutation-time hook, 385 cases", P + "v07_probed.json"),
    ("LangGraph deterministic", P + "langgraph_nollm_v07_probed.json"),
    ("LangGraph $+$ hook", P + "langgraph_v07_probed.json"),
    (r"\sysMem{} \code{infer=False}", P + "mem0_nollm_v07_probed.json"),
    (r"\sysMem{} \code{infer=True}", P + "mem0-infer_v07_probed.json"),
    (r"\sysPalace{}", P + "mempalace_nollm_v07_probed.json"),
    (r"\sysAmem{}", P + "amem_v07_probed.json"),
    ("Annotation arm", P + "inscribe_v07_probed.json"),
    ("Readable annotation arm", P + "inscribe-aware_v07_probed.json"),
    ("Merge-authority arm", P + "merge-inscribe_v07_probed.json"),
    ("Both arms", P + "inscribe+mutation_v07_probed.json"),
    ("External subset, deterministic", P + "nollm_external_probed.json"),
    ("External subset, hook", P + "langgraph_external_probed.json"),
]

sys.path.insert(0, str(ROOT / "scripts"))
from runs import resolve  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"\textbf{Experiment} & \textbf{cases} & \textbf{wall (s)} "
             r"& \textbf{LLM calls}\\", r"\midrule"]
    missing = []
    for label, fname in ROWS:
        try:
            p = resolve(fname)
        except Exception:
            missing.append(fname)
            continue
        if not p.exists():
            missing.append(fname)
            continue
        d = json.loads(p.read_text(encoding="utf-8-sig"))
        wall = d.get("wall_seconds")
        calls = (d.get("usage") or {}).get("calls")
        lines.append("%-32s & %d & %s & %s \\\\"
                     % (label, d.get("overall_total", 0),
                        "%.0f" % wall if wall else "---",
                        str(calls) if calls is not None else "---"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tab_repro.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("wrote paper/tab_repro.tex (%d rows)" % (len(lines) - 6))
    if missing:
        print("  no run file for: %s" % ", ".join(missing))


if __name__ == "__main__":
    main()
