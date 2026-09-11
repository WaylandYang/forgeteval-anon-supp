"""Per-case latency, derived from the same run files as the repro table.

Appendix M used to carry five hand-typed figures -- 74, 64, 191 and
514 ms/case and 2.3 s/case -- with no data file behind any of them, and
they disagreed with the wall-clock table by up to 7.7x on the same runs.
The only latency the released artifacts actually measure is wall_seconds,
so both tables now come from it and cannot diverge.

wall_seconds covers a whole run: harness setup, store construction, the
cases, and teardown. It is the cost of reproducing the experiment, which
is what the paper needs it for, and it is an upper bound on algorithmic
per-case cost rather than an isolation of it.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from runs import resolve  # noqa: E402

P = "openrouter_hook_deepseek_deepseek-v4-flash_"

ROWS = [
    (r"LangGraph \code{InMemoryStore}", P + "langgraph_nollm_v07_probed.json", "det"),
    (r"\sysLethe{}", P + "nollm_v07_probed.json", "det"),
    (r"\sysPalace{}", P + "mempalace_nollm_v07_probed.json", "det"),
    (r"\sysMem{} \code{infer=False}", P + "mem0_nollm_v07_probed.json", "det"),
    (r"\sysLethe{}$+$LLM", P + "v07_probed.json", "hook"),
    (r"LangGraph$+$LLM", P + "langgraph_v07_probed.json", "hook"),
    (r"\sysAmem{}", P + "amem_v07_probed.json", "det"),
]


def fmt(ms):
    return r"%.0f\,ms" % ms if ms < 1000 else r"%.1f\,s" % (ms / 1000)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"\textbf{System} & \textbf{wall (s)} & \textbf{cases} "
             r"& \textbf{per case}\\", r"\midrule"]
    seen = {}
    for label, fname, kind in ROWS:
        d = json.loads(resolve(fname).read_text(encoding="utf-8-sig"))
        wall, n = d.get("wall_seconds"), d.get("overall_total", 0)
        if not wall or not n:
            raise SystemExit("no wall_seconds for %s" % fname)
        ms = 1000 * wall / n
        seen[label] = (ms, kind)
        lines.append("%-34s & %d & %d & %s \\\\" % (label, round(wall), n, fmt(ms)))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tab_latency.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    det = [m for m, k in seen.values() if k == "det"]
    print("deterministic band: %.0f--%.0f ms/case" % (min(det), max(det)))
    for label, (ms, kind) in seen.items():
        print("  %-34s %8.0f ms  (%s)" % (label, ms, kind))


if __name__ == "__main__":
    main()
