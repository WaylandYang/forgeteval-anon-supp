"""Generate the Stage-3 label partition from the current designated runs.

The table was a snapshot taken on the 365-case bench before the survivor
repair and before the token-cap re-measurement, so nearly every cell had
moved: the manual partition is 132 cases rather than 112, Mem0 passes 6
of the llm_lift cases rather than 13, and the column totals no longer
matched the headline numbers anywhere in the paper.

Recomputing also sharpens the two definitional claims the section makes.
The labels were assigned by an earlier execution of the same two
configurations, so re-running moves a handful of cases across the
boundary: Lethe now passes 1 of the 55 llm_lift cases and Lethe+LLM
passes 1 of the 24 unsolvable ones. Those two exceptions are the honest
form of "by construction" and the caller prints them.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from runs import resolve  # noqa: E402

P = "openrouter_hook_deepseek_deepseek-v4-flash_"

SYSTEMS = [
    (r"\sysLethe{}", "nollm_v07_probed"),
    (r"\sysMem{}", "mem0_nollm_v07_probed"),
    (r"\sysLethe{}$+$LLM", "v07_probed"),
]

ORDER = ["manual", "easy", "llm_lift", "unsolvable"]
DISPLAY = {"manual": "manual (core)", "easy": "easy",
           "llm_lift": r"llm\_lift", "unsolvable": "unsolvable"}


def verdicts(name):
    s = resolve(P + name + ".json")
    p = s.with_name(s.name.replace(".json", "_ckpt.jsonl"))
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = bool(r["ok"])
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    lab = json.loads(
        (ROOT / "bench/forgeteval/adversarial_generated_labels.json")
        .read_text(encoding="utf-8"))
    runs = [(n, verdicts(f)) for n, f in SYSTEMS]
    allids = set(runs[0][1])

    groups = {k: [] for k in ORDER}
    for cid in sorted(allids):
        groups.setdefault(lab.get(cid, "manual"), []).append(cid)

    lines = [r"\begin{tabular}{lrccc}", r"\toprule",
             r"\textbf{Label} & \textbf{N} & "
             + " & ".join(r"\textbf{%s}" % n for n, _ in runs) + r"\\",
             r"\midrule"]
    tot = [0] * len(runs)
    for k in ORDER:
        ids = groups[k]
        cells = []
        for i, (_, v) in enumerate(runs):
            p = sum(1 for c in ids if v.get(c))
            tot[i] += p
            cells.append("%d (%d\\%%)" % (p, round(100 * p / len(ids))))
        lines.append("%-14s & %3d & %s \\\\"
                     % (DISPLAY[k], len(ids), " & ".join(cells)))
    n = len(allids)
    lines += [r"\midrule",
              r"\textbf{Total} & %d & %s \\"
              % (n, " & ".join("%d (%d\\%%)" % (t, round(100 * t / n))
                               for t in tot)),
              r"\bottomrule", r"\end{tabular}"]
    (ROOT / "paper" / "tab_label_partition.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("wrote paper/tab_label_partition.tex")

    # the two definitional exceptions, so the prose can name them
    det = dict(runs[0][1])
    hook = dict(runs[2][1])
    lift_pass = [c for c in groups["llm_lift"] if det.get(c)]
    uns_pass = [c for c in groups["unsolvable"] if hook.get(c)]
    print("  partition sizes: " + ", ".join(
        "%s %d" % (k, len(groups[k])) for k in ORDER))
    print("  totals: " + ", ".join(
        "%s %d/%d" % (n, t, len(allids)) for (n, _), t in zip(runs, tot)))
    print("  llm_lift cases the deterministic store now passes: %d %s"
          % (len(lift_pass), lift_pass))
    print("  unsolvable cases the hook now passes: %d %s"
          % (len(uns_pass), uns_pass))


if __name__ == "__main__":
    main()
