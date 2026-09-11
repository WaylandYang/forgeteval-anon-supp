"""Emit one worked case per attack category for the appendix.

The appendix promised "one per attack category, with the per-system
pass/fail breakdown" and contained that sentence and nothing else. The
cases and the per-case verdicts are both in the release, so the examples
are generated rather than transcribed -- which also means they cannot
drift from the runs the way a typed example would.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from runs import resolve  # noqa: E402

P = "openrouter_hook_deepseek_deepseek-v4-flash_"

SYSTEMS = [
    (r"\sysLethe{}", "nollm_v07_probed"),
    (r"\sysMem{}", "mem0_nollm_v07_probed"),
    ("LangGraph", "langgraph_nollm_v07_probed"),
    (r"\sysLethe{}$+$LLM", "v07_probed"),
]

CATS = ["substring_trap", "prefix_collision", "paraphrase_supersession",
        "negation_trap", "temporal_qualifier", "shared_attribute",
        "compound_fact", "identifier_obfuscation",
        "cross_lingual_identifier", "recursive_supersession"]


def tex(s):
    """LaTeX-safe, and short enough to sit in a table cell."""
    for a, b in (("\\", "\\textbackslash{}"), ("_", "\\_"), ("%", "\\%"),
                 ("&", "\\&"), ("#", "\\#"), ("$", "\\$"), ("{", "\\{"),
                 ("}", "\\}"), ("~", "\\textasciitilde{}"),
                 ("^", "\\textasciicircum{}")):
        s = s.replace(a, b)
    return s


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
    from scripts.repair_cross_lingual_queries import build_suite
    suite, _ = build_suite()
    by_id = {c.id: c for c in suite}
    runs = [(n, verdicts(f)) for n, f in SYSTEMS]

    out = []
    for cat in CATS:
        # pdflatex cannot set CJK, Cyrillic or Devanagari with the ICLR
        # template's fonts, so where a category has a Latin-script case
        # we show that one. The rest of the category is in the release.
        def renderable(c):
            blob = " ".join(c.setup_facts) + c.final_query + str(c.mutations)
            return all(ord(ch) <= 0x24F for ch in blob)

        pool = [c for c in suite
                if (re.match(r"adv_(.+)_\d+$", c.id) or [None])
                and re.match(r"adv_(.+)_\d+$", c.id)
                and re.match(r"adv_(.+)_\d+$", c.id).group(1) == cat]
        pick = next((c for c in pool if renderable(c)),
                    pool[0] if pool else None)
        if pick is None:
            print("  no case for %s" % cat, file=sys.stderr)
            continue

        facts = "\n".join(r"\item %s" % tex(f) for f in pick.setup_facts)
        # a supersede carries its replacement text as a third element;
        # printing only the key makes the request unreadable
        muts = ", ".join(
            r"\code{%s(%s)}" % (tex(m[0]),
                                ", ".join('``%s\'\'' % tex(x) for x in m[1:]))
            for m in pick.mutations)
        keep = ", ".join(r"\code{%s}" % tex(x) for x in pick.must_contain)
        drop = ", ".join(r"\code{%s}" % tex(x) for x in pick.must_not_contain)
        marks = " & ".join(
            (r"\checkmark" if v.get(pick.id) else r"$\times$")
            for _, v in runs)

        out.append(
            "\\paragraph{\\famname{%s} --- \\code{%s}.}\n"
            "\\begin{itemize}\\itemsep0pt\\parskip0pt\n%s\n\\end{itemize}\n"
            "\\noindent Request: %s.  Query: ``%s''.\\\\\n"
            "Must survive: %s.  Must be unreachable: %s.\\\\[2pt]\n"
            "\\begin{tabular}{%s}\n\\toprule\n%s\\\\\n\\midrule\n%s\\\\\n"
            "\\bottomrule\n\\end{tabular}\n"
            % (tex(cat), tex(pick.id), facts, muts, tex(pick.final_query),
               keep or "---", drop or "---",
               "c" * len(runs),
               " & ".join(n for n, _ in runs), marks))

    (ROOT / "paper" / "app_cases.tex").write_text(
        "\n\\medskip\n".join(out) + "\n", encoding="utf-8")
    print("wrote paper/app_cases.tex (%d categories)" % len(out))


if __name__ == "__main__":
    main()
