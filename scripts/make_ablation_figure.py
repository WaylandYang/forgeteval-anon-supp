"""The ablation as a shape rather than a table.

Table 4 carries six configurations by three categories and the reader has
to hold eighteen numbers to see the finding. The finding is a shape: two
lines climb as the control plane gains reach, and the third stays flat on
zero across every configuration that never sees the deletion request,
then jumps.

Data comes from the runs through runs.py, so this cannot drift from the
table it accompanies.
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import figstyle                          # noqa: E402
figstyle.apply()
import matplotlib.pyplot as plt          # noqa: E402
from runs import resolve                 # noqa: E402

OUT = ROOT / "paper" / "figures"
P = "openrouter_hook_deepseek_deepseek-v4-flash_"

# (x label, second line of label, run file) -- ordered by what the model sees
CELLS = [
    ("no model", "", "nollm_v07_probed.json"),
    ("annotate", "", "inscribe_v07_probed.json"),
    ("+ readable", "", "inscribe-aware_v07_probed.json"),
    ("merge", "", "merge-inscribe_v07_probed.json"),
    ("mutation", "", "v07_probed.json"),
    ("both", "", "inscribe+mutation_v07_probed.json"),
]

SERIES = [
    ("identifier_obfuscation", "identifier_obfuscation", 38,
     "#4a6f8a", "o", "-"),
    ("cross_lingual_identifier", "cross_lingual_identifier", 38,
     "#7fa3ba", "s", "-"),
    ("compound_fact", "compound_fact", 40, "#a05a4a", "D", "-"),
]


def main():
    vals = {}
    for _, _, f in CELLS:
        vals[f] = json.loads(resolve(P + f).read_text(encoding="utf-8-sig"))

    # Drawn at the same scale as the heatmap: that one is 4.6 in wide and
    # sits in a 0.60\linewidth box, so 0.72 of drawn size reaches the page.
    # Matching the ratio here keeps one type size one type size in print.
    fig, ax = plt.subplots(figsize=(2.9, 3.4))
    xs = range(len(CELLS))

    # The region where the model never sees the deletion request. The
    # span has to meet the axis edge exactly: a sliver of white between
    # the two reads as a category of its own.
    XLO, XHI = -0.45, len(CELLS) - 0.55
    ax.set_xlim(XLO, XHI)
    ax.axvspan(XLO, 3.5, color="#efebe6", zorder=0, linewidth=0)
    # Upper left: the two canonicalization lines climb through the middle
    # of the shaded region, so a centred label sits on top of them. The
    # left corner is empty because every series starts near zero.
    ax.text(XLO + 0.18, 97, "model never sees\nthe deletion request",
            ha="left", va="top", fontsize=figstyle.ANNOT, color="#8a7f75",
            style="italic", linespacing=1.35)

    for label, key, denom, colour, marker, ls in SERIES:
        ys = [100 * vals[f]["by_category"][key]["pass"] / denom
              for _, _, f in CELLS]
        ax.plot(list(xs), ys, ls, color=colour, marker=marker,
                markersize=4.5, linewidth=1.7, label=label, zorder=3,
                clip_on=False)

    ax.set_xticks(list(xs))
    ax.set_xticklabels(
        ["%s\n%s" % (a, b) if b else a for a, b, _ in CELLS],
        fontsize=figstyle.TICK, rotation=22, ha="right", rotation_mode="anchor")
    ax.set_ylim(-3, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("pass rate (%)", fontsize=figstyle.TITLE)
    ax.tick_params(labelsize=figstyle.TICK)
    ax.grid(axis="y", color="#e4e0da", linewidth=0.7, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999")
        ax.spines[side].set_linewidth(0.7)

    # Above the axes, one row: inside them it would straddle the shaded
    # boundary and imply a grouping that is not there.
    ax.legend(fontsize=figstyle.LEGEND, frameon=False, loc="lower left",
              bbox_to_anchor=(-0.02, 1.07), ncol=1, handlelength=1.5,
              labelspacing=0.25, borderaxespad=0.0)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = OUT / ("fig_ablation." + ext)
        fig.savefig(p, bbox_inches="tight",
                    **({"dpi": 150} if ext == "png" else {}))
        print("wrote %s" % p.name)


if __name__ == "__main__":
    main()
