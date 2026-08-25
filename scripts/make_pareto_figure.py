"""Recall-Forgetting Pareto frontier figure for the paper.

X-axis: recall-axis proxy = Memora-weekly overall pass rate (150 questions
        across 10 personas; substring-scored against Memora's released
        memory_evidence + forgetting_evidence fields).  We use Memora
        rather than LongMemEval-S because we have comparable numbers
        for three systems (Lethe, LangGraph, MemPalace).
Y-axis: forgetting-axis = ForgetEval-Adv overall pass rate, strict
        over all 385 cases. The values are read through runs.py, so
        they cannot drift from the tables; this line said 365 for as
        long as the old snapshot did.

Five points: 3 deterministic backends + 2 LLM-hooked variants.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# This figure sat in the sans-serif default while the two panels of
# Figure 1 were moved onto the body serif, so the paper carried three
# figures in two typefaces. figstyle also pins SOURCE_DATE_EPOCH, which
# is what makes the regenerate-and-diff check able to tell a real drift
# from a fresh /CreationDate stamp.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle                            # noqa: E402
figstyle.apply()
import matplotlib.pyplot as plt            # noqa: E402
import matplotlib.patches as mpatches      # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "paper" / "figures"

# (system, recall, forgetting, det_or_llm, marker_color, annot_dx, annot_dy)
POINTS = [
    # Forgetting axis reads through runs.py, so this figure cannot drift
    # from the tables the way it did when the values were typed in here.
    ("MemPalace",         40.0, 0.0, "det", "#888888", 1.0, 4),
    ("Lethe",             31.3, 63.6, "det", "#1f77b4", 1.0,  3),
    ("LangGraph",         44.7, 62.9, "det", "#2ca02c", 1.0, -5),
    ("Lethe+LLM",         31.3, 88.8, "llm", "#1f77b4", 1.0, 0),
    ("LangGraph+LLM",     44.7, 89.4, "llm", "#2ca02c", 1.0, 0),
]


def is_dominated(point, others):
    """A point is dominated if some other point has >= recall AND >= forgetting,
    with strict inequality on at least one axis."""
    _, r, f, *_ = point
    for op in others:
        if op[0] == point[0]:
            continue
        _, or_, of_, *_ = op
        if or_ >= r and of_ >= f and (or_ > r or of_ > f):
            return True
    return False


def main():
    fig, ax = plt.subplots(figsize=(5.2, 3.8))

    # Plot deterministic Pareto frontier: only connect non-dominated points.
    det_all = [p for p in POINTS if p[3] == "det"]
    det_nondom = [p for p in det_all if not is_dominated(p, det_all)]
    det_nondom_sorted = sorted(det_nondom, key=lambda p: p[1])
    rx = [p[1] for p in det_nondom_sorted]
    ry = [p[2] for p in det_nondom_sorted]
    ax.plot(rx, ry, "--", color="#bbbbbb", lw=1,
            label="deterministic frontier")

    # Plot per-point markers
    for point in POINTS:
        name, r, f, kind, color, dx, dy = point
        dominated = kind == "det" and is_dominated(point, det_all)
        if kind == "det":
            ax.scatter(r, f, s=85, color=color, marker="o",
                       edgecolor="black", lw=0.8, zorder=3,
                       alpha=0.4 if dominated else 1.0)
        else:
            ax.scatter(r, f, s=110, color=color, marker="*",
                       edgecolor="black", lw=0.8, zorder=4)
        label = name + " (dominated)" if dominated else name
        ax.annotate(label, (r, f),
                    xytext=(r + dx, f + dy),
                    fontsize=figstyle.TITLE,
                    ha="left",
                    va="center",
                    alpha=0.6 if dominated else 1.0)

    # Annotation arrows from deterministic points to LLM-hook points
    for det_name in ("Lethe", "LangGraph"):
        det = next(p for p in POINTS if p[0] == det_name)
        llm = next(p for p in POINTS if p[0] == det_name + "+LLM")
        ax.annotate("", xy=(llm[1], llm[2]), xytext=(det[1], det[2]),
                    arrowprops=dict(arrowstyle="-|>", color="#ff7f0e",
                                    lw=1.2, alpha=0.7))
    ax.text(48, 80, "LLM hook\n(architecture-agnostic)",
            color="#ff7f0e", fontsize=figstyle.TICK, style="italic",
            ha="left", va="center")

    ax.set_xlabel("Recall  (Memora-weekly overall pass rate, %)")
    ax.set_ylabel("Forgetting  (ForgetEval-Adv overall pass rate, %)")
    ax.set_xlim(20, 60)
    ax.set_ylim(-4, 102)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Custom legend
    det_handle = plt.Line2D([], [], marker="o", color="w",
                            markeredgecolor="black",
                            markerfacecolor="#444444", markersize=8,
                            label="deterministic backend")
    llm_handle = plt.Line2D([], [], marker="*", color="w",
                            markeredgecolor="black",
                            markerfacecolor="#444444", markersize=12,
                            label="with LLM hook (mutation time)")
    ax.legend(handles=[det_handle, llm_handle],
              loc="center left", bbox_to_anchor=(0.01, 0.45),
              fontsize=figstyle.TICK, frameon=True, edgecolor="#cccccc")

    plt.tight_layout()
    pdf_path = OUT / "fig_pareto.pdf"
    png_path = OUT / "fig_pareto.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=150)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
