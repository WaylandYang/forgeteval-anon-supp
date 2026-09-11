"""Redraw paper figures with matplotlib (PNG + PDF outputs).

This replaces the hand-rolled SVG figures with matplotlib renderings that
avoid the overlap problems in the v1 SVGs.  Outputs land in figures/:
  fig1_depth_axis.{png,pdf}
  fig2_variance.{png,pdf}
  fig3_distractors.{png,pdf}
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG  = ROOT / "paper" / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.labelsize":   11,
    "axes.titlesize":   12,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ─── Fig 1: depth axis schematic ─────────────────────────────────────

def fig1():
    """Depth axis as a clean schematic.

    Layout (top to bottom):
      operations as straight horizontal arrows on five staggered rows
      ────────────────── the depth axis ──────────────────
      depth values            (mono font, e.g. depth = 0)
      state names             (bold, colored)
    """
    fig, ax = plt.subplots(figsize=(10.5, 4.6))

    states = [
        (-2.5, r"$\mathrm{depth} < 0$",       "erased",    "#a02020"),
        (-1.0, r"$\mathrm{depth} = 0$",       "submerged", "#7a7a7a"),
        ( 0.5, r"$0 < \mathrm{depth} < 1$",   "sinking",   "#2a5fbf"),
        ( 2.0, r"$\mathrm{depth} = 1$",       "surface",   "#1f7a3f"),
        ( 3.5, r"$\mathrm{depth} = +\infty$", "pinned",    "#b8730a"),
    ]

    xs = [s[0] for s in states]
    ax.set_xlim(-3.5, 4.5)
    ax.set_ylim(-1.4, 3.4)

    y_axis = 0.0

    # Main horizontal axis.
    ax.plot([min(xs) - 0.4, max(xs) + 0.4], [y_axis, y_axis],
            color="#222", linewidth=2.0, zorder=3, solid_capstyle="round")

    # Waterline (depth = 0).
    waterline_x = -1.0
    ax.plot([waterline_x, waterline_x], [-1.05, 3.05],
            color="#bbb", linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.text(waterline_x + 0.06, 3.1, "waterline (depth = 0)",
            ha="left", va="top", color="#888", fontsize=9)

    # State ticks + dual labels under axis.
    for x, dlabel, state, color in states:
        ax.plot([x, x], [y_axis - 0.10, y_axis + 0.10],
                color=color, linewidth=2.6, zorder=4)
        ax.text(x, y_axis - 0.42, dlabel,
                ha="center", va="top", fontsize=10, color="#555")
        ax.text(x, y_axis - 0.90, state,
                ha="center", va="top", fontsize=11.5, color=color,
                fontweight="bold")

    # Operations as straight arrows on five staggered y-levels.  No curves
    # so labels never collide.  inscribe() is a downward arrow into the
    # axis at the surface position; the rest are horizontal arrows from
    # source state to target state.
    ops = [
        # (source x, target x, label, y, color, is_vertical)
        ( 2.0,  2.0, "inscribe()",          1.05, "#1f7a3f", True),
        ( 2.0,  0.5, "consolidate()",       1.65, "#2a5fbf", False),
        ( 0.5, -1.0, "surrender(release)",  2.25, "#7a7a7a", False),
        ( 2.0,  3.5, "pin()",               2.85, "#b8730a", False),
        (-1.0, -2.5, "surrender(purge)",    0.45, "#a02020", False),
    ]
    for sx, ex, label, y, color, vertical in ops:
        if vertical:
            # Downward arrow landing on the axis.
            ax.annotate(
                "", xy=(sx, y_axis + 0.12), xytext=(sx, y),
                arrowprops=dict(arrowstyle="-|>",
                                color=color, lw=1.8,
                                mutation_scale=12,
                                shrinkA=0, shrinkB=0),
                zorder=5,
            )
            ax.text(sx + 0.12, y + 0.03, label, ha="left", va="center",
                    fontsize=10, color=color,
                    family="DejaVu Sans Mono")
        else:
            ax.annotate(
                "", xy=(ex, y), xytext=(sx, y),
                arrowprops=dict(arrowstyle="-|>",
                                color=color, lw=1.8,
                                mutation_scale=12,
                                shrinkA=4, shrinkB=4),
                zorder=5,
            )
            mx = (sx + ex) / 2
            ax.text(mx, y + 0.16, label, ha="center", va="bottom",
                    fontsize=10, color=color,
                    family="DejaVu Sans Mono")

    ax.set_title("Lethe state space: a single scalar axis "
                 r"(every operation is a force on $\mathrm{depth} \in \mathbb{R}$)",
                 loc="left", fontsize=12.5, pad=14)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig1_depth_axis.{ext}")
    plt.close(fig)
    print(f"wrote {FIG / 'fig1_depth_axis.png'} (+pdf)")


# ─── Fig 2: variance bar chart ───────────────────────────────────────

def fig2():
    runs = json.loads((DATA / "variance.json").read_text())
    families = ["supersession", "decay", "amnesia", "purge", "drift"]
    means = []
    sds   = []
    for f in families:
        vals = [r["by_family"][f]["rate"] * 100 for r in runs]
        means.append(statistics.mean(vals))
        sds.append(statistics.pstdev(vals))
    overall = [r["overall_rate"] * 100 for r in runs]
    om = statistics.mean(overall)
    osd = statistics.pstdev(overall)

    fig, ax = plt.subplots(figsize=(8, 4.2))

    x = list(range(len(families)))
    bars = ax.bar(x, means, yerr=sds,
                  capsize=6, color="#3a6ea5", alpha=0.92,
                  edgecolor="#1e3e60", linewidth=0.8,
                  error_kw=dict(ecolor="#1e3e60", elinewidth=1.4, capthick=1.4))

    # Annotate each bar with mean ± σ
    for xi, m, s in zip(x, means, sds):
        ax.text(xi, m + s + 0.6, f"{m:.1f} ± {s:.1f}",
                ha="center", va="bottom", fontsize=9.5, color="#1e3e60")

    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=10)
    ax.set_ylim(85, 102)
    ax.set_ylabel("Pass rate (%)")
    ax.yaxis.grid(True, linestyle=":", color="#aaa", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title(
        f"ForgetEval variance across 5 seeds  ·  "
        f"overall = {om:.2f} ± {osd:.2f}%  ·  scale=50",
        fontsize=11.5, loc="left", pad=10,
    )

    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig2_variance.{ext}")
    plt.close(fig)
    print(f"wrote {FIG / 'fig2_variance.png'} (+pdf)")


# ─── Fig 3: distractor curve ─────────────────────────────────────────

def fig3():
    runs = json.loads((DATA / "distractors.json").read_text())
    by_d: dict[int, list] = {}
    for r in runs:
        by_d.setdefault(r["distractors"], []).append(r)

    ds = sorted(by_d)
    overall_m, overall_s = [], []
    amnesia_m, amnesia_s = [], []
    for d in ds:
        rs = by_d[d]
        ovs = [r["overall_rate"] * 100 for r in rs]
        ams = [r["by_family"]["amnesia"]["rate"] * 100 for r in rs]
        overall_m.append(statistics.mean(ovs))
        overall_s.append(statistics.pstdev(ovs))
        amnesia_m.append(statistics.mean(ams))
        amnesia_s.append(statistics.pstdev(ams))

    fig, ax = plt.subplots(figsize=(8, 4.4))

    ax.errorbar(ds, overall_m, yerr=overall_s,
                fmt="o-", color="#3a6ea5", lw=1.8, ms=6,
                capsize=4, capthick=1.2, elinewidth=1.2,
                label="overall")
    ax.errorbar(ds, amnesia_m, yerr=amnesia_s,
                fmt="s--", color="#cc5500", lw=1.8, ms=6,
                capsize=4, capthick=1.2, elinewidth=1.2,
                label="amnesia only")

    # Value labels offset alternately to avoid collision with the
    # other series.
    for d, m in zip(ds, overall_m):
        ax.annotate(f"{m:.1f}", (d, m), xytext=(0, 11),
                    textcoords="offset points",
                    ha="center", fontsize=9, color="#3a6ea5")
    for d, m in zip(ds, amnesia_m):
        ax.annotate(f"{m:.1f}", (d, m), xytext=(0, -16),
                    textcoords="offset points",
                    ha="center", fontsize=9, color="#cc5500")

    ax.set_xscale("log")
    ax.set_xticks(ds)
    ax.set_xticklabels([str(d) for d in ds])
    ax.set_xlabel("filler facts per case (d)")
    ax.set_ylabel("Pass rate (%)")
    ax.set_ylim(78, 103)
    ax.yaxis.grid(True, linestyle=":", color="#aaa", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False)

    ax.set_title(
        "Distractor robustness: overall vs. amnesia  ·  "
        "mean ± σ over seeds 42–44, scale=50",
        fontsize=11.5, loc="left", pad=10,
    )

    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig3_distractors.{ext}")
    plt.close(fig)
    print(f"wrote {FIG / 'fig3_distractors.png'} (+pdf)")


def main():
    fig1()
    fig2()
    fig3()


if __name__ == "__main__":
    main()
