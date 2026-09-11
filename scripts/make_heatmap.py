"""Per-category × per-system heatmap for the paper.

Rows: 9 systems grouped by placement regime
  (no-LLM deterministic / no-LLM-with-vector / inscribe-time LLM /
   KG abstraction / mutation-time LLM hook).
Cols: 10 attack categories, grouped by failure-mode family.
Cell: pass rate %; N/A drawn grey.

This replaces the §5.7 3-regime placement table.
"""
from __future__ import annotations

import matplotlib
# Force TrueType outlines (Type 42), not Type 3 bitmaps.
import pathlib as _pl
import sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import figstyle
figstyle.apply()
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "paper" / "figures"
OUT.mkdir(exist_ok=True)

# System order: groups by placement regime
SYSTEMS = [
    ("MemPalace",       "no-del"),
    ("Lethe",           "det"),
    ("Mem0",            "det"),
    ("LangGraph",       "det"),
    ("Letta",           "vec"),
    ("OpenMemory",      "vec"),
    ("Mem0+v3",         "ins"),
    ("A-MEM",           "ins"),
    ("Graphiti",        "kg"),
    ("HippoRAG",        "kg"),
    # Letta+LLM is a mutation-time hook, not a joint placement:
    # run_letta_llm_bench.py leaves inscribe as plain archival POST
    # and calls the model only from supersede and purge. Letta
    # embeds on write, but so does every vector store in this
    # figure; the inscribe-LLM block is about a model placed there.
    ("Letta+LLM",       "mut"),
    ("Lethe+LLM",       "mut"),
    ("LangGraph+LLM",   "mut"),
]
DAGGER = "†"

NOT_REMEASURED = {
    "HippoRAG",
}

GROUP_BOUNDARIES = [1, 4, 6, 8, 10]  # KG block rows 8-9

CATEGORIES = [
    "substring_trap",
    "prefix_collision",
    "paraphrase",
    "negation_trap",
    "temporal_qualifier",
    "shared_attribute",
    "compound_fact",
    "identifier_obfuscation",
    "cross_lingual_identifier",
    "recursive_supersession",
]
COL_LABELS = [
    "sub-trap", "prefix-coll", "paraphrase", "negation", "temporal",
    "shared-attr", "compound", "ident-obf", "cross-ling", "recursive",
]

# Pass-rate % cells; None = N/A
NA = None
DATA = {
    "MemPalace":         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Lethe":             [92, 82, 82, 95, 100, 88, 0, 5, 3, 92],
    "Mem0":              [83, 33, 82, 95, 100, 92, 0, 61, 18, 92],
    "LangGraph":         [97, 69, 82, 95, 100, 88, 0, 5, 3, 92],
    "Letta":             [53, 0, 84, 48, 57, 0, 0, 0, 0, 92],
    "OpenMemory":        [47, 0, 21, 35, 86, 0, 8, 0, 0, 79],
    "Mem0+v3":           [78, 31, 42, 90, 78, 92, 42, 71, 39, 31],
    "A-MEM":             [56, 0, 82, 48, 100, 0, 0, 0, 0, 92],
    "Graphiti":          [6, 0, 18, 5, 11, 0, 5, 0, 0, 31],
    "HippoRAG":          [22, 0, 3, 2, 57, 0, 0, 0, 0, 0],   # strict, not re-run
    "Letta+LLM":         [53, 67, 74, 52, 30, 0, 0, 100, 100, 26],
    "Lethe+LLM":         [100, 82, 82, 90, 100, 100, 68, 92, 84, 92],
    "LangGraph+LLM":     [100, 82, 82, 88, 100, 95, 68, 92, 97, 92],
}


def main():
    n_sys = len(SYSTEMS)
    n_cat = len(CATEGORIES)
    mat = np.full((n_sys, n_cat), np.nan)
    na_mask = np.zeros((n_sys, n_cat), dtype=bool)
    for r, (sys_name, _) in enumerate(SYSTEMS):
        row = DATA[sys_name]
        for c, v in enumerate(row):
            if v is None:
                na_mask[r, c] = True
            else:
                mat[r, c] = v

    # Softer red→yellow→green palette: less saturated, more academic.
    # Muted blue-to-terracotta rather than red-to-green. Two reasons:
    # the figure is mostly large flat blocks, where low saturation reads
    # better in print, and a red/green axis is the one axis a colour-blind
    # reader cannot separate -- which matters here because the claim *is*
    # the split between two colours.
    cmap = LinearSegmentedColormap.from_list(
        "morandi_div",
        [(0.62, 0.38, 0.33),   # terracotta        (0)
         (0.80, 0.63, 0.55),   # clay              (30)
         (0.90, 0.87, 0.82),   # warm grey         (50)
         (0.55, 0.65, 0.70),   # dusty blue        (70)
         (0.27, 0.44, 0.56)])  # deep slate        (100)])  # deeper green (100)

    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=100, aspect="equal")

    # N/A cells: light grey
    for r in range(n_sys):
        for c in range(n_cat):
            if na_mask[r, c]:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                            facecolor="#d8d8d8",
                                            edgecolor="white", lw=0.5))

    # Rows measured under weaker requirements: hatched, so a high cell
    # here cannot be read the same way as a hatch-free one.
    for r, (sys_name, _) in enumerate(SYSTEMS):
        if sys_name in NOT_REMEASURED:
            # A light wash, no hatch. Six rows needed to recede far
            # enough that a bright cell could not be misread against a
            # measured one; one row needs only to be findable.
            ax.add_patch(plt.Rectangle(
                (-0.5, r - 0.5), n_cat, 1, facecolor="#ffffff",
                edgecolor="none", alpha=0.30, zorder=2))

    # Cell labels — consistent: light cells get dark text, dark cells get white.
    for r in range(n_sys):
        for c in range(n_cat):
            if na_mask[r, c]:
                txt, color = "N/A", "#555"
            else:
                v = mat[r, c]
                txt = f"{int(round(v))}"
                # White text on dark red/dark green; black on mid yellow/light.
                color = "#ffffff" if (v < 22 or v > 78) else "#2b2b2b"
            if SYSTEMS[r][0] in NOT_REMEASURED:
                color = "#3a3a3a"
            ax.text(c, r, txt, ha="center", va="center",
                    fontsize=figstyle.TITLE, color=color, zorder=4)

    ax.set_xticks(range(n_cat))
    ax.set_xticklabels(COL_LABELS, rotation=30, ha="right", fontsize=figstyle.TITLE)
    ax.set_yticks(range(n_sys))
    ax.set_yticklabels(
        [s[0] + (" " + DAGGER if s[0] in NOT_REMEASURED else "")
         for s in SYSTEMS], fontsize=figstyle.TITLE)
    for lab, (sys_name, _) in zip(ax.get_yticklabels(), SYSTEMS):
        if sys_name in NOT_REMEASURED:
            lab.set_color("#777")
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    # Light cell borders
    ax.set_xticks(np.arange(-0.5, n_cat, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_sys, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)

    # Group boundary lines (subtle, just for visual rhythm).
    for b in GROUP_BOUNDARIES:
        ax.axhline(b - 0.5, color="#222", lw=1.0)

    # Right-side regime labels, plain text outside the colorbar.
    regime_brackets = [
        (0, 0, "no-del"),
        (1, 3, "deterministic"),
        (4, 5, "vec only"),
        (6, 7, "inscribe-LLM"),
        (8, 9, "KG abstr."),
        (10, 12, "mutation-LLM"),
    ]
    # No colorbar: every cell prints its value, so the bar restates the
    # numbers at the cost of a quarter of the width.
    fig.subplots_adjust(left=0.20, right=0.80)

    def brace(x, y0, y1, width=0.014):
        """A curly brace spanning y0..y1 at x, opening left."""
        from matplotlib.path import Path as MPath
        import matplotlib.patches as mpatches
        ym = (y0 + y1) / 2
        w = width
        verts = [
            (x, y0), (x + w, y0), (x + w, ym - (ym - y0) * 0.12), (x + w, ym),
            (x + w, ym), (x + w * 2, ym),
            (x + w, ym), (x + w, ym),
            (x + w, ym + (y1 - ym) * 0.12), (x + w, y1), (x, y1),
        ]
        codes = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO,
                 MPath.LINETO, MPath.LINETO,
                 MPath.MOVETO, MPath.LINETO,
                 MPath.CURVE3, MPath.CURVE3, MPath.LINETO]
        fig.add_artist(mpatches.PathPatch(
            # Same colour as the label it brackets: the two are one
            # element, and grey is spoken for by the un-re-measured
            # row. Thinner so it frames rather than competes.
            MPath(verts, codes), fill=False, edgecolor="#2b2b2b",
            linewidth=0.7, transform=fig.transFigure, clip_on=False))

    box = ax.get_position()
    x_brace = box.x1 + 0.010
    x_text = box.x1 + 0.048
    for r0, r1, label in regime_brackets:
        y_top = box.y1 - (r0 + 0.06) / n_sys * box.height
        y_bot = box.y1 - (r1 + 0.94) / n_sys * box.height
        brace(x_brace, y_bot, y_top)
        # Dark, not grey: grey marks the un-re-measured row above and
        # must not also mean "this is an annotation". Italics and the
        # brace already carry that.
        fig.text(x_text, (y_top + y_bot) / 2, label, ha="left", va="center",
                 fontsize=figstyle.CELL, color="#2b2b2b", fontstyle="italic")
    pdf = OUT / "fig_heatmap.pdf"
    png = OUT / "fig_heatmap.png"
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, bbox_inches="tight", dpi=150)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
