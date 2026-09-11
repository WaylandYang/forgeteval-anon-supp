"""One typographic setting for every figure in the paper.

The heatmap set a Times-like serif to match the body text and the
ablation chart did not, so the two halves of Figure 1 sat side by side in
different typefaces. They also ran six type sizes between them. Both now
import from here, and a new figure gets the same treatment by calling
apply() rather than by remembering to copy a preamble.

Sizes are chosen for a figure drawn at its final width -- the ICLR text
block is 5.5 in -- so nothing is scaled down by \\includegraphics and no
label lands below about 7 pt on the page.
"""
from __future__ import annotations

import os

import matplotlib

# matplotlib stamps /CreationDate into every PDF, so regenerating a figure
# changes five bytes and nothing else. The regenerate-and-diff check reads
# that as "the figure drifted from its data", which is the one thing it
# exists to detect -- a real drift and a fresh timestamp look the same.
# matplotlib honours SOURCE_DATE_EPOCH, so pin it and the output is
# byte-identical for anyone who reruns the generators.
os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")   # 2026-01-01Z

# Type 42 keeps the outlines as TrueType rather than Type 3 bitmaps,
# which some conference PDF checkers reject.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

SERIF = ["Times New Roman", "Nimbus Roman", "DejaVu Serif"]

# one scale, named by what it labels
TITLE = 9.5      # axis labels, row and column names
TICK = 8.5       # tick labels
LEGEND = 8.5     # legend entries
ANNOT = 8.0      # in-plot annotations
CELL = 8.0       # numbers inside heatmap cells
SMALL = 7.0      # group brackets, footnote marks


def apply():
    """Body-matching serif, applied before any figure is created."""
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = SERIF
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["axes.unicode_minus"] = False
