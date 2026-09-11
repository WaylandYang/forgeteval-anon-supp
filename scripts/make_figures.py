"""Generate paper figures from data/*.json.

Figures produced:
  - fig1_depth_axis.svg   — schematic of the depth axis
  - fig2_variance.svg     — per-family bar chart with σ across 5 seeds
  - fig3_distractors.svg  — distractor-density sweep line plot
  - fig4_ablations.svg    — ablation grouped bar chart

Uses pure SVG strings (no matplotlib) so the figures stay reproducible
and tiny.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(exist_ok=True)


SVG_HEADER = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="Inter, Helvetica, sans-serif">'
SVG_FOOTER = "</svg>"


def fig1_depth_axis() -> str:
    w, h = 900, 360
    cx, cy0 = 80, 200
    cx2 = 820
    parts = [SVG_HEADER.format(w=w, h=h)]
    parts.append(f'<text x="20" y="40" font-size="22" font-weight="600">'
                 f'Lethe state space: a single scalar axis</text>')
    parts.append(f'<text x="20" y="70" font-size="14" fill="#666">'
                 f'every operation is a force on depth ∈ ℝ</text>')
    # axis line
    parts.append(f'<line x1="{cx}" y1="{cy0}" x2="{cx2}" y2="{cy0}" '
                 f'stroke="#333" stroke-width="2"/>')
    # ticks
    ticks = [
        (cx + 80,   "depth < 0",  "erased",        "#aa0000"),
        (cx + 200,  "depth = 0",  "submerged",     "#888"),
        (cx + 360,  "0 < d < 1",  "sinking",       "#3366cc"),
        (cx + 520,  "depth = 1",  "surface",       "#229922"),
        (cx + 680,  "depth = +∞", "pinned",        "#cc7700"),
    ]
    for tx, label, state, color in ticks:
        parts.append(f'<line x1="{tx}" y1="{cy0-8}" x2="{tx}" y2="{cy0+8}" '
                     f'stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{tx}" y="{cy0+30}" text-anchor="middle" '
                     f'font-size="12" font-family="monospace" fill="#333">'
                     f'{label}</text>')
        parts.append(f'<text x="{tx}" y="{cy0+50}" text-anchor="middle" '
                     f'font-size="14" font-weight="600" fill="{color}">'
                     f'{state}</text>')
    # operations as arrows above the axis
    ops = [
        (cx+520, cx+520, "inscribe()", -60, "#229922"),
        (cx+520, cx+360, "consolidate()", -90, "#3366cc"),
        (cx+360, cx+200, "surrender(release)", -120, "#888"),
        (cx+520, cx+680, "pin()", -150, "#cc7700"),
        (cx+200, cx+80,  "surrender(purge)", -90, "#aa0000"),
    ]
    for sx, ex, label, dy, color in ops:
        ay = cy0 + dy
        mx = (sx + ex) // 2
        parts.append(f'<path d="M{sx},{cy0-10} Q{mx},{ay} {ex},{cy0-10}" '
                     f'stroke="{color}" stroke-width="1.5" fill="none" '
                     f'marker-end="url(#arrow{color})"/>')
        parts.append(f'<text x="{mx}" y="{ay-5}" text-anchor="middle" '
                     f'font-family="monospace" font-size="12" '
                     f'fill="{color}">{label}</text>')
    # arrow defs
    parts.append('<defs>')
    for _, _, _, _, color in ops:
        parts.append(f'<marker id="arrow{color}" viewBox="0 0 10 10" '
                     f'refX="9" refY="5" markerWidth="5" markerHeight="5" '
                     f'orient="auto"><path d="M0,0 L10,5 L0,10 z" '
                     f'fill="{color}"/></marker>')
    parts.append('</defs>')
    # waterline
    parts.append(f'<line x1="{cx+200}" y1="{cy0-100}" x2="{cx+200}" y2="{cy0+100}" '
                 f'stroke="#999" stroke-width="1" stroke-dasharray="3,3"/>')
    parts.append(f'<text x="{cx+200}" y="{cy0+100}" text-anchor="middle" '
                 f'font-size="11" fill="#999">waterline</text>')
    parts.append(SVG_FOOTER)
    return "\n".join(parts)


def fig2_variance() -> str:
    runs = json.loads((DATA / "variance.json").read_text())
    families = ["supersession", "decay", "amnesia", "purge", "drift"]
    by_fam = {f: [] for f in families}
    for r in runs:
        for f in families:
            by_fam[f].append(r["by_family"][f]["rate"])

    w, h = 800, 440
    parts = [SVG_HEADER.format(w=w, h=h)]
    parts.append(f'<text x="40" y="36" font-size="20" font-weight="600">'
                 f'Variance across 5 seeds (scale=50, distractors=4)</text>')
    parts.append(f'<text x="40" y="60" font-size="12" fill="#666">'
                 f'each bar = mean over seeds 42–46; error bars = ±1σ</text>')

    # Axes
    plot_x0, plot_y0 = 80, 380
    plot_w, plot_h = 680, 300
    parts.append(f'<line x1="{plot_x0}" y1="{plot_y0}" '
                 f'x2="{plot_x0+plot_w}" y2="{plot_y0}" stroke="#333"/>')
    parts.append(f'<line x1="{plot_x0}" y1="{plot_y0}" '
                 f'x2="{plot_x0}" y2="{plot_y0-plot_h}" stroke="#333"/>')

    # y-axis ticks  (80, 90, 100)
    for v in [80, 85, 90, 95, 100]:
        y = plot_y0 - (v - 70) / 30 * plot_h
        parts.append(f'<line x1="{plot_x0-4}" y1="{y}" x2="{plot_x0}" y2="{y}" stroke="#333"/>')
        parts.append(f'<text x="{plot_x0-8}" y="{y+4}" text-anchor="end" '
                     f'font-size="11" fill="#444">{v}%</text>')
        if v < 100:
            parts.append(f'<line x1="{plot_x0}" y1="{y}" x2="{plot_x0+plot_w}" y2="{y}" '
                         f'stroke="#eee"/>')

    # Bars
    bar_w = plot_w / (len(families) * 1.6)
    gap   = (plot_w - bar_w * len(families)) / (len(families) + 1)
    for i, f in enumerate(families):
        rates = by_fam[f]
        mean = statistics.mean(rates) * 100
        sd   = statistics.pstdev(rates) * 100 if len(rates) > 1 else 0.0
        x = plot_x0 + gap + i * (bar_w + gap)
        h_bar = (mean - 70) / 30 * plot_h
        y_top = plot_y0 - h_bar
        parts.append(f'<rect x="{x}" y="{y_top}" width="{bar_w}" '
                     f'height="{h_bar}" fill="#3a6ea5"/>')
        # error bar
        eh = sd / 30 * plot_h
        parts.append(f'<line x1="{x+bar_w/2}" y1="{y_top-eh}" '
                     f'x2="{x+bar_w/2}" y2="{y_top+eh}" '
                     f'stroke="#222" stroke-width="1.5"/>')
        parts.append(f'<line x1="{x+bar_w/2-6}" y1="{y_top-eh}" '
                     f'x2="{x+bar_w/2+6}" y2="{y_top-eh}" stroke="#222"/>')
        parts.append(f'<line x1="{x+bar_w/2-6}" y1="{y_top+eh}" '
                     f'x2="{x+bar_w/2+6}" y2="{y_top+eh}" stroke="#222"/>')
        # value label
        parts.append(f'<text x="{x+bar_w/2}" y="{y_top-12}" '
                     f'text-anchor="middle" font-size="11" '
                     f'fill="#222">{mean:.1f} ± {sd:.1f}</text>')
        # x-axis label
        parts.append(f'<text x="{x+bar_w/2}" y="{plot_y0+18}" '
                     f'text-anchor="middle" font-size="12" '
                     f'fill="#222">{f}</text>')

    # overall
    overall_rates = [r["overall_rate"] for r in runs]
    om = statistics.mean(overall_rates) * 100
    osd = statistics.pstdev(overall_rates) * 100 if len(overall_rates) > 1 else 0.0
    parts.append(f'<text x="{plot_x0+plot_w-10}" y="{plot_y0-plot_h+30}" '
                 f'text-anchor="end" font-size="16" font-weight="600" fill="#3a6ea5">'
                 f'overall = {om:.2f} ± {osd:.2f}%</text>')

    parts.append(SVG_FOOTER)
    return "\n".join(parts)


def fig3_distractors() -> str:
    runs = json.loads((DATA / "distractors.json").read_text())
    families = ["supersession", "decay", "amnesia", "purge", "drift"]
    by_d: dict[int, list] = {}
    for r in runs:
        by_d.setdefault(r["distractors"], []).append(r)

    d_values = sorted(by_d)
    ovs = []
    amnesia = []
    for d in d_values:
        rs = by_d[d]
        ovs.append((statistics.mean(r["overall_rate"] for r in rs),
                    statistics.pstdev(r["overall_rate"] for r in rs) if len(rs) > 1 else 0))
        amnesia.append((statistics.mean(r["by_family"]["amnesia"]["rate"] for r in rs),
                        statistics.pstdev(r["by_family"]["amnesia"]["rate"] for r in rs) if len(rs) > 1 else 0))

    w, h = 800, 440
    parts = [SVG_HEADER.format(w=w, h=h)]
    parts.append(f'<text x="40" y="36" font-size="20" font-weight="600">'
                 f'Distractor robustness: overall vs. amnesia</text>')
    parts.append(f'<text x="40" y="60" font-size="12" fill="#666">'
                 f'mean over seeds 42–44 at scale=50; error bars = ±1σ; '
                 f'x-axis: filler facts per case</text>')

    px0, py0 = 80, 380
    pw, ph = 680, 300
    parts.append(f'<line x1="{px0}" y1="{py0}" x2="{px0+pw}" y2="{py0}" stroke="#333"/>')
    parts.append(f'<line x1="{px0}" y1="{py0}" x2="{px0}" y2="{py0-ph}" stroke="#333"/>')

    for v in [80, 85, 90, 95, 100]:
        y = py0 - (v - 70) / 30 * ph
        parts.append(f'<line x1="{px0-4}" y1="{y}" x2="{px0}" y2="{y}" stroke="#333"/>')
        parts.append(f'<text x="{px0-8}" y="{y+4}" text-anchor="end" font-size="11" fill="#444">{v}%</text>')
        if v < 100:
            parts.append(f'<line x1="{px0}" y1="{y}" x2="{px0+pw}" y2="{y}" stroke="#eee"/>')

    # Log-ish x axis: d ∈ {4, 10, 25, 50, 100}
    def x_of(d):
        import math
        x_min, x_max = math.log10(4), math.log10(100)
        f = (math.log10(d) - x_min) / (x_max - x_min)
        return px0 + 30 + (pw - 60) * f

    for d in d_values:
        x = x_of(d)
        parts.append(f'<text x="{x}" y="{py0+18}" text-anchor="middle" font-size="12" fill="#222">d={d}</text>')

    def plot_series(ys: list, color: str, label: str, y_label: int):
        path_d = []
        for i, d in enumerate(d_values):
            x = x_of(d)
            m, s = ys[i]
            y = py0 - (m * 100 - 70) / 30 * ph
            cmd = "M" if i == 0 else "L"
            path_d.append(f"{cmd}{x},{y}")
            # error bar
            eh = s * 100 / 30 * ph
            parts.append(f'<line x1="{x}" y1="{y-eh}" x2="{x}" y2="{y+eh}" stroke="{color}" stroke-width="1.2"/>')
            # point
            parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
            # value label
            parts.append(f'<text x="{x}" y="{y-12}" text-anchor="middle" font-size="10" fill="{color}">{m*100:.1f}</text>')
        parts.append(f'<path d="{" ".join(path_d)}" stroke="{color}" stroke-width="2" fill="none"/>')
        # legend
        parts.append(f'<circle cx="{px0+pw-160}" cy="{y_label}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{px0+pw-150}" y="{y_label+4}" font-size="13" fill="{color}">{label}</text>')

    plot_series(ovs, "#3a6ea5", "overall", 100)
    plot_series(amnesia, "#cc5500", "amnesia only", 122)

    parts.append(SVG_FOOTER)
    return "\n".join(parts)


def main() -> None:
    f1 = fig1_depth_axis()
    (FIG / "fig1_depth_axis.svg").write_text(f1, encoding="utf-8")
    print(f"wrote {FIG / 'fig1_depth_axis.svg'}")

    if (DATA / "variance.json").exists():
        f2 = fig2_variance()
        (FIG / "fig2_variance.svg").write_text(f2, encoding="utf-8")
        print(f"wrote {FIG / 'fig2_variance.svg'}")

    if (DATA / "distractors.json").exists():
        f3 = fig3_distractors()
        (FIG / "fig3_distractors.svg").write_text(f3, encoding="utf-8")
        print(f"wrote {FIG / 'fig3_distractors.svg'}")


if __name__ == "__main__":
    main()
