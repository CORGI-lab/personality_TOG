"""
=============================================================================
FIGURE SCRIPT — Spearman Trait-Level Correlation Plots
Paper: Persona-Guided LLM Agents for Task-Oriented Dialogue
=============================================================================

WHAT THIS PRODUCES
------------------
Five figures, all saved as PDF (paper-ready) and PNG (preview):

  Figure 1 — Four-panel overview (Score/Rank × I-N/T-N)          [main figure]
  Figure 2 — Score ↔ I-N with Spearman rank conversion detail    [step-by-step]
  Figure 3 — Score ↔ I-N vs Rank ↔ I-N side-by-side             [Inform only]
  Figure 4 — Score ↔ T-N vs Rank ↔ T-N side-by-side             [Try only]
  Figure 5 — Sensitivity: all cells (n=60) with model markers    [cell-level]

OUTPUT FILES
------------
  fig1_four_panel.pdf / .png
  fig2_rank_conversion.pdf / .png
  fig3_inform_comparison.pdf / .png
  fig4_try_comparison.pdf / .png
  fig5_cell_level.pdf / .png

HOW TO RUN
----------
  python3 figures_spearman_trait.py

  Outputs are written to the same directory as this script.

AVG. RANK SCALE (important for interpretation)
----------------------------------------------
  1 = intended trait was the MOST evident trait detected
  2 = second most evident
  3 = third most evident
  4 = intended trait was ABSENT from top-3 (sentinel value —
      complete detection failure, not just a low rank)
  Values near 4 (introversive=3.49, closed=3.95) mean the trait
  almost never appeared in the evaluator's top-3 predictions.

DEPENDENCIES
------------
  pip install numpy scipy matplotlib
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import rankdata
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── colour / style constants ──────────────────────────────────────────────────
C_STRONG   = "#1D9E75"   # score > 4.5  (strongly realized)
C_MODERATE = "#EF9F27"   # score 2.0–4.5
C_WEAK     = "#E24B4A"   # score < 2.0
C_MIRROR   = "#888888"   # disagreeable (mirroring artefact)
C_FIT_SIG  = "#1D9E75"   # fit line when significant
C_FIT_NS   = "#BBBBBB"   # fit line when not significant
ALPHA_PT   = 0.85

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         9,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.color":       "#EEEEEE",
    "grid.linewidth":   0.6,
    "figure.dpi":       150,
})


# =============================================================================
# DATA
# =============================================================================

TRAITS = [
    "introversive", "closed", "stable", "Extraversion", "Agreeableness",
    "Conscientiousness", "Neuroticism", "Openness", "disagreeable",
    "unconscientious",
]
SHORT = [
    "Intro.", "Closed", "Stable", "Extra.", "Agree.",
    "Consc.", "Neuro.", "Open.", "Disagr.", "Uncon.",
]

# Trait-averaged values (mean of 6 model-domain combinations each)
# Source: Tables 5 & 6, averaged per trait
SC  = np.array([0.919, 0.102, 3.486, 4.991, 4.847,
                4.187, 4.928, 4.909, 4.788, 2.737])   # Avg. Intended Score
RK  = np.array([3.488, 3.945, 2.003, 1.000, 1.005,
                1.201, 1.004, 1.028, 1.007, 2.213])   # Avg. Rank (1-4)
IN  = np.array([-20.0,   1.3,  18.0,  52.0,  44.3,
                 30.7,  26.3,  70.0, -43.3,  18.3])   # Avg. I-N%
TN  = np.array([ 45.0,  29.0,  36.3,  29.3,  35.3,
                 19.3,  38.7,  21.0,  -6.0,  34.0])   # Avg. T-N%
IN_SD = np.array([38.2, 23.8, 23.0, 20.7, 31.9,
                  18.6, 25.0,  7.9, 44.9, 16.2])      # SD of I-N across datasets
TN_SD = np.array([17.3, 16.9, 15.0,  7.1, 14.5,
                  25.4, 17.5, 29.0, 21.3, 13.9])      # SD of T-N across datasets

# Cell-level data (n=60) for Figure 5
hotel_gpt4 = [
    ("introversive",-22,38,0.207,3.913),("closed",-42,10,0.020,3.987),
    ("stable",-12,26,3.173,2.653),("Extraversion",60,28,4.993,1.000),
    ("Agreeableness",56,20,4.887,1.000),("Conscientiousness",-8,-10,4.047,1.327),
    ("Neuroticism",10,20,4.920,1.007),("Openness",68,8,4.793,1.013),
    ("disagreeable",-12,-24,4.580,1.020),("unconscientious",-10,10,2.940,2.187),
]
hotel_qwen3 = [
    ("introversive",4,16,0.267,3.873),("closed",22,12,0.000,4.000),
    ("stable",32,10,3.553,1.833),("Extraversion",20,16,5.000,1.000),
    ("Agreeableness",52,36,4.820,1.000),("Conscientiousness",44,-4,4.053,1.053),
    ("Neuroticism",40,48,4.993,1.000),("Openness",76,0,4.987,1.000),
    ("disagreeable",-56,-32,4.780,1.020),("unconscientious",20,38,2.193,2.553),
]
hotel_gemini = [
    ("introversive",-40,40,1.487,3.133),("closed",8,28,0.060,3.980),
    ("stable",-2,52,3.693,1.753),("Extraversion",36,36,4.993,1.000),
    ("Agreeableness",52,40,4.993,1.000),("Conscientiousness",28,32,4.840,1.000),
    ("Neuroticism",40,68,5.000,1.000),("Openness",84,84,4.980,1.007),
    ("disagreeable",-80,-12,4.960,1.000),("unconscientious",8,32,4.373,1.033),
]
rest_gpt4 = [
    ("introversive",24,64,1.420,3.173),("closed",8,22,0.393,3.760),
    ("stable",32,52,3.260,2.213),("Extraversion",76,34,4.973,1.000),
    ("Agreeableness",52,60,4.593,1.020),("Conscientiousness",40,40,3.507,1.807),
    ("Neuroticism",0,16,4.673,1.020),("Openness",60,22,4.827,1.020),
    ("disagreeable",40,28,4.527,1.000),("unconscientious",40,56,1.260,3.413),
]
rest_qwen3 = [
    ("introversive",-92,44,0.240,3.867),("closed",-16,46,0.000,4.000),
    ("stable",54,44,3.580,1.747),("Extraversion",44,26,4.987,1.000),
    ("Agreeableness",78,40,4.787,1.013),("Conscientiousness",48,58,4.073,1.020),
    ("Neuroticism",0,36,4.980,1.000),("Openness",64,4,4.913,1.040),
    ("disagreeable",-60,16,4.960,1.000),("unconscientious",20,40,2.027,2.660),
]
rest_gemini = [
    ("introversive",6,68,1.895,2.969),("closed",28,56,0.140,3.940),
    ("stable",4,34,3.660,1.820),("Extraversion",76,36,5.000,1.000),
    ("Agreeableness",-24,16,5.000,1.000),("Conscientiousness",32,0,4.600,1.000),
    ("Neuroticism",68,44,5.000,1.000),("Openness",68,8,4.953,1.087),
    ("disagreeable",-92,-12,4.920,1.000),("unconscientious",32,28,3.627,1.433),
]

ALL_DS = {
    ("Hotel",      "GPT-4") : hotel_gpt4,
    ("Hotel",      "Qwen3") : hotel_qwen3,
    ("Hotel",      "Gemini"): hotel_gemini,
    ("Restaurant", "GPT-4") : rest_gpt4,
    ("Restaurant", "Qwen3") : rest_qwen3,
    ("Restaurant", "Gemini"): rest_gemini,
}

MODEL_MARKERS = {"GPT-4": "o", "Qwen3": "s", "Gemini": "^"}
MODEL_COLORS  = {"GPT-4": "#534AB7", "Qwen3": "#1D9E75", "Gemini": "#EF9F27"}


# =============================================================================
# HELPERS
# =============================================================================

def pt_color(sc, trait):
    if trait == "disagreeable": return C_MIRROR
    if sc > 4.5:  return C_STRONG
    if sc >= 2.0: return C_MODERATE
    return C_WEAK

def pt_size_bubble(sd, base=60, scale=3.5):
    return max(35, min(260, base + sd * scale))

def spearman_fit_line(x, y):
    """Return two endpoints of a line through the first and last sorted x."""
    idx = np.argsort(x)
    return x[idx[[0, -1]]], y[idx[[0, -1]]]

def add_spearman_fit(ax, x, y, sig, lw=1.5):
    xs, ys = spearman_fit_line(x, y)
    color = C_FIT_SIG if sig else C_FIT_NS
    ax.plot(xs, ys, linestyle="--", color=color, linewidth=lw,
            zorder=1, alpha=0.8)

def label_trait(ax, xi, yi, label, invert_x=False, custom_offset=None):
    """
    Place label with a thin connector arrow to its point.
    Stable and Uncon. are deliberately placed on opposite sides
    so their labels never cross even when the points are close.
    custom_offset = (dx, dy) in data-space units.
    """
    defaults = {
        "Disagr.": (-0.55, -14),
        "Open.":   ( 0.08,   8),
        "Closed":  (-0.55,   8) if invert_x else ( 0.08,  -9),
        "Intro.":  ( 0.08,   9),
        "Extra.":  ( 0.08,  -9),
        "Consc.":  ( 0.08,   8),
        "Uncon.":  ( 0.08,  -9),   # right + below
        "Neuro.":  (-0.55,   8),
        "Agree.":  ( 0.08,   8),
        "Stable":  (-0.55,  -9),   # left + below  (opposite side from Uncon.)
    }
    dx, dy = custom_offset if custom_offset else defaults.get(label, (0.08, 6))
    ax.annotate(
        label,
        xy=(xi, yi),
        xytext=(xi + dx, yi + dy),
        fontsize=7.5,
        color="#444444",
        annotation_clip=False,
        arrowprops=dict(
            arrowstyle="-",
            color="#AAAAAA",
            lw=0.7,
            shrinkA=0,
            shrinkB=3,
        ),
    )

def rho_text(ax, rho, p, pid):
    """
    Place stat annotation in the corner that has the least data.
    Panels (a)(b): data clusters top-right → place box bottom-left.
    Panels (c)(d): data clusters top-right and middle → place box bottom-left.
    Box has a white background so it never merges with circles.
    """
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    # Two-line text: ρ on first line, p + stars on second
    txt = f"ρ = {rho:+.3f}\np = {p:.3f}  {stars}"
    # Significant → dark teal; not significant → dark gray
    # Both are distinct from the circle colors (green/orange/red/gray)
    fc   = "#0B4F3A" if p < 0.05 else "#3A3A3A"
    ec   = "#1D9E75" if p < 0.05 else "#AAAAAA"
    ax.text(
        0.04, 0.06, txt,
        transform=ax.transAxes,
        fontsize=7.8,
        ha="left", va="bottom",
        color="white",
        fontweight="bold" if p < 0.05 else "normal",
        linespacing=1.5,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor=fc,
            edgecolor=ec,
            linewidth=0.8,
            alpha=0.88,
        ),
    )

def hline_zero(ax):
    ax.axhline(0, color="#CCCCCC", linewidth=0.9, zorder=0)

def save(fig, name):
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=200)
    print(f"  Saved: {name}.pdf  /  {name}.png")

def make_legend(extra_patch=None):
    patches = [
        mpatches.Patch(color=C_STRONG,   label="Score > 4.5  (strongly realized)"),
        mpatches.Patch(color=C_MODERATE, label="Score 2.0–4.5  (moderate)"),
        mpatches.Patch(color=C_WEAK,     label="Score < 2.0  (weakly realized)"),
        mpatches.Patch(color=C_MIRROR,   label="disagreeable†  (mirroring artefact)"),
    ]
    if extra_patch:
        patches.append(extra_patch)
    return patches


# =============================================================================
# FIGURE 1 — Four-panel overview (2 × 2)
# =============================================================================

def fig1_four_panel():
    print("\nGenerating Figure 1 — four-panel overview...")

    # Per-panel label offsets (dx in x-axis units, dy in pp).
    # Small values keep labels tight against their circles.
    OFFSETS = {
        "a": {  # Score ↔ I-N
            "Intro.":  ( 0.08,   5), "Closed":  ( 0.08,   4),
            "Stable":  (-0.42,  -6), "Extra.":  ( 0.08,  -6),
            "Agree.":  ( 0.08,   5), "Consc.":  (-0.45,   5),
            "Neuro.":  (-0.45,  -6), "Open.":   ( 0.08,   5),
            "Disagr.": (-0.45,  -8), "Uncon.":  ( 0.08,   5),
        },
        "b": {  # Rank ↔ I-N (x inverted)
            "Intro.":  ( 0.06,  -8), "Closed":  (-0.42,   5),
            "Stable":  ( 0.06,  -6), "Extra.":  ( 0.06,  -6),
            "Agree.":  ( 0.06,   5), "Consc.":  ( 0.32,   5),
            "Neuro.":  ( 0.32,  -6), "Open.":   ( 0.06,   5),
            "Disagr.": ( 0.06,  -8), "Uncon.":  (-0.42,   5),
        },
        "c": {  # Score ↔ T-N
            "Intro.":  ( 0.08,   5), "Closed":  ( 0.08,  -6),
            "Stable":  (-0.42,  -6), "Extra.":  ( 0.08,  -6),
            "Agree.":  ( 0.08,   5), "Consc.":  (-0.45,  -6),
            "Neuro.":  ( 0.35,   5), "Open.":   (-0.45,  -6),
            "Disagr.": ( 0.08,  -6), "Uncon.":  (-0.42,   5),
        },
        "d": {  # Rank ↔ T-N (x inverted)
            "Intro.":  ( 0.06,   5), "Closed":  (-0.42,  -6),
            "Stable":  (-0.42,   5), "Extra.":  ( 0.06,  -6),
            "Agree.":  ( 0.35,   5), "Consc.":  ( 0.06,  -6),
            "Neuro.":  ( 0.35,   5), "Open.":   (-0.42,  -6),
            "Disagr.": ( 0.06,  -6), "Uncon.":  (-0.42,   5),
        },
    }

    panels = [
        (SC, IN, IN_SD, "Avg. Intended Score  →",    "Δ I−N  (pp)",
         *stats.spearmanr(SC, IN), False, "a"),
        (RK, IN, IN_SD, "Avg. Rank  (← stronger)",  "Δ I−N  (pp)",
         *stats.spearmanr(RK, IN), True,  "b"),
        (SC, TN, TN_SD, "Avg. Intended Score  →",    "Δ T−N  (pp)",
         *stats.spearmanr(SC, TN), False, "c"),
        (RK, TN, TN_SD, "Avg. Rank  (← stronger)",  "Δ T−N  (pp)",
         *stats.spearmanr(RK, TN), True,  "d"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 6.4))
    fig.subplots_adjust(hspace=0.52, wspace=0.44)

    for ax, (xs, ys, sds, xl, yl, rho, p, inv, pid) in zip(axes.flat, panels):
        sig = p < 0.05
        colors = [pt_color(SC[i], TRAITS[i]) for i in range(10)]
        sizes  = [pt_size_bubble(sds[i]) for i in range(10)]

        hline_zero(ax)
        add_spearman_fit(ax, xs, ys, sig)

        for i in range(10):
            ax.scatter(xs[i], ys[i], s=sizes[i], c=colors[i],
                       edgecolors="white", linewidths=0.5,
                       zorder=3, alpha=ALPHA_PT)
            dx, dy = OFFSETS[pid][SHORT[i]]
            ax.annotate(
                SHORT[i], xy=(xs[i], ys[i]),
                xytext=(xs[i] + dx, ys[i] + dy),
                fontsize=7.5, color="#333333", annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color="#BBBBBB",
                                lw=0.7, shrinkA=0, shrinkB=3),
            )

        if inv:
            ax.invert_xaxis()

        ax.set_xlabel(xl, fontsize=8.5)
        ax.set_ylabel(yl, fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.text(0.03, 0.97, f"({pid})", transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", color="#333333")
        rho_text(ax, rho, p, pid)

    fig.legend(handles=make_legend(),
               loc="lower center", ncol=4, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.04))

    # fig.suptitle(
    #     "Figure 1 — Trait Realization vs. Pairwise Satisfaction Delta  (n = 10 traits)",
    #     fontsize=9.5, y=1.01, color="#222222")

    fig.text(0.5, -0.07,
        "Point size ∝ cross-dataset SD.  Dashed line = Spearman monotone fit.  "
        "†disagreeable: system-mirroring in Qwen3/Gemini.",
        ha="center", fontsize=7.2, color="#666666", style="italic")

    save(fig, "fig1_four_panel")
    plt.close(fig)


# =============================================================================
# FIGURE 2 — Score ↔ I-N with rank conversion detail (step-by-step)
# =============================================================================

def fig2_rank_conversion():
    print("Generating Figure 2 — rank conversion detail...")

    rho, p = stats.spearmanr(SC, IN)
    sc_ranks = rankdata(SC)
    in_ranks = rankdata(IN)

    fig = plt.figure(figsize=(10, 4.2))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.42)
    ax1 = fig.add_subplot(gs[0])   # raw scatter
    ax2 = fig.add_subplot(gs[1])   # after rank conversion
    ax3 = fig.add_subplot(gs[2])   # bar: Spearman rank of each trait's I-N

    colors = [pt_color(SC[i], TRAITS[i]) for i in range(10)]
    sizes  = [pt_size_bubble(IN_SD[i]) for i in range(10)]

    # Panel A — raw scatter
    hline_zero(ax1)
    add_spearman_fit(ax1, SC, IN, sig=True)
    for i in range(10):
        ax1.scatter(SC[i], IN[i], s=sizes[i], c=colors[i],
                    edgecolors="white", linewidths=0.5, zorder=3, alpha=ALPHA_PT)
        label_trait(ax1, SC[i], IN[i], SHORT[i])
    rho_text(ax1, rho, p, "a")
    ax1.set_xlabel("Avg. Intended Score  →", fontsize=8.5)
    ax1.set_ylabel("Δ I−N  (pp)", fontsize=8.5)
    ax1.text(0.03, 0.97, "(a) Raw values", transform=ax1.transAxes,
             fontsize=9, fontweight="bold", va="top", color="#333333")

    # Panel B — rank-converted scatter (this is what Spearman correlates)
    hline_zero(ax2)
    for i in range(10):
        ax2.scatter(sc_ranks[i], in_ranks[i], s=sizes[i], c=colors[i],
                    edgecolors="white", linewidths=0.5, zorder=3, alpha=ALPHA_PT)
        ax2.annotate(SHORT[i],
                     (sc_ranks[i], in_ranks[i]),
                     xytext=(sc_ranks[i] + 0.15, in_ranks[i] + 0.25),
                     fontsize=7.5, color="#444444")
    # fit line on ranks
    r_pearson_on_ranks, _ = stats.pearsonr(sc_ranks, in_ranks)
    xs_fit = np.linspace(sc_ranks.min(), sc_ranks.max(), 50)
    m, b = np.polyfit(sc_ranks, in_ranks, 1)
    ax2.plot(xs_fit, m * xs_fit + b, "--", color=C_FIT_SIG, lw=1.5, alpha=0.8)
    ax2.text(0.97, 0.97,
             f"Pearson r on ranks = {r_pearson_on_ranks:+.3f}\n(= Spearman ρ  ✓)",
             transform=ax2.transAxes, fontsize=8, ha="right", va="top",
             color=C_FIT_SIG, fontweight="bold")
    ax2.set_xlabel("Rank of Avg. Score  (1=lowest score)", fontsize=8.5)
    ax2.set_ylabel("Rank of Δ I−N  (1=most negative)", fontsize=8.5)
    ax2.text(0.03, 0.97, "(b) After rank conversion\n(what Spearman computes)",
             transform=ax2.transAxes, fontsize=9, fontweight="bold",
             va="top", color="#333333")

    # Panel C — table as bar chart showing rank alignment
    x_pos = np.arange(10)
    width = 0.38
    order = np.argsort(sc_ranks)   # sort by Score rank low→high
    bar_colors = [colors[i] for i in order]
    ax3.bar(x_pos - width/2, sc_ranks[order], width,
            color=bar_colors, alpha=0.7, label="Rank of Score", edgecolor="white")
    ax3.bar(x_pos + width/2, in_ranks[order],  width,
            color=bar_colors, alpha=0.35, label="Rank of I−N",
            edgecolor="white", hatch="///")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([SHORT[i] for i in order], rotation=40, ha="right", fontsize=7.5)
    ax3.set_ylabel("Spearman rank  (1=smallest value)", fontsize=8)
    ax3.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax3.text(0.03, 0.97, "(c) Rank alignment per trait\n(similar heights → high ρ)",
             transform=ax3.transAxes, fontsize=9, fontweight="bold",
             va="top", color="#333333")

    fig.suptitle(
        "Figure 2 — How Spearman Computes ρ: Score ↔ I−N  (n = 10 traits)",
        fontsize=9.5, y=1.02, color="#222222")
    fig.legend(handles=make_legend(),
               loc="lower center", ncol=4, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.06))

    save(fig, "fig2_rank_conversion")
    plt.close(fig)


# =============================================================================
# FIGURE 3 — Inform only: Score ↔ I-N  vs  Rank ↔ I-N
# =============================================================================

def fig3_inform_comparison():
    print("Generating Figure 3 — Inform condition (I-N) comparison...")

    rho_sc, p_sc = stats.spearmanr(SC, IN)
    rho_rk, p_rk = stats.spearmanr(RK, IN)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.2))
    fig.subplots_adjust(wspace=0.40)

    colors = [pt_color(SC[i], TRAITS[i]) for i in range(10)]
    sizes  = [pt_size_bubble(IN_SD[i]) for i in range(10)]

    for ax, xs, xl, rho, p, inv in [
        (ax1, SC, "Avg. Intended Score  →", rho_sc, p_sc, False),
        (ax2, RK, "Avg. Rank  (← stronger realization)", rho_rk, p_rk, True),
    ]:
        hline_zero(ax)
        add_spearman_fit(ax, xs, IN, p < 0.05)
        for i in range(10):
            ax.scatter(xs[i], IN[i], s=sizes[i], c=colors[i],
                       edgecolors="white", linewidths=0.5,
                       zorder=3, alpha=ALPHA_PT)
            label_trait(ax, xs[i], IN[i], SHORT[i], invert_x=inv)
        if inv:
            ax.invert_xaxis()
            ax.set_xlabel(xl, fontsize=8.5)
        else:
            ax.set_xlabel(xl, fontsize=8.5)
        ax.set_ylabel("Δ I−N  (pp)  [Inform − Neutral]", fontsize=8.5)
        rho_text(ax, rho, p, "fig3")

        # annotate the two extreme traits
        for i, trait in enumerate(TRAITS):
            if trait in ("introversive", "closed"):
                ax.annotate(
                    "rank≈4\n(absent\nfrom top-3)" if trait == "closed" else "",
                    (xs[i], IN[i]),
                    xytext=(xs[i] + (0.3 if not inv else -0.3), IN[i] - 12),
                    fontsize=6.5, color="#E24B4A",
                    arrowprops=dict(arrowstyle="->", color="#E24B4A", lw=0.8))

    ax1.set_title("(a) Avg. Intended Score → I−N", fontsize=9, pad=8)
    ax2.set_title("(b) Avg. Rank → I−N", fontsize=9, pad=8)

    fig.suptitle(
        "Figure 3 — Inform Condition: Realization Strength Predicts Satisfaction Gain  "
        "(n = 10 traits)",
        fontsize=9.5, y=1.03, color="#222222")
    fig.legend(handles=make_legend(),
               loc="lower center", ncol=4, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.07))

    foot = ("Point size ∝ SD across 6 model–domain datasets.  "
            "Rank 4 = intended trait absent from evaluator top-3 (detection failure).")
    fig.text(0.5, -0.10, foot, ha="center", fontsize=7.2,
             color="#666666", style="italic")

    save(fig, "fig3_inform_comparison")
    plt.close(fig)


# =============================================================================
# FIGURE 4 — Try only: Score ↔ T-N  vs  Rank ↔ T-N
# =============================================================================

def fig4_try_comparison():
    print("Generating Figure 4 — Try condition (T-N) comparison...")

    rho_sc, p_sc = stats.spearmanr(SC, TN)
    rho_rk, p_rk = stats.spearmanr(RK, TN)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.2))
    fig.subplots_adjust(wspace=0.40)

    colors = [pt_color(SC[i], TRAITS[i]) for i in range(10)]
    sizes  = [pt_size_bubble(TN_SD[i]) for i in range(10)]

    for ax, xs, xl, rho, p, inv in [
        (ax1, SC, "Avg. Intended Score  →", rho_sc, p_sc, False),
        (ax2, RK, "Avg. Rank  (← stronger realization)", rho_rk, p_rk, True),
    ]:
        hline_zero(ax)
        add_spearman_fit(ax, xs, TN, p < 0.05)
        for i in range(10):
            ax.scatter(xs[i], TN[i], s=sizes[i], c=colors[i],
                       edgecolors="white", linewidths=0.5,
                       zorder=3, alpha=ALPHA_PT)
            label_trait(ax, xs[i], TN[i], SHORT[i], invert_x=inv)
        if inv:
            ax.invert_xaxis()
        ax.set_xlabel(xl, fontsize=8.5)
        ax.set_ylabel("Δ T−N  (pp)  [Try − Neutral]", fontsize=8.5)
        rho_text(ax, rho, p, "fig4")

    ax1.set_title("(a) Avg. Intended Score → T−N", fontsize=9, pad=8)
    ax2.set_title("(b) Avg. Rank → T−N", fontsize=9, pad=8)

    # add flat-trend annotation to show non-significance visually
    for ax in (ax1, ax2):
        ax.text(0.5, 0.97,
                "No significant trend — Try gains are\nrealization-independent (all p > .15)",
                transform=ax.transAxes, fontsize=7.5, ha="center", va="top",
                color="#888888", style="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DDDDDD", alpha=0.85))

    fig.suptitle(
        "Figure 4 — Try Condition: No Dependence on Realization Strength  "
        "(n = 10 traits)",
        fontsize=9.5, y=1.03, color="#222222")
    fig.legend(handles=make_legend(),
               loc="lower center", ncol=4, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.07))

    save(fig, "fig4_try_comparison")
    plt.close(fig)


# =============================================================================
# FIGURE 5 — Cell-level scatter (n=60) with model markers
# =============================================================================

def fig5_cell_level():
    print("Generating Figure 5 — cell-level scatter (n=60)...")

    # collect all 60 cells
    cells_IN, cells_SC, cells_model, cells_trait = [], [], [], []
    for (dom, mod), ds in ALL_DS.items():
        for row in ds:
            cells_IN.append(row[1])
            cells_SC.append(row[3])
            cells_model.append(mod)
            cells_trait.append(row[0])

    cells_IN = np.array(cells_IN)
    cells_SC = np.array(cells_SC)

    rho, p = stats.spearmanr(cells_SC, cells_IN)

    # sensitivity without 4 mirroring cells
    mirror = {("Hotel","Qwen3","disagreeable"),("Hotel","Gemini","disagreeable"),
              ("Restaurant","Qwen3","disagreeable"),("Restaurant","Gemini","disagreeable")}
    mask = []
    for (dom, mod), ds in ALL_DS.items():
        for row in ds:
            mask.append((dom, mod, row[0]) not in mirror)
    mask = np.array(mask)
    rho_s, p_s = stats.spearmanr(cells_SC[mask], cells_IN[mask])

    fig, ax = plt.subplots(figsize=(7, 5.2))

    hline_zero(ax)
    add_spearman_fit(ax, cells_SC, cells_IN, sig=True)

    for i, (sc, in_val, mod, trait) in enumerate(
            zip(cells_SC, cells_IN, cells_model, cells_trait)):
        c = pt_color(sc, trait)
        m = MODEL_MARKERS[mod]
        ax.scatter(sc, in_val, s=55, c=c, marker=m,
                   edgecolors="white", linewidths=0.4,
                   zorder=3, alpha=0.75)

    # add model-marker legend entries
    model_handles = [
        plt.scatter([], [], marker=MODEL_MARKERS[m], c="#888888",
                    s=55, label=m, edgecolors="white")
        for m in ["GPT-4", "Qwen3", "Gemini"]
    ]

    # annotate the 4 mirroring cells
    for (dom, mod), ds in ALL_DS.items():
        for row in ds:
            if (dom, mod, row[0]) in mirror:
                ax.annotate("",
                    xy=(row[3], row[1]),
                    xytext=(row[3] + 0.15, row[1] - 8),
                    arrowprops=dict(arrowstyle="->", color="#E24B4A",
                                   lw=1.0, alpha=0.6))

    ax.text(4.2, -72,
            "←  mirroring cells\n    (Qwen3/Gemini,\n    disagreeable)",
            fontsize=7.5, color="#E24B4A", va="top")

    rho_text(ax, rho, p, "cell")
    ax.text(0.97, 0.88,
            f"Excl. mirroring:  ρ = {rho_s:+.3f}  (n=56)",
            transform=ax.transAxes, fontsize=7.5, ha="right",
            color=C_FIT_SIG, style="italic")

    ax.set_xlabel("Avg. Intended Score  →", fontsize=9)
    ax.set_ylabel("Δ I−N  (pp)  [Inform − Neutral]", fontsize=9)
    ax.set_title(
        "Figure 5 — Cell-Level Scatter: Avg. Intended Score ↔ I−N  "
        "(n = 60 cells)",
        fontsize=9.5, pad=10, color="#222222")

    color_handles = make_legend()
    all_handles = color_handles + model_handles
    ax.legend(handles=all_handles, fontsize=7.5, frameon=False,
              loc="upper left", ncol=1)

    foot = ("Each point = one trait × model × domain cell (10 traits × 6 datasets).  "
            "Marker shape = model.\nOverall n=60 ρ=+0.421 is conservative; "
            "excluding mirroring cells raises ρ to +0.557.")
    fig.text(0.5, -0.04, foot, ha="center", fontsize=7.2,
             color="#666666", style="italic")

    save(fig, "fig5_cell_level")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Generating all Spearman trait-level correlation figures")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 60)

    fig1_four_panel()
    fig2_rank_conversion()
    fig3_inform_comparison()
    fig4_try_comparison()
    fig5_cell_level()

    print("\nDone. Five figures saved (PDF + PNG each):")
    for name in ["fig1_four_panel", "fig2_rank_conversion",
                 "fig3_inform_comparison", "fig4_try_comparison",
                 "fig5_cell_level"]:
        print(f"  {name}.pdf  /  {name}.png")
