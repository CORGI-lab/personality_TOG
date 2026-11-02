"""
Realization-to-satisfaction figures, averaged over two judges (Gemma + GPT-5-mini).

Reads the result files directly (no hard-coded numbers):
  - User Trait Score: llm_evaluation/USER/New_1to5_scale/User_trait_score/{gemma,gpt5_mini}
  - Pairwise satisfaction: llm_metric_SYS/New_1to5_scale/User_Satisfaction_Pairly/{gemma,gpt5-mini}

Both quantities are averaged over the two judges per model x domain x trait cell.
  Delta O-N = Oracle win rate - Neutral win rate (pp), Delta T-N analogously.

Outputs (in Tests/):
  fig_realization_two_judges_n10.{pdf,png}  trait level, 10 points per panel
  fig_realization_two_judges_n60.{pdf,png}  cell level, 60 points per panel
"""
import glob
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure import (  # reuse colours, legend and stat box of the original figures
    C_FIT_SIG, C_FIT_NS, ALPHA_PT, MODEL_MARKERS, pt_color, pt_size_bubble,
    rho_text, hline_zero, make_legend,
)

OUT_DIR = "figures"
H = "./storage/chatgpt_new/metric/llm_metric/hpc"
PAIR = f"{H}/llm_metric_SYS/New_1to5_scale/User_Satisfaction_Pairly"
TS = f"{H}/llm_evaluation/USER/New_1to5_scale/User_trait_score"

TRAITS = ["introversive", "closed", "stable", "Extraversion", "Agreeableness",
          "Conscientiousness", "Neuroticism", "Openness", "disagreeable", "unconscientious"]
SHORT = dict(zip(TRAITS, ["Intro.", "Closed", "Stable", "Extra.", "Agree.",
                          "Consc.", "Neuro.", "Open.", "Disagr.", "Uncon."]))
LLMS = {"gpt4": "GPT-4", "qwen3": "Qwen3", "gemeni": "Gemini"}
DOMS = ["Hotel", "Restaurant"]
CELLS = [(d, l, t) for d in DOMS for l in LLMS for t in TRAITS]
MIRROR = {(d, l, "disagreeable") for d in DOMS for l in ("qwen3", "gemeni")}


# ----------------------------------------------------------------------------- data

def trait_score(judge):
    out = {}
    for d in DOMS:
        s = json.load(open(f"{TS}/{judge}/{d}/avg_across_cases_by_llm.json"))[d]
        for l in LLMS:
            for t in TRAITS:
                out[(d, l, t)] = s[l][t]["avg_intended_score_all_across_cases"]
    return out


def delta_gemma(comp, cell):
    """Gemma ran in fixed order (Neutral = Dialogue A), so its winner is case-level."""
    d, l, t = cell
    c = Counter()
    for f in glob.glob(f"{PAIR}/gemma/{d}/{comp}/{l}/{t}/pair_*/response.txt"):
        try:
            c[json.loads(re.search(r"\{.*\}", open(f).read(), re.S).group(0))["winner"]] += 1
        except Exception:
            pass
    return 100 * (c["B"] - c["A"]) / (c["A"] + c["B"] + c["tie"])


def delta_gpt5(comp, cell):
    d, l, t = cell
    s = json.load(open(f"{PAIR}/gpt5-mini/{d}/{comp}/{l}/{t}/summary.json"))
    return 100 * (s["win_rate_B"] - s["win_rate_A"])


def load():
    tg, tm = trait_score("gemma"), trait_score("gpt5_mini")
    score = {c: (tg[c] + tm[c]) / 2 for c in CELLS}
    delta = {}
    for comp, name in [("neutral_vs_trait", "O-N"), ("neutral_vs_try", "T-N")]:
        delta[name] = {c: (delta_gemma(comp, c) + delta_gpt5(comp, c)) / 2 for c in CELLS}
    return score, delta


# ----------------------------------------------------------------------------- plotting

YLABEL = {"O-N": "Δ O−N  (pp)  [Oracle − Neutral]", "T-N": "Δ T−N  (pp)  [Try − Neutral]"}


def stat_box(ax, rho, p, x=0.04, y=0.06, va="bottom"):
    stars = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
    ptxt = "p < .001" if p < .001 else f"p = {p:.3f}"
    sig = p < .05
    ax.text(x, y, f"ρ = {rho:+.3f}\n{ptxt}  {stars}", transform=ax.transAxes, fontsize=7.8,
            ha="left", va=va, color="white", fontweight="bold" if sig else "normal", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#0B4F3A" if sig else "#3A3A3A",
                      edgecolor="#1D9E75" if sig else "#AAAAAA", linewidth=0.8, alpha=0.88), zorder=5)


def fit_line(ax, x, y, sig):
    """Theil-Sen trend line (median of pairwise slopes): rank-based like the Spearman
    test, so its direction always agrees with rho -- a least-squares line can disagree
    when a few large values pull it (e.g. Delta T-N: OLS slope < 0 while rho > 0)."""
    m, b, _, _ = stats.theilslopes(y, x)
    xs = np.array([min(x), max(x)])
    ax.plot(xs, m * xs + b, linestyle="--", color=C_FIT_SIG if sig else C_FIT_NS,
            linewidth=1.6, zorder=1, alpha=0.85)


def label_points(ax, xs, ys, labels, yspan):
    # offsets in (x units, fraction of y-span); spread labels around crowded top-right
    off = {"Open.": (0.0, .07), "Agree.": (0.06, .05), "Extra.": (0.12, -.03), "Neuro.": (0.10, -.06),
           "Consc.": (-0.45, .06), "Disagr.": (-0.95, 0.0), "Stable": (-0.40, -.07),
           "Uncon.": (0.08, .06), "Intro.": (0.08, .07), "Closed": (0.08, .07)}
    for x, y, lab in zip(xs, ys, labels):
        dx, dy = off.get(lab, (0.08, .05))
        ax.annotate(lab, xy=(x, y), xytext=(x + dx, y + dy * yspan), fontsize=7.5,
                    color="#444444", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.7, shrinkA=0, shrinkB=3))


def fig_trait_level(score, delta):
    fig, axes = plt.subplots(2, 1, figsize=(4.6, 7.4))
    fig.subplots_adjust(hspace=0.32)
    xs = np.array([np.mean([score[(d, l, t)] for d in DOMS for l in LLMS]) for t in TRAITS])
    for ax, key, pid in zip(axes, ["O-N", "T-N"], ["(a)", "(b)"]):
        cells = np.array([[delta[key][(d, l, t)] for d in DOMS for l in LLMS] for t in TRAITS])
        ys, sds = cells.mean(axis=1), cells.std(axis=1, ddof=1)
        rho, p = stats.spearmanr(xs, ys)
        hline_zero(ax)
        fit_line(ax, xs, ys, p < 0.05)
        for i, t in enumerate(TRAITS):
            ax.scatter(xs[i], ys[i], s=pt_size_bubble(sds[i]), c=pt_color(xs[i], t),
                       edgecolors="white", linewidths=0.5, zorder=3, alpha=ALPHA_PT)
        yspan = ys.max() - ys.min()
        label_points(ax, xs, ys, [SHORT[t] for t in TRAITS], yspan)
        stat_box(ax, rho, p)
        ax.set_xlim(-0.15, 5.35)
        ax.set_ylim(ys.min() - 0.18 * yspan, ys.max() + 0.18 * yspan)
        ax.set_xlabel("Avg. User Trait Score  →", fontsize=8.5)
        ax.set_ylabel(YLABEL[key], fontsize=8.5)
        ax.text(0.03, 0.97, pid, transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="top", color="#333333")
    fig.legend(handles=make_legend(), loc="lower center", ncol=2, fontsize=6.8,
               frameon=False, bbox_to_anchor=(0.5, -0.045))
    fig.text(0.5, -0.075, "Trait level (n = 10); each point averages 6 model × domain cells and two judges "
             "(Gemma, GPT-5-mini).\nPoint size ∝ SD across cells. Dashed line = Theil–Sen trend; statistics are Spearman.",
             ha="center", fontsize=6.6, color="#666666", style="italic")
    save(fig, "fig_realization_two_judges_n10")
    return fig


def fig_cell_level(score, delta):
    fig, axes = plt.subplots(2, 1, figsize=(4.6, 7.4))
    fig.subplots_adjust(hspace=0.32)
    xs = np.array([score[c] for c in CELLS])
    keep = np.array([c not in MIRROR for c in CELLS])
    for ax, key, pid in zip(axes, ["O-N", "T-N"], ["(a)", "(b)"]):
        ys = np.array([delta[key][c] for c in CELLS])
        rho, p = stats.spearmanr(xs, ys)
        hline_zero(ax)
        fit_line(ax, xs, ys, p < 0.05)
        for c, x, y in zip(CELLS, xs, ys):
            ax.scatter(x, y, s=42, c=pt_color(x, c[2]), marker=MODEL_MARKERS[LLMS[c[1]]],
                       edgecolors="white", linewidths=0.4, zorder=3, alpha=0.8)
        stat_box(ax, rho, p, x=0.30, y=0.04, va="bottom")
        if key == "O-N":
            r2, p2 = stats.spearmanr(xs[keep], ys[keep])
            ax.text(0.30, 0.30, f"excl. disagreeable × Qwen3/Gemini\n(4 cells): ρ = {r2:+.3f}, p {'< .001' if p2 < .001 else f'= {p2:.3f}'}, n = 56",
                    transform=ax.transAxes, fontsize=6.8, ha="left", va="top",
                    color=C_FIT_SIG, style="italic")
        ax.set_xlim(-0.15, 5.25)
        ax.set_xlabel("User Trait Score (cell)  →", fontsize=8.5)
        ax.set_ylabel(YLABEL[key], fontsize=8.5)
        ax.text(0.03, 0.97, pid, transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="top", color="#333333")
    model_handles = [plt.scatter([], [], marker=MODEL_MARKERS[m], c="#888888", s=40, label=m,
                                 edgecolors="white") for m in ["GPT-4", "Qwen3", "Gemini"]]
    fig.legend(handles=make_legend() + model_handles, loc="lower center", ncol=2, fontsize=6.8,
               frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.text(0.5, -0.095, "Cell level (n = 60 = 10 traits × 3 models × 2 domains), averaged over two judges. "
             "Marker = model.", ha="center", fontsize=6.6, color="#666666", style="italic")
    save(fig, "fig_realization_two_judges_n60")
    return fig


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    print(f"Saved {name}.pdf / .png")


if __name__ == "__main__":
    score, delta = load()
    for key in ("O-N", "T-N"):
        xs = [np.mean([score[(d, l, t)] for d in DOMS for l in LLMS]) for t in TRAITS]
        ys = [np.mean([delta[key][(d, l, t)] for d in DOMS for l in LLMS]) for t in TRAITS]
        r10, p10 = stats.spearmanr(xs, ys)
        r60, p60 = stats.spearmanr([score[c] for c in CELLS], [delta[key][c] for c in CELLS])
        print(f"{key}: n=10 rho={r10:+.3f} p={p10:.3f} | n=60 rho={r60:+.3f} p={p60:.4f}")
    fig_trait_level(score, delta)
    fig_cell_level(score, delta)
