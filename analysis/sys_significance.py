"""Significance of Try/Oracle vs Neutral on CS, IR, TR for both system-quality judges."""
import glob, json, os, re
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

H = "./storage/chatgpt_new/metric/llm_metric/hpc/llm_metric_SYS/New_1to5_scale"
JUDGES = {
    "Gemma": (f"{H}/System_metrics/gemma4", {"CS": "Constraint Satisfaction Rate", "IR": "Inform Rate", "TR": "Dialogue Hallucination Rate"}),
    "GPT-5-mini": (f"{H}/System_metrics/gpt5_mini", {"CS": "Constraint Satisfaction Rate", "IR": "Inform Rate", "TR": "Dialogue Truthfulness Rate"}),
}
CASES = {"N": "neutral", "T": "try", "O": "trait"}


def scores(path, keys):
    s = open(path, encoding="utf-8").read()
    out = {}
    for m, k in keys.items():
        # TR was named "Dialogue Hallucination Rate" in most Gemma runs and
        # "Dialogue Truthfulness Rate" in later ones -- accept either.
        names = ["Dialogue Hallucination Rate", "Dialogue Truthfulness Rate"] if m == "TR" else [k]
        for name in names:
            hit = re.search(rf'"{re.escape(name)}"\s*:\s*\{{\s*"score"\s*:\s*(\d+(?:\.\d+)?)', s)
            if hit:
                out[m] = float(hit.group(1))
                break
    return out if len(out) == 3 else None


def load(root, keys):
    """rows: (domain, llm, trait, case, dialogue) -> {CS, IR, TR}"""
    rows, bad = {}, 0
    for f in glob.glob(f"{root}/*/*/*/*/*/response.txt"):
        dom, case, llm, trait, name = f.split("/")[-6:-1]
        sc = scores(f, keys)
        if sc is None:
            bad += 1
            continue
        rows[(dom, llm, trait, case, name)] = sc
    return rows, bad


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def fmt_p(p):
    return "<.001" if p < .001 else f"{p:.3f}"


data = {j: load(*v) for j, v in JUDGES.items()}
# Two-judge average per dialogue (both judges scored the same dialogues)
common = set(data["Gemma"][0]) & set(data["GPT-5-mini"][0])
data["Average"] = ({k: {m: (data["Gemma"][0][k][m] + data["GPT-5-mini"][0][k][m]) / 2 for m in ("CS", "IR", "TR")}
                    for k in common}, 0)

for judge, (rows, bad) in data.items():
    print(f"\n==================== {judge}  ({len(rows)} dialogues scored, {bad} unparseable) ====================")
    tests = []
    for m in ("CS", "IR", "TR"):
        by_case = {c: [v[m] for k, v in rows.items() if k[3] == cname] for c, cname in CASES.items()}
        cells = {}
        for (dom, llm, trait, case, _), v in rows.items():
            cells.setdefault((dom, llm, trait), {}).setdefault(case, []).append(v[m])
        cell_keys = sorted(k for k in cells if all(c in cells[k] for c in CASES.values()))
        cm = {c: np.array([np.mean(cells[k][cname]) for k in cell_keys]) for c, cname in CASES.items()}
        for c in ("T", "O"):
            a, b = np.array(by_case[c]), np.array(by_case["N"])
            u = mannwhitneyu(a, b, alternative="two-sided")
            rb = 2 * u.statistic / (len(a) * len(b)) - 1          # rank-biserial, + means c > N
            d = cm[c] - cm["N"]
            w = wilcoxon(d, alternative="two-sided", zero_method="wilcox")
            tests.append(dict(metric=m, cmp=f"{c} vs N", mean_c=a.mean(), mean_n=b.mean(), n=(len(a), len(b)),
                              p_mw=u.pvalue, rb=rb, cells=len(d), cells_up=int((d > 0).sum()),
                              cells_down=int((d < 0).sum()), p_w=w.pvalue))
    adj_mw, adj_w = holm(np.array([t["p_mw"] for t in tests])), holm(np.array([t["p_w"] for t in tests]))
    print(f"{'metric':<4}{'cmp':<9}{'mean':>16}{'MW p(Holm)':>12}{'r_rb':>7}   {'cells +/-':>10}{'Wilcoxon p(Holm)':>18}")
    for t, pm, pw in zip(tests, adj_mw, adj_w):
        print(f"{t['metric']:<4}{t['cmp']:<9}{t['mean_c']:>7.3f} vs {t['mean_n']:.3f}{fmt_p(pm):>12}{t['rb']:>7.3f}   "
              f"{t['cells_up']:>4}/{t['cells_down']:<5}{fmt_p(pw):>18}")
