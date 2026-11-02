"""System adaptation score (1-5) per trait x condition (N/T/O), averaged over 3 LLMs x 2 domains.
Gemma: summary JSONs (Restaurant introversive only in *_full.json). gpt-5-mini: summary JSONs from the new run."""
import json
from scipy.stats import spearmanr
G = "./storage/chatgpt_new/metric/llm_metric/hpc/sys_adaptation"
TRAITS = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness",
          "introversive", "disagreeable", "unconscientious", "stable", "closed"]
LLMS, DOMS, CASES = ["gpt4", "qwen3", "gemeni"], ["Hotel", "Restaurant"], [("neutral", "N"), ("try", "T"), ("trait", "O")]

def load(root, full_restaurant=False):
    out = {}
    for d in DOMS:
        f = f"{root}/{d}/system_trait_alignment_summary{'_full' if full_restaurant and d == 'Restaurant' else ''}.json"
        s = json.load(open(f))[d]
        for c, _ in CASES:
            for l in LLMS:
                for t in TRAITS:
                    v = s.get(c, {}).get(l, {}).get(t)
                    if v: out[(d, c, l, t)] = (v["avg_system_trait_alignment_score"], v["n_dialogues"])
    return out

J = {"Gemma": load(G, True), "GPT-5-mini": load(f"{G}/gpt5_mini")}
def avg(data, c, t=None, d=None):
    v = [data[k][0] for k in data if k[1] == c and (t is None or k[3] == t) and (d is None or k[0] == d)]
    return sum(v) / len(v) if v else float("nan")

print(f"{'Trait':<18}" + "".join(f"| {n:^20}" for n in list(J) + ["Two-judge avg"]))
print(f"{'':<18}" + "| N     T     O      " * 3)
rows = TRAITS + [None]
for t in rows:
    line = f"{t or 'Average':<18}"
    for n in J:
        line += "| " + " ".join(f"{avg(J[n], c, t):.2f}" for c, _ in CASES) + "   "
    line += "| " + " ".join(f"{(avg(J['Gemma'], c, t) + avg(J['GPT-5-mini'], c, t)) / 2:.2f}" for c, _ in CASES)
    print(line)
print("\nPer domain averages:")
for d in DOMS:
    print(d, {n: [round(avg(J[n], c, d=d), 2) for c, _ in CASES] for n in J})
print("\nCoverage (n dialogues):", {n: sum(v[1] for v in J[n].values()) for n in J})
keys = sorted(set(J["Gemma"]) & set(J["GPT-5-mini"]))
r, p = spearmanr([J["Gemma"][k][0] for k in keys], [J["GPT-5-mini"][k][0] for k in keys])
print(f"Judge agreement over {len(keys)} cells (domain x case x llm x trait): Spearman rho={r:.3f}, p={p:.2g}")

from scipy.stats import wilcoxon
print("\nWilcoxon over 60 matched cells (domain x llm x trait):")
cells = [(d, l, t) for d in DOMS for l in LLMS for t in TRAITS]
avg2 = {k: (J["Gemma"][k][0] + J["GPT-5-mini"][k][0]) / 2 for k in keys}
for name, data in [("Gemma", {k: v[0] for k, v in J["Gemma"].items()}), ("GPT-5-mini", {k: v[0] for k, v in J["GPT-5-mini"].items()}), ("Two-judge avg", avg2)]:
    res = []
    for a, b in [("try", "neutral"), ("trait", "neutral"), ("trait", "try")]:
        x = [data[(d, a, l, t)] - data[(d, b, l, t)] for d, l, t in cells]
        res.append(f"{a[0].upper() if a!='trait' else 'O'}-{b[0].upper() if b!='trait' else 'O'}: mean {sum(x)/60:+.2f}, p={wilcoxon(x).pvalue:.1e}")
    print(f"  {name:<14} " + " | ".join(res))
# do the judges agree on the gain per trait?
for a, b in [("trait", "neutral"), ("try", "neutral")]:
    g = [avg(J["Gemma"], a, t) - avg(J["Gemma"], b, t) for t in TRAITS]
    m = [avg(J["GPT-5-mini"], a, t) - avg(J["GPT-5-mini"], b, t) for t in TRAITS]
    print(f"Per-trait gain {a}-{b}: Spearman(Gemma, GPT-5-mini) over 10 traits rho={spearmanr(g, m)[0]:+.3f}")
