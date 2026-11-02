"""Table I (personality realization) and the RQ1 tests.

- User Trait Score per trait, averaged over dialogues, conditions, LLMs, domains and the two
  judges (Gemma, GPT-5-mini)  -> paper average 3.520.
- Top-3 detection rate per trait and judge with a one-sided binomial test against the 30%
  chance baseline (3 of 10 labels), Holm-corrected.
- Friedman test: does User Trait Score differ across Neutral / Try / Oracle?
Run from the repository root:  python analysis/table_trait_realization.py
"""
import glob, json
from collections import defaultdict
import numpy as np
from scipy.stats import binomtest, friedmanchisquare, wilcoxon

RAW = "storage/chatgpt_new/metric/llm_metric/hpc/llm_metric_USER/New_1to5_scale/User_trait_score"
SCORED = "storage/chatgpt_new/metric/llm_metric/hpc/llm_evaluation/USER/New_1to5_scale/User_trait_score"
JUDGES = ["gemma", "gpt5_mini"]
TRAITS = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness",
          "introversive", "disagreeable", "unconscientious", "stable", "closed"]
LLMS, DOMS, CASES = ["gpt4", "qwen3", "gemeni"], ["Hotel", "Restaurant"], ["neutral", "try", "trait"]


def holm(p):
    order, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


# per-dialogue accuracy.txt files: {"intended_score": x, "intended_rank": r} (rank 4 = not in top 3)
present, total = defaultdict(int), defaultdict(int)
for j in JUDGES:
    for f in glob.glob(f"{RAW}/{j}/*/*/*/*/*/accuracy.txt"):
        t = f.split("/")[-3]
        r = json.load(open(f))["intended_rank"]
        total[(j, t)] += 1
        present[(j, t)] += int(r is not None and r <= 3)

# cell scores (domain, case, llm, trait) per judge
cell = {}
for j in JUDGES:
    for d in DOMS:
        s = json.load(open(f"{SCORED}/{j}/{d}/avg_across_cases_by_llm.json"))[d]
        for l in LLMS:
            for t in TRAITS:
                for c in CASES:
                    cell[(j, d, c, l, t)] = s[l][t]["case_level_results"][c]["avg_intended_score_all"]
avg = lambda keyf: np.mean([np.mean([cell[(j,) + k] for j in JUDGES]) for k in keyf])

print(f"{'Trait':<18} {'TraitScore':>10} {'Gemma':>7} {'GPT5m':>7} | top-3 rate (two-judge)")
rates = {}
for t in TRAITS:
    keys = [(d, c, l, t) for d in DOMS for c in CASES for l in LLMS]
    per = [np.mean([cell[(j,) + k] for k in keys]) for j in JUDGES]
    k = sum(present[(j, t)] for j in JUDGES); n = sum(total[(j, t)] for j in JUDGES)
    rates[t] = (k, n)
    print(f"{t:<18} {np.mean(per):>10.3f} {per[0]:>7.3f} {per[1]:>7.3f} | {100 * k / n:5.1f}%")
allk = [(d, c, l, t) for d in DOMS for c in CASES for l in LLMS for t in TRAITS]
print(f"{'Average':<18} {avg(allk):>10.3f}")

print("\nTop-3 detection rate per judge, one-sided binomial test vs 30% chance, Holm-corrected:")
for j in JUDGES:
    p_raw = np.array([binomtest(present[(j, t)], total[(j, t)], 0.3, alternative="greater").pvalue for t in TRAITS])
    print(f"  {j}: " + ", ".join(f"{t} {100 * present[(j, t)] / total[(j, t)]:.1f}% (p={p:.2g})"
                                for t, p in zip(TRAITS, holm(p_raw))))

print("\nFriedman test across conditions (60 domain x LLM x trait cells):")
for name, js in [("Two-judge avg", JUDGES), ("Gemma", ["gemma"]), ("GPT-5-mini", ["gpt5_mini"])]:
    x = {c: np.array([np.mean([cell[(j, d, c, l, t)] for j in js]) for d in DOMS for l in LLMS for t in TRAITS]) for c in CASES}
    chi, p = friedmanchisquare(*x.values())
    post = holm(np.array([wilcoxon(x[a], x[b]).pvalue for a, b in [("neutral", "try"), ("neutral", "trait"), ("try", "trait")]]))
    print(f"  {name:<14} chi2={chi:.2f} p={p:.3f}  means N/T/O = "
          + "/".join(f"{x[c].mean():.2f}" for c in CASES) + f"  post-hoc Holm p (N-T, N-O, T-O) = {np.round(post, 3).tolist()}")
