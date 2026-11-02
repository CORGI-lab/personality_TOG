"""Spearman: User Trait Score vs pairwise gains (O-N, T-N), trait level (n=10) and
cell level (n=60 = 2 domains x 3 LLMs x 10 traits), for Gemma-only and for the
Gemma + GPT-5-mini average (both for trait score and for pairwise satisfaction)."""
import glob, json, re
from collections import Counter
import numpy as np
from scipy.stats import spearmanr

H = "./storage/chatgpt_new/metric/llm_metric/hpc"
PAIR = f"{H}/llm_metric_SYS/New_1to5_scale/User_Satisfaction_Pairly"
TS = f"{H}/llm_evaluation/USER/New_1to5_scale/User_trait_score"
TRAITS = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness",
          "introversive", "disagreeable", "unconscientious", "stable", "closed"]
LLMS, DOMS = ["gpt4", "qwen3", "gemeni"], ["Hotel", "Restaurant"]
CELLS = [(d, l, t) for d in DOMS for l in LLMS for t in TRAITS]


def trait_score(judge):
    out = {}
    for d in DOMS:
        s = json.load(open(f"{TS}/{judge}/{d}/avg_across_cases_by_llm.json"))[d]
        for l in LLMS:
            for t in TRAITS:
                out[(d, l, t)] = s[l][t]["avg_intended_score_all_across_cases"]
    return out


def delta_gemma(comp, cell):
    d, l, t = cell
    c = Counter()
    for f in glob.glob(f"{PAIR}/gemma/{d}/{comp}/{l}/{t}/pair_*/response.txt"):
        try:
            c[json.loads(re.search(r"\{.*\}", open(f).read(), re.S).group(0))["winner"]] += 1
        except Exception:
            pass
    n = c["A"] + c["B"] + c["tie"]
    return 100 * (c["B"] - c["A"]) / n          # case_b (Oracle/Try) minus Neutral


def delta_gpt5(comp, cell):
    d, l, t = cell
    s = json.load(open(f"{PAIR}/gpt5-mini/{d}/{comp}/{l}/{t}/summary.json"))
    return 100 * (s["win_rate_B"] - s["win_rate_A"])


ts_g, ts_m = trait_score("gemma"), trait_score("gpt5_mini")
data = {}
for comp, name in [("neutral_vs_trait", "O-N"), ("neutral_vs_try", "T-N")]:
    data[("gemma", name)] = {c: delta_gemma(comp, c) for c in CELLS}
    data[("gpt5", name)] = {c: delta_gpt5(comp, c) for c in CELLS}
SETUPS = {
    "Gemma only (original setup)": (ts_g, {k: data[("gemma", k)] for k in ("O-N", "T-N")}),
    "Gemma + GPT-5-mini average": (
        {c: (ts_g[c] + ts_m[c]) / 2 for c in CELLS},
        {k: {c: (data[("gemma", k)][c] + data[("gpt5", k)][c]) / 2 for c in CELLS} for k in ("O-N", "T-N")},
    ),
}


def sp(x, y):
    r, p = spearmanr(x, y)
    return f"rho={r:+.3f}, p={p:.3f}" if p >= .001 else f"rho={r:+.3f}, p<.001"


for label, (ts, deltas) in SETUPS.items():
    print(f"\n=========== {label} ===========")
    tts = [np.mean([ts[(d, l, t)] for d in DOMS for l in LLMS]) for t in TRAITS]
    print("trait-level User Trait Score: " + ", ".join(f"{t} {v:.2f}" for t, v in zip(TRAITS, tts)))
    for k in ("O-N", "T-N"):
        dl = deltas[k]
        tdl = [np.mean([dl[(d, l, t)] for d in DOMS for l in LLMS]) for t in TRAITS]
        print(f"  Delta {k}:")
        print(f"    trait level (n=10):          {sp(tts, tdl)}")
        print(f"    cell level  (n=60):          {sp([ts[c] for c in CELLS], [dl[c] for c in CELLS])}")
        for l, nm in [("gpt4", "GPT-4o"), ("qwen3", "Qwen3"), ("gemeni", "Gemini")]:
            cs = [c for c in CELLS if c[1] == l]
            print(f"    {nm:<7}     (n=20):          {sp([ts[c] for c in cs], [dl[c] for c in cs])}")
        if k == "O-N":
            keep = [c for c in CELLS if not (c[2] == "disagreeable" and c[1] in ("qwen3", "gemeni"))]
            print(f"    excl. disagreeable x Qwen3/Gemini (n={len(keep)}): {sp([ts[c] for c in keep], [dl[c] for c in keep])}")
