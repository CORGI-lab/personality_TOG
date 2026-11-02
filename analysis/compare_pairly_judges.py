"""Per-trait pairwise satisfaction (O/N, T/N, T/O) for the two judges and their average (Table II), averaged over
3 system LLMs x 2 domains. A cell 'x/y' = first condition win % / second condition win %."""
import glob, json, os, re
from collections import Counter

P = "./storage/chatgpt_new/metric/llm_metric/hpc/llm_metric_SYS/New_1to5_scale/User_Satisfaction_Pairly"
TRAITS = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness",
          "introversive", "disagreeable", "unconscientious", "stable", "closed"]
LLMS, DOMS = ["gpt4", "qwen3", "gemeni"], ["Hotel", "Restaurant"]
# column -> (comparison folder, which case-level side is the "first" condition)
COLS = [("O/N", "neutral_vs_trait", "B"), ("T/N", "neutral_vs_try", "B"), ("T/O", "try_vs_trait", "A")]


def cell_from_responses(d):
    """Gemma: fixed order, neutral/try always Dialogue A -> response winner is case-level."""
    c = Counter()
    for f in glob.glob(f"{d}/pair_*/response.txt"):
        try:
            c[json.loads(re.search(r"\{.*\}", open(f).read(), re.S).group(0))["winner"]] += 1
        except Exception:
            pass
    n = c["A"] + c["B"] + c["tie"]
    return (c["A"] / n, c["B"] / n, c["tie"] / n) if n else None


def cell_from_summary(d):
    try:
        s = json.load(open(f"{d}/summary.json"))
    except OSError:
        return None
    return s["win_rate_A"], s["win_rate_B"], s["tie_rate"]


def cell_first_order_only(d):
    """gpt-5-mini 'both' run: use only the original-order judgement (neutral/try shown
    first), i.e. the same setup Gemma was run with."""
    c = Counter()
    for f in glob.glob(f"{d}/pair_*/verdict.json"):
        v = json.load(open(f))
        j = [x for x in v["judgements"] if x["case_a_shown_first"]]
        if j:
            c[j[0]["case_winner"]] += 1
    n = c["A"] + c["B"] + c["tie"]
    return (c["A"] / n, c["B"] / n, c["tie"] / n) if n else None


JUDGES = [
    ("Gemma (fixed order)", "gemma", cell_from_responses),
    ("GPT-5-mini (random order)", "gpt5-mini", cell_from_summary),
]


def table(root, fn):
    out = {}
    for t in TRAITS:
        row = []
        for _, comp, first in COLS:
            vals = [fn(f"{P}/{root}/{d}/{comp}/{l}/{t}") for d in DOMS for l in LLMS]
            vals = [v for v in vals if v]
            if not vals:
                row.append(None)
                continue
            a = sum(v[0] for v in vals) / len(vals)
            b = sum(v[1] for v in vals) / len(vals)
            row.append((b, a, len(vals)) if first == "B" else (a, b, len(vals)))
        out[t] = row
    out["Average"] = [
        (sum(out[t][i][0] for t in TRAITS) / 10, sum(out[t][i][1] for t in TRAITS) / 10, None)
        if all(out[t][i] for t in TRAITS) else None
        for i in range(3)
    ]
    return out


def position_stats(root):
    j = fs = inc = n = 0
    for f in glob.glob(f"{P}/{root}/*/*/*/*/summary.json"):
        s = json.load(open(f)).get("position_stats")
        if s:
            j += s["judgements"]; fs += s["first_shown_wins"]; inc += s["order_inconsistent"]
            n += sum(json.load(open(f))["winner_counts"][k] for k in ("A", "B", "tie", "invalid"))
    return j, fs, inc, n


tables = {name: table(root, fn) for name, root, fn in JUDGES}
fmt = lambda c: "   -   " if c is None else f"{100 * c[0]:.0f}/{100 * c[1]:.0f}"
print(f"{'Trait':<18}" + "".join(f" | {name[:24]:^24}" for name, _, _ in JUDGES))
print(f"{'':<18}" + " | O/N    T/N    T/O    " * len(JUDGES))
for t in TRAITS + ["Average"]:
    print(f"{t:<18}" + "".join(" | " + " ".join(f"{fmt(tables[n][t][i]):<6}" for i in range(3)) + " " for n, _, _ in JUDGES))

print()
for name, root in [("GPT-5-mini", "gpt5-mini")]:
    j, fs, inc, n = position_stats(root)
    if j:
        print(f"{name}: first-shown dialogue wins {fs}/{j} = {100 * fs / j:.1f}% of judgements"
              + (f"; order-inconsistent pairs {inc}/{n} = {100 * inc / n:.1f}%" if inc else ""))

# Two-judge average (the US column of Table II)
g, m = tables[JUDGES[0][0]], tables[JUDGES[1][0]]
print("\nAverage of Gemma and GPT-5-mini (Table II, US column):")
for t in TRAITS + ["Average"]:
    print(f"{t:<18} " + ", ".join(f"{100 * (g[t][i][0] + m[t][i][0]) / 2:.0f}/{100 * (g[t][i][1] + m[t][i][1]) / 2:.0f}" for i in range(3)))
