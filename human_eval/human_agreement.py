"""Human evaluation (Section V-D): agreement of annotators with each other and with the two
pairwise-satisfaction judges (Gemma, GPT-5-mini).

Inputs
  human_eval/human_ratings.csv              anonymized ratings: study, pair_id, rater (H1/H2), choice_display (A/B/tie)
  storage/chatgpt_new/qualtrics_studies/annotator_1/study_XX.csv   pair metadata (display order A/B -> condition)
  judge outputs under storage/chatgpt_new/metric/llm_metric/hpc/llm_metric_SYS/New_1to5_scale/User_Satisfaction_Pairly

All labels are mapped to the displayed A/B of the annotator study, so kappa is computed on the
same A/B/tie categories the annotators saw. Run from the repository root:
  python human_eval/human_agreement.py
"""
import csv, json, os, re
from collections import defaultdict

Q = "storage/chatgpt_new/qualtrics_studies/annotator_1"
P = "storage/chatgpt_new/metric/llm_metric/hpc/llm_metric_SYS/New_1to5_scale/User_Satisfaction_Pairly"
COMP = {"TN": ("neutral_vs_try", "neutral", "try"), "ON": ("neutral_vs_trait", "neutral", "trait"),
        "TO": ("try_vs_trait", "try", "trait")}
NAME = {"Neutral": "neutral", "Try": "try", "Oracle": "trait", "Trait": "trait"}


def kappa(a, b, cats):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po


def gemma_winner(path):
    try:
        return json.loads(re.search(r"\{.*\}", open(path).read(), re.S).group(0))["winner"]
    except Exception:
        return None


ratings = defaultdict(dict)
for r in csv.DictReader(open("human_eval/human_ratings.csv")):
    ratings[(int(r["study"]), r["pair_id"])][r["rater"]] = r["choice_display"]

rows = []
for (study, pid), h in sorted(ratings.items()):
    with open(f"{Q}/study_{study:02d}.csv") as f:
        next(f)  # instruction line
        meta = {m["pair_id"]: m for m in csv.DictReader(f)}[pid]
    disp = {k.strip(): NAME[v.strip()] for k, v in (x.split("=") for x in meta["display_true_mapping"].split(","))}
    to_disp = lambda cond: "tie" if cond == "tie" else ("A" if disp["A"] == cond else "B")
    typ, rest = pid.split("_", 1)
    trait, llm, dom, names = re.match(r"(.+)_(gpt4|qwen3|gemeni)_(Hotel|Restaurant)_(.+)$", rest).groups()
    folder, ca, cb = COMP[typ]
    case = {"A": ca, "B": cb, "tie": "tie"}
    # 3 study pairs are not in the main runs (the cell was regenerated and re-paired after the
    # study was built); they were judged separately with the same setup (human_study_extra/).
    cell = f"{dom}/{folder}/{llm}/{trait}/pair_{names}"
    g5_path = next((x for x in (f"{P}/gpt5-mini/{cell}/verdict.json", f"{P}/human_study_extra/gpt5-mini/{cell}/verdict.json")
                    if os.path.exists(x)), None)
    g5 = json.load(open(g5_path))["winner"] if g5_path else None
    gm_path = next((x for x in (f"{P}/gemma/{cell}/response.txt", f"{P}/human_study_extra/gemma/{cell}/response.txt")
                    if os.path.exists(x)), None)
    gm = gemma_winner(gm_path) if gm_path else None
    rows.append(dict(h=[h["H1"], h["H2"]], gemma=gm and to_disp(case[gm]), gpt5=g5 and to_disp(case[g5]),
                     type=typ, trait=trait))

print(f"pairs: {len(rows)}; scored by both judges: {sum(bool(r['gemma'] and r['gpt5']) for r in rows)}")
rows = [r for r in rows if r["gemma"] and r["gpt5"]]
for r in rows:
    r["consensus"] = r["gemma"] if r["gemma"] == r["gpt5"] else "tie"


def report(name, H, J):
    k, po = kappa(H, J, ["A", "B", "tie"])
    nt = [(x, y) for x, y in zip(H, J) if "tie" not in (x, y)]
    k2, po2 = kappa([x for x, _ in nt], [y for _, y in nt], ["A", "B"])
    print(f"{name:<34} incl. ties {100 * po:5.1f}% kappa={k:.3f} | no-tie n={len(nt):>3} {100 * po2:5.1f}% kappa={k2:.3f}")
    return k, po, k2, po2


H1 = [r["h"][0] for r in rows]; H2 = [r["h"][1] for r in rows]
report("Annotator 1 vs annotator 2", H1, H2)
res = {}
for j in ("gemma", "gpt5", "consensus"):
    res[j] = report(f"Humans vs {j}" + (" (judges agree, else tie)" if j == "consensus" else ""),
                    H1 + H2, [r[j] for r in rows] * 2)
avg = [(res["gemma"][i] + res["gpt5"][i]) / 2 for i in range(4)]
print(f"{'Humans vs judges (average of two)':<34} incl. ties {100 * avg[1]:5.1f}% kappa={avg[0]:.3f} | "
      f"no-tie {100 * avg[3]:5.1f}% kappa={avg[2]:.3f}")
print(f"\njudge tie rate: Gemma {100 * sum(r['gemma'] == 'tie' for r in rows) / len(rows):.1f}%, "
      f"GPT-5-mini {100 * sum(r['gpt5'] == 'tie' for r in rows) / len(rows):.1f}%; "
      f"annotator tie rate {100 * (H1 + H2).count('tie') / (2 * len(rows)):.1f}%")
print(f"judges agree on {sum(r['gemma'] == r['gpt5'] for r in rows)}/{len(rows)} pairs")
for j in ("gemma", "gpt5"):
    against = [r for r in rows if r["h"][0] == r["h"][1] != "tie" and r[j] not in ("tie", r["h"][0])]
    print(f"pairs where both annotators agree against {j}: {len(against)} "
          f"({', '.join(sorted(r['type'] + ':' + r['trait'] for r in against))})")
