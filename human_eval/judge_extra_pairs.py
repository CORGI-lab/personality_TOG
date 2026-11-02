"""Judge the 3 human-study pairs that have no verdict in the main pairwise runs, with the same
prompt and settings as the main runs. Output: User_Satisfaction_Pairly/human_study_extra/."""
import json, os, random, re, sys
sys.path.insert(0, ".")
from LLM_Interaction.llm_models_api import judge_response
from LLM_Interaction.prompts.consts import p2_descriptions, p2_descriptions_reversed

PROMPT = open("LLM_Interaction/prompts/llm_evaluator_prompt_SYS_satisfaction_pairly.yml").read()
OUT = "storage/chatgpt_new/metric/llm_metric/hpc/llm_metric_SYS/New_1to5_scale/User_Satisfaction_Pairly/human_study_extra"
PAIRS = [  # (comparison folder, case_a, case_b, trait, llm, domain, dialogue_a, dialogue_b)
    ("neutral_vs_try", "neutral", "try", "introversive", "gemeni", "Restaurant", "03-02_20-28", "03-03_16-00"),
    ("neutral_vs_trait", "neutral", "trait", "introversive", "gemeni", "Restaurant", "10-21_14-35", "03-02_13-56"),
    ("neutral_vs_trait", "neutral", "trait", "introversive", "gemeni", "Restaurant", "03-02_20-17", "10-21_14-14"),
]
definition = lambda t: p2_descriptions.get(t) or p2_descriptions_reversed.get(t) or ""
log = lambda d, c, l, t, n: open(f"storage/chatgpt_new/{d}/{c}/{l}/{d.lower()}_{c}_{l}/addExample/{t}/{n}/sys_log.txt").read().strip()

gemma = lambda p: judge_response(p, "gemma4:31b-cloud", "ollama")
gpt5 = lambda p: judge_response(p, "openai/gpt-5-mini", "open_route", max_tokens=4000, reasoning_effort="low")


def winner(text):
    try: return json.loads(re.search(r"\{.*\}", text, re.S).group(0))["winner"]
    except Exception:
        m = re.search(r'"winner"\s*:\s*"(A|B|tie)"', text); return m.group(1) if m else "invalid"

for comp, ca, cb, t, l, d, a, b in PAIRS:
    da, db = log(d, ca, l, t, a), log(d, cb, l, t, b)
    pid = f"pair_{a}&{b}"
    # Gemma: fixed order (case_a shown as Dialogue A), as in the main Gemma run
    out = f"{OUT}/gemma/{d}/{comp}/{l}/{t}/{pid}"; os.makedirs(out, exist_ok=True)
    p = PROMPT.format(DIALOGUE_A=da, DIALOGUE_B=db, INTENDED_TRAIT=t, TRAIT_DEFINITION=definition(t))
    resp = gemma(p)
    open(f"{out}/prompt.txt", "w").write(p); open(f"{out}/response.txt", "w").write(resp)
    print("gemma", comp, pid, "->", winner(resp))
    # GPT-5-mini: seeded random display order, as in the main GPT-5-mini run
    swapped = random.Random(f"42-{comp}-{l}-{t}-{pid}").random() < 0.5
    out = f"{OUT}/gpt5-mini/{d}/{comp}/{l}/{t}/{pid}"; os.makedirs(out, exist_ok=True)
    first, second = (db, da) if swapped else (da, db)
    p = PROMPT.format(DIALOGUE_A=first, DIALOGUE_B=second, INTENDED_TRAIT=t, TRAIT_DEFINITION=definition(t))
    resp = gpt5(p); w = winner(resp)
    cw = ("B" if w == "A" else "A") if swapped and w in ("A", "B") else w
    open(f"{out}/prompt.txt", "w").write(p); open(f"{out}/response.txt", "w").write(resp)
    json.dump({"winner": cw, "case_a": ca, "case_b": cb, "order_mode": "random",
               "judgements": [{"case_a_shown_first": not swapped, "display_winner": w, "case_winner": cw,
                               "reason": (json.loads(re.search(r"\{.*\}", resp, re.S).group(0)).get("reason", "") if w != "invalid" else "")}]},
              open(f"{out}/verdict.json", "w"), indent=2)
    print("gpt5-mini", comp, pid, "swapped" if swapped else "original", "->", w, "(case-level", cw + ")")
