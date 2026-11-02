"""User trait-expression evaluation: judge, score, and summarize in one script.

Stages (choose with `stages` in config_files/llm_evaluation.yaml):
  judge     - ask the LLM judge for the user's top-3 evident traits per dialogue
              -> <output_path>/<Domain>/<case>/<llm>/<trait>/<dialogue>/response.txt
  score     - turn judge responses into per-cell metrics
              -> per-dialogue accuracy.txt next to each response.txt
              -> <score_output_path>/<Domain>/<case>/<llm>/accuracy.txt
              -> <score_output_path>/<Domain>/avg_across_cases_by_llm.json
  summarize - per-trait User Trait Score table (Hotel, Restaurant, both domains)
              -> <score_output_path>/user_trait_score_table.json
              optionally side by side with another judge (`compare_score_path`)

User Trait Score of a dialogue = the judge's 1-5 score for the intended trait if
it is among the judge's top-3 evident traits, else 0 (avg_intended_score_all).
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


import hydra
from omegaconf import DictConfig
from dotmap import DotMap

from LLM_Interaction.llm_models_api import judge_response

LLMS = ["gpt4", "qwen3", "gemeni"]
CASES = ["trait", "try", "neutral"]
TRAITS = [
    "introversive", "closed", "stable", "Extraversion", "Agreeableness",
    "Conscientiousness", "Neuroticism", "Openness", "disagreeable", "unconscientious"
]
#TRAITS=["introversive"]

# Order used for the paper tables
TABLE_TRAITS = [
    "Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness",
    "introversive", "disagreeable", "unconscientious", "stable", "closed"
]

CELL_METRICS = [
    "top1_match_rate",
    "top2_match_rate",
    "avg_intended_score_all",
    "avg_intended_score_when_present",
    "avg_intended_rank",
    "avg_top1_predicted_score",
]

CANONICAL_TRAITS = {t.lower(): t for t in TRAITS}


class LLM_Metric:

    def __init__(self, cfg):
        self.cfg = DotMap(cfg)
        self.project_root = cfg.project_root
        self.cfg.project_root = Path(cfg.project_root)
        self.output_path = cfg.output_path
        # Where scored metrics go; defaults to the judge output folder
        self.score_output_path = cfg.get("score_output_path") or cfg.output_path
        # Optional: score folder of another judge to print next to this one
        self.compare_score_path = cfg.get("compare_score_path")
        # `domain` may be one name ("Hotel") or a list ([Restaurant, Hotel])
        domains = cfg.domain
        self.domains = [domains] if isinstance(domains, str) else list(domains)
        self.domain = self.domains[0]
        self.date = cfg.date
        #gpt_setting
        self.model_name = cfg.get("judge_model") or cfg.get("model_name")
        self.api_source = cfg.get("api_source")
        self.temperature = cfg.get("gpt_temperature")
        self.gpt_max_tokens = cfg.gpt_max_tokens
        #prompt
        self.instruction_system_path = cfg.sys_prompt_path
        self.dialog_path = cfg.dialog_path
        self.prompt_name = cfg.prompt_name
        # Resume an interrupted judge run instead of re-calling the API
        self.skip_existing = bool(cfg.get("skip_existing", True))
        # LLM judge (see LLM_Interaction/llm_models_api.judge_response):
        #   GPT-5-mini: judge_provider=open_route, judge_model=openai/gpt-5-mini, reasoning_effort=low
        #   Gemma:      judge_provider=ollama,     judge_model=gemma4:31b-cloud
        self.judge_provider = cfg.get("judge_provider", "open_route")
        self.judge_model = cfg.get("judge_model")
        self.reasoning_effort = cfg.get("reasoning_effort")  # e.g. "low" for reasoning models
        self.num_workers = int(cfg.get("num_workers", 1))
        self.stages = list(cfg.get("stages", ["judge", "score", "summarize"]))

    # ------------------------------------------------------------------ #
    # Driver
    # ------------------------------------------------------------------ #

    def run_all(self):
        for stage in self.stages:
            if stage not in ("judge", "score", "summarize"):
                raise ValueError(f"Unknown stage {stage!r}; use judge, score, summarize")

        for domain in self.domains:
            self.domain = domain
            print(f"################ Domain: {domain} ################")
            if "judge" in self.stages:
                self.run()
            if "score" in self.stages:
                self.score()
                self.average_accuracy_across_cases_per_llm()

        if "summarize" in self.stages:
            self.summarize()

    # ------------------------------------------------------------------ #
    # Stage 1: judge
    # ------------------------------------------------------------------ #

    def call_judge(self, prompt):
        return judge_response(prompt, self.judge_model, self.judge_provider,
                              max_tokens=self.gpt_max_tokens, reasoning_effort=self.reasoning_effort)

    def judge_one(self, job):
        trait, sys_log_path, output_dir = job
        with open(sys_log_path, "r") as f:
            dialog = f.read()
        prompt = self.instruction_prompt_system.format(dialogue=dialog, INTENDED_TRAIT=trait)
        response = self.call_judge(prompt)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "response.txt"), "w", encoding="utf-8") as f:
            f.write(response)

    def run(self):

        with open(self.instruction_system_path, "r") as f:
            self.instruction_prompt_system = f.read()

        jobs = self.collect_judge_jobs()
        print(f"{self.domain}: {len(jobs)} dialogues to judge")
        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                for i, _ in enumerate(ex.map(self.judge_one, jobs), 1):
                    if i % 200 == 0:
                        print(f"  {self.domain}: {i}/{len(jobs)} judged", flush=True)
        else:
            for job in jobs:
                self.judge_one(job)

    def collect_judge_jobs(self):
        jobs = []
        dialog_path = os.path.join(self.dialog_path, self.domain)
        for case in CASES:
            dialog_path_case = os.path.join(dialog_path, case)
            for llm in LLMS:
                dialog_path_llm = os.path.join(dialog_path_case, llm)
                if llm != "gpt4":
                    self.prompt_name = f'{self.domain.lower()}_{case}_{llm}'
                else:
                    self.prompt_name = f'{self.domain.lower()}_{case}_gpt'
                dialog_path_name = os.path.join(dialog_path_llm, self.prompt_name, "addExample")

                for trait in TRAITS:
                    dialog_path_trait = os.path.join(dialog_path_name, trait)
                    for name in sorted(os.listdir(dialog_path_trait)):
                        sys_log_path = os.path.join(dialog_path_trait, name, "sys_log.txt")
                        if not os.path.isfile(sys_log_path):
                            continue
                        output_dir = os.path.join(self.output_path, self.domain, case, llm, trait, name)
                        if self.skip_existing and os.path.isfile(os.path.join(output_dir, "response.txt")):
                            continue
                        jobs.append((trait, sys_log_path, output_dir))
        return jobs

    # ------------------------------------------------------------------ #
    # Stage 2: score
    # ------------------------------------------------------------------ #

    def load_and_repair_json(self, file_path: str) -> dict:
        with open(file_path, "r", encoding="utf-8") as f:
            s = f.read().strip()

        if not s:
            raise ValueError(f"Empty file: {file_path}")

        s = s.replace("```json", "").replace("```", "").strip()
        s = s.replace("\\'", "'")

        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass

        try:
            return json.loads(self.repair_common_json_issues(s))
        except json.JSONDecodeError:
            pass

        # Last resort: some judges leave the free-text "reason" unquoted, which
        # breaks JSON but leaves trait names and scores intact -- extract those.
        traits = dict(re.findall(r'"top(\d)_evident_trait"\s*:\s*"([^"]+)"', s))
        scores = re.findall(r'"score"\s*:\s*(\d+(?:\.\d+)?)', s)
        if all(str(k) in traits for k in (1, 2, 3)) and len(scores) >= 3:
            return {
                **{f"top{k}_evident_trait": traits[str(k)] for k in (1, 2, 3)},
                **{f"top{k}_evident_trait_expression": {"score": float(scores[k - 1]), "reason": ""}
                   for k in (1, 2, 3)},
            }
        raise ValueError(f"Unparseable judge response: {file_path}")

    def repair_common_json_issues(self, s: str) -> str:
        s = s.strip()

        first_brace = s.find("{")
        if first_brace != -1:
            s = s[first_brace:]

        last_end = max(s.rfind("}"), s.rfind("]"))
        if last_end != -1:
            s = s[: last_end + 1]

        if s.count("}") < s.count("{"):
            s += "}" * (s.count("{") - s.count("}"))
        if s.count("]") < s.count("["):
            s += "]" * (s.count("[") - s.count("]"))

        return s.replace(",}", "}").replace(",]", "]")

    def canonical_trait(self, trait_name):
        """Fix only formatting (case, spaces); does NOT merge aliases or opposites."""
        if trait_name is None:
            return None
        trait_name = str(trait_name).strip()
        return CANONICAL_TRAITS.get(trait_name.lower(), trait_name)

    @staticmethod
    def _score_value(data, rank):
        try:
            return float(data.get(f"top{rank}_evident_trait_expression", {}).get("score"))
        except (TypeError, ValueError):
            return None

    def score(self):
        """Score every judge response of the current domain; write accuracy.txt files."""
        for case in CASES:
            for llm in LLMS:
                accuracy = {}
                for trait in TRAITS:
                    trait_dir = os.path.join(self.output_path, self.domain, case, llm, trait)
                    if not os.path.isdir(trait_dir):
                        print(f"Missing judge outputs: {trait_dir}")
                        continue

                    target_trait = self.canonical_trait(trait)
                    total = invalid = 0
                    top1_match_count = top2_match_count = 0
                    intended_score_sum_all = intended_score_sum_when_present = 0.0
                    intended_present_count = 0
                    intended_rank_sum = top1_score_sum = 0.0
                    predicted_trait_counts = defaultdict(int)
                    confusion_counts = defaultdict(int)
                    confusion_score_sums = defaultdict(float)

                    for name in sorted(os.listdir(trait_dir)):
                        response_path = os.path.join(trait_dir, name, "response.txt")
                        if not os.path.isfile(response_path):
                            continue

                        try:
                            data = self.load_and_repair_json(response_path)
                        except Exception as e:
                            print(f"Skipping invalid file: {response_path} ({e})")
                            invalid += 1
                            continue

                        top_traits = [
                            (rank, self.canonical_trait(data.get(f"top{rank}_evident_trait")), self._score_value(data, rank))
                            for rank in (1, 2, 3)
                        ]
                        top_traits = [(r, t, s) for r, t, s in top_traits if t and s is not None]
                        if not top_traits or top_traits[0][0] != 1:
                            invalid += 1
                            continue

                        total += 1
                        top1_pred_trait, top1_score = top_traits[0][1], top_traits[0][2]
                        top_trait_names = [t for _, t, _ in top_traits]

                        predicted_trait_counts[top1_pred_trait] += 1
                        top1_score_sum += top1_score

                        if target_trait == top1_pred_trait:
                            top1_match_count += 1
                        else:
                            confusion_counts[top1_pred_trait] += 1
                            confusion_score_sums[top1_pred_trait] += top1_score

                        if target_trait in top_trait_names[:2]:
                            top2_match_count += 1

                        intended_rank, intended_score = 4, 0.0
                        for rank, pred_trait, s in top_traits:
                            if target_trait == pred_trait:
                                intended_rank, intended_score = rank, s
                                break

                        intended_rank_sum += intended_rank
                        intended_score_sum_all += intended_score
                        if intended_rank <= 3:
                            intended_present_count += 1
                            intended_score_sum_when_present += intended_score

                        with open(os.path.join(trait_dir, name, "accuracy.txt"), "w", encoding="utf-8") as f:
                            json.dump({"intended_score": intended_score, "intended_rank": intended_rank},
                                      f, indent=2, ensure_ascii=False)

                    if total == 0:
                        print(f"No valid responses in {trait_dir}")
                        continue

                    accuracy[trait] = {
                        "target_trait_original": trait,
                        "target_trait": target_trait,
                        "total_examples": total,
                        "invalid_responses": invalid,
                        "top1_match_rate": top1_match_count / total,
                        "top2_match_rate": top2_match_count / total,
                        "avg_intended_score_all": intended_score_sum_all / total,
                        "avg_intended_score_when_present": (
                            intended_score_sum_when_present / intended_present_count
                            if intended_present_count else 0.0
                        ),
                        "avg_intended_rank": intended_rank_sum / total,
                        "avg_top1_predicted_score": top1_score_sum / total,
                        "predicted_trait_distribution_rate": {
                            k: v / total for k, v in sorted(predicted_trait_counts.items())
                        },
                        "confusion_distribution_rate": {
                            k: v / total for k, v in sorted(confusion_counts.items())
                        },
                        "confusion_avg_score": {
                            k: confusion_score_sums[k] / confusion_counts[k]
                            for k in sorted(confusion_counts) if confusion_counts[k] > 0
                        },
                    }

                output_dir = os.path.join(self.score_output_path, self.domain, case, llm)
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, "accuracy.txt"), "w", encoding="utf-8") as f:
                    json.dump(accuracy, f, indent=2, ensure_ascii=False)

    def average_accuracy_across_cases_per_llm(self):
        """Average each trait metric across cases, per LLM, for the current domain."""
        summary = {self.domain: {}}

        for llm in LLMS:
            summary[self.domain][llm] = {}
            for trait in TRAITS:
                trait_case_results = {}
                missing_cases = []
                for case in CASES:
                    accuracy_path = os.path.join(self.score_output_path, self.domain, case, llm, "accuracy.txt")
                    try:
                        with open(accuracy_path, "r", encoding="utf-8") as f:
                            trait_data = json.load(f)[trait]
                    except (OSError, KeyError, json.JSONDecodeError):
                        missing_cases.append(case)
                        continue
                    trait_case_results[case] = trait_data

                averaged = {}
                for key in CELL_METRICS:
                    vals = [float(d[key]) for d in trait_case_results.values() if d.get(key) is not None]
                    averaged[key] = sum(vals) / len(vals) if vals else 0.0

                summary[self.domain][llm][trait] = {
                    "target_trait_original": trait,
                    "target_trait": self.canonical_trait(trait),
                    "averaged_over_cases": list(trait_case_results.keys()),
                    "num_valid_cases": len(trait_case_results),
                    "missing_cases": missing_cases,
                    "total_examples_sum": sum(int(d.get("total_examples", 0)) for d in trait_case_results.values()),
                    **{f"avg_{k.removeprefix('avg_')}_across_cases": averaged[k] for k in CELL_METRICS},
                    "case_level_results": trait_case_results,
                }

        output_domain_dir = os.path.join(self.score_output_path, self.domain)
        os.makedirs(output_domain_dir, exist_ok=True)
        summary_path = os.path.join(output_domain_dir, "avg_across_cases_by_llm.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved averaged summary to: {summary_path}")
        return summary

    # ------------------------------------------------------------------ #
    # Stage 3: summarize
    # ------------------------------------------------------------------ #

    def trait_table(self, score_root):
        """Per trait: mean over cases x LLMs of the cell metrics, per domain and pooled."""
        cells = {}  # (domain, trait) -> list of cell dicts
        for domain in self.domains:
            for case in CASES:
                for llm in LLMS:
                    path = os.path.join(score_root, domain, case, llm, "accuracy.txt")
                    if not os.path.isfile(path):
                        continue
                    with open(path, "r", encoding="utf-8") as f:
                        acc = json.load(f)
                    for trait, d in acc.items():
                        cells.setdefault((domain, trait), []).append(d)

        def mean(dicts, key):
            vals = [float(d[key]) for d in dicts if d.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        table = {}
        for trait in TABLE_TRAITS:
            row = {}
            for domain in self.domains:
                row[domain] = mean(cells.get((domain, trait), []), "avg_intended_score_all")
            pooled = [d for domain in self.domains for d in cells.get((domain, trait), [])]
            row["both_domains"] = mean(pooled, "avg_intended_score_all")
            row["both_when_present"] = mean(pooled, "avg_intended_score_when_present")
            row["both_top1_match_rate"] = mean(pooled, "top1_match_rate")
            row["num_cells"] = len(pooled)
            table[trait] = row

        cols = list(self.domains) + ["both_domains", "both_when_present", "both_top1_match_rate"]
        table["Average"] = {
            c: (sum(table[t][c] for t in TABLE_TRAITS if table[t][c] is not None) / len(TABLE_TRAITS)
                if all(table[t][c] is not None for t in TABLE_TRAITS) else None)
            for c in cols
        }
        return table

    def summarize(self):
        table = self.trait_table(self.score_output_path)
        compare = self.trait_table(self.compare_score_path) if self.compare_score_path else None

        os.makedirs(self.score_output_path, exist_ok=True)
        out = {"judge": self.model_name, "score_root": self.score_output_path, "table": table}
        if compare:
            out.update({"compare_score_root": self.compare_score_path, "compare_table": compare})
        out_path = os.path.join(self.score_output_path, "user_trait_score_table.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        fmt = lambda v, pct=False: "   -  " if v is None else (f"{v * 100:5.1f}%" if pct else f"{v:6.3f}")
        header = f"{'Trait':<18}" + "".join(f"{d[:10]:>11}" for d in self.domains) + f"{'Both':>9}{'Present':>9}{'Top1':>8}"
        if compare:
            header += f"   | compare: {'Both':>6}{'Present':>9}{'Top1':>8}"
        print(f"\nUser Trait Score ({self.model_name}) -- saved to {out_path}")
        print(header)
        for trait in TABLE_TRAITS + ["Average"]:
            r = table[trait]
            line = f"{trait:<18}" + "".join(f"{fmt(r[d]):>11}" for d in self.domains)
            line += f"{fmt(r['both_domains']):>9}{fmt(r['both_when_present']):>9}{fmt(r['both_top1_match_rate'], True):>8}"
            if compare:
                c = compare[trait]
                line += f"   |          {fmt(c['both_domains']):>6}{fmt(c['both_when_present']):>9}{fmt(c['both_top1_match_rate'], True):>8}"
            print(line)
        return out


@hydra.main(config_path="../config_files", config_name="llm_evaluation")
def hydra_start(cfg: DictConfig) -> None:
    LLM_Eval = LLM_Metric(cfg)
    LLM_Eval.run_all()


if __name__ == "__main__":
    hydra_start()
