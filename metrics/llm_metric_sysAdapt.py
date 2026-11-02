import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hydra
import json
import re
from omegaconf import DictConfig
from pathlib import Path
from dotmap import DotMap
from LLM_Interaction.llm_models_api import judge_response
from datetime import datetime
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from LLM_Interaction.prompts.consts import (
    p2_descriptions,
    p2_descriptions_reversed,
)

LLMS = ["gpt4", "qwen3", "gemeni"]
CASES = ["trait", "try", "neutral"]
#CASES = ["neutral"]
#LLMS = ["gemeni"]

TRAITS = [
    "introversive", "closed", "stable", "Extraversion", "Agreeableness",
    "Conscientiousness", "Neuroticism", "Openness", "disagreeable", "unconscientious"
]
#TRAITS=["introversive"] #,"Openness","disagreeable","unconscientious"]
# TRAITS2=["disagreeable","unconscientious"]
class LLM_Metric:
    
    def __init__(self, cfg):
        self.cfg = DotMap(cfg)
        self.project_root = cfg.project_root
        self.cfg.project_root = Path(cfg.project_root)
        self.output_path = cfg.output_path
        self.domain = cfg.domain
        self.date = cfg.date
        #gpt_setting
        self.model_name = cfg.get("judge_model") or cfg.get("model_name")
        self.api_source = cfg.get("api_source")
        self.temperature = cfg.get("gpt_temperature")
        self.gpt_max_tokens = cfg.gpt_max_tokens 
        #prompt
        self.instruction_system_path =cfg.sys_prompt_path
        self.dialog_path = cfg.dialog_path
        self.prompt_name = cfg.prompt_name

        # running per (case, llm, trait) [score_sum, n] totals for
        # system_trait_alignment_score, updated after every judged dialogue
        # so the summary stays current even if the run is interrupted
        self.score_sums = {}
        self.lock = threading.Lock()
        # LLM judge (see LLM_Interaction/llm_models_api.judge_response):
        #   GPT-5-mini: judge_provider=open_route, judge_model=openai/gpt-5-mini, reasoning_effort=low
        #   Gemma:      judge_provider=ollama,     judge_model=gemma4:31b-cloud
        self.judge_provider = cfg.get("judge_provider", "open_route")
        self.judge_model = cfg.get("judge_model")
        self.reasoning_effort = cfg.get("reasoning_effort", None)
        self.num_workers = int(cfg.get("num_workers", 1))
        self.skip_existing = bool(cfg.get("skip_existing", True))
     
        
    # def get_dated_folder(self, base_path, date_prefix):
    #     # List all folders/files in base_path
    #     for name in os.listdir(base_path):
    #         full_path = os.path.join(base_path, name)
    #         # Check if it's a directory and starts with the given date prefix
    #         if os.path.isdir(full_path) and name.startswith(date_prefix):
    #             return full_path
    #     return None

    def parse_json_str(self, s: str, label: str = "<string>") -> dict:
        s = s.strip()
        if not s:
            raise ValueError(f"Empty response: {label}")

        s = s.replace("```json", "").replace("```", "").strip()
        s = s.replace("\\'", "'")

        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass

        repaired = self.repair_common_json_issues(s)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            # e.g. invalid backslash escapes inside "evidence": recover the score alone
            m = re.search(r'"system_trait_alignment_score"\s*:\s*"?(\d+(?:\.\d+)?)', s)
            if m:
                return {"system_trait_alignment_score": float(m.group(1))}
            print(f"\nInvalid JSON in: {label}")
            print(f"Error: {e}")
            print("Content preview:")
            print(repaired[:1500])
            raise

    def load_and_repair_json(self, file_path: str) -> dict:
        with open(file_path, "r", encoding="utf-8") as f:
            s = f.read()
        return self.parse_json_str(s, label=file_path)

    def repair_common_json_issues(self, s: str) -> str:
        s = s.strip()

        first_brace = s.find("{")
        if first_brace != -1:
            s = s[first_brace:]

        last_obj = s.rfind("}")
        last_arr = s.rfind("]")
        last_end = max(last_obj, last_arr)
        if last_end != -1:
            s = s[: last_end + 1]

        open_curly = s.count("{")
        close_curly = s.count("}")
        if close_curly < open_curly:
            s += "}" * (open_curly - close_curly)

        open_square = s.count("[")
        close_square = s.count("]")
        if close_square < open_square:
            s += "]" * (open_square - close_square)

        s = s.replace(",}", "}").replace(",]", "]")
        return s

    def record_score(self, case, llm, trait, response_text, domain_dir):
        """Parses system_trait_alignment_score out of a just-received judge
        response, folds it into the running (case, llm, trait) average, and
        merges that one cell into the domain-level summary JSON on disk.

        Reads-then-writes the existing file (instead of rebuilding it purely
        from this process's in-memory totals) so cells written by an earlier
        run — e.g. a different TRAITS subset — are never clobbered just
        because this process doesn't know about them."""
        try:
            data = self.parse_json_str(response_text, label=f"{case}/{llm}/{trait}")
            score = float(data["system_trait_alignment_score"])
        except Exception as e:
            print(f"Could not parse judge score for {case}/{llm}/{trait}: {e}")
            return

        by_llm = self.score_sums.setdefault(case, {})
        by_trait = by_llm.setdefault(llm, {})
        sum_count = by_trait.setdefault(trait, [0.0, 0])
        sum_count[0] += score
        sum_count[1] += 1

        os.makedirs(domain_dir, exist_ok=True)
        summary_path = os.path.join(domain_dir, "system_trait_alignment_summary.json")

        on_disk = {}
        if os.path.isfile(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    on_disk = json.load(f).get(self.domain, {})
            except Exception as e:
                print(f"Could not read existing summary, will not merge: {summary_path}\n  {e}")

        on_disk.setdefault(case, {}).setdefault(llm, {})[trait] = {
            "n_dialogues": sum_count[1],
            "avg_system_trait_alignment_score": round(sum_count[0] / sum_count[1], 2),
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({self.domain: on_disk}, f, indent=2, ensure_ascii=False)

    def aggregate_scores(self):
        """Reads every response.txt written by run() under
        output_path/domain/case/llm/trait/*/response.txt, averages
        system_trait_alignment_score per (case, llm, trait), and writes
        output_path/domain/system_trait_alignment_summary.json.

        Safe to call on a partial run (e.g. only some trait folders
        generated so far) — missing folders/files are just skipped.

        Walks whatever case/llm/trait folders actually exist on disk rather
        than the module-level CASES/LLMS/TRAITS constants, so a full
        recompute always reflects everything ever judged for this domain —
        including traits/cases from an earlier run with a different subset
        configured — not just the currently-active subset.
        """
        domain_dir = os.path.join(self.output_path, self.domain)
        summary = {}

        for case in sorted(os.listdir(domain_dir)) if os.path.isdir(domain_dir) else []:
            case_dir = os.path.join(domain_dir, case)
            if not os.path.isdir(case_dir):
                continue
            summary[case] = {}

            for llm in sorted(os.listdir(case_dir)):
                llm_dir = os.path.join(case_dir, llm)
                if not os.path.isdir(llm_dir):
                    continue
                summary[case][llm] = {}

                for trait in sorted(os.listdir(llm_dir)):
                    trait_dir = os.path.join(llm_dir, trait)
                    if not os.path.isdir(trait_dir):
                        continue

                    score_sum = 0.0
                    n_scored = 0

                    for name in os.listdir(trait_dir):
                        response_path = os.path.join(trait_dir, name, "response.txt")
                        if not os.path.isfile(response_path):
                            continue

                        try:
                            data = self.load_and_repair_json(response_path)
                            score = float(data["system_trait_alignment_score"])
                        except Exception as e:
                            print(f"Skipping unreadable/invalid response: {response_path}\n  {e}")
                            continue

                        score_sum += score
                        n_scored += 1

                    if n_scored == 0:
                        continue

                    summary[case][llm][trait] = {
                        "n_dialogues": n_scored,
                        "avg_system_trait_alignment_score": round(score_sum / n_scored, 2),
                    }

        os.makedirs(domain_dir, exist_ok=True)
        summary_path = os.path.join(domain_dir, "system_trait_alignment_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({self.domain: summary}, f, indent=2, ensure_ascii=False)

        print(f"\nSaved summary to: {summary_path}")
        return summary

    def call_judge(self, prompt):
        return judge_response(prompt, self.judge_model, self.judge_provider,
                              max_tokens=self.gpt_max_tokens, reasoning_effort=self.reasoning_effort)

    def collect_jobs(self):
        jobs = []
        dialog_path = os.path.join(self.dialog_path, self.domain)
        for case in CASES:
            for llm in LLMS:
                tag = "gpt" if llm == "gpt4" else llm
                prompt_name = f'{self.domain.lower()}_{case}_{tag}'
                dialog_path_name = os.path.join(dialog_path, case, llm, prompt_name, "addExample")
                for trait in TRAITS:
                    dialog_path_trait = os.path.join(dialog_path_name, trait)
                    for name in sorted(os.listdir(dialog_path_trait)):
                        sys_log = os.path.join(dialog_path_trait, name, "sys_log.txt")
                        if not os.path.isfile(sys_log):
                            continue
                        output_dir = os.path.join(self.output_path, self.domain, case, llm, trait, name)
                        if self.skip_existing and os.path.isfile(os.path.join(output_dir, "response.txt")):
                            continue
                        jobs.append((case, llm, trait, sys_log, output_dir))
        return jobs

    def judge_one(self, job):
        case, llm, trait, sys_log, output_dir = job
        with open(sys_log, "r") as f:
            dialog = f.read()
        prompt = self.instruction_prompt_system.format(dialogue=dialog, INTENDED_TRAIT=trait)
        try:
            response = self.call_judge(prompt)
        except Exception as e:
            print(f"FAILED {output_dir}: {e}")
            return
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "response.txt"), "w", encoding="utf-8") as f:
            f.write(response)
        with self.lock:
            self.record_score(case, llm, trait, response, os.path.join(self.output_path, self.domain))

    def run(self):
        with open(self.instruction_system_path, "r") as f:
            self.instruction_prompt_system = f.read()
        jobs = self.collect_jobs()
        print(f"[{self.domain}] {len(jobs)} dialogues to judge")
        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                for i, _ in enumerate(ex.map(self.judge_one, jobs), 1):
                    if i % 100 == 0:
                        print(f"[{self.domain}] {i}/{len(jobs)}", flush=True)
        else:
            for job in jobs:
                self.judge_one(job)


@hydra.main(config_path="../config_files", config_name="llm_metric_sysAdapt")
def hydra_start(cfg: DictConfig) -> None:
    domains = cfg.domain if isinstance(cfg.domain, (list, tuple)) or hasattr(cfg.domain, "__iter__") and not isinstance(cfg.domain, str) else [cfg.domain]
    for domain in domains:
        LLM_Eval = LLM_Metric(cfg)
        LLM_Eval.domain = domain
        LLM_Eval.run()  # generates judge scores via real API calls
        LLM_Eval.aggregate_scores()  # full recompute from response.txt files

    
if __name__ == "__main__":
    hydra_start()
    # wandb.finish()
    print(os.getcwd())
