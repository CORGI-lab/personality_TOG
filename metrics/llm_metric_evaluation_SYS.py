import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hydra
import json
import re

from omegaconf import DictConfig
from pathlib import Path
from dotmap import DotMap
from datetime import datetime
from LLM_Interaction.prompts.consts import (
    p2_descriptions,
    p2_descriptions_reversed,
)
LLMS = ["gpt4", "qwen3", "gemeni"]
CASES = ["trait", "try", "neutral"]
#CASES = ["neutral"]
TRAITS = [
    "introversive", "closed", "stable", "Extraversion", "Agreeableness",
    "Conscientiousness", "Neuroticism", "Openness", "disagreeable", "unconscientious"
]
#TRAITS = ["introversive"]

# Criteria returned by llm_evaluator_prompt_SYS.yml. The Gemma run used an earlier prompt version
# that named the third criterion "Dialogue Hallucination Rate" (same 1-5 direction: 5 = no
# unsupported claims); both names are accepted and reported as Truthfulness.
METRIC_KEYS = ["Constraint Satisfaction Rate", "Inform Rate", "Dialogue Truthfulness Rate"]
TRUTH_ALIASES = ["Dialogue Truthfulness Rate", "Dialogue Hallucination Rate"]


def get_scores(data):
    truth = next((data[k] for k in TRUTH_ALIASES if k in data), None)
    if truth is None:
        raise KeyError("no Truthfulness / Hallucination criterion")
    return [float(data["Constraint Satisfaction Rate"]["score"]), float(data["Inform Rate"]["score"]),
            float(truth["score"])]


class LLM_Metric:
    
    def __init__(self, cfg):
        self.cfg = DotMap(cfg)
        self.project_root = cfg.project_root
        self.cfg.project_root = Path(cfg.project_root)
        self.output_path = cfg.output_path
        # `domain` may be one name ("Hotel") or a list ([Restaurant, Hotel]);
        # run_all_domains() loops over the list.
        domains = cfg.domain
        self.domains = [domains] if isinstance(domains, str) else list(domains)
        self.domain = self.domains[0]
        self.date = cfg.date
        #prompt
        self.instruction_system_path =cfg.sys_prompt_path
        self.dialog_path = cfg.dialog_path
        self.prompt_name = cfg.prompt_name
        self.accuracy = {}
        
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

        repaired = self.repair_common_json_issues(s)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # Last resort: a judge sometimes leaves the free-text "reason" unquoted,
        # which breaks JSON but leaves the criterion scores intact -- extract them.
        data = {}
        for key in METRIC_KEYS[:2] + TRUTH_ALIASES:
            m = re.search(rf'"{re.escape(key)}"\s*:\s*\{{\s*"score"\s*:\s*(\d+(?:\.\d+)?)', s)
            if m:
                data[key] = {"score": float(m.group(1)), "reason": ""}
        if data:
            return data
        raise ValueError(f"Unparseable judge response: {file_path}")

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
    def run_all_domains(self):
        for domain in self.domains:
            self.domain = domain
            print(f"################ Domain: {domain} ################")
            self.run()

    def run(self):
        # NOTE: dialog_path here is the JUDGE OUTPUT folder (the response.txt
        # files written by the judge step), not the folder of sys_log.txt dialogues.
        dialog_path = os.path.join(self.dialog_path, self.domain)

        for case in CASES:
                dialog_path_case = os.path.join(dialog_path, case)

                for llm in LLMS:
                        dialog_path_llm = os.path.join(dialog_path_case, llm)
                        self.accuracy = {}

                        for trait in TRAITS:
                            print(f"*********************** trait *********************: {trait}")

                            dialog_path_trait = os.path.join(dialog_path_llm, trait)
                            if not os.path.isdir(dialog_path_trait):
                                print(f"Missing judge outputs: {dialog_path_trait}")
                                continue

                            constraint_satisfaction= 0
                            Truthfulness = 0
                            Inform_Rate = 0
                            #user_satisfaction = 0
                            total = 0
                            skipped = 0
                            for name in sorted(os.listdir(dialog_path_trait)):
                                dialog_path_name = os.path.join(dialog_path_trait, name)
                                dialog_path_name = os.path.join(dialog_path_name, "response.txt")
                                if not os.path.isfile(dialog_path_name):
                                    continue

                                try:
                                    data = self.load_and_repair_json(dialog_path_name)
                                    scores = get_scores(data)
                                except Exception as e:
                                    print(f"Skipping invalid file: {dialog_path_name} ({e})")
                                    skipped += 1
                                    continue
                                constraint_satisfaction += scores[0]
                                Inform_Rate += scores[1]
                                Truthfulness += scores[2]

                                #user_satisfaction += data["User Satisfaction Rate"]["score"]
                                total += 1

                            if total == 0:
                                print(f"No valid responses in {dialog_path_trait}")
                                continue

                            constraint_satisfaction_rate =  (constraint_satisfaction / total)
                            Truthfulness_rate =  (Truthfulness / total)
                            Inform_Rating =  (Inform_Rate / total)
                            #user_satisfaction_rate =  (user_satisfaction / total)


                            self.accuracy[trait] = {
                                "Constraint Satisfaction Rate": constraint_satisfaction_rate,
                                "Truthfulness_rate": Truthfulness_rate,
                                "Inform Rate": Inform_Rating,
                                #"user_satisfaction_rate": user_satisfaction_rate,
                                "total_examples": total,
                                "invalid_responses": skipped,
                            }
                        output_dir = os.path.join(self.output_path, self.domain)
                        output_dir = os.path.join(output_dir, case)
                                                    
                        output_dir = os.path.join(output_dir, llm)
                                    #output_dir = os.path.join(output_dir, trait)
                            #output_dir = os.path.join(output_dir, name)
                    

                                    # Create directory if it doesn't exist
                        if not os.path.exists(output_dir):
                                        os.makedirs(output_dir, exist_ok=True)

                            
                        with open(os.path.join(output_dir, "accuracy2.txt"), "w", encoding="utf-8") as f:
                                        json.dump( self.accuracy,f, indent=2, ensure_ascii=False)

 
            
        

        
@hydra.main(config_path="../config_files", config_name="llm_evaluation")
def hydra_start(cfg: DictConfig) -> None:
    LLM_Eval = LLM_Metric(cfg)


    LLM_Eval.run_all_domains()
            

    
if __name__ == "__main__":
    hydra_start()
    # wandb.finish()
    print(os.getcwd())
