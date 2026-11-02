import sys
import os
import json
import re
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hydra
from omegaconf import DictConfig
from dotmap import DotMap

from LLM_Interaction.llm_models_api import generate_response

from LLM_Interaction.prompts.consts import (
    p2_descriptions,
    p2_descriptions_reversed,
)


LLMS = ["gpt4", "qwen3", "gemeni"]

P2_TRAITS = [
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness",
]

TRAITS = [
    "introversive",
    "closed",
    "stable",
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness",
    "disagreeable",
    "unconscientious",
]

#TRAITS = ["closed"]

class LLM_Metric:

    def __init__(self, cfg):
        self.cfg = DotMap(cfg)

        self.project_root = Path(cfg.project_root)
        self.output_path = Path(cfg.output_path)

        # GPT settings
        self.model_name = cfg.model_name
        self.api_source = cfg.api_source
        self.temperature = cfg.gpt_temperature
        self.gpt_max_tokens = cfg.gpt_max_tokens
        # Keep existing responses (the released BFI outputs) instead of re-querying
        self.skip_existing = bool(cfg.get("skip_existing", True))

        # Prompt
        self.instruction_system_path = Path(cfg.sys_prompt_path)

        # BFI items JSON
        self.bfi_items_path = Path(cfg.bfi_items_path)
        self.bfi_data = self.load_bfi_items()

    # -------------------------------------------------------
    # Load BFI item metadata
    # -------------------------------------------------------
    def load_bfi_items(self):
        if not self.bfi_items_path.exists():
            raise FileNotFoundError(
                f"BFI items JSON file not found: {self.bfi_items_path}"
            )

        with open(self.bfi_items_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_trait_key(self, trait):
        """
        Converts trait names from TRAITS to JSON keys.
        Example:
        Extraversion -> extraversion
        Agreeableness -> agreeableness
        introversive -> introversive
        """
        return trait.lower()

    def get_trait_bfi_items_text(self, trait):
        trait_key = self.get_trait_key(trait)

        if trait_key not in self.bfi_data["traits"]:
            raise KeyError(
                f"Trait '{trait_key}' not found in bfi_trait_items.json. "
                f"Available traits: {list(self.bfi_data['traits'].keys())}"
            )

        items = self.bfi_data["traits"][trait_key]["items"]

        items_text = "\n".join(
            f'{item["number"]}. {item["text"]}' for item in items
        )

        return items_text

    def get_trait_definition(self, trait):
        """
        High-pole traits use p2_descriptions.
        Low-pole traits use p2_descriptions_reversed.
        """

        if trait in p2_descriptions:
            return p2_descriptions[trait]

        if trait in p2_descriptions_reversed:
            return p2_descriptions_reversed[trait]

        trait_key = trait.lower()

        if trait_key in p2_descriptions:
            return p2_descriptions[trait_key]

        if trait_key in p2_descriptions_reversed:
            return p2_descriptions_reversed[trait_key]

        raise KeyError(f"No definition found for trait: {trait}")

    # -------------------------------------------------------
    # Generate BFI responses from LLM
    # -------------------------------------------------------
    def run(self):
        with open(self.instruction_system_path, "r", encoding="utf-8") as f:
            instruction_prompt_system = f.read()

        for trait in TRAITS:
            print(f"*********************** trait: {trait} ***********************")

            trait_bfi_items = self.get_trait_bfi_items_text(trait)
            trait_definition = self.get_trait_definition(trait)

            sys_GPT_input = instruction_prompt_system.format(
                TARGET_TRAIT=trait,
                TRAIT_DEFINITION=trait_definition,
                TRAIT_SPECIFIC_BFI_ITEMS=trait_bfi_items,
            )

            output_dir = self.output_path / self.model_name / trait
            if self.skip_existing and (output_dir / "response.txt").exists():
                print(f"Exists, skipping: {output_dir / 'response.txt'}")
                continue

            # The probe is answered by the generation model itself (same settings as dialogue generation)
            response = generate_response(sys_GPT_input, self.model_name, self.api_source,
                                         max_tokens=self.gpt_max_tokens, temperature=self.temperature)
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_dir / "prompt.txt", "w", encoding="utf-8") as f:
                f.write(sys_GPT_input)

            with open(output_dir / "response.txt", "w", encoding="utf-8") as f:
                f.write(response)

            print(f"Saved response to: {output_dir / 'response.txt'}")

    # -------------------------------------------------------
    # JSON loading / repair
    # -------------------------------------------------------
    def load_and_repair_json(self, file_path) -> dict:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            s = f.read().strip()

        if not s:
            raise ValueError(f"Empty file: {file_path}")

        # Remove markdown fences
        s = re.sub(r"^```json\s*", "", s.strip())
        s = re.sub(r"^```\s*", "", s.strip())
        s = re.sub(r"\s*```$", "", s.strip())
        s = s.strip()

        # Try direct JSON parse
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON object from surrounding text
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if match:
            s = match.group(0)

        repaired = self.repair_common_json_issues(s)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            print(f"\nInvalid JSON in file: {file_path}")
            print(f"Error: {e}")
            print("Content preview:")
            print(repaired[:1500])
            raise

    def repair_common_json_issues(self, s: str) -> str:
        s = s.strip()

        # Keep only content starting from first {
        first_brace = s.find("{")
        if first_brace != -1:
            s = s[first_brace:]

        # Keep only content until last } or ]
        last_obj = s.rfind("}")
        last_arr = s.rfind("]")
        last_end = max(last_obj, last_arr)

        if last_end != -1:
            s = s[: last_end + 1]

        # Remove trailing commas
        s = re.sub(r",\s*}", "}", s)
        s = re.sub(r",\s*]", "]", s)

        # Balance braces if needed
        open_curly = s.count("{")
        close_curly = s.count("}")

        if close_curly < open_curly:
            s += "}" * (open_curly - close_curly)

        open_square = s.count("[")
        close_square = s.count("]")

        if close_square < open_square:
            s += "]" * (open_square - close_square)

        return s

    # -------------------------------------------------------
    # Score calculation
    # -------------------------------------------------------
    def calculate_single_trait_score(self, trait, data):
        trait_key = self.get_trait_key(trait)

        if trait_key not in self.bfi_data["traits"]:
            raise KeyError(
                f"Trait '{trait_key}' not found in bfi_trait_items.json."
            )

        trait_info = self.bfi_data["traits"][trait_key]
        items = trait_info["items"]

        if "responses" not in data:
            raise KeyError(f"'responses' key not found for trait {trait}")

        responses = data["responses"]

        scored_items = {}
        scored_values = []

        for item in items:
            item_num = str(item["number"])

            if item_num not in responses:
                raise KeyError(
                    f"Missing response for item {item_num} in trait {trait}"
                )

            raw_score = int(responses[item_num])

            if raw_score < 1 or raw_score > 5:
                raise ValueError(
                    f"Invalid score for item {item_num}: {raw_score}. "
                    "Score must be between 1 and 5."
                )

            if item["reverse"]:
                scored_score = 6 - raw_score
            else:
                scored_score = raw_score

            scored_values.append(scored_score)

            scored_items[item_num] = {
                "text": item["text"],
                "raw_score": raw_score,
                "reverse": item["reverse"],
                "scored_score": scored_score,
            }

        bfi_dimension_score = sum(scored_values) / len(scored_values)

        # For low-pole traits, higher intended_direction_score should mean better match.
        if trait_info["direction"] == "low":
            intended_direction_score = 6 - bfi_dimension_score
        else:
            intended_direction_score = bfi_dimension_score

        result = {
            "target_trait": trait,
            "bfi_dimension": trait_info["bfi_dimension"],
            "direction": trait_info["direction"],
            "num_items": len(items),
            "bfi_dimension_score": round(bfi_dimension_score, 4),
            "intended_direction_score": round(intended_direction_score, 4),
            "scored_items": scored_items,
        }

        return result

    def calculate_avg(self):
        all_results = {}

        for trait in TRAITS:
            print(f"*********************** trait: {trait} ***********************")

            output_dir = self.output_path / self.model_name / trait
            response_path = output_dir / "response.txt"

            try:
                data = self.load_and_repair_json(response_path)
                result = self.calculate_single_trait_score(trait, data)

                all_results[trait] = result

                print(
                    f"{trait} | "
                    f"BFI dimension: {result['bfi_dimension']} | "
                    f"direction: {result['direction']} | "
                    f"BFI score: {result['bfi_dimension_score']} | "
                    f"intended score: {result['intended_direction_score']}"
                )

                with open(output_dir / "bfi_score.json", "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

            except Exception as e:
                print(f"Error processing trait {trait}: {e}")

                all_results[trait] = {
                    "target_trait": trait,
                    "error": str(e),
                }

        summary_dir = self.output_path / self.model_name
        #summary_dir.mkdir(parents=True, exist_ok=True)

        summary_path = summary_dir / "bfi_scores_summary.json"

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"Saved summary to: {summary_path}")

        return all_results


@hydra.main(
    config_path="../config_files",
    config_name="llm_bfi",
)
def hydra_start(cfg: DictConfig) -> None:
    llm_eval = LLM_Metric(cfg)

    # Generate responses:
    llm_eval.run()

    # Calculate BFI scores:
    llm_eval.calculate_avg()


if __name__ == "__main__":
    hydra_start()
    print(os.getcwd())