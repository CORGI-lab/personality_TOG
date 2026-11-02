import sys
import os
import random
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hydra
from omegaconf import DictConfig
from dotmap import DotMap

from LLM_Interaction.llm_models_api import judge_response

from LLM_Interaction.prompts.consts import (
    p2_descriptions,
    p2_descriptions_reversed,
)


LLMS = ["gpt4", "qwen3", "gemeni"]


TRAITS = [
    "introversive", "closed", "stable", "Extraversion", "Agreeableness",
    "Conscientiousness", "Neuroticism", "Openness", "disagreeable", "unconscientious"
]

# For testing only:
#TRAITS = ["introversive"]


class LLM_Metric:

    def __init__(self, cfg):
        self.cfg = DotMap(cfg)

        self.project_root = Path(cfg.project_root)
        self.output_path = Path(cfg.output_path)

        # `domain` may be a single name ("Restaurant") or a list
        # ([Restaurant, Hotel]); run_all_domains() loops over the list and
        # sets self.domain to the domain currently being processed.
        domains = cfg.domain
        self.domains = [domains] if isinstance(domains, str) else list(domains)
        self.domain = self.domains[0]
        self.date = cfg.date

        # GPT settings
        self.model_name = cfg.get("judge_model") or cfg.get("model_name")
        self.api_source = cfg.get("api_source")
        self.temperature = cfg.get("gpt_temperature")
        self.gpt_max_tokens = cfg.gpt_max_tokens

        # Prompt
        self.instruction_system_path = Path(cfg.sys_prompt_path)

        # Dialogue path
        self.dialog_path = Path(cfg.dialog_path)
        self.prompt_name = cfg.prompt_name

        # Pairwise cases
        # Add these in yaml:
        # case_a: neutral
        # case_b: trait
        self.case_pairs = cfg.get("case_pairs", [["neutral", "trait"], ["neutral", "try"]])

        # For reproducible random pairing without replacement
        self.seed = int(cfg.get("seed", 42))

        # Position-bias control for which dialogue is shown as "Dialogue A":
        #   fixed  - case_a always shown first (original behaviour)
        #   random - seeded per-pair coin flip decides the display order
        #   both   - judge each pair in both orders; keep the verdict only if
        #            the two orders agree, otherwise count it as a tie
        # Verdicts are always mapped back to case level: "A" = case_a wins.
        # LLM judge (see LLM_Interaction/llm_models_api.judge_response):
        #   GPT-5-mini: judge_provider=open_route, judge_model=openai/gpt-5-mini, reasoning_effort=low
        #   Gemma:      judge_provider=ollama,     judge_model=gemma4:31b-cloud
        self.judge_provider = cfg.get("judge_provider", "open_route")
        self.judge_model = cfg.get("judge_model")
        self.reasoning_effort = cfg.get("reasoning_effort")
        self.num_workers = int(cfg.get("num_workers", 1))
        # Reuse verdict.json of pairs already judged (resume an interrupted run)
        self.skip_existing = bool(cfg.get("skip_existing", True))

        self.order_mode = cfg.get("order_mode", "fixed")
        if self.order_mode not in ("fixed", "random", "both"):
            raise ValueError(f"order_mode must be fixed/random/both, got {self.order_mode!r}")

    def get_trait_definition(self, trait):
        if trait in p2_descriptions:
            return p2_descriptions[trait]

        if trait in p2_descriptions_reversed:
            return p2_descriptions_reversed[trait]

        trait_lower = trait.lower()

        if trait_lower in p2_descriptions:
            return p2_descriptions[trait_lower]

        if trait_lower in p2_descriptions_reversed:
            return p2_descriptions_reversed[trait_lower]

        return ""

    def get_prompt_folder_name(self, case, llm):
        """
        Your original structure:
        Restaurant/trait/gemeni/restaurant_trait_gemeni/addExample/Extraversion/...
        Restaurant/trait/gpt4/restaurant_trait_gpt/addExample/Extraversion/...
        """

        if llm != "gpt4":
            return f"{self.domain.lower()}_{case}_{llm}"
        else:
            return f"{self.domain.lower()}_{case}_gpt"

    def get_trait_dialogue_dir(self, case, llm, trait):
        prompt_folder = self.get_prompt_folder_name(case, llm)

        return (
            self.dialog_path
            / self.domain
            / case
            / llm
            / prompt_folder
            / "addExample"
            / trait
        )

    def collect_dialogue_files(self, case, llm, trait):
        """
        Returns list of tuples:
        [
          {
            "dialogue_id": name,
            "path": path_to_sys_log
          }
        ]
        """

        trait_dir = self.get_trait_dialogue_dir(case, llm, trait)

        if not trait_dir.exists():
            print(f"Missing directory: {trait_dir}")
            return []

        files = []

        for name in sorted(os.listdir(trait_dir)):
            dialogue_folder = trait_dir / name

            if not dialogue_folder.is_dir():
                continue

            sys_log_path = dialogue_folder / "sys_log.txt"

            if sys_log_path.exists():
                files.append({
                    "dialogue_id": name,
                    "path": sys_log_path,
                })

        return files
    def rebuild_summary_from_existing_pair_folders(self):
        """
        Rebuild summaries by reading already existing pair folders only.

        This does NOT:
        - call the LLM
        - recreate pairs
        - depend on original dialogue folders

        It only scans:
        output_path/domain/comparison_name/llm/trait/pair_*/response.txt
        """

        all_summary = {}

        domain_dir = self.output_path / self.domain

        for case_a, case_b in self.case_pairs:
            comparison_name = f"{case_a}_vs_{case_b}"
            comparison_dir = domain_dir / comparison_name

            all_summary[comparison_name] = {}

            print(f"\n================ Rebuilding from existing folders: {comparison_name} ================\n")

            if not comparison_dir.exists():
                print(f"Missing comparison directory: {comparison_dir}")
                continue

            for llm in LLMS:
                llm_dir = comparison_dir / llm
                all_summary[comparison_name][llm] = {}

                if not llm_dir.exists():
                    print(f"Missing LLM directory: {llm_dir}")
                    continue

                for trait in TRAITS:
                    output_trait_dir = llm_dir / trait

                    print(
                        f"*********************** comparison: {comparison_name} | "
                        f"trait: {trait} | llm: {llm} ***********************"
                    )

                    winner_counts = {
                        "A": 0,
                        "B": 0,
                        "tie": 0,
                        "invalid": 0,
                        "missing": 0,
                    }

                    pair_records = []

                    if not output_trait_dir.exists():
                        print(f"Missing trait directory: {output_trait_dir}")
                        all_summary[comparison_name][llm][trait] = {
                            "comparison": comparison_name,
                            "trait": trait,
                            "llm": llm,
                            "case_a": case_a,
                            "case_b": case_b,
                            "num_existing_pair_folders": 0,
                            "winner_counts": winner_counts,
                            "win_rate_A": 0.0,
                            "win_rate_B": 0.0,
                            "tie_rate": 0.0,
                        }
                        continue

                    pair_dirs = [
                        p for p in sorted(output_trait_dir.iterdir())
                        if p.is_dir() and p.name.startswith("pair_")
                    ]

                    for idx, pair_dir in enumerate(pair_dirs):
                        response_path = pair_dir / "response.txt"
                        prompt_path = pair_dir / "prompt.txt"

                        record = {
                            "pair_id": idx,
                            "pair_folder": pair_dir.name,
                            "case_a": case_a,
                            "case_b": case_b,
                            "response_file": str(response_path),
                            "prompt_file": str(prompt_path) if prompt_path.exists() else None,
                        }

                        # Runs with order control store the case-level verdict separately;
                        # their response.txt may refer to a flipped display order.
                        verdict_path = pair_dir / "verdict.json"
                        if verdict_path.exists():
                            with open(verdict_path, "r", encoding="utf-8") as f:
                                verdict = json.load(f)
                            winner = verdict.get("winner", "invalid")
                            if winner not in ("A", "B", "tie"):
                                winner = "invalid"
                            winner_counts[winner] += 1
                            record.update(verdict)
                            record["winner"] = winner
                            pair_records.append(record)
                            continue

                        if not response_path.exists():
                            winner_counts["missing"] += 1
                            record["winner"] = "missing"
                            record["error"] = "response.txt not found"
                            pair_records.append(record)
                            continue

                        with open(response_path, "r", encoding="utf-8") as f:
                            response = f.read().strip()

                        try:
                            parsed = self.parse_json_response(response)
                            winner = parsed.get("winner", "invalid")

                            if winner not in ["A", "B", "tie"]:
                                winner = "invalid"

                            winner_counts[winner] += 1
                            record.update(parsed)
                            record["winner"] = winner

                        except Exception as e:
                            winner_counts["invalid"] += 1
                            record["winner"] = "invalid"
                            record["error"] = str(e)
                            record["raw_response"] = response

                        pair_records.append(record)

                    valid_total = (
                        winner_counts["A"]
                        + winner_counts["B"]
                        + winner_counts["tie"]
                    )

                    if valid_total > 0:
                        win_rate_a = winner_counts["A"] / valid_total
                        win_rate_b = winner_counts["B"] / valid_total
                        tie_rate = winner_counts["tie"] / valid_total
                    else:
                        win_rate_a = 0.0
                        win_rate_b = 0.0
                        tie_rate = 0.0

                    trait_summary = {
                        "comparison": comparison_name,
                        "trait": trait,
                        "llm": llm,
                        "case_a": case_a,
                        "case_b": case_b,
                        "num_existing_pair_folders": len(pair_dirs),
                        "num_valid_responses": valid_total,
                        "num_invalid_responses": winner_counts["invalid"],
                        "num_missing_responses": winner_counts["missing"],
                        "winner_counts": winner_counts,
                        "win_rate_A": win_rate_a,
                        "win_rate_B": win_rate_b,
                        "tie_rate": tie_rate,
                    }

                    all_summary[comparison_name][llm][trait] = trait_summary

                    with open(output_trait_dir / "pair_results.json", "w", encoding="utf-8") as f:
                        json.dump(pair_records, f, indent=2, ensure_ascii=False)

                    with open(output_trait_dir / "summary.json", "w", encoding="utf-8") as f:
                        json.dump(trait_summary, f, indent=2, ensure_ascii=False)

                    print(
                        f"{comparison_name} | {llm} | {trait} | "
                        f"A win: {win_rate_a:.3f} | "
                        f"B win: {win_rate_b:.3f} | "
                        f"tie: {tie_rate:.3f} | "
                        f"invalid: {winner_counts['invalid']} | "
                        f"missing: {winner_counts['missing']} | "
                        f"pairs: {len(pair_dirs)}"
                    )

        summary_path = domain_dir / "pairwise_summary_all_comparisons.json"

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_summary, f, indent=2, ensure_ascii=False)

        print(f"\nSaved rebuilt summary to: {summary_path}")

        return all_summary
    def make_pairs_without_replacement(self, files_a, files_b):
        """
        Pair two dialogue lists randomly without replacement.
        Extra dialogues from the larger list are ignored.
        """

        rng = random.Random(self.seed)

        files_a = list(files_a)
        files_b = list(files_b)

        rng.shuffle(files_a)
        rng.shuffle(files_b)

        n = min(len(files_a), len(files_b))

        return list(zip(files_a[:n], files_b[:n]))

    def read_text(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def judge_pair(self, item_a, item_b, orders, trait, trait_definition, pair_output_dir):
        """Judge one pair in the given display order(s); returns
        [(swapped, display_winner, case_winner, parsed), ...]. With skip_existing,
        a pair already judged in this order_mode is read back from verdict.json."""
        verdict_path = pair_output_dir / "verdict.json"
        if self.skip_existing and verdict_path.exists():
            with open(verdict_path, "r", encoding="utf-8") as f:
                verdict = json.load(f)
            # Reuse any stored judgement in the display order(s) needed now -- e.g. a
            # pair judged earlier in both orders already has the order random mode picks.
            stored = {not j["case_a_shown_first"]: j for j in verdict.get("judgements", [])}
            if all(swapped in stored for swapped in orders):
                return [
                    (swapped, stored[swapped]["display_winner"], stored[swapped]["case_winner"],
                     {"reason": stored[swapped].get("reason", "")})
                    for swapped in orders
                ]

        dialogue_a = self.read_text(item_a["path"])
        dialogue_b = self.read_text(item_b["path"])
        pair_output_dir.mkdir(parents=True, exist_ok=True)

        judged = []
        for swapped in orders:
            first, second = (dialogue_b, dialogue_a) if swapped else (dialogue_a, dialogue_b)
            prompt, response, parsed, display_winner = self.judge_once(first, second, trait, trait_definition)

            suffix = "_swapped" if (swapped and self.order_mode == "both") else ""
            with open(pair_output_dir / f"prompt{suffix}.txt", "w", encoding="utf-8") as f:
                f.write(prompt)
            with open(pair_output_dir / f"response{suffix}.txt", "w", encoding="utf-8") as f:
                f.write(response)

            judged.append((swapped, display_winner, self.to_case_winner(display_winner, swapped), parsed))
        return judged

    def call_llm(self, prompt):
        return judge_response(prompt, self.judge_model, self.judge_provider,
                              max_tokens=self.gpt_max_tokens, reasoning_effort=self.reasoning_effort)

    def run_all_domains(self, step):
        """Call `step` (run or rebuild_summary_from_existing_pair_folders) once per domain."""
        results = {}
        for domain in self.domains:
            self.domain = domain
            print(f"\n################ Domain: {domain} ################\n")
            results[domain] = step()
        return results

    def judge_once(self, dialogue_first, dialogue_second, trait, trait_definition):
        """Judge one display order. Returns (prompt, response, parsed, display_winner)."""
        prompt = self.instruction_prompt_system.format(
            DIALOGUE_A=dialogue_first,
            DIALOGUE_B=dialogue_second,
            INTENDED_TRAIT=trait,
            TRAIT_DEFINITION=trait_definition,
        )
        response = self.call_llm(prompt)

        try:
            parsed = self.parse_json_response(response)
            winner = parsed.get("winner", "invalid")
            if winner not in ("A", "B", "tie"):
                winner = "invalid"
        except Exception as e:
            parsed = {"error": str(e), "raw_response": response}
            winner = "invalid"

        return prompt, response, parsed, winner

    @staticmethod
    def to_case_winner(display_winner, swapped):
        """Map a verdict on the displayed order back to case level ("A" = case_a)."""
        if swapped and display_winner in ("A", "B"):
            return "B" if display_winner == "A" else "A"
        return display_winner

    @staticmethod
    def combine_orders(case_winner_original, case_winner_swapped):
        """A verdict that flips when the order flips is position-driven, so it becomes a tie."""
        if "invalid" in (case_winner_original, case_winner_swapped):
            return "invalid"
        if case_winner_original == case_winner_swapped:
            return case_winner_original
        return "tie"

    def run(self):

        with open(self.instruction_system_path, "r", encoding="utf-8") as f:
            self.instruction_prompt_system = f.read()

        all_summary = {}

        for case_a, case_b in self.case_pairs:
            comparison_name = f"{case_a}_vs_{case_b}"
            all_summary[comparison_name] = {}

            print(f"\n================ Running comparison: {comparison_name} ================\n")

            for llm in LLMS:
                #if case_a != 'neutral' and case_b != 'trait' and llm != 'gpt4':
                    all_summary[comparison_name][llm] = {}


                    for trait in TRAITS:
                        print(f"*********************** comparison: {comparison_name} | trait: {trait} | llm: {llm} ***********************")

                        files_a = self.collect_dialogue_files(case_a, llm, trait)
                        files_b = self.collect_dialogue_files(case_b, llm, trait)

                        pairs = self.make_pairs_without_replacement(files_a, files_b)

                        print(
                            f"{case_a}: {len(files_a)} dialogues | "
                            f"{case_b}: {len(files_b)} dialogues | "
                            f"pairs: {len(pairs)}"
                        )

                        output_trait_dir = (
                            self.output_path
                            / self.domain
                            / comparison_name
                            / llm
                            / trait
                        )
                        output_trait_dir.mkdir(parents=True, exist_ok=True)

                        # Case-level counts: "A" = case_a wins, "B" = case_b wins
                        winner_counts = {
                            "A": 0,
                            "B": 0,
                            "tie": 0,
                            "invalid": 0,
                        }
                        # Diagnostics: how often the dialogue shown first wins,
                        # and (order_mode=both) how often the verdict flips with order
                        position_stats = {
                            "judgements": 0,
                            "first_shown_wins": 0,
                            "second_shown_wins": 0,
                            "order_inconsistent": 0,
                        }

                        pair_records = []

                        trait_definition = self.get_trait_definition(trait)

                        # Deterministic per-cell RNG so random flips are reproducible
                        flip_rng = random.Random(f"{self.seed}-{comparison_name}-{llm}-{trait}")

                        # Display orders are drawn sequentially so they stay reproducible,
                        # then the (slow) judge calls run in parallel.
                        tasks = []
                        for item_a, item_b in pairs:
                            if self.order_mode == "fixed":
                                orders = [False]
                            elif self.order_mode == "random":
                                orders = [flip_rng.random() < 0.5]
                            else:
                                orders = [False, True]
                            pair_output_dir = output_trait_dir / f"pair_{item_a['dialogue_id']}&{item_b['dialogue_id']}"
                            tasks.append((item_a, item_b, orders, trait, trait_definition, pair_output_dir))

                        if self.num_workers > 1:
                            with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                                all_judged = list(ex.map(lambda t: self.judge_pair(*t), tasks))
                        else:
                            all_judged = [self.judge_pair(*t) for t in tasks]

                        for idx, ((item_a, item_b), (_, _, _, _, _, pair_output_dir), judged) in enumerate(
                                zip(pairs, tasks, all_judged)):

                            for _, display_winner, _, _ in judged:
                                if display_winner in ("A", "B"):
                                    position_stats["judgements"] += 1
                                    key = "first_shown_wins" if display_winner == "A" else "second_shown_wins"
                                    position_stats[key] += 1

                            if self.order_mode == "both":
                                winner = self.combine_orders(judged[0][2], judged[1][2])
                                if "invalid" not in (judged[0][2], judged[1][2]) and judged[0][2] != judged[1][2]:
                                    position_stats["order_inconsistent"] += 1
                            else:
                                winner = judged[0][2]

                            winner_counts[winner] += 1

                            record = {
                                "pair_id": idx,
                                "case_a": case_a,
                                "case_b": case_b,
                                "dialogue_a_id": item_a["dialogue_id"],
                                "dialogue_b_id": item_b["dialogue_id"],
                                "dialogue_a_file": str(item_a["path"]),
                                "dialogue_b_file": str(item_b["path"]),
                                "response_file": str(pair_output_dir / "response.txt"),
                                "order_mode": self.order_mode,
                                "winner": winner,
                                "judgements": [
                                    {
                                        "case_a_shown_first": not swapped,
                                        "display_winner": display_winner,
                                        "case_winner": case_winner,
                                        "reason": parsed.get("reason", ""),
                                        **({"error": parsed["error"]} if "error" in parsed else {}),
                                    }
                                    for swapped, display_winner, case_winner, parsed in judged
                                ],
                            }

                            # Case-level verdict on disk, so summaries can be rebuilt
                            # without re-deriving the display order from response.txt
                            with open(pair_output_dir / "verdict.json", "w", encoding="utf-8") as f:
                                json.dump(
                                    {"winner": winner, "case_a": case_a, "case_b": case_b,
                                     "order_mode": self.order_mode, "judgements": record["judgements"]},
                                    f, indent=2, ensure_ascii=False,
                                )

                            pair_records.append(record)

                        valid_total = winner_counts["A"] + winner_counts["B"] + winner_counts["tie"]

                        if valid_total > 0:
                            win_rate_a = winner_counts["A"] / valid_total
                            win_rate_b = winner_counts["B"] / valid_total
                            tie_rate = winner_counts["tie"] / valid_total
                        else:
                            win_rate_a = 0.0
                            win_rate_b = 0.0
                            tie_rate = 0.0

                        trait_summary = {
                            "comparison": comparison_name,
                            "trait": trait,
                            "llm": llm,
                            "case_a": case_a,
                            "case_b": case_b,
                            "num_case_a_dialogues": len(files_a),
                            "num_case_b_dialogues": len(files_b),
                            "num_pairs": len(pairs),
                            "order_mode": self.order_mode,
                            "winner_counts": winner_counts,
                            "win_rate_A": win_rate_a,
                            "win_rate_B": win_rate_b,
                            "tie_rate": tie_rate,
                            "position_stats": position_stats,
                            "first_shown_win_rate": (
                                position_stats["first_shown_wins"] / position_stats["judgements"]
                                if position_stats["judgements"] else 0.0
                            ),
                        }

                        all_summary[comparison_name][llm][trait] = trait_summary

                        with open(output_trait_dir / "pair_results.json", "w", encoding="utf-8") as f:
                            json.dump(pair_records, f, indent=2, ensure_ascii=False)

                        with open(output_trait_dir / "summary.json", "w", encoding="utf-8") as f:
                            json.dump(trait_summary, f, indent=2, ensure_ascii=False)

                        print(
                            f"{comparison_name} | {trait} | "
                            f"{case_a} win: {win_rate_a:.3f} | "
                            f"{case_b} win: {win_rate_b:.3f} | "
                            f"tie: {tie_rate:.3f} | "
                            f"first-shown win: {trait_summary['first_shown_win_rate']:.3f} | "
                            f"order-inconsistent: {position_stats['order_inconsistent']}"
                        )

            output_summary_dir = self.output_path / self.domain
            output_summary_dir.mkdir(parents=True, exist_ok=True)

            with open(output_summary_dir / "pairwise_summary_all_comparisons.json", "w", encoding="utf-8") as f:
                json.dump(all_summary, f, indent=2, ensure_ascii=False)

            print(f"Saved all pairwise summaries to: {output_summary_dir / 'pairwise_summary_all_comparisons.json'}")
            
    def parse_json_response(self, response):
        s = response.strip()

        s = s.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass

        first = s.find("{")
        last = s.rfind("}")

        if first != -1 and last != -1:
            s = s[first:last + 1]

        return json.loads(s)


@hydra.main(
    config_path="../config_files",
    config_name="llm_metric",
)
def hydra_start(cfg: DictConfig) -> None:
    LLM_Eval = LLM_Metric(cfg)
    LLM_Eval.run_all_domains(LLM_Eval.run)
    #LLM_Eval.run_all_domains(LLM_Eval.rebuild_summary_from_existing_pair_folders)

if __name__ == "__main__":
    hydra_start()
    print(os.getcwd())