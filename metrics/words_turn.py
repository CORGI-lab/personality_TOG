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
TRAITS = [
    "introversive", "closed", "stable", "Extraversion", "Agreeableness",
    "Conscientiousness", "Neuroticism", "Openness", "disagreeable", "unconscientious"
]

SCRIPTED_CLOSING = "that's all i need"
SPEAKER_RE = re.compile(r'^(user|system)\s*:', re.IGNORECASE | re.MULTILINE)

# Per-domain API methods: "search" returns a set of candidate options,
# "commit" is the booking call that ends the search phase.
DOMAIN_METHODS = {
    "hotel": {"search": ["SearchHotel"], "commit": ["ReserveHotel"]},
    "restaurant": {"search": ["FindRestaurants"], "commit": ["ReserveRestaurant"]},
}


class LLM_Metric:

    def __init__(self, cfg):
        self.cfg = DotMap(cfg)
        self.project_root = cfg.project_root
        self.cfg.project_root = Path(cfg.project_root)
        self.output_path = cfg.output_path
        self.domains = list(cfg.domains)
        self.domain = None  # set per-iteration in run()
        #self.date = cfg.date
        #gpt_setting
        #self.model_name = cfg.model_name
        #self.api_source = cfg.api_source
        #self.temperature = cfg.gpt_temperature
        #self.gpt_max_tokens = cfg.gpt_max_tokens
        #prompt
        #self.instruction_system_path = cfg.sys_prompt_path
        self.dialog_path = cfg.dialog_path
        self.prompt_name = cfg.prompt_name

    def parse_turns(self, dialogue_text):
        """Split a sys_log.txt transcript into (speaker, text) turns.

        Consecutive lines from the same speaker (e.g. a scripted "User:"
        preamble immediately followed by "USER:" with the actual message)
        are merged into one turn, since only a single reply follows them.
        """
        matches = list(SPEAKER_RE.finditer(dialogue_text))
        turns = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(dialogue_text)
            speaker = "User" if m.group(1).lower() == "user" else "System"
            text = SPEAKER_RE.sub("", dialogue_text[start:end], count=1).strip()

            if turns and turns[-1][0] == speaker:
                prev_speaker, prev_text = turns[-1]
                turns[-1] = (prev_speaker, prev_text + "\n" + text)
            else:
                turns.append((speaker, text))
        return turns

    def get_user_turns(self, dialogue_turns, exclude_scripted_close=True, speaker="User"):
        """dialogue_turns: list of (speaker, text) tuples in order."""
        user_turns = [text for spk, text in dialogue_turns if spk == "User"]
        sys_turns = [text for spk, text in dialogue_turns if spk == "System"]
        if exclude_scripted_close and user_turns:
            if SCRIPTED_CLOSING in user_turns[-1].lower():
                user_turns = user_turns[:-1]
        return user_turns if speaker=="User" else sys_turns

    def tokenize(self, text):
        return re.findall(r"[A-Za-z']+", text.lower())

    def count_api_calls(self, dialogue_text):
        return len(re.findall(r"APICall\s*\(", dialogue_text))

    def extract_search_result_blocks(self, system_turn_text):
        """Finds all 'Search Results ... End Search Results' blocks in a
        system turn and returns their raw content strings. A block that is
        never closed with 'End Search Results' (the response got truncated
        right after the results) is also captured, up to the end of the turn."""
        pattern = r"Search Results\s*(.*?)\s*(?:End Search Results|\Z)"
        return re.findall(pattern, system_turn_text, flags=re.DOTALL)

    def is_nonempty_result(self, result_block_text):
        """Returns True if the results block contains actual result entries,
        False if it's empty (e.g., '[]', blank, or whitespace only). Checked
        generically (presence of a dict entry) so it works across domains
        whose result records don't share a common field."""
        stripped = result_block_text.strip()
        if not stripped:
            return False
        if stripped in ("[]", "{}", "None"):
            return False
        return "{" in stripped

    def acceptance_metrics(self, turns, domain):
        """Walks the turns in order and finds the first booking/rental
        ('commit') API call. Returns:
          - first_option_accepted: 1 if exactly one successful search
            happened before that commit (user went with the first set of
            options), 0 if multiple successful searches happened (user
            rejected earlier results), None if no commit call occurred or
            the domain has no commit concept (e.g. Travel).
          - turns_to_acceptance: number of user turns that occurred before
            the commit call, i.e. how quickly the user committed. None if
            no commit call occurred.
        """
        methods = DOMAIN_METHODS.get(domain.lower())
        if not methods or not methods["commit"]:
            return {"first_option_accepted": None, "turns_to_acceptance": None}

        search_re = re.compile(
            r"APICall\s*\(\s*method\s*=\s*['\"](" + "|".join(re.escape(m) for m in methods["search"]) + r")['\"]"
        )
        commit_re = re.compile(
            r"APICall\s*\(\s*method\s*=\s*['\"](" + "|".join(re.escape(m) for m in methods["commit"]) + r")['\"]"
        )

        user_turns_seen = 0
        successful_searches = 0

        for speaker, text in turns:
            if speaker == "User":
                user_turns_seen += 1
                continue

            if commit_re.search(text):
                first_option_accepted = None
                if successful_searches > 0:
                    first_option_accepted = 1 if successful_searches == 1 else 0
                return {
                    "first_option_accepted": first_option_accepted,
                    "turns_to_acceptance": user_turns_seen,
                }

            if search_re.search(text):
                for block in self.extract_search_result_blocks(text):
                    if self.is_nonempty_result(block):
                        successful_searches += 1

        return {"first_option_accepted": None, "turns_to_acceptance": None}

    def dialogue_stats(self, dialogue_text, domain):
        turns = self.parse_turns(dialogue_text)
        user_turns = self.get_user_turns(turns)
        system_turns = self.get_user_turns(turns, speaker = "System")
        word_counts = [len(self.tokenize(t)) for t in user_turns]
        word_counts_sys = [len(self.tokenize(t)) for t in system_turns]
        acceptance = self.acceptance_metrics(turns, domain)

        return {
            "n_user_turns": len(user_turns),
            "n_sys_turns": len(system_turns),
            "total_words_user": sum(word_counts),
            "total_words_sys": sum(word_counts_sys),
            "n_api_calls": self.count_api_calls(dialogue_text),
            "first_option_accepted": acceptance["first_option_accepted"],
            "turns_to_acceptance": acceptance["turns_to_acceptance"],
        }

    def run(self):
        overall_summary = {}

        for domain in self.domains:
            self.domain = domain
            overall_summary[domain] = self.run_domain(domain)

        return overall_summary

    def run_domain(self, domain):
        dialog_path = os.path.join(self.dialog_path, domain)
        summary = {}

        for case in CASES:
            dialog_path_case = os.path.join(dialog_path, case)
            summary[case] = {}

            for llm in LLMS:
                dialog_path_llm = os.path.join(dialog_path_case, llm)
                if llm != "gpt4":
                    self.prompt_name = f'{domain.lower()}_{case}_{llm}'
                else:
                    self.prompt_name = f'{domain.lower()}_{case}_gpt'
                dialog_path_name = os.path.join(dialog_path_llm, self.prompt_name, "addExample")

                summary[case][llm] = {}

                for trait in TRAITS:
                    print(f" ***********************trait*********************: {domain}/{case}/{llm}/{trait}")
                    dialog_path_trait = os.path.join(dialog_path_name, trait)

                    if not os.path.isdir(dialog_path_trait):
                        print(f"Missing trait folder, skipping: {dialog_path_trait}")
                        continue

                    n_dialogues = 0
                    total_user_turns = 0
                    total_sys_turns = 0
                    total_words = 0
                    total_words_sys = 0
                    total_api_calls = 0

                    n_committed = 0
                    turns_to_acceptance_sum = 0
                    n_first_option_known = 0
                    n_first_option_accepted = 0

                    for name in os.listdir(dialog_path_trait):
                        dialog_file_sys = os.path.join(dialog_path_trait, name, "sys_log.txt")

                        if not os.path.isfile(dialog_file_sys):
                            continue

                        with open(dialog_file_sys, "r", encoding="utf-8") as f:
                            dialogue_text = f.read()

                        stats = self.dialogue_stats(dialogue_text, domain)

                        n_dialogues += 1
                        total_user_turns += stats["n_user_turns"]
                        total_sys_turns += stats["n_sys_turns"]
                        total_words += stats["total_words_user"]
                        total_words_sys += stats["total_words_sys"]
                        total_api_calls += stats["n_api_calls"]

                        if stats["turns_to_acceptance"] is not None:
                            n_committed += 1
                            turns_to_acceptance_sum += stats["turns_to_acceptance"]

                        if stats["first_option_accepted"] is not None:
                            n_first_option_known += 1
                            if stats["first_option_accepted"] == 1:
                                n_first_option_accepted += 1

                    if n_dialogues == 0:
                        continue

                    summary[case][llm][trait] = {
                        "n_dialogues": n_dialogues,
                        "avg_turn_count": round(total_user_turns / n_dialogues, 2),
                        "avg_words_per_turn": round(total_words / total_user_turns, 2) if total_user_turns else 0.0,
                        "avg_words_per_turn_sys": round(total_words_sys / total_sys_turns, 2) if total_sys_turns else 0.0,
                        "avg_api_count": round(total_api_calls / n_dialogues, 2),
                        "total_user_turns": total_user_turns,
                        "total_sys_turns": total_sys_turns,
                        "total_api_calls": total_api_calls,

                        "n_committed_dialogues": n_committed,
                        "avg_turns_to_acceptance": round(turns_to_acceptance_sum / n_committed, 2) if n_committed else None,
                        "first_option_accepted_rate": round(n_first_option_accepted / n_first_option_known, 2) if n_first_option_known else None,
                    }

        output_dir = os.path.join(self.output_path, domain)
        os.makedirs(output_dir, exist_ok=True)

        summary_path = os.path.join(output_dir, "user_turns_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({domain: summary}, f, indent=2, ensure_ascii=False)

        print(f"\nSaved summary to: {summary_path}")
        return summary


@hydra.main(config_path="../config_files", config_name="llm_metric_trait")
def hydra_start(cfg: DictConfig) -> None:
    LLM_Eval = LLM_Metric(cfg)

    LLM_Eval.run()


if __name__ == "__main__":
    hydra_start()
    # wandb.finish()
    print(os.getcwd())
