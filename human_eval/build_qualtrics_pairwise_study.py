#!/usr/bin/env python3
"""Build Qualtrics pairwise user-satisfaction annotation datasets from
dialogue sys_log.txt files and LLM-judge pairwise results.

Sampling and A/B flip decisions are generated once (seed=SEED) into a
fixed pool of N_PAIRS records. Per-annotator study files are independent
shuffles/splits of that same fixed pool (annotator_1 uses SEED, annotator_2
uses SEED+1), so both annotators see the same 300 pairs, same dialogue
content, and same A/B flips -- only presentation order differs.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42
ANNOTATOR_SEEDS = {"annotator_1": 42, "annotator_2": 43}
N_PAIRS = 300
N_STUDIES = 12
PAIRS_PER_CELL_GROUP = 10  # per (trait, comparison_type)

BASE_DIR = Path("./storage/chatgpt_new")
JUDGE_BASE = (
    BASE_DIR
    / "metric/llm_metric/hpc/llm_metric_SYS/New_1to5_scale/User_Satisfaction_Pairly"
)
OUTPUT_DIR = BASE_DIR / "qualtrics_studies"

DOMAINS = ["Hotel", "Restaurant"]
DOMAIN_LOWER = {"Hotel": "hotel", "Restaurant": "restaurant"}
DOMAIN_CATEGORY = {"Hotel": "constraint-driven", "Restaurant": "preference-driven"}

LLMS = ["gpt4", "qwen3", "gemeni"]
LLM_SUFFIX = {"gpt4": "gpt", "qwen3": "qwen3", "gemeni": "gemeni"}

TRAITS = [
    "Extraversion", "introversive",
    "Agreeableness", "disagreeable",
    "Conscientiousness", "unconscientious",
    "Neuroticism", "stable",
    "Openness", "closed",
]

COMPARISON_TYPES = ["TN", "ON", "TO"]
COMP_DIR = {"TN": "neutral_vs_try", "ON": "neutral_vs_trait", "TO": "try_vs_trait"}
# (condition of id1, condition of id2) per comparison type -- id1 is always
# the "A" side of the original judge prompt, id2 is always "B".
COMP_CONDITIONS = {
    "TN": ("neutral", "try"),
    "ON": ("neutral", "trait"),
    "TO": ("try", "trait"),
}
CONDITION_LABEL = {"neutral": "Neutral", "try": "Try", "trait": "Oracle"}

CELLS = [(llm, domain) for llm in LLMS for domain in DOMAINS]

SCRIPTED_CLOSING = "Thanks a lot! That's all I need."

ANNOTATOR_INSTRUCTIONS = (
    "Compare two dialogues with the same intended user trait. Choose which "
    "SYSTEM would feel more satisfying to the USER, considering the whole "
    "interaction. Focus on helpfulness, tone, responsiveness, patience, "
    "clarity, naturalness, and how well the system fits the user's "
    "communication style. Do not evaluate task success. Do not penalize "
    "a system only because search results were empty unless it handles "
    "that poorly. Do not use the scripted closing sentence as evidence "
    "of satisfaction. Choose tie only if both are similarly satisfying."
)

OUTPUT_COLUMNS = [
    "pair_id", "comparison_type", "trait", "llm_model", "domain",
    "domain_category", "dialogue_a", "dialogue_b", "display_true_mapping",
    "judge_verdict_display", "judge_reason",
]

PAIR_DIR_RE = re.compile(r"^pair_(.+)&(.+)$")
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PairRecord:
    pair_id: str
    comparison_type: str
    trait: str
    llm_model: str
    domain: str
    domain_category: str
    dialogue_a: str
    dialogue_b: str
    display_true_mapping: str
    judge_verdict_display: str
    judge_reason: str
    true_condition_a: str
    true_condition_b: str


class MissingLog:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, msg: str) -> None:
        self.lines.append(msg)

    def write(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + ("\n" if self.lines else ""), encoding="utf-8")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def dialogue_path(domain: str, case: str, llm: str, trait: str, run_id: str) -> Path:
    domain_lower = DOMAIN_LOWER[domain]
    suffix = LLM_SUFFIX[llm]
    return (
        BASE_DIR / domain / case / llm / f"{domain_lower}_{case}_{suffix}"
        / "addExample" / trait / run_id / "sys_log.txt"
    )


def judge_trait_dir(domain: str, comp_type: str, llm: str, trait: str) -> Path:
    return JUDGE_BASE / domain / COMP_DIR[comp_type] / llm / trait


def list_pair_dirs(domain: str, comp_type: str, llm: str, trait: str) -> list[Path]:
    d = judge_trait_dir(domain, comp_type, llm, trait)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_dir() and p.name.startswith("pair_"))


def parse_pair_ids(pair_dir_name: str) -> Optional[tuple[str, str]]:
    m = PAIR_DIR_RE.match(pair_dir_name)
    if not m:
        return None
    return m.group(1), m.group(2)


# ---------------------------------------------------------------------------
# Dialogue preprocessing -- renders each sys_log.txt into the Qualtrics
# HTML card template (Customer / Assistant bubbles, APICall and Search
# Results shown as distinct monospace blocks).
# ---------------------------------------------------------------------------

TURN_START_RE = re.compile(r"^(User|System):\s*(.*)$")
APICALL_INLINE_RE = re.compile(r"APICall\s*\(")

CUSTOMER_LABEL_STYLE = (
    "font-size:12px;text-transform:uppercase;letter-spacing:0.5px;"
    "color:#656c74;margin-bottom:3px;"
)
BUBBLE_TEXT_STYLE = "margin:0;font-size:17px;line-height:1.5;"
MONO_BLOCK_STYLE = (
    "margin-bottom:14px;font-family:ui-monospace,Menlo,Consolas,monospace;"
    "font-size:13px;color:#6b6355;background:#f1eee6;border-radius:6px;"
    "padding:8px 11px;white-space:pre-wrap;"
)
SEARCH_JSON_STYLE = MONO_BLOCK_STYLE + "overflow-x:auto;"
SEARCH_LABEL_STYLE = "margin-bottom:8px;font-size:12px;color:#656c74;font-style:italic;"

SEARCH_LABEL_START = f'<div style="{SEARCH_LABEL_STYLE}">Search Results</div>'
SEARCH_LABEL_END = f'<div style="{SEARCH_LABEL_STYLE}">End Search Results</div>'


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def render_bubble(role: str, text: str) -> str:
    label = "Customer" if role == "User" else "Assistant"
    escaped = _esc(text)
    if role == "User":
        return (
            '<div style="margin-bottom:14px;">'
            f'<div style="{CUSTOMER_LABEL_STYLE}">{label}</div>'
            f'<p style="{BUBBLE_TEXT_STYLE}">{escaped}</p>'
            "</div>"
        )
    return (
        '<div style="margin-bottom:14px;padding-left:14px;border-left:3px solid #2f6f6d;">'
        f'<div style="{CUSTOMER_LABEL_STYLE}">{label}</div>'
        f'<p style="{BUBBLE_TEXT_STYLE}">{escaped}</p>'
        "</div>"
    )


def render_apicall(text: str) -> str:
    return f'<div style="{MONO_BLOCK_STYLE}">{_esc(text)}</div>'


def render_search_json(text: str) -> str:
    return f'<div style="{SEARCH_JSON_STYLE}">{_esc(text)}</div>'


def split_apicalls(text: str) -> list[tuple[str, str]]:
    """Split a line into ('text', ...) / ('apicall', ...) pieces, in
    order, balancing parens on 'APICall(' / 'APICall (' so dict-valued
    parameters don't truncate the call early."""
    pieces: list[tuple[str, str]] = []
    i, n = 0, len(text)
    while i < n:
        m = APICALL_INLINE_RE.search(text, i)
        if not m:
            rest = text[i:].strip()
            if rest:
                pieces.append(("text", rest))
            break
        before = text[i:m.start()].strip()
        if before:
            pieces.append(("text", before))
        depth = 1
        j = m.end()
        while j < n and depth > 0:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
            j += 1
        pieces.append(("apicall", text[m.start():j].strip()))
        i = j
    return pieces


def render_html_dialogue(raw_text: str) -> str:
    """Extract alternating User/System turns from a sys_log.txt and render
    them as the Qualtrics HTML card template.

    Turn content is not always on the same line as the "User:"/"System:"
    marker -- some logs (notably qwen3) put the marker alone on a line
    followed by the message on the next line(s), so continuation lines
    are grouped under the last-seen marker. Within a turn, an inline or
    standalone APICall(...) becomes its own monospace block (splitting
    any narrative text around it into separate bubbles), and a
    Search Results / End Search Results pair brackets its JSON payload
    in its own monospace block -- matching the example template.
    """
    turns: list[list] = []  # [role, [raw_line, raw_line, ...]]
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = TURN_START_RE.match(stripped)
        if m:
            role, inline = m.group(1), m.group(2).strip()
            turns.append([role, [inline] if inline else []])
            continue
        if turns:
            turns[-1][1].append(stripped)

    blocks: list[str] = []
    for role, lines in turns:
        narrative: list[str] = []
        in_search = False
        search_lines: list[str] = []

        def flush_narrative() -> None:
            if not narrative:
                return
            text = " ".join(narrative).strip()
            narrative.clear()
            if not text:
                return
            if role == "User" and SCRIPTED_CLOSING in text:
                return
            blocks.append(render_bubble(role, text))

        for raw_line in lines:
            if raw_line.startswith("Search Results"):
                flush_narrative()
                in_search = True
                search_lines = []
                blocks.append(SEARCH_LABEL_START)
                continue
            if raw_line.startswith("End Search Results"):
                if search_lines:
                    blocks.append(render_search_json(" ".join(search_lines).strip()))
                    search_lines = []
                blocks.append(SEARCH_LABEL_END)
                in_search = False
                continue
            if in_search:
                search_lines.append(raw_line)
                continue
            for kind, piece in split_apicalls(raw_line):
                if kind == "apicall":
                    flush_narrative()
                    blocks.append(render_apicall(piece))
                else:
                    narrative.append(piece)

        flush_narrative()
        if search_lines:
            blocks.append(render_search_json(" ".join(search_lines).strip()))
            blocks.append(SEARCH_LABEL_END)

    return "".join(blocks)


# ---------------------------------------------------------------------------
# Judge response parsing
# ---------------------------------------------------------------------------

def parse_judge_response(path: Path) -> Optional[dict]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    m = JSON_FENCE_RE.search(raw)
    candidate = m.group(1) if m else raw.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def select_pair_dirs_for_group(
    trait: str, comp_type: str, rng: random.Random, missing: MissingLog
) -> list[tuple[str, str, Path]]:
    """Return list of (llm, domain, pair_dir) selections for one
    (trait, comparison_type) group, following the sampling design:
    one pair per cell (6 cells, no replacement) + 4 more from random
    cells with replacement, never reusing a pair folder within a cell."""
    cell_pairs = {
        cell: list_pair_dirs(cell[1], comp_type, cell[0], trait) for cell in CELLS
    }
    used: dict[tuple[str, str], set[str]] = {cell: set() for cell in CELLS}
    selected: list[tuple[str, str, Path]] = []

    first_pass_cells = CELLS[:]
    rng.shuffle(first_pass_cells)
    for cell in first_pass_cells:
        available = [d for d in cell_pairs[cell] if d.name not in used[cell]]
        if not available:
            missing.log(
                f"No judge pairs available for cell llm={cell[0]} domain={cell[1]} "
                f"trait={trait} comp={comp_type}"
            )
            continue
        choice = rng.choice(available)
        used[cell].add(choice.name)
        selected.append((cell[0], cell[1], choice))

    needed = PAIRS_PER_CELL_GROUP - len(selected)
    attempts = 0
    max_attempts = needed * 50 + 100
    while needed > 0 and attempts < max_attempts:
        attempts += 1
        cell = rng.choice(CELLS)
        available = [d for d in cell_pairs[cell] if d.name not in used[cell]]
        if not available:
            continue
        choice = rng.choice(available)
        used[cell].add(choice.name)
        selected.append((cell[0], cell[1], choice))
        needed -= 1

    if needed > 0:
        missing.log(
            f"Only found {len(selected)}/{PAIRS_PER_CELL_GROUP} pairs for "
            f"trait={trait} comp={comp_type} (pools exhausted)"
        )

    return selected


def build_record(
    domain: str, comp_type: str, llm: str, trait: str, pair_dir: Path,
    rng: random.Random, missing: MissingLog,
) -> Optional[PairRecord]:
    ids = parse_pair_ids(pair_dir.name)
    if ids is None:
        missing.log(f"Cannot parse pair id from folder name: {pair_dir}")
        return None
    id1, id2 = ids
    cond1, cond2 = COMP_CONDITIONS[comp_type]

    path1 = dialogue_path(domain, cond1, llm, trait, id1)
    path2 = dialogue_path(domain, cond2, llm, trait, id2)
    if not path1.is_file():
        missing.log(f"Missing dialogue file for id1 ({pair_dir}): {path1}")
        return None
    if not path2.is_file():
        missing.log(f"Missing dialogue file for id2 ({pair_dir}): {path2}")
        return None

    text1 = render_html_dialogue(path1.read_text(encoding="utf-8"))
    text2 = render_html_dialogue(path2.read_text(encoding="utf-8"))
    if not text1 or not text2:
        missing.log(f"Empty rendered dialogue for pair {pair_dir}")
        return None

    response_path = pair_dir / "response.txt"
    parsed = parse_judge_response(response_path)
    if parsed is None or "winner" not in parsed:
        missing.log(f"Cannot parse judge response: {response_path}")
        return None
    winner = str(parsed["winner"]).strip().lower()
    if winner not in ("a", "b", "tie"):
        missing.log(f"Unexpected winner value {parsed.get('winner')!r} in {response_path}")
        return None
    reason = str(parsed.get("reason", ""))

    # Original judge mapping: A = id1/cond1, B = id2/cond2.
    flip = rng.random() < 0.5
    if flip:
        dialogue_a, dialogue_b = text2, text1
        true_cond_a, true_cond_b = cond2, cond1
        verdict_map = {"a": "B", "b": "A", "tie": "tie"}
    else:
        dialogue_a, dialogue_b = text1, text2
        true_cond_a, true_cond_b = cond1, cond2
        verdict_map = {"a": "A", "b": "B", "tie": "tie"}

    display_true_mapping = (
        f"A={CONDITION_LABEL[true_cond_a]}, B={CONDITION_LABEL[true_cond_b]}"
    )
    judge_verdict_display = verdict_map[winner]
    pair_id = f"{comp_type}_{trait}_{llm}_{domain}_{id1}&{id2}"

    return PairRecord(
        pair_id=pair_id,
        comparison_type=comp_type,
        trait=trait,
        llm_model=llm,
        domain=domain,
        domain_category=DOMAIN_CATEGORY[domain],
        dialogue_a=dialogue_a,
        dialogue_b=dialogue_b,
        display_true_mapping=display_true_mapping,
        judge_verdict_display=judge_verdict_display,
        judge_reason=reason,
        true_condition_a=true_cond_a,
        true_condition_b=true_cond_b,
    )


def build_all_pairs(
    rng: random.Random, missing: MissingLog, plan_only: bool = False
):
    """Returns list[PairRecord] normally, or list of planned
    (trait, comp_type, llm, domain, pair_dir_name) tuples if plan_only."""
    plan: list[tuple[str, str, str, str, str]] = []
    records: list[PairRecord] = []
    for trait in TRAITS:
        for comp_type in COMPARISON_TYPES:
            selections = select_pair_dirs_for_group(trait, comp_type, rng, missing)
            for llm, domain, pair_dir in selections:
                if plan_only:
                    plan.append((trait, comp_type, llm, domain, pair_dir.name))
                    continue
                record = build_record(domain, comp_type, llm, trait, pair_dir, rng, missing)
                if record is not None:
                    records.append(record)
    return plan if plan_only else records


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_plan_summary(plan) -> None:
    from collections import Counter

    total = len(plan)
    by_comp = Counter(p[1] for p in plan)
    by_trait_comp = Counter((p[0], p[1]) for p in plan)
    by_cell = Counter((p[2], p[3]) for p in plan)

    print(f"Planned pairs: {total} (target {N_PAIRS})")
    print("\nBy comparison_type:")
    for comp in COMPARISON_TYPES:
        print(f"  {comp}: {by_comp.get(comp, 0)}")

    print("\nBy (trait, comparison_type) -- expect 10 each:")
    for trait in TRAITS:
        row = " ".join(f"{comp}={by_trait_comp.get((trait, comp), 0):2d}" for comp in COMPARISON_TYPES)
        print(f"  {trait:<18} {row}")

    print("\nBy cell (llm, domain) across all traits/comparisons:")
    for llm, domain in CELLS:
        print(f"  {llm:<6} {domain:<10} {by_cell.get((llm, domain), 0)}")


def write_coverage_report(records: list[PairRecord], path: Path) -> None:
    from collections import Counter

    lines = [f"Total pairs: {len(records)} (target {N_PAIRS})", ""]

    lines.append("By comparison_type:")
    c = Counter(r.comparison_type for r in records)
    for comp in COMPARISON_TYPES:
        lines.append(f"  {comp}: {c.get(comp, 0)}")

    lines.append("\nBy (trait, comparison_type):")
    c = Counter((r.trait, r.comparison_type) for r in records)
    for trait in TRAITS:
        row = " ".join(f"{comp}={c.get((trait, comp), 0):2d}" for comp in COMPARISON_TYPES)
        lines.append(f"  {trait:<18} {row}")

    lines.append("\nBy (llm_model, domain):")
    c = Counter((r.llm_model, r.domain) for r in records)
    for llm, domain in CELLS:
        lines.append(f"  {llm:<6} {domain:<10} {c.get((llm, domain), 0)}")

    lines.append("\nBy (llm_model, domain, comparison_type):")
    c = Counter((r.llm_model, r.domain, r.comparison_type) for r in records)
    for llm, domain in CELLS:
        row = " ".join(f"{comp}={c.get((llm, domain, comp), 0):2d}" for comp in COMPARISON_TYPES)
        lines.append(f"  {llm:<6} {domain:<10} {row}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_balance_check(records: list[PairRecord], path: Path) -> None:
    from collections import Counter

    lines = ["A/B flip balance per comparison_type", "(flip means displayed A/B differ from original judge A/B)", ""]
    flip_counts: dict[str, Counter] = {comp: Counter() for comp in COMPARISON_TYPES}
    for r in records:
        cond1, _cond2 = COMP_CONDITIONS[r.comparison_type]
        flipped = r.true_condition_a != cond1
        flip_counts[r.comparison_type]["flipped" if flipped else "not_flipped"] += 1

    for comp in COMPARISON_TYPES:
        cnt = flip_counts[comp]
        total = cnt["flipped"] + cnt["not_flipped"]
        lines.append(f"{comp}: flipped={cnt['flipped']} not_flipped={cnt['not_flipped']} total={total}")

    lines.append("\njudge_verdict_display distribution per comparison_type:")
    verdict_counts: dict[str, Counter] = {comp: Counter() for comp in COMPARISON_TYPES}
    for r in records:
        verdict_counts[r.comparison_type][r.judge_verdict_display] += 1
    for comp in COMPARISON_TYPES:
        cnt = verdict_counts[comp]
        lines.append(f"{comp}: A={cnt.get('A', 0)} B={cnt.get('B', 0)} tie={cnt.get('tie', 0)}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_master_all_pairs(records: list[PairRecord], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for r in records:
            row = {col: getattr(r, col) for col in OUTPUT_COLUMNS}
            writer.writerow(row)


def write_judge_labels(records: list[PairRecord], path: Path) -> None:
    fieldnames = [
        "pair_id", "display_true_mapping", "true_condition_a", "true_condition_b",
        "judge_verdict_display", "judge_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({col: getattr(r, col) for col in fieldnames})


def write_annotator_studies(
    records: list[PairRecord], annotator_dir: Path, shuffle_seed: int
) -> None:
    order = records[:]
    random.Random(shuffle_seed).shuffle(order)

    per_study = N_PAIRS // N_STUDIES
    assert per_study * N_STUDIES == N_PAIRS, "N_PAIRS must be evenly divisible by N_STUDIES"

    annotator_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = OUTPUT_COLUMNS + ["study_id", "pair_number_in_study"]

    for study_idx in range(N_STUDIES):
        study_id = study_idx + 1
        chunk = order[study_idx * per_study:(study_idx + 1) * per_study]
        study_path = annotator_dir / f"study_{study_id:02d}.csv"
        with study_path.open("w", newline="", encoding="utf-8") as f:
            f.write(f"# Annotator instruction: {ANNOTATOR_INSTRUCTIONS}\n")
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, r in enumerate(chunk, start=1):
                row = {col: getattr(r, col) for col in OUTPUT_COLUMNS}
                row["study_id"] = study_id
                row["pair_number_in_study"] = i
                writer.writerow(row)


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def verify_output(output_dir: Path) -> None:
    master_path = output_dir / "master" / "all_pairs.csv"
    print(f"\n--- Verifying {master_path} ---")
    with master_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Rows: {len(rows)}, columns: {reader.fieldnames}")
    for row in rows[:3]:
        print(f"  pair_id={row['pair_id']} mapping={row['display_true_mapping']} "
              f"verdict={row['judge_verdict_display']} "
              f"dialogue_a_chars={len(row['dialogue_a'])} dialogue_b_chars={len(row['dialogue_b'])}")

    study_path = output_dir / "annotator_1" / "study_01.csv"
    print(f"\n--- Verifying {study_path} ---")
    with study_path.open(encoding="utf-8") as f:
        first_line = f.readline()
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Comment line: {first_line.strip()[:80]}...")
    print(f"Rows: {len(rows)}")
    for row in rows[:3]:
        print(f"  study_id={row['study_id']} pair_number_in_study={row['pair_number_in_study']} "
              f"pair_id={row['pair_id']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print sampling plan and coverage without reading dialogue/judge files.")
    parser.add_argument("--verify", action="store_true", help="After building, re-read CSVs and print sample rows.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    missing = MissingLog()

    if args.dry_run:
        rng = random.Random(SEED)
        plan = build_all_pairs(rng, missing, plan_only=True)
        print_plan_summary(plan)
        if missing.lines:
            print(f"\n{len(missing.lines)} warnings (would go to missing_files.log):")
            for line in missing.lines[:20]:
                print(f"  {line}")
        return

    rng = random.Random(SEED)
    records = build_all_pairs(rng, missing, plan_only=False)
    print(f"Built {len(records)} pair records (target {N_PAIRS}).")
    if len(records) != N_PAIRS:
        print(f"WARNING: expected {N_PAIRS} records, got {len(records)}. See missing_files.log.")

    output_dir = args.output_dir
    master_dir = output_dir / "master"
    master_dir.mkdir(parents=True, exist_ok=True)

    write_master_all_pairs(records, master_dir / "all_pairs.csv")
    write_judge_labels(records, master_dir / "judge_labels.csv")
    write_coverage_report(records, master_dir / "coverage_report.txt")
    write_balance_check(records, master_dir / "balance_check.txt")
    missing.write(master_dir / "missing_files.log")

    for annotator, seed in ANNOTATOR_SEEDS.items():
        write_annotator_studies(records, output_dir / annotator, seed)

    print(f"Wrote output to {output_dir}")
    print(f"Missing/skip warnings logged: {len(missing.lines)}")

    if args.verify:
        verify_output(output_dir)


if __name__ == "__main__":
    main()
