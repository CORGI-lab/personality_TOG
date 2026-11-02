# Persona-Guided LLM Agents for Task-Oriented Dialogue

Code, generated dialogues, evaluator prompts, LLM-judge outputs, human-evaluation data and
analysis scripts for the paper

> **Persona-Guided LLM Agents for Task-Oriented Dialogue**
> Maryam Shoaeinaeini, Brent Harrison, A.B. Siddique — University of Kentucky

A personality-conditioned **user agent** talks to an API-grounded **system agent** on the
Hotel and Restaurant domains of the Schema-Guided Dialogue (SGD) dataset. The system runs
under three personality-access conditions:

| Condition | Folder name | What the system receives |
|---|---|---|
| Neutral | `neutral` | no personality information |
| Try | `try` | instruction to adapt to personality cues observed in the dialogue (no label) |
| Oracle | `trait` | the user's target trait and its description |

System LLMs: GPT-4o (`gpt4`), Qwen3-Next-80B (`qwen3`), Gemini 2.0 Flash (`gemeni`).
User traits: Extraversion / `introversive`, Agreeableness / `disagreeable`,
Conscientiousness / `unconscientious`, Neuroticism / `stable`, Openness / `closed`.
2 domains × 3 conditions × 3 LLMs × 10 traits × 50 dialogues = **9,000 dialogues**.

Every LLM-judged metric is scored by two judges, **Gemma-4-31B** (Ollama Cloud) and
**GPT-5-mini** (OpenRouter), and the paper reports their average.

---

## Repository layout

```
generate_dialogues.py            # two-agent dialogue generation (Hydra entry point)
LLM_Interaction/
  llm_models_api.py              # generate_response (agents) and judge_response (judges); keys from .env
  sys_simulator.py, utils.py     # dialogue loop, API-call parsing, database lookup
  schemas/schema.txt             # SGD schemas
  prompts/
    consts.py                    # P^2 trait descriptions (both poles) and demonstration dialogues
    Hotel/, Restaurant/          # user and system prompts per domain / LLM / condition
    llm_metric_prompt_USER_evident_trait.yml          # judge: User Trait Score
    llm_evaluator_prompt_SYS.yml                      # judge: Constraint Satisfaction, Inform Rate, Truthfulness
    llm_evaluator_prompt_SYS_satisfaction_pairly.yml  # judge: pairwise user satisfaction
    llm_metric_prompt.yml                             # judge: System Adaptation Score
    llm_bfiEvaluator_prompt_SYS.yml                   # BFI-44 probe of the trait prompts
config_files/                    # Hydra configs for generation and every metric
metrics/
  task_metrics/dialog_completion.py, task_success.py  # Dialogue Completion, Task Success (no LLM)
  llm_metric_trait_score.py      # User Trait Score (judge -> score -> summarize)
  llm_metric_evaluation_SYS.py   # aggregates CS / IR / Truthfulness judge responses
  llm_eval_pairly.py             # pairwise user satisfaction (A/B order control)
  llm_metric_sysAdapt.py         # System Adaptation Score
  bfi_eval.py                    # BFI-44 probe
  words_turn.py                  # judge-free words-per-turn statistics
analysis/                        # scripts that produce the numbers, tables and figures in the paper
human_eval/                      # human study: builder, anonymized ratings, agreement script
figures/                         # Figure 1 (Overview1.png), Figure 2 (and the cell-level variant)
service_results/                 # SGD Hotels_4 / Restaurants_1 databases (original + cleaned)
data_cleaning.py                 # removes duplicate / inconsistent records from the databases
bfi_44/bfi.json                  # BFI-44 items
storage/chatgpt_new/             # all released data (dialogues and metric outputs, see below)
```

### Released data (`storage/chatgpt_new/`)

| Path | Content |
|---|---|
| `{Hotel,Restaurant}/{neutral,try,trait}/{gpt4,qwen3,gemeni}/<prompt_name>/` | `prompt_system.txt`, `prompt_user.txt` (exact prompts used) and `addExample/<trait>/<run_id>/sys_log.txt` (one dialogue each, 50 per trait) |
| `metric/dialog_completion/...` | Dialogue Completion per domain / condition / LLM / trait |
| `metric/task_completion/task_completion2/...` | Task Success per domain / condition / LLM / trait |
| `metric/llm_metric/hpc/llm_metric_USER/New_1to5_scale/User_trait_score/{gemma,gpt5_mini}/` | raw User-Trait judge responses per dialogue (+ per-dialogue `accuracy.txt`) |
| `metric/llm_metric/hpc/llm_evaluation/USER/New_1to5_scale/User_trait_score/{gemma,gpt5_mini}/` | aggregated User Trait Scores (`avg_across_cases_by_llm.json`, `user_trait_score_table.json`) |
| `metric/llm_metric/hpc/llm_metric_SYS/New_1to5_scale/System_metrics/{gemma4,gpt5_mini}/` | raw CS / IR / Truthfulness judge responses per dialogue |
| `metric/llm_metric/hpc/llm_metric_SYS/New_1to5_scale/User_Satisfaction_Pairly/{gemma,gpt5-mini}/` | pairwise judgements per pair (`response.txt` for Gemma; `verdict.json` + `summary.json` for GPT-5-mini) |
| `metric/llm_metric/hpc/llm_metric_SYS/New_1to5_scale/User_Satisfaction_Pairly/human_study_extra/` | both judges' verdicts for the 3 human-study pairs that are not in the main runs (see Notes) |
| `metric/llm_metric/hpc/sys_adaptation/` | System Adaptation Score: Gemma in the top level, GPT-5-mini in `gpt5_mini/` |
| `metric/llm_metric/hpc/llm_metric_SYS/BFI/` | BFI-44 probe answers per prompt model |
| `qualtrics_studies/` | the 12 Qualtrics study files (2 annotator orders) and the master pair list |

Folder-name conventions: `trait` = Oracle, `gemeni` = Gemini 2.0 Flash, `gpt4` = GPT-4o.
Pairwise folders `neutral_vs_trait`, `neutral_vs_try`, `try_vs_trait` correspond to O/N, T/N
and T/O in the paper; inside a pair folder `pair_<dialogueA>&<dialogueB>`, the first id belongs to
the first condition of the folder name.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Python >= 3.9
pip install -r requirements.txt
cp .env.example .env                                     # then fill in the keys you need
```

All scripts are run **from the repository root**; every path in the configs is relative to it
(Hydra is configured with `job.chdir: false`). Configuration values can be overridden on the
command line, e.g. `python metrics/llm_metric_sysAdapt.py domain=[Hotel] num_workers=4`.

> **Cost note.** Generation and every judge stage make one API call per dialogue (or pair).
> The released outputs let you reproduce all paper numbers **without any API call** — start
> with the analysis scripts below.

---

## Reproducing the paper from the released outputs (no API calls)

| Paper item | Command |
|---|---|
| Table I – User Trait Score per trait (avg. 3.520), top-3 detection binomial tests, Friedman test across conditions (§V-A, §V-C) | `python analysis/table_trait_realization.py` |
| Table II – CS / IR / Truthfulness and Wilcoxon tests (§V-B) | `python analysis/sys_significance.py` |
| Table II – pairwise user satisfaction per trait, per judge and averaged | `python analysis/compare_pairly_judges.py` |
| Table III – System Adaptation Score, Wilcoxon tests, judge agreement (§V-C) | `python analysis/compare_sysadapt.py` |
| Spearman correlations (Table "realization_corr", §V-C) | `python analysis/realization_corr.py` |
| Figure 2 (trait level, n = 10) and the cell-level variant (n = 60) | `python analysis/figure_realization_two_judges.py` |
| Human evaluation (§V-D) | `python human_eval/human_agreement.py` |

Task-outcome numbers in Table I (Dialogue Completion, Task Success) are stored in
`storage/chatgpt_new/metric/dialog_completion/` and `storage/chatgpt_new/metric/task_completion/task_completion2/`;
BFI scores are in `storage/chatgpt_new/metric/llm_metric/hpc/llm_metric_SYS/BFI/`.

---

## Re-running the pipeline

### 1. Generate dialogues

`config_files/gpt-interactive-trait.yaml` selects one domain / LLM / condition per run:

```bash
python generate_dialogues.py \
  domain=Restaurant service_name=Restaurants_1 \
  dataset_path=./service_results/Restaurants_1_normalized.csv \
  model_name=gpt-4o api_source=openai model=gpt4 prompt_name=restaurant_trait_gpt \
  sys_prompt_path=./LLM_Interaction/prompts/Restaurant/gpt4/restaurant_prompt_trait_SYS.yml \
  user_prompt_path=./LLM_Interaction/prompts/Restaurant/restaurant_prompt_trait_USER.yml \
  output_path=./storage/chatgpt_new/Restaurant
```

Generation settings (as in `config_files/gpt-interactive-trait.yaml` and `llm_models_api.py`):
temperature 0.7, top-p 1.0 and at most 250 output tokens per call for all three models (GPT-4o,
Gemini 2.0 Flash, Qwen3-Next-80B). A dialogue stops at the
scripted closing phrase, after `max_turns` (40) turns (each user turn, system turn and API call
counts as one; at most ~20 user turns), or after 5 API calls; one in-domain
demonstration dialogue is used (`add_example: true`). Hotel uses `Hotels_4` / `Hotels_4_normalized2.csv`.

**Model access.** All model calls go through `LLM_Interaction/llm_models_api.py`, which keeps the
two roles separate and never hard-codes a model:

* `generate_response(prompt, model_name, api_source, ...)` — dialogue generation. The user agent
  (`generate_dialogues.py`) and the system agent (`InteractiveChatSession.generate` in
  `LLM_Interaction/sys_simulator.py`, used for normal turns and API-call retries) both use the
  generation model from the config. The BFI probe (`metrics/bfi_eval.py`) uses it too.
* `judge_response(prompt, judge_model, judge_provider, ...)` — every LLM judge.

| Role | Model | `api_source` / `judge_provider` | model id |
|---|---|---|---|
| generation | GPT-4o | `openai` | `gpt-4o` |
| generation | Gemini 2.0 Flash | `open_route` | `google/gemini-2.0-flash-001` |
| generation | Qwen3-Next-80B | `ollama` | `qwen3-next:80b` |
| judge | GPT-5-mini | `open_route` | `openai/gpt-5-mini` (`reasoning_effort: low`, `gpt_max_tokens: 4000`) |
| judge | Gemma-4-31B | `ollama` | `gemma4:31b-cloud` |

### 2. Task outcomes (no LLM)

```bash
python metrics/task_metrics/dialog_completion.py   # config_files/dialog_completion.yaml
python metrics/task_metrics/task_success.py        # config_files/task_completion.yaml
```

### 3. LLM-judged metrics

Each judge script reads `judge_provider`, `judge_model`, `reasoning_effort`, `gpt_max_tokens`,
`num_workers` and `skip_existing` (resume) from its config. The configs default to the GPT-5-mini
judge; for Gemma pass `judge_provider=ollama judge_model=gemma4:31b-cloud reasoning_effort=null`.
With `skip_existing: true` and the
released outputs in place, the judge stage makes no API calls.

| Metric | Script | Config | Prompt |
|---|---|---|---|
| User Trait Score | `metrics/llm_metric_trait_score.py` | `llm_evaluation.yaml` (`stages: [judge, score, summarize]`) | `llm_metric_prompt_USER_evident_trait.yml` |
| CS / IR / Truthfulness | `metrics/llm_metric_trait_score.py` with `stages=[judge]` and the SYS prompt, then `metrics/llm_metric_evaluation_SYS.py` to aggregate | `llm_evaluation.yaml` | `llm_evaluator_prompt_SYS.yml` |
| Pairwise satisfaction | `metrics/llm_eval_pairly.py` | `llm_metric.yaml` (`order_mode: random`) | `llm_evaluator_prompt_SYS_satisfaction_pairly.yml` |
| System Adaptation | `metrics/llm_metric_sysAdapt.py` | `llm_metric_sysAdapt.yaml` | `llm_metric_prompt.yml` |
| BFI-44 probe | `metrics/bfi_eval.py` | `llm_bfi.yaml` | `llm_bfiEvaluator_prompt_SYS.yml` |
| Words per turn | `metrics/words_turn.py` | `llm_metric_trait.yaml` | – |

Example (GPT-5-mini System Adaptation run used in the paper, 9,000 calls):

```bash
python metrics/llm_metric_sysAdapt.py                     # GPT-5-mini (config default)
python metrics/llm_metric_sysAdapt.py judge_provider=ollama judge_model=gemma4:31b-cloud \
  reasoning_effort=null output_path=./storage/chatgpt_new/metric/llm_metric/hpc/sys_adaptation/gemma_rerun
```

---

## Human evaluation

* `human_eval/build_qualtrics_pairwise_study.py` builds the 300-pair stratified pool and the
  Qualtrics loop-and-merge files in `storage/chatgpt_new/qualtrics_studies/` (12 studies × 25 pairs,
  two display orders).
* The paper analyses the 7 studies with complete, quality-screened responses (studies 2, 4, 5,
  9, 10, 11, 12; 175 pairs, 2 annotators each, 350 ratings).
* `human_eval/human_ratings.csv` contains the anonymized ratings (`study, pair_id, rater,
  choice_display`). Raw Qualtrics exports (IP addresses, locations, Prolific ids, free-text
  explanations) are **not** released.
* `human_eval/human_agreement.py` computes annotator–annotator and annotator–judge agreement
  (percent agreement and Cohen's κ, with and without ties) for both judges.

---

## Notes

* **Position bias.** GPT-5-mini and Gemma 4 judged each pair in a seeded random A/B order
  (`order_mode: random`, recorded in each `verdict.json`). 
* **System Adaptation, Gemma.** Raw per-dialogue responses are released for Restaurant; for Hotel
  only the per-cell summary (`Hotel/system_trait_alignment_summary.json`) is available.
  Restaurant `introversive` is in `system_trait_alignment_summary_full.json`.
  One Gemma Hotel/Try/GPT-4o dialogue is missing (8,999 of 9,000).
* **Human-study judge labels.** The `judge_verdict_display` / `judge_reason` columns in
  `qualtrics_studies/` come from an earlier Gemma run and were used only to build the study;
  the paper's agreement numbers use the released Gemma and GPT-5-mini judgements.
* Three human-study pairs (all Gemini × Restaurant × introversive) are not in the main pairwise
  runs: dialogues in that cell were regenerated after the study was built, and the judges' seeded
  random pairing then matched them differently. These 3 pairs were judged separately by both
  judges with the same prompt and settings (`human_eval/judge_extra_pairs.py`, outputs in
  `User_Satisfaction_Pairly/human_study_extra/`). They are used only for the human-agreement
  analysis, not for Table II.
* Figure 1 (framework overview, `figures/Overview1.png`) was drawn by hand and is not generated by code.
* Some Qwen3 Hotel prompt files carry `claude` in their file name for historical reasons; the
  exact prompt used for every run is stored next to the dialogues (`prompt_system.txt`,
  `prompt_user.txt`).

## Citation

```bibtex
@inproceedings{shoaeinaeini2026persona,
  title     = {Persona-Guided {LLM} Agents for Task-Oriented Dialogue},
  author    = {Shoaeinaeini, Maryam and Harrison, Brent and Siddique, A. B.},
  year      = {2026}
}
```

## Acknowledgment

This work was supported in part by the National Science Foundation (NSF) under grant IIS-2401685.
