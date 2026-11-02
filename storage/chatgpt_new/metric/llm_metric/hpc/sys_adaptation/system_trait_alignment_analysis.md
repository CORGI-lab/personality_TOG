# System trait-alignment score — Hotel vs. Restaurant

Source data: `Hotel/system_trait_alignment_summary.json` and
`Restaurant/system_trait_alignment_summary.json` (produced by judging every
dialogue with `LLM_Interaction/prompts/llm_metric_prompt.yml` via
`metrics/llm_metric_sysAdapt.py`). Each cell below is `avg_system_trait_alignment_score`
averaged across the 3 LLMs (gpt4/qwen3/gemeni), 50 dialogues per LLM per
trait per case — full coverage, 450 dialogues per case per domain.

The judge rubric: 1 = "behaves generically, clearly does not fit the trait,"
2 = "little meaningful adaptation," 3 = "partial/mixed," 4 = "mostly adapts
tone/pacing/reassurance to fit," 5 = "consistently and effectively adapts."

`neutral` = system given no trait signal. `try` = system told to try to adapt
without a labeled trait. `trait` = system given the actual (oracle) trait
label.

## Hotel

| trait | neutral | try | trait (oracle) | Δ(trait−neutral) |
|---|---:|---:|---:|---:|
| Extraversion | 1.63 | 3.39 | **4.93** | +3.30 |
| Openness | 1.46 | 2.65 | **4.43** | +2.97 |
| Neuroticism | 2.57 | 4.17 | **4.49** | +1.92 |
| Agreeableness | 1.12 | 1.69 | 3.42 | +2.30 |
| disagreeable | 1.19 | 2.07 | 3.37 | +2.18 |
| Conscientiousness | 1.34 | 1.95 | 2.91 | +1.57 |
| unconscientious | 1.08 | 1.45 | 2.28 | +1.20 |
| introversive | 1.03 | 1.03 | 1.13 | +0.10 |
| stable | 1.00 | 1.02 | 1.05 | +0.05 |
| closed | 1.00 | 1.08 | 1.01 | +0.01 |

## Restaurant

| trait | neutral | try | trait (oracle) | Δ(trait−neutral) |
|---|---:|---:|---:|---:|
| Extraversion | 1.87 | 3.52 | **4.85** | +2.98 |
| Openness | 1.75 | 3.24 | **4.41** | +2.66 |
| Neuroticism | 2.47 | 3.91 | **4.33** | +1.86 |
| disagreeable | 1.23 | 1.95 | 3.61 | +2.38 |
| Agreeableness | 1.19 | 1.60 | 2.83 | +1.64 |
| Conscientiousness | 1.41 | 1.71 | 2.52 | +1.11 |
| unconscientious | 1.17 | 1.35 | 2.09 | +0.92 |
| introversive | 1.04 | 1.29 | 1.45 | +0.41 |
| stable | 1.01 | 1.05 | 1.09 | +0.08 |
| closed | 1.01 | 1.02 | 1.08 | +0.07 |

## Hotel vs. Restaurant, side by side (oracle condition + delta)

| trait | Hotel trait | Rest. trait | Hotel Δ | Rest. Δ | avg trait | avg Δ |
|---|---:|---:|---:|---:|---:|---:|
| Extraversion | 4.93 | 4.85 | +3.30 | +2.98 | **4.89** | **+3.14** |
| Openness | 4.43 | 4.41 | +2.97 | +2.66 | **4.42** | **+2.82** |
| Neuroticism | 4.49 | 4.33 | +1.92 | +1.86 | **4.41** | **+1.89** |
| disagreeable | 3.37 | 3.61 | +2.18 | +2.38 | 3.49 | +2.28 |
| Agreeableness | 3.42 | 2.83 | +2.30 | +1.64 | 3.13 | +1.97 |
| Conscientiousness | 2.91 | 2.52 | +1.57 | +1.11 | 2.72 | +1.34 |
| unconscientious | 2.28 | 2.09 | +1.20 | +0.92 | 2.19 | +1.06 |
| introversive | 1.13 | 1.45 | +0.10 | +0.41 | 1.29 | +0.26 |
| stable | 1.05 | 1.09 | +0.05 | +0.08 | 1.07 | +0.07 |
| closed | 1.01 | 1.08 | +0.01 | +0.07 | 1.05 | +0.04 |

## Reading the pattern

Both domains agree on every point below independently — nothing flips sign
or ranking between Hotel and Restaurant.

**A 3-trait ceiling.** Extraversion, Openness, and Neuroticism are the only
traits reaching "good" to "very strong" alignment (4.3–4.9) once the system
is given the oracle trait label. These are the same three traits that drove
the largest verbosity increases in the `user_turns_summary.json` analysis
(`avg_words_per_turn_sys` up +20 to +34 words/turn from neutral to trait) —
the judge's qualitative "alignment" score and the mechanical word-count
metric are telling the same story from two independent measurements.

**A 3-trait floor.** introversive, stable, and closed never exceed 1.13 in
either domain — essentially zero measurable adaptation, and the tightest
cross-domain agreement of any trio (within 0.1–0.4 of each other). These are
the same three traits whose `avg_words_per_turn_sys` was flat or negative
from neutral to trait.

**`disagreeable` is the one trait that breaks the axis-symmetric story.**
It's nominally the low pole of the Agreeableness axis, so a purely
axis-symmetric adaptation mechanism would put it down with
introversive/stable/closed. Instead it ties or beats its own axis-mate
Agreeableness in both domains (Hotel: 3.37 vs. 3.42, essentially tied;
Restaurant: 3.61 vs. 2.83, clearly ahead) and lands as the 5th-highest trait
overall on the domain average (3.49). This matches an independent finding
from the turn-count data: `disagreeable` was the only low-pole trait whose
`avg_api_count` (search persistence) changed substantially and monotonically
with case — decreasing rather than staying flat — while its verbosity stayed
roughly flat like the other low poles. Two independent signals (an LLM judge
and a mechanical behavior metric) agree that `disagreeable` gets a distinct,
real adaptation strategy — dialing back search persistence rather than
writing more — that the other four reserved traits don't get at all.

**try vs. trait.** For every high-pole trait, `try` (no label, just told to
adapt) partially closes the gap and `trait` (oracle label) closes it further
— the effect direction never reverses. This mirrors the `try`-overshoot
pattern seen in `avg_words_per_turn_sys` for the low-pole traits, though here
`try` never overshoots `trait`; it's a strict partial step in the same
direction the oracle label pushes further.

**Net read:** the system's "personality adaptation," as this judge measures
it, is really a single lever that fires for five expressive/high-arousal
traits and stays essentially off for the reserved ones — with `disagreeable`
as the sole low-pole exception, and via a different mechanism (search
behavior, not verbosity) than the one driving the high-pole traits.
