# LLM Benchmarking Framework

A reproducible framework for evaluating and comparing free-tier Large Language Models
(Gemini, Mistral, Groq) across ten core capability dimensions, per the project proposal.

## Directory Structure

```
LLM-Benchmarking-Framework/
├── 01_Proposal/
├── 02_Reports/
├── 03_Code/
├── 04_Datasets/         # prompts.json, rubric.docx, human_validation_sample.json
├── 05_Logs_Results/
│   ├── Gemini_Logs/ Mistral_Logs/ Groq_Logs/ OpenRouter_Logs/
│   ├── environment_specs/
│   ├── Combined_Results/    Readable_Results/
│   ├── Scored_Results/
│   │   ├── Gemini_Scores/ Mistral_Scores/ Groq_Scores/
│   │   ├── Combined_Scores/ Readable_Scores/
│   ├── Survey/
│   │   ├── Human_Validation_Reading.docx
│   │   ├── human_scores_rater1.json
│   │   ├── human_scores_rater2.json
│   │   ├── human_scores_rater3.json
│   │   └── agreement_report.json
│   └── tests_logs/Week_2/  Week_5/
└── 06_Final_Report/
```

## Research Hypotheses

- **H1:** Reasoning vs. retrieval performance variance across models.
- **H2:** Instruction-following degradation with prompt complexity.
- **H3:** Provider-specific hallucination patterns.

## Prompt Dataset (Week 3) — Complete

`04_Datasets/prompts.json` — 220 prompts, 10 categories (22 each), per
proposal Section 3. 20 marked `stochastic_sample: true` for 5x repetition
(Reliability requirement). See `rubric.docx` for scoring methodology.

## Data Collection (Week 4) — Complete

Gemini (`gemini-3.5-flash-lite`), Mistral (`open-mistral-nemo`), Groq
(`openai/gpt-oss-120b`) — temperature (fixed 0.7) and measured latency per
call, across all 220 prompts (300 calls per provider including the
stochastic sample).

## Automated Scoring (Week 5) — Complete

`llm_judge_scorer.py` scored every response run against explicit 0-5
anchors (3 = success threshold), Success/Failure computed in code.

## Human-in-the-Loop Validation (Week 6) — Complete

**In plain terms:** Week 5's AI judge scored 660 responses automatically —
but an AI judge's reliability can't just be assumed, it has to be checked.
Week 6 had three real humans independently score a representative sample of
the same responses, then measured whether the humans and the AI agree, and
whether the humans agree with each other.

- **`select_human_validation_sample.py`** — stratified by category AND
  difficulty: 1 Easy, 1 Medium, 1 Hard prompt from EACH of the 10
  categories = 30 prompts, explicitly excluding the 20 stochastic-sample
  prompts (guaranteed zero overlap). Fixed seed (42), fully reproducible.
- **`build_human_validation_materials.py`** — generated a shared read-only
  reading document plus three separate scoring templates (90 items each:
  30 prompts x 3 providers), saved in `05_Logs_Results/Survey/`. Judge
  scores were withheld so each of the 3 raters scored independently.
- **`compare_human_vs_judge.py`** — computed exact-match rate,
  within-1-point rate, mean absolute difference, Pearson correlation, and
  **Cohen's Kappa** (standard + quadratic-weighted, cross-validated against
  scikit-learn's certified implementation) — both rater-vs-judge
  (validation) and rater-vs-rater (inter-rater agreement).

**Total completed:** 90 responses x 3 raters = 270 real scoring actions,
219 criteria compared per rater pairing. Real results: strong within-1-point
agreement across all raters (93-96%) with meaningful inter-rater correlation
(0.51-0.59) and Kappa (0.25-0.34), giving genuine evidence for evaluating
the automated scoring pipeline's reliability.

All scripts were verified against controlled data with known,
hand-calculable expected outcomes before being used on real project data.

## Statistical Analysis (Week 7) — Not Started

Mean/median/standard deviation/coefficient of variation on the 20
stochastic-sample prompts' 5 scored runs each, plus confidence intervals,
hypothesis testing, effect size, correlation, and regression across the
full scored dataset.

## Automation Pipeline

Every script shares: proactive throttling, reactive backoff + jitter,
persistent-error detection, atomic cache writes, resume-by-cache-file.

## Environment Setup

```bash
pip install -r requirements.txt
pip install scikit-learn   # only needed if re-validating Cohen's Kappa
```

`.env`:
```
OPENROUTER_API_KEY=...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
GROQ_API_KEY=...
```

## Full Run Order — Completed Through Week 6

```
Week 4:  capture_environment.py -> run_{gemini,mistral,groq}_benchmark.py
         -> build_results_tables.py -> build_results_docx.py
Week 5:  test_llm_judge_scorer.py -> llm_judge_scorer.py
         -> build_scored_tables.py -> build_scored_docx.py
Week 6:  select_human_validation_sample.py -> build_human_validation_materials.py
         -> 3 raters independently scored Survey/human_scores_raterN.json
         -> compare_human_vs_judge.py
```

## Status

| Week | Focus | Status |
|------|-------|--------|
| 1 | Framework Initialization & Rubric Design | Complete |
| 2 | Automation Pipeline & Reproducible Environment | Complete |
| 3 | Dataset Curation (10 proposal categories) | Complete |
| 4 | Multi-Provider Collection | Complete |
| 5 | Automated Scoring with Success/Failure Anchors | Complete |
| 6 | Human-in-the-Loop Validation (30-prompt stratified, 3 raters, Cohen's Kappa) | Complete |
| 7 | Statistical Analysis | Not started |
| 8 | Final Report Compilation | Not started |

See `02_Reports/` for detailed week-by-week progress reports.