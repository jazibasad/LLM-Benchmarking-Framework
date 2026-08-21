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
call.

## Automated Scoring (Week 5) — Complete

`llm_judge_scorer.py` scores every response run against explicit 0-5
anchors (3 = success threshold), Success/Failure computed in code.

## Human-in-the-Loop Validation (Week 6)

**In plain terms:** Week 5's AI judge scored 660 responses automatically —
but an AI judge's reliability can't just be assumed, it has to be checked.
Week 6 has real humans independently score a sample of the same responses,
then measures whether the humans and the AI agree. If they agree well, the
automated scores for the rest of the dataset can be trusted.

- **`select_human_validation_sample.py`** — **stratified by category AND
  difficulty**: 1 Easy, 1 Medium, 1 Hard prompt from EACH of the 10
  categories = **30 prompts**, guaranteeing every difficulty tier is
  represented in every category (not left to chance). Explicitly excludes
  the 20 stochastic-sample prompts (verified: they don't even cover all
  three difficulties in every category, so couldn't support this
  stratification if reused). Fixed seed (42), fully reproducible.
- **`build_human_validation_materials.py`** — generates a shared read-only
  reading document plus **three separate scoring templates** (90 items
  each: 30 prompts × 3 providers), saved in `05_Logs_Results/Survey/`.
  Judge scores deliberately withheld for independent scoring.
- **`compare_human_vs_judge.py`** — computes exact-match rate,
  within-1-point rate, mean absolute difference, Pearson correlation, and
  **Cohen's Kappa** (standard + quadratic-weighted, cross-validated against
  scikit-learn's certified implementation) — both rater-vs-judge
  (validation) and rater-vs-rater (inter-rater agreement).

**Total workload:** 30 prompts × 3 providers × 3 raters = 270 individual
scoring actions.

All scripts verified against controlled data with known, hand-calculable
expected outcomes, and re-tested end-to-end after the sampling redesign.

## Statistical Analysis (Week 7) — Not Started

Mean/median/standard deviation/coefficient of variation on the 20
stochastic-sample prompts' 5 scored runs each, plus confidence intervals,
hypothesis testing, effect size, correlation, and regression.

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

## Full Run Order

```
Week 4:  capture_environment.py → run_{gemini,mistral,groq}_benchmark.py
         → build_results_tables.py → build_results_docx.py
Week 5:  test_llm_judge_scorer.py → llm_judge_scorer.py
         → build_scored_tables.py → build_scored_docx.py
Week 6:  select_human_validation_sample.py → build_human_validation_materials.py
         → [3 raters independently score Survey/human_scores_raterN.json]
         → compare_human_vs_judge.py
```

## Status

| Week | Focus | Status |
|------|-------|--------|
| 1 | Framework Initialization & Rubric Design | ✅ Complete |
| 2 | Automation Pipeline & Reproducible Environment | ✅ Complete |
| 3 | Dataset Curation (10 proposal categories) | ✅ Complete |
| 4 | Multi-Provider Collection | ✅ Complete |
| 5 | Automated Scoring with Success/Failure Anchors | ✅ Complete |
| 6 | Human-in-the-Loop Validation (30-prompt stratified, 3 raters, Cohen's Kappa) | 🟡 Code complete, manual scoring + real execution pending |
| 7 | Statistical Analysis | ⏳ Not started |
| 8 | Final Report Compilation | ⏳ Not started |

See `02_Reports/` for detailed week-by-week progress reports.
