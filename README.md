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
│   │   ├── human_scores_rater1.json / rater2.json / rater3.json
│   │   └── agreement_report.json
│   ├── Statistical_Analysis/
│   │   ├── reliability_statistics.json
│   │   ├── statistical_analysis.json
│   │   └── Week7_Statistical_Report.docx
│   └── tests_logs/Week_2/  Week_5/
└── 06_Final_Report/
```

## Research Hypotheses

- **H1:** Reasoning vs. retrieval performance variance across models.
- **H2:** Instruction-following degradation with prompt complexity.
- **H3:** Provider-specific hallucination patterns.

## Prompt Dataset (Week 3) — Complete

`04_Datasets/prompts.json` — 220 prompts, 10 categories (22 each), per
proposal Section 3. 20 marked `stochastic_sample: true` for 5x repetition.

## Data Collection (Week 4) — Complete

Gemini (`gemini-3.5-flash-lite`), Mistral (`open-mistral-nemo`), Groq
(`openai/gpt-oss-120b`) — temperature (fixed 0.7) and measured latency per
call, across all 220 prompts (300 calls per provider).

## Automated Scoring (Week 5) — Complete

`llm_judge_scorer.py` scored every response run against explicit 0-5
anchors (3 = success threshold), Success/Failure computed in code.

## Human-in-the-Loop Validation (Week 6) — Complete

3 independent raters scored a 30-prompt stratified sample (1 Easy + 1
Medium + 1 Hard per category), compared against the automated judge and
against each other using exact-match rate, correlation, and Cohen's Kappa.

## Statistical Analysis (Week 7) — Complete

Per the proposal's Statistical Analysis requirements, computed on the
scored dataset:

- **`compute_reliability_statistics.py`** — mean, median, standard
  deviation, and coefficient of variation across the 5 scored runs of each
  of the 20 stochastic-sample prompts, per provider (the proposal's
  Reliability requirement).
- **`statistical_analysis.py`** — 95% confidence intervals per provider;
  hypothesis testing for H1 (reasoning vs. retrieval), H2
  (instruction-following vs. difficulty), and H3 (hallucination scores
  across providers) using Welch's t-test with an exact Student's
  t-distribution p-value (via the regularized incomplete beta function,
  rather than a normal approximation); Cohen's d effect sizes; Pearson
  correlation (difficulty/latency vs. score); simple linear regression.
  Also incorporates Week 6's inter-rater agreement results, since the
  proposal lists this under the same Statistical Analysis section.
- **`build_statistical_report_docx.py`** — consolidates both scripts'
  output into one formatted Word report with color-coded tables.

## Automation Pipeline

Every data-collection/scoring script shares: proactive throttling, reactive
backoff + jitter, persistent-error detection, atomic cache writes,
resume-by-cache-file.

## Environment Setup

```bash
pip install -r requirements.txt
pip install python-docx scikit-learn
```

`.env`:
```
OPENROUTER_API_KEY=...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
GROQ_API_KEY=...
```

## Full Run Order — Completed Through Week 7

```
Week 4:  capture_environment.py -> run_{gemini,mistral,groq}_benchmark.py
         -> build_results_tables.py -> build_results_docx.py
Week 5:  test_llm_judge_scorer.py -> llm_judge_scorer.py
         -> build_scored_tables.py -> build_scored_docx.py
Week 6:  select_human_validation_sample.py -> build_human_validation_materials.py
         -> 3 raters independently scored Survey/human_scores_raterN.json
         -> compare_human_vs_judge.py
Week 7:  compute_reliability_statistics.py -> statistical_analysis.py
         -> build_statistical_report_docx.py
```

## Status

| Week | Focus | Status |
|------|-------|--------|
| 1 | Framework Initialization & Rubric Design | Complete |
| 2 | Automation Pipeline & Reproducible Environment | Complete |
| 3 | Dataset Curation (10 proposal categories) | Complete |
| 4 | Multi-Provider Collection | Complete |
| 5 | Automated Scoring with Success/Failure Anchors | Complete |
| 6 | Human-in-the-Loop Validation (3 raters, Cohen's Kappa) | Complete |
| 7 | Statistical Analysis (Reliability, CI, Hypothesis Testing, Correlation, Regression) | Complete |
| 8 | Final Report Compilation | Not started |

See `02_Reports/` for detailed week-by-week progress reports.
