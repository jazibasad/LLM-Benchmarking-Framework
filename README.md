# LLM Benchmarking Framework

A reproducible framework for evaluating and comparing free-tier Large Language Models
(Gemini, Mistral, Groq) across ten core capability dimensions, per the project proposal.

**Project Status: Complete.** All eight weeks executed, documented, and the final
research report, presentation, and reproducibility package delivered. Weeks 1-7
supervisor-approved.

## Directory Structure

```
LLM-Benchmarking-Framework/
├── 01_Proposal/
├── 02_Reports/
│   ├── week1_progress_report.docx ... week8_progress_report.docx
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
    ├── Final_Research_Report.docx           # 53 pages
    ├── Final_Presentation.pptx               # 15 slides
    └── Reproducibility_Package_Checklist.docx
```

## Research Hypotheses

- **H1:** Reasoning vs. retrieval performance variance across models.
- **H2:** Instruction-following degradation with prompt complexity.
- **H3:** Provider-specific hallucination patterns.

## Framework Initialization & Rubric Design (Week 1) — Complete

Repository structure established (`01_Proposal/` through `06_Final_Report/`).
Original master scoring rubric authored, defining the 0-5 score anchors
later refined and finalized in Week 3. The three research hypotheses
(H1, H2, H3) were formulated at this stage and carried through unchanged
to the final hypothesis testing in Week 7.

## Automation Pipeline & Reproducible Environment (Week 2) — Complete

`benchmark_runner.py` built and verified against OpenRouter as an initial
development and testing provider, establishing the core automation pattern
used by all later provider runners: proactive throttling, reactive
backoff with jitter, atomic cache writes, and resume-by-cache-file logic.
`test_benchmark_runner.py` — 9/9 unit tests passing. Python virtual
environment and `requirements.txt` established for reproducible dependency
management.

**Real incident:** a named OpenRouter free-tier model was discontinued
mid-verification (persistent HTTP 404), correctly detected by the
persistent-error logic after 6 retries with increasing backoff. Resolved
by switching to OpenRouter's `openrouter/free` router, which selects a
currently available free model rather than depending on one fixed name
liable to disappear.

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
against each other using exact-match rate, correlation, and Cohen's kappa.

## Statistical Analysis (Week 7) — Complete

Reliability (mean/median/stdev/coefficient of variation on the stochastic
sample), 95% confidence intervals, hypothesis testing (Welch's t-test with
exact p-values), Cohen's d effect sizes, Pearson correlation, and
regression — all computed on the real scored dataset. Real result: Groq's
hosted model achieved the highest mean score, lowest score variance, and
the study's one statistically significant, large-effect finding
(hallucination resistance vs. Mistral, p = 0.01, d = 0.84).

## Final Report, Presentation & Reproducibility Package (Week 8) — Complete

- **`Final_Research_Report.docx`** — 53-page report covering
  introduction, related work, full methodology, experimental setup with
  real incident history, results, discussion (including a formal
  construct/internal/external/conclusion validity analysis), ethical
  considerations, conclusion, references, and six appendices (formulas,
  reproducibility package, dataset examples, rubric walkthrough, glossary,
  derivations). Eight figures and thirteen tables, all real data.
- **`Final_Presentation.pptx`** — 15-slide summary deck covering
  motivation, methodology, real results charts, the headline hallucination
  finding, limitations, and conclusions.
- **`Reproducibility_Package_Checklist.docx`** — inventory of all 32 real
  deliverables across source code, dataset, configuration, logs, human
  validation records, statistical outputs, and final documentation, each
  traced to the week it was built.

## Automation Pipeline

Every data-collection/scoring script shares: proactive throttling, reactive
backoff + jitter, persistent-error detection, atomic cache writes,
resume-by-cache-file.

## Environment Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` includes all project dependencies, including
`python-docx` and `scikit-learn` (used for report generation and Cohen's
Kappa validation respectively).

`.env`:
```
OPENROUTER_API_KEY=...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
GROQ_API_KEY=...
```

## Full Run Order

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
Week 8:  Final report, presentation, and reproducibility checklist compiled
         from the verified outputs of Weeks 1-7.
```

## Status

| Week | Focus | Status |
|------|-------|--------|
| 1 | Framework Initialization & Rubric Design | Complete — Approved |
| 2 | Automation Pipeline & Reproducible Environment | Complete — Approved |
| 3 | Dataset Curation (10 proposal categories) | Complete — Approved |
| 4 | Multi-Provider Collection | Complete — Approved |
| 5 | Automated Scoring with Success/Failure Anchors | Complete — Approved |
| 6 | Human-in-the-Loop Validation (3 raters, Cohen's Kappa) | Complete — Approved |
| 7 | Statistical Analysis (Reliability, CI, Hypothesis Testing, Correlation, Regression) | Complete — Approved |
| 8 | Final Report, Presentation & Reproducibility Package | Complete |

See `02_Reports/` for detailed week-by-week progress reports.
