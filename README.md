# LLM Benchmarking Framework

A reproducible framework for evaluating and comparing free-tier Large Language Models
(Gemini, OpenAI, Groq) across five core capability dimensions.

## Project Goals

- Build a disciplined, version-controlled research environment for LLM evaluation.
- Define an objective, weighted 0–5 point scoring rubric across five capability dimensions.
- Collect empirical response data across multiple providers under real-world rate-limit conditions.
- Aggregate, analyze, and compare model performance against three research hypotheses (H1, H2, H3).
- Produce a final report synthesizing quantitative scores and qualitative failure analysis.

## Directory Structure

```
LLM-Benchmarking-Framework/
├── 01_Proposal/         # Official project proposal (PDF)
├── 02_Reports/          # Weekly progress reports (Week 1–8)
├── 03_Code/             # All automation scripts and unit tests
├── 04_Datasets/         # Prompt sets (P001–P220) and rubric definition
├── 05_Logs_Results/     # Per-model JSON response logs (partitioned by provider)
│   ├── Gemini_Logs/
│   ├── OpenAI_Logs/
│   └── Groq_Logs/
└── 06_Final_Report/     # Consolidated final research document
```

## Research Hypotheses

- **H1:** Models will show measurable variance in multi-step reasoning performance
  relative to simple knowledge-retrieval performance.
- **H2:** Instruction-following accuracy will degrade as prompt complexity
  (number of constraints) increases, at different rates across models.
- **H3:** Models will exhibit provider-specific hallucination patterns under
  adversarial stress-test prompts, rather than a uniform failure rate.

## Evaluation Rubric

See [`04_Datasets/rubric.docx`](04_Datasets/rubric.docx) for the full weighted 0–5
scoring criteria across: Knowledge Retrieval, Multi-step Reasoning, Instruction
Following, Hallucination Stress Test, and Coding Tasks.

## Status

| Week | Focus | Status |
|------|-------|--------|
| 1 | Framework Initialization & Rubric Design | ✅ Complete |
| 2 | Automation Architecture & Pipeline Initialization | ⏳ Not started |
| 3 | Multi-Model Decoupling & Automated Resumption | ⏳ Not started |
| 4 | Experimental Data Collection & Quantitative Analysis | ⏳ Not started |
| 5 | Statistical Aggregation & Comparative Scoring | ⏳ Not started |
| 6 | Error Analysis & Failure Mode Taxonomy | ⏳ Not started |
| 7 | Discussion, Hypothesis Testing & Synthesis | ⏳ Not started |
| 8 | Final Report Compilation & Deliverable Packaging | ⏳ Not started |

See `02_Reports/` for detailed week-by-week progress reports.