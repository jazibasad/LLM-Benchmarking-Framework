# LLM Benchmarking Framework

A reproducible framework for evaluating and comparing free-tier Large Language Models
(Gemini, Mistral, Groq) across ten core capability dimensions, per the project proposal.

## Project Goals

- Build a disciplined, version-controlled research environment for LLM evaluation.
- Define an objective, weighted scoring rubric across ten capability dimensions.
- Collect empirical response data across multiple providers under real-world rate-limit conditions.
- Aggregate, analyze, and compare model performance against three research hypotheses (H1, H2, H3).
- Produce a final report synthesizing quantitative scores and qualitative failure analysis.

## Directory Structure

```
LLM-Benchmarking-Framework/
├── 01_Proposal/         # Official project proposal (PDF)
├── 02_Reports/          # Weekly progress reports (Week 1–8)
├── 03_Code/             # All automation scripts, validation scripts, and unit tests
├── 04_Datasets/         # Curated prompt set (P001–P220) and rubric definition
├── 05_Logs_Results/
│   ├── Gemini_Logs/            # Raw per-call response files
│   ├── Mistral_Logs/
│   ├── Groq_Logs/
│   ├── OpenRouter_Logs/
│   ├── environment_specs/      # Timestamped environment snapshots
│   ├── Combined_Results/       # Machine-readable consolidated raw JSON (per provider)
│   ├── Readable_Results/       # Human-readable raw response Word tables (per provider)
│   ├── Scored_Results/
│   │   ├── Gemini_Scores/      # Per-run scored JSON files
│   │   ├── Mistral_Scores/
│   │   ├── Groq_Scores/
│   │   ├── Combined_Scores/    # Machine-readable consolidated scores (per provider)
│   │   └── Readable_Scores/    # Human-readable scored Word tables (per provider)
│   └── tests_logs/
│       ├── Week_2/
│       └── Week_5/
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

`04_Datasets/rubric.docx` documents the full scoring methodology: the
standardized 0-5 score anchors (0 = Complete Failure through 5 = Complete
Success, 3 = Minimum Success Threshold), the Success/Failure determination
logic, all 10 benchmark categories, multi-module scoring for Hard-tier
prompts, and the stochastic reliability sample. Each of the 220 prompts also
carries its own specific `evaluation_criteria` in `prompts.json`, applied
using this common scale.

## Prompt Dataset (Week 3)

`04_Datasets/prompts.json` — 220 prompts across all 10 proposal categories
(22 each): Knowledge Retrieval, Multi-step Reasoning, Instruction Following,
Hallucination Stress Test, Coding & System Architecture, Ambiguity Handling,
Long-context Retention, Data Transformation, Summarization, Multilingual Tasks.

20 prompts (2 per category) are marked `stochastic_sample: true`,
`repeat_count: 5` for the proposal's Reliability requirement.

Run `python 03_Code/validate_prompts.py` to verify integrity.

## Controlled Environment Specification

`python 03_Code/capture_environment.py` — records OS, CPU, RAM, Python
version, real SDK versions (via `importlib.metadata`, not a fragile
`__version__` attribute lookup), network status, and model configuration.
Run once per collection session.

## Data Collection (Week 4)

Three decoupled provider runners, each recording `temperature` (fixed 0.7)
and measured `latency_seconds` per call, with full repeat-count support for
the stochastic sample:

- **Gemini** — `gemini-3.5-flash-lite` → `Gemini_Logs/`
- **Mistral** — `open-mistral-nemo` → `Mistral_Logs/`
- **Groq** — `openai/gpt-oss-120b` → `Groq_Logs/`

`build_results_tables.py` / `build_results_docx.py` consolidate raw
responses into machine-readable JSON and human-readable Word tables
(dedicated Temperature and Latency columns), correctly grouping all 5 runs
of each stochastic-sample prompt.

### Real incidents encountered and fixed during Week 4

This project has repeatedly hit free-tier model catalog volatility - a
genuine, recurring finding, not just inconvenience:

- **OpenRouter** (Week 2): a named free model was silently retired mid-project.
- **Gemini**: `gemini-2.5-flash` retired → replaced with `gemini-3.5-flash`
  → found to enforce a strict 20 requests/day cap → replaced again with
  `gemini-3.5-flash-lite`.
- **Cerebras**: introduced a mandatory payment-method requirement mid-project;
  replaced entirely with Mistral (permanent free tier, no card).
- **Groq**: `llama-3.3-70b-versatile` officially deprecated June 17, 2026
  (confirmed via Groq's own deprecation page) → replaced with
  `openai/gpt-oss-120b`, Groq's own recommended migration target.

Each incident was caught by the persistent-error detection mechanism,
which stops a run cleanly rather than wasting time retrying every
remaining prompt against a dead model or expired key.

## Automated Scoring (Week 5)

`llm_judge_scorer.py` scores **every individual response run** (not just
one per prompt) using a free LLM judge via OpenRouter, against the specific
evaluation criteria for that exact prompt.

**Explicit success/failure anchors** (per supervisor requirement, Aug 18):
every criterion is scored 0-5 against a standardized, documented scale
(full detail in `rubric.docx`). The Success/Failure outcome per criterion,
and the overall outcome per prompt run, are **computed in code** from this
documented threshold (score ≥ 3 = Success) — never trusted from the judge's
own labeling. A prompt run's overall outcome is Success only if every one
of its criteria individually succeeded.

`build_scored_tables.py` / `build_scored_docx.py` consolidate scored runs
into machine-readable JSON (with aggregate success/failure counts) and
human-readable Word tables (color-coded Success/Failure, failed rows and
stochastic-sample prompts visually highlighted).

### Real incident encountered and fixed during Week 5

Mid-scoring-run, OpenRouter returned a `401 Unauthorized: User not found`
error - an authentication failure, not a rate limit. This exposed a real
bug in the persistent-error detection: the keyword match for `"not_found"`
(underscore) didn't match the actual error text `"User not found"` (space),
so the script would have kept retrying every remaining prompt against the
broken key instead of stopping cleanly. Fixed by broadening the keyword
list to include `401`, `unauthorized`, `user not found`, and other common
auth-failure phrasings. Verified against the exact real error message
before being reused.

## Automation Pipeline (All Weeks)

Every script shares the same safety mechanisms:
- **Proactive throttling**, **reactive backoff + jitter**
- **Persistent-error detection** — stops early on systemic failures (quota,
  retired models, auth failures, billing issues)
- **Atomic cache writes** — no corruption from a hard interrupt
- **Resume-by-cache-file** — operates per individual run, so any
  interruption (including the auth failure above) never loses or repeats
  already-completed work

## Environment Setup

```bash
# In VS Code: Command Palette -> "Python: Create Environment" -> Venv
pip install -r requirements.txt
```

`.env` (never committed):
```
OPENROUTER_API_KEY=your_real_key_here
GEMINI_API_KEY=your_real_key_here
MISTRAL_API_KEY=your_real_key_here
GROQ_API_KEY=your_real_key_here
```

## Full Run Order (Weeks 4–5) — Completed

```
1. python 03_Code/capture_environment.py
2. python 03_Code/run_gemini_benchmark.py    (any order, independent)
3. python 03_Code/run_mistral_benchmark.py
4. python 03_Code/run_groq_benchmark.py
5. python 03_Code/build_results_tables.py
6. python 03_Code/build_results_docx.py
7. python 03_Code/tests/test_llm_judge_scorer.py   (10 passed)
8. python 03_Code/llm_judge_scorer.py
9. python 03_Code/build_scored_tables.py
10. python 03_Code/build_scored_docx.py
```

All 10 steps above have been executed for real — raw responses, temperature/latency,
environment specification, and automated scores for all 220 prompts (660 raw
responses, 300 calls per provider including the stochastic sample) are complete
across all three providers.

## Status

| Week | Focus | Status |
|------|-------|--------|
| 1 | Framework Initialization & Rubric Design | ✅ Complete |
| 2 | Automation Pipeline, Live OpenRouter Verification & Reproducible Environment | ✅ Complete |
| 3 | Dataset Curation (all 10 proposal categories) | ✅ Complete |
| 4 | Multi-Provider Collection, Temperature/Latency, Environment Spec, Repeat Logic | ✅ Complete |
| 5 | Automated Scoring with Explicit Success/Failure Anchors | ✅ Complete |
| 6 | Human Evaluation & Score Validation | ⏳ Not started |
| 7 | Statistical Analysis (mean, median, std dev, hypothesis testing) | ⏳ Not started |
| 8 | Final Report Compilation | ⏳ Not started |

**Open item:** "surveys" (mentioned by supervisor, Aug 12) — meaning unclear,
deferred until Week 6 when a human-evaluation form is actually needed.

See `02_Reports/` for detailed week-by-week progress reports.