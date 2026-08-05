# LLM Benchmarking Framework

A reproducible framework for evaluating and comparing free-tier Large Language Models
(Gemini, Cerebras, Groq) across five core capability dimensions.

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
├── 03_Code/             # All automation scripts, validation scripts, and unit tests
├── 04_Datasets/         # Curated prompt set (P001–P220) and rubric definition
├── 05_Logs_Results/     # Per-model JSON response logs + weekly test result logs
│   ├── Gemini_Logs/
│   ├── Cerebras_Logs/
│   ├── Groq_Logs/
│   ├── OpenRouter_Logs/
│   └── tests_logs/
│       ├── Week_2/
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

See [`04_Datasets/rubric.docx`](04_Datasets/rubric.docx) for the master weighted
0–5 scoring criteria across: Knowledge Retrieval, Multi-step Reasoning, Instruction
Following, Hallucination Stress Test, and Coding Tasks.

## Prompt Dataset (Week 3)

`04_Datasets/prompts.json` contains 220 curated evaluation prompts, each with:

- `id` — P001 through P220
- `category` — Knowledge Retrieval, Multi-step Reasoning, Instruction Following,
  Hallucination Stress Test, or Coding & System Architecture
- `difficulty` — Easy, Medium, or Hard
- `prompt` — the prompt text itself
- `evaluation_criteria` — specific, per-prompt scoring criteria (extends the
  master rubric with prompt-level granularity)
- `max_score` — the maximum achievable score for that prompt

**Verified breakdown** (see `03_Code/validate_prompts.py`):
- By category: Multi-step Reasoning 50, Coding & System Architecture 50,
  Knowledge Retrieval 40, Instruction Following 40, Hallucination Stress Test 40
- By difficulty: Hard 100, Medium 66, Easy 54

Run `python 03_Code/validate_prompts.py` to re-verify the dataset's structural
integrity (field completeness, unique IDs, valid difficulty values) at any time.

## Automation Pipeline (Week 2)

`03_Code/benchmark_runner.py` implements a single concrete BenchmarkRunner
class that calls OpenRouter's OpenAI-compatible API directly, with two
independent safety mechanisms to prevent free-tier rate-limit bans:

- **Proactive throttling** — a fixed minimum delay before every API call
- **Reactive backoff + jitter** — a growing randomized delay after a failure

Plus disk-based JSON caching per prompt and resume logic so an interrupted
run never repeats an already-completed API call.

The model is set to `openrouter/free`, OpenRouter's own router that
automatically selects a currently-available free model — this avoids
depending on one named free model, which can be discontinued without
notice (encountered directly during Week 2 development).

**Provider plan:** OpenRouter is used in Week 2 for pipeline development
and verification. Gemini, Cerebras, and Groq are the three fixed, named
providers used in Week 4's real benchmarking data collection.

## Testing

`03_Code/tests/test_benchmark_runner.py` verifies the pipeline using an
injected fake client — 9/9 tests passing, zero real API calls required.

Running this file directly (VS Code Run button) automatically saves a
JSON summary of that week's test results to its own dated folder:
`05_Logs_Results/tests_logs/Week_N/test_results.json`. Each new week's
test file only needs its `WEEK_LABEL` constant updated.

Week 3 introduced no automation code, so `03_Code/validate_prompts.py`
serves as that week's equivalent verification step, confirming the
curated dataset's structural integrity instead.

## Environment Setup

This project uses an isolated virtual environment so its dependencies never
conflict with anything else on your machine, and can be recreated exactly
by anyone.

```bash
# In VS Code: Command Palette -> "Python: Create Environment" -> Venv
pip install -r requirements.txt
```

Required environment variables (create a `.env` file in the project root,
never committed to GitHub):

```
OPENROUTER_API_KEY=your_real_key_here
```

(Gemini, Cerebras, and Groq API keys are added in Week 4 when their
runners are built.)

## Status

| Week | Focus | Status |
|------|-------|--------|
| 1 | Framework Initialization & Rubric Design | ✅ Complete |
| 2 | Automation Pipeline, Live OpenRouter Verification & Reproducible Environment | ✅ Complete |
| 3 | Dataset Curation & Repository Restructuring (OpenAI --> Cerebras) | ✅ Complete |
| 4 | Multi-Provider Decoupling & Real Data Collection (Gemini, Cerebras, Groq) | ⏳ Not started |
| 5 | Continued Data Collection & Quantitative Analysis | ⏳ Not started |
| 6 | Error Analysis & Failure Mode Taxonomy | ⏳ Not started |
| 7 | Discussion, Hypothesis Testing & Synthesis | ⏳ Not started |
| 8 | Final Report Compilation & Deliverable Packaging | ⏳ Not started |

See `02_Reports/` for detailed week-by-week progress reports.