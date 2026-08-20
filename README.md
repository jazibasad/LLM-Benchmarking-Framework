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
├── 05_Logs_Results/     # Per-model JSON response logs + weekly test result logs
│   ├── Gemini_Logs/
│   ├── Mistral_Logs/
│   ├── Groq_Logs/
│   ├── OpenRouter_Logs/
│   ├── environment_specs/
│   └── tests_logs/
│       └── Week_2/
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
scoring criteria. Each prompt also carries its own specific `evaluation_criteria`,
scored individually rather than against a single generic rubric.

## Prompt Dataset (Week 3 — Revised)

`04_Datasets/prompts.json` contains **220 curated evaluation prompts across all
10 categories specified in the project proposal (Section 3)**, 22 prompts per
category:

| # | Category | Count |
|---|---|---|
| 1 | Knowledge Retrieval | 22 |
| 2 | Multi-step Reasoning | 22 |
| 3 | Instruction Following | 22 |
| 4 | Hallucination Stress Test | 22 |
| 5 | Coding & System Architecture | 22 |
| 6 | Ambiguity Handling | 22 |
| 7 | Long-context Retention | 22 |
| 8 | Data Transformation | 22 |
| 9 | Summarization | 22 |
| 10 | Multilingual Tasks | 22 |
| | **Total** | **220** |

Each entry has: `id`, `category`, `difficulty` (Easy/Medium/Hard), `prompt`,
`evaluation_criteria`, `max_score`, `stochastic_sample`, and `repeat_count`.

### Stochastic Reliability Sample

Per the proposal's Reliability requirement, 20 prompts (2 per category) are
marked `"stochastic_sample": true` with `"repeat_count": 5` — selected for
genuinely open-ended/interpretive characteristics where model output would
plausibly vary between runs. These 20 are run 5x each per provider (300 total
calls per provider instead of 220) to compute mean, median, standard
deviation, and coefficient of variation. The remaining 200 prompts run once.

Run `python 03_Code/validate_prompts.py` to verify the dataset's structural
integrity, including confirming all 10 categories are present, none fall
below the proposal's 20-prompt minimum, and the stochastic sample is
correctly marked (20 total, 2 per category).

## Data Collection Providers (Week 4)

Three decoupled provider runners, each writing to its own partitioned log
folder, each supporting the repeat-count logic for the stochastic sample:

- **Gemini** (`run_gemini_benchmark.py` → `Gemini_Logs/`) — `gemini-3.5-flash-lite`
- **Mistral** (`run_mistral_benchmark.py` → `Mistral_Logs/`) — `open-mistral-nemo`
- **Groq** (`run_groq_benchmark.py` → `Groq_Logs/`) — `llama-3.3-70b-versatile`

All three record `temperature` (fixed at 0.7, a controlled variable) and
`latency_seconds` (measured per call) in every saved response, per the
proposal's Variables requirement (Section 2).

**Provider history:** OpenAI → Cerebras (original Week 3) → Mistral (Week 4),
after Cerebras introduced a mandatory payment-method requirement that could
not be authorized. Gemini's original model (`gemini-2.5-flash`) was found
retired mid-development and replaced with `gemini-3.5-flash`, which was
later found to enforce a strict 20 requests/day cap and replaced again with
`gemini-3.5-flash-lite`.

## Automation Pipeline

Every runner shares the same safety mechanisms:
- **Proactive throttling** — provider-specific delay before every API call
- **Reactive backoff + jitter** — growing randomized delay after a transient failure
- **Persistent-error detection** — stops the run early on systemic failures
  (quota exhaustion, retired models, billing issues)
- **Atomic cache writes** — no corrupted files from a hard interrupt
- **Resume-by-cache-file** — re-running a stopped script picks up exactly
  where it left off, including partially-completed repeat-runs

## Controlled Environment Specification

Run `python 03_Code/capture_environment.py` to record a timestamped snapshot
of the OS, CPU, RAM, Python version, SDK versions, network status, and model
configuration — satisfying the proposal's Controlled Environment requirement
(Section 2). Output saved to `05_Logs_Results/environment_specs/`.

## Scoring Methodology (Week 5)

Automated scores are derived from an LLM judge (`llm_judge_scorer.py`)
evaluating each response against the SPECIFIC evaluation criteria written
for that exact prompt. Each criterion is scored 0-5 against explicit,
standardized anchors (0 = Complete Failure, 3 = Minimum Success Threshold,
5 = Complete Success), and a Success/Failure outcome is computed in code
using this documented threshold — never trusted from the judge's own
labeling. A prompt's overall outcome is "Success" only if every one of its
criteria individually succeeded.

## Results Consolidation

- `build_results_tables.py` — consolidates raw per-prompt/per-run logs into
  one structured JSON per provider, correctly grouping the 5 runs of each
  stochastic-sample prompt together.
- `build_results_docx.py` — human-readable Word tables per provider, showing
  all 5 response variants for stochastic-sample prompts (highlighted),
  single responses for standard prompts.

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

## Status

| Week | Focus | Status |
|------|-------|--------|
| 1 | Framework Initialization & Rubric Design | ✅ Complete |
| 2 | Automation Pipeline, Live OpenRouter Verification & Reproducible Environment | ✅ Complete |
| 3 | Dataset Curation (Revised — all 10 proposal categories) | ✅ Complete |
| 4 | Multi-Provider Decoupling, Repeat Logic, Temperature/Latency, Environment Spec | ✅ Code complete, real execution pending |
| 5 | Automated Scoring with Success/Failure Anchors | ✅ Code complete, awaiting Week 4 data |
| 6 | Human Evaluation & Score Validation | ⏳ Not started |
| 7 | Statistical Analysis | ⏳ Not started |
| 8 | Final Report Compilation | ⏳ Not started |

See `02_Reports/` for detailed week-by-week progress reports.
