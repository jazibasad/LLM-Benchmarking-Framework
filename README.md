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
│   ├── Combined_Results/       # Machine-readable consolidated JSON (per provider)
│   ├── Readable_Results/       # Human-readable Word tables (per provider)
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
10 categories specified in the project proposal (Section 3)**, 22 prompts per category:

Knowledge Retrieval · Multi-step Reasoning · Instruction Following ·
Hallucination Stress Test · Coding & System Architecture · Ambiguity Handling ·
Long-context Retention · Data Transformation · Summarization · Multilingual Tasks

Each entry has: `id`, `category`, `difficulty` (Easy/Medium/Hard), `prompt`,
`evaluation_criteria`, `max_score`, `stochastic_sample`, `repeat_count`.

**Stochastic reliability sample:** 20 prompts (2 per category) are marked
`stochastic_sample: true`, `repeat_count: 5` — run 5x each to compute mean,
median, standard deviation, and coefficient of variation on model
consistency, per the proposal's Reliability requirement. The remaining 200
run once. A full collection run makes **300 API calls per provider**, not 220.

Run `python 03_Code/validate_prompts.py` to verify dataset integrity.

## Controlled Environment Specification

`03_Code/capture_environment.py` records a timestamped snapshot of the OS,
CPU, RAM, Python version, installed SDK versions, network status, and fixed
model/temperature configuration — satisfying the proposal's Controlled
Environment requirement (Section 2). Run once per collection session, saved
to `05_Logs_Results/environment_specs/`.

## Data Collection Providers (Week 4)

Three decoupled provider runners, each writing to its own partitioned log
folder:

- **Gemini** (`run_gemini_benchmark.py` → `Gemini_Logs/`) — `gemini-3.5-flash-lite`
- **Mistral** (`run_mistral_benchmark.py` → `Mistral_Logs/`) — `open-mistral-nemo`
- **Groq** (`run_groq_benchmark.py` → `Groq_Logs/`) — `openai/gpt-oss-120b`

Every call records `temperature` (fixed at 0.7, a controlled variable across
all three providers) and `latency_seconds` (measured per call), per the
proposal's Variables requirement (Section 2).

**Provider history:** OpenAI → Cerebras (original Week 3) → Mistral (Week 4),
after Cerebras introduced a mandatory payment-method requirement that could
not be authorized. Gemini's model went `gemini-2.5-flash` (retired) →
`gemini-3.5-flash` (20/day cap discovered) → `gemini-3.5-flash-lite` (current).

## Automation Pipeline

Every runner shares the same safety mechanisms:
- **Proactive throttling** — provider-specific delay before every API call
- **Reactive backoff + jitter** — growing randomized delay after a transient failure
- **Persistent-error detection** — stops the run early on systemic failures
- **Atomic cache writes** — no corrupted files from a hard interrupt
- **Resume-by-cache-file** — operates per individual run, so an interrupted
  5x-repeat set resumes only the missing runs, not the whole prompt

## Scoring Methodology (Week 5)

`llm_judge_scorer.py` scores each response using an LLM judge against the
SPECIFIC evaluation criteria for that exact prompt. Every criterion is
scored 0-5 against explicit, standardized anchors (0 = Complete Failure,
3 = Minimum Success Threshold, 5 = Complete Success), and a Success/Failure
outcome is computed **in code** using this documented threshold — never
trusted from the judge's own labeling. A prompt's overall outcome is
"Success" only if every one of its criteria individually succeeded.

## Results Consolidation

- **`build_results_tables.py`** — machine-readable, one structured JSON per
  provider, correctly grouping all 5 runs of each stochastic-sample prompt
  together (with their individual temperature and latency values), and
  flagging any prompt with an incomplete run count.
- **`build_results_docx.py`** — human-readable Word tables per provider,
  with dedicated **Temperature** and **Latency (s)** columns aligned
  per-run, plus all response text, satisfying the requirement for separate,
  structured, human-readable prompt+answer files.

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

## Full Week 4 Run Order

```
1. python 03_Code/capture_environment.py
2. python 03_Code/run_gemini_benchmark.py    (any order, independent)
3. python 03_Code/run_mistral_benchmark.py
4. python 03_Code/run_groq_benchmark.py
5. python 03_Code/build_results_tables.py
6. python 03_Code/build_results_docx.py
```

## Status

| Week | Focus | Status |
|------|-------|--------|
| 1 | Framework Initialization & Rubric Design | ✅ Complete |
| 2 | Automation Pipeline, Live OpenRouter Verification & Reproducible Environment | ✅ Complete |
| 3 | Dataset Curation (Revised — all 10 proposal categories) | ✅ Complete |
| 4 | Multi-Provider Collection, Temperature/Latency, Environment Spec, Repeat Logic | ✅ Code complete, real execution pending |
| 5 | Automated Scoring with Success/Failure Anchors | ✅ Code complete, awaiting Week 4 data |
| 6 | Human Evaluation & Score Validation | ⏳ Not started |
| 7 | Statistical Analysis | ⏳ Not started |
| 8 | Final Report Compilation | ⏳ Not started |

**Open item:** "surveys" (mentioned by supervisor, Aug 12) — meaning unclear,
deferred until Week 6 when a human-evaluation form is actually needed.

See `02_Reports/` for detailed week-by-week progress reports.
