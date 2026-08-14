# LLM Benchmarking Framework

A reproducible framework for evaluating and comparing free-tier Large Language Models
(Gemini, Mistral, Groq) across five core capability dimensions.

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
├── 05_Logs_Results/     # Per-model JSON response logs + consolidated outputs
│   ├── Gemini_Logs/           # Raw per-prompt logs - 220/220 complete
│   ├── Mistral_Logs/          # Raw per-prompt logs - 220/220 complete
│   ├── Groq_Logs/              # Raw per-prompt logs - 220/220 complete
│   ├── OpenRouter_Logs/
│   ├── Combined_Results/       # Structured JSON tables, one per provider
│   ├── Readable_Results/       # Human-readable Word tables, one per provider
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
0–5 scoring criteria across: Knowledge Retrieval, Multi-step Reasoning, Instruction
Following, Hallucination Stress Test, and Coding Tasks.

## Prompt Dataset (Week 3)

`04_Datasets/prompts.json` contains 220 curated evaluation prompts, each with:
`id`, `category`, `difficulty` (Easy/Medium/Hard), `prompt`, `evaluation_criteria`,
and `max_score`. Run `python 03_Code/validate_prompts.py` to verify integrity.

## Data Collection Providers (Week 4)

Real benchmarking data collected across three providers, each with its own
decoupled runner script and partitioned log folder:

- **Gemini** (`run_gemini_benchmark.py` → `Gemini_Logs/`) — `gemini-3.5-flash` — 220/220 ✅
  — limited to 20 requests/day on the free tier; collection proceeds
  incrementally across multiple days using the resume mechanism
- **Mistral** (`run_mistral_benchmark.py` → `Mistral_Logs/`) — `open-mistral-nemo` — 220/220 ✅
- **Groq** (`run_groq_benchmark.py` → `Groq_Logs/`) — `llama-3.3-70b-versatile` — 220/220 ✅

**Provider history:** OpenAI → Cerebras (Week 3) → Mistral (Week 4), after Cerebras
introduced a mandatory payment-method requirement that could not be authorized.
Gemini's original model (`gemini-2.5-flash`) was also found retired mid-development
and replaced with `gemini-3.5-flash`. Both incidents motivated a persistent-error
detection mechanism applied to all three runners: a run stops immediately on a
quota/billing/model-availability error instead of retrying every remaining prompt
uselessly, and resumes exactly where it left off on re-run.

## Automation Pipeline

Every runner shares the same safety mechanisms:
- **Proactive throttling** — provider-specific delay before every call
- **Reactive backoff + jitter** — growing delay after a transient failure
- **Persistent-error detection** — stops the run early on systemic failures
- **Atomic cache writes** — no corrupted files from a hard interrupt
- **Resume-by-cache-file** — re-running picks up exactly where it stopped

Verified for real: Groq's run was interrupted at 189/220 and, on re-running the
unmodified script, correctly resumed and completed the remaining 31 without
repeating or losing data.

## Results Consolidation (Week 4)

Two scripts consolidate the raw per-prompt logs into submission-ready structured
formats, satisfying the requirement that prompts and their corresponding results
be submitted as separate files, with all raw data in structured form:

- **`build_results_tables.py`** — consolidates each provider's logs into one
  structured JSON array (`Combined_Results/`), validated for completeness
  against the full 220-prompt dataset
- **`build_results_docx.py`** — produces one human-readable Word table per
  provider (`Readable_Results/`), with prompt, response, evaluation criteria,
  and score together in one row, and proper line breaks for multi-line content
  (code responses, multi-part criteria)

## Testing

Weeks 2-3 verified pipeline logic using injected fake clients (zero real API
calls). Week 4 verification is through real execution against live provider
APIs, since the objective was genuine data collection. The two consolidation
scripts were verified with synthetic test data before running against real
collected data.

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
| 3 | Dataset Curation & Repository Restructuring | ✅ Complete |
| 4 | Multi-Provider Decoupling, Real Data Collection & Results Consolidation | ✅ Complete | 
| 5 | Statistical Aggregation & Comparative Scoring | ⏳ Not started |
| 6 | Error Analysis & Failure Mode Taxonomy | ⏳ Not started |
| 7 | Discussion, Hypothesis Testing & Synthesis | ⏳ Not started |
| 8 | Final Report Compilation & Deliverable Packaging | ⏳ Not started |

See `02_Reports/` for detailed week-by-week progress reports.