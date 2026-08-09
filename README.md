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
├── 05_Logs_Results/     # Per-model JSON response logs + weekly test result logs
│   ├── Gemini_Logs/
│   ├── Mistral_Logs/
│   ├── Groq_Logs/
│   ├── OpenRouter_Logs/
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

- `id` — P001 through P220
- `category` — Knowledge Retrieval, Multi-step Reasoning, Instruction Following,
  Hallucination Stress Test, or Coding & System Architecture
- `difficulty` — Easy, Medium, or Hard
- `prompt` — the prompt text itself
- `evaluation_criteria` — specific, per-prompt scoring criteria
- `max_score` — the maximum achievable score for that prompt

Run `python 03_Code/validate_prompts.py` to verify the dataset's structural
integrity at any time.

## Data Collection Providers (Week 4)

Real benchmarking data is collected across three providers, each with its own
decoupled runner script and partitioned log folder:

- **Gemini** (`run_gemini_benchmark.py` → `Gemini_Logs/`)
- **Mistral** (`run_mistral_benchmark.py` → `Mistral_Logs/`)
- **Groq** (`run_groq_benchmark.py` → `Groq_Logs/`)

**Provider history note:** OpenAI was originally swapped for Cerebras in Week 3.
During Week 4 development, Cerebras introduced a mandatory payment-method
requirement to activate API access, and available local payment methods could
not be authorized for international billing. Mistral AI was selected instead as
a provider with a genuinely permanent free tier requiring no payment method.

## Automation Pipeline (Week 2)

`03_Code/benchmark_runner.py` implements the original pipeline pattern, calling
OpenRouter's OpenAI-compatible API directly, with two independent safety
mechanisms to prevent free-tier rate-limit bans:

- **Proactive throttling** — a fixed minimum delay before every API call
- **Reactive backoff + jitter** — a growing randomized delay after a failure

This same pattern is reused in each Week 4 provider runner, extended with:
- **Persistent-error detection** — stops the run early (rather than retrying
  every remaining prompt uselessly) if a quota is exhausted or a model has been
  retired/is unavailable
- **Atomic cache writes** — prevents corrupted cache files from a hard interrupt
- **Resume-by-cache-file** — re-running a stopped script automatically picks up
  exactly where it left off

## Testing

Each provider runner has a matching test file using an injected fake client —
zero real API calls required to verify the automation logic. Running a test
file directly (VS Code Run button) saves a JSON summary of that week's results
to `05_Logs_Results/tests_logs/Week_N/`.

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
| 4 | Multi-Provider Decoupling & Real Data Collection (Gemini, Mistral, Groq) | ⏳ In progress |
| 5 | Continued Data Collection & Quantitative Analysis | ⏳ Not started |
| 6 | Error Analysis & Failure Mode Taxonomy | ⏳ Not started |
| 7 | Discussion, Hypothesis Testing & Synthesis | ⏳ Not started |
| 8 | Final Report Compilation & Deliverable Packaging | ⏳ Not started |

See `02_Reports/` for detailed week-by-week progress reports.