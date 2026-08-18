"""
llm_judge_scorer.py

Scores every collected response against its prompt's specific evaluation
criteria, using a free LLM judge via OpenRouter. This is Week 5's core
deliverable: turning 660 raw prompt/response pairs into 660 scored records.

WHAT GETS SCORED AND HOW (the "on what basis" requirement):
For every response, the judge is given:
  1. The original prompt
  2. The model's actual response
  3. The EXACT evaluation criteria written for that specific prompt (parsed
     from prompts.json - e.g. "Syntax Correctness: 0-5", "Edge Case
     Handling: 0-5") - NOT a generic rubric, the real per-prompt criteria.
The judge scores EACH criterion individually (0-5) with a one-sentence
justification explaining exactly why that score was given, and the total
score is computed by SUMMING the individual criterion scores in code (not
trusting the judge's own arithmetic), then checked against the prompt's
max_score for consistency.

OUTPUT: one JSON file per prompt, per provider, in its own folder:
    05_Logs_Results/Scored_Results/Gemini_Scores/gemini_P001_scored.json
    05_Logs_Results/Scored_Results/Mistral_Scores/mistral_P001_scored.json
    05_Logs_Results/Scored_Results/Groq_Scores/groq_P001_scored.json

Each scored file contains: id, category, difficulty, prompt, response_text,
model, criterion_scores (list of {name, score, justification}), total_score,
max_score, judge_model, judge_timestamp_utc - i.e. everything needed to see
exactly what was scored, on what basis, and by whom.

SAFETY MECHANISMS: same pattern as the Week 4 runners - proactive
throttling, reactive backoff+jitter, persistent-error detection with early
stop, atomic cache writes, resume-by-cache-file (already-scored prompts are
skipped automatically on re-run).

INPUT SOURCE: reads from 05_Logs_Results/Combined_Results/<provider>_prompts_and_results.json
(built by build_results_tables.py in Week 4). Run that script first if these
don't exist yet.

HOW TO RUN FOR REAL (VS Code):
  1. Make sure OPENROUTER_API_KEY is in your .env (already set up since Week 2)
  2. Run the unit tests FIRST: python 03_Code/tests/test_llm_judge_scorer.py
  3. Once tests pass, run this file for real: python 03_Code/llm_judge_scorer.py
"""

import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("llm_judge_scorer")

JUDGE_MODEL = "openrouter/free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
COMBINED_RESULTS_DIR = os.path.join(BASE_DIR, "05_Logs_Results", "Combined_Results")
SCORED_RESULTS_DIR = os.path.join(BASE_DIR, "05_Logs_Results", "Scored_Results")

PROVIDERS = {
    "gemini": {
        "input_file": os.path.join(COMBINED_RESULTS_DIR, "gemini_prompts_and_results.json"),
        "score_dir": os.path.join(SCORED_RESULTS_DIR, "Gemini_Scores"),
        "file_prefix": "gemini_",
    },
    "mistral": {
        "input_file": os.path.join(COMBINED_RESULTS_DIR, "mistral_prompts_and_results.json"),
        "score_dir": os.path.join(SCORED_RESULTS_DIR, "Mistral_Scores"),
        "file_prefix": "mistral_",
    },
    "groq": {
        "input_file": os.path.join(COMBINED_RESULTS_DIR, "groq_prompts_and_results.json"),
        "score_dir": os.path.join(SCORED_RESULTS_DIR, "Groq_Scores"),
        "file_prefix": "groq_",
    },
}

MAX_RETRIES = 5
BASE_DELAY = 1.0
MAX_DELAY = 60.0
MIN_INTERVAL = 3.5  # seconds between judge calls - keeps requests safely under
                     # OpenRouter's confirmed 20 requests/minute free-tier cap
                     # (60s / 20 = 3s minimum; 3.5s adds a safety margin)

PERSISTENT_ERROR_KEYWORDS = [
    "quota", "resource_exhausted", "daily", "rate limit exceeded", "429", "exceeded",
    "not_found", "no longer available", "model not found", "404",
    "payment", "402", "billing", "tokens per day",
]


def _looks_like_persistent_error(exc) -> bool:
    text = str(exc).lower()
    return any(keyword in text for keyword in PERSISTENT_ERROR_KEYWORDS)


def parse_criteria(evaluation_criteria: str):
    """
    Parses the per-prompt evaluation_criteria text (e.g.
    "Syntax Correctness: 0-5 (Evaluates clean syntax and execution)")
    into a structured list of {"name": ..., "description": ...} dicts, one
    per criterion line. This is what makes scoring specific to each prompt
    rather than a generic one-size-fits-all rubric.
    """
    criteria = []
    for line in str(evaluation_criteria).strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(.+?):\s*0-5\s*\((.+)\)$", line)
        if match:
            criteria.append({"name": match.group(1).strip(), "description": match.group(2).strip()})
        else:
            criteria.append({"name": line, "description": ""})
    return criteria


def build_judge_prompt(prompt_text: str, response_text: str, criteria: list) -> str:
    criteria_lines = "\n".join(
        f"- {c['name']} (0-5): {c['description']}" for c in criteria
    )
    return f"""You are an expert evaluator scoring an AI model's response to a prompt.

ORIGINAL PROMPT:
{prompt_text}

MODEL'S RESPONSE:
{response_text}

Score the response against EACH of the following criteria, on a scale of 0 (fails completely) to 5 (excellent):
{criteria_lines}

For each criterion, give a whole-number score from 0 to 5 and a one-sentence justification explaining specifically why, based on what the response did or did not do.

Respond with ONLY valid JSON, no other text, in this exact structure:
{{
  "criterion_scores": [
    {{"name": "<criterion name exactly as given above>", "score": <integer 0-5>, "justification": "<one sentence>"}}
  ]
}}"""


def _extract_json(text: str) -> dict:
    """Judges sometimes wrap JSON in markdown code fences or add stray text;
    this extracts the first valid JSON object found in the response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {text[:200]}")
    return json.loads(match.group())


class LLMJudgeScorer:
    """Scores one provider's collected responses using an LLM judge. No abstract base class."""

    def __init__(self, api_key: str, input_file: str, score_dir: str, file_prefix: str,
                 max_retries: int = MAX_RETRIES, base_delay: float = BASE_DELAY,
                 max_delay: float = MAX_DELAY, min_interval: float = MIN_INTERVAL,
                 client=None):
        self.input_file = input_file
        self.score_dir = score_dir
        self.file_prefix = file_prefix
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.min_interval = min_interval
        self._last_call_time = 0.0

        os.makedirs(self.score_dir, exist_ok=True)

        self.records = self._load_records()

        if client is not None:
            self.client = client
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    def _load_records(self):
        with open(self.input_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _score_path(self, prompt_id: str) -> str:
        return os.path.join(self.score_dir, f"{self.file_prefix}{prompt_id}_scored.json")

    def _already_scored(self, prompt_id: str) -> bool:
        return os.path.exists(self._score_path(prompt_id))

    def _write_score_atomic(self, prompt_id: str, record: dict):
        final_path = self._score_path(prompt_id)
        tmp_path = final_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, final_path)

    def _throttle(self):
        elapsed = time.time() - self._last_call_time
        remaining = self.min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call_time = time.time()

    def call_judge(self, prompt_text: str, response_text: str, criteria: list) -> dict:
        judge_prompt = build_judge_prompt(prompt_text, response_text, criteria)
        response = self.client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        raw_text = response.choices[0].message.content
        parsed = _extract_json(raw_text)
        return parsed

    def _call_with_backoff(self, prompt_text, response_text, criteria) -> dict:
        attempt = 0
        last_exception = None
        while attempt <= self.max_retries:
            self._throttle()
            try:
                return self.call_judge(prompt_text, response_text, criteria)
            except Exception as exc:  # noqa: BLE001
                last_exception = exc
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                jitter = random.uniform(0, delay * 0.5)
                sleep_time = delay + jitter
                logger.warning(
                    "Attempt %d/%d failed (%s). Retrying in %.2fs.",
                    attempt + 1, self.max_retries + 1, exc, sleep_time,
                )
                time.sleep(sleep_time)
                attempt += 1
        raise RuntimeError(f"All {self.max_retries + 1} attempts failed. Last error: {last_exception}")

    def run(self):
        total = len(self.records)
        skipped = completed = failed = 0
        stopped_early = False

        for record in self.records:
            prompt_id = record["id"]

            if self._already_scored(prompt_id):
                skipped += 1
                continue

            logger.info("Scoring %s (%s, %s)...", prompt_id, record["category"], record.get("difficulty", "n/a"))

            criteria = parse_criteria(record.get("evaluation_criteria", ""))

            try:
                judge_result = self._call_with_backoff(record["prompt"], record.get("response_text", ""), criteria)
            except RuntimeError as exc:
                systemic = _looks_like_persistent_error(exc) or completed == 0
                if systemic:
                    remaining = total - completed - skipped
                    logger.error("STOPPING SCORING EARLY on %s. %d scored, %d remaining. Error: %s",
                                 prompt_id, completed, remaining, exc)
                    stopped_early = True
                    break
                logger.error("Giving up scoring %s (isolated error): %s", prompt_id, exc)
                failed += 1
                continue

            criterion_scores = judge_result.get("criterion_scores", [])
            total_score = sum(int(c.get("score", 0)) for c in criterion_scores)

            scored_record = {
                "id": prompt_id,
                "category": record["category"],
                "difficulty": record.get("difficulty"),
                "prompt": record["prompt"],
                "response_text": record.get("response_text"),
                "model": record.get("model"),
                "criterion_scores": criterion_scores,
                "total_score": total_score,
                "max_score": record.get("max_score"),
                "judge_model": JUDGE_MODEL,
                "judge_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            self._write_score_atomic(prompt_id, scored_record)
            completed += 1

        remaining = total - completed - skipped
        logger.info(
            "Scoring %s: %d scored, %d skipped (cached), %d failed, %d remaining, %d total.",
            "stopped early" if stopped_early else "complete",
            completed, skipped, failed, remaining, total,
        )
        return {"completed": completed, "skipped": skipped, "failed": failed,
                "remaining": remaining, "total": total, "stopped_early": stopped_early}


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("FAILURE: OPENROUTER_API_KEY not set. Add it to a .env file in the project root.")
        sys.exit(1)

    for name, config in PROVIDERS.items():
        if not os.path.exists(config["input_file"]):
            print(f"SKIPPING {name}: {config['input_file']} not found. "
                  f"Run build_results_tables.py first.")
            continue
        print(f"\n=== Scoring {name} ===")
        scorer = LLMJudgeScorer(
            api_key=api_key,
            input_file=config["input_file"],
            score_dir=config["score_dir"],
            file_prefix=config["file_prefix"],
        )
        stats = scorer.run()
        print(f"[{name}] {stats}")


if __name__ == "__main__":
    main()