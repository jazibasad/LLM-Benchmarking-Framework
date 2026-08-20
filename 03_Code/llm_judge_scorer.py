"""
llm_judge_scorer.py

Scores every collected response against its prompt's specific evaluation
criteria, using a free LLM judge via OpenRouter. This is Week 5's core
deliverable: turning collected prompt/response data into scored records.

PER-RUN SCORING: reads from 05_Logs_Results/Combined_Results/<provider>_
prompts_and_results.json (built by Week 4's build_results_tables.py), where
each prompt entry contains a "runs" array - 1 entry for standard prompts,
5 entries for the stochastic reliability sample. EACH RUN IS SCORED
SEPARATELY, not just the first one. This is deliberate: scoring every run
of the 20 stochastic-sample prompts independently lets you measure not just
response variance but SCORE variance across repeated calls - directly
feeding the mean/median/standard deviation/coefficient of variation
statistics required by the project proposal's Reliability section.

SUCCESS/FAILURE SCORING (per supervisor requirement, Aug 18): every
criterion is scored against explicit, standardized anchors (0 = Complete
Failure through 5 = Complete Success), and a Success/Failure outcome is
computed IN CODE using a documented threshold (score >= 3 = Success) -
never trusted from the judge's own labeling, exactly like total_score is
computed by summing rather than trusting the judge's arithmetic. A prompt's
overall outcome is "Success" only if every one of its criteria individually
succeeded.

OUTPUT: one JSON file per prompt per run, per provider:
    Scored_Results/Gemini_Scores/gemini_P001_scored.json          (single-run)
    Scored_Results/Gemini_Scores/gemini_P037_run1_scored.json     (multi-run)
    ... through run5 ...

Each scored file contains: id, category, difficulty, prompt, response_text,
run_number, temperature, latency_seconds, model, criterion_scores (each with
name, score, justification, outcome), total_score, max_score,
success_threshold, overall_outcome, judge_model, judge_timestamp_utc.

SAFETY MECHANISMS: proactive throttling, reactive backoff+jitter,
persistent-error detection with early stop, atomic cache writes, and
resume-by-cache-file (per individual run).

HOW TO RUN FOR REAL (VS Code):
  1. Make sure OPENROUTER_API_KEY is in your .env
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
MIN_INTERVAL = 3.5  # seconds - respects OpenRouter's confirmed 20 requests/minute cap

PERSISTENT_ERROR_KEYWORDS = [
    "quota", "resource_exhausted", "daily", "rate limit exceeded", "429", "exceeded",
    "not_found", "not found", "no longer available", "model not found", "404",
    "payment", "402", "billing", "tokens per day",
    "401", "unauthorized", "user not found", "invalid api key", "authentication",
]

# Explicit, standardized score anchors - what each 0-5 value means, and the
# exact threshold separating "success" from "failure". This is what makes
# automated scoring defensible: a score isn't just a number, it's derived
# from a documented, consistent definition applied identically everywhere.
SCORE_ANCHORS = """0 = Complete Failure: the response does not attempt this criterion at all, or is entirely wrong/absent.
1 = Major Failure: the response attempts this criterion but fails on most key aspects.
2 = Partial Failure: the response meets some minimal aspects but falls below an acceptable standard.
3 = Minimum Success: the response adequately meets the core requirement, with only minor gaps. (This is the SUCCESS THRESHOLD - any score of 3 or higher counts as a Success for this criterion.)
4 = Strong Success: the response meets the requirement well, with only trivial gaps.
5 = Complete Success: the response fully and excellently meets the requirement."""

SUCCESS_THRESHOLD = 3  # scores >= this value are classified as "Success", computed in code below


def _looks_like_persistent_error(exc) -> bool:
    text = str(exc).lower()
    return any(keyword in text for keyword in PERSISTENT_ERROR_KEYWORDS)


def parse_criteria(evaluation_criteria: str):
    """Parses per-prompt evaluation_criteria text into a structured list of
    {"name": ..., "description": ...} dicts, one per criterion line."""
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
    criteria_lines = "\n".join(f"- {c['name']} (0-5): {c['description']}" for c in criteria)
    return f"""You are an expert evaluator scoring an AI model's response to a prompt.

ORIGINAL PROMPT:
{prompt_text}

MODEL'S RESPONSE:
{response_text}

Score the response against EACH of the following criteria, using this EXACT scoring scale for every criterion:
{SCORE_ANCHORS}

Criteria to score:
{criteria_lines}

For each criterion, give a whole-number score from 0 to 5 and a one-sentence justification explaining specifically why, based on what the response did or did not do, referencing the scale definitions above.

Respond with ONLY valid JSON, no other text, in this exact structure:
{{
  "criterion_scores": [
    {{"name": "<criterion name exactly as given above>", "score": <integer 0-5>, "justification": "<one sentence>"}}
  ]
}}"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {text[:200]}")
    return json.loads(match.group())


class LLMJudgeScorer:
    """Scores one provider's collected responses, one run at a time. No abstract base class."""

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

        self.prompt_entries = self._load_entries()

        if client is not None:
            self.client = client
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    def _load_entries(self):
        with open(self.input_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _score_path(self, prompt_id: str, run_number: int, is_multi_run: bool) -> str:
        if is_multi_run:
            return os.path.join(self.score_dir, f"{self.file_prefix}{prompt_id}_run{run_number}_scored.json")
        return os.path.join(self.score_dir, f"{self.file_prefix}{prompt_id}_scored.json")

    def _already_scored(self, prompt_id: str, run_number: int, is_multi_run: bool) -> bool:
        return os.path.exists(self._score_path(prompt_id, run_number, is_multi_run))

    def _write_score_atomic(self, prompt_id: str, run_number: int, is_multi_run: bool, record: dict):
        final_path = self._score_path(prompt_id, run_number, is_multi_run)
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
        return _extract_json(response.choices[0].message.content)

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
                logger.warning("Attempt %d/%d failed (%s). Retrying in %.2fs.",
                                attempt + 1, self.max_retries + 1, exc, sleep_time)
                time.sleep(sleep_time)
                attempt += 1
        raise RuntimeError(f"All {self.max_retries + 1} attempts failed. Last error: {last_exception}")

    def _score_one_run(self, entry, run, completed, total, skipped):
        prompt_id = entry["id"]
        run_number = run.get("run_number", 1)
        is_multi_run = len(entry.get("runs", [])) > 1

        if self._already_scored(prompt_id, run_number, is_multi_run):
            return "skipped"

        run_label = f" (run {run_number})" if is_multi_run else ""
        logger.info("Scoring %s%s (%s, %s)...", prompt_id, run_label, entry["category"], entry.get("difficulty", "n/a"))

        criteria = parse_criteria(entry.get("evaluation_criteria", ""))
        response_text = run.get("response_text", "")

        try:
            judge_result = self._call_with_backoff(entry["prompt"], response_text, criteria)
        except RuntimeError as exc:
            systemic = _looks_like_persistent_error(exc) or completed == 0
            if systemic:
                remaining = total - completed - skipped
                logger.error("STOPPING SCORING EARLY on %s%s. %d scored, %d remaining. Error: %s",
                              prompt_id, run_label, completed, remaining, exc)
                return "stop"
            logger.error("Giving up scoring %s%s (isolated error): %s", prompt_id, run_label, exc)
            return "failed"

        criterion_scores = judge_result.get("criterion_scores", [])
        for c in criterion_scores:
            score_val = int(c.get("score", 0))
            c["outcome"] = "Success" if score_val >= SUCCESS_THRESHOLD else "Failure"

        total_score = sum(int(c.get("score", 0)) for c in criterion_scores)
        all_succeeded = all(c["outcome"] == "Success" for c in criterion_scores) if criterion_scores else False
        overall_outcome = "Success" if all_succeeded else "Failure"

        scored_record = {
            "id": prompt_id,
            "category": entry["category"],
            "difficulty": entry.get("difficulty"),
            "prompt": entry["prompt"],
            "response_text": response_text,
            "run_number": run_number,
            "temperature": run.get("temperature"),
            "latency_seconds": run.get("latency_seconds"),
            "model": entry.get("model"),
            "criterion_scores": criterion_scores,
            "total_score": total_score,
            "max_score": entry.get("max_score"),
            "success_threshold": SUCCESS_THRESHOLD,
            "overall_outcome": overall_outcome,
            "judge_model": JUDGE_MODEL,
            "judge_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._write_score_atomic(prompt_id, run_number, is_multi_run, scored_record)
        return "completed"

    def run(self):
        total_runs = sum(len(entry.get("runs", [])) for entry in self.prompt_entries)
        completed = skipped = failed = 0
        stopped_early = False

        for entry in self.prompt_entries:
            for run in entry.get("runs", []):
                outcome = self._score_one_run(entry, run, completed, total_runs, skipped)
                if outcome == "skipped":
                    skipped += 1
                elif outcome == "completed":
                    completed += 1
                elif outcome == "failed":
                    failed += 1
                elif outcome == "stop":
                    stopped_early = True
                    break
            if stopped_early:
                break

        remaining = total_runs - completed - skipped
        logger.info("Scoring %s: %d scored, %d skipped (cached), %d failed, %d remaining, %d total runs.",
                     "stopped early" if stopped_early else "complete",
                     completed, skipped, failed, remaining, total_runs)
        return {"completed": completed, "skipped": skipped, "failed": failed,
                "remaining": remaining, "total_runs": total_runs, "stopped_early": stopped_early}


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
            print(f"SKIPPING {name}: {config['input_file']} not found. Run build_results_tables.py first.")
            continue
        print(f"\n=== Scoring {name} ===")
        scorer = LLMJudgeScorer(api_key=api_key, input_file=config["input_file"],
                                  score_dir=config["score_dir"], file_prefix=config["file_prefix"])
        stats = scorer.run()
        print(f"[{name}] {stats}")


if __name__ == "__main__":
    main()