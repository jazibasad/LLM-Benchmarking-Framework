"""
run_groq_benchmark.py

Real Groq API runner for Week 4 data collection, reading the full
220-prompt, 10-category curated dataset. This is one of three decoupled
provider runners (alongside run_gemini_benchmark.py and
run_mistral_benchmark.py), each writing to its own partitioned log directory.

STOCHASTIC RELIABILITY SAMPLE: 20 of the 220 prompts are marked
"stochastic_sample": true with "repeat_count": 5 in prompts.json. Those 20
prompts are called 5 times each, producing 5 separately-saved response files
per prompt (groq_P037_run1.json ... groq_P037_run5.json), so run-to-run
variation can be measured later (mean, median, std dev, coefficient of
variation). The remaining 200 prompts are called once. A full run makes 300
total API calls (200 x 1 + 20 x 5), not 220.

MODEL: llama-3.3-70b-versatile.

SAFETY MECHANISMS: proactive throttling, reactive backoff+jitter,
persistent-error detection with early stop, atomic cache writes, and
resume-by-cache-file (works per-run for repeated prompts).

HOW TO RUN FOR REAL (VS Code):
  1. Create a .env file in the project root:  GROQ_API_KEY=your_real_key_here
  2. pip install -r requirements.txt   (one-time, in the VS Code terminal)
  3. Open this file -> click the Run button (top-right)
"""

import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("groq_runner")

MODEL_NAME = "openai/gpt-oss-120b"
# NOTE: llama-3.3-70b-versatile (the original model here) was officially
# deprecated by Groq on June 17, 2026, and now returns a 404 model_not_found
# error. Confirmed via Groq's own deprecation page, which recommends
# migrating to openai/gpt-oss-120b or qwen/qwen3.6-27b. This is the same
# failure pattern already hit twice before in this project (OpenRouter in
# Week 2, Gemini in Week 4) - free-tier model catalogs churn regularly. If
# this model also 404s in the future, check console.groq.com/docs/models
# or console.groq.com/docs/deprecations for the current recommended model.
TEMPERATURE = 0.7  # controlled independent variable (proposal Section 2) - fixed
                    # across all three providers for fair comparison
RESET_INFO = "midnight UTC (Groq's daily quota reset)"

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "05_Logs_Results", "Groq_Logs")
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "04_Datasets", "prompts.json")

MAX_RETRIES = 5
BASE_DELAY = 1.0
MAX_DELAY = 60.0
MIN_INTERVAL = 2.0  # seconds - minimum gap enforced before every call

PERSISTENT_ERROR_KEYWORDS = [
    "quota", "resource_exhausted", "daily", "rate limit exceeded", "429", "exceeded",
    "not_found", "no longer available", "model not found", "404",
    "payment", "402", "billing", "tokens per day",
]


def _looks_like_persistent_error(exc) -> bool:
    text = str(exc).lower()
    return any(keyword in text for keyword in PERSISTENT_ERROR_KEYWORDS)


class GroqBenchmarkRunner:
    """Concrete runner: calls Groq directly in call_model(). No abstract base class."""

    def __init__(self, api_key: str, log_dir: str = LOG_DIR, prompts_path: str = PROMPTS_PATH,
                 max_retries: int = MAX_RETRIES, base_delay: float = BASE_DELAY,
                 max_delay: float = MAX_DELAY, min_interval: float = MIN_INTERVAL,
                 client=None):
        self.log_dir = log_dir
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.min_interval = min_interval
        self._last_call_time = 0.0

        os.makedirs(self.log_dir, exist_ok=True)

        self.prompts = self._load_prompts(prompts_path)

        if client is not None:
            self.client = client
        else:
            from groq import Groq
            self.client = Groq(api_key=api_key)

    def _load_prompts(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _cache_path(self, prompt_id: str, run_number: int = None) -> str:
        if run_number is None:
            return os.path.join(self.log_dir, f"groq_{prompt_id}.json")
        return os.path.join(self.log_dir, f"groq_{prompt_id}_run{run_number}.json")

    def _already_done(self, prompt_id: str, run_number: int = None) -> bool:
        return os.path.exists(self._cache_path(prompt_id, run_number))

    def _write_cache_atomic(self, prompt_id: str, record: dict, run_number: int = None):
        final_path = self._cache_path(prompt_id, run_number)
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

    def call_model(self, prompt_text: str) -> dict:
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=TEMPERATURE,
        )
        latency_seconds = round(time.time() - start_time, 4)
        return {
            "response_text": response.choices[0].message.content,
            "raw": {"model": MODEL_NAME},
            "temperature": TEMPERATURE,
            "latency_seconds": latency_seconds,
        }

    def _call_with_backoff(self, prompt_text: str) -> dict:
        attempt = 0
        last_exception = None
        while attempt <= self.max_retries:
            self._throttle()
            try:
                return self.call_model(prompt_text)
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

    def _score_one_call(self, item, run_number, completed, total, skipped):
        prompt_id = item["id"]

        if self._already_done(prompt_id, run_number):
            return "skipped"

        run_label = f" (run {run_number}/{item.get('repeat_count', 1)})" if run_number else ""
        logger.info("Running %s%s (%s, %s)...", prompt_id, run_label, item["category"], item.get("difficulty", "n/a"))

        try:
            result = self._call_with_backoff(item["prompt"])
        except RuntimeError as exc:
            systemic = _looks_like_persistent_error(exc) or completed == 0
            if systemic:
                remaining = total - completed - skipped
                logger.error(
                    "STOPPING RUN EARLY on %s%s. %d completed, %d remaining. "
                    "If quota-related, resets at %s. Re-run to resume.",
                    prompt_id, run_label, completed, remaining, RESET_INFO,
                )
                return "stop"
            logger.error("Giving up on %s%s (isolated error): %s", prompt_id, run_label, exc)
            return "failed"

        record = {
            "id": prompt_id,
            "category": item["category"],
            "difficulty": item.get("difficulty"),
            "prompt": item["prompt"],
            "evaluation_criteria": item.get("evaluation_criteria"),
            "max_score": item.get("max_score"),
            "model": "groq",
            "response_text": result.get("response_text"),
            "raw": result.get("raw"),
            "temperature": result.get("temperature"),
            "latency_seconds": result.get("latency_seconds"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_number": run_number if run_number else 1,
            "repeat_count": item.get("repeat_count", 1),
        }
        self._write_cache_atomic(prompt_id, record, run_number)
        return "completed"

    def run(self):
        total_calls = sum(item.get("repeat_count", 1) for item in self.prompts)
        completed = skipped = failed = 0
        stopped_early = False

        for item in self.prompts:
            repeat_count = item.get("repeat_count", 1)
            run_numbers = range(1, repeat_count + 1) if repeat_count > 1 else [None]

            for run_number in run_numbers:
                outcome = self._score_one_call(item, run_number, completed, total_calls, skipped)
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

        remaining = total_calls - completed - skipped
        logger.info(
            "Run %s: %d completed, %d skipped (cached), %d failed, %d remaining, %d total calls.",
            "stopped early" if stopped_early else "complete",
            completed, skipped, failed, remaining, total_calls,
        )
        return {
            "completed": completed, "skipped": skipped, "failed": failed,
            "remaining": remaining, "total_calls": total_calls, "stopped_early": stopped_early,
        }


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("FAILURE: GROQ_API_KEY not set. Add it to a .env file in the project root.")
        sys.exit(1)

    runner = GroqBenchmarkRunner(api_key=api_key)
    stats = runner.run()
    print(f"\nDone. {stats}")
    if stats["stopped_early"]:
        print(f"Re-run this script after {RESET_INFO} to continue.")


if __name__ == "__main__":
    main()