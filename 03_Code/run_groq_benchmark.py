"""
run_groq_benchmark.py

Real Groq API runner for Week 4 data collection, reading the full 220-prompt
curated dataset directly. This is one of three decoupled provider runners
(alongside run_gemini_benchmark.py and run_cerebras_benchmark.py), each
writing to its own partitioned log directory.

SAFETY MECHANISMS: see run_gemini_benchmark.py docstring for the full
explanation - proactive throttling, reactive backoff+jitter, quota-exhaustion
detection with early stop, atomic cache writes, and resume-by-cache-file.

GROQ QUOTA RESET: daily limits reset at midnight UTC. For llama-3.3-70b-
versatile, the free tier is approximately 1,000 requests/day but only around
100,000 tokens/day - the TOKEN cap is the more likely thing to be hit first
given this project's longer "Hard" difficulty prompts, not the request count.

MODEL NAME CAVEAT: Groq's free-tier model catalog also changes over time (a
previously-available model has been observed dropped from a real account's
list without notice). If MODEL_NAME below 404s, check console.groq.com for
the current model list and update it.

HOW TO RUN FOR REAL (VS Code):
  1. Create a .env file in the project root:  GROQ_API_KEY=your_real_key_here
  2. pip install -r requirements.txt   (one-time, in the VS Code terminal)
  3. Open this file -> click the Run button (top-right)
  4. If it stops early due to quota, re-run it after midnight UTC.
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

MODEL_NAME = "llama-3.3-70b-versatile"
RESET_INFO = "midnight UTC (Groq's daily quota reset)"

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "05_Logs_Results", "Groq_Logs")
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "04_Datasets", "prompts.json")

MAX_RETRIES = 5
BASE_DELAY = 1.0
MAX_DELAY = 60.0
MIN_INTERVAL = 2.0  # seconds - minimum gap enforced before every call

QUOTA_KEYWORDS = ["quota", "resource_exhausted", "daily", "rate limit exceeded", "429", "exceeded", "tokens per day"]


def _looks_like_quota_exhausted(exc) -> bool:
    text = str(exc).lower()
    return any(keyword in text for keyword in QUOTA_KEYWORDS)


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

        self.prompts_path = prompts_path
        self.prompts = self._load_prompts()

        if client is not None:
            self.client = client
        else:
            from groq import Groq
            self.client = Groq(api_key=api_key)

    def _load_prompts(self):
        with open(self.prompts_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _cache_path(self, prompt_id: str) -> str:
        return os.path.join(self.log_dir, f"groq_{prompt_id}.json")

    def _already_done(self, prompt_id: str) -> bool:
        return os.path.exists(self._cache_path(prompt_id))

    def _write_cache_atomic(self, prompt_id: str, record: dict):
        final_path = self._cache_path(prompt_id)
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
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt_text}],
        )
        return {
            "response_text": response.choices[0].message.content,
            "raw": {"model": MODEL_NAME},
        }

    def _call_with_backoff(self, prompt_text: str) -> dict:
        attempt = 0
        last_exception = None

        while attempt <= self.max_retries:
            self._throttle()
            try:
                return self.call_model(prompt_text)
            except Exception as exc:  # noqa: BLE001 - broad on purpose, provider errors vary
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

        raise RuntimeError(
            f"All {self.max_retries + 1} attempts failed. Last error: {last_exception}"
        )

    def run(self):
        total = len(self.prompts)
        skipped = completed = failed = 0
        stopped_early = False

        for item in self.prompts:
            prompt_id = item["id"]

            if self._already_done(prompt_id):
                skipped += 1
                continue

            logger.info("Running %s (%s, %s)...", prompt_id, item["category"], item.get("difficulty", "n/a"))

            try:
                result = self._call_with_backoff(item["prompt"])
            except RuntimeError as exc:
                if _looks_like_quota_exhausted(exc):
                    remaining = total - completed - skipped
                    logger.error(
                        "Quota exhaustion detected on %s after exhausting all retries: %s",
                        prompt_id, exc,
                    )
                    logger.error(
                        "STOPPING RUN EARLY. %d completed, %d remaining. "
                        "This provider resets at %s. Re-run this script after that "
                        "time - already-completed prompts are cached and will be "
                        "skipped automatically, so it will continue from %s.",
                        completed, remaining, RESET_INFO, prompt_id,
                    )
                    stopped_early = True
                    break
                logger.error("Giving up on %s (non-quota error): %s", prompt_id, exc)
                failed += 1
                continue

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
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            self._write_cache_atomic(prompt_id, record)
            completed += 1

        remaining = total - completed - skipped
        logger.info(
            "Run %s: %d completed, %d skipped (cached), %d failed, %d remaining, %d total.",
            "stopped early due to quota" if stopped_early else "complete",
            completed, skipped, failed, remaining, total,
        )
        return {
            "completed": completed, "skipped": skipped, "failed": failed,
            "remaining": remaining, "total": total, "stopped_early": stopped_early,
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