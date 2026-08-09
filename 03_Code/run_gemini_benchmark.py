"""
run_gemini_benchmark.py

Real Gemini API runner for Week 4 data collection, reading the full 220-prompt
curated dataset directly. This is one of three decoupled provider runners
(alongside run_cerebras_benchmark.py and run_groq_benchmark.py), each writing
to its own partitioned log directory.

SAFETY MECHANISMS:
  1. PROACTIVE THROTTLING - fixed minimum delay before every call.
  2. REACTIVE BACKOFF + JITTER - growing randomized delay after a failure.
  3. QUOTA-EXHAUSTION DETECTION - if a prompt still fails after all retries
     are exhausted, and the error looks like a persistent daily-quota issue
     (not a one-off transient error), the ENTIRE RUN STOPS IMMEDIATELY rather
     than burning through the remaining prompts, each failing after minutes
     of futile retries. A clear message tells you exactly how many prompts
     are done, how many remain, and when this provider's quota resets.
  4. ATOMIC CACHE WRITES - each response is written to a temp file then
     atomically renamed, so a hard interruption (Ctrl+C, crash, power loss)
     can never leave a corrupted, half-written cache file.
  5. RESUME LOGIC - a prompt only gets a cache file once it fully succeeds.
     Simply re-running this script later automatically skips every prompt
     that already has a cached response and continues with the rest, until
     all 220 prompts are complete. No manual bookkeeping needed.

GEMINI QUOTA RESET: daily request quotas (RPD) reset at midnight PACIFIC
TIME. Google significantly cut free-tier RPD limits in December 2025, and
the exact current number varies by model and can change without notice -
check the live limit shown in your Google AI Studio project rather than
trusting any hardcoded figure. If this script stops early with a quota
message, re-run it any time after midnight Pacific Time.

HOW TO RUN FOR REAL (VS Code):
  1. Create a .env file in the project root:  GEMINI_API_KEY=your_real_key_here
  2. pip install -r requirements.txt   (one-time, in the VS Code terminal)
  3. Open this file -> click the Run button (top-right)
  4. If it stops early due to quota, just re-run it after the reset time
     shown - it will pick up exactly where it left off.
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
logger = logging.getLogger("gemini_runner")

MODEL_NAME = "gemini-2.5-flash"
RESET_INFO = "midnight Pacific Time (Gemini's daily RPD quota reset)"

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "05_Logs_Results", "Gemini_Logs")
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "04_Datasets", "prompts.json")

MAX_RETRIES = 5
BASE_DELAY = 1.0
MAX_DELAY = 60.0
MIN_INTERVAL = 4.5  # seconds - keeps requests safely under Gemini's free-tier RPM cap

QUOTA_KEYWORDS = ["quota", "resource_exhausted", "daily", "rate limit exceeded", "429", "exceeded"]


def _looks_like_quota_exhausted(exc) -> bool:
    """
    Heuristic: if a prompt still fails after ALL backoff retries have been
    exhausted (meaning we already waited through roughly a minute of growing
    delays), a transient per-minute rate limit would normally have cleared by
    then. Persisting past that point, combined with quota-style wording in
    the error, is a strong signal this is a genuine daily quota exhaustion
    rather than a one-off blip.
    """
    text = str(exc).lower()
    return any(keyword in text for keyword in QUOTA_KEYWORDS)


class GeminiBenchmarkRunner:
    """Concrete runner: calls Gemini directly in call_model(). No abstract base class."""

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
            from google import genai
            self.client = genai.Client(api_key=api_key)

    def _load_prompts(self):
        with open(self.prompts_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _cache_path(self, prompt_id: str) -> str:
        return os.path.join(self.log_dir, f"gemini_{prompt_id}.json")

    def _already_done(self, prompt_id: str) -> bool:
        return os.path.exists(self._cache_path(prompt_id))

    def _write_cache_atomic(self, prompt_id: str, record: dict):
        """Write to a temp file then atomically rename, so an interrupted
        write can never leave a corrupted cache file behind."""
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
        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt_text,
        )
        return {
            "response_text": response.text,
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
                "model": "gemini",
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

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("FAILURE: GEMINI_API_KEY not set. Add it to a .env file in the project root.")
        sys.exit(1)

    runner = GeminiBenchmarkRunner(api_key=api_key)
    stats = runner.run()
    print(f"\nDone. {stats}")
    if stats["stopped_early"]:
        print(f"Re-run this script after {RESET_INFO} to continue.")


if __name__ == "__main__":
    main()