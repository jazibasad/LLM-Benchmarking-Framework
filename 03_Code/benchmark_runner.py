"""
benchmark_runner.py

A single, self-contained benchmark runner that calls the real OpenRouter API
directly (no abstract base class, no subclassing needed).

OpenRouter provides an OpenAI-compatible API that can route requests to many
different underlying models (including free-tier ones) through one endpoint.
This project uses OpenRouter in Week 2 for pipeline development and testing;
Groq is used separately in Week 4 as one of the three providers benchmarked
in the full data collection run.

Two independent safety mechanisms are built in to avoid getting rate-limited
or banned by OpenRouter's free tier:

  1. PROACTIVE THROTTLING - a fixed minimum delay is enforced BEFORE every
     single API call, regardless of whether previous calls succeeded. This
     keeps the average request rate below the provider's free-tier limit.

  2. REACTIVE BACKOFF + JITTER - if a call still fails (timeout, 429 rate
     limit, connection error), the wait time before retrying grows
     exponentially (1s, 2s, 4s, 8s...) with randomized jitter added, so
     repeated retries don't all pile up at the same instant.

Also provides:
  - Per-response disk caching as JSON (openrouter_[prompt_id].json)
  - Resume logic: already-cached prompt IDs are skipped on restart

PROMPT SOURCE (Week 2): this runner reads from 04_Datasets/prompts_test.json
by default - a small 5-prompt set (one per rubric category) used to verify
the pipeline end-to-end quickly. The full 220-prompt set (prompts.json) is
used starting Week 4's real data collection run, once all three provider
runners are decoupled (Week 3).

HOW TO RUN FOR REAL (VS Code):
  1. Create a .env file in the project root:  OPENROUTER_API_KEY=your_real_key_here
  2. pip install -r requirements.txt   (one-time, in the VS Code terminal)
  3. Open this file -> click the Run button (top-right)

This calls the real OpenRouter API for every prompt in
04_Datasets/prompts_test.json that doesn't already have a cached log in
05_Logs_Results/OpenRouter_Logs/.
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
logger = logging.getLogger("openrouter_runner")

# "openrouter/free" is OpenRouter's own router: it automatically selects a
# currently-available free model for each request, instead of pointing at
# one named free model that can get discontinued without notice (individual
# ":free" model slugs on OpenRouter churn frequently - this happened during
# Week 2 development, when a previously-working model was pulled).
#
# NOTE: because this is a random router, different calls may be answered by
# different underlying models. That's fine for Week 2 (proving the pipeline
# works end-to-end), but NOT appropriate for Week 4's real benchmarking data
# collection, where each provider needs one fixed, named model for a valid
# comparison. Week 4's runners will pin specific model names instead.
MODEL_NAME = "openrouter/free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "05_Logs_Results", "OpenRouter_Logs")

# Week 2 default: the small 5-prompt test set, not the full 220-prompt file.
# Switch this to "prompts.json" once you're ready for the full run (Week 4).
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "04_Datasets", "prompts_test.json")

MAX_RETRIES = 5
BASE_DELAY = 1.0
MAX_DELAY = 60.0
MIN_INTERVAL = 2.0  # seconds - minimum gap enforced before every call


class BenchmarkRunner:
    """
    Concrete runner: calls OpenRouter directly in call_model(). No abstract
    base class - this is the whole pipeline in one file.
    """

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

        # `client` can be injected for testing (see test_benchmark_runner.py).
        # In real use, we build the real OpenRouter client (OpenAI-compatible
        # SDK pointed at OpenRouter's base URL) from the API key.
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    def _load_prompts(self):
        with open(self.prompts_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _cache_path(self, prompt_id: str) -> str:
        return os.path.join(self.log_dir, f"openrouter_{prompt_id}.json")

    def _already_done(self, prompt_id: str) -> bool:
        """Resume logic: skip prompts that already have a cached log on disk."""
        return os.path.exists(self._cache_path(prompt_id))

    def _throttle(self):
        """PROACTIVE safety mechanism: enforce a minimum gap between calls."""
        elapsed = time.time() - self._last_call_time
        remaining = self.min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call_time = time.time()

    def call_model(self, prompt_text: str) -> dict:
        """Makes the real (or injected) OpenRouter API call for a single prompt."""
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt_text}],
        )
        return {
            "response_text": response.choices[0].message.content,
            "raw": {"model": MODEL_NAME},
        }

    def _call_with_backoff(self, prompt_text: str) -> dict:
        """REACTIVE safety mechanism: exponential backoff with jitter on failure."""
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
        """Executes the full benchmark pass, skipping already-cached prompts."""
        total = len(self.prompts)
        skipped = completed = failed = 0

        for item in self.prompts:
            prompt_id = item["id"]

            if self._already_done(prompt_id):
                skipped += 1
                continue

            logger.info("Running %s (%s)...", prompt_id, item["category"])

            try:
                result = self._call_with_backoff(item["prompt"])
            except RuntimeError as exc:
                logger.error("Giving up on %s: %s", prompt_id, exc)
                failed += 1
                continue

            record = {
                "id": prompt_id,
                "category": item["category"],
                "prompt": item["prompt"],
                "model": "openrouter",
                "response_text": result.get("response_text"),
                "raw": result.get("raw"),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }

            with open(self._cache_path(prompt_id), "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)

            completed += 1

        logger.info(
            "Run complete: %d completed, %d skipped (cached), %d failed, %d total.",
            completed, skipped, failed, total,
        )
        return {"completed": completed, "skipped": skipped, "failed": failed, "total": total}


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

    runner = BenchmarkRunner(api_key=api_key)
    stats = runner.run()
    print(f"\nDone. {stats}")


if __name__ == "__main__":
    main()