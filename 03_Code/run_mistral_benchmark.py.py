"""
run_mistral_benchmark.py

Real Mistral AI runner for Week 4 data collection, reading the full 220-prompt
curated dataset directly. This is one of three decoupled provider runners
(alongside run_gemini_benchmark.py and run_groq_benchmark.py), each writing
to its own partitioned log directory.

PROVIDER HISTORY: Mistral replaces Cerebras as the third benchmarked provider.
Cerebras was originally chosen in Week 3 (replacing OpenAI), but during Week 4
development Cerebras introduced a mandatory payment-method requirement to
activate API access, and the available local payment method could not be
authorized for international billing. Mistral AI was selected instead: a
permanent (not trial/expiring) free tier requiring no payment method at all.

SAFETY MECHANISMS: see run_gemini_benchmark.py docstring for the full
explanation - proactive throttling, reactive backoff+jitter, persistent-error
detection with early stop (covers both quota exhaustion AND model
retirement/unavailability - this project has hit both failure modes during
development), atomic cache writes, and resume-by-cache-file.

MISTRAL RATE LIMITS: the free "Experiment" tier allows roughly 1 request/
second and high monthly token volume (order of 1 billion tokens/month),
which comfortably covers a 220-prompt run in one sitting. No daily quota
reset time is documented as a hard wall the way Gemini/Groq have one - if a
persistent error does occur, check console.mistral.ai for current limits.

HOW TO RUN FOR REAL (VS Code):
  1. Create a .env file in the project root:  MISTRAL_API_KEY=your_real_key_here
  2. pip install -r requirements.txt   (one-time, in the VS Code terminal;
     requires Python 3.10+ for the mistralai SDK)
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
logger = logging.getLogger("mistral_runner")

# "open-mistral-nemo" is Mistral's recommended free-tier starting model -
# an open-weight model (128k context, trained with NVIDIA) distinct from the
# paid "mistral-large-latest" / "mistral-small-latest" premier models. If
# this 404s or is retired, check console.mistral.ai for the current free
# model list and update it here.
MODEL_NAME = "open-mistral-nemo"
RESET_INFO = "check console.mistral.ai for current limits (no fixed daily reset documented)"

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "05_Logs_Results", "Mistral_Logs")
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "04_Datasets", "prompts.json")

MAX_RETRIES = 5
BASE_DELAY = 1.0
MAX_DELAY = 60.0
MIN_INTERVAL = 1.2  # seconds - keeps requests under Mistral's ~1 request/second free-tier cap

PERSISTENT_ERROR_KEYWORDS = [
    "quota", "resource_exhausted", "daily", "rate limit exceeded", "429", "exceeded",
    "not_found", "no longer available", "model not found", "404",
    "payment", "402", "billing",
]


def _looks_like_persistent_error(exc) -> bool:
    """
    Heuristic: if a prompt still fails after ALL backoff retries have been
    exhausted, and the error contains quota/model-availability/billing
    wording, every remaining prompt will almost certainly fail the exact
    same way - so the run stops rather than grinding through all of them
    uselessly. This project has already hit this exact pattern twice
    (OpenRouter model retirement in Week 2, Gemini model retirement and
    Cerebras billing requirement in Week 4).
    """
    text = str(exc).lower()
    return any(keyword in text for keyword in PERSISTENT_ERROR_KEYWORDS)


class MistralBenchmarkRunner:
    """Concrete runner: calls Mistral directly in call_model(). No abstract base class."""

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

        # `client` can be injected for testing (see tests/test_mistral_runner.py).
        if client is not None:
            self.client = client
        else:
            from mistralai.client import Mistral
            self.client = Mistral(api_key=api_key)

    def _load_prompts(self):
        with open(self.prompts_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _cache_path(self, prompt_id: str) -> str:
        return os.path.join(self.log_dir, f"mistral_{prompt_id}.json")

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
        response = self.client.chat.complete(
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
                systemic = _looks_like_persistent_error(exc) or completed == 0
                if systemic:
                    remaining = total - completed - skipped
                    reason = (
                        "quota/model-availability/billing wording detected"
                        if _looks_like_persistent_error(exc)
                        else "the very first prompt failed after all retries - "
                             "likely a systemic issue (model name, API key, or endpoint), "
                             "not a per-prompt problem"
                    )
                    logger.error(
                        "Persistent error detected on %s after exhausting all retries (%s): %s",
                        prompt_id, reason, exc,
                    )
                    logger.error(
                        "STOPPING RUN EARLY. %d completed, %d remaining. %s. "
                        "Already-completed prompts are cached and will be skipped "
                        "automatically, so re-running will continue from %s.",
                        completed, remaining, RESET_INFO, prompt_id,
                    )
                    stopped_early = True
                    break
                logger.error("Giving up on %s (isolated error): %s", prompt_id, exc)
                failed += 1
                continue

            record = {
                "id": prompt_id,
                "category": item["category"],
                "difficulty": item.get("difficulty"),
                "prompt": item["prompt"],
                "evaluation_criteria": item.get("evaluation_criteria"),
                "max_score": item.get("max_score"),
                "model": "mistral",
                "response_text": result.get("response_text"),
                "raw": result.get("raw"),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            self._write_cache_atomic(prompt_id, record)
            completed += 1

        remaining = total - completed - skipped
        logger.info(
            "Run %s: %d completed, %d skipped (cached), %d failed, %d remaining, %d total.",
            "stopped early" if stopped_early else "complete",
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

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("FAILURE: MISTRAL_API_KEY not set. Add it to a .env file in the project root.")
        sys.exit(1)

    runner = MistralBenchmarkRunner(api_key=api_key)
    stats = runner.run()
    print(f"\nDone. {stats}")
    if stats["stopped_early"]:
        print(f"Check the log above for why, then re-run to continue.")


if __name__ == "__main__":
    main()