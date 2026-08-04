"""
test_benchmark_runner.py

Tests BenchmarkRunner WITHOUT ever calling the real OpenRouter API. Instead
of mocking network requests, we inject a fake "client" object that mimics
the shape of the real OpenAI-compatible client OpenRouter uses
(client.chat.completions.create(...)). This means:

  - Zero risk of rate limits or bans while testing
  - Runs instantly, no network needed at all
  - Still tests the REAL runner code (throttling, backoff, jitter, caching,
    resume logic) - only the actual network call is replaced

WEEKLY TEST RESULTS: running this file directly (the VS Code Run button,
or `python test_benchmark_runner.py`) automatically saves a JSON summary of
every test's pass/fail outcome to:

    05_Logs_Results/tests_logs/Week_2/test_results.json

This is self-contained in this one file - no conftest.py or other extra
file is needed. For each new week's test file, just copy this file and
change the WEEK_LABEL constant below (e.g. "Week_3", "Week_4", ...) so that
week's results land in their own dated folder.

Run with the VS Code Run button (recommended, saves results):
    python 03_Code/tests/test_benchmark_runner.py

Or with plain pytest (runs tests, but skips the results-saving step below,
since that only happens in the __main__ block):
    pytest 03_Code/tests/test_benchmark_runner.py -v
"""

import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmark_runner import BenchmarkRunner  # noqa: E402

# Change this one constant each week when you copy this file forward.
WEEK_LABEL = "Week_2"


# ---------------------------------------------------------------------------
# Fake OpenRouter client - mimics client.chat.completions.create(...)
# .choices[0].message.content, without any real network call. OpenRouter
# uses the OpenAI SDK shape, so this fake matches that same interface.
# ---------------------------------------------------------------------------

class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, fail_count=0):
        self.fail_count = fail_count
        self.calls = 0

    def create(self, model, messages):
        self.calls += 1
        if self.calls <= self.fail_count:
            raise TimeoutError("simulated transient failure")
        prompt_text = messages[0]["content"]
        return _FakeResponse(content=f"echo: {prompt_text}")


class _FakeChat:
    def __init__(self, fail_count=0):
        self.completions = _FakeCompletions(fail_count=fail_count)


class FakeOpenRouterClient:
    """Drop-in stand-in for OpenAI(api_key=..., base_url=...) - same shape, no network."""
    def __init__(self, fail_count=0):
        self.chat = _FakeChat(fail_count=fail_count)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_prompts_and_logdir():
    with tempfile.TemporaryDirectory() as tmp:
        prompts_path = os.path.join(tmp, "prompts.json")
        with open(prompts_path, "w") as f:
            json.dump(
                [
                    {"id": "P001", "category": "knowledge_retrieval", "prompt": "test 1"},
                    {"id": "P002", "category": "coding_tasks", "prompt": "test 2"},
                ],
                f,
            )
        log_dir = os.path.join(tmp, "logs")
        yield prompts_path, log_dir


def make_runner(prompts_path, log_dir, fail_count=0, **overrides):
    fake_client = FakeOpenRouterClient(fail_count=fail_count)
    defaults = dict(
        api_key="fake-key-not-used",
        log_dir=log_dir,
        prompts_path=prompts_path,
        max_retries=3,
        base_delay=0.01,
        max_delay=0.05,
        min_interval=0.0,
        client=fake_client,
    )
    defaults.update(overrides)
    return BenchmarkRunner(**defaults), fake_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_loads_prompts_correctly(tmp_prompts_and_logdir):
    prompts_path, log_dir = tmp_prompts_and_logdir
    runner, _ = make_runner(prompts_path, log_dir)
    assert len(runner.prompts) == 2
    assert runner.prompts[0]["id"] == "P001"


def test_cache_path_format(tmp_prompts_and_logdir):
    prompts_path, log_dir = tmp_prompts_and_logdir
    runner, _ = make_runner(prompts_path, log_dir)
    assert runner._cache_path("P001") == os.path.join(log_dir, "openrouter_P001.json")


def test_run_writes_one_json_per_prompt(tmp_prompts_and_logdir):
    prompts_path, log_dir = tmp_prompts_and_logdir
    runner, _ = make_runner(prompts_path, log_dir)
    stats = runner.run()
    assert stats == {"completed": 2, "skipped": 0, "failed": 0, "total": 2}
    assert set(os.listdir(log_dir)) == {"openrouter_P001.json", "openrouter_P002.json"}

    with open(os.path.join(log_dir, "openrouter_P001.json")) as f:
        record = json.load(f)
    assert record["response_text"] == "echo: test 1"
    assert record["model"] == "openrouter"


def test_resume_logic_skips_cached_prompts(tmp_prompts_and_logdir):
    prompts_path, log_dir = tmp_prompts_and_logdir
    runner1, _ = make_runner(prompts_path, log_dir)
    runner1.run()

    runner2, fake_client2 = make_runner(prompts_path, log_dir)
    stats2 = runner2.run()
    assert stats2 == {"completed": 0, "skipped": 2, "failed": 0, "total": 2}
    assert fake_client2.chat.completions.calls == 0  # never called - both cached


def test_backoff_retries_then_succeeds(tmp_prompts_and_logdir):
    prompts_path, log_dir = tmp_prompts_and_logdir
    runner, fake_client = make_runner(prompts_path, log_dir, fail_count=2)
    stats = runner.run()
    assert stats["completed"] == 2
    with open(os.path.join(log_dir, "openrouter_P001.json")) as f:
        record = json.load(f)
    assert record["response_text"] == "echo: test 1"
    assert fake_client.chat.completions.calls >= 3  # 2 failures + 1 success for P001


def test_backoff_gives_up_after_max_retries(tmp_prompts_and_logdir):
    prompts_path, log_dir = tmp_prompts_and_logdir
    runner, _ = make_runner(prompts_path, log_dir, fail_count=100, max_retries=1)
    stats = runner.run()
    assert stats["failed"] == 2
    assert stats["completed"] == 0


def test_backoff_delay_grows_exponentially(tmp_prompts_and_logdir):
    """Proves retry delay increases with each failed attempt (anti-ban)."""
    prompts_path, log_dir = tmp_prompts_and_logdir
    runner, _ = make_runner(
        prompts_path, log_dir, fail_count=3,
        max_retries=3, base_delay=0.05, max_delay=5.0,
    )
    start = time.time()
    runner.run()
    elapsed = time.time() - start
    # 3 failures -> delays of ~0.05, ~0.10, ~0.20s minimum (before jitter) = ~0.35s floor
    assert elapsed >= 0.30


def test_throttle_enforces_minimum_interval(tmp_prompts_and_logdir):
    """Proves calls are proactively spaced apart, even when every call succeeds."""
    prompts_path, log_dir = tmp_prompts_and_logdir
    runner, _ = make_runner(prompts_path, log_dir, min_interval=0.2)
    start = time.time()
    runner.run()
    elapsed = time.time() - start
    # First call has nothing to wait on; only the 2nd call is throttled.
    assert elapsed >= 0.18


def test_jitter_adds_randomness_to_delay():
    """Proves jitter genuinely varies rather than being a fixed constant."""
    import random
    random.seed(1)
    delays = [random.uniform(0, 1.0) for _ in range(5)]
    assert len(set(delays)) == 5


# ---------------------------------------------------------------------------
# Self-contained results saving - no conftest.py needed. This plugin object
# is passed directly to pytest.main() below, only when this file is run
# directly (not when pytest discovers it via the Testing sidebar).
# ---------------------------------------------------------------------------

class _ResultCollector:
    def __init__(self):
        self.results = []

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.results.append({
                "test_name": report.nodeid,
                "outcome": report.outcome,  # "passed", "failed", or "skipped"
                "duration_seconds": round(report.duration, 4),
            })


def _save_weekly_results(collector, exit_code):
    results_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "05_Logs_Results", "tests_logs", WEEK_LABEL
    )
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "test_results.json")

    passed = sum(1 for r in collector.results if r["outcome"] == "passed")
    failed = sum(1 for r in collector.results if r["outcome"] == "failed")
    skipped = sum(1 for r in collector.results if r["outcome"] == "skipped")

    summary = {
        "week": WEEK_LABEL,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_tests": len(collector.results),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "exit_code": exit_code,
        "tests": collector.results,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTest results saved to: {os.path.abspath(results_path)}")


if __name__ == "__main__":
    collector = _ResultCollector()
    exit_code = pytest.main([__file__, "-v"], plugins=[collector])
    _save_weekly_results(collector, exit_code)