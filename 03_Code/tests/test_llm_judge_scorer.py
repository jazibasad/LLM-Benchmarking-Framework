"""
test_llm_judge_scorer.py

Tests LLMJudgeScorer WITHOUT calling the real OpenRouter API. A fake
OpenAI-compatible client is injected that returns pre-scripted judge
responses, so this verifies the actual scoring logic (criteria parsing,
prompt construction, response parsing, total-score computation, caching,
resume, persistent-error detection, atomic writes) with zero real API cost.

RUN THIS FIRST, BEFORE running llm_judge_scorer.py for real - this is the
required Week 5 unit-testing step, verifying the scoring pipeline logic is
correct before it ever spends real API calls on the actual 660 responses.

Running this file directly (VS Code Run button) saves a JSON summary of
results to 05_Logs_Results/tests_logs/Week_5/test_results.json.

Run with the VS Code Run button:
    python 03_Code/tests/test_llm_judge_scorer.py
"""

import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm_judge_scorer import (  # noqa: E402
    LLMJudgeScorer, parse_criteria, build_judge_prompt, _extract_json,
)

WEEK_LABEL = "Week_5"
RESULTS_FILENAME = "test_results.json"


# ---------------------------------------------------------------------------
# Fake OpenRouter/OpenAI-compatible client - mimics
# client.chat.completions.create(...).choices[0].message.content, returning
# a scripted judge JSON response instead of a real network call.
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
    def __init__(self, fail_count=0, judge_response=None):
        self.fail_count = fail_count
        self.calls = 0
        self.judge_response = judge_response or json.dumps({
            "criterion_scores": [
                {"name": "Syntax Correctness", "score": 4, "justification": "Runs correctly with minor style issues."},
                {"name": "Edge Case Handling", "score": 3, "justification": "Handles most cases but misses one edge case."},
            ]
        })

    def create(self, model, messages):
        self.calls += 1
        if self.calls <= self.fail_count:
            raise TimeoutError("simulated transient failure")
        return _FakeResponse(content=self.judge_response)


class _FakeChat:
    def __init__(self, fail_count=0, judge_response=None):
        self.completions = _FakeCompletions(fail_count=fail_count, judge_response=judge_response)


class FakeJudgeClient:
    """Drop-in stand-in for OpenAI(api_key=..., base_url=...) - same shape, no network."""
    def __init__(self, fail_count=0, judge_response=None):
        self.chat = _FakeChat(fail_count=fail_count, judge_response=judge_response)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_input_and_scoredir():
    """
    Builds the test fixture from the REAL Week 2 prompts_test.json (5 real
    prompts, one per rubric category), attaching a synthetic model response
    to each so the scoring pipeline can be exercised realistically - the
    prompts and evaluation_criteria are real project data, only the
    response_text is invented since this fixture never calls a real model.
    """
    real_prompts_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "04_Datasets", "prompts_test.json"
    )
    with open(real_prompts_path, "r", encoding="utf-8") as f:
        real_prompts = json.load(f)

    synthetic_responses = {
        "P001": "def is_palindrome(s):\n    clean = ''.join(c.lower() for c in s if c.isalnum())\n    return clean == clean[::-1]",
        "P041": "Train A travels at 60 km/h, Train B at 70 km/h, combined closing speed 130 km/h. "
                "Train B has a 45-minute head start covering 52.5 km, leaving 157.5 km. "
                "At 130 km/h that takes 1.2115 hours (~1h 13min), so they meet at approximately 10:58 AM.",
        "P091": "Pros:\n1. Flexible schedule\n2. No commute\n3. Better work-life balance\n\n"
                "Cons:\n1. Isolation\n2. Communication challenges\n3. Harder to separate work and home",
        "P131": "There is no credible peer-reviewed study proving this. Reading in dim light can "
                "cause temporary eye strain, but no permanent damage - this is a common myth.",
        "P171": "def reverse_linked_list(head):\n    prev = None\n    while head:\n        nxt = head.next\n        head.next = prev\n        prev = head\n        head = nxt\n    return prev",
    }

    with tempfile.TemporaryDirectory() as tmp:
        records = []
        for p in real_prompts:
            record = dict(p)
            record["model"] = "gemini"
            record["response_text"] = synthetic_responses.get(p["id"], "Sample response.")
            records.append(record)

        input_file = os.path.join(tmp, "gemini_prompts_and_results.json")
        with open(input_file, "w") as f:
            json.dump(records, f)

        score_dir = os.path.join(tmp, "scores")
        yield input_file, score_dir


def make_scorer(input_file, score_dir, fail_count=0, judge_response=None, **overrides):
    fake_client = FakeJudgeClient(fail_count=fail_count, judge_response=judge_response)
    defaults = dict(
        api_key="fake-key-not-used",
        input_file=input_file,
        score_dir=score_dir,
        file_prefix="gemini_",
        max_retries=3,
        base_delay=0.01,
        max_delay=0.05,
        min_interval=0.0,
        client=fake_client,
    )
    defaults.update(overrides)
    return LLMJudgeScorer(**defaults), fake_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_criteria_parsing_simple():
    criteria = parse_criteria("Syntax Correctness: 0-5 (Clean code)\nEdge Case Handling: 0-5 (Handles edges)")
    assert len(criteria) == 2
    assert criteria[0]["name"] == "Syntax Correctness"
    assert criteria[0]["description"] == "Clean code"


def test_criteria_parsing_multi_module():
    criteria = parse_criteria(
        "Module 1 Score: 0-5 (A)\nModule 2 Score: 0-5 (B)\nModule 3 Score: 0-5 (C)\nSystem Integration Score: 0-5 (D)"
    )
    assert len(criteria) == 4
    assert criteria[3]["name"] == "System Integration Score"


def test_judge_prompt_includes_prompt_and_criteria():
    criteria = parse_criteria("Accuracy: 0-5 (Correct facts)")
    judge_prompt = build_judge_prompt("What is the capital of France?", "Paris", criteria)
    assert "capital of France" in judge_prompt
    assert "Accuracy" in judge_prompt
    assert "JSON" in judge_prompt


def test_extract_json_handles_markdown_fences():
    wrapped = '```json\n{"criterion_scores": [{"name": "X", "score": 5, "justification": "Good."}]}\n```'
    result = _extract_json(wrapped)
    assert result["criterion_scores"][0]["score"] == 5


def test_extract_json_handles_plain_json():
    plain = '{"criterion_scores": [{"name": "X", "score": 3, "justification": "OK."}]}'
    result = _extract_json(plain)
    assert result["criterion_scores"][0]["score"] == 3


def test_run_scores_all_records_and_computes_total(tmp_input_and_scoredir):
    input_file, score_dir = tmp_input_and_scoredir
    scorer, _ = make_scorer(input_file, score_dir)
    stats = scorer.run()
    assert stats == {"completed": 5, "skipped": 0, "failed": 0, "remaining": 0, "total": 5, "stopped_early": False}

    with open(os.path.join(score_dir, "gemini_P001_scored.json")) as f:
        scored = json.load(f)
    # Total score must be the SUM of criterion scores, computed in code,
    # not trusted from the judge's own text.
    assert scored["total_score"] == 4 + 3  # matches the fake judge_response above
    assert scored["max_score"] == "10"
    assert scored["judge_model"] == "openrouter/free"
    assert len(scored["criterion_scores"]) == 2
    assert "justification" in scored["criterion_scores"][0]
    # Confirm this is real Week 2 prompt data, not invented fixture text
    assert "palindrome" in scored["prompt"].lower()


def test_resume_skips_already_scored(tmp_input_and_scoredir):
    input_file, score_dir = tmp_input_and_scoredir
    scorer1, _ = make_scorer(input_file, score_dir)
    scorer1.run()

    scorer2, fake_client2 = make_scorer(input_file, score_dir)
    stats2 = scorer2.run()
    assert stats2 == {"completed": 0, "skipped": 5, "failed": 0, "remaining": 0, "total": 5, "stopped_early": False}
    assert fake_client2.chat.completions.calls == 0  # never called - all already scored


def test_backoff_retries_then_succeeds(tmp_input_and_scoredir):
    input_file, score_dir = tmp_input_and_scoredir
    scorer, fake_client = make_scorer(input_file, score_dir, fail_count=2)
    stats = scorer.run()
    assert stats["completed"] == 5
    assert fake_client.chat.completions.calls >= 3  # 2 failures + success on first record, then 4 more successes


def test_persistent_error_stops_early_on_first_record(tmp_input_and_scoredir):
    input_file, score_dir = tmp_input_and_scoredir
    scorer, _ = make_scorer(input_file, score_dir, fail_count=100, max_retries=1)
    stats = scorer.run()
    assert stats["stopped_early"] is True
    assert stats["completed"] == 0


def test_malformed_judge_response_is_treated_as_failure(tmp_input_and_scoredir):
    """A judge response that isn't valid JSON should trigger the normal
    retry/failure path, not crash the whole script."""
    input_file, score_dir = tmp_input_and_scoredir
    scorer, _ = make_scorer(input_file, score_dir, judge_response="this is not json at all",
                             max_retries=1)
    stats = scorer.run()
    # First record fails entirely (bad response every retry) -> systemic stop
    assert stats["stopped_early"] is True


def test_atomic_write_leaves_no_tmp_files(tmp_input_and_scoredir):
    input_file, score_dir = tmp_input_and_scoredir
    scorer, _ = make_scorer(input_file, score_dir)
    scorer.run()
    files = os.listdir(score_dir)
    assert not any(f.endswith(".tmp") for f in files)


# ---------------------------------------------------------------------------
# Self-contained weekly results saving - no conftest.py needed.
# ---------------------------------------------------------------------------

class _ResultCollector:
    def __init__(self):
        self.results = []

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.results.append({
                "test_name": report.nodeid,
                "outcome": report.outcome,
                "duration_seconds": round(report.duration, 4),
            })


def _save_weekly_results(collector, exit_code):
    results_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "05_Logs_Results", "tests_logs", WEEK_LABEL
    )
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, RESULTS_FILENAME)

    passed = sum(1 for r in collector.results if r["outcome"] == "passed")
    failed = sum(1 for r in collector.results if r["outcome"] == "failed")
    skipped = sum(1 for r in collector.results if r["outcome"] == "skipped")

    summary = {
        "week": WEEK_LABEL,
        "script_under_test": "llm_judge_scorer.py",
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
