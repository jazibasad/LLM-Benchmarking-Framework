"""
test_llm_judge_scorer.py

Tests LLMJudgeScorer WITHOUT calling the real OpenRouter API. A fake
OpenAI-compatible client is injected. Tests the REBUILT per-run scoring
logic: each run of each prompt (1 for standard, 5 for stochastic-sample)
is scored independently, with explicit success/failure outcomes computed
in code from a documented threshold.

RUN THIS FIRST, before running llm_judge_scorer.py for real.

Running this file directly (VS Code Run button) saves results to
05_Logs_Results/tests_logs/Week_5/test_results.json.

Run with the VS Code Run button:
    python 03_Code/tests/test_llm_judge_scorer.py
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm_judge_scorer import (  # noqa: E402
    LLMJudgeScorer, parse_criteria, build_judge_prompt, _extract_json, SUCCESS_THRESHOLD,
)

WEEK_LABEL = "Week_5"
RESULTS_FILENAME = "test_results.json"


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


class FakeJudgeClient:
    def __init__(self, fail_count=0, judge_response=None):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(fail_count=fail_count, judge_response=judge_response)})()


@pytest.fixture
def tmp_input_and_scoredir():
    """Builds fixture data matching EXACTLY the shape build_results_tables.py
    (Week 4) produces: entries with a 'runs' array, 1 for standard prompts,
    5 for stochastic-sample prompts."""
    with tempfile.TemporaryDirectory() as tmp:
        input_file = os.path.join(tmp, "gemini_prompts_and_results.json")
        entries = [
            {
                "id": "P001", "category": "Knowledge Retrieval", "difficulty": "Easy",
                "prompt": "Write a Python function to check if a string is a palindrome.",
                "evaluation_criteria": "Syntax Correctness: 0-5 (Evaluates clean syntax and execution)\nEdge Case Handling: 0-5 (Evaluates handling of spaces, punctuation, and casing)",
                "max_score": "10", "model": "gemini", "stochastic_sample": False, "repeat_count": 1,
                "runs": [{"run_number": 1, "response_text": "def is_palindrome(s): return s == s[::-1]",
                          "temperature": 0.7, "latency_seconds": 1.234, "timestamp_utc": "2026-08-20T10:00:00+00:00"}],
            },
            {
                "id": "P037", "category": "Multi-step Reasoning", "difficulty": "Hard",
                "prompt": "Solve a multi-server load balancing optimization problem.",
                "evaluation_criteria": "Algorithm Execution: 0-5 (Evaluates weighted distribution math)\nHeadroom Adjustment: 0-5 (Evaluates dynamic capacity recalculation)",
                "max_score": "10", "model": "gemini", "stochastic_sample": True, "repeat_count": 5,
                "runs": [
                    {"run_number": i, "response_text": f"solution variant {i}", "temperature": 0.7,
                     "latency_seconds": 1.5 + i * 0.1, "timestamp_utc": f"2026-08-20T10:0{i}:00+00:00"}
                    for i in range(1, 6)
                ],
            },
        ]
        with open(input_file, "w") as f:
            json.dump(entries, f)
        score_dir = os.path.join(tmp, "scores")
        yield input_file, score_dir


def make_scorer(input_file, score_dir, fail_count=0, judge_response=None, **overrides):
    fake_client = FakeJudgeClient(fail_count=fail_count, judge_response=judge_response)
    defaults = dict(
        api_key="fake-key-not-used", input_file=input_file, score_dir=score_dir,
        file_prefix="gemini_", max_retries=3, base_delay=0.01, max_delay=0.05,
        min_interval=0.0, client=fake_client,
    )
    defaults.update(overrides)
    return LLMJudgeScorer(**defaults), fake_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_criteria_parsing():
    criteria = parse_criteria("Syntax Correctness: 0-5 (Clean code)\nEdge Case Handling: 0-5 (Handles edges)")
    assert len(criteria) == 2
    assert criteria[0]["name"] == "Syntax Correctness"


def test_judge_prompt_includes_score_anchors():
    criteria = parse_criteria("Accuracy: 0-5 (Correct facts)")
    judge_prompt = build_judge_prompt("What is the capital of France?", "Paris", criteria)
    assert "Complete Failure" in judge_prompt
    assert "Complete Success" in judge_prompt
    assert "SUCCESS THRESHOLD" in judge_prompt


def test_extract_json_handles_markdown_fences():
    wrapped = '```json\n{"criterion_scores": [{"name": "X", "score": 5, "justification": "Good."}]}\n```'
    result = _extract_json(wrapped)
    assert result["criterion_scores"][0]["score"] == 5


def test_single_run_prompt_scored_correctly(tmp_input_and_scoredir):
    input_file, score_dir = tmp_input_and_scoredir
    scorer, _ = make_scorer(input_file, score_dir)
    stats = scorer.run()

    # 1 run for P001 + 5 runs for P037 = 6 total
    assert stats == {"completed": 6, "skipped": 0, "failed": 0, "remaining": 0, "total_runs": 6, "stopped_early": False}

    # Single-run prompt: no _run suffix in filename
    with open(os.path.join(score_dir, "gemini_P001_scored.json")) as f:
        scored = json.load(f)
    assert scored["total_score"] == 4 + 3
    assert scored["run_number"] == 1
    assert scored["temperature"] == 0.7
    assert scored["latency_seconds"] == 1.234
    assert scored["success_threshold"] == SUCCESS_THRESHOLD


def test_multi_run_prompt_scored_separately_per_run(tmp_input_and_scoredir):
    input_file, score_dir = tmp_input_and_scoredir
    scorer, _ = make_scorer(input_file, score_dir)
    scorer.run()

    # All 5 runs should exist as SEPARATE scored files
    for i in range(1, 6):
        path = os.path.join(score_dir, f"gemini_P037_run{i}_scored.json")
        assert os.path.exists(path), f"Missing scored file for run {i}"
        with open(path) as f:
            scored = json.load(f)
        assert scored["run_number"] == i
        # Each run's own latency should be preserved distinctly
        assert scored["latency_seconds"] == round(1.5 + i * 0.1, 10) or abs(scored["latency_seconds"] - (1.5 + i * 0.1)) < 1e-6


def test_success_failure_outcome_boundary(tmp_input_and_scoredir):
    """Score of exactly 3 (the threshold) must count as Success, not Failure."""
    input_file, score_dir = tmp_input_and_scoredir
    boundary_response = json.dumps({
        "criterion_scores": [
            {"name": "Syntax Correctness", "score": 3, "justification": "Exactly at threshold."},
            {"name": "Edge Case Handling", "score": 2, "justification": "Below threshold."},
        ]
    })
    scorer, _ = make_scorer(input_file, score_dir, judge_response=boundary_response)
    scorer.run()

    with open(os.path.join(score_dir, "gemini_P001_scored.json")) as f:
        scored = json.load(f)
    assert scored["criterion_scores"][0]["outcome"] == "Success"
    assert scored["criterion_scores"][1]["outcome"] == "Failure"
    assert scored["overall_outcome"] == "Failure"  # one criterion failed -> overall fails


def test_resume_skips_already_scored_runs(tmp_input_and_scoredir):
    input_file, score_dir = tmp_input_and_scoredir
    scorer1, _ = make_scorer(input_file, score_dir)
    scorer1.run()

    scorer2, fake_client2 = make_scorer(input_file, score_dir)
    stats2 = scorer2.run()
    assert stats2 == {"completed": 0, "skipped": 6, "failed": 0, "remaining": 0, "total_runs": 6, "stopped_early": False}
    assert fake_client2.chat.completions.calls == 0


def test_backoff_retries_then_succeeds(tmp_input_and_scoredir):
    input_file, score_dir = tmp_input_and_scoredir
    scorer, fake_client = make_scorer(input_file, score_dir, fail_count=2)
    stats = scorer.run()
    assert stats["completed"] == 6
    assert fake_client.chat.completions.calls >= 3


def test_persistent_error_stops_on_first_run(tmp_input_and_scoredir):
    input_file, score_dir = tmp_input_and_scoredir
    scorer, _ = make_scorer(input_file, score_dir, fail_count=100, max_retries=1)
    stats = scorer.run()
    assert stats["stopped_early"] is True
    assert stats["completed"] == 0


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
    results_dir = os.path.join(os.path.dirname(__file__), "..", "..", "05_Logs_Results", "tests_logs", WEEK_LABEL)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, RESULTS_FILENAME)

    passed = sum(1 for r in collector.results if r["outcome"] == "passed")
    failed = sum(1 for r in collector.results if r["outcome"] == "failed")
    skipped = sum(1 for r in collector.results if r["outcome"] == "skipped")

    summary = {
        "week": WEEK_LABEL, "script_under_test": "llm_judge_scorer.py",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_tests": len(collector.results), "passed": passed, "failed": failed,
        "skipped": skipped, "exit_code": exit_code, "tests": collector.results,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTest results saved to: {os.path.abspath(results_path)}")


if __name__ == "__main__":
    collector = _ResultCollector()
    exit_code = pytest.main([__file__, "-v"], plugins=[collector])
    _save_weekly_results(collector, exit_code)
