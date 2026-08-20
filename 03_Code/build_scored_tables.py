"""
build_scored_tables.py

Consolidates the scattered per-run SCORED JSON files (written by
llm_judge_scorer.py, one file per prompt-run per provider) into ONE
structured, machine-readable table per provider. Correctly groups the 5
scored runs of each stochastic-sample prompt together into a single record
with a "scored_runs" array, matching the same grouping pattern used by
Week 4's build_results_tables.py for raw (unscored) data.

OUTPUT FILES (written to 05_Logs_Results/Scored_Results/Combined_Scores/):
  - gemini_scores_table.json
  - mistral_scores_table.json
  - groq_scores_table.json

Each also gets a completeness check against the full 220-prompt dataset,
verifying every prompt has the correct number of scored runs.

Run with the VS Code Run button:
    python 03_Code/build_scored_tables.py
"""

import json
import os
import re
import glob
from collections import defaultdict

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROMPTS_PATH = os.path.join(BASE_DIR, "04_Datasets", "prompts.json")
SCORED_RESULTS_DIR = os.path.join(BASE_DIR, "05_Logs_Results", "Scored_Results")
OUTPUT_DIR = os.path.join(SCORED_RESULTS_DIR, "Combined_Scores")

PROVIDERS = {
    "gemini": {
        "score_dir": os.path.join(SCORED_RESULTS_DIR, "Gemini_Scores"),
        "file_prefix": "gemini_",
        "output_name": "gemini_scores_table.json",
    },
    "mistral": {
        "score_dir": os.path.join(SCORED_RESULTS_DIR, "Mistral_Scores"),
        "file_prefix": "mistral_",
        "output_name": "mistral_scores_table.json",
    },
    "groq": {
        "score_dir": os.path.join(SCORED_RESULTS_DIR, "Groq_Scores"),
        "file_prefix": "groq_",
        "output_name": "groq_scores_table.json",
    },
}

# Matches both "gemini_P001_scored.json" (single-run) and
# "gemini_P037_run3_scored.json" (multi-run)
FILENAME_PATTERN = re.compile(r"^(?P<prefix>\w+_)(?P<id>P\d{3})(?:_run(?P<run>\d+))?_scored\.json$")


def _prompt_id_sort_key(prompt_id):
    match = re.search(r"\d+", prompt_id)
    return int(match.group()) if match else 0


def load_expected_prompts():
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        return {p["id"]: p for p in json.load(f)}


def build_table_for_provider(name, config, expected_prompts):
    score_dir = config["score_dir"]
    prefix = config["file_prefix"]

    pattern = os.path.join(score_dir, f"{prefix}*_scored.json")
    files = [f for f in glob.glob(pattern) if not f.endswith(".tmp")]

    grouped = defaultdict(list)
    for filepath in files:
        filename = os.path.basename(filepath)
        match = FILENAME_PATTERN.match(filename)
        if not match:
            print(f"  WARNING: unrecognized filename pattern, skipping: {filename}")
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                record = json.load(f)
            except json.JSONDecodeError:
                print(f"  WARNING: could not parse {filepath}, skipping.")
                continue
        grouped[match.group("id")].append(record)

    consolidated = []
    incomplete_prompts = []

    for prompt_id in sorted(grouped.keys(), key=_prompt_id_sort_key):
        scored_runs = grouped[prompt_id]
        scored_runs.sort(key=lambda r: r.get("run_number", 1))

        first_run = scored_runs[0]
        expected = expected_prompts.get(prompt_id, {})
        expected_repeat_count = expected.get("repeat_count", 1)

        if len(scored_runs) != expected_repeat_count:
            incomplete_prompts.append(
                f"{prompt_id}: expected {expected_repeat_count} scored run(s), found {len(scored_runs)}"
            )

        # Compute aggregate statistics across runs (useful directly for the
        # 20 stochastic-sample prompts; trivially just one value for others)
        total_scores = [r.get("total_score", 0) for r in scored_runs]
        outcomes = [r.get("overall_outcome") for r in scored_runs]

        consolidated.append({
            "id": prompt_id,
            "category": first_run.get("category"),
            "difficulty": first_run.get("difficulty"),
            "prompt": first_run.get("prompt"),
            "max_score": first_run.get("max_score"),
            "model": first_run.get("model"),
            "stochastic_sample": expected.get("stochastic_sample", False),
            "repeat_count": expected_repeat_count,
            "success_count": outcomes.count("Success"),
            "failure_count": outcomes.count("Failure"),
            "scored_runs": [
                {
                    "run_number": r.get("run_number", 1),
                    "response_text": r.get("response_text"),
                    "temperature": r.get("temperature"),
                    "latency_seconds": r.get("latency_seconds"),
                    "criterion_scores": r.get("criterion_scores"),
                    "total_score": r.get("total_score"),
                    "overall_outcome": r.get("overall_outcome"),
                    "judge_model": r.get("judge_model"),
                    "judge_timestamp_utc": r.get("judge_timestamp_utc"),
                }
                for r in scored_runs
            ],
        })

    found_ids = set(grouped.keys())
    all_ids = set(expected_prompts.keys())
    missing_ids = sorted(all_ids - found_ids, key=_prompt_id_sort_key)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, config["output_name"])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2, ensure_ascii=False)

    total_runs_written = sum(len(entry["scored_runs"]) for entry in consolidated)
    print(f"[{name}] {len(consolidated)} / {len(expected_prompts)} prompts "
          f"({total_runs_written} total scored runs) written to {output_path}")
    if missing_ids:
        print(f"  Missing {len(missing_ids)} prompt(s) entirely: {', '.join(missing_ids)}")
    if incomplete_prompts:
        print(f"  {len(incomplete_prompts)} prompt(s) with incomplete scored run counts:")
        for msg in incomplete_prompts:
            print(f"    - {msg}")
    if not missing_ids and not incomplete_prompts:
        print(f"  Complete - all {len(expected_prompts)} prompts scored with correct run counts.")

    return len(consolidated), missing_ids, incomplete_prompts


def main():
    print(f"Building consolidated score tables from: {os.path.abspath(SCORED_RESULTS_DIR)}\n")

    expected_prompts = load_expected_prompts()
    print(f"Full dataset: {len(expected_prompts)} prompts\n")

    for name, config in PROVIDERS.items():
        build_table_for_provider(name, config, expected_prompts)
        print()


if __name__ == "__main__":
    main()
