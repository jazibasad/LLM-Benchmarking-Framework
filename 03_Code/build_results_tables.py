"""
build_results_tables.py

Consolidates the scattered per-prompt JSON log files (in
05_Logs_Results/<Provider>_Logs/) into ONE structured, machine-readable
table per provider - satisfying the requirement that all raw data (prompts,
results, logs) be submitted in structured form, with prompts and their
corresponding answers together.

UPDATED FOR THE STOCHASTIC RELIABILITY SAMPLE: 20 of the 220 prompts were
called 5 times each, producing separate files
(gemini_P037_run1.json ... run5.json) rather than one file. This script
correctly detects and groups those into a single record per prompt with a
"runs" array containing all repetitions, rather than treating each run as
an unrelated separate entry. Every record - whether single-run or
multi-run - has a uniform "runs" array, so downstream code (aggregation,
docx generation) doesn't need to special-case the two shapes.

OUTPUT FILES (written to 05_Logs_Results/Combined_Results/):
  - gemini_prompts_and_results.json
  - mistral_prompts_and_results.json
  - groq_prompts_and_results.json

Each also gets a completeness check against the full 220-prompt dataset,
verifying every prompt has the correct number of runs (1 for standard
prompts, 5 for stochastic-sample prompts).

Run with the VS Code Run button:
    python 03_Code/build_results_tables.py
"""

import json
import os
import re
import glob
from collections import defaultdict

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROMPTS_PATH = os.path.join(BASE_DIR, "04_Datasets", "prompts.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "05_Logs_Results", "Combined_Results")

PROVIDERS = {
    "gemini": {
        "log_dir": os.path.join(BASE_DIR, "05_Logs_Results", "Gemini_Logs"),
        "file_prefix": "gemini_",
        "output_name": "gemini_prompts_and_results.json",
    },
    "mistral": {
        "log_dir": os.path.join(BASE_DIR, "05_Logs_Results", "Mistral_Logs"),
        "file_prefix": "mistral_",
        "output_name": "mistral_prompts_and_results.json",
    },
    "groq": {
        "log_dir": os.path.join(BASE_DIR, "05_Logs_Results", "Groq_Logs"),
        "file_prefix": "groq_",
        "output_name": "groq_prompts_and_results.json",
    },
}

# Matches both "gemini_P001.json" (single-run) and "gemini_P037_run3.json" (multi-run)
FILENAME_PATTERN = re.compile(r"^(?P<prefix>\w+_)(?P<id>P\d{3})(?:_run(?P<run>\d+))?\.json$")


def _prompt_id_sort_key(prompt_id):
    match = re.search(r"\d+", prompt_id)
    return int(match.group()) if match else 0


def load_expected_prompts():
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        return {p["id"]: p for p in json.load(f)}


def build_table_for_provider(name, config, expected_prompts):
    log_dir = config["log_dir"]
    prefix = config["file_prefix"]

    pattern = os.path.join(log_dir, f"{prefix}*.json")
    files = [f for f in glob.glob(pattern) if not f.endswith(".tmp")]

    # Group files by prompt ID, collecting all runs for that prompt
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
        runs = grouped[prompt_id]
        runs.sort(key=lambda r: r.get("run_number", 1))

        first_run = runs[0]
        expected = expected_prompts.get(prompt_id, {})
        expected_repeat_count = expected.get("repeat_count", 1)

        if len(runs) != expected_repeat_count:
            incomplete_prompts.append(
                f"{prompt_id}: expected {expected_repeat_count} run(s), found {len(runs)}"
            )

        consolidated.append({
            "id": prompt_id,
            "category": first_run.get("category"),
            "difficulty": first_run.get("difficulty"),
            "prompt": first_run.get("prompt"),
            "evaluation_criteria": first_run.get("evaluation_criteria"),
            "max_score": first_run.get("max_score"),
            "model": first_run.get("model"),
            "stochastic_sample": expected.get("stochastic_sample", False),
            "repeat_count": expected_repeat_count,
            "runs": [
                {
                    "run_number": r.get("run_number", 1),
                    "response_text": r.get("response_text"),
                    "raw": r.get("raw"),
                    "temperature": r.get("temperature"),
                    "latency_seconds": r.get("latency_seconds"),
                    "timestamp_utc": r.get("timestamp_utc"),
                }
                for r in runs
            ],
        })

    found_ids = set(grouped.keys())
    all_ids = set(expected_prompts.keys())
    missing_ids = sorted(all_ids - found_ids, key=_prompt_id_sort_key)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, config["output_name"])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2, ensure_ascii=False)

    total_runs_written = sum(len(entry["runs"]) for entry in consolidated)
    print(f"[{name}] {len(consolidated)} / {len(expected_prompts)} prompts "
          f"({total_runs_written} total response runs) written to {output_path}")
    if missing_ids:
        print(f"  Missing {len(missing_ids)} prompt(s) entirely: {', '.join(missing_ids)}")
    if incomplete_prompts:
        print(f"  {len(incomplete_prompts)} prompt(s) with incomplete run counts:")
        for msg in incomplete_prompts:
            print(f"    - {msg}")
    if not missing_ids and not incomplete_prompts:
        print(f"  Complete - all {len(expected_prompts)} prompts present with correct run counts.")

    return len(consolidated), missing_ids, incomplete_prompts


def main():
    print(f"Building consolidated prompt+result tables from: {os.path.abspath(BASE_DIR)}\n")

    expected_prompts = load_expected_prompts()
    print(f"Full dataset: {len(expected_prompts)} prompts\n")

    for name, config in PROVIDERS.items():
        build_table_for_provider(name, config, expected_prompts)
        print()


if __name__ == "__main__":
    main()
