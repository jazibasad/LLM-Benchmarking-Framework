"""
build_scored_tables.py

Consolidates the scattered per-prompt SCORED JSON files (written by
llm_judge_scorer.py, one file per prompt per provider in
05_Logs_Results/Scored_Results/<Provider>_Scores/) into ONE structured
machine-readable table per provider - the scored equivalent of Week 4's
build_results_tables.py.

Each output row contains everything needed to see exactly what was scored,
on what basis, and by whom: id, category, difficulty, prompt, response_text,
model, criterion_scores (each with name, score, justification), total_score,
max_score, judge_model, judge_timestamp_utc.

OUTPUT FILES (written to 05_Logs_Results/Scored_Results/Combined_Scores/):
  - gemini_scores_table.json
  - mistral_scores_table.json
  - groq_scores_table.json

Each also gets a completeness check against the full 220-prompt dataset, so
any prompt not yet scored for that provider is reported clearly.

Run with the VS Code Run button:
    python 03_Code/build_scored_tables.py
"""

import json
import os
import re
import glob

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


def _prompt_id_sort_key(record):
    match = re.search(r"\d+", record.get("id", "0"))
    return int(match.group()) if match else 0


def load_full_prompt_ids():
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return {p["id"] for p in prompts}


def build_table_for_provider(name, config, all_prompt_ids):
    score_dir = config["score_dir"]
    prefix = config["file_prefix"]

    pattern = os.path.join(score_dir, f"{prefix}*_scored.json")
    files = [f for f in glob.glob(pattern) if not f.endswith(".tmp")]

    records = []
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                records.append(json.load(f))
            except json.JSONDecodeError:
                print(f"  WARNING: could not parse {filepath}, skipping.")

    records.sort(key=_prompt_id_sort_key)

    found_ids = {r.get("id") for r in records}
    missing_ids = sorted(all_prompt_ids - found_ids, key=lambda pid: int(re.search(r"\d+", pid).group()))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, config["output_name"])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"[{name}] {len(records)} / {len(all_prompt_ids)} scored prompts written to {output_path}")
    if missing_ids:
        print(f"  Missing {len(missing_ids)} prompt(s) - not yet scored: {', '.join(missing_ids)}")
    else:
        print(f"  Complete - all {len(all_prompt_ids)} prompts scored.")

    return len(records), missing_ids


def main():
    print(f"Building consolidated score tables from: {os.path.abspath(SCORED_RESULTS_DIR)}\n")

    all_prompt_ids = load_full_prompt_ids()
    print(f"Full dataset: {len(all_prompt_ids)} prompts\n")

    summary = {}
    for name, config in PROVIDERS.items():
        count, missing = build_table_for_provider(name, config, all_prompt_ids)
        summary[name] = {"scored": count, "missing": missing}
        print()

    print("=== Summary ===")
    for name, info in summary.items():
        status = "COMPLETE" if not info["missing"] else f"INCOMPLETE ({len(info['missing'])} not yet scored)"
        print(f"  {name:10s} {info['scored']:3d}/{len(all_prompt_ids)}  {status}")


if __name__ == "__main__":
    main()
