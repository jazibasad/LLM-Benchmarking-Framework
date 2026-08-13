"""
build_results_tables.py

Consolidates the scattered per-prompt JSON log files (one file per prompt,
per provider, in 05_Logs_Results/<Provider>_Logs/) into ONE structured file
per provider containing every prompt together with its corresponding answer
and all raw data - exactly as required by the supervisor's instruction:

    "Include separate files for actual prompts and their corresponding
    answers. The raw data must also be submitted as well in structured
    form. Every data. Prompts, results, datasets, logs results, surveys, etc."

For each provider (Gemini, Groq, Mistral), this produces one JSON file that
is a table structure: an array of 220 records, one row per prompt, every
row with the same set of columns (id, category, difficulty, prompt,
evaluation_criteria, max_score, model, response_text, raw, timestamp_utc).

OUTPUT FILES (written to 05_Logs_Results/Combined_Results/):
  - gemini_prompts_and_results.json
  - groq_prompts_and_results.json
  - mistral_prompts_and_results.json

Each also gets a matching completeness check against the full 220-prompt
dataset, so any missing prompt IDs for that provider are reported clearly
rather than silently producing an incomplete table.

Run with the VS Code Run button:
    python 03_Code/build_results_tables.py
"""

import json
import os
import re
import glob

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROMPTS_PATH = os.path.join(BASE_DIR, "04_Datasets", "prompts.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "05_Logs_Results", "Combined_Results")

PROVIDERS = {
    "gemini": {
        "log_dir": os.path.join(BASE_DIR, "05_Logs_Results", "Gemini_Logs"),
        "file_prefix": "gemini_",
        "output_name": "gemini_prompts_and_results.json",
    },
    "groq": {
        "log_dir": os.path.join(BASE_DIR, "05_Logs_Results", "Groq_Logs"),
        "file_prefix": "groq_",
        "output_name": "groq_prompts_and_results.json",
    },
    "mistral": {
        "log_dir": os.path.join(BASE_DIR, "05_Logs_Results", "Mistral_Logs"),
        "file_prefix": "mistral_",
        "output_name": "mistral_prompts_and_results.json",
    },
}

# Natural sort so P002 comes before P010, not after P1 alphabetically
def _prompt_id_sort_key(record):
    match = re.search(r"\d+", record.get("id", "0"))
    return int(match.group()) if match else 0


def load_full_prompt_ids():
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return {p["id"] for p in prompts}


def build_table_for_provider(name, config, all_prompt_ids):
    log_dir = config["log_dir"]
    prefix = config["file_prefix"]

    pattern = os.path.join(log_dir, f"{prefix}*.json")
    files = [f for f in glob.glob(pattern) if not f.endswith(".tmp")]

    records = []
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                record = json.load(f)
            except json.JSONDecodeError:
                print(f"  WARNING: could not parse {filepath}, skipping.")
                continue
            records.append(record)

    records.sort(key=_prompt_id_sort_key)

    found_ids = {r.get("id") for r in records}
    missing_ids = sorted(all_prompt_ids - found_ids, key=lambda pid: int(re.search(r"\d+", pid).group()))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, config["output_name"])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"[{name}] {len(records)} / {len(all_prompt_ids)} prompts written to {output_path}")
    if missing_ids:
        print(f"  Missing {len(missing_ids)} prompt(s): {', '.join(missing_ids)}")
    else:
        print(f"  Complete - all {len(all_prompt_ids)} prompts present.")

    return len(records), missing_ids


def main():
    print(f"Building consolidated prompt+result tables from: {os.path.abspath(BASE_DIR)}\n")

    all_prompt_ids = load_full_prompt_ids()
    print(f"Full dataset: {len(all_prompt_ids)} prompts (from prompts.json)\n")

    summary = {}
    for name, config in PROVIDERS.items():
        count, missing = build_table_for_provider(name, config, all_prompt_ids)
        summary[name] = {"completed": count, "missing": missing}
        print()

    print("=== Summary ===")
    for name, info in summary.items():
        status = "COMPLETE" if not info["missing"] else f"INCOMPLETE ({len(info['missing'])} missing)"
        print(f"  {name:10s} {info['completed']:3d}/{len(all_prompt_ids)}  {status}")


if __name__ == "__main__":
    main()