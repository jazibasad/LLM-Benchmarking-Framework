"""
compute_reliability_statistics.py

Computes the Reliability statistics required by the project proposal:
"Prompts with stochasticity will be repeated 5-10 times to compute
statistical measures (mean, median, standard deviation, coefficient of
variation)."

This is computed on the 20 stochastic-sample prompts (Week 3), each of
which was called 5 times per provider (Week 4) and scored 5 times
independently (Week 5). For each stochastic-sample prompt, per provider,
this script collects the 5 total_score values across the 5 runs and
computes:
  - Mean: average score across the 5 runs
  - Median: middle value across the 5 runs
  - Standard deviation: how much the 5 scores vary from the mean
  - Coefficient of variation (CV = std dev / mean): a NORMALIZED measure of
    variability, allowing fair comparison of consistency across prompts
    with different max_scores (a std dev of 1 point means something very
    different for a 10-point prompt vs a 20-point Hard prompt - CV corrects
    for this by expressing variability as a fraction of the mean)

A LOW coefficient of variation means the model gave consistent scores
across repeated calls to the same prompt (reliable). A HIGH CV means the
model's output quality (as scored) varied a lot run-to-run (less reliable).

Uses only Python's built-in statistics module - no external dependencies
required.

OUTPUT: 05_Logs_Results/Statistical_Analysis/reliability_statistics.json

Run with the VS Code Run button:
    python 03_Code/compute_reliability_statistics.py
"""

import json
import os
import statistics
import glob
import re

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROMPTS_PATH = os.path.join(BASE_DIR, "04_Datasets", "prompts.json")
SCORED_RESULTS_DIR = os.path.join(BASE_DIR, "05_Logs_Results", "Scored_Results")
OUTPUT_DIR = os.path.join(BASE_DIR, "05_Logs_Results", "Statistical_Analysis")

PROVIDERS = {
    "gemini": os.path.join(SCORED_RESULTS_DIR, "Gemini_Scores"),
    "mistral": os.path.join(SCORED_RESULTS_DIR, "Mistral_Scores"),
    "groq": os.path.join(SCORED_RESULTS_DIR, "Groq_Scores"),
}

FILENAME_PATTERN = re.compile(r"^(?P<prefix>\w+_)(?P<id>P\d{3})_run(?P<run>\d+)_scored\.json$")


def load_stochastic_prompt_ids():
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return sorted(
        {p["id"] for p in prompts if p.get("stochastic_sample")},
        key=lambda x: int(x[1:])
    )


def load_run_scores(score_dir, prompt_id):
    """Loads all available run scores (total_score) for one stochastic-sample prompt."""
    pattern = os.path.join(score_dir, f"*_{prompt_id}_run*_scored.json")
    files = glob.glob(pattern)
    run_scores = {}
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            record = json.load(f)
        run_scores[record.get("run_number")] = record.get("total_score")
    return run_scores


def compute_stats(scores):
    if len(scores) < 2:
        return None
    mean = statistics.mean(scores)
    median = statistics.median(scores)
    stdev = statistics.stdev(scores)  # sample standard deviation
    cv = (stdev / mean) if mean != 0 else None
    return {
        "n_runs": len(scores),
        "scores": scores,
        "mean": round(mean, 4),
        "median": round(median, 4),
        "standard_deviation": round(stdev, 4),
        "coefficient_of_variation": round(cv, 4) if cv is not None else None,
    }


def main():
    stochastic_ids = load_stochastic_prompt_ids()
    print(f"Computing reliability statistics for {len(stochastic_ids)} stochastic-sample prompts, "
          f"across {len(PROVIDERS)} providers.\n")

    report = {"providers": {}}

    for provider, score_dir in PROVIDERS.items():
        print(f"=== {provider} ===")
        provider_results = {}
        all_cvs = []

        for prompt_id in stochastic_ids:
            run_scores_dict = load_run_scores(score_dir, prompt_id)
            if len(run_scores_dict) < 2:
                print(f"  {prompt_id}: only {len(run_scores_dict)} run(s) found - skipping (need >= 2 for stdev)")
                continue

            ordered_scores = [run_scores_dict[k] for k in sorted(run_scores_dict.keys())]
            stats = compute_stats(ordered_scores)
            provider_results[prompt_id] = stats
            if stats["coefficient_of_variation"] is not None:
                all_cvs.append(stats["coefficient_of_variation"])

            print(f"  {prompt_id}: mean={stats['mean']}  median={stats['median']}  "
                  f"stdev={stats['standard_deviation']}  CV={stats['coefficient_of_variation']}")

        overall_cv_mean = round(statistics.mean(all_cvs), 4) if all_cvs else None
        report["providers"][provider] = {
            "per_prompt": provider_results,
            "n_prompts_analyzed": len(provider_results),
            "average_coefficient_of_variation": overall_cv_mean,
        }
        print(f"  --> Average CV across all analyzed prompts: {overall_cv_mean}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "reliability_statistics.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Full report saved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
