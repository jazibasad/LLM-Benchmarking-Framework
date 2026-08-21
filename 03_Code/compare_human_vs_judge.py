"""
compare_human_vs_judge.py

Compares scores from ALL rater files (human_scores_rater1.json,
human_scores_rater2.json, ...) against the LLM judge's scores, AND against
each other. This satisfies both parts of the proposal: "compare automated
scores against human-evaluated scores" (Week 6) and "Inter-rater agreement"
(listed under Week 7's Statistical Analysis, computed here since it needs
this same underlying data).

Each rater's individual results are kept SEPARATE throughout - never
merged or averaged together before comparison - so each rater's judgment
remains individually traceable.

METRICS COMPUTED (for every rater-vs-judge pair AND every rater-vs-rater
pair): exact match rate, within-1-point rate, mean absolute difference,
Pearson correlation - overall and per-provider.

Run with the VS Code Run button:
    python 03_Code/compare_human_vs_judge.py
"""

import glob
import json
import os
import re
from collections import defaultdict
from itertools import combinations

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
HUMAN_VALIDATION_DIR = os.path.join(BASE_DIR, "05_Logs_Results", "Survey")
SCORED_RESULTS_DIR = os.path.join(BASE_DIR, "05_Logs_Results", "Scored_Results", "Combined_Scores")
OUTPUT_PATH = os.path.join(HUMAN_VALIDATION_DIR, "agreement_report.json")

PROVIDER_FILES = {
    "gemini": "gemini_scores_table.json",
    "mistral": "mistral_scores_table.json",
    "groq": "groq_scores_table.json",
}

RATER_FILE_PATTERN = re.compile(r"human_scores_rater(\d+)\.json$")


def discover_rater_files():
    files = glob.glob(os.path.join(HUMAN_VALIDATION_DIR, "human_scores_rater*.json"))
    raters = {}
    for f in files:
        match = RATER_FILE_PATTERN.search(os.path.basename(f))
        if match:
            raters[int(match.group(1))] = f
    return dict(sorted(raters.items()))


def load_rater_scores(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    scores = {}
    for item in data["items"]:
        for crit in item["your_scores"]:
            if crit["your_score"] is not None:
                scores[(item["id"], item["provider"], crit["criterion_name"])] = crit["your_score"]
    return scores


def load_judge_scores():
    judge_lookup = {}
    for provider, filename in PROVIDER_FILES.items():
        path = os.path.join(SCORED_RESULTS_DIR, filename)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found - has build_scored_tables.py been run?")
            continue
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            scored_runs = entry.get("scored_runs", [])
            if not scored_runs:
                continue
            for c in scored_runs[0].get("criterion_scores", []):
                judge_lookup[(entry["id"], provider, c["name"])] = c["score"]
    return judge_lookup


def pearson_correlation(x, y):
    n = len(x)
    if n < 2:
        return None
    mean_x, mean_y = sum(x) / n, sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
    std_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
    if std_x == 0 or std_y == 0:
        return None
    return cov / (std_x * std_y)


def cohens_kappa(x, y, weighted=None):
    """
    Computes Cohen's Kappa between two lists of ordinal ratings (0-5) - the
    standard, purpose-built statistic for inter-rater agreement, which
    corrects for the agreement expected by chance alone (something raw
    exact-match rate or correlation does not account for).

    weighted=None gives standard (unweighted) Kappa - treats any
    disagreement equally regardless of size.
    weighted="quadratic" gives quadratic-weighted Kappa - more appropriate
    for ordinal 0-5 scores, since it penalizes a 0-vs-5 disagreement far
    more than a 3-vs-4 disagreement, rather than treating all misses the
    same.
    """
    n = len(x)
    if n == 0:
        return None
    categories = sorted(set(x) | set(y))
    k = len(categories)
    if k < 2:
        return None  # undefined if only one score value appears at all

    cat_index = {c: i for i, c in enumerate(categories)}
    matrix = [[0] * k for _ in range(k)]
    for xi, yi in zip(x, y):
        matrix[cat_index[xi]][cat_index[yi]] += 1

    row_marginals = [sum(matrix[i][j] for j in range(k)) for i in range(k)]
    col_marginals = [sum(matrix[i][j] for i in range(k)) for j in range(k)]

    if weighted is None:
        po = sum(matrix[i][i] for i in range(k)) / n
        pe = sum(row_marginals[i] * col_marginals[i] for i in range(k)) / (n * n)
        if pe >= 1:
            return None
        return (po - pe) / (1 - pe)
    else:  # quadratic weighted
        def w(i, j):
            return ((i - j) ** 2) / ((k - 1) ** 2) if k > 1 else 0

        do = sum(matrix[i][j] * w(i, j) for i in range(k) for j in range(k)) / n
        de = sum(row_marginals[i] * col_marginals[j] * w(i, j) for i in range(k) for j in range(k)) / (n * n)
        if de == 0:
            return None
        return 1 - (do / de)


def compute_agreement(scores_a, scores_b, label_a, label_b):
    shared_keys = set(scores_a.keys()) & set(scores_b.keys())
    if not shared_keys:
        return None

    a_vals, b_vals = [], []
    per_provider = defaultdict(lambda: {"a": [], "b": []})
    for key in shared_keys:
        pid, provider, crit = key
        a_vals.append(scores_a[key])
        b_vals.append(scores_b[key])
        per_provider[provider]["a"].append(scores_a[key])
        per_provider[provider]["b"].append(scores_b[key])

    n = len(a_vals)
    exact = sum(1 for a, b in zip(a_vals, b_vals) if a == b)
    within_1 = sum(1 for a, b in zip(a_vals, b_vals) if abs(a - b) <= 1)
    mean_abs_diff = sum(abs(a - b) for a, b in zip(a_vals, b_vals)) / n
    corr = pearson_correlation(a_vals, b_vals)
    kappa = cohens_kappa(a_vals, b_vals)
    weighted_kappa = cohens_kappa(a_vals, b_vals, weighted="quadratic")

    result = {
        "comparison": f"{label_a} vs {label_b}",
        "n_criteria_compared": n,
        "exact_match_rate": round(exact / n, 4),
        "within_1_point_rate": round(within_1 / n, 4),
        "mean_absolute_difference": round(mean_abs_diff, 4),
        "pearson_correlation": round(corr, 4) if corr is not None else None,
        "cohens_kappa": round(kappa, 4) if kappa is not None else None,
        "cohens_kappa_quadratic_weighted": round(weighted_kappa, 4) if weighted_kappa is not None else None,
        "per_provider": {},
    }
    for provider, vals in per_provider.items():
        pn = len(vals["a"])
        if pn == 0:
            continue
        p_exact = sum(1 for a, b in zip(vals["a"], vals["b"]) if a == b)
        p_corr = pearson_correlation(vals["a"], vals["b"])
        result["per_provider"][provider] = {
            "n": pn,
            "exact_match_rate": round(p_exact / pn, 4),
            "pearson_correlation": round(p_corr, 4) if p_corr is not None else None,
        }
    return result


def main():
    rater_files = discover_rater_files()
    if not rater_files:
        print(f"FAILURE: no human_scores_rater*.json files found in {HUMAN_VALIDATION_DIR}. "
              f"Run build_human_validation_materials.py first.")
        return

    print(f"Found {len(rater_files)} rater file(s): {list(rater_files.keys())}\n")

    rater_scores = {rid: load_rater_scores(path) for rid, path in rater_files.items()}
    judge_scores = load_judge_scores()

    report = {"raters_found": list(rater_files.keys()), "human_vs_judge": {}, "inter_rater": {}}

    print("=== Human vs. LLM Judge Agreement (per rater) ===\n")
    for rid, scores in rater_scores.items():
        result = compute_agreement(scores, judge_scores, f"Rater{rid}", "Judge")
        if result is None:
            print(f"Rater {rid}: no comparable scored items found.")
            continue
        report["human_vs_judge"][f"rater{rid}"] = result
        print(f"Rater {rid} vs Judge: n={result['n_criteria_compared']}  "
              f"exact_match={result['exact_match_rate']*100:.1f}%  "
              f"within_1={result['within_1_point_rate']*100:.1f}%  "
              f"correlation={result['pearson_correlation']}  "
              f"kappa={result['cohens_kappa']}  "
              f"weighted_kappa={result['cohens_kappa_quadratic_weighted']}")

    if len(rater_scores) >= 2:
        print("\n=== Inter-Rater Agreement (human vs. human) ===\n")
        for rid_a, rid_b in combinations(rater_scores.keys(), 2):
            result = compute_agreement(rater_scores[rid_a], rater_scores[rid_b], f"Rater{rid_a}", f"Rater{rid_b}")
            if result is None:
                print(f"Rater {rid_a} vs Rater {rid_b}: no shared scored items found.")
                continue
            key = f"rater{rid_a}_vs_rater{rid_b}"
            report["inter_rater"][key] = result
            print(f"Rater {rid_a} vs Rater {rid_b}: n={result['n_criteria_compared']}  "
                  f"exact_match={result['exact_match_rate']*100:.1f}%  "
                  f"correlation={result['pearson_correlation']}  "
                  f"kappa={result['cohens_kappa']}  "
                  f"weighted_kappa={result['cohens_kappa_quadratic_weighted']}")
    else:
        print("\nOnly 1 rater found - inter-rater agreement requires 2+ raters. "
              "Set NUM_RATERS >= 2 in build_human_validation_materials.py to enable this.")

    os.makedirs(HUMAN_VALIDATION_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
