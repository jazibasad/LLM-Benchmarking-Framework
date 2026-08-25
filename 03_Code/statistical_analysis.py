"""
statistical_analysis.py

Computes the remaining Statistical Analysis measures required by the
project proposal: Confidence intervals, Hypothesis testing, Effect size,
Correlation, Regression. (Inter-rater agreement was completed in Week 6;
Reliability - mean/median/stdev/CV - is computed separately by
compute_reliability_statistics.py.)

Also directly tests the project's three research hypotheses using this
same real scored data:
  H1: Multi-step Reasoning scores differ from Knowledge Retrieval scores
  H2: Instruction Following scores decrease as difficulty increases
  H3: Hallucination Stress Test scores differ significantly BETWEEN providers
      (a non-uniform pattern, not all providers hallucinating equally)

All statistics are implemented in pure Python (no external dependencies)
and were cross-validated against scipy.stats during development to confirm
correctness.

Uses each prompt's FIRST scored run only (for the 20 stochastic-sample
prompts) plus all 200 standard prompts' single run - i.e. one score per
prompt per provider, 220 x 3 = 660 data points, matching the main scored
dataset. The 5-run variance itself is analyzed separately in
compute_reliability_statistics.py.

OUTPUT: 05_Logs_Results/Statistical_Analysis/statistical_analysis.json

Run with the VS Code Run button:
    python 03_Code/statistical_analysis.py
"""

import json
import os
import glob
import re
import statistics
import math

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROMPTS_PATH = os.path.join(BASE_DIR, "04_Datasets", "prompts.json")
SCORED_RESULTS_DIR = os.path.join(BASE_DIR, "05_Logs_Results", "Scored_Results")
OUTPUT_DIR = os.path.join(BASE_DIR, "05_Logs_Results", "Statistical_Analysis")
WEEK6_AGREEMENT_PATH = os.path.join(BASE_DIR, "05_Logs_Results", "Survey", "agreement_report.json")

PROVIDERS = {
    "gemini": os.path.join(SCORED_RESULTS_DIR, "Gemini_Scores"),
    "mistral": os.path.join(SCORED_RESULTS_DIR, "Mistral_Scores"),
    "groq": os.path.join(SCORED_RESULTS_DIR, "Groq_Scores"),
}

DIFFICULTY_RANK = {"Easy": 1, "Medium": 2, "Hard": 3}


# ---------------------------------------------------------------------------
# Core statistics (pure Python, cross-validated against scipy during dev)
# ---------------------------------------------------------------------------

def confidence_interval_95(scores):
    """95% confidence interval for the mean, using the t-distribution.
    Uses a fixed t-critical-value lookup for common sample sizes rather than
    requiring scipy, falling back to the normal approximation (1.96) for
    larger samples where t and normal are nearly identical anyway."""
    n = len(scores)
    if n < 2:
        return None
    mean = statistics.mean(scores)
    stdev = statistics.stdev(scores)
    se = stdev / math.sqrt(n)

    # t-critical values (two-tailed, 95%) for small df; normal approx for large n
    t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
               7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086,
               30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}
    df = n - 1
    if df in t_table:
        t_crit = t_table[df]
    elif df > 120:
        t_crit = 1.96
    else:
        # linear interpolation between nearest known df values
        keys = sorted(t_table.keys())
        lower = max(k for k in keys if k <= df)
        upper = min(k for k in keys if k >= df)
        if lower == upper:
            t_crit = t_table[lower]
        else:
            frac = (df - lower) / (upper - lower)
            t_crit = t_table[lower] + frac * (t_table[upper] - t_table[lower])

    margin = t_crit * se
    return {"mean": round(mean, 4), "margin_of_error": round(margin, 4),
            "ci_lower": round(mean - margin, 4), "ci_upper": round(mean + margin, 4), "n": n}


def welch_t_test(sample_a, sample_b):
    """Welch's t-test (does not assume equal variances) - two-tailed.
    Returns t-statistic and an approximate p-value via a standard normal
    approximation to the t-distribution (accurate to ~2 decimal places for
    the sample sizes in this project, which is sufficient for this use)."""
    n1, n2 = len(sample_a), len(sample_b)
    if n1 < 2 or n2 < 2:
        return None
    mean1, mean2 = statistics.mean(sample_a), statistics.mean(sample_b)
    var1, var2 = statistics.variance(sample_a), statistics.variance(sample_b)

    se = math.sqrt(var1 / n1 + var2 / n2)
    if se == 0:
        return None
    t_stat = (mean1 - mean2) / se

    # Welch-Satterthwaite degrees of freedom
    df = ((var1 / n1 + var2 / n2) ** 2) / (
        ((var1 / n1) ** 2) / (n1 - 1) + ((var2 / n2) ** 2) / (n2 - 1)
    )

    p_value = _t_dist_two_tailed_p(abs(t_stat), df)

    return {"t_statistic": round(t_stat, 4), "degrees_of_freedom": round(df, 2),
            "p_value": round(p_value, 4), "mean_a": round(mean1, 4), "mean_b": round(mean2, 4),
            "significant_at_0.05": p_value < 0.05}


def _t_dist_two_tailed_p(t_abs, df):
    """
    Exact two-tailed p-value for the Student's t-distribution, computed via
    the regularized incomplete beta function (the standard closed-form
    relationship between the t-distribution's CDF and the beta function -
    the same approach used internally by most statistics libraries), rather
    than a normal approximation. This avoids the ~0.01-0.02 error margin a
    normal approximation introduces at small-to-moderate degrees of
    freedom, which could otherwise flip a borderline significance
    conclusion right at the p=0.05 threshold.
    """
    x = df / (df + t_abs * t_abs)
    # I_x(df/2, 1/2) is the standard closed-form relationship that gives
    # the two-tailed p-value DIRECTLY - no additional factor needed.
    p_two_tailed = _regularized_incomplete_beta(x, df / 2, 0.5)
    return p_two_tailed


def _regularized_incomplete_beta(x, a, b):
    """Regularized incomplete beta function I_x(a, b), via a continued
    fraction expansion (Numerical Recipes' betacf algorithm) - a standard,
    accurate, dependency-free numerical method."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1 - x))

    if x < (a + 1) / (a + b + 2):
        return front * _betacf(x, a, b) / a
    else:
        return 1 - front * _betacf(1 - x, b, a) / b


def _betacf(x, a, b, max_iter=200, eps=1e-10):
    """Continued fraction component of the incomplete beta function."""
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < eps:
            break

    return h


def cohens_d(sample_a, sample_b):
    """Effect size for the difference between two group means, in units of
    pooled standard deviation. |d| < 0.2 = negligible, ~0.5 = medium,
    > 0.8 = large effect."""
    n1, n2 = len(sample_a), len(sample_b)
    if n1 < 2 or n2 < 2:
        return None
    mean1, mean2 = statistics.mean(sample_a), statistics.mean(sample_b)
    var1, var2 = statistics.variance(sample_a), statistics.variance(sample_b)
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return None
    return round((mean1 - mean2) / pooled_std, 4)


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


def simple_linear_regression(x, y):
    """y = intercept + slope * x, via ordinary least squares, plus R-squared."""
    n = len(x)
    if n < 2:
        return None
    mean_x, mean_y = sum(x) / n, sum(y) / n
    ss_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    ss_xx = sum((xi - mean_x) ** 2 for xi in x)
    if ss_xx == 0:
        return None
    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x

    y_pred = [intercept + slope * xi for xi in x]
    ss_res = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred))
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else None

    return {"slope": round(slope, 4), "intercept": round(intercept, 4),
            "r_squared": round(r_squared, 4) if r_squared is not None else None, "n": n}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_first_run_scores():
    """Loads one score per prompt per provider (first available run), with
    category/difficulty metadata - the main analysis dataset."""
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts = {p["id"]: p for p in json.load(f)}

    data = {provider: [] for provider in PROVIDERS}

    for provider, score_dir in PROVIDERS.items():
        pattern = os.path.join(score_dir, "*_scored.json")
        files = glob.glob(pattern)

        by_prompt = {}
        for filepath in files:
            filename = os.path.basename(filepath)
            match = re.match(r"^\w+_(P\d{3})(?:_run(\d+))?_scored\.json$", filename)
            if not match:
                continue
            pid = match.group(1)
            run_num = int(match.group(2)) if match.group(2) else 1
            if pid not in by_prompt or run_num < by_prompt[pid][0]:
                with open(filepath, "r", encoding="utf-8") as f:
                    record = json.load(f)
                by_prompt[pid] = (run_num, record)

        for pid, (run_num, record) in by_prompt.items():
            meta = prompts.get(pid, {})
            data[provider].append({
                "id": pid, "category": meta.get("category"), "difficulty": meta.get("difficulty"),
                "total_score": record.get("total_score"), "max_score": record.get("max_score"),
                "latency_seconds": record.get("latency_seconds"),
            })

    return data


def load_inter_rater_agreement():
    """
    Loads Week 6's human-vs-judge and inter-rater agreement results
    (exact match, within-1-point, correlation, Cohen's Kappa - both
    standard and quadratic-weighted). Included here because the proposal
    lists "Inter-rater agreement" under the same Statistical Analysis
    section as confidence intervals, hypothesis testing, correlation, and
    regression - this consolidates all of Week 7's required statistics
    into one place rather than leaving it siloed in Week 6's own folder.
    """
    if not os.path.exists(WEEK6_AGREEMENT_PATH):
        print(f"  NOTE: Week 6 agreement report not found at {WEEK6_AGREEMENT_PATH} - "
              f"run compare_human_vs_judge.py first if you want inter-rater agreement included.")
        return None
    with open(WEEK6_AGREEMENT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    data = load_all_first_run_scores()
    report = {"confidence_intervals": {}, "hypothesis_tests": {}, "correlation": {}, "regression": {}}

    print("=== Confidence Intervals (95%) on Total Score, per Provider ===\n")
    for provider, records in data.items():
        scores = [r["total_score"] for r in records if r["total_score"] is not None]
        ci = confidence_interval_95(scores)
        report["confidence_intervals"][provider] = ci
        if ci:
            print(f"  {provider}: mean={ci['mean']}, 95% CI [{ci['ci_lower']}, {ci['ci_upper']}], n={ci['n']}")
        else:
            print(f"  {provider}: insufficient data (n={len(scores)})")

    print("\n=== H1: Multi-step Reasoning vs Knowledge Retrieval (per provider) ===\n")
    h1_results = {}
    for provider, records in data.items():
        reasoning = [r["total_score"] for r in records if r["category"] == "Multi-step Reasoning" and r["total_score"] is not None]
        retrieval = [r["total_score"] for r in records if r["category"] == "Knowledge Retrieval" and r["total_score"] is not None]
        test = welch_t_test(reasoning, retrieval)
        effect = cohens_d(reasoning, retrieval)
        h1_results[provider] = {"t_test": test, "cohens_d": effect}
        if test:
            print(f"  {provider}: reasoning_mean={test['mean_a']} retrieval_mean={test['mean_b']} "
                  f"p={test['p_value']} significant={test['significant_at_0.05']} d={effect}")
        else:
            print(f"  {provider}: insufficient data")
    report["hypothesis_tests"]["H1_reasoning_vs_retrieval"] = h1_results

    print("\n=== H2: Instruction Following score vs Difficulty (correlation, per provider) ===\n")
    h2_results = {}
    for provider, records in data.items():
        if_records = [r for r in records if r["category"] == "Instruction Following" and r["total_score"] is not None]
        difficulties = [DIFFICULTY_RANK[r["difficulty"]] for r in if_records]
        scores = [r["total_score"] for r in if_records]
        corr = pearson_correlation(difficulties, scores)
        h2_results[provider] = {"correlation_difficulty_vs_score": round(corr, 4) if corr is not None else None, "n": len(if_records)}
        print(f"  {provider}: correlation(difficulty, score) = {round(corr, 4) if corr is not None else None} (n={len(if_records)})")
    report["hypothesis_tests"]["H2_instruction_following_vs_difficulty"] = h2_results

    print("\n=== H3: Hallucination Stress Test - Provider Comparison ===\n")
    h3_scores = {provider: [r["total_score"] for r in records if r["category"] == "Hallucination Stress Test" and r["total_score"] is not None]
                 for provider, records in data.items()}
    h3_results = {}
    providers_list = list(h3_scores.keys())
    for i in range(len(providers_list)):
        for j in range(i + 1, len(providers_list)):
            p1, p2 = providers_list[i], providers_list[j]
            test = welch_t_test(h3_scores[p1], h3_scores[p2])
            effect = cohens_d(h3_scores[p1], h3_scores[p2])
            key = f"{p1}_vs_{p2}"
            h3_results[key] = {"t_test": test, "cohens_d": effect}
            if test:
                print(f"  {key}: p={test['p_value']} significant={test['significant_at_0.05']} d={effect}")
    report["hypothesis_tests"]["H3_hallucination_provider_comparison"] = h3_results

    print("\n=== Correlation: Difficulty vs Score, and Latency vs Score (overall, per provider) ===\n")
    for provider, records in data.items():
        valid = [r for r in records if r["total_score"] is not None and r["difficulty"] in DIFFICULTY_RANK]
        diffs = [DIFFICULTY_RANK[r["difficulty"]] for r in valid]
        scores = [r["total_score"] for r in valid]
        corr_diff = pearson_correlation(diffs, scores)

        valid_lat = [r for r in records if r["total_score"] is not None and r["latency_seconds"] is not None]
        latencies = [r["latency_seconds"] for r in valid_lat]
        scores_lat = [r["total_score"] for r in valid_lat]
        corr_lat = pearson_correlation(latencies, scores_lat)

        report["correlation"][provider] = {
            "difficulty_vs_score": round(corr_diff, 4) if corr_diff is not None else None,
            "latency_vs_score": round(corr_lat, 4) if corr_lat is not None else None,
        }
        print(f"  {provider}: corr(difficulty,score)={round(corr_diff,4) if corr_diff is not None else None}  "
              f"corr(latency,score)={round(corr_lat,4) if corr_lat is not None else None}")

    print("\n=== Regression: Predicting Score from Difficulty (overall, per provider) ===\n")
    for provider, records in data.items():
        valid = [r for r in records if r["total_score"] is not None and r["difficulty"] in DIFFICULTY_RANK]
        diffs = [DIFFICULTY_RANK[r["difficulty"]] for r in valid]
        scores = [r["total_score"] for r in valid]
        reg = simple_linear_regression(diffs, scores)
        report["regression"][provider] = reg
        if reg:
            print(f"  {provider}: score = {reg['intercept']} + {reg['slope']} * difficulty_rank, R^2={reg['r_squared']}")

    print("\n=== Inter-Rater Agreement (from Week 6) ===\n")
    inter_rater_data = load_inter_rater_agreement()
    report["inter_rater_agreement"] = inter_rater_data
    if inter_rater_data:
        print("  Human vs. Judge:")
        for rater, result in inter_rater_data.get("human_vs_judge", {}).items():
            print(f"    {rater}: exact_match={result['exact_match_rate']*100:.1f}%  "
                  f"kappa={result['cohens_kappa']}  weighted_kappa={result['cohens_kappa_quadratic_weighted']}")
        print("  Inter-Rater (human vs. human):")
        for pair, result in inter_rater_data.get("inter_rater", {}).items():
            print(f"    {pair}: exact_match={result['exact_match_rate']*100:.1f}%  "
                  f"kappa={result['cohens_kappa']}  weighted_kappa={result['cohens_kappa_quadratic_weighted']}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "statistical_analysis.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
