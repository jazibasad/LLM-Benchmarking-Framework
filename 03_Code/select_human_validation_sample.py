"""
select_human_validation_sample.py

Selects an independent, stratified random sample of prompts for Week 6's
Human-in-the-Loop validation, per the project proposal's requirement:
"Human-in-the-Loop: In Week 6, we will compare automated scores against
human-evaluated scores to validate our framework's accuracy."

METHODOLOGY: stratified by BOTH category AND difficulty - exactly 1 Easy,
1 Medium, and 1 Hard prompt is randomly selected from EACH of the 10
categories (30 prompts total, ~14% of the 220-prompt dataset). This
guarantees every difficulty tier is represented within every category,
rather than leaving difficulty coverage to chance as a category-only
stratification would.

The 20 stochastic reliability sample prompts (Week 3/5, used to measure
response VARIANCE across repeated runs) are EXPLICITLY EXCLUDED from the
eligible pool before sampling - not just avoided by chance - to guarantee
independence from a sample selected for a different purpose. This was
verified necessary: the stochastic sample does not have Easy or Medium
prompts in every category (2 categories have only Hard multi-step
prompts), so it could not support this stratification even if reused, and
reusing it would in any case bias the validation sample toward atypical,
open-ended prompts rather than being representative.

Fixed random seed (42) for reproducibility.

OUTPUT: 04_Datasets/human_validation_sample.json - the 30 selected prompt
IDs with their category, difficulty, and the random seed used.

Run with the VS Code Run button:
    python 03_Code/select_human_validation_sample.py
"""

import json
import os
import random
from collections import defaultdict

RANDOM_SEED = 42
DIFFICULTIES = ["Easy", "Medium", "Hard"]

PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "04_Datasets", "prompts.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "04_Datasets", "human_validation_sample.json")


def main():
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    stochastic_ids = {p["id"] for p in prompts if p.get("stochastic_sample")}
    eligible_prompts = [p for p in prompts if p["id"] not in stochastic_ids]

    # Stratify by (category, difficulty) - two-dimensional grouping
    by_cat_diff = defaultdict(list)
    for p in eligible_prompts:
        by_cat_diff[(p["category"], p["difficulty"])].append(p)

    rng = random.Random(RANDOM_SEED)

    categories = sorted({p["category"] for p in eligible_prompts})
    selected = []
    missing_strata = []

    for category in categories:
        for difficulty in DIFFICULTIES:
            pool = by_cat_diff.get((category, difficulty), [])
            if not pool:
                missing_strata.append(f"{category} / {difficulty}")
                continue
            chosen = rng.choice(pool)
            selected.append({
                "id": chosen["id"],
                "category": chosen["category"],
                "difficulty": chosen["difficulty"],
                "prompt": chosen["prompt"],
                "evaluation_criteria": chosen["evaluation_criteria"],
                "max_score": chosen["max_score"],
            })

    selected.sort(key=lambda p: int(p["id"][1:]))

    output = {
        "methodology": "Stratified random sample: 1 prompt per difficulty "
                        "(Easy, Medium, Hard) per category = 3 per category, "
                        "10 categories = 30 total. EXPLICITLY EXCLUDES the "
                        "20 stochastic reliability sample prompts from the "
                        "eligible pool (not just avoided by chance). Fixed "
                        "random seed for reproducibility.",
        "random_seed": RANDOM_SEED,
        "strata": "category x difficulty (10 x 3 = 30 strata, 1 prompt each)",
        "excluded_stochastic_sample_ids": sorted(stochastic_ids, key=lambda x: int(x[1:])),
        "missing_strata": missing_strata,  # should be empty - every category has all 3 difficulties among standard prompts
        "total_prompts_sampled": len(selected),
        "total_responses_for_validation": len(selected) * 3,  # x3 providers
        "sampled_prompt_ids": [p["id"] for p in selected],
        "overlap_with_stochastic_sample": sorted(
            {p["id"] for p in selected} & stochastic_ids
        ),  # should always be empty - included for transparency/verification
        "prompts": selected,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Stratified sample selected: {len(selected)} prompts "
          f"(1 per difficulty x 3 difficulties x 10 categories, seed={RANDOM_SEED})")
    print(f"Total responses for human validation: {len(selected) * 3} (across 3 providers)\n")

    if missing_strata:
        print(f"WARNING: {len(missing_strata)} category/difficulty combinations had no eligible "
              f"prompts and were skipped: {missing_strata}\n")

    print("Selected prompt IDs by category:")
    by_cat_selected = defaultdict(list)
    for p in selected:
        by_cat_selected[p["category"]].append(f"{p['id']}({p['difficulty'][0]})")
    for cat, ids in sorted(by_cat_selected.items()):
        print(f"  {cat:35s} {', '.join(ids)}")

    print(f"\nSaved to: {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
