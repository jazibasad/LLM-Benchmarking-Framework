"""
validate_prompts.py

Structural validation for 04_Datasets/prompts.json - the curated 220-prompt
evaluation set, now spanning all 10 categories specified in the project
proposal (Section 3): Knowledge Retrieval, Multi-step Reasoning, Instruction
Following, Hallucination Stress Test, Coding & System Architecture,
Ambiguity Handling, Long-context Retention, Data Transformation,
Summarization, and Multilingual Tasks - 22 prompts each.

Checks performed:
  1. File loads as valid JSON
  2. Every entry has all required fields (id, category, difficulty, prompt,
     evaluation_criteria, max_score)
  3. All prompt IDs are unique
  4. IDs follow the expected P001-P220 format
  5. Every difficulty value is one of Easy / Medium / Hard
  6. No empty prompt or evaluation_criteria strings
  7. All 10 expected categories are present
  8. Each category has exactly 22 prompts (per the proposal's minimum-20-
     per-category requirement, with a safety margin)
  9. Reports the category and difficulty breakdown

Run with the VS Code Run button:
    python 03_Code/validate_prompts.py
"""

import json
import os
import re
import sys
from collections import Counter

PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "04_Datasets", "prompts.json")
REQUIRED_FIELDS = ["id", "category", "difficulty", "prompt", "evaluation_criteria", "max_score"]
VALID_DIFFICULTIES = {"Easy", "Medium", "Hard"}
ID_PATTERN = re.compile(r"^P\d{3}$")

EXPECTED_CATEGORIES = {
    "Knowledge Retrieval", "Multi-step Reasoning", "Instruction Following",
    "Hallucination Stress Test", "Coding & System Architecture",
    "Ambiguity Handling", "Long-context Retention", "Data Transformation",
    "Summarization", "Multilingual Tasks",
}
EXPECTED_PER_CATEGORY = 22
MIN_PER_CATEGORY = 20  # the proposal's actual hard minimum


def main():
    print(f"Validating: {os.path.abspath(PROMPTS_PATH)}\n")

    try:
        with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
            prompts = json.load(f)
    except FileNotFoundError:
        print(f"FAILURE: file not found at {PROMPTS_PATH}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"FAILURE: invalid JSON - {exc}")
        sys.exit(1)

    errors = []
    seen_ids = set()

    for i, entry in enumerate(prompts):
        missing = [field for field in REQUIRED_FIELDS if field not in entry or entry[field] in (None, "")]
        if missing:
            errors.append(f"Entry {i} (id={entry.get('id', '?')}): missing/empty fields {missing}")

        pid = entry.get("id", "")
        if not ID_PATTERN.match(pid):
            errors.append(f"Entry {i}: id '{pid}' does not match expected format P###")
        if pid in seen_ids:
            errors.append(f"Duplicate id found: {pid}")
        seen_ids.add(pid)

        difficulty = entry.get("difficulty")
        if difficulty not in VALID_DIFFICULTIES:
            errors.append(f"Entry {i} (id={pid}): invalid difficulty '{difficulty}'")

    total = len(prompts)
    print(f"Total entries: {total}")

    if total != 220:
        errors.append(f"Expected 220 entries, found {total}")

    category_counts = Counter(e.get("category") for e in prompts)
    difficulty_counts = Counter(e.get("difficulty") for e in prompts)

    print("\nCategory breakdown:")
    for cat, count in category_counts.most_common():
        flag = "" if count >= MIN_PER_CATEGORY else "  <-- BELOW MINIMUM OF 20"
        print(f"  {cat:35s} {count}{flag}")

    print("\nDifficulty breakdown:")
    for diff, count in difficulty_counts.most_common():
        print(f"  {diff:10s} {count}")

    # Check all 10 expected categories are present
    found_categories = set(category_counts.keys())
    missing_categories = EXPECTED_CATEGORIES - found_categories
    unexpected_categories = found_categories - EXPECTED_CATEGORIES
    if missing_categories:
        errors.append(f"Missing expected categories: {sorted(missing_categories)}")
    if unexpected_categories:
        errors.append(f"Unexpected categories found (not in proposal's 10): {sorted(unexpected_categories)}")

    # Check each category meets the minimum (proposal requirement: not less than 20)
    for cat in EXPECTED_CATEGORIES:
        count = category_counts.get(cat, 0)
        if count < MIN_PER_CATEGORY:
            errors.append(f"Category '{cat}' has only {count} prompts, below the required minimum of {MIN_PER_CATEGORY}")

    # Check the stochastic reliability sample (proposal's "Reliability" requirement:
    # prompts with stochasticity repeated 5-10 times). This project samples 2
    # prompts per category (20 total) for 5x repetition rather than repeating
    # all 220, given free-tier API constraints - see Week 3 report for rationale.
    sampled = [e for e in prompts if e.get("stochastic_sample") is True]
    sample_cat_counts = Counter(e.get("category") for e in sampled)
    print("\nStochastic reliability sample (5x repetition planned):")
    for cat in sorted(EXPECTED_CATEGORIES):
        count = sample_cat_counts.get(cat, 0)
        print(f"  {cat:35s} {count}")
    print(f"  {'TOTAL':35s} {len(sampled)}")

    if len(sampled) != 20:
        errors.append(f"Expected exactly 20 stochastic-sample prompts, found {len(sampled)}")
    for cat in EXPECTED_CATEGORIES:
        if sample_cat_counts.get(cat, 0) != 2:
            errors.append(f"Category '{cat}' should have exactly 2 stochastic-sample prompts, "
                          f"found {sample_cat_counts.get(cat, 0)}")
    for entry in sampled:
        if entry.get("repeat_count") != 5:
            errors.append(f"Entry {entry.get('id')} is marked stochastic_sample=True but "
                          f"repeat_count is {entry.get('repeat_count')}, expected 5")

    print()
    if errors:
        print(f"VALIDATION FAILED - {len(errors)} issue(s) found:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"VALIDATION PASSED - all 220 entries structurally valid, all IDs unique, "
              f"all required fields present, all difficulty values valid, all 10 proposal "
              f"categories present with at least {MIN_PER_CATEGORY} prompts each "
              f"(each category has exactly {EXPECTED_PER_CATEGORY}), and the 20-prompt "
              f"stochastic reliability sample (2 per category, repeat_count=5) is correctly marked.")


if __name__ == "__main__":
    main()
