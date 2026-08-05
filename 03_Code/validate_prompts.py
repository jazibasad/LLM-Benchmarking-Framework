"""
validate_prompts.py

Structural validation for 04_Datasets/prompts.json - the curated 220-prompt
evaluation set. There is no automation/API code to unit-test in Week 3 (that
starts Week 4), but the dataset itself is this week's real deliverable, so
this script verifies it programmatically rather than just eyeballing it.

Checks performed:
  1. File loads as valid JSON
  2. Every entry has all required fields (id, category, difficulty, prompt,
     evaluation_criteria, max_score)
  3. All prompt IDs are unique
  4. IDs follow the expected P001-P220 format
  5. Every difficulty value is one of Easy / Medium / Hard
  6. No empty prompt or evaluation_criteria strings
  7. Reports the category and difficulty breakdown

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
        print(f"  {cat:35s} {count}")

    print("\nDifficulty breakdown:")
    for diff, count in difficulty_counts.most_common():
        print(f"  {diff:10s} {count}")

    print()
    if errors:
        print(f"VALIDATION FAILED - {len(errors)} issue(s) found:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("VALIDATION PASSED - all 220 entries structurally valid, all IDs unique, "
              "all required fields present, all difficulty values valid.")


if __name__ == "__main__":
    main()
