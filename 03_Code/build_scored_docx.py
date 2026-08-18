"""
build_scored_docx.py

Creates one professional, human-readable Word document per provider,
showing every prompt with its response, its per-criterion scores WITH
justifications, total score, and judge details - the scored equivalent of
Week 4's build_results_docx.py.

This is the document that answers "on what basis was this response scored
this way": every criterion is listed by name with its individual 0-5 score
and a one-sentence justification explaining the judge's reasoning.

Reads from 05_Logs_Results/Scored_Results/<Provider>_Scores/ directly (does
not require build_scored_tables.py to have been run first).

OUTPUT FILES (written to 05_Logs_Results/Scored_Results/Readable_Scores/):
  - gemini_scores.docx
  - mistral_scores.docx
  - groq_scores.docx

Requires: pip install python-docx

Run with the VS Code Run button:
    python 03_Code/build_scored_docx.py
"""

import json
import os
import re
import glob

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
SCORED_RESULTS_DIR = os.path.join(BASE_DIR, "05_Logs_Results", "Scored_Results")
OUTPUT_DIR = os.path.join(SCORED_RESULTS_DIR, "Readable_Scores")

PROVIDERS = {
    "gemini": {
        "score_dir": os.path.join(SCORED_RESULTS_DIR, "Gemini_Scores"),
        "file_prefix": "gemini_",
        "output_name": "gemini_scores.docx",
        "title": "Gemini - Scored Results",
    },
    "mistral": {
        "score_dir": os.path.join(SCORED_RESULTS_DIR, "Mistral_Scores"),
        "file_prefix": "mistral_",
        "output_name": "mistral_scores.docx",
        "title": "Mistral - Scored Results",
    },
    "groq": {
        "score_dir": os.path.join(SCORED_RESULTS_DIR, "Groq_Scores"),
        "file_prefix": "groq_",
        "output_name": "groq_scores.docx",
        "title": "Groq - Scored Results",
    },
}

COLUMNS = [
    ("ID", 0.5),
    ("Category", 1.0),
    ("Difficulty", 0.6),
    ("Prompt", 1.7),
    ("Model Response", 2.0),
    ("Criterion Scores & Justification", 3.2),
    ("Total", 0.5),
    ("Max", 0.4),
]


def _prompt_id_sort_key(record):
    match = re.search(r"\d+", record.get("id", "0"))
    return int(match.group()) if match else 0


def _load_records(score_dir, prefix):
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
    return records


def _format_criterion_scores(criterion_scores) -> str:
    """
    Turns the list of {name, score, justification} dicts into a readable
    multi-line string: this is the actual "on what basis" documentation -
    every criterion, its score, and the judge's reasoning, all visible
    together for each response.
    """
    lines = []
    for c in criterion_scores:
        name = c.get("name", "?")
        score = c.get("score", "?")
        justification = c.get("justification", "")
        lines.append(f"{name}: {score}/5")
        if justification:
            lines.append(f"   {justification}")
    return "\n".join(lines) if lines else "(no criterion scores recorded)"


def _set_cell_shading(cell, hex_color):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), hex_color)
    cell._tc.get_or_add_tcPr().append(shd)


def _set_repeat_header(row):
    tr = row._tr
    trPr = tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)


def _add_multiline_text(paragraph, text, font_size, bold=False, color=None):
    """Real, visible line breaks for multi-line content - Word does not
    render raw '\\n' characters as line breaks on its own."""
    lines = str(text).split("\n")
    for i, line in enumerate(lines):
        if i > 0:
            paragraph.add_run().add_break(WD_BREAK.LINE)
        run = paragraph.add_run(line)
        run.font.size = Pt(font_size)
        run.bold = bold
        if color:
            run.font.color.rgb = color


def build_doc_for_provider(name, config):
    records = _load_records(config["score_dir"], config["file_prefix"])

    doc = Document()

    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    new_width, new_height = section.page_height, section.page_width
    section.page_width, section.page_height = new_width, new_height
    section.left_margin = Inches(0.4)
    section.right_margin = Inches(0.4)
    section.top_margin = Inches(0.5)
    section.bottom_margin = Inches(0.5)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run(config["title"])
    run.bold = True
    run.font.size = Pt(18)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = subtitle.add_run(
        f"{len(records)} scored responses - LLM judge: openrouter/free - "
        f"each criterion scored 0-5 with judge justification"
    )
    sub_run.italic = True
    sub_run.font.size = Pt(10)
    sub_run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    table = doc.add_table(rows=1, cols=len(COLUMNS))
    table.style = "Table Grid"
    table.autofit = False

    header_row = table.rows[0]
    _set_repeat_header(header_row)
    for i, (col_name, width) in enumerate(COLUMNS):
        cell = header_row.cells[i]
        cell.width = Inches(width)
        p = cell.paragraphs[0]
        r = p.add_run(col_name)
        r.bold = True
        r.font.size = Pt(9)
        _set_cell_shading(cell, "D9E2F3")

    for record in records:
        row = table.add_row()

        total_score = record.get("total_score", "")
        max_score = record.get("max_score", "")
        criterion_text = _format_criterion_scores(record.get("criterion_scores", []))

        values = [
            record.get("id", ""),
            record.get("category", ""),
            record.get("difficulty", ""),
            record.get("prompt", ""),
            record.get("response_text", ""),
            criterion_text,
            str(total_score),
            str(max_score),
        ]
        for i, (col_name, width) in enumerate(COLUMNS):
            cell = row.cells[i]
            cell.width = Inches(width)
            p = cell.paragraphs[0]
            if col_name == "Total":
                _add_multiline_text(p, values[i], font_size=9, bold=True)
            else:
                _add_multiline_text(p, values[i], font_size=8)

        # Highlight low scores (potential concern) for quick visual scanning
        try:
            if max_score and int(total_score) < int(max_score) * 0.5:
                _set_cell_shading(row.cells[6], "FCE4E4")
        except (ValueError, TypeError):
            pass

    # Footer note explaining methodology, for anyone opening this file cold
    doc.add_paragraph()
    footer = doc.add_paragraph()
    footer_run = footer.add_run(
        "Scoring methodology: each response was scored by an LLM judge (openrouter/free) "
        "against the SPECIFIC evaluation criteria written for that exact prompt (not a "
        "generic rubric). Each criterion was scored independently on a 0-5 scale with a "
        "justification, and the total score is the sum of all criterion scores for that "
        "prompt, computed programmatically rather than trusted from the judge's own text."
    )
    footer_run.italic = True
    footer_run.font.size = Pt(8)
    footer_run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, config["output_name"])
    doc.save(output_path)

    print(f"[{name}] {len(records)} scored rows written to {output_path}")
    return len(records)


def main():
    print(f"Building human-readable scored tables in: {os.path.abspath(OUTPUT_DIR)}\n")
    for name, config in PROVIDERS.items():
        build_doc_for_provider(name, config)
    print("\nDone. Each document shows every response's per-criterion scores and judge justifications.")


if __name__ == "__main__":
    main()
