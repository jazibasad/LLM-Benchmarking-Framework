"""
build_results_docx.py

Creates one human-readable Word document per provider (Gemini, Groq,
Mistral), each containing every prompt alongside its model response,
evaluation criteria, and max score in a single table - designed for a human
evaluator to sit down and score each response directly, rather than reading
raw JSON files.

Reads directly from the per-prompt JSON logs in 05_Logs_Results/<Provider>_Logs/
(the same source used by build_results_tables.py), so this can be run
independently at any time - it does not require build_results_tables.py to
have been run first.

OUTPUT FILES (written to 05_Logs_Results/Readable_Results/):
  - gemini_results.docx
  - groq_results.docx
  - mistral_results.docx

Each document is landscape-oriented with columns: ID, Category, Difficulty,
Prompt, Model Response, Evaluation Criteria, Max Score, Timestamp. Multi-line
content (code responses, multi-part evaluation criteria) renders with real,
visible line breaks rather than collapsing into one run-on paragraph.

Requires: pip install python-docx  (add to requirements.txt if not present)

Run with the VS Code Run button:
    python 03_Code/build_results_docx.py
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
OUTPUT_DIR = os.path.join(BASE_DIR, "05_Logs_Results", "Readable_Results")

PROVIDERS = {
    "gemini": {
        "log_dir": os.path.join(BASE_DIR, "05_Logs_Results", "Gemini_Logs"),
        "file_prefix": "gemini_",
        "output_name": "gemini_results.docx",
        "title": "Gemini - Prompts and Results",
    },
    "groq": {
        "log_dir": os.path.join(BASE_DIR, "05_Logs_Results", "Groq_Logs"),
        "file_prefix": "groq_",
        "output_name": "groq_results.docx",
        "title": "Groq - Prompts and Results",
    },
    "mistral": {
        "log_dir": os.path.join(BASE_DIR, "05_Logs_Results", "Mistral_Logs"),
        "file_prefix": "mistral_",
        "output_name": "mistral_results.docx",
        "title": "Mistral - Prompts and Results",
    },
}

COLUMNS = [
    ("ID", 0.5),
    ("Category", 1.1),
    ("Difficulty", 0.7),
    ("Prompt", 2.3),
    ("Model Response", 2.6),
    ("Evaluation Criteria", 2.3),
    ("Max Score", 0.6),
    ("Timestamp", 1.1),
]


def _prompt_id_sort_key(record):
    match = re.search(r"\d+", record.get("id", "0"))
    return int(match.group()) if match else 0


def _load_records(log_dir, prefix):
    pattern = os.path.join(log_dir, f"{prefix}*.json")
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


def _set_cell_shading(cell, hex_color):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), hex_color)
    cell._tc.get_or_add_tcPr().append(shd)


def _set_repeat_header(row):
    """Marks a table row to repeat as a header on every page."""
    tr = row._tr
    trPr = tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)


def _add_multiline_text(paragraph, text, font_size, bold=False):
    """
    Adds text to a paragraph with real, visible line breaks wherever the
    source text contains '\\n'. Word does NOT render raw newline characters
    inside a single run as visible line breaks - without this, multi-line
    content (evaluation criteria with several scored items, or code-heavy
    model responses) would collapse into one unreadable run-on paragraph.
    """
    lines = str(text).split("\n")
    for i, line in enumerate(lines):
        if i > 0:
            paragraph.add_run().add_break(WD_BREAK.LINE)
        run = paragraph.add_run(line)
        run.font.size = Pt(font_size)
        run.bold = bold


def build_doc_for_provider(name, config):
    records = _load_records(config["log_dir"], config["file_prefix"])

    doc = Document()

    # Landscape orientation for wide tables
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    new_width, new_height = section.page_height, section.page_width
    section.page_width, section.page_height = new_width, new_height
    section.left_margin = Inches(0.5)
    section.right_margin = Inches(0.5)
    section.top_margin = Inches(0.5)
    section.bottom_margin = Inches(0.5)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run(config["title"])
    run.bold = True
    run.font.size = Pt(18)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = subtitle.add_run(f"{len(records)} prompts and responses - generated for human evaluation")
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
        cell.text = ""
        p = cell.paragraphs[0]
        r = p.add_run(col_name)
        r.bold = True
        r.font.size = Pt(9)
        _set_cell_shading(cell, "D9E2F3")

    for record in records:
        row = table.add_row()
        values = [
            record.get("id", ""),
            record.get("category", ""),
            record.get("difficulty", ""),
            record.get("prompt", ""),
            record.get("response_text", ""),
            record.get("evaluation_criteria", ""),
            str(record.get("max_score", "")),
            record.get("timestamp_utc", ""),
        ]
        for i, (col_name, width) in enumerate(COLUMNS):
            cell = row.cells[i]
            cell.width = Inches(width)
            p = cell.paragraphs[0]
            _add_multiline_text(p, values[i], font_size=8.5)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, config["output_name"])
    doc.save(output_path)

    print(f"[{name}] {len(records)} rows written to {output_path}")
    return len(records)


def main():
    print(f"Building human-readable Word tables in: {os.path.abspath(OUTPUT_DIR)}\n")
    for name, config in PROVIDERS.items():
        build_doc_for_provider(name, config)
    print("\nDone. Open each .docx to review prompts, responses, and scoring criteria side by side.")


if __name__ == "__main__":
    main()