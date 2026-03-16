"""
Build MATHEMATICAL_MODEL.docx from MATHEMATICAL_MODEL.md
Uses python-docx with proper formatting: headings, equations, tables, references.
"""
import re
import os
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

HERE = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(HERE, 'MATHEMATICAL_MODEL.md')
DOCX_PATH = os.path.join(HERE, 'MATHEMATICAL_MODEL.docx')


def set_style_defaults(doc):
    """Configure document-wide styles."""
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Cambria'
    font.size = Pt(11)
    pf = style.paragraph_format
    pf.space_after = Pt(6)
    pf.space_before = Pt(0)
    pf.line_spacing = 1.15

    for level in range(1, 4):
        hstyle = doc.styles[f'Heading {level}']
        hstyle.font.color.rgb = RGBColor(0, 51, 102)
        hstyle.font.name = 'Calibri'
        if level == 1:
            hstyle.font.size = Pt(16)
            hstyle.paragraph_format.space_before = Pt(24)
        elif level == 2:
            hstyle.font.size = Pt(14)
            hstyle.paragraph_format.space_before = Pt(18)
        else:
            hstyle.font.size = Pt(12)
            hstyle.paragraph_format.space_before = Pt(12)


def add_equation(doc, latex_text):
    """Add a centered equation paragraph with the LaTeX rendered as italic text."""
    # Clean up LaTeX for display
    eq = latex_text.strip()
    # Remove display math delimiters
    eq = re.sub(r'^\$\$\s*', '', eq)
    eq = re.sub(r'\s*\$\$$', '', eq)
    eq = eq.strip()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(eq)
    run.font.name = 'Cambria Math'
    run.font.size = Pt(11)
    run.italic = True
    return p


def process_inline_formatting(paragraph, text):
    """Add text to paragraph with bold/italic/math inline formatting."""
    # Split on bold, italic, and inline math markers
    parts = re.split(r'(\$[^$]+\$|\*\*[^*]+\*\*|\*[^*]+\*)', text)
    for part in parts:
        if not part:
            continue
        if part.startswith('**') and part.endswith('**'):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif part.startswith('*') and part.endswith('*') and not part.startswith('**'):
            run = paragraph.add_run(part[1:-1])
            run.italic = True
        elif part.startswith('$') and part.endswith('$'):
            math_text = part[1:-1]
            run = paragraph.add_run(math_text)
            run.font.name = 'Cambria Math'
            run.italic = True
        else:
            paragraph.add_run(part)


def add_table_from_rows(doc, header_row, data_rows):
    """Add a formatted table to the document."""
    n_cols = len(header_row)
    table = doc.add_table(rows=1 + len(data_rows), cols=n_cols)
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header
    for j, cell_text in enumerate(header_row):
        cell = table.rows[0].cells[j]
        cell.text = ''
        p = cell.paragraphs[0]
        process_inline_formatting(p, cell_text.strip())
        for run in p.runs:
            run.bold = True
            run.font.size = Pt(10)

    # Data rows
    for i, row_data in enumerate(data_rows):
        for j, cell_text in enumerate(row_data):
            if j >= n_cols:
                break
            cell = table.rows[i + 1].cells[j]
            cell.text = ''
            p = cell.paragraphs[0]
            process_inline_formatting(p, cell_text.strip())
            for run in p.runs:
                run.font.size = Pt(10)

    doc.add_paragraph()  # spacing after table


def parse_table_line(line):
    """Parse a markdown table row into cells."""
    cells = line.strip().strip('|').split('|')
    return [c.strip() for c in cells]


def is_separator_row(line):
    """Check if this is a markdown table separator (|---|---|)."""
    stripped = line.strip().strip('|')
    parts = stripped.split('|')
    return all(re.match(r'^[\s\-:]+$', p) for p in parts)


def build_document():
    with open(MD_PATH, 'r') as f:
        lines = f.readlines()

    doc = Document()

    # Page setup
    section = doc.sections[0]
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.54)
    section.right_margin = Cm(2.54)

    set_style_defaults(doc)

    # Title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(4)
    run = title.add_run('Mathematical Framework for Cell-Driven\nGranular Scaffold Remodeling')
    run.bold = True
    run.font.size = Pt(18)
    run.font.name = 'Calibri'
    run.font.color.rgb = RGBColor(0, 51, 102)

    # Subtitle
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(12)
    run = subtitle.add_run('GELLS-DEM V1.7 -- March 2026')
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(100, 100, 100)

    # Process the markdown
    i = 0
    # Skip the title lines (already handled)
    while i < len(lines) and not lines[i].startswith('## 1.'):
        i += 1

    in_display_math = False
    math_buffer = []
    table_buffer_header = None
    table_buffer_rows = []
    in_table = False
    paragraph_buffer = []
    list_items = []

    def flush_paragraph():
        nonlocal paragraph_buffer
        if paragraph_buffer:
            text = ' '.join(paragraph_buffer)
            text = text.strip()
            if text:
                p = doc.add_paragraph()
                process_inline_formatting(p, text)
            paragraph_buffer = []

    def flush_table():
        nonlocal table_buffer_header, table_buffer_rows, in_table
        if table_buffer_header and table_buffer_rows:
            add_table_from_rows(doc, table_buffer_header, table_buffer_rows)
        table_buffer_header = None
        table_buffer_rows = []
        in_table = False

    def flush_list():
        nonlocal list_items
        for item in list_items:
            p = doc.add_paragraph(style='List Bullet')
            process_inline_formatting(p, item)
        list_items = []

    while i < len(lines):
        line = lines[i]
        raw = line.rstrip('\n')

        # Display math block
        if raw.strip().startswith('$$') and not in_display_math:
            flush_paragraph()
            flush_list()
            if raw.strip().endswith('$$') and len(raw.strip()) > 4:
                # Single-line display math
                add_equation(doc, raw.strip())
            else:
                in_display_math = True
                math_buffer = [raw.strip()]
            i += 1
            continue

        if in_display_math:
            math_buffer.append(raw.strip())
            if '$$' in raw:
                eq_text = ' '.join(math_buffer)
                add_equation(doc, eq_text)
                in_display_math = False
                math_buffer = []
            i += 1
            continue

        # Horizontal rule
        if raw.strip() == '---':
            flush_paragraph()
            flush_list()
            flush_table()
            i += 1
            continue

        # Headings
        heading_match = re.match(r'^(#{1,3})\s+(.*)', raw)
        if heading_match:
            flush_paragraph()
            flush_list()
            flush_table()
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            h = doc.add_heading(text, level=level)
            i += 1
            continue

        # Table detection
        if '|' in raw and not raw.strip().startswith('$$'):
            cells = parse_table_line(raw)
            if len(cells) >= 2:
                if not in_table:
                    flush_paragraph()
                    flush_list()
                    # Start new table
                    table_buffer_header = cells
                    in_table = True
                    i += 1
                    continue
                elif is_separator_row(raw):
                    # Skip separator
                    i += 1
                    continue
                else:
                    table_buffer_rows.append(cells)
                    i += 1
                    continue

        # If we were in a table and now we're not, flush
        if in_table and '|' not in raw:
            flush_table()

        # Blank line
        if raw.strip() == '':
            flush_paragraph()
            flush_list()
            i += 1
            continue

        # Numbered list
        num_match = re.match(r'^(\d+)\.\s+(.*)', raw.strip())
        if num_match:
            flush_paragraph()
            text = num_match.group(2)
            p = doc.add_paragraph(style='List Number')
            process_inline_formatting(p, text)
            i += 1
            continue

        # Bullet list
        bullet_match = re.match(r'^[-*]\s+(.*)', raw.strip())
        if bullet_match:
            flush_paragraph()
            text = bullet_match.group(1)
            list_items.append(text)
            i += 1
            continue

        # Regular paragraph text
        if list_items and not bullet_match:
            flush_list()

        # Check if this is a continuation of indented content under a bullet
        if raw.startswith('  ') and list_items:
            list_items[-1] += ' ' + raw.strip()
            i += 1
            continue

        paragraph_buffer.append(raw.strip())
        i += 1

    # Final flush
    flush_paragraph()
    flush_list()
    flush_table()

    # Add page break before references
    doc.add_page_break()

    doc.save(DOCX_PATH)
    print(f'Saved: {DOCX_PATH}')
    print(f'  Pages: ~{len(doc.paragraphs) // 40 + 1} (estimated)')


if __name__ == '__main__':
    build_document()
