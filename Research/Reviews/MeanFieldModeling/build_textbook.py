#!/usr/bin/env python3
"""
Build a Word document (.docx) formatted as a textbook from the
Mean Field Modeling Guide content + generated figures.

Run from repo root:
    python CodeLog/MeanFieldModeling/build_textbook.py

Produces:
    CodeLog/MeanFieldModeling/Mean_Field_Modeling_Guide.docx
"""

import os, sys, textwrap
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml

HERE = Path(__file__).resolve().parent
FIGURES = HERE / 'figures'
OUTPATH = HERE / 'Mean_Field_Modeling_Guide.docx'

# ── Helpers ──────────────────────────────────────────────────────────

def _set_cell_shading(cell, color_hex):
    """Set background shading on a table cell."""
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def _add_bottom_border(paragraph, color="4472C4", size=6):
    """Add a colored bottom border to a paragraph (for chapter titles)."""
    pPr = paragraph._p.get_or_add_pPr()
    pBdr = parse_xml(
        f'<w:pBdr {nsdecls("w")}>'
        f'  <w:bottom w:val="single" w:sz="{size}" w:space="1" w:color="{color}"/>'
        f'</w:pBdr>'
    )
    pPr.append(pBdr)


def _set_repeat_header_row(row):
    """Mark a table row to repeat as header on each page."""
    trPr = row._tr.get_or_add_trPr()
    trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))


# ── Style Setup ──────────────────────────────────────────────────────

def setup_styles(doc):
    """Configure all custom styles for the textbook."""
    styles = doc.styles

    # -- Default paragraph font --
    style_normal = styles['Normal']
    font = style_normal.font
    font.name = 'Cambria'
    font.size = Pt(11)
    font.color.rgb = RGBColor(0x2D, 0x2D, 0x2D)
    pf = style_normal.paragraph_format
    pf.space_after = Pt(6)
    pf.space_before = Pt(0)
    pf.line_spacing = 1.15

    # -- Heading 1: Chapter title --
    h1 = styles['Heading 1']
    h1.font.name = 'Calibri'
    h1.font.size = Pt(26)
    h1.font.bold = True
    h1.font.color.rgb = RGBColor(0x1B, 0x3A, 0x5C)
    h1.paragraph_format.space_before = Pt(36)
    h1.paragraph_format.space_after = Pt(12)
    h1.paragraph_format.page_break_before = True

    # -- Heading 2: Section --
    h2 = styles['Heading 2']
    h2.font.name = 'Calibri'
    h2.font.size = Pt(18)
    h2.font.bold = True
    h2.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)
    h2.paragraph_format.space_before = Pt(18)
    h2.paragraph_format.space_after = Pt(6)

    # -- Heading 3: Subsection --
    h3 = styles['Heading 3']
    h3.font.name = 'Calibri'
    h3.font.size = Pt(14)
    h3.font.bold = True
    h3.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)
    h3.paragraph_format.space_before = Pt(12)
    h3.paragraph_format.space_after = Pt(4)

    # -- Code block style --
    if 'Code Block' not in [s.name for s in styles]:
        code_style = styles.add_style('Code Block', WD_STYLE_TYPE.PARAGRAPH)
        code_style.font.name = 'Consolas'
        code_style.font.size = Pt(9)
        code_style.font.color.rgb = RGBColor(0x1E, 0x1E, 0x1E)
        code_style.paragraph_format.space_before = Pt(2)
        code_style.paragraph_format.space_after = Pt(2)
        code_style.paragraph_format.line_spacing = 1.0
        code_style.paragraph_format.left_indent = Cm(0.5)

    # -- Key concept callout --
    if 'Callout' not in [s.name for s in styles]:
        callout = styles.add_style('Callout', WD_STYLE_TYPE.PARAGRAPH)
        callout.font.name = 'Cambria'
        callout.font.size = Pt(11)
        callout.font.italic = True
        callout.font.color.rgb = RGBColor(0x1B, 0x3A, 0x5C)
        callout.paragraph_format.left_indent = Cm(1.0)
        callout.paragraph_format.right_indent = Cm(1.0)
        callout.paragraph_format.space_before = Pt(8)
        callout.paragraph_format.space_after = Pt(8)
        callout.paragraph_format.line_spacing = 1.15

    # -- Figure caption --
    if 'Caption' not in [s.name for s in styles]:
        cap = styles.add_style('Caption', WD_STYLE_TYPE.PARAGRAPH)
    else:
        cap = styles['Caption']
    cap.font.name = 'Calibri'
    cap.font.size = Pt(9.5)
    cap.font.italic = True
    cap.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(4)
    cap.paragraph_format.space_after = Pt(12)

    # -- Equation style --
    if 'Equation' not in [s.name for s in styles]:
        eq = styles.add_style('Equation', WD_STYLE_TYPE.PARAGRAPH)
        eq.font.name = 'Cambria Math'
        eq.font.size = Pt(12)
        eq.font.color.rgb = RGBColor(0x1E, 0x1E, 0x1E)
        eq.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        eq.paragraph_format.space_before = Pt(8)
        eq.paragraph_format.space_after = Pt(8)

    return doc


# ── Content builders ─────────────────────────────────────────────────

def add_para(doc, text, style='Normal', bold=False):
    p = doc.add_paragraph(style=style)
    run = p.add_run(text)
    run.bold = bold
    return p


def add_callout(doc, text):
    """Add a key-concept callout box."""
    p = doc.add_paragraph(style='Callout')
    run = p.add_run(text)
    # Add left border via XML
    pPr = p._p.get_or_add_pPr()
    pBdr = parse_xml(
        f'<w:pBdr {nsdecls("w")}>'
        f'  <w:left w:val="single" w:sz="18" w:space="8" w:color="2E74B5"/>'
        f'</w:pBdr>'
    )
    pPr.append(pBdr)
    return p


def add_equation(doc, text):
    p = doc.add_paragraph(style='Equation')
    p.add_run(text)
    return p


def add_code_block(doc, code_text):
    """Add a code block with grey background shading."""
    lines = code_text.strip().split('\n')
    for i, line in enumerate(lines):
        p = doc.add_paragraph(style='Code Block')
        p.add_run(line if line else ' ')
        # Grey background
        pPr = p._p.get_or_add_pPr()
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:val="clear" w:fill="F2F2F2"/>')
        pPr.append(shd)


def add_figure(doc, filename, caption, width=Inches(5.5)):
    """Add a centered figure with caption."""
    fig_path = FIGURES / filename
    if not fig_path.exists():
        add_para(doc, f'[Figure not found: {filename}]')
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(fig_path), width=width)
    # Caption
    cap = doc.add_paragraph(style='Caption')
    cap.add_run(caption)


def add_table(doc, headers, rows):
    """Add a formatted table."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    hdr = table.rows[0]
    _set_repeat_header_row(hdr)
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = h
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True
                run.font.size = Pt(10)
                run.font.name = 'Calibri'
        _set_cell_shading(cell, '2E74B5')
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    # Data rows
    for r_idx, row_data in enumerate(rows):
        row = table.rows[r_idx + 1]
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.text = str(val)
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(10)
                    run.font.name = 'Cambria'
            if r_idx % 2 == 1:
                _set_cell_shading(cell, 'EDF2F9')

    doc.add_paragraph()  # spacing
    return table


# ── Main Document Builder ────────────────────────────────────────────

def build():
    doc = Document()

    # Page setup
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    setup_styles(doc)

    # ═══════════════════════════════════════════════════════════════
    # TITLE PAGE
    # ═══════════════════════════════════════════════════════════════
    for _ in range(6):
        doc.add_paragraph()

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run('A Beginner\'s Guide to\nMean Field Modeling')
    run.font.name = 'Calibri'
    run.font.size = Pt(36)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x1B, 0x3A, 0x5C)

    doc.add_paragraph()

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run('From Coffee Cooling to Cell-Driven Scaffold Compaction')
    run.font.name = 'Cambria'
    run.font.size = Pt(16)
    run.font.italic = True
    run.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)

    for _ in range(4):
        doc.add_paragraph()

    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = info.add_run('GELS Project\nMcGhee Lab, University of Illinois Urbana-Champaign')
    run.font.name = 'Cambria'
    run.font.size = Pt(13)
    run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    doc.add_paragraph()

    ver = doc.add_paragraph()
    ver.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = ver.add_run('Version 2.4 — March 2026')
    run.font.name = 'Cambria'
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    # ═══════════════════════════════════════════════════════════════
    # TABLE OF CONTENTS (manual, since python-docx can't auto-TOC)
    # ═══════════════════════════════════════════════════════════════
    doc.add_page_break()

    toc_title = doc.add_paragraph()
    toc_title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = toc_title.add_run('Table of Contents')
    run.font.name = 'Calibri'
    run.font.size = Pt(26)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x1B, 0x3A, 0x5C)
    _add_bottom_border(toc_title, color="2E74B5", size=8)

    doc.add_paragraph()

    chapters = [
        ('Chapter 1', 'What Is a Mean Field Model?'),
        ('Chapter 2', 'Your First Mean Field Model: Coffee Cooling'),
        ('Chapter 3', 'Adding Nonlinearity: Population Growth'),
        ('Chapter 4', 'Two Competing Forces: A Thermostat'),
        ('Chapter 5', 'From Particles to Fields: The Porous Medium'),
        ('Chapter 6', 'Putting It All Together: The GELS Mean Field Model'),
        ('Chapter 7', 'Hands-On: Running the Model on Real DEM Data'),
        ('Chapter 8', 'Comparing Across Experiments: The DOE Sweep'),
        ('Chapter 9', 'From Scalar ODE to Spatial PDE'),
        ('Chapter 10', 'The Energy Landscape: A Thermodynamic View'),
        ('Chapter 11', 'Summary and Key Takeaways'),
        ('Appendix A', 'File Reference'),
        ('Appendix B', 'Generating the Figures'),
        ('Appendix C', 'Quick-Start Cheat Sheet'),
    ]
    for ch_num, ch_title in chapters:
        p = doc.add_paragraph()
        run_num = p.add_run(f'{ch_num}   ')
        run_num.font.name = 'Calibri'
        run_num.font.size = Pt(12)
        run_num.font.bold = True
        run_num.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)
        run_title = p.add_run(ch_title)
        run_title.font.name = 'Cambria'
        run_title.font.size = Pt(12)

    # ═══════════════════════════════════════════════════════════════
    # CHAPTER 1: What Is a Mean Field Model?
    # ═══════════════════════════════════════════════════════════════
    h = doc.add_heading('Chapter 1: What Is a Mean Field Model?', level=1)
    _add_bottom_border(h, color="2E74B5", size=8)

    doc.add_heading('The Core Idea', level=2)

    add_para(doc, (
        'Imagine you have a jar filled with 500 rubber balls, and someone is slowly '
        'squeezing the jar from the outside. You want to predict how tightly packed '
        'the balls become over time. You have two choices:'
    ))

    add_para(doc, (
        'Option A: Track every ball. Compute the position, velocity, and contacts of '
        'all 500 balls at every instant. This is a Discrete Element Method (DEM) '
        'simulation \u2014 it\'s accurate but expensive. Our new_dem_0.py does exactly this.'
    ), bold=False)

    add_para(doc, (
        'Option B: Track the average. Instead of 500 positions, track one number: '
        'the average packing fraction. Write a simple equation for how that number '
        'changes over time. This is a mean field model.'
    ), bold=False)

    add_para(doc, (
        'The term \u201cmean field\u201d means we replace the complex interactions between '
        'individual particles with their average (mean) effect. Instead of asking '
        '\u201cwhat force does ball #237 feel from its six neighbours?\u201d, we ask '
        '\u201cwhat is the average force a typical ball feels in a region with this '
        'packing density?\u201d'
    ))

    doc.add_heading('Why Bother?', level=2)

    add_table(doc,
        ['Feature', 'DEM Simulation', 'Mean Field Model'],
        [
            ['Accuracy', 'High (resolves individual contacts)', 'Approximate (averages over details)'],
            ['Speed', 'Minutes to hours', 'Milliseconds'],
            ['Parameters explored', '1 run at a time', '200,000 in seconds'],
            ['Physical insight', 'Hard to extract (too much data)', 'Built into the equations'],
            ['Design optimization', 'Impractical', 'Natural fit'],
        ]
    )

    add_para(doc, (
        'A mean field model is not a replacement for DEM \u2014 it\'s a companion. '
        'We use DEM to validate the mean field model, then use the mean field model '
        'to explore the vast parameter space that DEM cannot reach.'
    ))

    doc.add_heading('The Recipe', level=2)

    add_para(doc, 'Every mean field model follows the same five-step recipe:')

    steps = [
        'Choose your state variable(s) \u2014 what single number captures the system\'s state?',
        'Write the rate equation \u2014 how does that number change over time? This is almost always an ODE: dx/dt = f(x, t).',
        'Identify the driving forces \u2014 what pushes the system forward? What resists?',
        'Solve the ODE \u2014 analytically if possible, numerically if not.',
        'Validate against data \u2014 compare to experiments or detailed simulations.',
    ]
    for i, step in enumerate(steps, 1):
        p = doc.add_paragraph()
        run_num = p.add_run(f'{i}.  ')
        run_num.bold = True
        run_num.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)
        p.add_run(step)

    add_para(doc, 'Let\u2019s build this intuition step by step with increasingly complex examples.')

    # ═══════════════════════════════════════════════════════════════
    # CHAPTER 2: Coffee Cooling
    # ═══════════════════════════════════════════════════════════════
    h = doc.add_heading('Chapter 2: Coffee Cooling', level=1)
    _add_bottom_border(h, color="2E74B5", size=8)

    doc.add_heading('The Physical Setup', level=2)
    add_para(doc, (
        'You pour a cup of coffee at 90\u00b0C and set it on your desk. The room is 22\u00b0C. '
        'How does the coffee temperature change over time?'
    ))

    doc.add_heading('Step 1: Choose the State Variable', level=3)
    add_para(doc, (
        'T(t) = temperature of the coffee at time t (\u00b0C). We are \u201caveraging\u201d over all '
        'the complex fluid dynamics inside the cup and replacing it with a single number.'
    ))

    doc.add_heading('Step 2: Write the Rate Equation', level=3)
    add_para(doc, (
        'Newton\u2019s law of cooling says: the rate of temperature change is proportional '
        'to the temperature difference between the coffee and the room:'
    ))
    add_equation(doc, 'dT/dt = \u2212k \u00d7 (T \u2212 T_room)')
    add_para(doc, (
        'where k is a cooling constant (depends on cup material, surface area, etc.), '
        'T_room = 22\u00b0C (the environment), and the negative sign means the coffee loses '
        'heat when it\u2019s hotter than the room. This is our first mean field model!'
    ))

    doc.add_heading('Step 3: Solve', level=3)
    add_para(doc, 'This ODE has an exact analytical solution:')
    add_equation(doc, 'T(t) = T_room + (T\u2080 \u2212 T_room) \u00d7 exp(\u2212k \u00d7 t)')

    doc.add_heading('Step 4: Code It Up', level=3)
    add_code_block(doc, textwrap.dedent("""\
        import numpy as np

        T_0 = 90.0       # initial temperature (°C)
        T_room = 22.0    # room temperature (°C)
        k = 0.1          # cooling rate (1/min)

        t = np.linspace(0, 60, 200)
        T = T_room + (T_0 - T_room) * np.exp(-k * t)
    """))

    add_figure(doc, 'example_01_coffee_cooling.png',
               'Figure 2.1: Coffee cooling \u2014 exponential decay toward room temperature. '
               'One equation and one parameter (k) captures the entire system.')

    add_callout(doc, (
        'Key concept: A mean field model trades microscopic detail for macroscopic '
        'predictability. The parameter k is an effective parameter \u2014 it doesn\u2019t correspond '
        'to any single physical process, but captures their combined effect.'
    ))

    # ═══════════════════════════════════════════════════════════════
    # CHAPTER 3: Logistic Growth
    # ═══════════════════════════════════════════════════════════════
    h = doc.add_heading('Chapter 3: Adding Nonlinearity \u2014 Population Growth', level=1)
    _add_bottom_border(h, color="2E74B5", size=8)

    doc.add_heading('The Physical Setup', level=2)
    add_para(doc, (
        'A colony of bacteria doubles every hour in a petri dish. But the dish can only '
        'hold 1,000,000 bacteria. What happens?'
    ))

    doc.add_heading('The Logistic Equation', level=2)
    add_para(doc, (
        'If bacteria double freely: dN/dt = r \u00d7 N, giving exponential growth forever. '
        'But that\u2019s unrealistic. Add a resistance term:'
    ))
    add_equation(doc, 'dN/dt = r \u00d7 N \u00d7 (1 \u2212 N/K)')
    add_para(doc, (
        'This is the logistic equation. When N is small, (1 \u2212 N/K) \u2248 1, so growth is '
        'nearly exponential. When N \u2192 K, the growth rate \u2192 0. The system saturates. '
        'This is our first nonlinear mean field model.'
    ))

    add_figure(doc, 'example_02_logistic_growth.png',
               'Figure 3.1: Logistic growth vs uncapped exponential. The (1 \u2212 N/K) term '
               'creates the S-shaped saturation curve \u2014 the same idea we\'ll use for granular jamming.')

    add_callout(doc, (
        'Key concept: Mean field models become powerful when they include competing effects '
        '\u2014 a driving force and a resistance. The interplay between them creates rich '
        'dynamics from simple equations.'
    ))

    # ═══════════════════════════════════════════════════════════════
    # CHAPTER 4: Thermostat
    # ═══════════════════════════════════════════════════════════════
    h = doc.add_heading('Chapter 4: Two Competing Forces \u2014 A Thermostat', level=1)
    _add_bottom_border(h, color="2E74B5", size=8)

    doc.add_heading('Why This Example Matters', level=2)
    add_para(doc, (
        'The GELS mean field model is fundamentally about two competing stresses: '
        'cells pulling the scaffold together, and granule contacts resisting further '
        'compression. Before we get there, let\u2019s build intuition with a simpler system.'
    ))

    doc.add_heading('The Setup', level=2)
    add_para(doc, (
        'A room with a heater and an air conditioner. The heater turns on gradually '
        '(like cells maturing and forming bridges), and the AC resists temperature '
        'increases above a setpoint (like granular jamming resistance).'
    ))
    add_equation(doc, 'dT/dt = [Q_heater(t) \u2212 Q_cooling(T)] / C')
    add_para(doc, (
        'where Q_heater(t) ramps up over time (like cell maturation), '
        'Q_cooling(T) kicks in above a threshold (like jamming resistance), '
        'and C is thermal mass (like granular viscosity).'
    ))

    add_figure(doc, 'example_03_competing_forces.png',
               'Figure 4.1: Two competing forces. Left: temperature trajectory with threshold. '
               'Right: driving force vs resistance \u2014 shaded regions show net compaction vs arrest.')

    doc.add_heading('The Three Regimes', level=2)

    regimes = [
        ('Early (0\u20135 h):', 'The heater ramps up but the room is still below threshold. No resistance. Temperature rises freely. In our scaffold: cells are maturing but haven\u2019t started bridging.'),
        ('Middle (5\u201315 h):', 'Heater at full power, resistance growing. Temperature still rises but decelerates. In our scaffold: bridges forming, contacts stiffening.'),
        ('Late (15+ h):', 'Driving equals resistance. Dynamic equilibrium \u2014 not because forces disappeared, but because they exactly balance. In our scaffold: compaction arrests.'),
    ]
    for label, desc in regimes:
        p = doc.add_paragraph()
        run = p.add_run(label + '  ')
        run.bold = True
        run.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)
        p.add_run(desc)

    add_callout(doc, (
        'Key concept: The final state isn\u2019t determined by either force alone \u2014 it emerges '
        'from their balance. This is the central idea of the GELS mean field model.'
    ))

    # ═══════════════════════════════════════════════════════════════
    # CHAPTER 5: Porous Medium
    # ═══════════════════════════════════════════════════════════════
    h = doc.add_heading('Chapter 5: From Particles to Fields \u2014 The Porous Medium', level=1)
    _add_bottom_border(h, color="2E74B5", size=8)

    doc.add_heading('The Kozeny-Carman Equation', level=2)
    add_para(doc, (
        'One of the most useful results in porous media physics connects the void fraction '
        '(empty space between granules) to permeability (how easily fluid flows through):'
    ))
    add_equation(doc, 'K = \u03c6_v\u00b3 \u00d7 d\u00b2 / [180 \u00d7 (1 \u2212 \u03c6_v)\u00b2]')
    add_para(doc, (
        'where K is permeability (\u00b5m\u00b2), \u03c6_v is void fraction (0 to 1), and d is grain '
        'diameter (\u00b5m). This is a mean field relationship \u2014 it replaces complex pore '
        'geometry with a single effective parameter.'
    ))

    add_figure(doc, 'example_04_kozeny_carman.png',
               'Figure 5.1: Kozeny-Carman permeability. A small change in void fraction causes '
               'a huge change in permeability \u2014 the cubic dependence makes scaffold compaction critical.')

    doc.add_heading('Why This Matters for Tissue Engineering', level=2)
    add_para(doc, (
        'Going from \u03c6_v = 0.30 to \u03c6_v = 0.20 (a 33% reduction in void space) drops '
        'permeability by roughly 75%. This creates a feedback loop: cells compact the scaffold, '
        'reducing permeability, which starves cells of nutrients. The mean field model captures '
        'this coupling between structure and transport.'
    ))

    add_callout(doc, (
        'Key concept: The Kozeny-Carman equation converts structural information (packing '
        'fraction) into functional information (transport). This is how our mean field model '
        'connects cell-driven compaction to scaffold performance.'
    ))

    # ═══════════════════════════════════════════════════════════════
    # CHAPTER 6: The GELS Mean Field Model
    # ═══════════════════════════════════════════════════════════════
    h = doc.add_heading('Chapter 6: The GELS Mean Field Model', level=1)
    _add_bottom_border(h, color="2E74B5", size=8)

    doc.add_heading('6.1  The Physical Picture', level=2)
    add_para(doc, (
        'Our scaffold is a packed bed of hydrogel granules, seeded with living cells. '
        'There are two types of granules: functional (cell-laden, type 0) and inert '
        '(passive, type 1). Cells attach to functional granules, mature their focal '
        'adhesions, form bridges to neighbouring granules, and pull. This compacts the '
        'functional region while expelling void space into the inert region.'
    ))

    doc.add_heading('6.2  The State Variable', level=2)
    add_para(doc, (
        'We track one number: x_f(t) \u2014 the fraction of the domain occupied by the '
        '\u201cfunctional zone\u201d (functional granules + their local void space). '
        'Because granules are incompressible, the total solid volume \u03c6_f + \u03c6_i = \u03c6_solid '
        'is constant. What changes is how that solid is distributed.'
    ))

    doc.add_heading('6.3  Volume Conservation', level=2)
    add_para(doc, (
        'This is the most important constraint. The solid fractions \u03c6_f and \u03c6_i are constant '
        '(granules don\u2019t appear or disappear). But as x_f decreases, functional granules '
        'pack tighter locally:'
    ))
    add_equation(doc, '\u03c6_v,func = 1 \u2212 \u03c6_f / x_f        (less void, tighter packing)')
    add_equation(doc, '\u03c6_v,inert = 1 \u2212 \u03c6_i / (1 \u2212 x_f)    (more void, looser packing)')
    add_equation(doc, '\u03c6_v,global = 1 \u2212 \u03c6_solid           (always constant!)')

    add_figure(doc, 'example_05_volume_conservation.png',
               'Figure 6.1: Volume conservation. Left: local void fractions vs x_f \u2014 the global void '
               '(dashed) is exactly constant while the local voids change. Right: stacked bars showing '
               'how void redistributes from functional to inert zone during compaction.')

    doc.add_heading('6.4  The Driving Force: Cell Traction', level=2)
    add_para(doc, (
        'Cells pull on the scaffold through the motor-clutch model (Chan & Odde 2008). '
        'Motor proteins (myosin) pull inward, while clutch springs (integrins) grip the '
        'substrate. The cell \u201csenses\u201d substrate stiffness through the force balance:'
    ))
    add_equation(doc, '\u03b2 = k_sub / (k_sub + k_opt)')
    add_equation(doc, 'F_cell = F_stall \u00d7 \u03b2 \u00d7 engagement')
    add_para(doc, (
        'On soft substrates (small E): k_sub \u226a k_opt, so \u03b2 \u2192 0 and cells barely pull. '
        'On stiff substrates (large E): k_sub \u226b k_opt, so \u03b2 \u2192 1 and cells pull at full force.'
    ))

    add_figure(doc, 'example_06_motor_clutch.png',
               'Figure 6.2: Motor-clutch model \u2014 cell traction force vs substrate stiffness. '
               'Hydrogels (1\u201350 kPa) sit on the rising portion of the curve.')

    add_para(doc, (
        'But cells don\u2019t pull at full force from the start. Three biological processes '
        'ramp up over time: (1) FA maturation, (2) bridge formation via a Poisson process, '
        'and (3) force ramp for newly formed bridges. The total cell stress is:'
    ))
    add_equation(doc, '\u03c3_cell(t) = n_density \u00d7 F_cell \u00d7 f_bridge(t) \u00d7 maturity(t) \u00d7 ramp(t)')

    doc.add_heading('6.5  The Resistance: Granular Jamming', level=2)
    add_para(doc, (
        'As cells compact the functional zone, the local packing fraction \u03c6_f/x_f increases. '
        'When it approaches the random close packing (RCP) fraction \u03c6_RCP (~0.64 for 3D spheres, '
        '~0.82 for 2D), the granules jam:'
    ))
    add_equation(doc, '\u03c3_resist(x_f) = \u03c3\u2080 \u00d7 (\u03c6_f/x_f / \u03c6_RCP \u2212 1)^\u03b1    when \u03c6_f/x_f > \u03c6_RCP')
    add_equation(doc, '\u03c3_resist(x_f) = 0                           otherwise')

    add_figure(doc, 'example_07_jamming_resistance.png',
               'Figure 6.3: Jamming resistance for different exponents \u03b1. '
               'The dashed line marks x_f at the jamming threshold.')

    doc.add_heading('6.6  The Central ODE', level=2)
    add_para(doc, 'Combining driving force and resistance:')
    add_equation(doc, 'dx_f/dt = \u2212x_f \u00d7 (\u03c3_cell(t) \u2212 \u03c3_resist(x_f)) / \u03b7_eff')

    add_table(doc,
        ['Term', 'Meaning', 'Analogy to Ch. 4'],
        [
            ['dx_f/dt', 'Rate of compaction', 'dT/dt'],
            ['x_f', 'Current state', 'T'],
            ['\u03c3_cell(t)', 'Cell traction (time-dependent)', 'Q_heater(t)'],
            ['\u03c3_resist(x_f)', 'Jamming resistance (state-dependent)', 'Q_cooling(T)'],
            ['\u03b7_eff', 'Effective granular viscosity', 'C (thermal mass)'],
        ]
    )

    add_para(doc, 'The three fitted parameters are:', bold=True)

    fitted = [
        ('\u03b7_eff', 'effective granular viscosity. Controls how fast compaction proceeds.'),
        ('\u03c3\u2080', 'jamming stress prefactor. Controls how strongly the jammed state resists.'),
        ('\u03b1', 'jamming exponent. Controls how sharply resistance turns on at jamming.'),
    ]
    for sym, desc in fitted:
        p = doc.add_paragraph()
        run = p.add_run(f'{sym} \u2014 ')
        run.bold = True
        run.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)
        p.add_run(desc)

    add_para(doc, (
        'Everything else (E_modulus, cell count, bridge kinetics, geometry) is set by the '
        'simulation parameters. Only these three are fitted to match DEM data.'
    ))

    doc.add_heading('6.7  Full Worked Example', level=2)
    add_para(doc, (
        'The following figure shows the complete mean field model solved for a typical '
        'scaffold (\u03c6_f = 0.40, \u03c6_i = 0.25, E = 10 kPa, R = 40 \u00b5m, 8 cells per granule):'
    ))

    add_figure(doc, 'example_08_full_model.png',
               'Figure 6.4: Complete GELS mean field model. (a) Compaction trajectory x_f(t). '
               '(b) Void redistribution between zones. (c) Stress balance \u2014 cell traction vs '
               'jamming resistance. (d) Functional zone permeability drops by ~10\u00d7.',
               width=Inches(6.0))

    doc.add_heading('6.8  The Biological Timeline', level=2)
    add_para(doc, (
        'The driving force has a characteristic S-shape because it is the product of three '
        'biological ramp functions: FA maturity, bridge fraction, and force ramp.'
    ))

    add_figure(doc, 'example_09_biological_timeline.png',
               'Figure 6.5: Biological timeline. Hours 0\u20133: spreading. Hours 3\u20134: FA maturation begins. '
               'Hours 4\u20138: bridges form rapidly. Hours 8+: force saturates.')

    doc.add_heading('6.9  Parameter Sensitivity', level=2)
    add_para(doc, (
        'One of the biggest advantages of a mean field model is that you can instantly see '
        'how parameters affect the outcome:'
    ))

    add_figure(doc, 'example_10_parameter_sensitivity.png',
               'Figure 6.6: Parameter sensitivity. (a) Viscosity controls the timescale. '
               '(b) Jamming prefactor controls the final state. (c) Jamming exponent controls '
               'sharpness of arrest. (d) Substrate stiffness controls cell traction magnitude.',
               width=Inches(6.0))

    # ═══════════════════════════════════════════════════════════════
    # CHAPTER 7: Hands-On with Real Data
    # ═══════════════════════════════════════════════════════════════
    h = doc.add_heading('Chapter 7: Running the Model on Real DEM Data', level=1)
    _add_bottom_border(h, color="2E74B5", size=8)

    doc.add_heading('7.1  What\u2019s in a DEM Run?', level=2)
    add_para(doc, (
        'Each DOE trial directory (e.g., results/LHC/DOE_2D_0006/) contains the complete '
        'output of a DEM simulation: params.json (all parameters), history.csv (time series '
        'of bulk metrics at ~49 timesteps), snapshots (per-particle data), and fields '
        '(phase field arrays).'
    ))

    doc.add_heading('7.2  Fitting a Single Run', level=2)
    add_para(doc, (
        'The simplest workflow is a one-line command that loads DEM data, constructs the '
        'mean field model from the simulation parameters, and fits the three free parameters:'
    ))
    add_code_block(doc, textwrap.dedent("""\
        from analysis.mean_field_model import from_run

        model, fit, data = from_run('results/LHC/DOE_2D_0010')

        print(f"eta_eff = {fit['eta_eff']:.4g}")
        print(f"sigma_0 = {fit['sigma_0']:.4g}")
        print(f"alpha   = {fit['alpha']:.3f}")
        print(f"R^2     = {fit['R2']:.4f}")
    """))

    doc.add_heading('7.3  The Diagnostic Plots', level=2)
    add_para(doc, (
        'The figure below shows the four-panel diagnostic for DOE_2D_0010 (E = 4.0 kPa, '
        'a soft hydrogel). Panel (a) overlays the DEM data points with the mean field fit. '
        'Panel (b) shows that global porosity remains approximately constant (volume conservation). '
        'Panel (c) shows the stress balance, and panel (d) compares model and DEM permeability.'
    ))

    add_figure(doc, 'example_11_dem_single_fit.png',
               'Figure 7.1: Mean field model fit to DEM simulation DOE_2D_0010. '
               'The model captures the compaction trajectory, stress evolution, and permeability trend.',
               width=Inches(6.0))

    doc.add_heading('7.4  Understanding the Fitted Parameters', level=2)

    add_table(doc,
        ['Condition', '\u03b7_eff', '\u03c3\u2080', '\u03b1', 'Physical Meaning'],
        [
            ['Soft (E~4 kPa)', '0.1\u20131.0', '~0.001', '1.0\u20131.5', 'Low viscosity, weak jamming, gradual arrest'],
            ['Stiff (E~48 kPa)', '1.0\u201310.0', '~0.01+', '1.0\u20132.0', 'Higher viscosity, stronger resistance'],
        ]
    )

    # ═══════════════════════════════════════════════════════════════
    # CHAPTER 8: DOE Comparison
    # ═══════════════════════════════════════════════════════════════
    h = doc.add_heading('Chapter 8: Comparing Across Experiments', level=1)
    _add_bottom_border(h, color="2E74B5", size=8)

    doc.add_heading('8.1  Why Compare Multiple Runs?', level=2)
    add_para(doc, (
        'A single DEM run gives you one data point. Our DOE (Design of Experiments) gives '
        '19 runs spanning different modulus (4\u2013100 kPa), composition (func_ratio 0.33\u20130.93), '
        'and granule size (R = 45\u2013100 \u00b5m). By fitting the mean field model to all of them, '
        'we see how the physics varies across conditions.'
    ))

    add_figure(doc, 'example_12_doe_comparison.png',
               'Figure 8.1: DOE comparison across 20 DEM trials. (a) Compaction vs stiffness, '
               'colored by functional ratio. (b) Compaction vs composition, colored by modulus. '
               '(c) Structure-transport relationship with Kozeny-Carman overlay. '
               '(d) Peak bridge count vs granule size.',
               width=Inches(6.0))

    doc.add_heading('8.2  The Multi-Run Overlay', level=2)
    add_para(doc, (
        'Overlaying trajectories from all runs reveals the full range of scaffold behaviours '
        'achievable across the design space:'
    ))

    add_figure(doc, 'example_13_multi_run_overlay.png',
               'Figure 8.2: All 20 DOE trajectories colored by stiffness. (a) Compaction ranges '
               'from minimal to >40%. (b) Porosity fluctuations are small (volume conservation). '
               '(c) Bridge formation dynamics vary widely with composition.',
               width=Inches(6.0))

    # ═══════════════════════════════════════════════════════════════
    # CHAPTER 9: ODE to PDE
    # ═══════════════════════════════════════════════════════════════
    h = doc.add_heading('Chapter 9: From Scalar ODE to Spatial PDE', level=1)
    _add_bottom_border(h, color="2E74B5", size=8)

    doc.add_heading('9.1  The Limitation of the Scalar Model', level=2)
    add_para(doc, (
        'The ODE assumes the scaffold compacts uniformly everywhere. In reality, the center '
        'has more neighbours (higher bridge probability, more compaction) while the edge '
        'has fewer. This gradient is captured by extending the scalar ODE to a 1D radial PDE:'
    ))
    add_equation(doc, '\u2202x_f/\u2202t = \u2212x_f(\u03be,t) \u00d7 [\u03c3_cell(\u03be,t) \u2212 \u03c3_resist(x_f)] / \u03b7_eff + D_eff \u00d7 \u2202\u00b2x_f/\u2202\u03be\u00b2')
    add_para(doc, (
        'where \u03be \u2208 [0, 1] is a radial coordinate (0 = center, 1 = edge), and '
        'D_eff captures stress-driven redistribution of packing.'
    ))

    doc.add_heading('9.2  Spatially-Varying Bridge Rate', level=2)
    add_para(doc, (
        'A key addition is the neighbor_factor = 0.5 \u00d7 (1 + cos(\u03c0\u03be)), which is 1.0 at the '
        'center (all neighbours present) and 0.0 at the edge (no neighbours outside the '
        'scaffold). This makes bridges form faster in the center.'
    ))

    doc.add_heading('9.3  The Vectorised Sweep', level=2)
    add_para(doc, (
        'The analysis/parameter_sweep.py module solves this PDE for 200,000 parameter '
        'combinations simultaneously using vectorised NumPy. Eleven dimensions are swept '
        '(granule size, stiffness, composition, shape, cell loading), and for each sample '
        'the model predicts compaction, permeability, tissue growth, and distance to 7 organ targets.'
    ))

    doc.add_heading('9.4  Tissue Volume Growth', level=2)
    add_para(doc, (
        'The PDE model also includes spatially-varying tissue growth (V2.1+): tissue fills '
        'the void space created by compaction, with more bridges locally promoting more ECM '
        'secretion. This is important because tissue volume affects the final architecture '
        'descriptors used for organ matching.'
    ))

    # ═══════════════════════════════════════════════════════════════
    # CHAPTER 10: Energy Landscape
    # ═══════════════════════════════════════════════════════════════
    h = doc.add_heading('Chapter 10: The Energy Landscape', level=1)
    _add_bottom_border(h, color="2E74B5", size=8)

    doc.add_heading('10.1  From Forces to Energy', level=2)
    add_para(doc, (
        'Everything so far is in the force picture: driving stress vs resistance stress. '
        'But there\u2019s a complementary energy picture that gives deeper insight. The '
        'analysis/energy_landscape.py module computes a free energy landscape G(\u03be) where '
        '\u03be is a compaction coordinate (\u03be = 0 means no compaction, \u03be \u2192 1 means fully compacted).'
    ))

    doc.add_heading('10.2  Six Energy Terms', level=2)
    add_equation(doc, 'G_total(\u03be) = G_cell + G_elastic + G_yield + G_void + G_inert + G_surface')

    add_table(doc,
        ['Term', 'Physical Origin', 'Sign', 'Meaning'],
        [
            ['G_cell', 'Cell traction', 'Negative', 'Cells lower energy by compacting'],
            ['G_elastic', 'Hertzian contacts', 'Positive', 'Contacts store elastic energy'],
            ['G_yield', 'Granular friction', 'Positive', 'Activation barrier to start compaction'],
            ['G_void', 'Osmotic pressure', 'Positive', 'Cost of redistributing void'],
            ['G_inert', 'Inert frustration', 'Positive', 'Inert granules block geometrically'],
            ['G_surface', 'Interface tension', 'Mixed', 'Interfacial energy at func/inert boundary'],
        ]
    )

    add_figure(doc, 'example_14_energy_landscape.png',
               'Figure 10.1: Energy landscape decomposition. (a) Individual energy terms \u2014 '
               'G_cell (driving) vs resistance terms. (b) Total free energy with equilibrium '
               'marked by a star.',
               width=Inches(6.0))

    doc.add_heading('10.3  Dimensionless Groups', level=2)
    add_para(doc, (
        'The energy landscape naturally defines dimensionless groups that collapse the '
        'parameter space:'
    ))

    add_table(doc,
        ['Group', 'Definition', 'Physical Meaning'],
        [
            ['\u03b2', '\u03c3_cell / \u03c3\u2080', 'Cell-to-jamming ratio'],
            ['Ca', '\u03c3_cell / \u03c3_yield', 'Cellular capillary number'],
            ['\u03a6_r', '\u03c6_i / \u03c6_f', 'Inert obstruction ratio'],
            ['\u03a8', '\u03c6_f,local / \u03c6_RCP', 'Proximity to jamming'],
            ['\u0393', '\u03b3_fi / \u03c3_cell', 'Interfacial number'],
        ]
    )

    # ═══════════════════════════════════════════════════════════════
    # CHAPTER 11: Summary
    # ═══════════════════════════════════════════════════════════════
    h = doc.add_heading('Chapter 11: Summary and Key Takeaways', level=1)
    _add_bottom_border(h, color="2E74B5", size=8)

    doc.add_heading('The Building Blocks', level=2)

    add_table(doc,
        ['Example', 'State Variable', 'Driving Force', 'Resistance', 'Key Lesson'],
        [
            ['Coffee cooling', 'Temperature T', '\u2014', '~ (T \u2212 T_room)', 'Exponential decay to equilibrium'],
            ['Population growth', 'Population N', 'r \u00d7 N', '(1 \u2212 N/K)', 'Nonlinearity creates saturation'],
            ['Thermostat', 'Temperature T', 'Q_heater(t)', 'Q_cool above threshold', 'Two forces \u2192 equilibrium'],
            ['Porous medium', 'Void fraction \u03c6_v', '\u2014', '\u2014', 'Structure \u2192 transport (K-C)'],
            ['GELS', 'Zone fraction x_f', '\u03c3_cell(t)', '\u03c3_resist(x_f)', 'All of the above combined'],
        ]
    )

    doc.add_heading('The Three Levels of the Mean Field Model', level=2)

    add_table(doc,
        ['Level', 'Code Module', 'State', 'Fitted Params', 'Best For'],
        [
            ['1. Scalar ODE', 'mean_field_model.py', 'x_f(t)', '3 (\u03b7, \u03c3\u2080, \u03b1)', 'Understanding, validation'],
            ['2. Radial PDE', 'parameter_sweep.py', 'x_f(\u03be, t)', '11 swept', 'Design optimization, organ targeting'],
            ['3. Energy Landscape', 'energy_landscape.py', 'G(\u03be)', '0 (all from physics)', 'Insight, scaling laws'],
        ]
    )

    doc.add_heading('Key Equations', level=2)

    add_table(doc,
        ['Equation', 'What It Does'],
        [
            ['F_cell = F_stall \u00d7 \u03b2 \u00d7 engagement', 'Motor-clutch cell traction'],
            ['\u03c3_cell = n \u00d7 F \u00d7 f_bridge \u00d7 maturity \u00d7 ramp', 'Total cell stress'],
            ['\u03c3_resist = \u03c3\u2080 \u00d7 (\u03c6_local/\u03c6_RCP \u2212 1)^\u03b1', 'Jamming resistance'],
            ['dx_f/dt = \u2212x_f(\u03c3_cell \u2212 \u03c3_resist)/\u03b7_eff', 'Compaction ODE'],
            ['K = \u03c6_v\u00b3 d\u00b2 / [180(1\u2212\u03c6_v)\u00b2]', 'Kozeny-Carman permeability'],
        ]
    )

    doc.add_heading('Tips for the Beginner', level=2)

    tips = [
        'Start simple. If your mean field model has more than 3\u20135 free parameters, it\u2019s probably overfitting.',
        'Validate against data. Always compute R\u00b2 and look at the residuals.',
        'Understand the limits. Mean field models average over spatial fluctuations \u2014 they work well for bulk properties but miss local effects.',
        'Use dimensional analysis. Dimensionless groups reduce the parameter space and give physical insight.',
        'The model is a tool, not the truth. Use it to generate hypotheses, guide experiments, and explore design space.',
        'Run generate_examples.py yourself. Modify parameters and re-run to build intuition.',
    ]
    for i, tip in enumerate(tips, 1):
        p = doc.add_paragraph()
        run = p.add_run(f'{i}.  ')
        run.bold = True
        run.font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)
        p.add_run(tip)

    # ═══════════════════════════════════════════════════════════════
    # APPENDICES
    # ═══════════════════════════════════════════════════════════════
    h = doc.add_heading('Appendix A: File Reference', level=1)
    _add_bottom_border(h, color="2E74B5", size=8)

    add_table(doc,
        ['File', 'Purpose'],
        [
            ['analysis/mean_field_model.py', 'Two-zone compaction ODE, fitting, plotting'],
            ['analysis/parameter_sweep.py', '11-D LHS sweep with 1D radial PDE + tissue growth'],
            ['analysis/energy_landscape.py', 'Free energy decomposition, kinetics, dimensionless groups'],
            ['analysis/tissue_descriptors.py', '18 tissue architecture descriptors'],
            ['analysis/organ_targets.py', '7 organ target vectors for design optimization'],
            ['viz/dimensionless.py', 'Dimensionless analysis and data collapse'],
            ['CodeLog/MeanFieldModeling/', 'This guide, figure generator, and output figures'],
        ]
    )

    h = doc.add_heading('Appendix B: Generating the Figures', level=1)
    _add_bottom_border(h, color="2E74B5", size=8)

    add_para(doc, 'All figures in this guide can be regenerated by running:')
    add_code_block(doc, 'python CodeLog/MeanFieldModeling/generate_examples.py')
    add_para(doc, (
        'This produces all 14 PNG figures in CodeLog/MeanFieldModeling/figures/. '
        'Examples 1\u201310 use synthetic data and always work. Examples 11\u201314 use real '
        'DEM data from results/LHC/ and are skipped if that directory is not present.'
    ))

    h = doc.add_heading('Appendix C: Quick-Start Cheat Sheet', level=1)
    _add_bottom_border(h, color="2E74B5", size=8)

    add_code_block(doc, textwrap.dedent("""\
        # === Fit mean field to one DEM run ===
        from analysis.mean_field_model import from_run
        model, fit, data = from_run('results/LHC/DOE_2D_0006')

        # === Generate all diagnostic plots ===
        from analysis.mean_field_model import run_all
        model, fit, data = run_all('results/LHC/DOE_2D_0006')

        # === Build a model from scratch ===
        from analysis.mean_field_model import CompactionModel, MotorClutchParams
        mc = MotorClutchParams()
        model = CompactionModel(
            E_modulus=10.0, phi_f0=0.40, phi_i0=0.25,
            R_func=40.0, n_cells_per_granule=8, N_func=160,
            domain_volume=800**2, mc_params=mc
        )
        sol = model.solve((0, 72), eta_eff=0.5, sigma_0=0.001, alpha=1.2)

        # === Run a 200k parameter sweep ===
        from analysis.parameter_sweep import generate_lhs, run_sweep_vectorised
        samples = generate_lhs(200_000, seed=42)
        outputs = run_sweep_vectorised(samples)
    """))

    # ── Save ─────────────────────────────────────────────────────
    doc.save(str(OUTPATH))
    print(f"\nSaved: {OUTPATH}")
    print(f"Size:  {OUTPATH.stat().st_size / 1024:.0f} KB")


if __name__ == '__main__':
    print("=" * 60)
    print("  Building Mean Field Modeling Textbook (.docx)")
    print("=" * 60)
    build()
    print("\nDone!")
