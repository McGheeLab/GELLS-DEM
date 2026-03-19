#!/usr/bin/env python3
"""
Build Mean Field Modeling textbook (.docx) using pandoc.

Workflow:
  1. Customize pandoc reference.docx with textbook styles
  2. Convert MEAN_FIELD_GUIDE.md → .docx via pandoc
  3. Post-process: embed figures, add title page, fix table styling

Requires: pypandoc_binary (includes pandoc), python-docx
"""

import os, sys, re, copy
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml
import pypandoc

HERE = Path(__file__).resolve().parent
GUIDE_MD = HERE / "MEAN_FIELD_GUIDE.md"
FIGURES_DIR = HERE / "figures"
REF_DOCX = HERE / "reference.docx"
OUTPUT_DOCX = HERE / "Mean_Field_Modeling_Guide.docx"

# ── Colors ──────────────────────────────────────────────────────────
NAVY = RGBColor(0x00, 0x2B, 0x5C)       # Chapter headings
DARK_BLUE = RGBColor(0x1A, 0x47, 0x8A)  # Section headings
STEEL_BLUE = RGBColor(0x3A, 0x6E, 0xA5) # Subsection headings
TABLE_HDR_BG = "002B5C"
TABLE_ALT_BG = "E8EEF4"
CALLOUT_BORDER = "3A6EA5"


def customize_reference():
    """Modify pandoc's default reference.docx for textbook styling."""
    doc = Document(str(REF_DOCX))

    # --- Heading 1 (Chapter) ---
    h1 = doc.styles['Heading 1']
    h1.font.name = 'Calibri'
    h1.font.size = Pt(26)
    h1.font.bold = True
    h1.font.color.rgb = NAVY
    h1.paragraph_format.space_before = Pt(36)
    h1.paragraph_format.space_after = Pt(18)
    # Bottom border
    pPr = h1.element.get_or_add_pPr()
    pBdr = parse_xml(
        f'<w:pBdr {nsdecls("w")}>'
        f'  <w:bottom w:val="single" w:sz="8" w:space="4" w:color="{TABLE_HDR_BG}"/>'
        f'</w:pBdr>'
    )
    pPr.append(pBdr)

    # --- Heading 2 (Section) ---
    h2 = doc.styles['Heading 2']
    h2.font.name = 'Calibri'
    h2.font.size = Pt(18)
    h2.font.bold = True
    h2.font.color.rgb = DARK_BLUE
    h2.paragraph_format.space_before = Pt(24)
    h2.paragraph_format.space_after = Pt(10)

    # --- Heading 3 (Subsection) ---
    h3 = doc.styles['Heading 3']
    h3.font.name = 'Calibri'
    h3.font.size = Pt(14)
    h3.font.bold = True
    h3.font.color.rgb = STEEL_BLUE
    h3.paragraph_format.space_before = Pt(14)
    h3.paragraph_format.space_after = Pt(6)

    # --- Normal (body text) ---
    normal = doc.styles['Normal']
    normal.font.name = 'Cambria'
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    # --- First Paragraph (same as normal, no indent) ---
    if 'First Paragraph' in [s.name for s in doc.styles]:
        fp = doc.styles['First Paragraph']
        fp.font.name = 'Cambria'
        fp.font.size = Pt(11)

    # --- Block Text / Source Code (code blocks) ---
    for style_name in ['Source Code', 'Block Text']:
        if style_name in [s.name for s in doc.styles]:
            st = doc.styles[style_name]
            st.font.name = 'Consolas'
            st.font.size = Pt(9)
            st.font.color.rgb = RGBColor(0x2A, 0x2A, 0x2A)
            st.paragraph_format.space_before = Pt(4)
            st.paragraph_format.space_after = Pt(4)
            # Grey background shading
            pPr = st.element.get_or_add_pPr()
            shd = parse_xml(f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="F0F0F0"/>')
            pPr.append(shd)

    # --- Table style (we'll handle inline) ---
    # --- Caption style ---
    style_names = [s.name for s in doc.styles]
    if 'Caption' in style_names:
        cap = doc.styles['Caption']
        cap.font.name = 'Calibri'
        cap.font.size = Pt(10)
        cap.font.italic = True
        cap.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
        cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.save(str(REF_DOCX))
    print(f"  Customized reference.docx styles")


def convert_md_to_docx():
    """Use pandoc to convert Markdown → .docx with reference styles."""
    pandoc_path = pypandoc.get_pandoc_path()

    # Read and preprocess the markdown
    md_text = GUIDE_MD.read_text(encoding='utf-8')

    # Convert text-based figure references to actual image embeds:
    #   > **Figure reference:** See `figures/example_01_coffee_cooling.png`
    # becomes:
    #   ![Figure: Coffee Cooling](absolute/path/to/figures/example_01_coffee_cooling.png)
    def ref_to_image(m):
        rel_path = m.group(1)  # e.g. "figures/example_01_coffee_cooling.png"
        abs_path = str(HERE / rel_path)
        if not Path(abs_path).exists():
            print(f"    WARNING: figure not found: {abs_path}")
            return m.group(0)  # keep original if file not found
        # Build a readable caption from the filename
        filename = Path(rel_path).stem  # e.g. "example_01_coffee_cooling"
        caption = filename.replace('_', ' ')
        # e.g. "example 01 coffee cooling" → "Figure 1: Coffee Cooling"
        parts = caption.split(' ', 2)  # ['example', '01', 'coffee cooling']
        if len(parts) >= 3:
            num = parts[1].lstrip('0') or '0'
            desc = parts[2].title()
            caption = f"Figure {num}: {desc}"
        else:
            caption = caption.title()
        return f'\n![{caption}]({abs_path})\n'

    md_text = re.sub(
        r'>\s*\*\*Figure reference:\*\*\s*See\s*`(figures/[^`]+\.png)`',
        ref_to_image, md_text
    )

    # Also convert any existing ![](figures/...) to absolute paths
    def fix_img_path(m):
        alt = m.group(1)
        rel_path = m.group(2)
        abs_path = str(HERE / rel_path)
        return f'![{alt}]({abs_path})'

    md_text = re.sub(r'!\[([^\]]*)\]\((figures/[^)]+)\)', fix_img_path, md_text)

    # Write preprocessed markdown to temp file
    tmp_md = HERE / "_guide_preprocessed.md"
    tmp_md.write_text(md_text, encoding='utf-8')

    # Run pandoc
    output = pypandoc.convert_file(
        str(tmp_md),
        'docx',
        outputfile=str(OUTPUT_DOCX),
        extra_args=[
            f'--reference-doc={REF_DOCX}',
            '--toc',
            '--toc-depth=3',
            '--syntax-highlighting=tango',
            '--resource-path', str(HERE),
        ]
    )
    tmp_md.unlink()
    print(f"  Pandoc conversion complete")


def post_process():
    """Post-process: add title page, style tables, adjust figure sizes."""
    doc = Document(str(OUTPUT_DOCX))

    # ── Insert title page at the beginning ──
    # We insert paragraphs at the very start
    body = doc.element.body

    # Create title page elements (inserted in reverse order at position 0)
    title_elements = []

    # Page break after title page
    pb_para = parse_xml(
        f'<w:p {nsdecls("w")}>'
        f'  <w:pPr><w:pageBreakBefore/></w:pPr>'
        f'</w:p>'
    )

    # Spacer
    def make_spacer(pts=48):
        return parse_xml(
            f'<w:p {nsdecls("w")}>'
            f'  <w:pPr><w:spacing w:before="{int(pts*20)}" w:after="0"/>'
            f'    <w:jc w:val="center"/></w:pPr>'
            f'</w:p>'
        )

    # Title
    title_para = parse_xml(
        f'<w:p {nsdecls("w")}>'
        f'  <w:pPr><w:spacing w:before="3600" w:after="200"/>'
        f'    <w:jc w:val="center"/></w:pPr>'
        f'  <w:r><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>'
        f'    <w:b/><w:sz w:val="72"/><w:color w:val="002B5C"/></w:rPr>'
        f'    <w:t>A Beginner&#x27;s Guide to</w:t></w:r>'
        f'</w:p>'
    )
    title_para2 = parse_xml(
        f'<w:p {nsdecls("w")}>'
        f'  <w:pPr><w:spacing w:before="0" w:after="400"/>'
        f'    <w:jc w:val="center"/></w:pPr>'
        f'  <w:r><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>'
        f'    <w:b/><w:sz w:val="72"/><w:color w:val="002B5C"/></w:rPr>'
        f'    <w:t>Mean Field Modeling</w:t></w:r>'
        f'</w:p>'
    )

    # Subtitle
    subtitle_para = parse_xml(
        f'<w:p {nsdecls("w")}>'
        f'  <w:pPr><w:spacing w:before="200" w:after="600"/>'
        f'    <w:jc w:val="center"/></w:pPr>'
        f'  <w:r><w:rPr><w:rFonts w:ascii="Cambria" w:hAnsi="Cambria"/>'
        f'    <w:i/><w:sz w:val="28"/><w:color w:val="3A6EA5"/></w:rPr>'
        f'    <w:t>From Coffee Cooling to Cell-Driven Scaffold Compaction</w:t></w:r>'
        f'</w:p>'
    )

    # Horizontal rule (border-bottom on empty para)
    hr_para = parse_xml(
        f'<w:p {nsdecls("w")}>'
        f'  <w:pPr><w:spacing w:before="200" w:after="200"/>'
        f'    <w:jc w:val="center"/>'
        f'    <w:pBdr><w:bottom w:val="single" w:sz="6" w:space="1" w:color="002B5C"/></w:pBdr>'
        f'  </w:pPr>'
        f'</w:p>'
    )

    # Project info
    info_para = parse_xml(
        f'<w:p {nsdecls("w")}>'
        f'  <w:pPr><w:spacing w:before="400" w:after="100"/>'
        f'    <w:jc w:val="center"/></w:pPr>'
        f'  <w:r><w:rPr><w:rFonts w:ascii="Cambria" w:hAnsi="Cambria"/>'
        f'    <w:sz w:val="24"/><w:color w:val="333333"/></w:rPr>'
        f'    <w:t>GELS Project</w:t></w:r>'
        f'</w:p>'
    )
    info_para2 = parse_xml(
        f'<w:p {nsdecls("w")}>'
        f'  <w:pPr><w:spacing w:before="100" w:after="100"/>'
        f'    <w:jc w:val="center"/></w:pPr>'
        f'  <w:r><w:rPr><w:rFonts w:ascii="Cambria" w:hAnsi="Cambria"/>'
        f'    <w:sz w:val="24"/><w:color w:val="333333"/></w:rPr>'
        f'    <w:t>McGhee Lab &#x2022; University of Illinois Urbana-Champaign</w:t></w:r>'
        f'</w:p>'
    )
    info_para3 = parse_xml(
        f'<w:p {nsdecls("w")}>'
        f'  <w:pPr><w:spacing w:before="400" w:after="100"/>'
        f'    <w:jc w:val="center"/></w:pPr>'
        f'  <w:r><w:rPr><w:rFonts w:ascii="Cambria" w:hAnsi="Cambria"/>'
        f'    <w:i/><w:sz w:val="22"/><w:color w:val="666666"/></w:rPr>'
        f'    <w:t>A teaching document for understanding mean field approximations</w:t></w:r>'
        f'</w:p>'
    )
    info_para4 = parse_xml(
        f'<w:p {nsdecls("w")}>'
        f'  <w:pPr><w:spacing w:before="100" w:after="100"/>'
        f'    <w:jc w:val="center"/></w:pPr>'
        f'  <w:r><w:rPr><w:rFonts w:ascii="Cambria" w:hAnsi="Cambria"/>'
        f'    <w:i/><w:sz w:val="22"/><w:color w:val="666666"/></w:rPr>'
        f'    <w:t>in the context of granular scaffold compaction</w:t></w:r>'
        f'</w:p>'
    )

    # Insert all title page elements at position 0 (in order)
    first_child = body[0] if len(body) > 0 else None
    for elem in [title_para, title_para2, subtitle_para, hr_para,
                 info_para, info_para2, info_para3, info_para4, pb_para]:
        if first_child is not None:
            first_child.addprevious(elem)
        else:
            body.append(elem)

    # ── Style all tables ──
    for table in doc.tables:
        # Set table width
        table.alignment = WD_TABLE_ALIGNMENT.CENTER

        for i, row in enumerate(table.rows):
            for cell in row.cells:
                # Header row styling
                if i == 0:
                    shading = parse_xml(
                        f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="{TABLE_HDR_BG}"/>'
                    )
                    cell._tc.get_or_add_tcPr().append(shading)
                    for para in cell.paragraphs:
                        for run in para.runs:
                            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                            run.font.bold = True
                            run.font.name = 'Calibri'
                            run.font.size = Pt(10)
                # Alternating rows
                elif i % 2 == 0:
                    shading = parse_xml(
                        f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="{TABLE_ALT_BG}"/>'
                    )
                    cell._tc.get_or_add_tcPr().append(shading)

                # All cells: font
                for para in cell.paragraphs:
                    for run in para.runs:
                        if run.font.name is None:
                            run.font.name = 'Cambria'
                        if run.font.size is None:
                            run.font.size = Pt(10)

    # ── Resize images to fit nicely ──
    for para in doc.paragraphs:
        for run in para.runs:
            for drawing in run.element.findall(qn('w:drawing')):
                # Find inline extent
                for inline in drawing.findall(qn('wp:inline')):
                    extent = inline.find(qn('wp:extent'))
                    if extent is not None:
                        cx = int(extent.get('cx', 0))
                        # If wider than 5.5 inches (5029200 EMU), scale down
                        max_width = Inches(5.5)
                        if cx > max_width:
                            ratio = max_width / cx
                            cy = int(extent.get('cy', 0))
                            extent.set('cx', str(int(max_width)))
                            extent.set('cy', str(int(cy * ratio)))

    # ── Style blockquotes as callout boxes ──
    for para in doc.paragraphs:
        pPr = para._element.find(qn('w:pPr'))
        if pPr is not None:
            # Check if paragraph has left indent (pandoc blockquote marker)
            ind = pPr.find(qn('w:ind'))
            if ind is not None:
                left = ind.get(qn('w:left'), '0')
                if int(left) >= 720:  # blockquotes get ~720 twips indent
                    # Add blue left border
                    pBdr = parse_xml(
                        f'<w:pBdr {nsdecls("w")}>'
                        f'  <w:left w:val="single" w:sz="18" w:space="8" w:color="{CALLOUT_BORDER}"/>'
                        f'</w:pBdr>'
                    )
                    pPr.append(pBdr)
                    # Light background
                    shd = parse_xml(
                        f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="F0F5FA"/>'
                    )
                    pPr.append(shd)
                    # Italic text
                    for run in para.runs:
                        run.font.italic = True
                        run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    # ── Set page margins ──
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(2.5)

    doc.save(str(OUTPUT_DOCX))
    print(f"  Post-processing complete")


def main():
    print("=" * 60)
    print("  Building Mean Field Modeling Textbook (pandoc)")
    print("=" * 60)
    print()

    # Step 1: Customize reference doc
    print("[1/3] Customizing reference styles...")
    customize_reference()

    # Step 2: Convert with pandoc
    print("[2/3] Converting Markdown → .docx via pandoc...")
    convert_md_to_docx()

    # Step 3: Post-process
    print("[3/3] Post-processing (title page, tables, figures)...")
    post_process()

    size_kb = OUTPUT_DOCX.stat().st_size / 1024
    print()
    print(f"Saved: {OUTPUT_DOCX}")
    print(f"Size:  {size_kb:.0f} KB")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
