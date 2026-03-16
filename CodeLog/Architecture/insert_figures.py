"""
Insert analysis figures into MATHEMATICAL_MODEL.docx.
Takes the pandoc-generated docx (with native OMML equations) and adds all
plots from results/analysis/ into the appropriate sections.
"""
import os
import json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DOCX_IN = os.path.join(HERE, 'MATHEMATICAL_MODEL.docx')
DOCX_OUT = os.path.join(HERE, 'MATHEMATICAL_MODEL.docx')
ANALYSIS = os.path.join(ROOT, 'results', 'analysis')

FIG_WIDTH = Inches(5.5)
FIG_WIDTH_HALF = Inches(3.0)
FIG_WIDTH_SMALL = Inches(2.5)


def add_figure(doc, img_path, caption, width=FIG_WIDTH, after_paragraph=None):
    """Add a figure with caption after a specific paragraph or at end."""
    if not os.path.isfile(img_path):
        print(f"  WARNING: missing {img_path}")
        return None

    # Image paragraph
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(2)
    run = p_img.add_run()
    run.add_picture(img_path, width=width)

    # Caption paragraph
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_after = Pt(10)
    run = p_cap.add_run(caption)
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(80, 80, 80)
    run.italic = True

    return p_img


def find_paragraph_index(doc, text_contains):
    """Find the index of the first paragraph containing the given text."""
    for i, p in enumerate(doc.paragraphs):
        if text_contains in p.text:
            return i
    return -1


def find_heading(doc, heading_text):
    """Find paragraph index of a heading containing the text."""
    for i, p in enumerate(doc.paragraphs):
        if p.style.name.startswith('Heading') and heading_text in p.text:
            return i
    return -1


def insert_after_paragraph(doc, para_index, img_path, caption, width=FIG_WIDTH):
    """Insert figure after a specific paragraph by index.

    Works by appending the figure at the end and then moving the XML elements.
    """
    if not os.path.isfile(img_path):
        print(f"  WARNING: missing {img_path}")
        return

    # Add image paragraph
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(2)
    run = p_img.add_run()
    run.add_picture(img_path, width=width)

    # Add caption
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_after = Pt(10)
    run = p_cap.add_run(caption)
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(80, 80, 80)
    run.italic = True

    # Move the two new paragraphs to after para_index
    body = doc.element.body
    ref_element = doc.paragraphs[para_index]._element

    # Move caption first (it'll be after img)
    body.remove(p_cap._element)
    ref_element.addnext(p_cap._element)

    # Then move image (it'll be before caption = right after ref)
    body.remove(p_img._element)
    ref_element.addnext(p_img._element)


def main():
    print("Loading pandoc-generated document...")
    doc = Document(DOCX_IN)

    # Load analysis summaries
    mf_path = os.path.join(ANALYSIS, 'mean_field_summary.json')
    cg_path = os.path.join(ANALYSIS, 'coarse_grain_summary.json')
    dim_path = os.path.join(ANALYSIS, 'dimensionless_summary.json')
    td_path = os.path.join(ANALYSIS, 'tissue_descriptors.json')

    with open(mf_path) as f:
        mf_data = json.load(f)
    with open(cg_path) as f:
        cg_data = json.load(f)

    # ================================================================
    # Strategy: Find each results subsection heading, then insert
    # figures after the last paragraph of that subsection (before the
    # next heading). We build the document by appending figures at the
    # end, then moving them into position using XML manipulation.
    #
    # For simplicity and reliability, we'll find each section and
    # insert figures right before the next section heading.
    # ================================================================

    # Collect section boundaries
    headings = []
    for i, p in enumerate(doc.paragraphs):
        if p.style.name.startswith('Heading'):
            headings.append((i, p.text, p.style.name))

    def find_section_end(section_text):
        """Find the paragraph index just before the next heading after section_text."""
        found = False
        for idx, text, style in headings:
            if section_text in text:
                found = True
                continue
            if found:
                return idx - 1
        # If it's the last section, return last paragraph
        return len(doc.paragraphs) - 1

    # ================================================================
    # 10.1 Mean-Field Model Fitting -- insert best-fit plots
    # ================================================================
    print("\nInserting mean-field plots...")

    # Sort runs by R² (best first)
    mf_ranked = []
    for name, v in mf_data.items():
        if isinstance(v, dict) and 'R2' in v:
            mf_ranked.append((name, v['R2'], v.get('E_modulus', '?')))
    mf_ranked.sort(key=lambda x: -x[1])

    sec_end = find_section_end('10.1')
    fig_num = 1

    # Insert top 5 best-fit compaction plots
    for name, r2, E in mf_ranked[:5]:
        img = os.path.join(ANALYSIS, 'mean_field', name, 'compaction_fit.png')
        caption = f"Figure {fig_num}. {name} compaction fit (E = {E} kPa, R\u00b2 = {r2:.3f})."
        insert_after_paragraph(doc, sec_end, img, caption)
        sec_end += 2  # account for inserted paragraphs
        fig_num += 1

    # Insert stress balance for best run
    best_name = mf_ranked[0][0]
    img = os.path.join(ANALYSIS, 'mean_field', best_name, 'stress_balance.png')
    caption = f"Figure {fig_num}. {best_name} stress balance -- cell stress vs. resistance."
    insert_after_paragraph(doc, sec_end, img, caption)
    sec_end += 2
    fig_num += 1

    # Permeability evolution for best run
    img = os.path.join(ANALYSIS, 'mean_field', best_name, 'permeability_evolution.png')
    caption = f"Figure {fig_num}. {best_name} Kozeny-Carman permeability evolution."
    insert_after_paragraph(doc, sec_end, img, caption)
    sec_end += 2
    fig_num += 1

    # Phase evolution for best run
    img = os.path.join(ANALYSIS, 'mean_field', best_name, 'phase_evolution.png')
    caption = f"Figure {fig_num}. {best_name} three-phase volume fraction evolution."
    insert_after_paragraph(doc, sec_end, img, caption)
    sec_end += 2
    fig_num += 1

    # ================================================================
    # 10.3 Dimensionless Analysis -- all 8 plots
    # ================================================================
    print("Inserting dimensionless analysis plots...")

    dim_dir = os.path.join(ANALYSIS, 'dimensionless')
    dim_plots = [
        ('beta_collapse.png', 'Data collapse by motor-clutch engagement ratio \u03b2'),
        ('Ca_scaling.png', 'Compaction scaling with cellular capillary number Ca'),
        ('jamming_diagram.png', 'Jamming diagram: compaction vs. \u03c6_solid/\u03c6_J'),
        ('composition_effects.png', 'Effect of composition ratio on compaction'),
        ('factor_main_effects.png', 'DOE main effects of 5 factors on \u0394\u03c6_f'),
        ('factor_interactions.png', 'DOE factor interaction effects'),
        ('dimensionless_dashboard.png', 'Dimensionless analysis dashboard (all groups)'),
        ('phase_space.png', 'Phase space trajectories across DOE'),
    ]

    sec_end = find_section_end('10.3')
    for fname, desc in dim_plots:
        img = os.path.join(dim_dir, fname)
        caption = f"Figure {fig_num}. {desc}."
        insert_after_paragraph(doc, sec_end, img, caption)
        sec_end += 2
        fig_num += 1

    # ================================================================
    # 10.5 Architectural Distance -- both plots
    # ================================================================
    print("Inserting architectural distance plots...")

    arch_dir = os.path.join(ANALYSIS, 'arch_distance')
    arch_plots = [
        ('distance_heatmap.png', 'Architectural distance heatmap: DOE runs vs. 7 organ targets'),
        ('landscape_trabecular_bone.png', 'Optimization landscape for trabecular bone target'),
    ]

    sec_end = find_section_end('10.5')
    for fname, desc in arch_plots:
        img = os.path.join(arch_dir, fname)
        caption = f"Figure {fig_num}. {desc}."
        insert_after_paragraph(doc, sec_end, img, caption)
        sec_end += 2
        fig_num += 1

    # ================================================================
    # Appendix: Per-Run Mean-Field Analysis
    # ================================================================
    print("Adding appendix with all per-run plots...")

    # Add page break and appendix heading
    doc.add_page_break()
    h = doc.add_heading('Appendix A: Per-Run Mean-Field Analysis', level=1)

    p = doc.add_paragraph()
    run = p.add_run(
        'This appendix contains the mean-field compaction model results for all '
        '23 DOE runs. Each run shows four panels: (a) compaction fit, '
        '(b) phase evolution, (c) permeability evolution, (d) stress balance.'
    )
    run.font.size = Pt(10)

    for i in range(1, 24):
        name = f"DOE_{i:02d}"
        run_dir = os.path.join(ANALYSIS, 'mean_field', name)
        if not os.path.isdir(run_dir):
            continue

        mf_info = mf_data.get(name, {})
        r2 = mf_info.get('R2', 'N/A')
        E = mf_info.get('E_modulus', '?')
        dphi = mf_info.get('phi_f_change', 0)

        r2_str = f"{r2:.3f}" if isinstance(r2, float) else str(r2)
        dphi_str = f"{dphi:.4f}" if isinstance(dphi, float) else str(dphi)

        # Run sub-heading
        doc.add_heading(f'{name} (E = {E} kPa, R\u00b2 = {r2_str}, \u0394\u03c6_f = {dphi_str})', level=2)

        plots = ['compaction_fit.png', 'phase_evolution.png',
                 'permeability_evolution.png', 'stress_balance.png']
        labels = ['(a) Compaction fit', '(b) Phase evolution',
                  '(c) Permeability evolution', '(d) Stress balance']

        for plot_name, label in zip(plots, labels):
            img = os.path.join(run_dir, plot_name)
            if os.path.isfile(img):
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.paragraph_format.space_before = Pt(4)
                p_img.paragraph_format.space_after = Pt(1)
                r = p_img.add_run()
                r.add_picture(img, width=FIG_WIDTH)

                p_cap = doc.add_paragraph()
                p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_cap.paragraph_format.space_after = Pt(6)
                r = p_cap.add_run(f"Figure {fig_num}. {name} -- {label}.")
                r.font.size = Pt(9)
                r.font.color.rgb = RGBColor(80, 80, 80)
                r.italic = True
                fig_num += 1

    # ================================================================
    # Save
    # ================================================================
    doc.save(DOCX_OUT)
    total_figs = fig_num - 1
    print(f"\nSaved: {DOCX_OUT}")
    print(f"  Total figures inserted: {total_figs}")
    print(f"  Paragraphs: {len(doc.paragraphs)}")
    print(f"  Tables: {len(doc.tables)}")


if __name__ == '__main__':
    main()
