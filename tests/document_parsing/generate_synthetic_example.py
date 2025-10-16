import io
import json
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# 1️⃣ Synthetic paper data
def generate_synthetic_paper():
    paper_data = {
        "title": "A Novel Approach to Synthetic Academic Paper Generation",
        "authors": [("Alice Smith", 1), ("Bob Johnson", 2), ("Carol Lee", 1)],
        "affiliations": {
            1: "Department of Computer Science, University of Example, Example City, Country",
            2: "AI Research Lab, TechCorp Inc., Another City, Country",
        },
        "corresponding_author": ("Alice Smith", "alice.smith@example.edu"),
        "abstract": (
            "In this work, we propose a novel approach for generating synthetic academic papers. "
            "We demonstrate the effectiveness of our pipeline for creating structured outputs in XML, Markdown, and PDF."
        ),
        "sections": [
            {
                "title": "Introduction",
                "content": "The introduction describes the problem context and motivation.",
            },
            {
                "title": "Related Work",
                "content": "We discuss related approaches and existing literature.",
            },
            {
                "title": "Methodology",
                "content": "We explain our synthetic paper generation method step by step.",
            },
            {
                "title": "Results",
                "content": "We evaluate the outputs and compare formats.",
            },
            {
                "title": "Conclusion",
                "content": "We summarize our contributions and future directions.",
            },
        ],
        "tables": [
            {
                "caption": "Example results table",
                "headers": ["Method", "Accuracy", "F1 Score"],
                "rows": [
                    ["Baseline", "85%", "0.82"],
                    ["Our Method", "92%", "0.91"],
                ],
            }
        ],
        "equations": [
            {
                "label": "eq1",
                "latex": r"E = mc^2",
                "description": "Mass–energy equivalence",
            }
        ],
    }
    return paper_data


# 2️⃣ XML generator
def generate_xml(paper_data):
    root = ET.Element("paper")
    ET.SubElement(root, "title").text = paper_data["title"]

    authors_el = ET.SubElement(root, "authors")
    for name, aff in paper_data["authors"]:
        author_el = ET.SubElement(authors_el, "author", affiliation=str(aff))
        author_el.text = name

    affiliations_el = ET.SubElement(root, "affiliations")
    for num, aff_text in paper_data["affiliations"].items():
        aff_el = ET.SubElement(affiliations_el, "affiliation", id=str(num))
        aff_el.text = aff_text

    corr_name, corr_email = paper_data["corresponding_author"]
    corr_el = ET.SubElement(root, "corresponding_author")
    ET.SubElement(corr_el, "name").text = corr_name
    ET.SubElement(corr_el, "email").text = corr_email

    ET.SubElement(root, "abstract").text = paper_data["abstract"]

    sections_el = ET.SubElement(root, "sections")
    for sec in paper_data["sections"]:
        sec_el = ET.SubElement(sections_el, "section", title=sec["title"])
        sec_el.text = sec["content"]

    return ET.tostring(root, encoding="unicode")


# 3️⃣ Markdown generator
def generate_md(paper_data):
    lines = []
    lines.append(f"# {paper_data['title']}\n")

    # Authors with superscripts
    author_line = ", ".join(f"{name}^{aff}" for name, aff in paper_data["authors"])
    lines.append(author_line + "\n")

    # Affiliations
    for num, aff_text in paper_data["affiliations"].items():
        lines.append(f"^{num} {aff_text}")
    lines.append("")

    # Corresponding author
    corr_name, corr_email = paper_data["corresponding_author"]
    lines.append(f"Corresponding author: {corr_name} ({corr_email})\n")

    # Abstract
    lines.append("## Abstract")
    lines.append(paper_data["abstract"] + "\n")

    # Sections
    for sec in paper_data["sections"]:
        lines.append(f"## {sec['title']}")
        lines.append(sec["content"] + "\n")

    return "\n".join(lines)


# 4️⃣ PDF generator


def render_latex_to_image(latex, width=4, height=1, fontsize=12):
    """
    Render LaTeX string to a PIL-compatible image buffer for ReportLab.
    width/height are in inches for matplotlib figure.
    """
    fig = plt.figure(figsize=(width, height))
    fig.text(0.5, 0.5, f"${latex}$", fontsize=fontsize, ha="center", va="center")
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(
        buf, format="png", bbox_inches="tight", pad_inches=0.1, transparent=True
    )
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf(paper_data, output_filename):
    styles = getSampleStyleSheet()
    story = []

    # ---------------- Title ----------------
    title_style = ParagraphStyle(
        "title",
        parent=styles["Title"],
        alignment=1,  # center
        spaceAfter=12,
        fontSize=16,
        leading=20,
    )
    story.append(Paragraph(paper_data["title"], title_style))
    story.append(Spacer(1, 0.2 * inch))

    # ---------------- Authors ----------------
    author_fragments = []
    for name, aff_ids in paper_data["authors"]:
        if isinstance(aff_ids, int):
            aff_ids = [aff_ids]
        superscripts = "".join(
            [f'<font size="6"><super>{num}</super></font>' for num in aff_ids]
        )
        author_fragments.append(f"{name}{superscripts}")

    author_text = ", ".join(author_fragments)
    author_style = ParagraphStyle(
        "authors", parent=styles["Normal"], alignment=1, fontSize=10, leading=12
    )
    story.append(Paragraph(author_text, author_style))
    story.append(Spacer(1, 0.05 * inch))

    # ---------------- Affiliations ----------------
    for num, aff_text in paper_data["affiliations"].items():
        aff_line = f'<font size="8"><super>{num}</super></font> {aff_text}'
        story.append(Paragraph(aff_line, author_style))
    story.append(Spacer(1, 0.2 * inch))

    # ---------------- Corresponding author ----------------
    corr_name, corr_email = paper_data["corresponding_author"]
    corr_text = f"Corresponding author: {corr_name} ({corr_email})"
    corr_style = ParagraphStyle(
        "corr",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        alignment=1,
        textColor=colors.darkgray,
    )
    story.append(Paragraph(corr_text, corr_style))
    story.append(Spacer(1, 0.3 * inch))

    # ---------------- Abstract ----------------
    story.append(Paragraph("Abstract", styles["Heading2"]))
    story.append(Spacer(1, 0.05 * inch))
    story.append(Paragraph(paper_data["abstract"], styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    # ---------------- Sections ----------------
    for section in paper_data["sections"]:
        story.append(Paragraph(section["title"], styles["Heading2"]))
        story.append(Spacer(1, 0.05 * inch))
        story.append(Paragraph(section["content"], styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))

    # ---------------- Tables ----------------
    for table in paper_data.get("tables", []):
        story.append(Paragraph(f"Table: {table['caption']}", styles["Heading3"]))
        t_data = [table["headers"]] + table["rows"]
        t = Table(t_data, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 0.2 * inch))

    # ---------------- Equations ----------------
    for eq in paper_data.get("equations", []):
        story.append(
            Paragraph(
                f"Equation ({eq['label']}): {eq.get('description', '')}",
                styles["Heading3"],
            )
        )
        buf = render_latex_to_image(eq["latex"], width=4, height=1, fontsize=12)
        img = Image(buf)
        # Let ReportLab scale proportionally
        img._restrictSize(5 * inch, 1.5 * inch)
        story.append(img)
        story.append(Spacer(1, 0.2 * inch))

    # ---------------- Build PDF ----------------
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )
    doc.build(story)
    print(f"PDF generated at {output_filename}")


# ✅ Example usage
paper_data = generate_synthetic_paper()
with open("expected_paper.json", "w", encoding="utf-8") as f:
    json.dump(paper_data, f, indent=4)
# XML
xml_content = generate_xml(paper_data)
with open("paper.xml", "w", encoding="utf-8") as f:
    f.write(xml_content)

# Markdown
md_content = generate_md(paper_data)
with open("paper.md", "w", encoding="utf-8") as f:
    f.write(md_content)

# PDF
generate_pdf(paper_data, "paper.pdf")
