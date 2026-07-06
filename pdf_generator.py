import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

REPORTS_DIR = "./reports"


def convert_markdown_to_html_tags(text: str) -> str:
    """Helper to convert basic markdown formatting (bold, italic) to HTML tags for ReportLab."""
    # Convert bold **text** to <b>text</b>
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    # Convert italic *text* to <i>text</i>
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
    # Convert inline code `code` to <code>code</code>
    text = re.sub(r"`(.*?)`", r"<font face='Courier'>\1</font>", text)
    return text


def generate_pdf_report(markdown_content: str, filename: str) -> str:
    """Converts the clinical decision support report from Markdown to a structured PDF."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    filepath = os.path.join(REPORTS_DIR, filename)

    doc = SimpleDocTemplate(
        filepath,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1e3a8a"),  # Deep Navy Blue
        spaceAfter=15,
        alignment=0,  # Left align
    )
    h1_style = ParagraphStyle(
        "Header1",
        parent=styles["Heading1"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        "Header2",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0f766e"),  # Teal
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#374151"),  # Dark gray
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "BulletCustom",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#374151"),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4,
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#b91c1c"),  # Warning Red
        spaceAfter=8,
        backColor=colors.HexColor("#fef2f2"),
        borderColor=colors.HexColor("#fecaca"),
        borderWidth=1,
        borderPadding=6,
    )

    story = []

    # Title
    story.append(Paragraph("Clinical Decision Support Report", title_style))
    story.append(Spacer(1, 10))

    lines = markdown_content.split("\n")
    in_disclaimer = False
    disclaimer_text = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Handle disclaimer sections or headers
        if "DISCLAIMER" in stripped.upper() or "IMPORTANT" in stripped.upper():
            in_disclaimer = True

        # Process Headers
        if stripped.startswith("# "):
            title_text = stripped[2:].strip()
            story.append(Paragraph(convert_markdown_to_html_tags(title_text), h1_style))
        elif stripped.startswith("## "):
            title_text = stripped[3:].strip()
            story.append(Paragraph(convert_markdown_to_html_tags(title_text), h1_style))
        elif stripped.startswith("### "):
            title_text = stripped[4:].strip()
            story.append(Paragraph(convert_markdown_to_html_tags(title_text), h2_style))
        # Process Lists
        elif stripped.startswith("- ") or stripped.startswith("* "):
            content = convert_markdown_to_html_tags(stripped[2:])
            story.append(Paragraph(f"&bull; {content}", bullet_style))
        elif re.match(r"^\d+\.\s", stripped):
            # Numbered list
            match = re.match(r"^(\d+)\.\s(.*)", stripped)
            num = match.group(1)
            content = convert_markdown_to_html_tags(match.group(2))
            story.append(Paragraph(f"{num}. {content}", bullet_style))
        # Process Paragraphs
        else:
            content = convert_markdown_to_html_tags(stripped)
            if in_disclaimer:
                disclaimer_text.append(content)
                if len(disclaimer_text) >= 2 or "licensed medical professionals" in content:
                    story.append(
                        Paragraph(" ".join(disclaimer_text), disclaimer_style)
                    )
                    disclaimer_text = []
                    in_disclaimer = False
            else:
                story.append(Paragraph(content, body_style))

    # Fallback for remaining disclaimer text
    if disclaimer_text:
        story.append(Paragraph(" ".join(disclaimer_text), disclaimer_style))

    # Add a nice footer line
    story.append(Spacer(1, 20))
    divider_data = [[""]]
    divider_table = Table(divider_data, colWidths=[doc.width], rowHeights=[1])
    divider_table.setStyle(
        TableStyle(
            [
                ("LINEABOVE", (0, 0), (-1, -1), 1, colors.HexColor("#e5e7eb")),
            ]
        )
    )
    story.append(divider_table)
    story.append(Spacer(1, 5))
    story.append(
        Paragraph(
            "Confidential Clinical Decision Support Document - For Use By Licensed Providers Only",
            ParagraphStyle(
                "FooterText", parent=body_style, fontSize=8, alignment=1, textColor=colors.HexColor("#9ca3af")
            ),
        )
    )

    doc.build(story)
    return filepath
