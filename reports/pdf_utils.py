"""PDF report generation utilities."""

import os
import streamlit as st
from typing import Dict, Any

# Try to import reportlab, handle gracefully if not installed
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def generate_pdf_report(username: str, record: Dict[str, Any], output_path: str) -> None:
    """
    Generate a PDF report for a prediction record.
    
    Args:
        username: Username of the patient
        record: Dictionary containing prediction data:
            - timestamp: ISO time string
            - inputs: dict of patient features
            - probability: float
            - prediction: int (0 or 1)
            - risk_level: str
            - guidance: dict with triage_level, rationale, recommendation_text, otc_considerations
        output_path: Path where PDF should be saved
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "reportlab is not installed. Please install it using: pip install reportlab==4.0.9"
        )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Create PDF document
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Title style
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=30,
        alignment=TA_CENTER,
    )
    
    # Heading style
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#34495e"),
        spaceAfter=12,
        spaceBefore=12,
    )
    
    # Normal text style
    normal_style = styles["Normal"]
    
    # Title
    story.append(Paragraph("Diabetes Risk Prediction Report", title_style))
    story.append(Spacer(1, 0.2 * inch))
    
    # User and timestamp
    timestamp_str = record.get("timestamp", "N/A")
    if timestamp_str != "N/A":
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except:
            pass
    
    story.append(Paragraph(f"<b>User:</b> {username}", normal_style))
    story.append(Paragraph(f"<b>Date:</b> {timestamp_str}", normal_style))
    story.append(Spacer(1, 0.3 * inch))
    
    # Input features table
    story.append(Paragraph("Patient Input Features", heading_style))
    inputs = record.get("inputs", {})
    
    if inputs:
        # Prepare table data
        table_data = [["Feature", "Value"]]
        for key, value in inputs.items():
            if isinstance(value, float):
                value_str = f"{value:.2f}"
            else:
                value_str = str(value)
            table_data.append([key, value_str])
        
        # Create table
        table = Table(table_data, colWidths=[3 * inch, 2 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3498db")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 0.3 * inch))
    
    # Prediction results
    story.append(Paragraph("Prediction Results", heading_style))
    
    probability = record.get("probability", 0)
    risk_level = record.get("risk_level", "Unknown")
    prediction = record.get("prediction", 0)
    
    result_data = [
        ["Metric", "Value"],
        ["Predicted Probability", f"{probability:.4f}"],
        ["Risk Level", risk_level],
        ["Prediction", "High Risk" if prediction == 1 else "Low Risk"],
    ]
    
    result_table = Table(result_data, colWidths=[3 * inch, 2 * inch])
    result_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e74c3c") if prediction == 1 else colors.HexColor("#2ecc71")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.grey),
            ]
        )
    )
    story.append(result_table)
    story.append(Spacer(1, 0.3 * inch))
    
    # Clinical guidance
    guidance = record.get("guidance", {})
    if guidance:
        story.append(Paragraph("Clinical Guidance", heading_style))
        
        triage_level = guidance.get("triage_level", "routine")
        story.append(Paragraph(f"<b>Triage Level:</b> {triage_level.upper()}", normal_style))
        story.append(Spacer(1, 0.1 * inch))
        
        rationale = guidance.get("rationale", "")
        if rationale:
            story.append(Paragraph(f"<b>Rationale:</b> {rationale}", normal_style))
            story.append(Spacer(1, 0.1 * inch))
        
        recommendation_text = guidance.get("recommendation_text", "")
        if recommendation_text:
            # Limit to 80 words
            words = recommendation_text.split()[:80]
            truncated_text = " ".join(words)
            story.append(Paragraph(f"<b>Recommendations:</b> {truncated_text}", normal_style))
            story.append(Spacer(1, 0.1 * inch))
        
        otc_considerations = guidance.get("otc_considerations", [])
        if otc_considerations:
            story.append(Paragraph("<b>OTC Considerations (Categories Only):</b>", normal_style))
            for item in otc_considerations:
                story.append(Paragraph(f"• {item}", normal_style))
            story.append(Spacer(1, 0.1 * inch))
        
        if guidance.get("see_doctor", False):
            story.append(Paragraph("<b>Clinical consultation recommended</b>", normal_style))
    
    # Disclaimer
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("Disclaimer", heading_style))
    disclaimer_text = (
        "This report is for educational purposes only and does not constitute medical advice. "
        "No medications or dosages are provided. Please consult a licensed healthcare provider "
        "for diagnosis and treatment."
    )
    story.append(Paragraph(disclaimer_text, normal_style))
    
    # Build PDF
    doc.build(story)


def download_pdf_button(pdf_path: str, button_label: str = "📄 Download PDF Report", file_name: str = "diabetes_report.pdf") -> None:
    """
    Create a Streamlit download button for a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        button_label: Label for the download button
        file_name: Name for the downloaded file
    """
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label=button_label,
                data=pdf_file.read(),
                file_name=file_name,
                mime="application/pdf",
            )
    else:
        st.error(f"PDF file not found at {pdf_path}")

