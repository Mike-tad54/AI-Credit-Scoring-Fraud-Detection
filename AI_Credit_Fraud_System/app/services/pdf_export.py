"""PDF export service — credit score reports and fraud detection reports.

Implements:
    * FR-08 — credit score report exportable as PDF
    * FR-13 — fraud detection report exportable as PDF

Uses ReportLab's Platypus framework (high-level page layout) so the
output is structured into flowables (Paragraphs, Tables, Spacers) rather
than manual canvas drawing. PDFs are streamed back to the browser via a
``Response`` with ``Content-Disposition: attachment``.
"""
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle)
from flask import Response


def _styles():
    """Build the paragraph style sheet used by all PDF reports."""
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(name="Title2",
                            parent=base["Title"],
                            fontSize=18,
                            textColor=colors.HexColor("#003366"),
                            spaceAfter=12))
    base.add(ParagraphStyle(name="H2",
                            parent=base["Heading2"],
                            fontSize=13,
                            textColor=colors.HexColor("#003366"),
                            spaceBefore=10,
                            spaceAfter=6))
    base.add(ParagraphStyle(name="Body2",
                            parent=base["BodyText"],
                            fontSize=10,
                            leading=14))
    return base


def _risk_color(risk_level):
    """Map a risk-level string to its documented color (UC-S06)."""
    return {
        "LOW": colors.HexColor("#1E7B34"),
        "MEDIUM": colors.HexColor("#FFC000"),
        "HIGH": colors.HexColor("#C00000")
    }.get(risk_level, colors.black)


def generate_credit_pdf(score, application, shap):
    """Build the credit score PDF (FR-08) and return a Flask ``Response``.

    Args:
        score (CreditScore): the persisted score record.
        application (CreditApplication): the parent application.
        shap (dict): top-5 SHAP feature contributions.

    Returns:
        Response: Flask response streaming the PDF as an attachment.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(buf,
                            pagesize=A4,
                            leftMargin=2 * cm,
                            rightMargin=2 * cm,
                            topMargin=2 * cm,
                            bottomMargin=2 * cm,
                            title=f"Credit_Report_{score.score_id}")
    s = _styles()
    elements = []

    elements.append(Paragraph("AI Credit Scoring Report", s["Title2"]))
    elements.append(
        Paragraph(
            f"Unity University — AI Credit &amp; Fraud System | "
            f"Generated: {score.generated_at.strftime('%d %b %Y %H:%M')}",
            s["Body2"]))
    elements.append(Spacer(1, 0.5 * cm))

    # ── Score summary block ───────────────────────────────────────────
    elements.append(Paragraph("Score Summary", s["H2"]))
    summary = [[
        "Applicant", application.full_name, "Region",
        application.region or "N/A"
    ],
               [
                   "Credit Score", f"{score.score:.1f} / 100", "Risk Level",
                   score.risk_level
               ],
               [
                   "Default Probability", f"{score.probability*100:.1f}%",
                   "Model Version", score.model_version
               ],
               [
                   "Loan Amount",
                   f"ETB {application.loan_amount:,.0f}", "Loan Purpose",
                   application.loan_purpose or "N/A"
               ]]
    t = Table(summary, colWidths=[4 * cm, 5 * cm, 4 * cm, 4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
        ("TEXTCOLOR", (3, 1), (3, 1), _risk_color(score.risk_level)),
        ("FONTNAME", (3, 1), (3, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.5 * cm))

    # ── BR-01 thresholds reference ────────────────────────────────────
    elements.append(Paragraph("Risk Tier Reference (BR-01)", s["H2"]))
    tiers = [["Score Range", "Risk Level", "Recommendation"],
             ["70 – 100", "LOW", "Recommended for Approval"],
             ["50 – 69", "MEDIUM", "Referred for Manual Review"],
             ["0 – 49", "HIGH", "Recommended for Decline"]]
    tt = Table(tiers, colWidths=[4 * cm, 4 * cm, 9 * cm])
    tt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F5F7FA")]),
    ]))
    elements.append(tt)
    elements.append(Spacer(1, 0.5 * cm))

    # ── SHAP explanation (FR-05, NFR-08) ──────────────────────────────
    elements.append(
        Paragraph("AI Explanation — Top 5 SHAP Features (FR-05)", s["H2"]))
    if shap:
        shap_rows = [["Feature", "Impact Score"]] + [[
            k.replace("_", " ").title(), f"{v:+.4f}"
        ] for k, v in shap.items()]
        st = Table(shap_rows, colWidths=[8 * cm, 4 * cm])
        st.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E75B6")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#E8F0F8")]),
        ]))
        elements.append(st)
    else:
        elements.append(Paragraph("(No SHAP data available)", s["Body2"]))

    elements.append(Spacer(1, 0.5 * cm))
    elements.append(
        Paragraph(
            "Generated by the AI Credit Scoring &amp; Fraud Detection System "
            "— Unity University, 2025/2026. This report is for academic "
            "demonstration only.", s["Body2"]))

    doc.build(elements)
    buf.seek(0)
    return Response(
        buf,
        mimetype="application/pdf",
        headers={
            "Content-Disposition":
            f"attachment;filename=credit_report_{score.score_id}.pdf"
        },
    )


def generate_fraud_pdf(batch_id, rows, totals, overrides=None):
    """Build the fraud detection PDF (FR-13) and return a Flask ``Response``.

    Args:
        batch_id (str): the batch identifier.
        rows (list[dict]): per-transaction result rows.
        totals (dict): ``{total, legit, susp, fraud}`` summary counts.
        overrides (list[dict] | None): per-transaction override info, each
            dict containing ``{txn_id, original_label, new_label, reason,
            overridden_at}``. If provided, an "Officer Override" section
            is appended to the PDF for audit transparency (Bug fix 1.3).

    Returns:
        Response: Flask response streaming the PDF as an attachment.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(buf,
                            pagesize=A4,
                            leftMargin=2 * cm,
                            rightMargin=2 * cm,
                            topMargin=2 * cm,
                            bottomMargin=2 * cm,
                            title=f"Fraud_Report_{batch_id}")
    s = _styles()
    elements = []

    elements.append(Paragraph("AI Fraud Detection Report", s["Title2"]))
    elements.append(
        Paragraph(
            f"Unity University — AI Credit &amp; Fraud System | "
            f"Batch ID: <b>{batch_id}</b>", s["Body2"]))
    elements.append(Spacer(1, 0.5 * cm))

    # ── Summary block ─────────────────────────────────────────────────
    elements.append(Paragraph("Batch Summary", s["H2"]))
    summary = [
        ["Total Transactions", str(totals["total"])],
        ["Legitimate", str(totals["legit"])],
        ["Suspicious", str(totals["susp"])],
        ["Fraud Detected", str(totals["fraud"])],
        [
            "Fraud Rate",
            f"{(totals['fraud']/totals['total']*100) if totals['total'] else 0:.2f}%"
        ],
    ]
    # If officer overrides exist, show AI-vs-officer comparison.
    if overrides:
        ai_fraud_count = sum(1 for o in overrides
                             if o.get("original_label") == "FRAUD")
        ai_legit_count = totals["total"] - ai_fraud_count
        summary.append(["AI Detected as Fraud", str(ai_fraud_count)])
        summary.append(["Officer Overridden", str(len(overrides))])
    t = Table(summary, colWidths=[6 * cm, 4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.5 * cm))

    # ── BR-02 threshold reference ────────────────────────────────────
    elements.append(Paragraph("Classification Thresholds (BR-02)", s["H2"]))
    th = [["Probability Range", "Label", "Action"],
          ["> 0.50", "FRAUD", "Blocked + Officer Alerted"],
          ["0.30 – 0.50", "SUSPICIOUS", "Flagged for Review"],
          ["< 0.30", "LEGITIMATE", "Approved"]]
    tht = Table(th, colWidths=[4 * cm, 4 * cm, 9 * cm])
    tht.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#C00000")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#FCE4EC")]),
    ]))
    elements.append(tht)
    elements.append(Spacer(1, 0.5 * cm))

    # ── Per-transaction table ────────────────────────────────────────
    elements.append(Paragraph("Transaction Results", s["H2"]))
    txn_rows = [["Txn ID", "Amount (ETB)", "Type", "Label", "Probability"]]
    for r in rows:
        txn_rows.append([
            f"#{r['txn_id']}",
            f"{r['amount']:,.2f}",
            r.get("txn_type") or "N/A",
            r["label"],
            f"{r['probability']:.4f}",
        ])
    tr = Table(txn_rows, colWidths=[2.5 * cm, 3.5 * cm, 3 * cm, 3 * cm, 3 * cm])
    tr.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F5F7FA")]),
    ]))
    elements.append(tr)

    # ── Bug fix 1.3: Officer Override section ────────────────────────
    # If any transactions were overridden by officers, show the override
    # history for audit transparency. This ensures the PDF reflects the
    # full decision trail, not just the final label.
    if overrides:
        elements.append(Spacer(1, 0.5 * cm))
        elements.append(
            Paragraph("Officer Override History (Audit Trail)", s["H2"]))
        ov_rows = [[
            "Txn ID", "AI Original Label", "Officer Corrected Label",
            "Reason"
        ]]
        for o in overrides:
            ov_rows.append([
                f"#{o.get('txn_id', '?')}",
                o.get("original_label", "?"),
                o.get("new_label", "?"),
                (o.get("reason") or "(no reason provided)")[:60],
            ])
        ovt = Table(ov_rows,
                    colWidths=[2 * cm, 3.5 * cm, 3.5 * cm, 8 * cm])
        ovt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FFC000")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25,
             colors.HexColor("#CCCCCC")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#FFF8DC")]),
        ]))
        elements.append(ovt)
        elements.append(Spacer(1, 0.2 * cm))
        elements.append(
            Paragraph(
                "Override records are preserved in the audit log (BR-05) "
                "with officer identity, timestamp, and reason.",
                s["Body2"]))

    elements.append(Spacer(1, 0.5 * cm))
    elements.append(
        Paragraph(
            "Generated by the AI Credit Scoring &amp; Fraud Detection System "
            "— Unity University, 2025/2026. This report is for academic "
            "demonstration only.", s["Body2"]))

    doc.build(elements)
    buf.seek(0)
    return Response(
        buf,
        mimetype="application/pdf",
        headers={
            "Content-Disposition":
            f"attachment;filename=fraud_report_{batch_id}.pdf"
        },
    )
