from __future__ import annotations

from io import BytesIO
from typing import Any, Optional, Sequence
from xml.sax.saxutils import escape

from protein_visualizer.services.snapshot import snapshot_to_summary_lines

REPORTLAB_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_AVAILABLE = True
except ModuleNotFoundError:
    pass

PDF_EXPORT_AVAILABLE = REPORTLAB_AVAILABLE


def _register_chinese_font() -> str:
    font_name = "STSong-Light"
    try:
        pdfmetrics.getFont(font_name)
    except Exception:
        pdfmetrics.registerFont(UnicodeCIDFont(font_name))
    return font_name


def _format_volume_text(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):,.1f} A³"
    except (TypeError, ValueError):
        return "-"


def _metric_table_rows(metrics: Sequence[tuple[str, Any]], font_name: str):
    cells = []
    for label, value in metrics:
        cells.append(
            [
                Paragraph(f"<font size='9' color='#64748b'>{escape(str(label))}</font>", _paragraph_style(font_name, 9, colors.HexColor("#64748b"))),
                Paragraph(f"<font size='15' color='#0f1724'>{escape(str(value))}</font>", _paragraph_style(font_name, 15, colors.HexColor("#0f1724"))),
            ]
        )
    return cells


def _paragraph_style(font_name: str, size: int = 10, text_color=None, alignment=None, leading: Optional[int] = None) -> ParagraphStyle:
    if not REPORTLAB_AVAILABLE:
        raise ModuleNotFoundError("No module named 'reportlab'. PDF export requires reportlab to be installed.")
    if alignment is None:
        alignment = TA_LEFT
    if text_color is None:
        text_color = colors.HexColor("#0f1724")
    return ParagraphStyle(
        name=f"{font_name}_{size}_{alignment}",
        fontName=font_name,
        fontSize=size,
        leading=leading or int(size * 1.35),
        textColor=text_color,
        alignment=alignment,
        spaceAfter=0,
        spaceBefore=0,
    )


def _section_heading(text: str, font_name: str):
    return Paragraph(
        escape(text),
        _paragraph_style(font_name, 13, colors.HexColor("#0f1724"), alignment=TA_LEFT, leading=18),
    )


def _build_preview_table(title: str, rows: Sequence[dict[str, Any]], columns: Sequence[str], font_name: str):
    if not rows:
        return []

    story = [Paragraph(escape(title), _paragraph_style(font_name, 11, colors.HexColor("#0f1724"))), Spacer(1, 2 * mm)]
    table_data = []
    header = [Paragraph(f"<font color='#475569'>{escape(column)}</font>", _paragraph_style(font_name, 9, colors.HexColor("#475569"))) for column in columns]
    table_data.append(header)

    for row in rows:
        table_data.append(
            [
                Paragraph(f"<font color='#0f1724'>{escape(str(row.get(column, '-')))}</font>", _paragraph_style(font_name, 9, colors.HexColor("#0f1724")))
                for column in columns
            ]
        )

    table = Table(table_data, repeatRows=1, colWidths=[None] * len(columns))
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d4ed8")),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#eff6ff")),
                ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#334155")),
                ("GRID", (0, 1), (-1, -1), 0.4, colors.HexColor("#dbe3ee")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table)
    return story


def _story_report_text(report_text: str, font_name: str):
    story = []
    body_style = _paragraph_style(font_name, 10, colors.HexColor("#0f1724"), alignment=TA_LEFT, leading=15)
    section_style = _paragraph_style(font_name, 11, colors.HexColor("#0f1724"), alignment=TA_LEFT, leading=16)

    for line in report_text.splitlines():
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 3 * mm))
            continue
        if stripped.endswith(":") and len(stripped) < 40:
            story.append(Spacer(1, 2 * mm))
            story.append(Paragraph(escape(stripped), section_style))
            continue
        if set(stripped) == {"="} or set(stripped) == {"-"}:
            continue
        story.append(Paragraph(escape(stripped), body_style))
    return story


def build_simple_pdf(report_text: str, snapshot: Optional[dict[str, Any]] = None) -> bytes:
    if not REPORTLAB_AVAILABLE:
        raise ModuleNotFoundError("No module named 'reportlab'. PDF export requires reportlab to be installed.")

    font_name = _register_chinese_font()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=14 * mm,
        title=(snapshot or {}).get("title", "ProteinInsight 分析报告"),
        author="ProteinInsight",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name="ProteinTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=20,
        leading=24,
        textColor=colors.white,
        alignment=TA_LEFT,
        spaceAfter=0,
        spaceBefore=0,
    )
    subtitle_style = ParagraphStyle(
        name="ProteinSubtitle",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#e0f2fe"),
        alignment=TA_LEFT,
        spaceAfter=0,
        spaceBefore=0,
    )
    section_style = ParagraphStyle(
        name="ProteinSection",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#0f1724"),
        spaceBefore=3,
        spaceAfter=5,
    )

    summary_snapshot = snapshot or {}
    summary = summary_snapshot.get("summary") or {}
    summary_lines = snapshot_to_summary_lines(summary_snapshot) if summary_snapshot else []
    joint_candidate_preview = summary_snapshot.get("joint_candidate_preview") or []

    story = []

    def decorate(canvas, document):
        canvas.saveState()
        width, height = A4
        canvas.setFillColor(colors.HexColor("#f6f9fc"))
        canvas.rect(0, 0, width, height, stroke=0, fill=1)
        canvas.setFillColor(colors.HexColor("#2b6cb0"))
        canvas.rect(0, height - 30 * mm, width, 30 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.HexColor("#e8efff"))
        canvas.setFont(font_name, 8)
        canvas.drawRightString(width - 16 * mm, height - 10 * mm, f"Page {document.page}")
        canvas.drawString(16 * mm, 10 * mm, "ProteinInsight report export")
        canvas.restoreState()

    if summary_snapshot:
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph(escape(summary_snapshot.get("title", "ProteinInsight 分析报告")), title_style))
        story.append(Spacer(1, 1.5 * mm))
        story.append(Paragraph(escape(summary_snapshot.get("generated_at", "")), subtitle_style))
        story.append(Spacer(1, 5 * mm))

        protein_volume = summary.get("protein_volume")
        if protein_volume is None:
            protein_volume = summary_snapshot.get("protein_volume")

        metrics = [
            ("残基总数", summary.get("residue_count", "-")),
            ("平均能量", summary.get("mean_energy", "-")),
            ("蛋白质体积", _format_volume_text(protein_volume)),
            ("热点残基", summary_snapshot.get("hotspot_count", 0)),
            ("口袋条目", summary_snapshot.get("pocket_rows", 0)),
            ("能量来源", summary.get("energy_source") or "未标注"),
        ]
        metric_rows = _metric_table_rows(metrics, font_name)
        metric_table = Table(metric_rows, colWidths=[38 * mm, 42 * mm])
        metric_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                    ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#dbe3ee")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dbe3ee")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(metric_table)
        story.append(Spacer(1, 5 * mm))

        if summary_lines:
            story.append(Paragraph("自动摘要", section_style))
            for line in summary_lines:
                story.append(Paragraph(f"• {escape(str(line))}", _paragraph_style(font_name, 10, colors.HexColor("#334155"), leading=14)))
            story.append(Spacer(1, 4 * mm))

        top_hotspots = summary_snapshot.get("top_hotspots") or []
        pocket_summary = summary_snapshot.get("pocket_summary") or []
        joint_candidate_preview = summary_snapshot.get("joint_candidate_preview") or []

        if top_hotspots:
            story.append(Paragraph("热点预览", section_style))
            story.extend(_build_preview_table("Top 热点", top_hotspots, ["label", "delta_total", "hotspot_rank"], font_name))
            story.append(Spacer(1, 4 * mm))

        if pocket_summary:
            story.append(Paragraph("口袋预览", section_style))
            pocket_columns = [
                "pocket_id",
                "evidence_quality_label",
                "evidence_quality_score",
                "detection_route",
                "consensus_methods",
                "method_vote_count",
                "consensus_score",
                "volume",
                "score",
                "hotspot_count",
            ]
            available_columns = [column for column in pocket_columns if any(column in row for row in pocket_summary)]
            story.extend(_build_preview_table("Pocket 概览", pocket_summary, available_columns or pocket_columns, font_name))
            story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("文本分析报告", section_style))
    if joint_candidate_preview:
        story.append(Paragraph("联合推荐预览", section_style))
        joint_columns = [
            "recommendation_rank",
            "pocket_id",
            "recommendation_label",
            "recommendation_score",
            "recommendation_action",
            "evidence_quality_label",
            "smart_rank_label",
            "triple_overlap_count",
        ]
        available_columns = [column for column in joint_columns if any(column in row for row in joint_candidate_preview)]
        story.extend(_build_preview_table("Joint Recommendation 概览", joint_candidate_preview, available_columns or joint_columns, font_name))
        story.append(Spacer(1, 4 * mm))

    story.extend(_story_report_text(report_text, font_name))

    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    return buffer.getvalue()
