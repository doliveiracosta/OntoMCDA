"""PDF export for OntoMCDA recommendations."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import BinaryIO
from xml.sax.saxutils import escape

import pandas as pd

from .constants import APP_NAME, APP_OWNER_LABEL, ATTR_LABELS, PREMISE_ATTRS


def write_pdf_report(
    output: str | BinaryIO,
    *,
    problem_text: str,
    premises: dict,
    nlp_metrics: dict | None = None,
    premise_diagnostics: list[dict] | None = None,
    recommendations: pd.DataFrame,
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output, pagesize=A4, rightMargin=1.5 * cm, leftMargin=1.5 * cm)
    story = []

    def paragraph(text: object, style: str = "Normal") -> Paragraph:
        return Paragraph(escape(str(text)), styles[style])

    def table(rows: list[list[object]], widths: list[float]) -> Table:
        wrapped = [[paragraph(value) for value in row] for row in rows]
        t = Table(wrapped, colWidths=widths, repeatRows=1)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9ca3af")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return t

    story.append(paragraph(APP_NAME, "Title"))
    story.append(paragraph("Relatorio de recomendacao semantica de metodos multicriterio", "Heading2"))
    story.append(paragraph(APP_OWNER_LABEL))
    story.append(paragraph(f"Data de geracao: {datetime.now().strftime('%d/%m/%Y %H:%M')}"))
    story.append(Spacer(1, 10))

    story.append(paragraph("1. Descricao textual do problema", "Heading1"))
    story.append(paragraph(problem_text or "Nao informado."))
    story.append(Spacer(1, 10))

    story.append(paragraph("2. Premissas inferidas", "Heading1"))
    premise_rows = [["Premissa", "Valor inferido"]]
    for attr in PREMISE_ATTRS:
        value = premises.get(attr)
        premise_rows.append([ATTR_LABELS.get(attr, attr), value or "Nao inferido"])
    story.append(table(premise_rows, [6.5 * cm, 9.2 * cm]))
    story.append(Spacer(1, 10))

    story.append(paragraph("3. Metricas quantitativas do PLN", "Heading1"))
    metrics = nlp_metrics or {}
    metric_rows = [
        ["Metrica", "Valor", "Interpretacao"],
        [
            "ICS-PLN",
            f"{float(metrics.get('ics_pln', 0.0)):.1f}%",
            "Indice de cobertura semantica das premissas inferidas.",
        ],
        [
            "Premissas inferidas",
            f"{int(metrics.get('inferred_premises', 0))}/{int(metrics.get('total_premises', len(PREMISE_ATTRS)))}",
            "Quantidade de premissas identificadas pelo PLN.",
        ],
        [
            "TNI-PLN",
            f"{float(metrics.get('tni_pln', 0.0)):.1f}%",
            "Taxa de premissas nao inferidas pelo PLN.",
        ],
        [
            "IET-PLN",
            f"{float(metrics.get('iet_pln', 0.0)):.1f}%",
            "Indice de evidencias textuais rastreaveis.",
        ],
        [
            "ICL-PLN",
            f"{float(metrics.get('icl_pln', 0.0)):.1f}%",
            "Confianca lexical media normalizada das premissas inferidas.",
        ],
    ]
    story.append(table(metric_rows, [3.5 * cm, 3.0 * cm, 9.2 * cm]))
    story.append(Spacer(1, 10))

    diagnostics = premise_diagnostics or []
    if diagnostics:
        story.append(paragraph("4. Diagnostico das premissas PLN", "Heading1"))
        diagnostic_rows = [["Premissa", "Valor", "Diagnostico", "Problema", "Sugestao"]]
        for row in diagnostics:
            diagnostic_rows.append(
                [
                    row.get("Premissa", ""),
                    row.get("Valor inferido", ""),
                    row.get("Diagnostico", ""),
                    row.get("Problema predominante", ""),
                    row.get("Sugestao", ""),
                ]
            )
        story.append(table(diagnostic_rows, [2.7 * cm, 2.6 * cm, 3.2 * cm, 2.7 * cm, 4.5 * cm]))
        story.append(Spacer(1, 10))

    story.append(paragraph("5. Metodos recomendados", "Heading1"))
    if recommendations.empty:
        story.append(paragraph("Nenhum metodo recomendado. Revise a descricao do problema."))
    else:
        rows = [["Metodo", "Aderencia", "Criterios", "Justificativa"]]
        for _, row in recommendations.head(10).iterrows():
            rows.append(
                [
                    row["Metodo"],
                    f"{float(row['Aderencia(%)']):.1f}%",
                    f"{row['Criterios atendidos']}/{row['Criterios comparados']}",
                    row.get("Justificativa", ""),
                ]
            )
        story.append(table(rows, [3.5 * cm, 2.0 * cm, 2.5 * cm, 7.7 * cm]))

    doc.build(story)


def pdf_bytes(**kwargs) -> bytes:
    buffer = BytesIO()
    write_pdf_report(buffer, **kwargs)
    return buffer.getvalue()
