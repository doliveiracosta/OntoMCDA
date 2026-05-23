"""Public Streamlit app for OntoMCDA semantic method recommendation."""

from __future__ import annotations

import base64
import mimetypes
from html import escape as html_escape
from pathlib import Path

import pandas as pd
import streamlit as st

from ontomcda.constants import APP_NAME, APP_OWNER_LABEL, APP_SUBTITLE, ATTR_LABELS, OWL_PATH, PREMISE_ATTRS
from ontomcda.metrics import build_premise_diagnostics, calculate_operational_nlp_metrics
from ontomcda.ontology import load_profiles
from ontomcda.recommender import recommend_methods
from ontomcda.report import pdf_bytes


@st.cache_data(show_spinner=False)
def cached_profiles(path: str) -> dict:
    return load_profiles(path)


def example_text() -> str:
    return (
        "Desejo priorizar projetos de inovacao em um comite com multiplos especialistas. "
        "Os criterios incluem custo, impacto, prazo e risco, com dados quantitativos e qualitativos. "
        "Ha incerteza nas estimativas e sera necessario considerar pesos de criterios."
    )


def asset_data_uri(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def fallback_nlp_metrics(result: object) -> dict[str, float | int]:
    premises = getattr(result, "premises", {}) or {}
    evidence = getattr(result, "evidence", {}) or {}
    query_profile = getattr(result, "query_profile", {}) or {}
    score_map = {attr: int(query_profile.get(attr, {}).get("score", 0) or 0) for attr in PREMISE_ATTRS}
    return calculate_operational_nlp_metrics(premises, evidence, score_map)


def fallback_premise_diagnostics(result: object) -> list[dict[str, object]]:
    premises = getattr(result, "premises", {}) or {}
    evidence = getattr(result, "evidence", {}) or {}
    query_profile = getattr(result, "query_profile", {}) or {}
    score_map = {attr: int(query_profile.get(attr, {}).get("score", 0) or 0) for attr in PREMISE_ATTRS}
    return build_premise_diagnostics(premises, evidence, score_map)


def lexical_score_bar(value: object) -> str:
    try:
        score = max(0.0, min(3.0, float(value)))
    except (TypeError, ValueError):
        score = 0.0

    colors = {
        0: "#dc2626",
        1: "#f97316",
        2: "#ca8a04",
        3: "#16a34a",
    }
    color = colors[int(round(score))]
    width = max(6, int((score / 3.0) * 100))
    return (
        '<div style="display:grid; grid-template-columns:minmax(95px, 1fr) 26px; '
        'align-items:center; gap:8px; min-width:145px;">'
        '<div style="height:14px; background:linear-gradient(90deg, #fee2e2, #ffedd5, #dcfce7); '
        'border-radius:999px; overflow:hidden; box-shadow:inset 0 0 0 1px rgba(17, 24, 39, 0.16);">'
        f'<div style="height:100%; width:{width}%; background:{color}; border-radius:999px;"></div>'
        "</div>"
        f'<span style="font-weight:700; color:#111827; text-align:right;">{score:.0f}</span>'
        "</div>"
    )


def render_diagnostic_dataframe(diagnostics: list[dict[str, object]]) -> None:
    df = pd.DataFrame(diagnostics)
    df = df.drop(columns=["Diagnostico", "Problema predominante"], errors="ignore")
    if "Escore lexical" not in df.columns:
        st.dataframe(df, use_container_width=True, hide_index=True)
        return

    columns = list(df.columns)
    header = "".join(f"<th>{html_escape(str(column))}</th>" for column in columns)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for column in columns:
            if column == "Escore lexical":
                cells.append(f"<td>{lexical_score_bar(row[column])}</td>")
            else:
                cells.append(f"<td>{html_escape(str(row[column]))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")

    st.markdown(
        f"""
        <style>
        .premise-diagnostic-table-wrap {{
            width: 100%;
            overflow-x: auto;
            border: 1px solid #e5e7eb;
            border-radius: 7px;
        }}
        .premise-diagnostic-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
        }}
        .premise-diagnostic-table th {{
            background: #f3f4f6;
            color: #6b7280;
            font-weight: 500;
            text-align: left;
            padding: 9px 10px;
            border-bottom: 1px solid #e5e7eb;
            white-space: nowrap;
        }}
        .premise-diagnostic-table td {{
            color: #111827;
            padding: 8px 10px;
            border-bottom: 1px solid #e5e7eb;
            border-right: 1px solid #e5e7eb;
            vertical-align: middle;
            white-space: nowrap;
        }}
        .premise-diagnostic-table tr:last-child td {{
            border-bottom: 0;
        }}
        </style>
        <div class="premise-diagnostic-table-wrap">
            <table class="premise-diagnostic-table">
                <thead><tr>{header}</tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_opening_cover() -> None:
    st.markdown(
        """
        <style>
        .institutional-logos {
            display: flex;
            align-items: center;
            gap: 22px;
            margin: 0.2rem 0 1.0rem;
        }
        .institutional-logos img {
            object-fit: contain;
            width: auto;
            display: block;
        }
        .institutional-logos .logo-upe {
            height: 52px;
        }
        .institutional-logos .logo-poli {
            height: 54px;
        }
        .institutional-logos .logo-ppgec {
            height: 48px;
        }
        .usage-guide {
            margin: 0.2rem 0 1.1rem;
            color: #4b5563;
            font-size: 0.94rem;
        }
        .usage-guide summary {
            cursor: pointer;
            color: #6b7280;
            text-decoration: none;
            width: fit-content;
            list-style: none;
        }
        .usage-guide summary:hover {
            color: #374151;
        }
        .usage-guide summary::-webkit-details-marker {
            display: none;
        }
        .usage-guide ol {
            margin: 0.75rem 0 0;
            padding-left: 1.25rem;
            line-height: 1.45;
        }
        .usage-guide li {
            margin-bottom: 0.42rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    logo_upe = Path("assets/logo_upe.jfif")
    logo_poli = Path("assets/logo_upe_poli.png")
    logo_ppgec = Path("assets/logo_ppgec.png")
    if logo_upe.exists() and logo_poli.exists() and logo_ppgec.exists():
        st.markdown(
            f"""
            <div class="institutional-logos">
                <img class="logo-upe" src="{asset_data_uri(logo_upe)}" alt="UPE">
                <img class="logo-poli" src="{asset_data_uri(logo_poli)}" alt="POLI">
                <img class="logo-ppgec" src="{asset_data_uri(logo_ppgec)}" alt="PPGEC">
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <details class="usage-guide">
            <summary>Como utilizar a plataforma</summary>
            <ol>
                <li><strong>Descreva o problema decisorio:</strong> escreva em linguagem natural o contexto, objetivo e alternativas analisadas.</li>
                <li><strong>Informe a problematica:</strong> indique se deseja escolher, ordenar, classificar ou descrever alternativas.</li>
                <li><strong>Explique os criterios:</strong> descreva os criterios relevantes e o tipo de dados disponiveis.</li>
                <li><strong>Indique a estrutura decisoria:</strong> informe se a decisao e individual, em grupo, por comite ou por multiplos avaliadores.</li>
                <li><strong>Declare o ambiente de decisao:</strong> mencione se ha certeza, risco, incerteza, ambiguidade ou dados imprecisos.</li>
                <li><strong>Explique o uso de pesos:</strong> informe se havera pesos, importancia relativa ou comparacao par-a-par.</li>
                <li><strong>Informe a compensatoriedade:</strong> indique se ha compensacao, compensacao parcial ou nao compensacao entre criterios.</li>
                <li><strong>Analise a recomendacao:</strong> clique em Analisar e recomendar para visualizar premissas inferidas, metricas de PLN e metodos recomendados.</li>
                <li><strong>Revise o diagnostico:</strong> use a matriz diagnostica para melhorar a descricao textual quando alguma premissa nao for inferida.</li>
                <li><strong>Exporte o relatorio:</strong> baixe o PDF para registrar as premissas, metricas e recomendacoes.</li>
            </ol>
        </details>
        """,
        unsafe_allow_html=True,
    )

    st.title(APP_NAME)
    st.markdown(f"### {APP_SUBTITLE}")
    st.caption(
        "Recomendacao explicavel de metodos multicriterio a partir de descricao textual, "
        "fundamentada por PLN e ontologia."
    )
    st.markdown(f"**{APP_OWNER_LABEL}**")

    logo_orcid = Path("assets/logo_orcid.svg")
    logo_linkedin = Path("assets/logo_linkedin.svg")
    if logo_orcid.exists() and logo_linkedin.exists():
        st.markdown(
            f"""
            <style>
            .author-links {{
                display: flex;
                align-items: center;
                gap: 18px;
                margin: 0.1rem 0 1.2rem;
            }}
            .author-links a {{
                display: inline-flex;
                align-items: center;
                gap: 7px;
                color: #6b7280;
                text-decoration: none;
                font-size: 0.92rem;
            }}
            .author-links a:hover {{
                color: #374151;
                text-decoration: none;
            }}
            .author-links img {{
                width: 18px;
                height: 18px;
                display: inline-block;
            }}
            </style>
            <div class="author-links">
                <a href="https://orcid.org/0000-0002-6138-7451" target="_blank">
                    <img src="{asset_data_uri(logo_orcid)}" alt="ORCID">
                    <span>Perfil academico</span>
                </a>
                <a href="https://www.linkedin.com/in/daviddeoliveiracosta" target="_blank">
                    <img src="{asset_data_uri(logo_linkedin)}" alt="LinkedIn">
                    <span>Perfil profissional</span>
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title=APP_NAME, layout="wide")
    render_opening_cover()

    owl_path = Path(OWL_PATH)
    if not owl_path.exists():
        st.error(f"Arquivo OWL nao encontrado: {owl_path}")
        return

    profiles = cached_profiles(str(owl_path))
    text = st.text_area(
        "Descreva o problema decisorio",
        value=example_text(),
        height=190,
        help="Informe objetivo, tipo de decisao, dados disponiveis, incerteza, pesos e quantidade de decisores.",
    )
    st.caption(f"Ontologia carregada: {len(profiles)} perfis MCDA | Base de conhecimento: {owl_path.name}")
    top_k = st.slider("Quantidade de metodos recomendados", min_value=5, max_value=30, value=15, step=5)

    if st.button("Analisar e recomendar", type="primary"):
        st.session_state.ontomcda_result = recommend_methods(text, profiles, top_k=top_k)
        st.session_state.ontomcda_text = text

    result = st.session_state.get("ontomcda_result")
    if result is None:
        return

    if result.missing_mandatory:
        missing = ", ".join(ATTR_LABELS.get(attr, attr) for attr in result.missing_mandatory)
        st.warning(f"Nao foi possivel inferir premissas obrigatorias: {missing}. Reforce a descricao textual.")

    st.divider()
    st.subheader("Metricas quantitativas do PLN")
    metrics = getattr(result, "nlp_metrics", fallback_nlp_metrics(result))
    metric_cols = st.columns(6)
    metric_cols[0].metric("ICS-PLN", f"{float(metrics['ics_pln']):.1f}%")
    metric_cols[1].metric(
        "Premissas inferidas",
        f"{int(metrics['inferred_premises'])}/{int(metrics['total_premises'])}",
    )
    metric_cols[2].metric("TNI-PLN", f"{float(metrics['tni_pln']):.1f}%")
    metric_cols[3].metric("IET-PLN", f"{float(metrics['iet_pln']):.1f}%")
    metric_cols[4].metric("ICL-PLN", f"{float(metrics.get('icl_pln', 0.0)):.1f}%")
    metric_cols[5].metric("IJL-PLN", f"{float(metrics.get('ijl_pln', 0.0)):.1f}%")
    st.caption(
        "ICS-PLN = indice de cobertura semantica das premissas inferidas. "
        "TNI-PLN = taxa de premissas nao inferidas. "
        "IET-PLN = indice de evidencias textuais rastreaveis. "
        "ICL-PLN = confianca lexical media normalizada das premissas inferidas. "
        "IJL-PLN = similaridade Jaccard lexical media entre texto e vocabulario controlado."
    )
    diagnostics = getattr(result, "premise_diagnostics", fallback_premise_diagnostics(result))
    with st.expander("Diagnostico por premissa para melhoria do PLN", expanded=True):
        render_diagnostic_dataframe(diagnostics)

    st.divider()
    summary_cols = st.columns(3)
    summary_cols[0].metric("Metodos recomendados", len(result.accepted))
    summary_cols[1].metric("Restricoes fortes violadas", len(result.rejected))
    top_score = 0.0 if result.accepted.empty else float(result.accepted.iloc[0]["Aderencia(%)"])
    summary_cols[2].metric("Maior aderencia", f"{top_score:.1f}%")

    st.subheader("Premissas inferidas")
    premise_rows = []
    for attr in PREMISE_ATTRS:
        value = result.premises.get(attr)
        evidence = "; ".join(result.evidence.get(attr, []))
        premise_rows.append(
            {
                "Premissa": ATTR_LABELS.get(attr, attr),
                "Valor inferido": value or "Nao inferido",
                "Evidencias": evidence,
            }
        )
    st.dataframe(pd.DataFrame(premise_rows), use_container_width=True, hide_index=True)

    st.subheader("Metodos recomendados")
    if result.accepted.empty:
        st.info("Nenhum metodo recomendado para as premissas atuais.")
    else:
        st.dataframe(result.accepted, use_container_width=True, hide_index=True)

    with st.expander("Metodos rejeitados por restricoes fortes"):
        if result.rejected.empty:
            st.write("Nenhum metodo rejeitado por restricao forte.")
        else:
            st.dataframe(result.rejected, use_container_width=True, hide_index=True)

    pdf_kwargs = {
        "problem_text": st.session_state.get("ontomcda_text", text),
        "premises": result.premises,
        "nlp_metrics": metrics,
        "premise_diagnostics": diagnostics,
        "recommendations": result.accepted,
    }
    try:
        report_data = pdf_bytes(**pdf_kwargs)
    except TypeError:
        # Compatibility with older deployed report.py versions that do not accept nlp_metrics yet.
        pdf_kwargs.pop("nlp_metrics", None)
        pdf_kwargs.pop("premise_diagnostics", None)
        report_data = pdf_bytes(**pdf_kwargs)

    st.download_button(
        "Baixar relatorio PDF",
        data=report_data,
        file_name="relatorio_ontomcda_recomendacao.pdf",
        mime="application/pdf",
        type="primary",
    )


if __name__ == "__main__":
    main()
