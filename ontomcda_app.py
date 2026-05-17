"""Public Streamlit app for OntoMCDA semantic method recommendation."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import pandas as pd
import streamlit as st

from ontomcda.constants import APP_NAME, APP_OWNER_LABEL, ATTR_LABELS, OWL_PATH, PREMISE_ATTRS
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
    total_premises = len(PREMISE_ATTRS)
    inferred_count = sum(1 for attr in PREMISE_ATTRS if premises.get(attr) is not None)
    not_inferred_count = total_premises - inferred_count
    evidence_count = sum(1 for attr in PREMISE_ATTRS if premises.get(attr) is not None and evidence.get(attr))
    ics_pln = 0.0 if total_premises == 0 else 100.0 * inferred_count / total_premises
    tni_pln = 0.0 if total_premises == 0 else 100.0 * not_inferred_count / total_premises
    iet_pln = 0.0 if inferred_count == 0 else 100.0 * evidence_count / inferred_count
    return {
        "total_premises": total_premises,
        "inferred_premises": inferred_count,
        "not_inferred_premises": not_inferred_count,
        "premises_with_evidence": evidence_count,
        "ics_pln": round(ics_pln, 2),
        "tni_pln": round(tni_pln, 2),
        "iet_pln": round(iet_pln, 2),
    }


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

    st.title(APP_NAME)
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
    metric_cols = st.columns(4)
    metric_cols[0].metric("ICS-PLN", f"{float(metrics['ics_pln']):.1f}%")
    metric_cols[1].metric(
        "Premissas inferidas",
        f"{int(metrics['inferred_premises'])}/{int(metrics['total_premises'])}",
    )
    metric_cols[2].metric("TNI-PLN", f"{float(metrics['tni_pln']):.1f}%")
    metric_cols[3].metric("IET-PLN", f"{float(metrics['iet_pln']):.1f}%")
    st.caption(
        "ICS-PLN = indice de cobertura semantica das premissas inferidas. "
        "TNI-PLN = taxa de premissas nao inferidas. "
        "IET-PLN = indice de evidencias textuais rastreaveis."
    )

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
                "Papel na consulta": result.query_profile[attr]["role"],
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

    st.download_button(
        "Baixar relatorio PDF",
        data=pdf_bytes(
            problem_text=st.session_state.get("ontomcda_text", text),
            premises=result.premises,
            nlp_metrics=result.nlp_metrics,
            recommendations=result.accepted,
        ),
        file_name="relatorio_ontomcda_recomendacao.pdf",
        mime="application/pdf",
        type="primary",
    )


if __name__ == "__main__":
    main()
