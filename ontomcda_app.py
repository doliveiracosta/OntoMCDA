"""Public Streamlit app for OntoMCDA semantic method recommendation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ontomcda.constants import APP_NAME, APP_OWNER_LABEL, ATTR_LABELS, OWL_PATH
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
            height: 42px;
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
            """
            <div class="institutional-logos">
                <img class="logo-upe" src="assets/logo_upe.jfif" alt="UPE">
                <img class="logo-poli" src="assets/logo_upe_poli.png" alt="POLI">
                <img class="logo-ppgec" src="assets/logo_ppgec.png" alt="PPGEC">
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
            """
            <style>
            .author-links {
                display: flex;
                align-items: center;
                gap: 18px;
                margin: 0.1rem 0 1.2rem;
            }
            .author-links a {
                display: inline-flex;
                align-items: center;
                gap: 7px;
                color: #6b7280;
                text-decoration: none;
                font-size: 0.92rem;
            }
            .author-links a:hover {
                color: #374151;
                text-decoration: none;
            }
            .author-links img {
                width: 18px;
                height: 18px;
                display: inline-block;
            }
            </style>
            <div class="author-links">
                <a href="https://orcid.org/0000-0002-6138-7451" target="_blank">
                    <img src="assets/logo_orcid.svg" alt="ORCID">
                    <span>Perfil academico</span>
                </a>
                <a href="https://www.linkedin.com/in/daviddeoliveiracosta" target="_blank">
                    <img src="assets/logo_linkedin.svg" alt="LinkedIn">
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

    summary_cols = st.columns(3)
    summary_cols[0].metric("Metodos recomendados", len(result.accepted))
    summary_cols[1].metric("Restricoes fortes violadas", len(result.rejected))
    top_score = 0.0 if result.accepted.empty else float(result.accepted.iloc[0]["Aderencia(%)"])
    summary_cols[2].metric("Maior aderencia", f"{top_score:.1f}%")

    st.subheader("Premissas inferidas")
    premise_rows = []
    for attr, value in result.premises.items():
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
            recommendations=result.accepted,
        ),
        file_name="relatorio_ontomcda_recomendacao.pdf",
        mime="application/pdf",
        type="primary",
    )


if __name__ == "__main__":
    main()
