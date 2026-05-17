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
    logo_poli = Path("assets/logo_upe_poli.png")
    logo_ppgec = Path("assets/logo_ppgec.png")
    if logo_poli.exists() and logo_ppgec.exists():
        poli_col, ppgec_col, _ = st.columns([1.1, 1.7, 3.8])
        with poli_col:
            st.image(str(logo_poli), use_container_width=True)
        with ppgec_col:
            st.image(str(logo_ppgec), use_container_width=True)

    st.title(APP_NAME)
    st.markdown(f"**{APP_OWNER_LABEL}**")
    st.caption("Recomendacao explicavel de metodos multicriterio a partir de descricao textual e ontologia OWL.")


def main() -> None:
    st.set_page_config(page_title=APP_NAME, layout="wide")
    render_opening_cover()
    st.info(
        "Versao publica de pesquisa: nao insira dados pessoais, sigilosos ou sensiveis. "
        "A recomendacao e um apoio metodologico e deve ser validada pelo pesquisador ou decisor."
    )

    owl_path = Path(OWL_PATH)
    if not owl_path.exists():
        st.error(f"Arquivo OWL nao encontrado: {owl_path}")
        return

    profiles = cached_profiles(str(owl_path))
    ontology_col, mode_col = st.columns([1, 1])
    ontology_col.metric("Perfis MCDA na ontologia", len(profiles))
    mode_col.metric("Base de conhecimento", owl_path.name)

    text = st.text_area(
        "Descreva o problema decisorio",
        value=example_text(),
        height=190,
        help="Informe objetivo, tipo de decisao, dados disponiveis, incerteza, pesos e quantidade de decisores.",
    )
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
