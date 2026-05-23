"""Quantitative NLP metrics for OntoMCDA."""

from __future__ import annotations

from typing import Optional

from .constants import ATTR_LABELS, PREMISE_ATTRS

LEXICAL_SCORE_CAP = 3

HARD_PREMISES = {"compensatoriedade", "monotonicidade", "ponderabilidade", "completude_pref"}

DIAGNOSTIC_GUIDANCE = {
    "tipo_problema": "Declare se o objetivo e escolher, ordenar, classificar ou descrever alternativas.",
    "compensatoriedade": "Declare se ha compensacao, compensacao parcial, veto ou nao compensacao entre criterios.",
    "tipo_variavel": "Informe se os dados sao quantitativos, qualitativos, mistos, fuzzy, grey ou intervalares.",
    "monotonicidade": "Use pistas como quanto maior melhor, quanto menor melhor, nao monotonico, limiar ou veto.",
    "estrutura_decisoria": "Informe se a decisao e individual, coletiva, em grupo, por comite ou por multiplos avaliadores.",
    "completude_pref": "Declare se as preferencias/comparacoes sao completas ou se ha informacao incompleta.",
    "ambiente_decisao": "Informe se o ambiente e de certeza, risco, incerteza, ambiguidade ou dados imprecisos.",
    "ponderabilidade": "Explique se ha pesos, importancia relativa, comparacao par-a-par ou ausencia de ponderacao.",
    "usa_pesos": "Declare se o metodo deve usar pesos, importancia relativa ou criterios ponderados.",
    "requer_pesos": "Declare se os pesos precisam ser definidos, estimados ou derivados pelo decisor.",
}


def calculate_operational_nlp_metrics(
    premises: dict[str, Optional[str]],
    evidence: dict[str, list[str]],
    score_map: dict[str, int] | None = None,
) -> dict[str, float | int]:
    """Calculate runtime NLP metrics for a single user text.

    ICS-PLN measures semantic coverage, TNI-PLN measures non-inference,
    IET-PLN measures traceable textual evidence, and ICL-PLN measures the
    average normalized lexical strength of inferred premises.
    """
    score_map = score_map or {}
    total_premises = len(PREMISE_ATTRS)
    inferred_attrs = [attr for attr in PREMISE_ATTRS if premises.get(attr) is not None]
    inferred_count = len(inferred_attrs)
    not_inferred_count = total_premises - inferred_count
    evidence_count = sum(1 for attr in inferred_attrs if evidence.get(attr))

    semantic_coverage = 0.0 if total_premises == 0 else 100.0 * inferred_count / total_premises
    not_inferred_rate = 0.0 if total_premises == 0 else 100.0 * not_inferred_count / total_premises
    textual_evidence = 0.0 if inferred_count == 0 else 100.0 * evidence_count / inferred_count

    lexical_scores = [
        min(max(int(score_map.get(attr, 0)), 0), LEXICAL_SCORE_CAP) / LEXICAL_SCORE_CAP
        for attr in inferred_attrs
    ]
    lexical_confidence = 0.0 if not lexical_scores else 100.0 * sum(lexical_scores) / len(lexical_scores)

    return {
        "total_premises": total_premises,
        "inferred_premises": inferred_count,
        "not_inferred_premises": not_inferred_count,
        "premises_with_evidence": evidence_count,
        "ics_pln": round(semantic_coverage, 2),
        "tni_pln": round(not_inferred_rate, 2),
        "iet_pln": round(textual_evidence, 2),
        "icl_pln": round(lexical_confidence, 2),
    }


def build_premise_diagnostics(
    premises: dict[str, Optional[str]],
    evidence: dict[str, list[str]],
    score_map: dict[str, int] | None = None,
) -> list[dict[str, object]]:
    """Build an explainable diagnostic matrix for premise-level NLP quality."""
    score_map = score_map or {}
    rows: list[dict[str, object]] = []

    for attr in PREMISE_ATTRS:
        value = premises.get(attr)
        attr_evidence = evidence.get(attr, [])
        score = int(score_map.get(attr, 0) or 0)

        if value is None:
            status = "Nao inferida"
            problem = "Texto e lexico" if attr in HARD_PREMISES else "Texto insuficiente"
        elif score <= 1:
            status = "Inferida com baixa confianca lexical"
            problem = "Lexico/regras"
        elif not attr_evidence:
            status = "Inferida por regra contextual"
            problem = "Texto"
        else:
            status = "Inferencia rastreavel"
            problem = "Adequado"

        rows.append(
            {
                "Premissa": ATTR_LABELS.get(attr, attr),
                "Valor inferido": value or "Nao inferido",
                "Evidencias": "; ".join(attr_evidence),
                "Diagnostico": status,
                "Problema predominante": problem,
                "Sugestao": DIAGNOSTIC_GUIDANCE.get(attr, "Reforce a descricao textual desta premissa."),
            }
        )

    return rows
