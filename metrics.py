"""Quantitative NLP metrics for OntoMCDA."""

from __future__ import annotations

from typing import Optional

from .constants import PREMISE_ATTRS

LEXICAL_SCORE_CAP = 3


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
