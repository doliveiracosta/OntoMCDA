"""Recommendation engine that matches inferred premises against ontology profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .constants import ATTRS, ATTR_LABELS, ATTR_WEIGHTS, MANDATORY_QUERY_ATTRS, PREMISE_ATTRS
from .metrics import build_premise_diagnostics, calculate_operational_nlp_metrics
from .text_inference import canon_value, infer_premises, jaccard_scores_by_premise


@dataclass(slots=True)
class RecommendationResult:
    premises: dict[str, Optional[str]]
    evidence: dict[str, list[str]]
    query_profile: dict[str, dict[str, object]]
    nlp_metrics: dict[str, float | int]
    premise_diagnostics: list[dict[str, object]]
    accepted: pd.DataFrame
    rejected: pd.DataFrame
    missing_mandatory: list[str]


def tipo_variavel_match(inferred: str, method_value: Optional[str]) -> bool:
    if method_value is None:
        return False
    method = canon_value("tipo_variavel", method_value)
    if inferred == "Mista":
        return method in {"Mista", "Cardinal", "Ordinal", "PseudoCardinal", "CardinalIntervalar", "CardinalFuzzy", "CardinalGrey"}
    if inferred == "Cardinal":
        return method in {"Cardinal", "Mista", "PseudoCardinal", "CardinalIntervalar", "CardinalFuzzy", "CardinalGrey"}
    if inferred == "Ordinal":
        return method in {"Ordinal", "Mista"}
    if inferred == "CardinalFuzzy":
        return method in {"CardinalFuzzy", "Cardinal", "Mista"}
    return inferred == method


def ontology_match(attr: str, problem_value: Optional[str], method_value: Optional[str]) -> tuple[Optional[bool], str]:
    if problem_value is None:
        return None, "premissa nao inferida"
    if method_value is None:
        return None, "atributo ausente na ontologia"

    problem = canon_value(attr, problem_value)
    method = canon_value(attr, method_value)

    if attr == "tipo_variavel":
        return tipo_variavel_match(str(problem), method), str(method)
    if attr == "estrutura_decisoria":
        if method in {"MonoMulti", "Mono_Multi", "MonoMultiDecisor", "Mono/Multi"}:
            return problem in {"Individual", "Grupo"}, str(method)
        return problem == method, str(method)
    if attr == "compensatoriedade" and problem == "ParcialmenteCompensatorio":
        return method in {"ParcialmenteCompensatorio", "Compensatorio", "NaoCompensatorio"}, str(method)

    return problem == method, str(method)


def readable_match(attr: str, match: Optional[bool], method_value: str) -> str:
    label = ATTR_LABELS[attr]
    if match is True:
        return f"{label}: {method_value}"
    if match is False:
        return f"{label}: {method_value} (divergente)"
    return f"{label}: atributo ausente na ontologia"


def build_query_profile(premises: dict[str, Optional[str]], score_map: dict[str, int]) -> dict[str, dict[str, object]]:
    query = {}
    for attr in ATTRS:
        value = premises.get(attr)
        score = score_map.get(attr, 0)
        if value is None:
            role = "ignore"
        elif attr in MANDATORY_QUERY_ATTRS:
            role = "hard"
        elif score >= 1:
            role = "soft"
        else:
            role = "ignore"
        query[attr] = {"value": value, "score": score, "role": role}
    return query


def consult_ontology(
    query_profile: dict[str, dict[str, object]],
    profiles: dict[str, dict[str, Optional[str]]],
    top_k: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted = []
    rejected = []
    compared_attrs = PREMISE_ATTRS

    for method_name, method_profile in profiles.items():
        hard_fail = []
        compared = 0
        attended = 0
        total_weight = 0.0
        attended_weight = 0.0
        marks = {}
        explanations = []

        for attr in PREMISE_ATTRS:
            query = query_profile[attr]
            query_value = query["value"]
            role = query["role"]
            compared += 1
            total_weight += ATTR_WEIGHTS.get(attr, 1.0)

            if query_value is None:
                marks[attr] = "-"
                continue

            match, explanation = ontology_match(attr, str(query_value), method_profile.get(attr))
            if role == "hard" and match is False:
                hard_fail.append(ATTR_LABELS[attr])

            if match is True:
                attended += 1
                attended_weight += ATTR_WEIGHTS.get(attr, 1.0)
                marks[attr] = "OK"
            elif match is False:
                marks[attr] = "Nao"
            else:
                marks[attr] = "-"
            explanations.append(readable_match(attr, match, explanation))

        if hard_fail:
            rejected.append({"Metodo": method_name, "Restricoes fortes violadas": ", ".join(hard_fail)})
            continue

        adherence = 0.0 if total_weight == 0 else 100.0 * attended_weight / total_weight
        accepted.append(
            {
                "Metodo": method_name,
                "Aderencia(%)": round(adherence, 2),
                "Criterios atendidos": attended,
                "Criterios comparados": compared,
                "Justificativa": "; ".join(explanations[:8]),
                **{ATTR_LABELS.get(attr, attr): marks[attr] for attr in compared_attrs},
            }
        )

    accepted_df = pd.DataFrame(accepted)
    rejected_df = pd.DataFrame(rejected)
    if not accepted_df.empty:
        accepted_df = accepted_df.sort_values(
            ["Aderencia(%)", "Criterios atendidos", "Metodo"],
            ascending=[False, False, True],
        ).head(top_k).reset_index(drop=True)

    return accepted_df, rejected_df


def recommend_methods(text: str, profiles: dict[str, dict[str, Optional[str]]], top_k: int = 15) -> RecommendationResult:
    premises, evidence, score_map = infer_premises(text)
    jaccard_map = jaccard_scores_by_premise(text)
    query_profile = build_query_profile(premises, score_map)
    nlp_metrics = calculate_operational_nlp_metrics(premises, evidence, score_map, jaccard_map)
    premise_diagnostics = build_premise_diagnostics(premises, evidence, score_map)
    missing = [attr for attr in MANDATORY_QUERY_ATTRS if premises.get(attr) is None]
    if missing:
        return RecommendationResult(
            premises,
            evidence,
            query_profile,
            nlp_metrics,
            premise_diagnostics,
            pd.DataFrame(),
            pd.DataFrame(),
            missing,
        )

    accepted, rejected = consult_ontology(query_profile, profiles, top_k=top_k)
    return RecommendationResult(premises, evidence, query_profile, nlp_metrics, premise_diagnostics, accepted, rejected, missing)
