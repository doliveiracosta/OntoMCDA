"""Rule-based NLP layer for extracting MCDA problem premises from text."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from .constants import ATTRS


LEXICON: dict[str, dict[str, list[str]]] = {
    "tipo_problema": {
        "Escolha": [
            "escolher",
            "selecionar",
            "melhor alternativa",
            "melhor opcao",
            "decisao final",
            "alternativa vencedora",
            "qual alternativa",
            "contratacao",
            "portfolio",
            "carteira",
            "orcamento limitado",
        ],
        "Ordenacao": [
            "ordenar",
            "ranking",
            "ranquear",
            "priorizar",
            "hierarquizar",
            "ordem de preferencia",
            "lista de prioridades",
            "da melhor para a pior",
            "priorizacao",
        ],
        "Classificacao": [
            "classificar",
            "categorizar",
            "classes",
            "categorias",
            "faixas",
            "niveis",
            "classes de risco",
            "alto medio baixo",
            "fornecedores estrategicos",
        ],
    },
    "compensatoriedade": {
        "Compensatorio": ["trade-off", "compensar", "compensacao", "compensatorio", "equilibrar criterios"],
        "NaoCompensatorio": ["veto", "nao compensa", "nao compensatorio", "sem compensacao", "discordancia"],
        "ParcialmenteCompensatorio": ["compensacao parcial", "parcialmente compensatorio", "compensacao limitada"],
    },
    "tipo_variavel": {
        "Mista": ["dados mistos", "variaveis mistas", "quantitativos e qualitativos", "medidas objetivas e subjetivas"],
        "Cardinal": ["cardinal", "quantitativo", "numerico", "custo", "tempo", "prazo", "nota", "percentual"],
        "Ordinal": ["ordinal", "ordem", "alto medio baixo", "escala verbal", "escala linguistica", "posto"],
        "PseudoCardinal": ["comparacoes par a par", "par a par", "escala de saaty", "matriz de comparacao"],
        "CardinalIntervalar": ["intervalar", "intervalo", "faixa de valores", "intervalos de confianca"],
        "CardinalFuzzy": ["fuzzy", "triangular", "trapezoidal", "linguistico fuzzy", "numero fuzzy"],
        "CardinalGrey": ["grey", "cinza", "sistema grey"],
    },
    "monotonicidade": {
        "Monotonico": ["monotonico", "dominancia", "pareto", "mais e melhor", "menos e melhor", "coerente"],
        "NaoMonotonico": ["nao monotonico", "nao linear", "veto", "limiar de preferencia", "limiar de indiferenca"],
    },
    "estrutura_decisoria": {
        "Individual": ["individual", "decisor unico", "um decisor", "um avaliador", "um especialista"],
        "Grupo": [
            "grupo",
            "decisao em grupo",
            "comite",
            "colegiado",
            "equipe",
            "especialistas",
            "multiplos decisores",
            "consenso entre avaliadores",
            "areas",
        ],
    },
    "completude_pref": {
        "Completo": ["preferencias completas", "comparacao completa", "todas as alternativas", "comparabilidade total"],
        "Incompleto": ["preferencias incompletas", "dados incompletos", "incomparabilidade", "comparacao parcial"],
    },
    "ambiente_decisao": {
        "Certeza": ["certeza", "deterministico", "sem incerteza", "dados certos", "informacao completa"],
        "Risco": ["risco", "probabilidade", "probabilistico", "distribuicao de probabilidade", "chance"],
        "Incerteza": ["incerteza", "cenario incerto", "imprecisao", "ambiguidade", "fuzzy", "subjetividade"],
    },
    "ponderabilidade": {
        "Ponderavel": ["ponderavel", "ponderacao", "atribuir pesos", "pesos dos criterios", "importancia relativa"],
        "NaoPonderavel": ["nao ponderavel", "sem ponderacao", "sem pesos", "dispensa pesos"],
        "SemiPonderavel": ["semi ponderavel", "ponderacao parcial", "pesos parciais"],
    },
    "usa_pesos": {
        "Sim": ["usa pesos", "utiliza pesos", "com pesos", "peso dos criterios", "importancia dos criterios"],
        "Nao": ["sem pesos", "nao usa pesos", "nao utiliza pesos", "dispensa pesos"],
    },
    "requer_pesos": {
        "Sim": ["requer pesos", "necessita pesos", "exige pesos", "depende de pesos", "pesos como entrada"],
        "Nao": ["nao requer pesos", "nao necessita pesos", "dispensa pesos"],
    },
    "gera_pesos": {
        "Sim": ["gera pesos", "calcula pesos", "estima pesos", "produz pesos", "deriva pesos"],
        "Nao": ["nao gera pesos", "nao calcula pesos", "nao produz pesos"],
    },
    "auxilia_gerar_pesos": {
        "Sim": ["auxilia gerar pesos", "apoia a geracao de pesos", "comparacoes par a par para pesos"],
        "Nao": ["nao auxilia gerar pesos", "nao ajuda a gerar pesos"],
    },
    "sugere_pesos": {
        "Sim": ["sugere pesos", "recomenda pesos", "propoe pesos", "indica pesos"],
        "Nao": ["nao sugere pesos", "nao recomenda pesos"],
    },
}


def normalize_text(value: str) -> str:
    text = str(value).lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def contains_term(text_norm: str, term: str) -> int:
    pattern = r"\b" + re.escape(normalize_text(term)) + r"\b"
    return len(re.findall(pattern, text_norm))


def canon_value(attr: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = normalize_text(value).replace("-", "").replace("_", "").replace(" ", "")
    maps = {
        "tipo_problema": {"escolha": "Escolha", "ordenacao": "Ordenacao", "classificacao": "Classificacao"},
        "compensatoriedade": {
            "compensatorio": "Compensatorio",
            "naocompensatorio": "NaoCompensatorio",
            "parcialmentecompensatorio": "ParcialmenteCompensatorio",
        },
        "tipo_variavel": {
            "mista": "Mista",
            "cardinal": "Cardinal",
            "ordinal": "Ordinal",
            "pseudocardinal": "PseudoCardinal",
            "cardinalintervalar": "CardinalIntervalar",
            "intervalar": "CardinalIntervalar",
            "cardinalfuzzy": "CardinalFuzzy",
            "fuzzy": "CardinalFuzzy",
            "cardinalgrey": "CardinalGrey",
            "grey": "CardinalGrey",
        },
        "monotonicidade": {"monotonico": "Monotonico", "naomonotonico": "NaoMonotonico"},
        "estrutura_decisoria": {
            "individual": "Individual",
            "grupo": "Grupo",
            "coletiva": "Grupo",
            "multidecisor": "Grupo",
            "monodecisor": "Individual",
            "monomulti": "MonoMulti",
        },
        "completude_pref": {"completo": "Completo", "incompleto": "Incompleto"},
        "ambiente_decisao": {"certeza": "Certeza", "risco": "Risco", "incerteza": "Incerteza"},
        "ponderabilidade": {
            "ponderavel": "Ponderavel",
            "naoponderavel": "NaoPonderavel",
            "semiponderavel": "SemiPonderavel",
        },
    }
    return maps.get(attr, {}).get(cleaned, value)


def infer_attribute(text_norm: str, attr: str) -> tuple[Optional[str], list[str], int]:
    scores: dict[str, int] = {}
    evidence: dict[str, list[str]] = {}
    for value, terms in LEXICON[attr].items():
        count = 0
        found = []
        for term in terms:
            term_count = contains_term(text_norm, term)
            if term_count:
                count += term_count
                found.append(f"{term}({term_count})")
        scores[value] = count
        evidence[value] = found

    best = max(scores, key=scores.get)
    best_score = scores[best]
    if best_score == 0 or list(scores.values()).count(best_score) > 1:
        return None, [], best_score
    return canon_value(attr, best), evidence[best], best_score


def infer_premises(text: str) -> tuple[dict[str, Optional[str]], dict[str, list[str]], dict[str, int]]:
    text_norm = normalize_text(text)
    premises: dict[str, Optional[str]] = {}
    evidence_map: dict[str, list[str]] = {}
    score_map: dict[str, int] = {}

    for attr in ATTRS:
        value, evidence, score = infer_attribute(text_norm, attr)
        premises[attr] = value
        evidence_map[attr] = evidence
        score_map[attr] = score

    # Contextual precedence: classification and ordering are often masked by generic "decision" wording.
    if any(contains_term(text_norm, term) for term in ["classificar", "categorizar", "classes", "categorias"]):
        premises["tipo_problema"] = "Classificacao"
        evidence_map["tipo_problema"] = evidence_map["tipo_problema"] + ["regra_precedencia_classificacao"]
        score_map["tipo_problema"] = max(score_map["tipo_problema"], 2)
    elif any(contains_term(text_norm, term) for term in ["priorizar", "ranking", "ordenar", "ranquear"]):
        premises["tipo_problema"] = "Ordenacao"
        evidence_map["tipo_problema"] = evidence_map["tipo_problema"] + ["regra_precedencia_ordenacao"]
        score_map["tipo_problema"] = max(score_map["tipo_problema"], 2)

    if (
        ("quantitativ" in text_norm and "qualitativ" in text_norm)
        or contains_term(text_norm, "dados mistos")
        or contains_term(text_norm, "variaveis mistas")
    ):
        premises["tipo_variavel"] = "Mista"
        evidence_map["tipo_variavel"] = evidence_map["tipo_variavel"] + ["regra_precedencia_variavel_mista"]
        score_map["tipo_variavel"] = max(score_map["tipo_variavel"], 2)

    if any(
        term in text_norm
        for term in ["incerteza", "incerto", "imprecis", "cenario futuro", "cenarios futuros", "subjetividade"]
    ):
        premises["ambiente_decisao"] = "Incerteza"
        evidence_map["ambiente_decisao"] = evidence_map["ambiente_decisao"] + ["regra_precedencia_incerteza"]
        score_map["ambiente_decisao"] = max(score_map["ambiente_decisao"], 2)

    if any(term in text_norm for term in ["incomplet", "comparacao parcial", "comparacoes parciais"]):
        premises["completude_pref"] = "Incompleto"
        evidence_map["completude_pref"] = evidence_map["completude_pref"] + ["regra_precedencia_incompleto"]
        score_map["completude_pref"] = max(score_map["completude_pref"], 2)
    elif any(term in text_norm for term in ["preferencias completas", "comparacao completa", "comparacoes completas"]):
        premises["completude_pref"] = "Completo"
        evidence_map["completude_pref"] = evidence_map["completude_pref"] + ["regra_precedencia_completo"]
        score_map["completude_pref"] = max(score_map["completude_pref"], 2)

    if any(term in text_norm for term in ["mais e melhor", "menos e melhor", "monotonic", "dominancia"]):
        premises["monotonicidade"] = "Monotonico"
        evidence_map["monotonicidade"] = evidence_map["monotonicidade"] + ["regra_precedencia_monotonico"]
        score_map["monotonicidade"] = max(score_map["monotonicidade"], 2)

    return premises, evidence_map, score_map
