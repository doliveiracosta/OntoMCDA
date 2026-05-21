"""Rule-based NLP layer for extracting MCDA problem premises from text."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from .constants import ATTRS


LEXICON: dict[str, dict[str, list[str]]] = {'tipo_problema': {'Escolha': ['escolher',
                               'selecionar',
                               'melhor alternativa',
                               'melhor opcao',
                               'decisao final',
                               'alternativa vencedora',
                               'qual alternativa',
                               'contratacao',
                               'portfolio',
                               'carteira',
                               'orcamento limitado',
                               'escolha',
                               'selecao',
                               'decidir',
                               'eleger',
                               'optar',
                               'adotar',
                               'definir a melhor alternativa',
                               'selecionar a melhor alternativa',
                               'escolher a melhor alternativa',
                               'alternativa mais adequada',
                               'alternativa mais apropriada',
                               'alternativa preferida',
                               'escolha da melhor',
                               'escolha final',
                               'qual alternativa escolher',
                               'qual alternativa selecionar',
                               'qual projeto escolher',
                               'qual projeto selecionar',
                               'selecao da melhor alternativa',
                               'selecao final',
                               'tomar a melhor decisao',
                               'tomada de decisao final',
                               'definir qual alternativa',
                               'definir qual projeto'],
                   'Ordenacao': ['ordenar',
                                 'ranking',
                                 'ranquear',
                                 'priorizar',
                                 'hierarquizar',
                                 'ordem de preferencia',
                                 'lista de prioridades',
                                 'da melhor para a pior',
                                 'priorizacao',
                                 'ordenacao',
                                 'ranqueamento',
                                 'hierarquia',
                                 'prioridade',
                                 'prioridades',
                                 'ordenar alternativas',
                                 'ordenar projetos',
                                 'ordenar os projetos',
                                 'ordenar as alternativas',
                                 'estabelecer ranking',
                                 'gerar ranking',
                                 'produzir ranking',
                                 'lista de prioridade',
                                 'classificar por prioridade',
                                 'hierarquia de alternativas',
                                 'hierarquia de projetos',
                                 'da mais preferida para a menos preferida',
                                 'ordem decrescente de preferencia',
                                 'definir prioridade',
                                 'definir prioridades',
                                 'estabelecer uma ordem',
                                 'estabelecer ordem de prioridade'],
                   'Classificacao': ['classificar',
                                     'categorizar',
                                     'classes',
                                     'categorias',
                                     'faixas',
                                     'niveis',
                                     'classes de risco',
                                     'alto medio baixo',
                                     'fornecedores estrategicos',
                                     'classificacao',
                                     'sorting',
                                     'alocar em categorias',
                                     'atribuir a classes',
                                     'enquadrar em classes',
                                     'distribuir em categorias',
                                     'separar por categorias',
                                     'niveis de desempenho',
                                     'categorias de risco',
                                     'categorias de desempenho',
                                     'categorias previamente definidas',
                                     'classes previamente definidas',
                                     'alocar em classes',
                                     'classificar em categorias',
                                     'classificar em classes',
                                     'atribuir categoria',
                                     'atribuir classe',
                                     'enquadrar em categorias',
                                     'enquadrar em faixas',
                                     'alocar alternativas em classes',
                                     'alocar alternativas em categorias',
                                     'monitorados',
                                     'criticos',
                                     'alta media baixa']},
 'compensatoriedade': {'Compensatorio': ['trade-off',
                                         'compensar',
                                         'compensacao',
                                         'compensatorio',
                                         'equilibrar criterios',
                                         'compensacoes',
                                         'aceita compensacao',
                                         'aceita compensacoes',
                                         'permite compensacao',
                                         'permite compensacoes',
                                         'desempenho em um criterio pode compensar outro',
                                         'um criterio compensa outro',
                                         'compensacao entre criterios'],
                       'NaoCompensatorio': ['veto',
                                            'nao compensa',
                                            'nao compensatorio',
                                            'sem compensacao',
                                            'discordancia',
                                            'limiar de veto',
                                            'nao admite compensacao',
                                            'nao admite compensacoes',
                                            'nao pode ser compensado',
                                            'nao deve ser compensado',
                                            'sem compensacoes',
                                            'proibe compensacao',
                                            'criterio de veto',
                                            'efeito veto'],
                       'ParcialmenteCompensatorio': ['compensacao parcial',
                                                     'parcialmente compensatorio',
                                                     'compensacao limitada',
                                                     'compensa parcialmente',
                                                     'nao deve ser totalmente compensado',
                                                     'nao pode ser totalmente compensado',
                                                     'compensado por bom desempenho em outros criterios',
                                                     'compensacao apenas parcial',
                                                     'compensacao parcial entre criterios',
                                                     'admite compensacao parcial']},
 'tipo_variavel': {'Mista': ['dados mistos',
                             'variaveis mistas',
                             'quantitativos e qualitativos',
                             'medidas objetivas e subjetivas',
                             'mista',
                             'misto',
                             'qualitativos e quantitativos',
                             'criterios quantitativos e qualitativos',
                             'julgamentos especialistas e medidas objetivas',
                             'informacoes quantitativas e qualitativas',
                             'indicadores quantitativos e qualitativos',
                             'criterios numericos e qualitativos'],
                   'Cardinal': ['cardinal',
                                'quantitativo',
                                'numerico',
                                'custo',
                                'tempo',
                                'prazo',
                                'nota',
                                'percentual',
                                'quantitativa',
                                'numerica',
                                'valores numericos',
                                'medida numerica',
                                'dados numericos',
                                'escala numerica',
                                'pontuacao',
                                'pontualidade',
                                'capacidade de resposta',
                                'estabilidade financeira',
                                'frequencia',
                                'idade dos equipamentos',
                                'tempo medio',
                                'probabilidade',
                                'porcentagem'],
                   'Ordinal': ['ordinal',
                               'ordem',
                               'alto medio baixo',
                               'escala verbal',
                               'escala linguistica',
                               'posto',
                               'preferencia ordinal',
                               'escala ordinal',
                               'ranking ordinal',
                               'muito alto',
                               'alto',
                               'medio',
                               'baixo',
                               'muito baixo',
                               'julgamentos ordinais',
                               'avaliacoes ordinais',
                               'preferencias qualitativas ordenadas',
                               'intensidade de preferencia'],
                   'PseudoCardinal': ['comparacoes par a par',
                                      'par a par',
                                      'escala de saaty',
                                      'matriz de comparacao',
                                      'pseudo-cardinal',
                                      'pseudocardinal',
                                      'pseudo cardinal',
                                      'comparacao entre criterios',
                                      'comparacao entre alternativas'],
                   'CardinalIntervalar': ['intervalar',
                                          'intervalo',
                                          'faixa de valores',
                                          'intervalos de confianca',
                                          'escala intervalar',
                                          'intervalos',
                                          'dados intervalares',
                                          'intervalo de valores'],
                   'CardinalFuzzy': ['fuzzy',
                                     'triangular',
                                     'trapezoidal',
                                     'linguistico fuzzy',
                                     'numero fuzzy',
                                     'linguistica fuzzy',
                                     'numeros fuzzy',
                                     'variavel fuzzy'],
                   'CardinalGrey': ['grey', 'cinza', 'sistema grey', 'grey system']},
 'monotonicidade': {'Monotonico': ['monotonico',
                                   'dominancia',
                                   'pareto',
                                   'mais e melhor',
                                   'menos e melhor',
                                   'coerente',
                                   'dominancia pareto',
                                   'consistente',
                                   'relacao monotonica'],
                    'NaoMonotonico': ['nao monotonico',
                                      'nao linear',
                                      'veto',
                                      'limiar de preferencia',
                                      'limiar de indiferenca',
                                      'efeito nao monotonico']},
 'estrutura_decisoria': {'Individual': ['individual',
                                        'decisor unico',
                                        'um decisor',
                                        'um avaliador',
                                        'um especialista',
                                        'avaliador unico',
                                        'tomada de decisao individual',
                                        'decisao individual'],
                         'Grupo': ['grupo',
                                   'decisao em grupo',
                                   'comite',
                                   'colegiado',
                                   'equipe',
                                   'especialistas',
                                   'multiplos decisores',
                                   'consenso entre avaliadores',
                                   'areas',
                                   'tomada de decisao em grupo',
                                   'painel de especialistas',
                                   'varios decisores',
                                   'coletiva',
                                   'decisao coletiva',
                                   'grupo de avaliadores',
                                   'grupo de decisores',
                                   'comissao',
                                   'banca',
                                   'painel avaliador',
                                   'gestores e especialistas',
                                   'multiplos avaliadores',
                                   'consenso entre decisores',
                                   'areas de suprimentos']},
 'completude_pref': {'Completo': ['preferencias completas',
                                  'comparacao completa',
                                  'todas as alternativas',
                                  'comparabilidade total',
                                  'preferencia completa',
                                  'todas as alternativas sao comparaveis',
                                  'ordem completa',
                                  'todas podem ser comparadas'],
                     'Incompleto': ['preferencias incompletas',
                                    'dados incompletos',
                                    'incomparabilidade',
                                    'comparacao parcial',
                                    'preferencia incompleta',
                                    'informacao incompleta',
                                    'nem todas as alternativas podem ser comparadas',
                                    'incomparaveis']},
 'ambiente_decisao': {'Certeza': ['certeza',
                                  'deterministico',
                                  'sem incerteza',
                                  'dados certos',
                                  'informacao completa',
                                  'ambiente de certeza',
                                  'sem risco',
                                  'cenario deterministico'],
                      'Risco': ['risco',
                                'probabilidade',
                                'probabilistico',
                                'distribuicao de probabilidade',
                                'chance',
                                'probabilidades',
                                'cenario probabilistico',
                                'chance de ocorrencia',
                                'cenario de risco'],
                      'Incerteza': ['incerteza',
                                    'cenario incerto',
                                    'imprecisao',
                                    'ambiguidade',
                                    'fuzzy',
                                    'subjetividade',
                                    'ambiente de incerteza',
                                    'informacao incompleta',
                                    'dados incompletos',
                                    'grey',
                                    'julgamentos especialistas',
                                    'projecoes',
                                    'eventos futuros',
                                    'avaliacoes subjetivas',
                                    'consenso entre decisores',
                                    'apreciacoes subjetivas',
                                    'informacao parcial']},
 'ponderabilidade': {'Ponderavel': ['ponderavel',
                                    'ponderacao',
                                    'atribuir pesos',
                                    'pesos dos criterios',
                                    'importancia relativa',
                                    'atribuicao de pesos',
                                    'criterios ponderaveis',
                                    'importancia relativa diferente',
                                    'niveis de importancia',
                                    'peso relativo',
                                    'pesos relativos',
                                    'criterios com pesos'],
                     'NaoPonderavel': ['nao ponderavel',
                                       'sem ponderacao',
                                       'sem pesos',
                                       'dispensa pesos',
                                       'nao utiliza pesos'],
                     'SemiPonderavel': ['semi ponderavel', 'ponderacao parcial', 'pesos parciais']},
 'usa_pesos': {'Sim': ['usa pesos',
                       'utiliza pesos',
                       'com pesos',
                       'peso dos criterios',
                       'importancia dos criterios',
                       'ponderacao dos criterios',
                       'relevancia relativa',
                       'pesos definidos',
                       'pesos atribuidos'],
               'Nao': ['sem pesos', 'nao usa pesos', 'nao utiliza pesos', 'dispensa pesos']},
 'requer_pesos': {'Sim': ['requer pesos',
                          'necessita pesos',
                          'exige pesos',
                          'depende de pesos',
                          'pesos como entrada',
                          'pesos de entrada',
                          'criterios com importancia relativa diferente',
                          'importancia relativa diferente',
                          'peso como parametro'],
                  'Nao': ['nao requer pesos', 'nao necessita pesos', 'dispensa pesos']},
 'gera_pesos': {'Sim': ['gera pesos',
                        'calcula pesos',
                        'estima pesos',
                        'produz pesos',
                        'deriva pesos',
                        'obtem pesos',
                        'determina pesos'],
                'Nao': ['nao gera pesos', 'nao calcula pesos', 'nao produz pesos']},
 'auxilia_gerar_pesos': {'Sim': ['auxilia gerar pesos',
                                 'apoia a geracao de pesos',
                                 'comparacoes par a par para pesos',
                                 'auxilia a gerar pesos',
                                 'ajuda a gerar pesos',
                                 'apoia a gerar pesos',
                                 'auxilia a definicao de pesos',
                                 'apoia a definicao de pesos',
                                 'comparacoes entre criterios',
                                 'importancia estabelecida pelos decisores',
                                 'pesos definidos pelos decisores'],
                         'Nao': ['nao auxilia gerar pesos', 'nao ajuda a gerar pesos', 'nao auxilia a gerar pesos']},
 'sugere_pesos': {'Sim': ['sugere pesos', 'recomenda pesos', 'propoe pesos', 'indica pesos'],
                  'Nao': ['nao sugere pesos', 'nao recomenda pesos']}}

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
