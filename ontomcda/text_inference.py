"""Rule-based NLP layer for extracting MCDA problem premises from text."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from .constants import ATTRS, PREMISE_ATTRS


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

LEXICON_EXTENSIONS: dict[str, dict[str, list[str]]] = {
    "tipo_problema": {
        "Escolha": [
            "selecionar fornecedores",
            "selecionar projetos",
            "definir a alternativa vencedora",
            "tomar uma decisao de selecao",
            "escolher entre alternativas",
            "selecionar a opcao mais adequada",
        ],
        "Ordenacao": [
            "priorizacao de alternativas",
            "priorizacao de projetos",
            "ordenacao de alternativas",
            "ranqueamento de fornecedores",
            "hierarquizacao de alternativas",
            "estabelecer prioridades",
            "ordenar por desempenho",
        ],
        "Classificacao": [
            "classificar fornecedores",
            "classificar alternativas",
            "enquadrar alternativas",
            "atribuir categorias",
            "categorizar por desempenho",
            "alocar em classes",
            "separar em grupos de risco",
        ],
    },
    "compensatoriedade": {
        "Compensatorio": [
            "trade off entre criterios",
            "troca entre criterios",
            "substituicao de desempenho",
            "bom desempenho compensa",
            "ganhos compensam perdas",
            "permite trade off",
            "admite compensacao plena",
        ],
        "ParcialmenteCompensatorio": [
            "compensacao com veto",
            "compensacao limitada",
            "compensacao parcial",
            "compensa parcialmente",
            "trade off limitado",
            "admite compensacao com restricoes",
            "ha limiar de veto",
            "veto parcial",
            "veto relativo",
            "limiar de veto parcial",
            "restricao com tolerancia",
            "requisito minimo com tolerancia",
        ],
        "NaoCompensatorio": [
            "nao permite compensacao",
            "nao admite trade off",
            "sem trade off",
            "criterio eliminatorio",
            "requisito minimo obrigatorio",
            "veto absoluto",
            "um criterio nao anula o outro",
            "restricao obrigatoria",
            "condicao eliminatoria",
            "nao admite substituicao entre criterios",
            "desempenho ruim nao pode ser compensado",
        ],
    },
    "tipo_variavel": {
        "Ordinal": [
            "escala ordinal",
            "baixo medio alto",
            "categorias ordenadas",
            "niveis de preferencia",
            "dados qualitativos ordenados",
        ],
        "Cardinal": [
            "dados numericos",
            "valores numericos",
            "medidas quantitativas",
            "indicadores mensuraveis",
            "custos em reais",
            "prazo em dias",
        ],
        "Mista": [
            "criterios quantitativos e qualitativos",
            "dados quantitativos e qualitativos",
            "indicadores objetivos e subjetivos",
            "variaveis quantitativas e qualitativas",
            "criterios numericos e julgamentos qualitativos",
        ],
        "CardinalFuzzy": [
            "dados fuzzy",
            "numero fuzzy",
            "escala fuzzy",
            "avaliacao linguistica fuzzy",
            "incerteza fuzzy",
        ],
        "CardinalIntervalar": [
            "dados intervalares",
            "valores em intervalo",
            "intervalos de variacao",
            "estimativas intervalares",
        ],
        "CardinalGrey": [
            "dados grey",
            "sistema grey",
            "informacao cinzenta",
            "grey numbers",
        ],
    },
    "monotonicidade": {
        "Monotonico": [
            "criterio de beneficio",
            "criterio de custo",
            "criterio monotonico de beneficio",
            "criterio monotonico de custo",
            "preferencia crescente",
            "preferencia decrescente",
            "funcao de preferencia crescente",
            "funcao de preferencia decrescente",
            "funcao de beneficio crescente",
            "funcao de custo decrescente",
            "maior valor e melhor",
            "menor valor e melhor",
            "maior desempenho e melhor",
            "menor custo e melhor",
            "quanto maior o desempenho melhor",
            "quanto menor o custo melhor",
            "maximizar desempenho",
            "minimizar custo",
            "maximizar beneficio",
            "minimizar perda",
            "relacao monotonicamente crescente",
            "relacao monotonicamente decrescente",
        ],
        "NaoMonotonico": [
            "faixa ideal",
            "zona ideal",
            "zona otima",
            "faixa otima",
            "intervalo ideal",
            "ideal entre",
            "valor otimo intermediario",
            "ponto otimo",
            "ponto de melhor desempenho",
            "ponto de saturacao",
            "preferencia em formato de u",
            "preferencia nao monotona",
            "nao e crescente",
            "nao e decrescente",
            "melhor em valores intermediarios",
            "piora depois de certo ponto",
            "aumento nao implica melhora",
            "reducao nao implica melhora",
        ],
    },
    "estrutura_decisoria": {
        "Grupo": [
            "decisao coletiva",
            "decisao em grupo",
            "comite decisor",
            "painel de especialistas",
            "multiplos decisores",
            "varios avaliadores",
            "equipe multidisciplinar",
            "decisao participativa",
            "colegiado",
        ],
        "Individual": [
            "decisor unico",
            "decisao individual",
            "um unico decisor",
            "responsavel unico",
            "avaliador unico",
        ],
    },
    "completude_pref": {
        "Completo": [
            "todas as comparacoes foram realizadas",
            "todas as preferencias conhecidas",
            "informacao totalmente disponivel",
            "avaliacao completa das alternativas",
            "sem lacunas de informacao",
            "preferencias sao completas",
            "comparacoes sao completas",
            "preferencias totalmente especificadas",
            "todos os julgamentos foram informados",
            "matriz de preferencias completa",
        ],
        "Incompleto": [
            "comparacoes faltantes",
            "preferencias parciais",
            "informacao faltante",
            "avaliacao incompleta",
            "dados nao informados",
            "nem todos os criterios foram avaliados",
            "preferencias nao sao completas",
            "comparacoes nao sao completas",
            "ha julgamentos faltantes",
            "matriz de preferencias incompleta",
        ],
    },
    "ambiente_decisao": {
        "Certeza": [
            "dados deterministico",
            "dados deterministas",
            "sem incerteza",
            "informacoes conhecidas",
            "cenario estavel",
        ],
        "Risco": [
            "probabilidades conhecidas",
            "probabilidade estimada",
            "cenarios probabilisticos",
            "risco mensuravel",
            "distribuicao de probabilidade",
        ],
        "Incerteza": [
            "probabilidades desconhecidas",
            "dados imprecisos",
            "estimativas incertas",
            "variabilidade desconhecida",
            "ambiente incerto",
            "cenarios incertos",
            "informacao imprecisa",
        ],
    },
    "ponderabilidade": {
        "Ponderavel": [
            "criterios com pesos",
            "atribuicao de importancia",
            "importancia dos criterios",
            "preferencias ponderadas",
            "vetor de pesos",
            "pesos elicitados",
        ],
        "SemiPonderavel": [
            "pesos aproximados",
            "pesos qualitativos",
            "pesos parcialmente definidos",
            "importancia qualitativa",
        ],
        "NaoPonderavel": [
            "sem ponderacao",
            "nao requer ponderacao",
            "criterios com mesma importancia",
            "todos os criterios equivalentes",
        ],
    },
    "usa_pesos": {
        "Sim": [
            "utiliza pesos",
            "considera pesos",
            "incorpora pesos",
            "criterios ponderados",
            "vetor de pesos",
        ],
        "Nao": [
            "nao utiliza pesos",
            "nao considera pesos",
            "sem vetor de pesos",
            "sem criterios ponderados",
        ],
    },
    "requer_pesos": {
        "Sim": [
            "pesos como entrada",
            "pesos devem ser informados",
            "necessita pesos",
            "exige vetor de pesos",
            "requer definicao de pesos",
        ],
        "Nao": [
            "nao exige pesos",
            "nao requer definicao de pesos",
            "pesos nao precisam ser informados",
        ],
    },
}


def extend_lexicon() -> None:
    for attr, values in LEXICON_EXTENSIONS.items():
        for value, terms in values.items():
            LEXICON.setdefault(attr, {}).setdefault(value, [])
            existing = set(LEXICON[attr][value])
            for term in terms:
                if term not in existing:
                    LEXICON[attr][value].append(term)
                    existing.add(term)


extend_lexicon()

JACCARD_STOPWORDS = {
    "a",
    "ao",
    "aos",
    "as",
    "com",
    "como",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "entre",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "ou",
    "para",
    "por",
    "que",
    "se",
    "um",
    "uma",
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


def tokenize_for_similarity(value: str) -> list[str]:
    text = normalize_text(value)
    tokens = re.findall(r"\b[a-z0-9]{3,}\b", text)
    return [token for token in tokens if token not in JACCARD_STOPWORDS]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def best_window_jaccard(text_tokens: list[str], term_tokens: list[str]) -> float:
    if not text_tokens or not term_tokens:
        return 0.0

    term_set = set(term_tokens)
    window_size = max(1, len(term_tokens))
    best = 0.0
    for start in range(0, len(text_tokens)):
        window = text_tokens[start : start + window_size]
        if not window:
            continue
        best = max(best, jaccard(set(window), term_set))

    return best


def jaccard_scores_by_premise(text: str) -> dict[str, float]:
    """Calculate lexical Jaccard similarity between text and each premise lexicon.

    The score is complementary and diagnostic: it does not replace symbolic rules.
    It uses the best local overlap between the input text and controlled terms for
    each premise, preserving explainability without adding external dependencies.
    """
    text_tokens = tokenize_for_similarity(text)
    scores: dict[str, float] = {}

    for attr in PREMISE_ATTRS:
        best = 0.0
        for terms in LEXICON.get(attr, {}).values():
            for term in terms:
                term_tokens = tokenize_for_similarity(term)
                best = max(best, best_window_jaccard(text_tokens, term_tokens))
        scores[attr] = round(100.0 * best, 2)

    return scores


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


def infer_compensatoriedade_by_rules(text_norm: str) -> tuple[Optional[str], list[str], int]:
    """Infer compensatoriedade with robust semantic patterns.

    This complements the lexicon because users often write variations such as
    "compensacao", "compensacao parcial", "nao compensatorio" or even minor
    spelling mistakes around the compensatory root.
    """
    root = r"(compens\w*|compesn\w*|trade\s*-?\s*off)"
    veto_root = r"(veto|limiar|restricao minima|restricao obrigatoria|requisito minimo|criterio minimo|condicao minima|criterio eliminatorio)"
    negative_before = rf"\b(nao|sem|nunca|jamais)\b[\w\s,.;:-]{{0,45}}\b{root}\b"
    negative_after = rf"\b{root}\b[\w\s,.;:-]{{0,45}}\b(nao|nunca|jamais)\b"
    partial_near = rf"\b{root}\b[\w\s,.;:-]{{0,45}}\b(parcial\w*|limitad\w*)\b"
    partial_before = rf"\b(parcial\w*|limitad\w*)\b[\w\s,.;:-]{{0,45}}\b{root}\b"
    comp_with_veto = rf"\b{root}\b[\w\s,.;:-]{{0,80}}\b{veto_root}\b|\b{veto_root}\b[\w\s,.;:-]{{0,80}}\b{root}\b"
    compensatory = rf"\b{root}\b"

    if any(
        phrase in text_norm
        for phrase in [
            "veto parcial",
            "veto relativo",
            "limiar de veto parcial",
            "restricao com tolerancia",
            "requisito minimo com tolerancia",
        ]
    ):
        return "ParcialmenteCompensatorio", ["regra_semantica_veto_parcial"], 3
    if re.search(negative_before, text_norm) or re.search(negative_after, text_norm):
        return "NaoCompensatorio", ["regra_regex_nao_compensatorio"], 3
    if re.search(comp_with_veto, text_norm):
        return "ParcialmenteCompensatorio", ["regra_regex_compensacao_com_veto"], 3
    if re.search(partial_near, text_norm) or re.search(partial_before, text_norm):
        return "ParcialmenteCompensatorio", ["regra_regex_compensacao_parcial"], 3
    if re.search(compensatory, text_norm):
        return "Compensatorio", ["regra_regex_compensatorio"], 2
    if any(
        phrase in text_norm
        for phrase in [
            "ganho em um criterio",
            "perda em outro criterio",
            "maior custo pode ser aceito",
            "desempenho superior em um criterio",
            "desempenho inferior em outro criterio",
            "equilibrio entre criterios",
            "substituicao entre criterios",
            "troca entre criterios",
            "ganhos compensam perdas",
        ]
    ):
        return "Compensatorio", ["regra_semantica_tradeoff_implicito"], 2
    if any(
        phrase in text_norm
        for phrase in [
            "criterio eliminatorio",
            "requisito minimo obrigatorio",
            "veto absoluto",
            "restricao obrigatoria",
            "condicao eliminatoria",
        ]
    ):
        return "NaoCompensatorio", ["regra_semantica_nao_compensatorio"], 3
    return None, [], 0


def infer_monotonicidade_by_rules(text_norm: str) -> tuple[Optional[str], list[str], int]:
    """Infer monotonicity from common MCDA wording."""
    non_monotonic_patterns = [
        r"\bnao\b[\w\s,.;:-]{0,35}\bmonotonic\w*",
        r"\bnao\b[\w\s,.;:-]{0,25}\be\b[\w\s,.;:-]{0,25}\bcrescent\w*",
        r"\bnao\b[\w\s,.;:-]{0,25}\be\b[\w\s,.;:-]{0,25}\bdecrescent\w*",
        r"\bnao\b[\w\s,.;:-]{0,35}\bcrescent\w*",
        r"\bnao\b[\w\s,.;:-]{0,35}\bdecrescent\w*",
        r"\bnao\b[\w\s,.;:-]{0,45}\bimplica\b[\w\s,.;:-]{0,25}\bmelhora\w*",
        r"\baumento\b[\w\s,.;:-]{0,45}\bnao\b[\w\s,.;:-]{0,35}\bmelhora\w*",
        r"\breducao\b[\w\s,.;:-]{0,45}\bnao\b[\w\s,.;:-]{0,35}\bmelhora\w*",
        r"\b(faixa|zona|intervalo)\b[\w\s,.;:-]{0,35}\b(ideal|otim\w*)\b",
        r"\b(ideal|otim\w*)\b[\w\s,.;:-]{0,25}\bentre\b",
        r"\bponto\b[\w\s,.;:-]{0,25}\botim\w*\b",
        r"\bmelhor\b[\w\s,.;:-]{0,35}\bvalores?\b[\w\s,.;:-]{0,20}\bintermediari\w*\b",
    ]
    if any(re.search(pattern, text_norm) for pattern in non_monotonic_patterns) or any(
        phrase in text_norm
        for phrase in [
            "relacao nao linear",
            "preferencia nao linear",
            "efeito limiar",
            "ponto de saturacao",
            "limiar de veto",
            "faixa ideal",
            "zona ideal",
            "zona otima",
            "faixa otima",
            "intervalo ideal",
            "ideal entre",
            "ponto otimo",
            "valor otimo intermediario",
            "preferencia em formato de u",
            "melhor em valores intermediarios",
            "piora depois de certo ponto",
        ]
    ):
        return "NaoMonotonico", ["regra_semantica_nao_monotonico"], 3

    monotonic_patterns = [
        r"\bquanto\s+maior\b[\w\s,.;:-]{0,60}\bmelhor\b",
        r"\bquanto\s+menor\b[\w\s,.;:-]{0,60}\bmelhor\b",
        r"\bmaior\b[\w\s,.;:-]{0,35}\b(melhor|preferivel|desejavel)\b",
        r"\bmenor\b[\w\s,.;:-]{0,35}\b(melhor|preferivel|desejavel)\b",
        r"\bmaximiz\w*\b[\w\s,.;:-]{0,35}\b(desempenho|beneficio|valor|qualidade|retorno)\b",
        r"\bminimiz\w*\b[\w\s,.;:-]{0,35}\b(custo|perda|risco|tempo|prazo)\b",
    ]
    monotonic_phrases = [
        "quanto maior melhor",
        "quanto maior, melhor",
        "quanto menor melhor",
        "quanto menor, melhor",
        "mais e melhor",
        "menos e melhor",
        "criterio beneficio",
        "criterio de beneficio",
        "criterio monotonico de beneficio",
        "criterio custo",
        "criterio de custo",
        "criterio monotonico de custo",
        "aumento melhora",
        "reducao melhora",
        "maior valor e melhor",
        "menor valor e melhor",
        "maior desempenho e melhor",
        "menor custo e melhor",
        "preferencia crescente",
        "preferencia decrescente",
        "funcao de preferencia crescente",
        "funcao de preferencia decrescente",
        "dominancia",
    ]
    if (
        any(phrase in text_norm for phrase in monotonic_phrases)
        or any(re.search(pattern, text_norm) for pattern in monotonic_patterns)
        or re.search(r"\bmonotonic\w*", text_norm)
    ):
        return "Monotonico", ["regra_semantica_monotonico"], 3

    return None, [], 0


def infer_ponderabilidade_by_rules(text_norm: str) -> tuple[Optional[str], list[str], int]:
    """Infer whether criteria can be weighted."""
    non_weighted_patterns = [
        r"\bsem\b[\w\s,.;:-]{0,25}\bpesos?\b",
        r"\bnao\b[\w\s,.;:-]{0,25}\busa\b[\w\s,.;:-]{0,15}\bpesos?\b",
        r"\bdispensa\b[\w\s,.;:-]{0,25}\bpesos?\b",
        r"\bnao\b[\w\s,.;:-]{0,25}\bponder\w*",
    ]
    if any(re.search(pattern, text_norm) for pattern in non_weighted_patterns):
        return "NaoPonderavel", ["regra_semantica_sem_pesos"], 3

    partial_phrases = [
        "pesos parciais",
        "ponderacao parcial",
        "importancia aproximada",
        "pesos qualitativos",
        "preferencia aproximada",
    ]
    if any(phrase in text_norm for phrase in partial_phrases):
        return "SemiPonderavel", ["regra_semantica_ponderacao_parcial"], 3

    weighted_phrases = [
        "pesos definidos",
        "pesos dos criterios",
        "peso dos criterios",
        "criterios ponderados",
        "importancia relativa",
        "atribuicao direta de pesos",
        "comparacao par a par",
        "comparacao par-a-par",
        "comparacoes par a par",
        "comparacoes par-a-par",
        "swing weighting",
        "best worst method",
        "bwm",
        "ahp",
        "entropia",
        "critic",
    ]
    if any(phrase in text_norm for phrase in weighted_phrases) or re.search(r"\bpesos?\b", text_norm):
        return "Ponderavel", ["regra_semantica_ponderavel"], 2

    return None, [], 0


def infer_completude_by_rules(text_norm: str) -> tuple[Optional[str], list[str], int]:
    """Infer preference completeness from textual evidence."""
    incomplete_patterns = [
        r"\bpreferencias?\b[\w\s,.;:-]{0,35}\bnao\b[\w\s,.;:-]{0,20}\bcomplet\w*",
        r"\bcomparacoes?\b[\w\s,.;:-]{0,35}\bnao\b[\w\s,.;:-]{0,20}\bcomplet\w*",
        r"\bjulgamentos?\b[\w\s,.;:-]{0,35}\bfaltant\w*",
        r"\bmatriz\b[\w\s,.;:-]{0,35}\bincomplet\w*",
    ]
    if any(re.search(pattern, text_norm) for pattern in incomplete_patterns):
        return "Incompleto", ["regra_regex_incompleto"], 3

    incomplete_phrases = [
        "preferencias incompletas",
        "informacao incompleta",
        "informacoes incompletas",
        "comparacao parcial",
        "comparacoes parciais",
        "dados ausentes",
        "lacunas",
        "incomparabilidade",
        "sem todas as comparacoes",
        "nem todas as alternativas",
        "nem todos os pares",
    ]
    if any(phrase in text_norm for phrase in incomplete_phrases) or "incomplet" in text_norm:
        return "Incompleto", ["regra_semantica_incompleto"], 3

    complete_patterns = [
        r"\bpreferencias?\b[\w\s,.;:-]{0,35}\bcomplet\w*",
        r"\bcomparacoes?\b[\w\s,.;:-]{0,35}\bcomplet\w*",
        r"\btodos?\b[\w\s,.;:-]{0,35}\bjulgamentos?\b[\w\s,.;:-]{0,35}\binformad\w*",
        r"\bmatriz\b[\w\s,.;:-]{0,35}\bcomplet\w*",
    ]
    if any(re.search(pattern, text_norm) for pattern in complete_patterns):
        return "Completo", ["regra_regex_completo"], 3

    complete_phrases = [
        "preferencias completas",
        "informacao completa",
        "informacoes completas",
        "comparacao completa",
        "comparacoes completas",
        "todas as alternativas",
        "todas as comparacoes",
        "todos os pares",
        "comparabilidade total",
        "matriz completa",
    ]
    if any(phrase in text_norm for phrase in complete_phrases):
        return "Completo", ["regra_semantica_completo"], 3

    return None, [], 0


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
        or contains_term(text_norm, "criterios numericos e qualitativos")
        or contains_term(text_norm, "indicadores objetivos e subjetivos")
        or ("numerico" in text_norm and "qualitativ" in text_norm)
    ):
        premises["tipo_variavel"] = "Mista"
        evidence_map["tipo_variavel"] = evidence_map["tipo_variavel"] + ["regra_precedencia_variavel_mista"]
        score_map["tipo_variavel"] = max(score_map["tipo_variavel"], 2)
    elif any(term in text_norm for term in ["dados numericos", "valores numericos", "indicadores mensuraveis"]):
        premises["tipo_variavel"] = "Cardinal"
        evidence_map["tipo_variavel"] = evidence_map["tipo_variavel"] + ["regra_precedencia_variavel_cardinal"]
        score_map["tipo_variavel"] = max(score_map["tipo_variavel"], 2)
    elif any(term in text_norm for term in ["escala ordinal", "categorias ordenadas", "baixo medio alto"]):
        premises["tipo_variavel"] = "Ordinal"
        evidence_map["tipo_variavel"] = evidence_map["tipo_variavel"] + ["regra_precedencia_variavel_ordinal"]
        score_map["tipo_variavel"] = max(score_map["tipo_variavel"], 2)

    if any(
        term in text_norm
        for term in [
            "incerteza",
            "incerto",
            "imprecis",
            "cenario futuro",
            "cenarios futuros",
            "subjetividade",
            "probabilidades desconhecidas",
            "variabilidade desconhecida",
            "ambiente incerto",
        ]
    ):
        premises["ambiente_decisao"] = "Incerteza"
        evidence_map["ambiente_decisao"] = evidence_map["ambiente_decisao"] + ["regra_precedencia_incerteza"]
        score_map["ambiente_decisao"] = max(score_map["ambiente_decisao"], 2)
    elif any(term in text_norm for term in ["probabilidade conhecida", "probabilidades conhecidas", "cenarios probabilisticos", "risco mensuravel"]):
        premises["ambiente_decisao"] = "Risco"
        evidence_map["ambiente_decisao"] = evidence_map["ambiente_decisao"] + ["regra_precedencia_risco"]
        score_map["ambiente_decisao"] = max(score_map["ambiente_decisao"], 2)
    elif any(term in text_norm for term in ["sem incerteza", "dados deterministico", "dados deterministas", "cenario estavel"]):
        premises["ambiente_decisao"] = "Certeza"
        evidence_map["ambiente_decisao"] = evidence_map["ambiente_decisao"] + ["regra_precedencia_certeza"]
        score_map["ambiente_decisao"] = max(score_map["ambiente_decisao"], 2)

    complete_value, complete_evidence, complete_score = infer_completude_by_rules(text_norm)
    if complete_value is not None:
        premises["completude_pref"] = complete_value
        evidence_map["completude_pref"] = evidence_map["completude_pref"] + complete_evidence
        score_map["completude_pref"] = max(score_map["completude_pref"], complete_score)

    monotonic_value, monotonic_evidence, monotonic_score = infer_monotonicidade_by_rules(text_norm)
    if monotonic_value is not None:
        premises["monotonicidade"] = monotonic_value
        evidence_map["monotonicidade"] = evidence_map["monotonicidade"] + monotonic_evidence
        score_map["monotonicidade"] = max(score_map["monotonicidade"], monotonic_score)

    ponder_value, ponder_evidence, ponder_score = infer_ponderabilidade_by_rules(text_norm)
    if ponder_value is not None:
        premises["ponderabilidade"] = ponder_value
        evidence_map["ponderabilidade"] = evidence_map["ponderabilidade"] + ponder_evidence
        score_map["ponderabilidade"] = max(score_map["ponderabilidade"], ponder_score)
        if ponder_value != "NaoPonderavel":
            premises["usa_pesos"] = premises.get("usa_pesos") or "Sim"
            premises["requer_pesos"] = premises.get("requer_pesos") or "Sim"
            score_map["usa_pesos"] = max(score_map["usa_pesos"], 2)
            score_map["requer_pesos"] = max(score_map["requer_pesos"], 2)
            evidence_map["usa_pesos"] = evidence_map["usa_pesos"] + ["regra_semantica_pesos"]
            evidence_map["requer_pesos"] = evidence_map["requer_pesos"] + ["regra_semantica_pesos"]
        else:
            premises["usa_pesos"] = "Nao"
            premises["requer_pesos"] = "Nao"
            score_map["usa_pesos"] = max(score_map["usa_pesos"], 2)
            score_map["requer_pesos"] = max(score_map["requer_pesos"], 2)
            evidence_map["usa_pesos"] = evidence_map["usa_pesos"] + ["regra_semantica_sem_pesos"]
            evidence_map["requer_pesos"] = evidence_map["requer_pesos"] + ["regra_semantica_sem_pesos"]

    comp_value, comp_evidence, comp_score = infer_compensatoriedade_by_rules(text_norm)
    if comp_value is not None:
        premises["compensatoriedade"] = comp_value
        evidence_map["compensatoriedade"] = evidence_map["compensatoriedade"] + comp_evidence
        score_map["compensatoriedade"] = max(score_map["compensatoriedade"], comp_score)

    return premises, evidence_map, score_map
