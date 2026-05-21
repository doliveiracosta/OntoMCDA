"""Constants used by the OntoMCDA recommendation engine."""

APP_NAME = "OntoMCDA"
APP_SUBTITLE = "Modelo Semantico para Recomendacao de Metodos Multicriterio"
APP_OWNER = "David de Oliveira Costa"
APP_OWNER_LABEL = f"Desenvolvido por {APP_OWNER}, Doutorando em Engenharia de Computacao, 2026."
OWL_PATH = "data/OntoMCDA_v2.owl"

NEW_NS = "http://www.ontomcda.org/onto#"
OLD_NS = "http://www.ontomcda.org/onto#OntologyMCDA#"

ATTRS = [
    "tipo_problema",
    "compensatoriedade",
    "tipo_variavel",
    "monotonicidade",
    "estrutura_decisoria",
    "completude_pref",
    "ambiente_decisao",
    "ponderabilidade",
    "usa_pesos",
    "requer_pesos",
    "gera_pesos",
    "auxilia_gerar_pesos",
    "sugere_pesos",
]

PREMISE_ATTRS = [
    "tipo_problema",
    "compensatoriedade",
    "tipo_variavel",
    "monotonicidade",
    "estrutura_decisoria",
    "completude_pref",
    "ambiente_decisao",
    "ponderabilidade",
    "usa_pesos",
    "requer_pesos",
]

MANDATORY_QUERY_ATTRS = ["tipo_problema"]
DATA_ATTRS = {"usa_pesos", "requer_pesos", "gera_pesos", "auxilia_gerar_pesos", "sugere_pesos"}

ATTR_LABELS = {
    "tipo_problema": "Tipo de problema",
    "compensatoriedade": "Compensatoriedade",
    "tipo_variavel": "Tipo de variavel",
    "monotonicidade": "Monotonicidade",
    "estrutura_decisoria": "Estrutura decisoria",
    "completude_pref": "Completude das preferencias",
    "ambiente_decisao": "Ambiente de decisao",
    "ponderabilidade": "Ponderabilidade",
    "usa_pesos": "Usa pesos",
    "requer_pesos": "Requer pesos",
    "gera_pesos": "Gera pesos",
    "auxilia_gerar_pesos": "Auxilia gerar pesos",
    "sugere_pesos": "Sugere pesos",
}

ATTR_WEIGHTS = {
    "tipo_problema": 4.0,
    "compensatoriedade": 3.5,
    "tipo_variavel": 3.0,
    "monotonicidade": 2.0,
    "estrutura_decisoria": 2.0,
    "completude_pref": 1.5,
    "ambiente_decisao": 2.0,
    "ponderabilidade": 1.5,
    "usa_pesos": 1.5,
    "requer_pesos": 1.2,
    "gera_pesos": 0.8,
    "auxilia_gerar_pesos": 1.0,
    "sugere_pesos": 0.5,
}

ATTR_TO_PROP = {
    "tipo_problema": ["temTipoProblema", "temTipoDeProblema"],
    "compensatoriedade": ["temCompensatoriedade"],
    "tipo_variavel": ["temTipoVariavel", "temTipoDeVariavel"],
    "monotonicidade": ["temMonotonicidade", "suportaMonotonicidade"],
    "estrutura_decisoria": ["temEstruturaDecisoria"],
    "completude_pref": ["temCompletudePref"],
    "ambiente_decisao": ["temAmbienteDecisao", "temAmbienteDeDecisao"],
    "ponderabilidade": ["temPonderabilidade"],
    "usa_pesos": ["usaPesos"],
    "requer_pesos": ["requerPesos"],
    "gera_pesos": ["geraPesos"],
    "auxilia_gerar_pesos": ["auxiliaGerarPesos"],
    "sugere_pesos": ["sugerePesos"],
}
