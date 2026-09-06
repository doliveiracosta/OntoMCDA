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

PREMISE_GLOSSARY = [
    {
        "premissa": "Tipo de problema",
        "definicao": "Finalidade decisoria predominante do problema ou do metodo.",
        "valores": "Escolha; Ordenacao; Classificacao.",
        "natureza": "Estrutural",
    },
    {
        "premissa": "Compensatoriedade",
        "definicao": "Indica se desempenhos inferiores em determinados criterios podem ser compensados por desempenhos superiores em outros criterios.",
        "valores": "Compensatorio; Parcialmente compensatorio; Nao compensatorio.",
        "natureza": "Estrutural",
    },
    {
        "premissa": "Tipo de variavel",
        "definicao": "Caracteriza a natureza das informacoes utilizadas para avaliar as alternativas.",
        "valores": "Cardinal; Ordinal; Mista; Pseudo-cardinal; Cardinal fuzzy; Cardinal intervalar; Cardinal grey.",
        "natureza": "Estrutural",
    },
    {
        "premissa": "Monotonicidade",
        "definicao": "Indica se a preferencia associada a um criterio segue uma direcao definida, como criterio de beneficio ou de custo.",
        "valores": "Monotonico; Monotonicidade fraca; Nao monotonico.",
        "natureza": "Estrutural",
    },
    {
        "premissa": "Estrutura decisoria",
        "definicao": "Representa a configuracao dos participantes do processo decisorio.",
        "valores": "Monodecisor; Multidecisor; Mono/Multi.",
        "natureza": "Estrutural",
    },
    {
        "premissa": "Completude das preferencias",
        "definicao": "Indica se o metodo exige preferencias integralmente definidas ou admite informacao parcial, incompleta ou progressiva.",
        "valores": "Completa; Incompleta.",
        "natureza": "Estrutural",
    },
    {
        "premissa": "Ambiente de decisao",
        "definicao": "Caracteriza o grau de conhecimento disponivel sobre dados, consequencias e cenarios do problema decisorio.",
        "valores": "Certeza; Risco; Incerteza.",
        "natureza": "Estrutural",
    },
    {
        "premissa": "Ponderabilidade",
        "definicao": "Indica se o metodo admite, depende ou dispensa a atribuicao de importancia relativa aos criterios.",
        "valores": "Ponderavel; Parcialmente ponderavel; Nao ponderavel.",
        "natureza": "Estrutural",
    },
    {
        "premissa": "Usa pesos",
        "definicao": "Indica se pesos dos criterios participam do procedimento decisorio, ainda que sejam obtidos de modo direto, indireto, endogeno ou implicito.",
        "valores": "Sim; Nao.",
        "natureza": "Operacional",
    },
    {
        "premissa": "Requer pesos",
        "definicao": "Indica se o metodo exige pesos explicitos como entrada para sua aplicacao.",
        "valores": "Sim; Nao.",
        "natureza": "Operacional",
    },
]

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
