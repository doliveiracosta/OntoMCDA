# OntoMCDA public deployment

Aplicacao Streamlit para recomendar metodos multicriterio a partir de descricao textual, usando a ontologia `OntoMCDA_v2.owl`.

## Arquivo principal

```text
ontomcda_app.py
```

## Estrutura necessaria

```text
data/OntoMCDA_v2.owl
ontomcda/
ontomcda_app.py
requirements_ontomcda.txt
```

No Streamlit Cloud, se o app usar `requirements.txt` por padrao, copie o conteudo de `requirements_ontomcda.txt` para `requirements.txt` ou crie um repositorio separado para a plataforma OntoMCDA. O pacote de upload gerado por este projeto ja inclui um `requirements.txt` proprio para a OntoMCDA.

## Comportamento do MVP

- Recebe uma descricao textual do problema decisorio.
- Extrai premissas por regras de PLN.
- Consulta perfis de metodos MCDA na ontologia OWL.
- Ranqueia metodos por aderencia.
- Explica criterios atendidos e restricoes fortes.
- Calcula metricas quantitativas operacionais de PLN.
- Gera relatorio PDF.

## Arquivos para upload publico

Use apenas os arquivos da pasta `github-upload-files-ontomcda-*` gerada localmente:

```text
.streamlit/config.toml
assets/logo_upe.jfif
assets/logo_upe_poli.png
assets/logo_ppgec.png
assets/logo_orcid.svg
assets/logo_linkedin.svg
data/OntoMCDA_v2.owl
ontomcda/
ontomcda_app.py
requirements.txt
README.md
DEPLOYMENT_ONTOMCDA.md
```

## Evolucao recomendada

- Ampliar o lexico com base na tese completa.
- Incluir embeddings/LLM supervisionado pela ontologia para melhorar a extracao semantica.
- Criar perguntas guiadas quando premissas obrigatorias estiverem ausentes.
- Persistir casos de uso anonimizados para validacao academica.
