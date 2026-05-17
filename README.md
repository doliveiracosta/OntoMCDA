# OntoMCDA: Recomendador Semantico MCDA

Plataforma Streamlit para recomendacao explicavel de metodos multicriterio a partir de uma descricao textual do problema decisorio e da ontologia `OntoMCDA_v2.owl`.

Desenvolvido por David de Oliveira Costa.

## O que a plataforma faz

- Recebe um texto livre descrevendo o problema de decisao.
- Infere premissas metodologicas por regras de PLN alinhadas a tese.
- Consulta a ontologia OWL como base de conhecimento.
- Ranqueia metodos MCDA por aderencia.
- Explica criterios atendidos e restricoes fortes violadas.
- Gera relatorio PDF para compartilhamento academico.

## Arquivo principal

```text
ontomcda_app.py
```

## Estrutura minima

```text
data/OntoMCDA_v2.owl
assets/logo_upe_poli.png
assets/logo_ppgec.png
ontomcda/
ontomcda_app.py
requirements.txt
```

## Rodar localmente

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run ontomcda_app.py
```

## Publicar no Streamlit Cloud

1. Crie um repositorio GitHub para a plataforma.
2. Envie os arquivos da estrutura minima.
3. No Streamlit Cloud, selecione `ontomcda_app.py` como arquivo principal.
4. Aguarde a instalacao das dependencias de `requirements.txt`.
5. Teste a recomendacao e o download do PDF antes de divulgar o link.

## Observacao academica

Esta versao e um MVP publicavel. Ela usa inferencia lexical e regras transparentes para preservar explicabilidade. A evolucao natural e incorporar embeddings ou LLM supervisionado pela ontologia, mantendo a rastreabilidade das premissas inferidas.
