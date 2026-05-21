# Benchmark de vocabulario OntoMCDA

Fonte de referencia: `OntoMCDA_final.ipynb`.

O lexico da plataforma foi comparado ao vocabulario do notebook final da tese e reforcado por premissa. Os termos foram normalizados para ASCII, minusculas e sem acentos, preservando compatibilidade com a normalizacao textual do app.

## Cobertura aproximada apos reforco

| Premissa | Cobertura aproximada |
|---|---:|
| tipo_problema | 91.7% |
| tipo_variavel | 77.4% |
| estrutura_decisoria | 73.8% |
| gera_pesos | 71.4% |
| ambiente_decisao | 70.8% |
| requer_pesos | 70.6% |
| auxilia_gerar_pesos | 70.0% |
| completude_pref | 66.7% |
| sugere_pesos | 66.7% |
| usa_pesos | 65.0% |
| ponderabilidade | 58.8% |
| compensatoriedade | 58.7% |
| monotonicidade | 57.7% |

## Efeito observado em texto de teste completo

Antes do reforco, textos ricos ainda podiam gerar cobertura parcial. Apos o reforco lexical, um texto com 10 premissas explicitas retornou:

```text
ICS-PLN: 100.0%
TNI-PLN: 0.0%
IET-PLN: 100.0%
ICL-PLN: 70.0%
Premissas inferidas: 10/10
```

## Interpretacao

O reforco melhora a cobertura semantica e a confianca lexical do PLN sem transformar o sistema em caixa-preta. A plataforma continua baseada em regras explicaveis e evidencias textuais rastreaveis, o que favorece a defesa metodologica da tese.

## Proximos reforcos recomendados

- Ampliar `monotonicidade`, `compensatoriedade` e `ponderabilidade`, que ainda apresentam menor cobertura relativa.
- Adicionar redacoes reais de avaliadores externos para validar estabilidade fora do vocabulario autoral.
- Comparar as inferencias com gabarito supervisionado usando accuracy, F1 macro, kappa e taxa de nao-inferido por premissa.
