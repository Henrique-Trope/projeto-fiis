# Projeto FIIs: Previsão de Outperformance em Relação ao IFIX

Projeto de Engenharia de Dados e Machine Learning desenvolvido para estimar se o **retorno total de Fundos de Investimento Imobiliário do segmento logístico superará o IFIX nos 7 pregões seguintes**.

O trabalho cobre todo o ciclo de dados: coleta, versionamento, arquitetura Medallion no Databricks, criação de features, validação temporal, comparação de modelos, backtest, teste final fora da amostra e explicabilidade com SHAP.

> **Aviso:** este projeto possui finalidade acadêmica e exploratória. Os resultados não constituem recomendação de investimento.

## Principais resultados

No teste final realizado em um período de 12 meses completamente separado do desenvolvimento, entre **01/03/2025 e 28/02/2026**, o modelo obteve:

| Métrica | Resultado |
|---|---:|
| ROC-AUC | **0,6175** |
| Accuracy | **58,30%** |
| Precision | **57,34%** |
| Recall | **56,48%** |
| F1-score | **56,90%** |
| Observações no holdout | **1.235** |
| Datas avaliadas | **247** |

O modelo demonstrou capacidade moderadamente superior ao acaso para classificar individualmente se um FII superaria o IFIX. Entretanto, essa capacidade não se converteu em um ranking Top 1 confiável:

| Avaliação de ranking | Resultado |
|---|---:|
| Top 1 diário, com horizontes sobrepostos | **47,77%** |
| Top 1 em 36 decisões não sobrepostas | **36,11%** |
| Top 1 como melhor FII absoluto | **22,22%** |

A conclusão principal é que o modelo funcionou melhor como **classificador individual de outperformance** do que como mecanismo para escolher, entre os cinco fundos, o melhor ativo da semana.

## Objetivo

Para cada FII e data de referência, o modelo responde:

> Qual é a probabilidade de o retorno total deste FII superar o retorno do IFIX nos próximos 7 pregões?

O retorno total considera:

- variação do preço da cota;
- dividendos associados à data ex-dividendo;
- comparação com o retorno do IFIX no mesmo horizonte.

## Universo analisado

Foram analisados cinco FIIs do segmento logístico:

- `HGLG11`
- `BTLG11`
- `XPLG11`
- `LVBI11`
- `VILG11`

A escolha de um universo fixo e pequeno facilita a auditoria, mas limita a generalização dos resultados para outros FIIs e segmentos.

## Fontes de dados

### FIIs

Os dados dos fundos foram obtidos a partir do **Status Invest**, após a identificação de lacunas nos históricos inicialmente consultados em outras fontes.

Foram coletados:

- cotações históricas;
- datas ex-dividendo;
- datas de pagamento;
- valores distribuídos;
- informações de ajuste dos eventos.

### IFIX

O histórico do IFIX foi obtido com **Yahoo Finance / yfinance**, consolidado em uma série diária entre 2011 e 2026.

### Indicadores macroeconômicos

As séries macroeconômicas foram obtidas pelo Sistema Gerenciador de Séries Temporais do Banco Central do Brasil:

- SELIC diária, SGS 11;
- dólar comercial de venda, SGS 1;
- IPCA mensal, SGS 433;
- taxa de desocupação PNAD Contínua, SGS 24369.

As variáveis mensais foram utilizadas com defasagens de disponibilidade para reduzir risco de uso antecipado de informações ainda não divulgadas.

## Tecnologias utilizadas

- Python
- Pandas
- PySpark
- Databricks Serverless
- Delta Lake
- Unity Catalog
- Git e GitHub
- yfinance
- scikit-learn
- XGBoost
- LightGBM
- SHAP
- Matplotlib e Seaborn

## Arquitetura

O projeto utiliza uma arquitetura Medallion:

```text
Fontes externas
      ↓
Arquivos versionados no GitHub
      ↓
Databricks Git Folder
      ↓
Bronze
      ↓
Silver
      ↓
Gold
      ↓
Machine Learning
      ↓
Walk-forward, backtest e holdout final
```

### Bronze

Responsável por consolidar os dados de origem com rastreabilidade.

| Tabela | Conteúdo |
|---|---|
| `workspace.bronze.fii_prices` | Cotações diárias dos cinco FIIs |
| `workspace.bronze.fii_dividends` | Eventos de dividendos |
| `workspace.bronze.ifix` | Histórico diário do IFIX |

As tabelas Bronze mantêm metadados como arquivo de origem e horário de ingestão.

### Silver

Responsável pela limpeza, seleção de campos de negócio e construção de retornos.

| Tabela | Conteúdo |
|---|---|
| `workspace.silver.fii_prices` | Preços e volume preparados para análise |
| `workspace.silver.fii_dividends` | Dividendos padronizados |
| `workspace.silver.ifix` | Série limpa do benchmark |
| `workspace.silver.fii_total_returns` | Retorno de preço, retorno de dividendos e retorno total |
| `workspace.silver.selic` | SELIC diária |
| `workspace.silver.ipca` | IPCA mensal |
| `workspace.silver.cotacao_dolar` | Dólar comercial diário |
| `workspace.silver.desemprego` | Taxa de desocupação mensal |

### Gold

A tabela principal de modelagem é:

```text
workspace.gold.fii_features_v1
```

A versão final possui 23 features, além dos identificadores e targets.

## Engenharia de features

### Retornos históricos do FII

- retorno de 1 pregão;
- retorno acumulado de 7 pregões;
- retorno acumulado de 30 pregões;
- retorno acumulado de 90 pregões.

### Retornos históricos do IFIX

- retorno de 1 pregão;
- retorno acumulado de 7 pregões;
- retorno acumulado de 30 pregões;
- retorno acumulado de 90 pregões.

### Risco e volatilidade

- volatilidade de 30 pregões;
- volatilidade de 90 pregões.

### Dividendos

- dividend yield dos últimos 365 dias corridos;
- quantidade de dias desde o último dividendo;
- presença de dividendo na data;
- quantidade de histórico disponível.

O dividend yield utiliza janela temporal de 365 dias corridos, e não 365 linhas de pregão.

### Comparação com o IFIX

- alpha de 30 pregões;
- alpha de 90 pregões.

### Macroeconomia

- SELIC;
- IPCA;
- dólar;
- desemprego.

## Definição do retorno total

Durante o projeto foi validado que o `adj_close` já incorpora ajustes retroativos. Somar dividendos explicitamente ao retorno calculado com `adj_close` causaria dupla contagem.

Por isso, o retorno total foi construído com preço de fechamento e dividendos explícitos:

```text
price_return = (close_atual / close_anterior) - 1

dividend_return = dividend_value / close_anterior

total_return = price_return + dividend_return
```

Os dividendos foram associados pela `ex_dividend_date`, que representa o momento econômico em que o preço tende a refletir o evento.

## Target

O target binário principal é:

```text
target_7d = retorno total futuro do FII > retorno futuro do IFIX
```

Também foi mantido um target contínuo auxiliar:

```text
target_alpha_7d = retorno futuro do FII - retorno futuro do IFIX
```

As features utilizam apenas dados históricos, enquanto o target utiliza exclusivamente os sete pregões posteriores à data de referência.

## Prevenção de data leakage

Foram aplicadas as seguintes medidas:

- splits exclusivamente temporais;
- ausência de embaralhamento aleatório;
- janelas históricas sem uso de dados futuros;
- purge gap de 7 pregões antes do holdout;
- defasagem de disponibilidade para IPCA e desemprego;
- fit de transformações somente no conjunto de treino;
- holdout de 12 meses não utilizado em seleção de features ou tuning;
- congelamento do modelo antes da execução final.

## Modelos avaliados

Foram comparados quatro algoritmos:

| Modelo | ROC-AUC no teste histórico |
|---|---:|
| Logistic Regression | 0,5941 |
| Random Forest | 0,6241 |
| LightGBM | 0,6251 |
| XGBoost | **0,6366** |

A Logistic Regression foi utilizada como baseline interpretável. O XGBoost apresentou o melhor desempenho histórico e foi selecionado para as etapas seguintes.

## Feature engineering V2 e estudo de ablação

Uma segunda versão da Gold foi criada com indicadores adicionais, como RSI, distância para médias móveis, beta, taxa de outperformance, drawdown e estabilidade dos dividendos.

Apesar de algumas features apresentarem importância no XGBoost, a Gold V2 reduziu o ROC-AUC no teste histórico. Um estudo de ablação mostrou que apenas `price_vs_ma_7d` melhorava isoladamente a validação, mas a comparação walk-forward indicou equivalência estatística entre a V1 e a versão mínima.

Pelo princípio da parcimônia, a Gold V1 foi mantida como versão final.

## Validação walk-forward

A validação utilizou janela expansiva e gap de 7 pregões. O modelo foi treinado somente com dados anteriores ao período avaliado.

Os resultados permaneceram acima de 0,50 nos diferentes períodos, demonstrando sinal preditivo temporal, embora com variação entre anos.

O tuning produziu uma versão do XGBoost com:

- 25% menos árvores;
- maior regularização;
- menor variabilidade entre folds;
- melhoria no pior fold;
- ganho médio moderado.

A versão ajustada foi escolhida por apresentar melhor equilíbrio entre performance, estabilidade e complexidade.

## Backtest

Foi realizado um backtest semanal com previsões estritamente fora da amostra.

A estratégia Top 1 apresentou:

- taxa histórica de outperformance de 57,94%;
- retorno bruto de 32,04%;
- turnover médio de 69,16%;
- alta sensibilidade a custos.

Com custo de 0,10% por lado, a estratégia ainda superou o IFIX por pequena margem. Com custo conservador de 0,20% por lado, o retorno líquido ficou negativo.

A conclusão foi que o sinal preditivo não se converteu em uma estratégia ativa robusta após custos realistas. A carteira Equal Weight dos cinco FIIs apresentou melhor desempenho líquido, porém não depende das previsões do modelo e foi tratada como benchmark passivo.

## Teste final fora da amostra

O teste final utilizou dados de **01/03/2025 a 28/02/2026**, período não consultado durante seleção de features, escolha do algoritmo ou tuning.

Foram avaliadas 1.235 observações, correspondentes a 247 datas e cinco FIIs.

### Classificação individual

| Métrica | Resultado |
|---|---:|
| ROC-AUC | **0,6175** |
| Accuracy | **58,30%** |
| Precision | **57,34%** |
| Recall | **56,48%** |
| F1-score | **56,90%** |

### Desempenho por ativo

- Melhor ROC-AUC: `BTLG11`, aproximadamente 0,70.
- Melhor accuracy: `XPLG11`, acima de 62%.
- Pior resultado: `HGLG11`, próximo do acaso em parte das métricas.

### Ranking Top 1

Em 36 decisões não sobrepostas, espaçadas a cada 7 pregões:

- o Top 1 superou o IFIX em 13 decisões;
- taxa de outperformance: **36,11%**;
- intervalo de confiança de 95%: **20,42% a 51,80%**;
- o Top 1 foi o melhor dos cinco FIIs em **22,22%** das decisões.

O resultado não forneceu evidência de que o ranking Top 1 atingisse a meta histórica de aproximadamente 58%.

## Explicabilidade com SHAP

O modelo final foi interpretado com SHAP sobre o holdout completo.

### Features mais influentes

1. `ifix_return_1d`
2. `days_since_last_dividend`
3. `dividend_yield_12m`
4. `volume`
5. `ifix_return_30d`

O modelo associou retornos positivos recentes do IFIX a menor chance de o FII superar o benchmark. Dividend yield mais alto, maior volume e mais dias desde o último dividendo estiveram associados a aumentos na probabilidade prevista.

As quatro variáveis macro representaram cerca de **17,5% da importância total segundo SHAP**. O modelo associou SELIC e dólar mais elevados a contribuições positivas, enquanto IPCA e desemprego mais elevados contribuíram negativamente para as previsões.

Essas relações descrevem o comportamento do modelo e não devem ser interpretadas como causalidade econômica.

### Por que o Top 1 foi fraco?

A análise identificou três fatores principais:

- baixa separação entre as probabilidades do primeiro e segundo colocados;
- seleção excessiva do HGLG11;
- probabilidades úteis para classificação, mas insuficientemente calibradas para ranking.

O HGLG11 foi escolhido como Top 1 em mais de 40% das previsões, mas apresentou baixa taxa de acerto quando selecionado.

## Estrutura do repositório

```text
projeto-fiis/
├── ingestion/
│   ├── fetch_data.py
│   ├── fetch_dividends.py
│   ├── fetch_ifix.py
│   └── data/
│       ├── macros/
│       ├── *_cotacoes.csv
│       ├── *_dividendos.csv
│       └── IFIX_completo.csv
├── 01_bronze_fii_prices.ipynb
├── 02_silver_fii.py
├── 03_gold_fii.py
├── 31_ml_eda.ipynb
├── 32_ml_baseline.ipynb
├── 33_ml_advanced.ipynb
├── 34_feature_engineering_v2.ipynb
├── 35_ml_v2.ipynb
├── 37_walk_forward_validation.ipynb
├── 38_hyperparameter_tuning.ipynb
├── 39A_walk_forward_predictions.ipynb
├── 39B_backtest.ipynb
├── 40_final_holdout_test.ipynb
├── 41_top1_non_overlapping_audit.ipynb
├── 42_model_explainability.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

> Alguns arquivos podem ser exportados pelo Databricks em formato `.py` com metadados de notebook, dependendo da configuração do Git Folder.

## Como reproduzir

### 1. Clonar o repositório

```bash
git clone https://github.com/Henrique-Trope/projeto-fiis.git
cd projeto-fiis
```

### 2. Criar o ambiente virtual

```bash
python -m venv .venv
```

No Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Executar a ingestão

```bash
python ingestion/fetch_data.py
python ingestion/fetch_dividends.py
python ingestion/fetch_ifix.py
```

### 5. Sincronizar com o Databricks

Crie um Git Folder no Databricks apontando para:

```text
https://github.com/Henrique-Trope/projeto-fiis.git
```

### 6. Executar os notebooks na ordem

```text
Bronze → Silver → Gold → EDA → Baseline → Modelos avançados
→ Walk-forward → Tuning → Backtest → Holdout → SHAP
```

## Limitações

- universo restrito a cinco FIIs logísticos;
- período histórico limitado;
- possíveis diferenças de qualidade entre fontes;
- variáveis macro sujeitas a revisão e defasagem de divulgação;
- apenas um ano de holdout final;
- dependência de um universo definido previamente;
- classificação individual não implica ranking eficiente;
- custos e slippage podem inviabilizar estratégias ativas;
- SHAP explica o modelo, não a realidade econômica;
- resultados passados não garantem desempenho futuro.

## Conclusão

O projeto demonstrou que existe informação útil para prever, com desempenho moderadamente superior ao acaso, se um FII superará o IFIX nos sete pregões seguintes.

O XGBoost final alcançou **58,30% de acerto e ROC-AUC de 0,6175** em um ano completamente fora da amostra. Entretanto, a tentativa de usar as probabilidades para selecionar exclusivamente o melhor FII não se mostrou robusta.

O principal aprendizado é que um modelo pode apresentar capacidade preditiva estatística e ainda assim não gerar uma estratégia de investimento viável após custos ou uma ordenação confiável dos ativos. A separação entre classificação, ranking e resultado econômico foi essencial para uma avaliação honesta.

## Possíveis trabalhos futuros

- testar horizonte mensal para reduzir turnover;
- calibrar probabilidades para melhorar ranking;
- investigar o comportamento específico do HGLG11;
- automatizar atualizações macroeconômicas;
- validar continuamente em novos períodos;
- estudar estratégias que utilizem o classificador sem concentração Top 1.

## Autor

**Henrique Almeida Trope**

- GitHub: [Henrique-Trope](https://github.com/Henrique-Trope)
- Repositório: [projeto-fiis](https://github.com/Henrique-Trope/projeto-fiis)

---

Se este projeto foi útil, considere deixar uma estrela no repositório.
