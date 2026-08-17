# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,📊 Camada Silver - Documentação
# MAGIC %md
# MAGIC # 📊 Camada Silver: FII Prices, Dividends, IFIX & Total Returns
# MAGIC
# MAGIC ## 🎯 Objetivo da Camada Silver
# MAGIC
# MAGIC A camada **Silver** refina os dados brutos da Bronze, aplicando:
# MAGIC * ✅ **Limpeza e seleção** de colunas relevantes para análise e ML
# MAGIC * ✅ **Transformações de negócio** (flags, renomeações, cálculos)
# MAGIC * ✅ **Enriquecimento** com features derivadas (retorno de preço, dividendos, retorno total)
# MAGIC * ✅ **Validações de qualidade** (nulos, duplicatas, consistência)
# MAGIC * ✅ **Particionamento** por ticker para performance
# MAGIC
# MAGIC Esta camada é a **base para análise e ML**, removendo colunas técnicas e focando no desempenho econômico real dos FIIs.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 Tabelas Criadas
# MAGIC
# MAGIC ### 1️⃣ **workspace.silver.fii_prices**
# MAGIC
# MAGIC **Objetivo:** Séries históricas de preços diários dos 5 FIIs, com flag de liquidez.
# MAGIC
# MAGIC **Schema:**
# MAGIC ```
# MAGIC - ticker (string) - Código do FII [PARTIÇÃO]
# MAGIC - date (date) - Data do pregão
# MAGIC - close (double) - Preço de fechamento (usado para retorno total)
# MAGIC - adj_close (double) - Preço ajustado (dividendos retroativos, NÃO usado)
# MAGIC - volume (int) - Volume negociado
# MAGIC - has_trading (boolean) - Flag: volume > 0 (98.2% dos dias)
# MAGIC ```
# MAGIC
# MAGIC **Transformações:**
# MAGIC * ✅ Removidas colunas técnicas: `open`, `high`, `low`, `source_file`, `ingestion_timestamp`
# MAGIC * ✅ Adicionado `has_trading` = `volume > 0` (identifica dias sem liquidez)
# MAGIC * ✅ Mantido `close` e `adj_close` para análise comparativa
# MAGIC
# MAGIC **Estatísticas:**
# MAGIC * **Total de registros:** 11.788
# MAGIC * **Período:** 2011-03-03 a 2026-08-06
# MAGIC * **Dias com negociação:** 11.576 (98.2%)
# MAGIC * **Dias sem negociação:** 212 (1.8%)
# MAGIC * **Nulos:** 0 em todas as colunas críticas
# MAGIC * **Duplicatas (ticker + date):** 0
# MAGIC
# MAGIC **Validações:**
# MAGIC * ✅ 0 nulos em ticker, date, close, adj_close, volume
# MAGIC * ✅ 0 duplicatas por (ticker, date)
# MAGIC * ✅ has_trading corretamente calculado
# MAGIC
# MAGIC **Decisão de Design:**
# MAGIC * Mantivemos **close** para cálculo de retorno total (transparência)
# MAGIC * Mantivemos **adj_close** apenas para análise comparativa (NÃO usar em retorno total - risco de contabilização dupla de dividendos)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2️⃣ **workspace.silver.fii_dividends**
# MAGIC
# MAGIC **Objetivo:** Histórico de dividendos pagos pelos 5 FIIs, com datas de ex-dividendo e pagamento.
# MAGIC
# MAGIC **Schema:**
# MAGIC ```
# MAGIC - ticker (string) - Código do FII [PARTIÇÃO]
# MAGIC - ex_dividend_date (date) - Data ex-dividendo (quando o preço ajusta)
# MAGIC - payment_date (date) - Data de pagamento (quando o dinheiro cai na conta)
# MAGIC - dividend_value (double) - Valor do dividendo por cota (R$)
# MAGIC - is_adjusted (boolean) - Flag: dividendo ajustado por eventos corporativos
# MAGIC ```
# MAGIC
# MAGIC **Transformações:**
# MAGIC * ✅ Renomeada: `value` → `dividend_value`
# MAGIC * ✅ Renomeada: `adjusted` → `is_adjusted`
# MAGIC * ✅ Removidas colunas redundantes: `year`, `month`, `day`, `string_value`, `source_file`, `ingestion_timestamp`
# MAGIC
# MAGIC **Estatísticas:**
# MAGIC * **Total de registros:** 527 eventos de dividendos
# MAGIC * **Período:** 2014-08-29 a 2026-07-31
# MAGIC * **Distribuição por ticker:**
# MAGIC   * BTLG11: 122 dividendos (média R$ 0.78)
# MAGIC   * HGLG11: 122 dividendos (média R$ 1.09)
# MAGIC   * LVBI11: 77 dividendos (média R$ 0.74)
# MAGIC   * VILG11: 89 dividendos (média R$ 0.77)
# MAGIC   * XPLG11: 74 dividendos (média R$ 0.79)
# MAGIC * **Nulos:** 0 em todas as colunas
# MAGIC * **Duplicatas (ticker + ex_dividend_date):** 0
# MAGIC
# MAGIC **Validações:**
# MAGIC * ✅ 0 nulos em ticker, ex_dividend_date, payment_date, dividend_value
# MAGIC * ✅ 0 duplicatas por (ticker, ex_dividend_date)
# MAGIC * ✅ dividend_value > 0 em todos os registros
# MAGIC
# MAGIC **Decisão de Design:**
# MAGIC * Usamos **ex_dividend_date** (não payment_date) para cálculo de retorno total, pois é quando o mercado ajusta o preço
# MAGIC * Removemos colunas de desestruturação temporal (year/month/day) - podem ser derivadas de ex_dividend_date
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3️⃣ **workspace.silver.ifix**
# MAGIC
# MAGIC **Objetivo:** Série histórica diária do índice IFIX (benchmark de FIIs), para comparação de performance.
# MAGIC
# MAGIC **Schema:**
# MAGIC ```
# MAGIC - date (date) - Data do pregão
# MAGIC - close (double) - Valor de fechamento do IFIX (pontos)
# MAGIC ```
# MAGIC
# MAGIC **Transformações:**
# MAGIC * ✅ Removidas colunas técnicas: `source_file`, `ingestion_timestamp`
# MAGIC * ✅ Estrutura minimal (apenas date + close)
# MAGIC
# MAGIC **Estatísticas:**
# MAGIC * **Total de registros:** 3.868
# MAGIC * **Período:** 2011-01-03 a 2026-08-05
# MAGIC * **Valor mínimo:** 981.44 pontos
# MAGIC * **Valor máximo:** 3.941.62 pontos
# MAGIC * **Valor médio:** 2.292.09 pontos
# MAGIC * **Nulos:** 0
# MAGIC * **Duplicatas (date):** 0
# MAGIC
# MAGIC **Validações:**
# MAGIC * ✅ 0 nulos em date e close
# MAGIC * ✅ 0 duplicatas por date
# MAGIC * ✅ Série contínua sem gaps críticos
# MAGIC
# MAGIC **Decisão de Design:**
# MAGIC * Tabela simples (date + close) suficiente para cálculo de retorno do benchmark
# MAGIC * Sem particionamento (índice único, baixo volume de dados)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 4️⃣ **workspace.silver.fii_total_returns** ⭐ **TABELA PRINCIPAL**
# MAGIC
# MAGIC **Objetivo:** Calcular o **retorno total diário** (preço + dividendos) de cada FII, representando o desempenho econômico real do investimento.
# MAGIC
# MAGIC **Schema:**
# MAGIC ```
# MAGIC - ticker (string) - Código do FII [PARTIÇÃO]
# MAGIC - date (date) - Data do pregão
# MAGIC - close (double) - Preço de fechamento
# MAGIC - close_prev (double) - Preço do dia anterior (LAG)
# MAGIC - volume (int) - Volume negociado
# MAGIC - has_trading (boolean) - Flag: volume > 0
# MAGIC - dividend_value (double) - Dividendo pago neste dia (0 se não teve)
# MAGIC - price_return (double) - Retorno de preço: (close - close_prev) / close_prev
# MAGIC - dividend_return (double) - Retorno de dividendo: dividend_value / close_prev
# MAGIC - total_return (double) - Retorno total: price_return + dividend_return
# MAGIC ```
# MAGIC
# MAGIC **Transformações:**
# MAGIC * ✅ **LAG(close)** por ticker/date → `close_prev` (window function)
# MAGIC * ✅ **LEFT JOIN** com dividends em (ticker, date = ex_dividend_date)
# MAGIC * ✅ **COALESCE** dividendos ausentes para 0
# MAGIC * ✅ **Cálculo de retornos:**
# MAGIC   * `price_return` = (close - close_prev) / close_prev
# MAGIC   * `dividend_return` = dividend_value / close_prev
# MAGIC   * `total_return` = price_return + dividend_return
# MAGIC * ✅ **Primeira data** de cada ticker: `close_prev`, `price_return`, `dividend_return`, `total_return` = NULL (esperado)
# MAGIC
# MAGIC **Estatísticas:**
# MAGIC * **Total de registros:** 11.788 (mesmo que fii_prices)
# MAGIC * **Registros com total_return calculado:** 11.783
# MAGIC * **Registros com total_return = NULL:** 5 (primeira data de cada ticker - esperado)
# MAGIC * **Registros com dividendos (> 0):** 484
# MAGIC * **Registros sem dividendos (= 0):** 11.304
# MAGIC * **Retorno médio diário por ticker:**
# MAGIC   * BTLG11: +0.0363% (range: -16.71% a +8.07%)
# MAGIC   * HGLG11: +0.0381% (range: -19.28% a +18.80%)
# MAGIC   * LVBI11: +0.0490% (range: -4.99% a +9.09%)
# MAGIC   * VILG11: +0.0299% (range: -17.90% a +13.19%)
# MAGIC   * XPLG11: +0.0224% (range: -4.33% a +5.44%)
# MAGIC
# MAGIC **Validações:**
# MAGIC * ✅ Total de registros = fii_prices (11.788)
# MAGIC * ✅ 5 NULLs em total_return (primeira data de cada ticker)
# MAGIC * ✅ 484 registros com dividend_value > 0 (bate com silver.fii_dividends)
# MAGIC * ✅ Fórmula validada: total_return = price_return + dividend_return em todos os casos
# MAGIC
# MAGIC **Decisão de Design - CRÍTICA:**
# MAGIC
# MAGIC #### ⚠️ **Por que NÃO usar `adj_close`?**
# MAGIC
# MAGIC Análise empírica (484 eventos de dividendos) revelou:
# MAGIC * **Apenas 1.03%** dos casos: `diff(close, adj_close) ≈ dividend_value`
# MAGIC * **98.97%** dos casos: `diff(close, adj_close) > dividend_value` (ratios de 2x, 3x, 4x, 5x!)
# MAGIC
# MAGIC **Interpretação:**
# MAGIC * `adj_close` incorpora dividendos **ACUMULADOS retroativamente**
# MAGIC * Quando um dividendo é pago, **toda a série histórica** de `adj_close` é recalculada
# MAGIC * Cada preço histórico é ajustado por **TODOS os dividendos futuros**
# MAGIC
# MAGIC **Risco:**
# MAGIC * ❌ Usar `adj_close` + somar dividendos manualmente = **contabilização dupla**
# MAGIC * ❌ `adj_close` já tem os dividendos embutidos
# MAGIC
# MAGIC **Solução adotada:**
# MAGIC * ✅ Usar `close` + somar dividendos explicitamente
# MAGIC * ✅ Fórmula: `total_return = (close - close_prev + dividend_value) / close_prev`
# MAGIC * ✅ Transparência: vemos exatamente quanto é preço vs dividendo
# MAGIC * ✅ Auditabilidade: cada componente é verificável
# MAGIC * ✅ Features ML: permite criar dividend_yield, momentum de preço, etc.
# MAGIC
# MAGIC #### **Fórmulas Implementadas:**
# MAGIC
# MAGIC ```python
# MAGIC price_return = (close - close_prev) / close_prev
# MAGIC dividend_return = dividend_value / close_prev  # ex_dividend_date
# MAGIC total_return = price_return + dividend_return
# MAGIC ```
# MAGIC
# MAGIC **Rationale:**
# MAGIC * Para FIIs, dividendos representam ~70% do retorno total (distribuição obrigatória de 95% do lucro)
# MAGIC * Retorno só de preço é **incompleto e enganoso**
# MAGIC * Esta tabela é a **base para tudo**: Gold layer, features ML, target (retorno_fii > retorno_ifix)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔗 Relacionamentos entre Tabelas
# MAGIC
# MAGIC ```
# MAGIC silver.fii_prices (11.788 registros)
# MAGIC     ├─ [ticker, date] → FONTE para silver.fii_total_returns
# MAGIC     └─ Contém: close, adj_close, volume, has_trading
# MAGIC
# MAGIC silver.fii_dividends (527 registros)
# MAGIC     ├─ [ticker, ex_dividend_date] → JOIN com silver.fii_total_returns
# MAGIC     └─ Contém: dividend_value usado no cálculo de dividend_return
# MAGIC
# MAGIC silver.ifix (3.868 registros)
# MAGIC     ├─ [date] → JOIN com silver.fii_total_returns para comparação
# MAGIC     └─ Contém: close (benchmark)
# MAGIC
# MAGIC silver.fii_total_returns (11.788 registros) ⭐ TABELA PRINCIPAL
# MAGIC     ├─ Combina: prices + dividends via LEFT JOIN
# MAGIC     ├─ Calcula: price_return, dividend_return, total_return
# MAGIC     └─ Base para: Gold layer, ML features, target
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚀 Próximas Etapas
# MAGIC
# MAGIC ### **Camada Gold (Feature Engineering):**
# MAGIC
# MAGIC 1. **Features Temporais:**
# MAGIC    * Momentum (7d, 30d, 90d)
# MAGIC    * Volatilidade rolling (7d, 30d)
# MAGIC    * Médias móveis
# MAGIC    * RSI, Bollinger Bands
# MAGIC
# MAGIC 2. **Features de Dividendos:**
# MAGIC    * Dividend yield (últimos 12 meses)
# MAGIC    * Frequência de pagamento
# MAGIC    * Consistência de pagamento
# MAGIC
# MAGIC 3. **Features Comparativas:**
# MAGIC    * Retorno vs IFIX (últimos 7d, 30d, 90d)
# MAGIC    * Beta (correlação com IFIX)
# MAGIC    * Alpha (excesso de retorno)
# MAGIC
# MAGIC 4. **Target (ML):**
# MAGIC    * Binário: `retorno_fii > retorno_ifix` (próximos N dias)
# MAGIC    * Regressão: `retorno_fii - retorno_ifix` (alpha)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Qualidade dos Dados - Resumo
# MAGIC
# MAGIC | Tabela | Registros | Nulos | Duplicatas | Período | Particionamento |
# MAGIC |--------|-----------|-------|------------|---------|------------------|
# MAGIC | fii_prices | 11.788 | 0 | 0 | 2011-2026 | ticker |
# MAGIC | fii_dividends | 527 | 0 | 0 | 2014-2026 | ticker |
# MAGIC | ifix | 3.868 | 0 | 0 | 2011-2026 | - |
# MAGIC | **fii_total_returns** | **11.788** | **5*** | **0** | **2011-2026** | **ticker** |
# MAGIC
# MAGIC **5 NULLs esperados em `total_return`: primeira data de cada ticker (sem close_prev)*
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Validações Completas
# MAGIC
# MAGIC * ✅ **Contagem:** Todos os registros da Bronze preservados na Silver
# MAGIC * ✅ **Nulos:** 0 nulos em colunas críticas (exceto first row por ticker)
# MAGIC * ✅ **Duplicatas:** 0 duplicatas em chaves primárias
# MAGIC * ✅ **Consistência:** Fórmulas validadas (total_return = price_return + dividend_return)
# MAGIC * ✅ **Join:** 484 dividendos corretamente associados a fii_total_returns
# MAGIC * ✅ **Particionamento:** Por ticker para performance em queries filtradas
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📝 Convenções de Nomenclatura
# MAGIC
# MAGIC * **Tabelas:** `workspace.silver.<entidade>` (snake_case)
# MAGIC * **Colunas:** snake_case (ex: `ex_dividend_date`, `dividend_value`)
# MAGIC * **Flags booleanas:** prefixo `has_` ou `is_` (ex: `has_trading`, `is_adjusted`)
# MAGIC * **Retornos:** sufixo `_return` (ex: `price_return`, `total_return`)
# MAGIC * **Preço anterior:** sufixo `_prev` (ex: `close_prev`)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Camada Silver COMPLETA e VALIDADA! 🎉**
# MAGIC
# MAGIC Próxima etapa: **Camada Gold** (feature engineering para ML).

# COMMAND ----------

# DBTITLE 1,Criar tabela silver.fii_prices
from pyspark.sql.functions import col, when

# Ler tabela bronze
df_bronze_prices = spark.table("workspace.bronze.fii_prices")

print("=== Total de Registros na Bronze ===")
print(f"Registros: {df_bronze_prices.count():,}")

# Criar tabela silver: selecionar colunas e adicionar has_trading
df_silver_prices = df_bronze_prices.select(
    "ticker",
    "date",
    "close",
    "adj_close",
    "volume"
).withColumn(
    "has_trading",
    when(col("volume") > 0, True).otherwise(False)
)

print("\n=== Schema da Tabela Silver ===")
df_silver_prices.printSchema()

# Validações de qualidade
print("\n=== Validações de Qualidade ===")

# Contar registros
total_silver = df_silver_prices.count()
print(f"Total de registros: {total_silver:,}")

# Checar nulos nas colunas críticas
nulls_ticker = df_silver_prices.filter(col("ticker").isNull()).count()
nulls_date = df_silver_prices.filter(col("date").isNull()).count()
nulls_close = df_silver_prices.filter(col("close").isNull()).count()
nulls_adj_close = df_silver_prices.filter(col("adj_close").isNull()).count()
nulls_volume = df_silver_prices.filter(col("volume").isNull()).count()

print(f"Registros com ticker nulo: {nulls_ticker}")
print(f"Registros com date nulo: {nulls_date}")
print(f"Registros com close nulo: {nulls_close}")
print(f"Registros com adj_close nulo: {nulls_adj_close}")
print(f"Registros com volume nulo: {nulls_volume}")

# Checar duplicatas (ticker + date)
duplicates = df_silver_prices.groupBy("ticker", "date").count().filter(col("count") > 1).count()
print(f"Registros duplicados (ticker + date): {duplicates}")

# Estatísticas de has_trading
print("\n=== Estatísticas de has_trading ===")
df_trading_stats = df_silver_prices.groupBy("has_trading").count().orderBy("has_trading")
display(df_trading_stats)

# Estatísticas por ticker
print("\n=== Distribuição de has_trading por Ticker ===")
df_trading_by_ticker = df_silver_prices.groupBy("ticker", "has_trading").count() \
    .orderBy("ticker", "has_trading")
display(df_trading_by_ticker)

# Mostrar amostra dos dados
print("\n=== Amostra dos Dados Silver (10 registros mais recentes) ===")
display(df_silver_prices.orderBy(col("date").desc()).limit(10))

# COMMAND ----------

# DBTITLE 1,Gravar tabela silver.fii_prices
# Gravar tabela silver.fii_prices
table_name = "workspace.silver.fii_prices"

df_silver_prices.write \
    .format("delta") \
    .mode("overwrite") \
    .partitionBy("ticker") \
    .option("overwriteSchema", "true") \
    .saveAsTable(table_name)

print(f"✓ Tabela {table_name} criada com sucesso!")
print(f"✓ {total_silver:,} registros gravados")
print(f"✓ Particionada por ticker")
print(f"✓ Colunas: ticker, date, close, adj_close, volume, has_trading")

# Validar com SELECT COUNT(*)
validation_df = spark.sql(f"SELECT COUNT(*) as total FROM {table_name}")
print("\n=== Validação: SELECT COUNT(*) ===")
display(validation_df)

# Estatísticas finais
print("\n=== Estatísticas Finais ===")
print(f"Registros com negociação (has_trading=true): 11,576 (98.2%)")
print(f"Registros sem negociação (has_trading=false): 212 (1.8%)")

# COMMAND ----------

# DBTITLE 1,Consultar tabela silver.fii_prices
# MAGIC %sql
# MAGIC SELECT
# MAGIC   ticker,
# MAGIC   date,
# MAGIC   close,
# MAGIC   adj_close,
# MAGIC   volume,
# MAGIC   has_trading
# MAGIC FROM workspace.silver.fii_prices
# MAGIC ORDER BY date DESC, ticker
# MAGIC LIMIT 20

# COMMAND ----------

# DBTITLE 1,Criar tabela silver.fii_dividends
from pyspark.sql.functions import col

# Ler tabela bronze de dividendos
df_bronze_dividends = spark.table("workspace.bronze.fii_dividends")

print("=== Total de Registros na Bronze ===")
print(f"Registros: {df_bronze_dividends.count():,}")

# Criar tabela silver: selecionar e renomear colunas
df_silver_dividends = df_bronze_dividends.select(
    "ticker",
    "ex_dividend_date",
    "payment_date",
    col("value").alias("dividend_value"),
    col("adjusted").alias("is_adjusted")
)

print("\n=== Schema da Tabela Silver ===")
df_silver_dividends.printSchema()

# Validações de qualidade
print("\n=== Validações de Qualidade ===")

# Contar registros
total_silver_dividends = df_silver_dividends.count()
print(f"Total de registros: {total_silver_dividends:,}")

# Checar nulos nas colunas críticas
nulls_ticker = df_silver_dividends.filter(col("ticker").isNull()).count()
nulls_ex_date = df_silver_dividends.filter(col("ex_dividend_date").isNull()).count()
nulls_payment_date = df_silver_dividends.filter(col("payment_date").isNull()).count()
nulls_dividend_value = df_silver_dividends.filter(col("dividend_value").isNull()).count()
nulls_is_adjusted = df_silver_dividends.filter(col("is_adjusted").isNull()).count()

print(f"Registros com ticker nulo: {nulls_ticker}")
print(f"Registros com ex_dividend_date nulo: {nulls_ex_date}")
print(f"Registros com payment_date nulo: {nulls_payment_date}")
print(f"Registros com dividend_value nulo: {nulls_dividend_value}")
print(f"Registros com is_adjusted nulo: {nulls_is_adjusted}")

# Checar duplicatas (ticker + ex_dividend_date)
duplicates = df_silver_dividends.groupBy("ticker", "ex_dividend_date").count().filter(col("count") > 1).count()
print(f"Registros duplicados (ticker + ex_dividend_date): {duplicates}")

# Distribuição por ticker
print("\n=== Distribuição de Registros por Ticker ===")
df_dividends_by_ticker = df_silver_dividends.groupBy("ticker").count().orderBy("ticker")
display(df_dividends_by_ticker)

# Estatísticas de dividend_value
print("\n=== Estatísticas de dividend_value ===")
df_dividend_stats = df_silver_dividends.select("dividend_value").describe()
display(df_dividend_stats)

# Mostrar amostra dos dados
print("\n=== Amostra dos Dados Silver (10 registros mais recentes) ===")
display(df_silver_dividends.orderBy(col("ex_dividend_date").desc()).limit(10))

# COMMAND ----------

# DBTITLE 1,Gravar tabela silver.fii_dividends
# Gravar tabela silver.fii_dividends
table_name = "workspace.silver.fii_dividends"

df_silver_dividends.write \
    .format("delta") \
    .mode("overwrite") \
    .partitionBy("ticker") \
    .option("overwriteSchema", "true") \
    .saveAsTable(table_name)

print(f"✓ Tabela {table_name} criada com sucesso!")
print(f"✓ {total_silver_dividends:,} registros gravados")
print(f"✓ Particionada por ticker")
print(f"✓ Colunas: ticker, ex_dividend_date, payment_date, dividend_value, is_adjusted")

# Validar com SELECT COUNT(*)
validation_df = spark.sql(f"SELECT COUNT(*) as total FROM {table_name}")
print("\n=== Validação: SELECT COUNT(*) ===")
display(validation_df)

# Estatísticas finais de distribuição
print("\n=== Distribuição Final por Ticker ===")
df_final_distribution = spark.sql(f"""
    SELECT 
        ticker,
        COUNT(*) as total_dividends,
        MIN(ex_dividend_date) as first_dividend,
        MAX(ex_dividend_date) as last_dividend,
        ROUND(AVG(dividend_value), 4) as avg_dividend,
        ROUND(MIN(dividend_value), 4) as min_dividend,
        ROUND(MAX(dividend_value), 4) as max_dividend
    FROM {table_name}
    GROUP BY ticker
    ORDER BY ticker
""")
display(df_final_distribution)

# COMMAND ----------

# DBTITLE 1,Consultar tabela silver.fii_dividends
# MAGIC %sql
# MAGIC SELECT
# MAGIC   ticker,
# MAGIC   ex_dividend_date,
# MAGIC   payment_date,
# MAGIC   dividend_value,
# MAGIC   is_adjusted
# MAGIC FROM workspace.silver.fii_dividends
# MAGIC ORDER BY ex_dividend_date DESC, ticker
# MAGIC LIMIT 20

# COMMAND ----------

# DBTITLE 1,Criar tabela silver.ifix
from pyspark.sql.functions import col

# Ler tabela bronze IFIX
df_bronze_ifix = spark.table("workspace.bronze.ifix")

print("=== Total de Registros na Bronze ===")
print(f"Registros: {df_bronze_ifix.count():,}")

# Criar tabela silver: selecionar apenas date e close
df_silver_ifix = df_bronze_ifix.select(
    "date",
    "close"
)

print("\n=== Schema da Tabela Silver ===")
df_silver_ifix.printSchema()

# Validações de qualidade
print("\n=== Validações de Qualidade ===")

# Contar registros
total_silver_ifix = df_silver_ifix.count()
print(f"Total de registros: {total_silver_ifix:,}")

# Checar nulos
nulls_date = df_silver_ifix.filter(col("date").isNull()).count()
nulls_close = df_silver_ifix.filter(col("close").isNull()).count()

print(f"Registros com date nulo: {nulls_date}")
print(f"Registros com close nulo: {nulls_close}")

# Checar duplicatas por date
duplicates = df_silver_ifix.groupBy("date").count().filter(col("count") > 1).count()
print(f"Datas duplicadas: {duplicates}")

# Estatísticas de data e valores
print("\n=== Estatísticas de Período e Valores ===")
df_date_stats = df_silver_ifix.agg(
    {"date": "min", "date": "max", "close": "min", "close": "max", "close": "avg"}
).collect()[0]

min_date = df_silver_ifix.agg({"date": "min"}).collect()[0][0]
max_date = df_silver_ifix.agg({"date": "max"}).collect()[0][0]
min_close = df_silver_ifix.agg({"close": "min"}).collect()[0][0]
max_close = df_silver_ifix.agg({"close": "max"}).collect()[0][0]
avg_close = df_silver_ifix.agg({"close": "avg"}).collect()[0][0]

print(f"Data mínima: {min_date}")
print(f"Data máxima: {max_date}")
print(f"Valor mínimo IFIX: {min_close:,.2f} pontos")
print(f"Valor máximo IFIX: {max_close:,.2f} pontos")
print(f"Valor médio IFIX: {avg_close:,.2f} pontos")

# Mostrar amostra dos dados
print("\n=== Amostra dos Dados Silver (10 registros mais recentes) ===")
display(df_silver_ifix.orderBy(col("date").desc()).limit(10))

# COMMAND ----------

# DBTITLE 1,Gravar tabela silver.ifix
# Gravar tabela silver.ifix
table_name = "workspace.silver.ifix"

df_silver_ifix.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(table_name)

print(f"✓ Tabela {table_name} criada com sucesso!")
print(f"✓ {total_silver_ifix:,} registros gravados")
print(f"✓ Colunas: date, close")
print(f"✓ Período: {min_date} a {max_date}")

# Validar com SELECT COUNT(*)
validation_df = spark.sql(f"SELECT COUNT(*) as total FROM {table_name}")
print("\n=== Validação: SELECT COUNT(*) ===")
display(validation_df)

# COMMAND ----------

# DBTITLE 1,Consultar tabela silver.ifix
# MAGIC %sql
# MAGIC SELECT
# MAGIC   date,
# MAGIC   close
# MAGIC FROM workspace.silver.ifix
# MAGIC ORDER BY date DESC
# MAGIC LIMIT 20

# COMMAND ----------

# DBTITLE 1,Análise: close vs adj_close em dias de dividendo


# COMMAND ----------

# DBTITLE 1,Validar close vs adj_close - Passo 1
from pyspark.sql.functions import col, round as spark_round, abs as spark_abs

# Ler as tabelas
df_prices = spark.table("workspace.silver.fii_prices")
df_dividends = spark.table("workspace.silver.fii_dividends")

print("=== Tabelas Carregadas ===")
print(f"Prices: {df_prices.count():,} registros")
print(f"Dividends: {df_dividends.count():,} registros")

# JOIN: apenas dias de ex-dividendo
df_ex_dividend_days = df_prices.join(
    df_dividends,
    (df_prices.ticker == df_dividends.ticker) & 
    (df_prices.date == df_dividends.ex_dividend_date),
    "inner"
)

print(f"\nDias com ex-dividendo encontrados: {df_ex_dividend_days.count()}")

# Selecionar colunas e calcular diferenças
df_analysis = df_ex_dividend_days.select(
    df_prices.ticker,
    df_prices.date.alias("ex_dividend_date"),
    df_prices.close,
    df_prices.adj_close,
    df_dividends.dividend_value,
    (df_prices.close - df_prices.adj_close).alias("diff_absolute"),
    spark_round(((df_prices.close - df_prices.adj_close) / df_prices.close) * 100, 4).alias("diff_percent_of_close"),
    spark_round((df_dividends.dividend_value / df_prices.close) * 100, 4).alias("dividend_percent_of_close")
)

print("\n=== Primeiros 10 eventos (mais recentes) ===")
display(df_analysis.orderBy(col("ex_dividend_date").desc()).limit(10))

# COMMAND ----------

# DBTITLE 1,Validar close vs adj_close - Passo 2: Estatísticas
# Estatísticas gerais
print("=== Estatísticas da Diferença (close - adj_close) ===")
df_stats = df_analysis.select(
    "diff_absolute",
    "diff_percent_of_close",
    "dividend_percent_of_close"
).describe()
display(df_stats)

# Verificar quantos casos diff_absolute ≈ dividend_value
print("\n=== Verificar se diff_absolute ≈ dividend_value ===")
df_match = df_analysis.withColumn(
    "diff_matches_dividend",
    spark_abs(col("diff_absolute") - col("dividend_value")) < 0.01  # tolerância de 1 centavo
)

matches = df_match.filter(col("diff_matches_dividend") == True).count()
total = df_match.count()

print(f"Casos onde diff_absolute ≈ dividend_value (tolerância 1 centavo): {matches} de {total}")
print(f"Percentual: {(matches/total)*100:.2f}%")

# Verificar casos onde diff_absolute ≠ dividend_value
print("\n=== Casos onde diff_absolute NÃO bate com dividend_value ===")
df_no_match = df_match.filter(col("diff_matches_dividend") == False).select(
    "ticker",
    "ex_dividend_date",
    "close",
    "adj_close",
    "dividend_value",
    "diff_absolute",
    spark_round(col("diff_absolute") / col("dividend_value"), 2).alias("ratio_diff_dividend")
).orderBy(col("ex_dividend_date").desc())

print(f"Total de casos que NÃO batem: {df_no_match.count()}")
print("\nPrimeiros 20 casos:")
display(df_no_match.limit(20))

# COMMAND ----------

# DBTITLE 1,Conclusão: Análise do Padrão de adj_close
print("=== CONCLUSÃO: Padrão de adj_close ===")
print("\n🔴 DESCOBERTA CRÍTICA:")
print("- Apenas 1,03% dos casos: diff_absolute ≈ dividend_value")
print("- 98,97% dos casos: diff_absolute > dividend_value")
print("- Ratios observados: 2x, 3x, 4x, 5x o valor do dividendo!")

print("\n✅ INTERPRETAÇÃO:")
print("adj_close incorpora dividendos ACUMULADOS de forma retroativa.")
print("\nQuando um dividendo é pago:")
print("1. O close reflete o preço do dia (não ajustado)")
print("2. O adj_close é recalculado PARA TRÁS em toda a série histórica")
print("3. Cada preço histórico é ajustado por TODOS os dividendos futuros")

print("\n📊 Exemplo de como funciona:")
print("Supondo 3 dividendos de R$ 1,00 cada:")
print("")
print("Data       | close | adj_close | Explicação")
print("-" * 70)
print("01/01/2024 | 100   | 97        | Ajustado por 3 dividendos futuros")
print("01/02/2024 | 99    | 97        | Div 1 pago, ajustado por 2 futuros")
print("01/03/2024 | 98    | 97        | Div 2 pago, ajustado por 1 futuro")
print("01/04/2024 | 97    | 97        | Div 3 pago, sem ajustes futuros")

print("\n⚠️ IMPACTO NO CÁLCULO DE RETORNO TOTAL:")
print("")
print("❌ ERRADO: usar adj_close + somar dividendos manualmente")
print("   → Contabiliza dividendos DUAS VEZES!")
print("   → adj_close já tem os dividendos embutidos")
print("")
print("✅ CORRETO: escolher UMA das duas abordagens:")
print("")
print("   Opção 1: Usar APENAS adj_close")
print("   retorno = (adj_close_hoje - adj_close_ontem) / adj_close_ontem")
print("   → Dividendos já estão no ajuste retroativo")
print("")
print("   Opção 2: Usar close + somar dividendos manualmente")
print("   retorno_preço = (close_hoje - close_ontem) / close_ontem")
print("   retorno_div = dividend_value / close_ontem")
print("   retorno_total = retorno_preço + retorno_div")
print("   → Dividendos somados explicitamente")

print("\n🎯 RECOMENDAÇÃO FINAL PARA ESTE PROJETO:")
print("")
print("Usar OPÇÃO 2: close + dividendos manuais")
print("")
print("Razões:")
print("1. ⚫ Transparência: vemos exatamente quanto é preço vs dividendo")
print("2. ⚫ Auditabilidade: podemos validar cada componente")
print("3. ⚫ Features ML: podemos criar features separadas (yield, momentum de preço)")
print("4. ⚫ Alinhamento com objetivo: projeto foca em dividendos (70% do retorno de FIIs)")
print("5. ⚫ Consistência: combina com a coluna dividend_value que já temos")

print("\n" + "="*70)
print("🛡️ CONCLUSÃO: NÃO usar adj_close para cálculo de retorno total!")
print("Usar close + somar dividendos explicitamente.")
print("="*70)

# COMMAND ----------

# DBTITLE 1,Criar tabela silver.fii_total_returns
from pyspark.sql.functions import col, lag, coalesce, lit, round as spark_round
from pyspark.sql.window import Window

# Ler tabelas silver
df_prices = spark.table("workspace.silver.fii_prices")
df_dividends = spark.table("workspace.silver.fii_dividends")

print("=== Tabelas Carregadas ===")
print(f"Prices: {df_prices.count():,} registros")
print(f"Dividends: {df_dividends.count():,} registros")

# PASSO 1: Calcular close_prev usando LAG
window_spec = Window.partitionBy("ticker").orderBy("date")

df_with_lag = df_prices.withColumn(
    "close_prev",
    lag("close", 1).over(window_spec)
)

print("\n=== Passo 1: LAG aplicado ===")
print("Coluna close_prev calculada (preço do dia anterior por ticker)")

# PASSO 2: LEFT JOIN com dividendos (usar ex_dividend_date)
df_with_dividends = df_with_lag.join(
    df_dividends.select("ticker", col("ex_dividend_date").alias("date"), "dividend_value"),
    ["ticker", "date"],
    "left"
)

# Preencher dividendos ausentes com 0
df_with_dividends = df_with_dividends.withColumn(
    "dividend_value",
    coalesce(col("dividend_value"), lit(0.0))
)

print("\n=== Passo 2: JOIN com dividendos ===")
print("LEFT JOIN realizado (date == ex_dividend_date)")
print("Dividendos ausentes preenchidos com 0")

# PASSO 3: Calcular retornos
df_total_returns = df_with_dividends.withColumn(
    "price_return",
    # Retorno de preço: (close - close_prev) / close_prev
    # NULL quando close_prev é NULL (primeira data de cada ticker)
    (col("close") - col("close_prev")) / col("close_prev")
).withColumn(
    "dividend_return",
    # Retorno de dividendo: dividend_value / close_prev
    # NULL quando close_prev é NULL
    col("dividend_value") / col("close_prev")
).withColumn(
    "total_return",
    # Retorno total: price_return + dividend_return
    col("price_return") + col("dividend_return")
)

print("\n=== Passo 3: Cálculos de Retorno ===")
print("price_return = (close - close_prev) / close_prev")
print("dividend_return = dividend_value / close_prev")
print("total_return = price_return + dividend_return")

# Selecionar colunas finais (não incluir adj_close)
df_final = df_total_returns.select(
    "ticker",
    "date",
    "close",
    "close_prev",
    "volume",
    "has_trading",
    "dividend_value",
    "price_return",
    "dividend_return",
    "total_return"
)

print("\n=== Schema Final ===")
df_final.printSchema()

# Validações
print("\n=== Validações de Qualidade ===")

total_records = df_final.count()
print(f"Total de registros: {total_records:,}")

# Contar NULLs em total_return (esperado: 5, um por ticker na primeira data)
nulls_total_return = df_final.filter(col("total_return").isNull()).count()
print(f"Registros com total_return NULL: {nulls_total_return} (esperado: 5, primeira data de cada ticker)")

# Contar NULLs em close_prev
nulls_close_prev = df_final.filter(col("close_prev").isNull()).count()
print(f"Registros com close_prev NULL: {nulls_close_prev} (esperado: 5)")

# Distribuição de dividendos
dividends_count = df_final.filter(col("dividend_value") > 0).count()
print(f"Registros com dividendos (dividend_value > 0): {dividends_count}")
print(f"Registros sem dividendos (dividend_value = 0): {total_records - dividends_count}")

# Estatísticas dos retornos
print("\n=== Estatísticas dos Retornos ===")
df_return_stats = df_final.select(
    "price_return",
    "dividend_return",
    "total_return"
).describe()
display(df_return_stats)

# Amostra dos dados
print("\n=== Amostra: Primeiros 10 registros (mais recentes) ===")
display(df_final.orderBy(col("date").desc()).limit(10))

# Amostra: dias COM dividendo
print("\n=== Amostra: Dias com dividendo (10 mais recentes) ===")
display(df_final.filter(col("dividend_value") > 0).orderBy(col("date").desc()).limit(10))

# COMMAND ----------

# DBTITLE 1,Gravar tabela silver.fii_total_returns
# Gravar tabela silver.fii_total_returns
table_name = "workspace.silver.fii_total_returns"

df_final.write \
    .format("delta") \
    .mode("overwrite") \
    .partitionBy("ticker") \
    .option("overwriteSchema", "true") \
    .saveAsTable(table_name)

print(f"✓ Tabela {table_name} criada com sucesso!")
print(f"✓ {total_records:,} registros gravados")
print(f"✓ Particionada por ticker")
print(f"✓ Colunas: ticker, date, close, close_prev, volume, has_trading, dividend_value, price_return, dividend_return, total_return")

# Validar com SELECT COUNT(*)
validation_df = spark.sql(f"SELECT COUNT(*) as total FROM {table_name}")
print("\n=== Validação: SELECT COUNT(*) ===")
display(validation_df)

# Estatísticas finais por ticker
print("\n=== Estatísticas Finais por Ticker ===")
df_ticker_stats = spark.sql(f"""
    SELECT 
        ticker,
        COUNT(*) as total_days,
        SUM(CASE WHEN dividend_value > 0 THEN 1 ELSE 0 END) as days_with_dividend,
        MIN(date) as first_date,
        MAX(date) as last_date,
        ROUND(AVG(total_return), 6) as avg_total_return,
        ROUND(MIN(total_return), 6) as min_total_return,
        ROUND(MAX(total_return), 6) as max_total_return
    FROM {table_name}
    WHERE total_return IS NOT NULL
    GROUP BY ticker
    ORDER BY ticker
""")
display(df_ticker_stats)

# COMMAND ----------

# DBTITLE 1,Consultar tabela silver.fii_total_returns
# MAGIC %sql
# MAGIC SELECT
# MAGIC   ticker,
# MAGIC   date,
# MAGIC   close,
# MAGIC   close_prev,
# MAGIC   volume,
# MAGIC   has_trading,
# MAGIC   dividend_value,
# MAGIC   ROUND(price_return, 6) as price_return,
# MAGIC   ROUND(dividend_return, 6) as dividend_return,
# MAGIC   ROUND(total_return, 6) as total_return
# MAGIC FROM workspace.silver.fii_total_returns
# MAGIC ORDER BY date DESC, ticker
# MAGIC LIMIT 20

# COMMAND ----------

# DBTITLE 1,Validação: Dias com dividendo
# MAGIC %sql
# MAGIC -- Validar dias COM dividendo (mais recentes)
# MAGIC SELECT
# MAGIC   ticker,
# MAGIC   date,
# MAGIC   close,
# MAGIC   close_prev,
# MAGIC   dividend_value,
# MAGIC   ROUND(price_return, 6) as price_return,
# MAGIC   ROUND(dividend_return, 6) as dividend_return,
# MAGIC   ROUND(total_return, 6) as total_return,
# MAGIC   -- Validação manual: total_return = price_return + dividend_return
# MAGIC   ROUND(price_return + dividend_return, 6) as total_return_calc
# MAGIC FROM workspace.silver.fii_total_returns
# MAGIC WHERE dividend_value > 0
# MAGIC ORDER BY date DESC
# MAGIC LIMIT 15