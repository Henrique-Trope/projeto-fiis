# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,📊 Gold Layer V1 - FII Features para Machine Learning
# MAGIC %md
# MAGIC # 📊 Gold Layer V1 - FII Features para Machine Learning
# MAGIC
# MAGIC ## 🎯 Objetivo
# MAGIC Criar dataset final para treinar modelos de ML que prevejam se um **FII superará o IFIX nos próximos 7 dias**.
# MAGIC
# MAGIC ## 🧠 Estratégia: Abordagem Incremental
# MAGIC - **Gold V1 (esta versão):** Features essenciais, ~28 colunas, foco em interpretabilidade
# MAGIC - **Gold V2 (futura):** Features técnicas avançadas (RSI, médias móveis, beta)
# MAGIC - **Gold V3 (futura):** Ajustes baseados em análise de importância de features
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📋 Schema da workspace.gold.fii_features_v1 (36 colunas)
# MAGIC
# MAGIC ### **1. IDENTIFICAÇÃO (5 colunas)**
# MAGIC - `ticker` (string) - Código do FII [PARTIÇÃO]
# MAGIC - `date` (date) - Data de referência
# MAGIC - `close` (double) - Preço de fechamento
# MAGIC - `volume` (int) - Volume negociado
# MAGIC - `has_trading` (boolean) - Flag de liquidez
# MAGIC
# MAGIC ### **2. RETORNOS HISTÓRICOS DO FII (4 colunas)**
# MAGIC - `return_1d` - Retorno total 1 dia atrás
# MAGIC - `return_7d` - Retorno acumulado últimos 7 dias
# MAGIC - `return_30d` - Retorno acumulado últimos 30 dias
# MAGIC - `return_90d` - Retorno acumulado últimos 90 dias
# MAGIC
# MAGIC ### **3. RETORNOS HISTÓRICOS DO IFIX (4 colunas)**
# MAGIC - `ifix_return_1d` - Retorno IFIX 1 dia atrás
# MAGIC - `ifix_return_7d` - Retorno IFIX últimos 7 dias
# MAGIC - `ifix_return_30d` - Retorno IFIX últimos 30 dias
# MAGIC - `ifix_return_90d` - Retorno IFIX últimos 90 dias
# MAGIC
# MAGIC ### **4. VOLATILIDADE (2 colunas)**
# MAGIC - `volatility_30d` - Desvio padrão retornos últimos 30 dias
# MAGIC - `volatility_90d` - Desvio padrão retornos últimos 90 dias
# MAGIC
# MAGIC ### **5. DIVIDENDOS (4 colunas)**
# MAGIC - `dividend_yield_12m` - Soma dividendos últimos 365 dias / preço atual
# MAGIC - `dividend_history_days` - Dias de histórico disponível (auditoria)
# MAGIC - `days_since_last_dividend` - Dias desde último ex-dividend
# MAGIC - `has_dividend_today` - Flag: dividendo hoje
# MAGIC
# MAGIC ### **6. ALPHA vs IFIX (2 colunas)**
# MAGIC - `alpha_30d` - return_30d - ifix_return_30d
# MAGIC - `alpha_90d` - return_90d - ifix_return_90d
# MAGIC
# MAGIC ### **7. VARIÁVEIS MACRO (4 colunas)**
# MAGIC - `selic` - Taxa SELIC diária (% ao ano)
# MAGIC - `ipca` - Variação IPCA mensal (%)
# MAGIC - `dolar` - Cotação USD/BRL
# MAGIC - `desemprego` - Taxa desemprego mensal (%)
# MAGIC
# MAGIC ### **8. TARGETS (2 colunas)**
# MAGIC - `target_7d` - FII superou IFIX nos próximos 7 dias?
# MAGIC - `target_alpha_7d` - Diferença de retorno nos próximos 7 dias
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ⚠️ Regras Anti-Data-Leakage
# MAGIC
# MAGIC ✅ **Features:** Usam APENAS informação até `date-1` (inclusive)
# MAGIC - Window functions: `ROWS/RANGE BETWEEN N PRECEDING AND 1 PRECEDING`
# MAGIC
# MAGIC ✅ **Targets:** Usam APENAS informação de `date+1` até `date+7` (inclusive)
# MAGIC - Window functions: `ROWS BETWEEN 1 FOLLOWING AND 7 FOLLOWING`
# MAGIC
# MAGIC ✅ **Macro Mensal (IPCA, Desemprego):**
# MAGIC - Usam defasagem de divulgação (~15 dias para IPCA, ~30 dias para desemprego)
# MAGIC - Se `DAY(date) < 15`: usar valor de 2 meses atrás
# MAGIC - Se `DAY(date) >= 15`: usar valor de 1 mês atrás
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Estimativa de Registros
# MAGIC
# MAGIC **Base:** 11.788 registros (silver.fii_total_returns)
# MAGIC
# MAGIC **Filtros:**
# MAGIC - Primeiras 90 linhas/ticker: -450
# MAGIC - Últimas 7 linhas/ticker: -35
# MAGIC - NULLs de macro: ~-50
# MAGIC - NULLs de IFIX: ~-10
# MAGIC
# MAGIC **Total esperado:** ~**11.243 registros válidos**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Validações Pós-Criação
# MAGIC 1. Contagem de registros
# MAGIC 2. Zero NULLs em features e targets
# MAGIC 3. Distribuição balanceada de `target_7d` (~50% True/False)
# MAGIC 4. Alpha médio próximo de 0
# MAGIC 5. Teste manual de data leakage

# COMMAND ----------

# DBTITLE 1,1. Imports e Setup
# Imports
from pyspark.sql import Window
from pyspark.sql.functions import (
    col, lag, lead, sum as _sum, stddev, exp, log1p, expm1,
    unix_timestamp, datediff, date_sub, add_months, date_trunc,
    year, month, dayofmonth, when, lit, coalesce, last_value,
    count, max as _max, min as _min
)

print("✅ Imports carregados")
print(f"📊 Spark version: {spark.version}")

# COMMAND ----------

# DBTITLE 1,2. Carregar Dados Base
# Carregar tabelas Silver
df_fii = spark.table("workspace.silver.fii_total_returns")
df_ifix = spark.table("workspace.silver.ifix")
df_selic = spark.table("workspace.silver.selic")
df_dolar = spark.table("workspace.silver.cotacao_dolar")
df_ipca = spark.table("workspace.silver.ipca")
df_desemprego = spark.table("workspace.silver.desemprego")

print(f"✅ Dados carregados:")
print(f"   - FII Total Returns: {df_fii.count():,} registros")
print(f"   - IFIX: {df_ifix.count():,} registros")
print(f"   - SELIC: {df_selic.count():,} registros")
print(f"   - Dólar: {df_dolar.count():,} registros")
print(f"   - IPCA: {df_ipca.count():,} registros")
print(f"   - Desemprego: {df_desemprego.count():,} registros")

# COMMAND ----------

# DBTITLE 1,3. Calcular Retorno Diário do IFIX
# Calcular retorno diário do IFIX para depois acumular em janelas
window_ifix = Window.orderBy("date")

df_ifix = df_ifix.withColumn(
    "ifix_close_prev",
    lag(col("close"), 1).over(window_ifix)
).withColumn(
    "ifix_daily_return",
    when(col("ifix_close_prev").isNotNull(),
         (col("close") - col("ifix_close_prev")) / col("ifix_close_prev")
    ).otherwise(lit(None))
)

print("✅ Retorno diário do IFIX calculado")
display(df_ifix.select("date", "close", "ifix_close_prev", "ifix_daily_return").limit(10))

# COMMAND ----------

# DBTITLE 1,4. JOIN FII + IFIX
# JOIN FII com IFIX (LEFT JOIN por data)
df_gold = df_fii.join(
    df_ifix.select("date", col("close").alias("ifix_close"), "ifix_daily_return"),
    on="date",
    how="left"
)

print(f"✅ JOIN FII + IFIX: {df_gold.count():,} registros")

# COMMAND ----------

# DBTITLE 1,5. Calcular Retornos Históricos do FII
# Window por ticker ordenado por date
window_ticker = Window.partitionBy("ticker").orderBy("date")

# Retornos lookback do FII (usando ROWS BETWEEN N PRECEDING AND 1 PRECEDING)
df_gold = df_gold.withColumn(
    "return_1d",
    lag(col("total_return"), 1).over(window_ticker)
).withColumn(
    "return_7d",
    expm1(_sum(log1p(col("total_return"))).over(
        window_ticker.rowsBetween(-7, -1)
    ))
).withColumn(
    "return_30d",
    expm1(_sum(log1p(col("total_return"))).over(
        window_ticker.rowsBetween(-30, -1)
    ))
).withColumn(
    "return_90d",
    expm1(_sum(log1p(col("total_return"))).over(
        window_ticker.rowsBetween(-90, -1)
    ))
)

print("✅ Retornos históricos do FII calculados (1d, 7d, 30d, 90d)")

# COMMAND ----------

# DBTITLE 1,6. Calcular Retornos Históricos do IFIX
# Retornos lookback do IFIX (mesma lógica, mas sem partition por ticker)
window_date = Window.orderBy("date")

df_gold = df_gold.withColumn(
    "ifix_return_1d",
    lag(col("ifix_daily_return"), 1).over(window_date)
).withColumn(
    "ifix_return_7d",
    expm1(_sum(log1p(col("ifix_daily_return"))).over(
        window_date.rowsBetween(-7, -1)
    ))
).withColumn(
    "ifix_return_30d",
    expm1(_sum(log1p(col("ifix_daily_return"))).over(
        window_date.rowsBetween(-30, -1)
    ))
).withColumn(
    "ifix_return_90d",
    expm1(_sum(log1p(col("ifix_daily_return"))).over(
        window_date.rowsBetween(-90, -1)
    ))
)

print("✅ Retornos históricos do IFIX calculados (1d, 7d, 30d, 90d)")

# COMMAND ----------

# DBTITLE 1,7. Calcular Volatilidade
# Volatilidade (desvio padrão dos retornos)
df_gold = df_gold.withColumn(
    "volatility_30d",
    stddev(col("total_return")).over(
        window_ticker.rowsBetween(-30, -1)
    )
).withColumn(
    "volatility_90d",
    stddev(col("total_return")).over(
        window_ticker.rowsBetween(-90, -1)
    )
)

print("✅ Volatilidade calculada (30d, 90d)")

# COMMAND ----------

# DBTITLE 1,8. Calcular Features de Dividendos
# Converter date para timestamp unix para usar RANGE BETWEEN
df_gold = df_gold.withColumn("ts", unix_timestamp(col("date")))

# Window com RANGE BETWEEN para 365 dias (31536000 segundos)
window_365d = Window.partitionBy("ticker").orderBy("ts").rangeBetween(-31536000, -1)

# Dividend yield: soma dividendos últimos 365 dias / close atual
df_gold = df_gold.withColumn(
    "dividend_sum_365d",
    _sum(col("dividend_value")).over(window_365d)
).withColumn(
    "dividend_yield_12m",
    col("dividend_sum_365d") / col("close")
)

# Dias de histórico disponível (para auditoria)
df_gold = df_gold.withColumn(
    "first_date_ticker",
    _min(col("date")).over(Window.partitionBy("ticker"))
).withColumn(
    "dividend_history_days",
    datediff(col("date"), col("first_date_ticker"))
)

# Dias desde último dividendo
window_ticker_unbounded = Window.partitionBy("ticker").orderBy("date").rowsBetween(Window.unboundedPreceding, -1)
df_gold = df_gold.withColumn(
    "last_dividend_date",
    when(col("dividend_value") > 0, col("date"))
).withColumn(
    "last_dividend_date_seen",
    _max(col("last_dividend_date")).over(window_ticker_unbounded)
).withColumn(
    "days_since_last_dividend",
    when(col("last_dividend_date_seen").isNotNull(),
         datediff(col("date"), col("last_dividend_date_seen"))
    ).otherwise(lit(None))
)

# Flag: dividendo hoje
df_gold = df_gold.withColumn(
    "has_dividend_today",
    col("dividend_value") > 0
)

# Remover colunas auxiliares
df_gold = df_gold.drop("ts", "dividend_sum_365d", "first_date_ticker", "last_dividend_date", "last_dividend_date_seen")

print("✅ Features de dividendos calculadas")
print("   - dividend_yield_12m (usando RANGE BETWEEN 365 dias)")
print("   - dividend_history_days")
print("   - days_since_last_dividend")
print("   - has_dividend_today")

# COMMAND ----------

# DBTITLE 1,9. Calcular Alpha
# Alpha: retorno FII - retorno IFIX
df_gold = df_gold.withColumn(
    "alpha_30d",
    col("return_30d") - col("ifix_return_30d")
).withColumn(
    "alpha_90d",
    col("return_90d") - col("ifix_return_90d")
)

print("✅ Features de alpha calculadas (30d, 90d)")

# COMMAND ----------

# DBTITLE 1,10. JOIN com Macro Diária (SELIC, Dólar)
# JOIN com SELIC (diária, direto por data)
df_gold = df_gold.join(
    df_selic.select(col("data_completa").alias("date_selic"), col("taxa_diaria").alias("selic")),
    df_gold.date == col("date_selic"),
    how="left"
).drop("date_selic")

# JOIN com Dólar (diária, direto por data)
df_gold = df_gold.join(
    df_dolar.select(col("data_completa").alias("date_dolar"), col("cotacao").alias("dolar")),
    df_gold.date == col("date_dolar"),
    how="left"
).drop("date_dolar")

print("✅ JOIN com macro diária (SELIC, Dólar)")

# COMMAND ----------

# DBTITLE 1,11. JOIN com Macro Mensal COM DEFASAGEM
# Criar colunas de referência com defasagem de divulgação
# IPCA: se dia < 15, usar 2 meses atrás; se dia >= 15, usar 1 mês atrás
df_gold = df_gold.withColumn(
    "ipca_reference_month",
    when(dayofmonth(col("date")) < 15,
         date_trunc("month", add_months(col("date"), -2))
    ).otherwise(
         date_trunc("month", add_months(col("date"), -1))
    )
)

# Desemprego: sempre usar 2 meses atrás (conservador)
df_gold = df_gold.withColumn(
    "desemprego_reference_month",
    date_trunc("month", add_months(col("date"), -2))
)

# JOIN com IPCA
df_gold = df_gold.join(
    df_ipca.select(col("data_completa").alias("ipca_ref"), col("variacao_percentual").alias("ipca")),
    df_gold.ipca_reference_month == col("ipca_ref"),
    how="left"
).drop("ipca_ref", "ipca_reference_month")

# JOIN com Desemprego
df_gold = df_gold.join(
    df_desemprego.select(col("data_completa").alias("desemp_ref"), col("taxa_percentual").alias("desemprego")),
    df_gold.desemprego_reference_month == col("desemp_ref"),
    how="left"
).drop("desemp_ref", "desemprego_reference_month")

print("✅ JOIN com macro mensal (IPCA, Desemprego) - COM DEFASAGEM DE DIVULGAÇÃO")

# COMMAND ----------

# DBTITLE 1,12. Calcular Targets (Forward-Looking)
# Targets: retornos futuros (próximos 7 dias)
# Window para futuro (FOLLOWING)
window_future_fii = Window.partitionBy("ticker").orderBy("date").rowsBetween(1, 7)
window_future_ifix = Window.orderBy("date").rowsBetween(1, 7)

# Retorno futuro FII (próximos 7 dias)
df_gold = df_gold.withColumn(
    "future_return_fii_7d",
    expm1(_sum(log1p(col("total_return"))).over(window_future_fii))
)

# Retorno futuro IFIX (próximos 7 dias)
df_gold = df_gold.withColumn(
    "future_return_ifix_7d",
    expm1(_sum(log1p(col("ifix_daily_return"))).over(window_future_ifix))
)

# Target binário: FII superou IFIX?
df_gold = df_gold.withColumn(
    "target_7d",
    col("future_return_fii_7d") > col("future_return_ifix_7d")
)

# Target contínuo: alpha futuro
df_gold = df_gold.withColumn(
    "target_alpha_7d",
    col("future_return_fii_7d") - col("future_return_ifix_7d")
)

# Remover colunas auxiliares de retorno futuro
df_gold = df_gold.drop("future_return_fii_7d", "future_return_ifix_7d")

print("✅ Targets calculados (target_7d, target_alpha_7d)")

# COMMAND ----------

# DBTITLE 1,13. Aplicar Filtros (Remover NULLs)
# Contar registros antes dos filtros
count_antes = df_gold.count()
print(f"📊 Registros ANTES dos filtros: {count_antes:,}")

# Filtro 1: Remover primeiras 90 linhas por ticker (return_90d NULL)
df_gold = df_gold.filter(col("return_90d").isNotNull())
count_apos_return = df_gold.count()
print(f"   Após filtrar return_90d NULL: {count_apos_return:,} ({count_antes - count_apos_return:,} removidos)")

# Filtro 2: Remover últimas 7 linhas por ticker (target_7d NULL)
df_gold = df_gold.filter(col("target_7d").isNotNull())
count_apos_target = df_gold.count()
print(f"   Após filtrar target_7d NULL: {count_apos_target:,} ({count_apos_return - count_apos_target:,} removidos)")

# Filtro 3: Remover NULLs de IFIX (crítico)
df_gold = df_gold.filter(col("ifix_daily_return").isNotNull())
count_apos_ifix = df_gold.count()
print(f"   Após filtrar IFIX NULL: {count_apos_ifix:,} ({count_apos_target - count_apos_ifix:,} removidos)")

# Filtro 4: Forward-fill macro diária (SELIC, Dólar) - usar último valor conhecido
window_date_unbounded = Window.orderBy("date").rowsBetween(Window.unboundedPreceding, 0)
df_gold = df_gold.withColumn(
    "selic",
    last_value(col("selic"), ignoreNulls=True).over(window_date_unbounded)
).withColumn(
    "dolar",
    last_value(col("dolar"), ignoreNulls=True).over(window_date_unbounded)
)

# Filtro 5: Verificar NULLs de macro mensal (IPCA, Desemprego)
null_ipca = df_gold.filter(col("ipca").isNull()).count()
null_desemprego = df_gold.filter(col("desemprego").isNull()).count()
print(f"   NULLs remanescentes - IPCA: {null_ipca}, Desemprego: {null_desemprego}")

if null_ipca > 0 or null_desemprego > 0:
    print("   ⚠️ Removendo registros com NULLs em IPCA/Desemprego (estratégia conservadora)")
    df_gold = df_gold.filter(col("ipca").isNotNull() & col("desemprego").isNotNull())
    count_final = df_gold.count()
    print(f"   Após filtrar macro mensal NULL: {count_final:,} ({count_apos_ifix - count_final:,} removidos)")

count_final = df_gold.count()
print(f"\n✅ TOTAL FINAL: {count_final:,} registros válidos para ML")

# COMMAND ----------

# DBTITLE 1,14. Selecionar Colunas Finais e Gravar Tabela
# Selecionar apenas as colunas finais do schema
df_gold_final = df_gold.select(
    # Identificação
    "ticker", "date", "close", "volume", "has_trading",
    # Retornos FII
    "return_1d", "return_7d", "return_30d", "return_90d",
    # Retornos IFIX
    "ifix_return_1d", "ifix_return_7d", "ifix_return_30d", "ifix_return_90d",
    # Volatilidade
    "volatility_30d", "volatility_90d",
    # Dividendos
    "dividend_yield_12m", "dividend_history_days", "days_since_last_dividend", "has_dividend_today",
    # Alpha
    "alpha_30d", "alpha_90d",
    # Macro
    "selic", "ipca", "dolar", "desemprego",
    # Targets
    "target_7d", "target_alpha_7d"
)

# Gravar como tabela Delta particionada por ticker
df_gold_final.write \
    .mode("overwrite") \
    .partitionBy("ticker") \
    .saveAsTable("workspace.gold.fii_features_v1")

print("✅ Tabela workspace.gold.fii_features_v1 criada com sucesso!")
print(f"   - Total de registros: {df_gold_final.count():,}")
print(f"   - Particionada por: ticker")
print(f"   - Total de colunas: {len(df_gold_final.columns)}")

# COMMAND ----------

# DBTITLE 1,15. Validações Pós-Criação
# Carregar tabela criada para validações
df_val = spark.table("workspace.gold.fii_features_v1")

print("🔍 VALIDAÇÕES PÓS-CRIAÇÃO\n")

# 1. Contagem de registros
print(f"1️⃣ Contagem: {df_val.count():,} registros")

# 2. Verificar NULLs em features críticas
nulls_return90 = df_val.filter(col("return_90d").isNull()).count()
nulls_vol90 = df_val.filter(col("volatility_90d").isNull()).count()
nulls_selic = df_val.filter(col("selic").isNull()).count()
print(f"\n2️⃣ NULLs em Features:")
print(f"   - return_90d: {nulls_return90}")
print(f"   - volatility_90d: {nulls_vol90}")
print(f"   - selic: {nulls_selic}")

# 3. Verificar NULLs em targets
nulls_target = df_val.filter(col("target_7d").isNull()).count()
print(f"\n3️⃣ NULLs em Target: {nulls_target}")

# 4. Distribuição de target_7d
print(f"\n4️⃣ Distribuição de target_7d:")
target_dist = df_val.groupBy("target_7d").count().orderBy("target_7d")
display(target_dist)

# 5. Sanity check: Alpha médio
print(f"\n5️⃣ Sanity Check - Alpha Médio:")
alpha_stats = df_val.selectExpr(
    "AVG(alpha_30d) as avg_alpha_30d",
    "AVG(alpha_90d) as avg_alpha_90d",
    "AVG(target_alpha_7d) as avg_target_alpha_7d"
)
display(alpha_stats)

print("\n✅ Validações concluídas!")