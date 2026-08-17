# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Inspecionar tabela Gold V1 atual
from pyspark.sql.functions import *

print("="*70)
print("INSPEÇÃO DA TABELA GOLD V1 ATUAL")
print("="*70)

# 1. Ler tabela atual
gold_atual = spark.table('workspace.gold.fii_features_v1')

print(f"\n1. Contagem total: {gold_atual.count():,} registros")

# 2. Schema completo
print("\n2. Schema completo:")
gold_atual.printSchema()

# 3. Colunas
colunas = gold_atual.columns
print(f"\n3. Total de colunas: {len(colunas)}")
print("\nColunas:")
for i, coluna in enumerate(colunas, 1):
    print(f"  {i:2}. {coluna}")

# 4. Estatísticas básicas
print("\n4. Estatísticas por data:")
gold_atual.agg(
    min('date').alias('min_data'),
    max('date').alias('max_data'),
    countDistinct('ticker').alias('tickers_unicos')
).show(truncate=False)

# 5. Amostra de dados
print("\n5. Amostra (3 registros mais recentes):")
gold_atual.orderBy(col('date').desc()).show(3, truncate=False, vertical=True)

# 6. Verificar features (excluindo ticker, date e targets)
features = [c for c in colunas if c not in ['ticker', 'date', 'target_7d', 'target_alpha_7d']]
print(f"\n6. Features identificadas: {len(features)}")
for i, feat in enumerate(features, 1):
    print(f"  {i:2}. {feat}")

# 7. Registros por ticker
print("\n7. Top 10 tickers por número de registros:")
gold_atual.groupBy('ticker').count() \
    .orderBy(col('count').desc()) \
    .show(10, truncate=False)

# COMMAND ----------

# DBTITLE 1,Criar backup da Gold V1 atual
from datetime import datetime

# Timestamp para o backup
backup_timestamp = "20260813"
backup_table = f"workspace.gold.fii_features_v1_backup_{backup_timestamp}"

print("="*70)
print("CRIANDO BACKUP DA GOLD V1")
print("="*70)

# Ler tabela atual
gold_atual = spark.table('workspace.gold.fii_features_v1')
count_atual = gold_atual.count()

print(f"\nTabela original: workspace.gold.fii_features_v1")
print(f"Registros: {count_atual:,}")

# Criar backup
gold_atual.write.mode("overwrite").saveAsTable(backup_table)

print(f"\n✓ Backup criado: {backup_table}")
print(f"  Registros: {count_atual:,}")
print("\n" + "="*70)

# COMMAND ----------

# DBTITLE 1,Buscar código original da Gold V1
# Buscar o notebook/arquivo que criou a Gold V1 original
import os

print("Buscando notebooks com 'gold' ou 'features' no nome...\n")

# Listar notebooks no diretório do projeto
for root, dirs, files in os.walk('/Workspace/Users/tropehe@outlook.com/projeto-fiis'):
    for file in files:
        if file.endswith('.py') or 'gold' in file.lower() or 'feature' in file.lower():
            print(f"  {os.path.join(root, file)}")

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

print("✓ Imports carregados")
print(f"📊 Spark version: {spark.version}")

# COMMAND ----------

# DBTITLE 1,2. Carregar Tabelas Silver
# Carregar tabelas Silver
df_fii = spark.table("workspace.silver.fii_total_returns")
df_ifix = spark.table("workspace.silver.ifix")
df_selic = spark.table("workspace.silver.selic")
df_dolar = spark.table("workspace.silver.cotacao_dolar")
df_ipca = spark.table("workspace.silver.ipca")
df_desemprego = spark.table("workspace.silver.desemprego")

print(f"✓ Dados carregados:")
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

print("✓ Retorno diário do IFIX calculado")

# COMMAND ----------

# DBTITLE 1,4. JOIN FII + IFIX
# JOIN FII com IFIX (LEFT JOIN por data)
df_gold = df_fii.join(
    df_ifix.select("date", col("close").alias("ifix_close"), "ifix_daily_return"),
    on="date",
    how="left"
)

print(f"✓ JOIN FII + IFIX: {df_gold.count():,} registros")

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

print("✓ Retornos históricos do FII calculados (1d, 7d, 30d, 90d)")

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

print("✓ Retornos históricos do IFIX calculados (1d, 7d, 30d, 90d)")

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

print("✓ Volatilidade calculada (30d, 90d)")

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

print("✓ Features de dividendos calculadas")

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

print("✓ Features de alpha calculadas (30d, 90d)")

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

print("✓ JOIN com macro diária (SELIC, Dólar)")

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

print("✓ JOIN com macro mensal (IPCA, Desemprego) - COM DEFASAGEM DE DIVULGAÇÃO")

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

print("✓ Targets calculados (target_7d, target_alpha_7d)")

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
print(f"\n✓ TOTAL FINAL: {count_final:,} registros válidos para ML")

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

print(f"📊 Preparando para gravar tabela:")
print(f"   - Total de registros: {df_gold_final.count():,}")
print(f"   - Total de colunas: {len(df_gold_final.columns)}")

# Gravar como tabela Delta particionada por ticker
df_gold_final.write \
    .mode("overwrite") \
    .partitionBy("ticker") \
    .saveAsTable("workspace.gold.fii_features_v1")

print("\n✓ Tabela workspace.gold.fii_features_v1 criada com sucesso!")

# COMMAND ----------

# DBTITLE 1,15. Validações Completas
# Carregar tabela recém-criada
df_new = spark.table("workspace.gold.fii_features_v1")

# Carregar backup para comparação
df_backup = spark.table("workspace.gold.fii_features_v1_backup_20260813")

print("="*70)
print("VALIDAÇÕES PÓS-RECONSTRUÇÃO")
print("="*70)

# 1. Quantidade total de registros
count_new = df_new.count()
count_backup = df_backup.count()
print(f"\n1️⃣ Contagem de Registros:")
print(f"   - Nova: {count_new:,}")
print(f"   - Anterior: {count_backup:,}")
print(f"   - Diferença: {count_new - count_backup:+,}")

# 2. Data mínima e máxima
print(f"\n2️⃣ Datas Mín/Máx:")
df_new.agg(
    _min('date').alias('min_data'),
    _max('date').alias('max_data')
).show(truncate=False)

# 3. Data máxima por ticker
print(f"\n3️⃣ Data Máxima por Ticker:")
df_new.groupBy('ticker').agg(
    _max('date').alias('max_data'),
    count('*').alias('registros')
).orderBy('ticker').show(truncate=False)

# 4. Número de registros por ticker
print(f"\n4️⃣ Registros por Ticker:")
df_new.groupBy('ticker').count().orderBy('ticker').show(truncate=False)

# 5. Duplicatas por ticker e date
print(f"\n5️⃣ Duplicatas por ticker + date:")
duplicatas = df_new.groupBy('ticker', 'date').count().filter(col('count') > 1).count()
print(f"   Total de duplicatas: {duplicatas}")

# 6. Nulos por coluna
print(f"\n6️⃣ Nulos por Coluna:")
for coluna in df_new.columns:
    nulos = df_new.filter(col(coluna).isNull()).count()
    if nulos > 0:
        print(f"   {coluna}: {nulos:,} nulos")
print("   (Apenas colunas com nulos são mostradas)")

# 7. Distribuição de target_7d
print(f"\n7️⃣ Distribuição de target_7d:")
df_new.groupBy('target_7d').count().orderBy('target_7d').show(truncate=False)

# 8. Presença das mesmas 23 features
print(f"\n8️⃣ Features (excluindo ticker, date, targets):")
features_new = [c for c in df_new.columns if c not in ['ticker', 'date', 'target_7d', 'target_alpha_7d']]
print(f"   Total: {len(features_new)} features")
for i, feat in enumerate(features_new, 1):
    print(f"   {i:2}. {feat}")

# 9. Confirmar que registros históricos até 14/02/2025 continuam iguais
print(f"\n9️⃣ Comparação Histórico até 2025-02-14:")
date_limite = "2025-02-14"
count_historico_new = df_new.filter(col('date') <= date_limite).count()
count_historico_backup = df_backup.filter(col('date') <= date_limite).count()
print(f"   Nova até {date_limite}: {count_historico_new:,}")
print(f"   Anterior até {date_limite}: {count_historico_backup:,}")
print(f"   Match: {'\u2713 SIM' if count_historico_new == count_historico_backup else '❌ NÃO'}")

# 10. Registros entre 01/03/2025 e 28/02/2026
print(f"\n🔟 Registros entre 01/03/2025 e 28/02/2026:")
count_novos = df_new.filter((col('date') >= "2025-03-01") & (col('date') <= "2026-02-28")).count()
print(f"   Total: {count_novos:,} registros")

# 11. Confirmar que target de 7 pregões está calculável no período final
print(f"\n🎯 Target 7d calculável no período final:")
max_date_with_target = df_new.filter(col('target_7d').isNotNull()).agg(_max('date')).collect()[0][0]
print(f"   Última data com target_7d: {max_date_with_target}")
from datetime import date as dt_date
print(f"   Target calculável até: {'✓ SIM' if max_date_with_target >= dt_date(2026, 2, 20) else '❌ NÃO'}")

print("\n" + "="*70)
print("✓ VALIDAÇÕES CONCLUÍDAS!")
print("="*70)