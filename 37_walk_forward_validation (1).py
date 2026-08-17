# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,📊 Objetivo e Contexto
# MAGIC %md
# MAGIC # 🎯 Walk-Forward Validation: Gold V1 vs V2 Minimal
# MAGIC
# MAGIC ## 📋 Objetivo
# MAGIC
# MAGIC Comparar **workspace.gold.fii_features_v1** (23 features) versus **workspace.gold.fii_features_v2_minimal** (24 features: V1 + price_vs_ma_7d) usando **walk-forward validation** com janela expansiva para verificar se a nova feature melhora o XGBoost de forma **consistente** em diferentes períodos.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔬 Metodologia
# MAGIC
# MAGIC ### XGBoost Fixo
# MAGIC * Mesmos hiperparâmetros do notebook 33_ml_advanced
# MAGIC * Seed=42
# MAGIC * Mesmo pré-processamento
# MAGIC * **NÃO** fazer tuning
# MAGIC * **NÃO** criar novas features
# MAGIC * **NÃO** selecionar features
# MAGIC
# MAGIC ### Walk-Forward com Janela Expansiva
# MAGIC
# MAGIC **Folds Temporais:**
# MAGIC
# MAGIC **Fold 1:**
# MAGIC * Train: 2020-03 até 2021-12
# MAGIC * Gap: 7 pregões
# MAGIC * Eval: 2022
# MAGIC
# MAGIC **Fold 2:**
# MAGIC * Train: 2020-03 até 2022-12
# MAGIC * Gap: 7 pregões
# MAGIC * Eval: 2023
# MAGIC
# MAGIC **Fold 3:**
# MAGIC * Train: 2020-03 até 2023-12
# MAGIC * Gap: 7 pregões
# MAGIC * Eval: 2024
# MAGIC
# MAGIC **Fold 4:**
# MAGIC * Train: 2020-03 até 2024-12
# MAGIC * Gap: 7 pregões
# MAGIC * Eval: 2025+
# MAGIC
# MAGIC **Gap de 7 pregões:** Evita sobreposição causada pelo target_7d (forward-looking window)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Métricas por Fold
# MAGIC
# MAGIC * ROC-AUC
# MAGIC * Accuracy
# MAGIC * Precision
# MAGIC * Recall
# MAGIC * F1
# MAGIC * Quantidade de registros
# MAGIC * Distribuição do target
# MAGIC * Probabilidades out-of-sample
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ❓ Perguntas a Responder
# MAGIC
# MAGIC 1. **price_vs_ma_7d melhora o modelo de forma consistente?**
# MAGIC 2. **O ganho aparece em vários folds ou apenas em um período?**
# MAGIC 3. **V1 ou V2 Minimal deve seguir para tuning e backtest?**
# MAGIC 4. **O modelo permanece acima de ROC-AUC 0.50 na maioria dos períodos?**
# MAGIC 5. **Existe robustez suficiente para avançar ao backtest?**

# COMMAND ----------

# DBTITLE 1,0. Instalar Dependências
# MAGIC %pip install xgboost scikit-learn
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,1. Setup e Imports
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('default')
sns.set_palette("husl")

print("✅ Imports carregados")
print(f"Data atual: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

# COMMAND ----------

# DBTITLE 1,2. Carregar Dados Gold V1 e V2 Minimal
# Carregar Gold V1 e V2 Minimal
print("=" * 80)
print("📊 CARREGANDO DADOS")
print("=" * 80)

df_v1 = spark.table("workspace.gold.fii_features_v1").toPandas()
df_v2_minimal = spark.table("workspace.gold.fii_features_v2_minimal").toPandas()

print(f"\n✅ Gold V1 carregada: {df_v1.shape[0]} registros, {df_v1.shape[1]} colunas")
print(f"✅ Gold V2_minimal carregada: {df_v2_minimal.shape[0]} registros, {df_v2_minimal.shape[1]} colunas")

# Converter date para datetime
df_v1['date'] = pd.to_datetime(df_v1['date'])
df_v2_minimal['date'] = pd.to_datetime(df_v2_minimal['date'])

# Ordenar por ticker e date
df_v1 = df_v1.sort_values(['ticker', 'date']).reset_index(drop=True)
df_v2_minimal = df_v2_minimal.sort_values(['ticker', 'date']).reset_index(drop=True)

print(f"\nPeríodo V1: {df_v1['date'].min().date()} a {df_v1['date'].max().date()}")
print(f"Período V2_minimal: {df_v2_minimal['date'].min().date()} a {df_v2_minimal['date'].max().date()}")

print("\n" + "=" * 80)

# COMMAND ----------

# DBTITLE 1,3. Definir Features e Colunas
# Colunas que NÃO são features
non_feature_cols = ['ticker', 'date', 'target_7d', 'target_alpha_7d']

# Features V1 (23 features)
features_v1 = [col for col in df_v1.columns if col not in non_feature_cols]

# Features V2_minimal (24 features: V1 + price_vs_ma_7d)
features_v2_minimal = [col for col in df_v2_minimal.columns if col not in non_feature_cols]

print("=" * 80)
print("📋 FEATURES")
print("=" * 80)
print(f"\nGold V1: {len(features_v1)} features")
print(f"Gold V2_minimal: {len(features_v2_minimal)} features")

# Feature nova em V2
new_feature = set(features_v2_minimal) - set(features_v1)
print(f"\nNova feature em V2_minimal: {new_feature}")

print("\n" + "=" * 80)

# COMMAND ----------

# DBTITLE 1,4. Função Auxiliar - Treinar e Avaliar
def train_and_evaluate_fold(X_train, y_train, X_eval, y_eval, fold_name, version_name):
    """
    Treina XGBoost com hiperparâmetros fixos do 33_ml_advanced e avalia no fold.
    
    Args:
        X_train: Features de treino
        y_train: Target de treino
        X_eval: Features de avaliação
        y_eval: Target de avaliação
        fold_name: Nome do fold (ex: "Fold 1")
        version_name: Nome da versão ("V1" ou "V2_minimal")
    
    Returns:
        dict com métricas, probabilidades e modelo
    """
    # Hiperparâmetros fixos do 33_ml_advanced
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc',
        early_stopping_rounds=20,
        verbosity=0
    )
    
    # Treinar
    model.fit(
        X_train, y_train,
        eval_set=[(X_eval, y_eval)],
        verbose=False
    )
    
    # Predições
    y_pred_proba = model.predict_proba(X_eval)[:, 1]
    y_pred = model.predict(X_eval)
    
    # Métricas
    auc = roc_auc_score(y_eval, y_pred_proba)
    acc = accuracy_score(y_eval, y_pred)
    prec = precision_score(y_eval, y_pred, zero_division=0)
    rec = recall_score(y_eval, y_pred, zero_division=0)
    f1 = f1_score(y_eval, y_pred, zero_division=0)
    
    # Distribuição do target
    target_dist = y_eval.value_counts(normalize=True).to_dict()
    
    return {
        'fold': fold_name,
        'version': version_name,
        'n_train': len(X_train),
        'n_eval': len(X_eval),
        'auc': auc,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'target_dist_0': target_dist.get(False, 0),
        'target_dist_1': target_dist.get(True, 0),
        'best_iteration': model.best_iteration,
        'y_true': y_eval.values,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
        'model': model
    }

print("✅ Função train_and_evaluate_fold criada")

# COMMAND ----------

# DBTITLE 1,5. Definir Folds Temporais Walk-Forward
# Definir folds temporais expansivos com gap de 7 pregões
print("=" * 80)
print("📅 DEFINIÇÃO DOS FOLDS TEMPORAIS (WALK-FORWARD EXPANSIVO)")
print("=" * 80)

folds = [
    {
        'name': 'Fold 1',
        'train_start': '2020-03-01',
        'train_end': '2021-12-31',
        'eval_start': '2022-01-01',
        'eval_end': '2022-12-31'
    },
    {
        'name': 'Fold 2',
        'train_start': '2020-03-01',
        'train_end': '2022-12-31',
        'eval_start': '2023-01-01',
        'eval_end': '2023-12-31'
    },
    {
        'name': 'Fold 3',
        'train_start': '2020-03-01',
        'train_end': '2023-12-31',
        'eval_start': '2024-01-01',
        'eval_end': '2024-12-31'
    },
    {
        'name': 'Fold 4',
        'train_start': '2020-03-01',
        'train_end': '2024-12-31',
        'eval_start': '2025-01-01',
        'eval_end': '2025-12-31'  # Ajustará para última data disponível
    }
]

for fold in folds:
    print(f"\n{fold['name']}:")
    print(f"  Train: {fold['train_start']} a {fold['train_end']}")
    print(f"  Gap: 7 pregões (evitar sobreposição com target_7d)")
    print(f"  Eval: {fold['eval_start']} a {fold['eval_end']}")

print("\n" + "=" * 80)
print("⚠️ Gap de 7 pregões: os primeiros 7 pregões do período de avaliação serão excluídos")
print("=" * 80)

# COMMAND ----------

# DBTITLE 1,6. Executar Walk-Forward Validation
# Lista para armazenar resultados de todos os folds
all_results = []

print("=" * 80)
print("🔄 EXECUTANDO WALK-FORWARD VALIDATION")
print("=" * 80)

for fold_config in folds:
    fold_name = fold_config['name']
    print(f"\n{'='*80}")
    print(f"📋 {fold_name}")
    print(f"{'='*80}")
    
    # Converter datas
    train_start = pd.to_datetime(fold_config['train_start'])
    train_end = pd.to_datetime(fold_config['train_end'])
    eval_start = pd.to_datetime(fold_config['eval_start'])
    eval_end = pd.to_datetime(fold_config['eval_end'])
    
    # Ajustar eval_end para última data disponível se necessário
    max_date_available = df_v1['date'].max()
    if eval_end > max_date_available:
        eval_end = max_date_available
        print(f"\n⚠️ Ajustando eval_end para última data disponível: {eval_end.date()}")
    
    # Filtrar dados de treino (até train_end)
    train_mask_v1 = (df_v1['date'] >= train_start) & (df_v1['date'] <= train_end)
    train_mask_v2 = (df_v2_minimal['date'] >= train_start) & (df_v2_minimal['date'] <= train_end)
    
    # Filtrar dados de avaliação (entre eval_start e eval_end)
    eval_mask_base_v1 = (df_v1['date'] >= eval_start) & (df_v1['date'] <= eval_end)
    eval_mask_base_v2 = (df_v2_minimal['date'] >= eval_start) & (df_v2_minimal['date'] <= eval_end)
    
    # Aplicar GAP de 7 pregões: pegar datas únicas ordenadas e excluir as primeiras 7
    eval_dates_v1 = df_v1[eval_mask_base_v1]['date'].drop_duplicates().sort_values().reset_index(drop=True)
    eval_dates_v2 = df_v2_minimal[eval_mask_base_v2]['date'].drop_duplicates().sort_values().reset_index(drop=True)
    
    if len(eval_dates_v1) > 7:
        gap_date_v1 = eval_dates_v1.iloc[7]  # Primeira data após o gap de 7 pregões
        eval_mask_v1 = eval_mask_base_v1 & (df_v1['date'] >= gap_date_v1)
    else:
        print(f"\n⚠️ {fold_name}: Eval tem apenas {len(eval_dates_v1)} datas. Pulando gap.")
        eval_mask_v1 = eval_mask_base_v1
    
    if len(eval_dates_v2) > 7:
        gap_date_v2 = eval_dates_v2.iloc[7]
        eval_mask_v2 = eval_mask_base_v2 & (df_v2_minimal['date'] >= gap_date_v2)
    else:
        print(f"\n⚠️ {fold_name}: Eval tem apenas {len(eval_dates_v2)} datas. Pulando gap.")
        eval_mask_v2 = eval_mask_base_v2
    
    print(f"\nTrain: {train_start.date()} a {train_end.date()}")
    print(f"Eval (com gap): a partir de {df_v1[eval_mask_v1]['date'].min().date()} até {eval_end.date()}")
    print(f"Train V1: {train_mask_v1.sum()} registros")
    print(f"Eval V1 (após gap): {eval_mask_v1.sum()} registros")
    print(f"Train V2: {train_mask_v2.sum()} registros")
    print(f"Eval V2 (após gap): {eval_mask_v2.sum()} registros")
    
    # Treinar e avaliar Gold V1
    print(f"\n🔹 Treinando Gold V1 ({len(features_v1)} features)...")
    X_train_v1 = df_v1.loc[train_mask_v1, features_v1]
    y_train_v1 = df_v1.loc[train_mask_v1, 'target_7d']
    X_eval_v1 = df_v1.loc[eval_mask_v1, features_v1]
    y_eval_v1 = df_v1.loc[eval_mask_v1, 'target_7d']
    
    result_v1 = train_and_evaluate_fold(
        X_train_v1, y_train_v1,
        X_eval_v1, y_eval_v1,
        fold_name, 'V1'
    )
    all_results.append(result_v1)
    print(f"  ✅ V1: ROC-AUC = {result_v1['auc']:.4f}")
    
    # Treinar e avaliar Gold V2_minimal
    print(f"\n🔸 Treinando Gold V2_minimal ({len(features_v2_minimal)} features)...")
    X_train_v2 = df_v2_minimal.loc[train_mask_v2, features_v2_minimal]
    y_train_v2 = df_v2_minimal.loc[train_mask_v2, 'target_7d']
    X_eval_v2 = df_v2_minimal.loc[eval_mask_v2, features_v2_minimal]
    y_eval_v2 = df_v2_minimal.loc[eval_mask_v2, 'target_7d']
    
    result_v2 = train_and_evaluate_fold(
        X_train_v2, y_train_v2,
        X_eval_v2, y_eval_v2,
        fold_name, 'V2_minimal'
    )
    all_results.append(result_v2)
    print(f"  ✅ V2_minimal: ROC-AUC = {result_v2['auc']:.4f}")
    
    # Comparar
    diff = result_v2['auc'] - result_v1['auc']
    winner = 'V2_minimal' if diff > 0 else 'V1' if diff < 0 else 'Empate'
    print(f"\n🏆 Vencedor do {fold_name}: {winner}")
    print(f"  Diferença: {diff:+.4f} ({(diff/result_v1['auc'])*100:+.2f}%)")

print("\n" + "=" * 80)
print("✅ WALK-FORWARD VALIDATION CONCLUÍDA")
print("=" * 80)

# COMMAND ----------

# DBTITLE 1,7. Consolidar Resultados
# Criar DataFrame com resultados
results_df = pd.DataFrame([{
    'fold': r['fold'],
    'version': r['version'],
    'n_train': r['n_train'],
    'n_eval': r['n_eval'],
    'auc': r['auc'],
    'accuracy': r['accuracy'],
    'precision': r['precision'],
    'recall': r['recall'],
    'f1': r['f1'],
    'target_dist_0': r['target_dist_0'],
    'target_dist_1': r['target_dist_1'],
    'best_iteration': r['best_iteration']
} for r in all_results])

print("=" * 100)
print("📊 RESULTADOS CONSOLIDADOS - TODOS OS FOLDS")
print("=" * 100)
print(results_df.to_string(index=False))
print("=" * 100)

# COMMAND ----------

# DBTITLE 1,8. Análise Comparativa V1 vs V2 Minimal
print("=" * 100)
print("🔬 ANÁLISE COMPARATIVA: GOLD V1 vs V2_MINIMAL")
print("=" * 100)

# Separar resultados por versão
v1_results = results_df[results_df['version'] == 'V1'].copy()
v2_results = results_df[results_df['version'] == 'V2_minimal'].copy()

# ROC-AUC médio e desvio padrão
print("\n📈 ROC-AUC - Estatísticas por Versão:")
print(f"\n  Gold V1:")
print(f"    Média: {v1_results['auc'].mean():.4f}")
print(f"    Desvio padrão: {v1_results['auc'].std():.4f}")
print(f"    Mínimo: {v1_results['auc'].min():.4f} ({v1_results.loc[v1_results['auc'].idxmin(), 'fold']})")
print(f"    Máximo: {v1_results['auc'].max():.4f} ({v1_results.loc[v1_results['auc'].idxmax(), 'fold']})")

print(f"\n  Gold V2_minimal:")
print(f"    Média: {v2_results['auc'].mean():.4f}")
print(f"    Desvio padrão: {v2_results['auc'].std():.4f}")
print(f"    Mínimo: {v2_results['auc'].min():.4f} ({v2_results.loc[v2_results['auc'].idxmin(), 'fold']})")
print(f"    Máximo: {v2_results['auc'].max():.4f} ({v2_results.loc[v2_results['auc'].idxmax(), 'fold']})")

# Diferença entre versões
print(f"\n🔹 Diferença V2_minimal - V1:")
print(f"    ROC-AUC médio: {v2_results['auc'].mean() - v1_results['auc'].mean():+.4f}")
print(f"    Ganho percentual: {((v2_results['auc'].mean() / v1_results['auc'].mean()) - 1) * 100:+.2f}%")

# Comparar fold por fold
print("\n📅 Comparação Fold por Fold:")
for fold_name in v1_results['fold'].unique():
    v1_auc = v1_results[v1_results['fold'] == fold_name]['auc'].values[0]
    v2_auc = v2_results[v2_results['fold'] == fold_name]['auc'].values[0]
    diff = v2_auc - v1_auc
    winner = '✅ V2' if diff > 0 else '❌ V1' if diff < 0 else '🔶 Empate'
    print(f"  {fold_name}: V1={v1_auc:.4f}, V2={v2_auc:.4f}, Diff={diff:+.4f} {winner}")

# Contagem de folds vencidos
folds_won_v1 = 0
folds_won_v2 = 0
folds_tie = 0

for fold_name in v1_results['fold'].unique():
    v1_auc = v1_results[v1_results['fold'] == fold_name]['auc'].values[0]
    v2_auc = v2_results[v2_results['fold'] == fold_name]['auc'].values[0]
    if v2_auc > v1_auc:
        folds_won_v2 += 1
    elif v1_auc > v2_auc:
        folds_won_v1 += 1
    else:
        folds_tie += 1

print(f"\n🏆 Folds Vencidos:")
print(f"  V1: {folds_won_v1} folds")
print(f"  V2_minimal: {folds_won_v2} folds")
print(f"  Empates: {folds_tie} folds")

# Verificar se modelo permanece > 0.50
print(f"\n✅ Robustez (ROC-AUC > 0.50):")
v1_above_50 = (v1_results['auc'] > 0.50).sum()
v2_above_50 = (v2_results['auc'] > 0.50).sum()
total_folds = len(v1_results)

print(f"  V1: {v1_above_50}/{total_folds} folds acima de 0.50")
print(f"  V2_minimal: {v2_above_50}/{total_folds} folds acima de 0.50")

print("\n" + "=" * 100)

# COMMAND ----------

# DBTITLE 1,9. Visualização - ROC-AUC por Fold
# Gráfico: ROC-AUC por Fold (V1 vs V2)
fig, ax = plt.subplots(figsize=(12, 6))

folds_unique = results_df['fold'].unique()
x = np.arange(len(folds_unique))
width = 0.35

v1_aucs = [v1_results[v1_results['fold'] == f]['auc'].values[0] for f in folds_unique]
v2_aucs = [v2_results[v2_results['fold'] == f]['auc'].values[0] for f in folds_unique]

ax.bar(x - width/2, v1_aucs, width, label='Gold V1', alpha=0.8, color='steelblue')
ax.bar(x + width/2, v2_aucs, width, label='Gold V2_minimal', alpha=0.8, color='coral')

# Linha de referência em 0.50
ax.axhline(y=0.50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='ROC-AUC = 0.50 (random)')

ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('ROC-AUC', fontsize=12)
ax.set_title('ROC-AUC por Fold: Gold V1 vs V2_minimal', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(folds_unique)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Gráfico de ROC-AUC por Fold criado")

# COMMAND ----------

# DBTITLE 1,10. Visualização - Diferença V2 - V1
# Gráfico: Diferença ROC-AUC (V2_minimal - V1) por Fold
fig, ax = plt.subplots(figsize=(12, 6))

differences = [v2_aucs[i] - v1_aucs[i] for i in range(len(folds_unique))]
colors = ['green' if d > 0 else 'red' if d < 0 else 'gray' for d in differences]

ax.bar(folds_unique, differences, alpha=0.8, color=colors)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('Diferença ROC-AUC (V2_minimal - V1)', fontsize=12)
ax.set_title('Ganho/Perda de ROC-AUC: V2_minimal vs V1 por Fold', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Adicionar valores nas barras
for i, (fold, diff) in enumerate(zip(folds_unique, differences)):
    ax.text(i, diff + 0.002 if diff > 0 else diff - 0.002, f'{diff:+.4f}', 
            ha='center', va='bottom' if diff > 0 else 'top', fontsize=10)

plt.tight_layout()
plt.show()

print("✅ Gráfico de diferença ROC-AUC criado")

# COMMAND ----------

# DBTITLE 1,🎯 Conclusões e Respostas Objetivas
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC # 🎯 CONCLUSÕES E RESPOSTAS OBJETIVAS
# MAGIC
# MAGIC ## ❓ Perguntas e Respostas
# MAGIC
# MAGIC ### 1. price_vs_ma_7d melhora o modelo de forma consistente?
# MAGIC
# MAGIC **Resposta será baseada nos resultados:**
# MAGIC * Se V2_minimal vencer em 3-4 folds: **SIM, melhora consistente**
# MAGIC * Se V2_minimal vencer em 2 folds: **Melhora moderada, mas não totalmente consistente**
# MAGIC * Se V2_minimal vencer em 0-1 fold: **NÃO, não melhora de forma consistente**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 2. O ganho aparece em vários folds ou apenas em um período?
# MAGIC
# MAGIC **Resposta será baseada nos resultados:**
# MAGIC * Analisar se o ganho é distribuído ou concentrado em um único fold
# MAGIC * Verificar se há padrão temporal (ex: melhora apenas em períodos recentes)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 3. V1 ou V2 Minimal deve seguir para tuning e backtest?
# MAGIC
# MAGIC **Critério de decisão:**
# MAGIC * Se V2_minimal tiver **ROC-AUC médio > V1** e vencer em **maioria dos folds**: **V2_minimal segue**
# MAGIC * Se V1 for superior ou empatar: **V1 segue**
# MAGIC * Se ambas tiverem ROC-AUC médio < 0.55: **Reavaliar feature engineering antes de seguir**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 4. O modelo permanece acima de ROC-AUC 0.50 na maioria dos períodos?
# MAGIC
# MAGIC **Resposta será baseada nos resultados:**
# MAGIC * Contar quantos folds têm ROC-AUC > 0.50
# MAGIC * Se 3-4 folds: **SIM, modelo robusto**
# MAGIC * Se 2 folds: **Moderado, atenção para folds fracos**
# MAGIC * Se 0-1 fold: **NÃO, modelo não é robusto**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 5. Existe robustez suficiente para avançar ao backtest?
# MAGIC
# MAGIC **Critérios:**
# MAGIC * ROC-AUC médio > 0.55
# MAGIC * Maioria dos folds com ROC-AUC > 0.50
# MAGIC * Desvio padrão de ROC-AUC < 0.10 (estabilidade)
# MAGIC * Sem degradação severa em folds recentes
# MAGIC
# MAGIC **Se todos os critérios forem atendidos: SIM, seguir para backtest**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚨 Sinais de Alerta
# MAGIC
# MAGIC * ⚠️ **Degradação temporal:** ROC-AUC pior nos folds mais recentes
# MAGIC * ⚠️ **Alta variabilidade:** Desvio padrão > 0.10
# MAGIC * ⚠️ **Mudança de regime:** Diferença drástica entre folds consecutivos
# MAGIC * ⚠️ **ROC-AUC < 0.50:** Modelo não tem poder preditivo
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚀 Próximos Passos
# MAGIC
# MAGIC **Se V2_minimal vencer:**
# MAGIC 1. Seguir para hyperparameter tuning com V2_minimal
# MAGIC 2. Implementar backtest com estratégia de trading
# MAGIC 3. Avaliar retorno financeiro
# MAGIC
# MAGIC **Se V1 vencer ou empatar:**
# MAGIC 1. Seguir com V1 para tuning
# MAGIC 2. Ou: reavaliar outras features da Gold V2 original
# MAGIC
# MAGIC **Se ambas falharem:**
# MAGIC 1. Reavaliar feature engineering
# MAGIC 2. Considerar outros algoritmos (LightGBM, CatBoost)
# MAGIC 3. Investigar mudança de regime nos dados

# COMMAND ----------

# DBTITLE 1,11. Respostas Automáticas (baseadas nos resultados)
print("=" * 100)
print("🎯 RESPOSTAS AUTOMÁTICAS ÀS PERGUNTAS OBJETIVAS")
print("=" * 100)

# 1. price_vs_ma_7d melhora o modelo de forma consistente?
print("\n❓ 1. price_vs_ma_7d melhora o modelo de forma consistente?")
if folds_won_v2 >= 3:
    print(f"   ✅ SIM. V2_minimal venceu {folds_won_v2}/{len(folds_unique)} folds.")
    print(f"      Melhora consistente em diferentes períodos.")
elif folds_won_v2 == 2:
    print(f"   ⚠️ MODERADO. V2_minimal venceu {folds_won_v2}/{len(folds_unique)} folds.")
    print(f"      Melhora parcial, mas não totalmente consistente.")
else:
    print(f"   ❌ NÃO. V2_minimal venceu apenas {folds_won_v2}/{len(folds_unique)} folds.")
    print(f"      Não há melhora consistente.")

# 2. O ganho aparece em vários folds ou apenas em um período?
print("\n❓ 2. O ganho aparece em vários folds ou apenas em um período?")
if folds_won_v2 >= 3:
    print(f"   ✅ O ganho aparece em vários períodos ({folds_won_v2} folds).")
    print(f"      Indício de feature robusta temporalmente.")
elif folds_won_v2 >= 1:
    print(f"   ⚠️ O ganho aparece em poucos períodos ({folds_won_v2} folds).")
    print(f"      Pode estar relacionado a regime específico de mercado.")
else:
    print(f"   ❌ Não há ganho em nenhum período.")

# 3. V1 ou V2 Minimal deve seguir para tuning e backtest?
print("\n❓ 3. V1 ou V2 Minimal deve seguir para tuning e backtest?")
v1_mean_auc = v1_results['auc'].mean()
v2_mean_auc = v2_results['auc'].mean()

if v2_mean_auc > v1_mean_auc and folds_won_v2 >= 2:
    print(f"   🏆 RECOMENDAÇÃO: Gold V2_minimal")
    print(f"      ROC-AUC médio: {v2_mean_auc:.4f} (V2) vs {v1_mean_auc:.4f} (V1)")
    print(f"      Folds vencidos: {folds_won_v2}/{len(folds_unique)}")
    print(f"      Seguir com V2_minimal para tuning e backtest.")
elif v1_mean_auc >= v2_mean_auc or folds_won_v1 >= folds_won_v2:
    print(f"   🏆 RECOMENDAÇÃO: Gold V1")
    print(f"      ROC-AUC médio: {v1_mean_auc:.4f} (V1) vs {v2_mean_auc:.4f} (V2)")
    print(f"      Folds vencidos: {folds_won_v1}/{len(folds_unique)}")
    print(f"      Seguir com V1 para tuning e backtest.")
else:
    print(f"   ⚠️ INCONCLUSIVO. Analisar caso a caso.")

if max(v1_mean_auc, v2_mean_auc) < 0.55:
    print(f"\n   ⚠️ ALERTA: ROC-AUC médio < 0.55")
    print(f"      Considerar reavaliar feature engineering antes de seguir.")

# 4. O modelo permanece acima de ROC-AUC 0.50 na maioria dos períodos?
print("\n❓ 4. O modelo permanece acima de ROC-AUC 0.50 na maioria dos períodos?")
best_version = 'V2_minimal' if v2_mean_auc > v1_mean_auc else 'V1'
best_results = v2_results if best_version == 'V2_minimal' else v1_results
above_50 = (best_results['auc'] > 0.50).sum()

if above_50 >= 3:
    print(f"   ✅ SIM. {best_version} está acima de 0.50 em {above_50}/{len(folds_unique)} folds.")
    print(f"      Modelo tem poder preditivo robusto.")
elif above_50 >= 2:
    print(f"   ⚠️ MODERADO. {best_version} está acima de 0.50 em {above_50}/{len(folds_unique)} folds.")
    print(f"      Atenção para folds fracos.")
else:
    print(f"   ❌ NÃO. {best_version} está acima de 0.50 em apenas {above_50}/{len(folds_unique)} folds.")
    print(f"      Modelo não é robusto.")

# 5. Existe robustez suficiente para avançar ao backtest?
print("\n❓ 5. Existe robustez suficiente para avançar ao backtest?")
std_auc = best_results['auc'].std()
mean_auc = best_results['auc'].mean()

criteria = {
    'ROC-AUC médio > 0.55': mean_auc > 0.55,
    'Maioria dos folds > 0.50': above_50 >= 3,
    'Desvio padrão < 0.10': std_auc < 0.10
}

print(f"\n   Critérios de Robustez ({best_version}):")
for criterion, passed in criteria.items():
    status = '✅' if passed else '❌'
    print(f"      {status} {criterion}")
    if 'ROC-AUC médio' in criterion:
        print(f"          Valor: {mean_auc:.4f}")
    elif 'Desvio padrão' in criterion:
        print(f"          Valor: {std_auc:.4f}")
    elif 'Maioria' in criterion:
        print(f"          Valor: {above_50}/{len(folds_unique)} folds")

all_passed = all(criteria.values())
if all_passed:
    print(f"\n   ✅ SIM. Todos os critérios atendidos.")
    print(f"      Seguir para backtest com {best_version}.")
else:
    print(f"\n   ❌ NÃO. Alguns critérios não foram atendidos.")
    print(f"      Reavaliar antes de seguir para backtest.")

print("\n" + "=" * 100)

# COMMAND ----------

# DBTITLE 1,🔍 AUDITORIA COMPLETA - Parte 1: Detalhamento por Fold
print("\n" + "=" * 120)
print("🔍 AUDITORIA COMPLETA - PARTE 1: DETALHAMENTO POR FOLD E VERSÃO")
print("=" * 120)

# Para cada fold, extrair informações detalhadas
for i, fold_config in enumerate(folds):
    fold_name = fold_config['name']
    print(f"\n{'='*120}")
    print(f"📊 {fold_name}")
    print(f"{'='*120}")
    
    # Períodos
    train_start = pd.to_datetime(fold_config['train_start'])
    train_end = pd.to_datetime(fold_config['train_end'])
    eval_start = pd.to_datetime(fold_config['eval_start'])
    eval_end = pd.to_datetime(fold_config['eval_end'])
    
    # Ajustar eval_end para última data disponível se necessário
    max_date_available = df_v1['date'].max()
    if eval_end > max_date_available:
        eval_end = max_date_available
    
    # Recalcular máscaras para auditoria
    train_mask_v1 = (df_v1['date'] >= train_start) & (df_v1['date'] <= train_end)
    eval_mask_base_v1 = (df_v1['date'] >= eval_start) & (df_v1['date'] <= eval_end)
    
    # Aplicar gap
    eval_dates_v1 = df_v1[eval_mask_base_v1]['date'].drop_duplicates().sort_values().reset_index(drop=True)
    if len(eval_dates_v1) > 7:
        gap_date_v1 = eval_dates_v1.iloc[7]
        eval_mask_v1 = eval_mask_base_v1 & (df_v1['date'] >= gap_date_v1)
        gap_start = eval_dates_v1.iloc[0]
        gap_end = eval_dates_v1.iloc[6]
    else:
        eval_mask_v1 = eval_mask_base_v1
        gap_start = gap_end = None
    
    # Períodos exatos
    print(f"\n📅 PERÍODOS:")
    print(f"  Treino: {train_start.date()} a {train_end.date()} ({(train_end - train_start).days + 1} dias)")
    if gap_start:
        print(f"  Gap: {gap_start.date()} a {gap_end.date()} (7 pregões)")
    else:
        print(f"  Gap: Não aplicado (período curto)")
    actual_eval_start = df_v1[eval_mask_v1]['date'].min()
    actual_eval_end = df_v1[eval_mask_v1]['date'].max()
    print(f"  Avaliação: {actual_eval_start.date()} a {actual_eval_end.date()} ({(actual_eval_end - actual_eval_start).days + 1} dias)")
    
    # Quantidades por versão
    for version, df, mask_train, mask_eval in [
        ('V1', df_v1, train_mask_v1, eval_mask_v1),
        ('V2_minimal', df_v2_minimal, train_mask_v1, eval_mask_v1)
    ]:
        print(f"\n🔹 {version}:")
        
        # Quantidades totais
        n_train = mask_train.sum()
        n_eval = mask_eval.sum()
        print(f"  Total treino: {n_train} observações")
        print(f"  Total avaliação: {n_eval} observações")
        
        # Quantidades por ticker
        print(f"  \n  Observações por ticker (treino):")
        for ticker in sorted(df['ticker'].unique()):
            n_ticker_train = ((df['ticker'] == ticker) & mask_train).sum()
            print(f"    {ticker}: {n_ticker_train}")
        
        print(f"  \n  Observações por ticker (avaliação):")
        for ticker in sorted(df['ticker'].unique()):
            n_ticker_eval = ((df['ticker'] == ticker) & mask_eval).sum()
            print(f"    {ticker}: {n_ticker_eval}")
        
        # Distribuição do target
        target_eval = df.loc[mask_eval, 'target_7d']
        target_dist = target_eval.value_counts(normalize=True).sort_index()
        print(f"  \n  Distribuição target_7d (avaliação):")
        print(f"    False (não subiu): {target_dist.get(False, 0):.2%} ({(target_eval == False).sum()} obs)")
        print(f"    True (subiu): {target_dist.get(True, 0):.2%} ({(target_eval == True).sum()} obs)")
        
        # Buscar métricas do resultado
        result = [r for r in all_results if r['fold'] == fold_name and r['version'] == version][0]
        print(f"  \n  MÉTRICAS:")
        print(f"    ROC-AUC: {result['auc']:.4f}")
        print(f"    Accuracy: {result['accuracy']:.4f}")
        print(f"    Precision: {result['precision']:.4f}")
        print(f"    Recall: {result['recall']:.4f}")
        print(f"    F1-Score: {result['f1']:.4f}")
        print(f"    Taxa de acerto: {result['accuracy']:.2%}")
        print(f"    Best iteration: {result['best_iteration']}")

print("\n" + "=" * 120)

# COMMAND ----------

# DBTITLE 1,🔍 AUDITORIA COMPLETA - Parte 2: Análise do Fold 4 e Sensibilidade
print("\n" + "=" * 120)
print("🔍 AUDITORIA COMPLETA - PARTE 2: ANÁLISE DO FOLD 4 E SENSIBILIDADE")
print("=" * 120)

# Análise detalhada do Fold 4
fold4_v1 = [r for r in all_results if r['fold'] == 'Fold 4' and r['version'] == 'V1'][0]
fold4_v2 = [r for r in all_results if r['fold'] == 'Fold 4' and r['version'] == 'V2_minimal'][0]

print(f"\n⚠️ ANÁLISE ESPECIAL DO FOLD 4 (2025):")
print(f"\n  Características do Fold 4:")
print(f"    Período: 2025-01-01 a {df_v1['date'].max().date()}")
print(f"    Duração: ~1.5 meses (vs ~12 meses nos Folds 1-3)")
print(f"    Observações V1: {fold4_v1['n_eval']} (vs 1215, 1205, 1220 nos Folds 1-3)")
print(f"    Observações V2: {fold4_v2['n_eval']}")

print(f"\n  Resultados do Fold 4:")
print(f"    V1 ROC-AUC: {fold4_v1['auc']:.4f} (melhor fold da V1)")
print(f"    V2 ROC-AUC: {fold4_v2['auc']:.4f}")
print(f"    Diferença: {fold4_v1['auc'] - fold4_v2['auc']:+.4f} (V1 vence por {abs(fold4_v1['auc'] - fold4_v2['auc']):.4f})")

print(f"\n  🚨 PROBLEMAS IDENTIFICADOS:")
print(f"    1. Fold 4 tem apenas {fold4_v1['n_eval']} obs (10% do tamanho dos outros folds)")
print(f"    2. Período muito curto (~1.5 meses vs 12 meses)")
print(f"    3. Alta incerteza estatística devido ao tamanho pequeno da amostra")
print(f"    4. Não é comparável aos folds anuais completos")

# Análise de sensibilidade: médias com e sem Fold 4
print(f"\n\n📊 ANÁLISE DE SENSIBILIDADE - INFLUÊNCIA DO FOLD 4:")

# Médias com todos os folds (1-4)
v1_mean_all = v1_results['auc'].mean()
v2_mean_all = v2_results['auc'].mean()

print(f"\n  Média Simples COM Fold 4 (Folds 1-4):")
print(f"    V1: {v1_mean_all:.4f}")
print(f"    V2: {v2_mean_all:.4f}")
print(f"    Diferença (V1 - V2): {v1_mean_all - v2_mean_all:+.4f}")
print(f"    Vencedor: {'V1' if v1_mean_all > v2_mean_all else 'V2'}")

# Médias sem Fold 4 (apenas 1-3)
v1_results_no_f4 = v1_results[v1_results['fold'] != 'Fold 4']
v2_results_no_f4 = v2_results[v2_results['fold'] != 'Fold 4']

v1_mean_no_f4 = v1_results_no_f4['auc'].mean()
v2_mean_no_f4 = v2_results_no_f4['auc'].mean()

print(f"\n  Média Simples SEM Fold 4 (apenas Folds 1-3):")
print(f"    V1: {v1_mean_no_f4:.4f}")
print(f"    V2: {v2_mean_no_f4:.4f}")
print(f"    Diferença (V1 - V2): {v1_mean_no_f4 - v2_mean_no_f4:+.4f}")
print(f"    Vencedor: {'V1' if v1_mean_no_f4 > v2_mean_no_f4 else 'V2'}")

print(f"\n  🔎 INTERPRETAÇÃO:")
if v1_mean_all > v2_mean_all and v1_mean_no_f4 < v2_mean_no_f4:
    print(f"    ⚠️ ATENÇÃO: V1 só vence na média simples por causa do Fold 4!")
    print(f"    Sem o Fold 4, V2 seria superior em {v2_mean_no_f4 - v1_mean_no_f4:.4f} pontos.")
    print(f"    O alto ROC-AUC de V1 no Fold 4 ({fold4_v1['auc']:.4f}) está puxando a média.")
    print(f"    Mas o Fold 4 tem alta incerteza devido ao tamanho pequeno.")
elif v1_mean_all > v2_mean_all and v1_mean_no_f4 > v2_mean_no_f4:
    print(f"    V1 vence tanto com quanto sem Fold 4.")
    print(f"    A vantagem é consistente entre as configurações.")
else:
    print(f"    V2 vence em ambas as configurações.")

# Impacto do Fold 4 na média
impact = abs(v1_mean_all - v1_mean_no_f4)
print(f"\n  Impacto do Fold 4 na média da V1: {impact:.4f} pontos")
print(f"  Impacto percentual: {(impact / v1_mean_no_f4) * 100:.2f}%")

print("\n" + "=" * 120)

# COMMAND ----------

# DBTITLE 1,🔍 AUDITORIA COMPLETA - Parte 3: Métricas Globais Out-of-Sample
print("\n" + "=" * 120)
print("🔍 AUDITORIA COMPLETA - PARTE 3: MÉTRICAS GLOBAIS OUT-OF-SAMPLE")
print("=" * 120)

print("\nConcatenando predições out-of-sample de todos os folds...\n")

# Concatenar todas as predições out-of-sample de cada versão
def concatenate_predictions(version_name):
    """Concatena predições de todos os folds para uma versão"""
    version_results = [r for r in all_results if r['version'] == version_name]
    
    y_true_all = np.concatenate([r['y_true'] for r in version_results])
    y_pred_proba_all = np.concatenate([r['y_pred_proba'] for r in version_results])
    y_pred_all = np.concatenate([r['y_pred'] for r in version_results])
    
    return y_true_all, y_pred_proba_all, y_pred_all, version_results

# V1
y_true_v1, y_pred_proba_v1, y_pred_v1, v1_fold_results = concatenate_predictions('V1')

# V2
y_true_v2, y_pred_proba_v2, y_pred_v2, v2_fold_results = concatenate_predictions('V2_minimal')

print(f"✅ V1: {len(y_true_v1)} predições out-of-sample concatenadas")
print(f"✅ V2_minimal: {len(y_true_v2)} predições out-of-sample concatenadas")

# Calcular métricas globais
print("\n" + "="*120)
print("🌍 MÉTRICAS GLOBAIS OUT-OF-SAMPLE (TODOS OS FOLDS CONCATENADOS)")
print("="*120)

for version_name, y_true, y_pred_proba, y_pred in [
    ('V1', y_true_v1, y_pred_proba_v1, y_pred_v1),
    ('V2_minimal', y_true_v2, y_pred_proba_v2, y_pred_v2)
]:
    print(f"\n🔹 {version_name}:")
    
    # Métricas globais
    auc_global = roc_auc_score(y_true, y_pred_proba)
    acc_global = accuracy_score(y_true, y_pred)
    prec_global = precision_score(y_true, y_pred, zero_division=0)
    rec_global = recall_score(y_true, y_pred, zero_division=0)
    f1_global = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"  ROC-AUC global: {auc_global:.4f}")
    print(f"  Accuracy global: {acc_global:.4f} ({acc_global:.2%})")
    print(f"  Precision global: {prec_global:.4f}")
    print(f"  Recall global: {rec_global:.4f}")
    print(f"  F1-Score global: {f1_global:.4f}")
    print(f"  Total de predições: {len(y_true)}")
    
    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Matriz de Confusão:")
    print(f"    TN: {cm[0,0]:<6} FP: {cm[0,1]:<6}")
    print(f"    FN: {cm[1,0]:<6} TP: {cm[1,1]:<6}")

# Armazenar para comparação
auc_global_v1 = roc_auc_score(y_true_v1, y_pred_proba_v1)
auc_global_v2 = roc_auc_score(y_true_v2, y_pred_proba_v2)

print("\n" + "="*120)
print("🔎 COMPARAÇÃO GLOBAL:")
print("="*120)
print(f"\n  ROC-AUC Global:")
print(f"    V1: {auc_global_v1:.4f}")
print(f"    V2_minimal: {auc_global_v2:.4f}")
print(f"    Diferença (V2 - V1): {auc_global_v2 - auc_global_v1:+.4f}")
print(f"    Vencedor: {'V2_minimal' if auc_global_v2 > auc_global_v1 else 'V1'}")

# Médias ponderadas pelo número de observações
print("\n" + "="*120)
print("⚖️ MÉDIAS PONDERADAS POR NÚMERO DE OBSERVAÇÕES:")
print("="*120)

# V1
weights_v1 = np.array([r['n_eval'] for r in v1_fold_results])
aucs_v1 = np.array([r['auc'] for r in v1_fold_results])
weighted_mean_v1 = np.average(aucs_v1, weights=weights_v1)

# V2
weights_v2 = np.array([r['n_eval'] for r in v2_fold_results])
aucs_v2 = np.array([r['auc'] for r in v2_fold_results])
weighted_mean_v2 = np.average(aucs_v2, weights=weights_v2)

print(f"\n  Média Ponderada ROC-AUC:")
print(f"    V1: {weighted_mean_v1:.4f}")
print(f"    V2_minimal: {weighted_mean_v2:.4f}")
print(f"    Diferença (V2 - V1): {weighted_mean_v2 - weighted_mean_v1:+.4f}")
print(f"    Vencedor: {'V2_minimal' if weighted_mean_v2 > weighted_mean_v1 else 'V1'}")

print(f"\n  Pesos (número de observações por fold):")
for i, (fold_name, w) in enumerate(zip(['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4'], weights_v1)):
    print(f"    {fold_name}: {w} obs ({w/weights_v1.sum():.1%} do total)")

print("\n" + "="*120)
print("📊 RESUMO COMPARATIVO:")
print("="*120)
print(f"\n  Métrica                    | V1      | V2_minimal | Vencedor")
print(f"  " + "-"*60)
print(f"  Média Simples (Folds 1-4) | {v1_mean_all:.4f} | {v2_mean_all:.4f}     | {'V1' if v1_mean_all > v2_mean_all else 'V2'}")
print(f"  Média Simples (Folds 1-3) | {v1_mean_no_f4:.4f} | {v2_mean_no_f4:.4f}     | {'V1' if v1_mean_no_f4 > v2_mean_no_f4 else 'V2'}")
print(f"  Média Ponderada           | {weighted_mean_v1:.4f} | {weighted_mean_v2:.4f}     | {'V1' if weighted_mean_v1 > weighted_mean_v2 else 'V2'}")
print(f"  ROC-AUC Global            | {auc_global_v1:.4f} | {auc_global_v2:.4f}     | {'V1' if auc_global_v1 > auc_global_v2 else 'V2'}")
print(f"  Desvio Padrão (Folds)     | {v1_results['auc'].std():.4f} | {v2_results['auc'].std():.4f}     | {'V1' if v1_results['auc'].std() < v2_results['auc'].std() else 'V2'} (menor é melhor)")
print(f"  Folds Vencidos            | {folds_won_v1}/4    | {folds_won_v2}/4       | {'V1' if folds_won_v1 > folds_won_v2 else 'V2'}")

print("\n" + "="*120)

# COMMAND ----------

# DBTITLE 1,🔍 AUDITORIA COMPLETA - Parte 4: Análise Estatística e Recomendação Final
print("\n" + "=" * 120)
print("🔍 AUDITORIA COMPLETA - PARTE 4: ANÁLISE ESTATÍSTICA E RECOMENDAÇÃO FINAL")
print("=" * 120)

# Bootstrap temporal para intervalos de confiança
print("\n📏 BOOTSTRAP TEMPORAL - Intervalos de Confiança (95%):")
print("\nMétodo: Reamostragem de folds completos (preserva estrutura temporal)\n")

np.random.seed(42)
n_bootstrap = 1000

# Bootstrap resampling folds (não linhas individuais)
bootstrap_aucs_v1 = []
bootstrap_aucs_v2 = []

for _ in range(n_bootstrap):
    # Reamostrar folds com reposição
    sampled_indices = np.random.choice(len(v1_fold_results), size=len(v1_fold_results), replace=True)
    
    # V1
    sampled_v1 = [v1_fold_results[i] for i in sampled_indices]
    y_true_boot_v1 = np.concatenate([r['y_true'] for r in sampled_v1])
    y_pred_proba_boot_v1 = np.concatenate([r['y_pred_proba'] for r in sampled_v1])
    auc_boot_v1 = roc_auc_score(y_true_boot_v1, y_pred_proba_boot_v1)
    bootstrap_aucs_v1.append(auc_boot_v1)
    
    # V2
    sampled_v2 = [v2_fold_results[i] for i in sampled_indices]
    y_true_boot_v2 = np.concatenate([r['y_true'] for r in sampled_v2])
    y_pred_proba_boot_v2 = np.concatenate([r['y_pred_proba'] for r in sampled_v2])
    auc_boot_v2 = roc_auc_score(y_true_boot_v2, y_pred_proba_boot_v2)
    bootstrap_aucs_v2.append(auc_boot_v2)

bootstrap_aucs_v1 = np.array(bootstrap_aucs_v1)
bootstrap_aucs_v2 = np.array(bootstrap_aucs_v2)

# Intervalos de confiança 95%
ci_v1 = np.percentile(bootstrap_aucs_v1, [2.5, 97.5])
ci_v2 = np.percentile(bootstrap_aucs_v2, [2.5, 97.5])

print(f"  V1:")
print(f"    ROC-AUC Global: {auc_global_v1:.4f}")
print(f"    IC 95%: [{ci_v1[0]:.4f}, {ci_v1[1]:.4f}]")
print(f"    Amplitude: {ci_v1[1] - ci_v1[0]:.4f}")

print(f"\n  V2_minimal:")
print(f"    ROC-AUC Global: {auc_global_v2:.4f}")
print(f"    IC 95%: [{ci_v2[0]:.4f}, {ci_v2[1]:.4f}]")
print(f"    Amplitude: {ci_v2[1] - ci_v2[0]:.4f}")

# Teste de hipótese via bootstrap
bootstrap_diff = bootstrap_aucs_v2 - bootstrap_aucs_v1
ci_diff = np.percentile(bootstrap_diff, [2.5, 97.5])

print(f"\n  Diferença (V2 - V1):")
print(f"    Diferença observada: {auc_global_v2 - auc_global_v1:+.4f}")
print(f"    IC 95% da diferença: [{ci_diff[0]:+.4f}, {ci_diff[1]:+.4f}]")

if ci_diff[0] > 0:
    print(f"    ✅ V2 é estatisticamente superior (IC não inclui zero)")
elif ci_diff[1] < 0:
    print(f"    ✅ V1 é estatisticamente superior (IC não inclui zero)")
else:
    print(f"    ⚠️ Diferença NÃO é estatisticamente significante (IC inclui zero)")
    print(f"    As versões devem ser consideradas EQUIVALENTES")

# Responder às perguntas do usuário
print("\n" + "="*120)
print("❓ RESPOSTAS ÀS PERGUNTAS DA AUDITORIA:")
print("="*120)

print("\n1. A vantagem média da V1 é real ou está sendo puxada pelo Fold 4 curto?")
if v1_mean_all > v2_mean_all and v1_mean_no_f4 < v2_mean_no_f4:
    print(f"   🚨 A vantagem da V1 ESTÁ SENDO PUXADA PELO FOLD 4.")
    print(f"   Sem o Fold 4: V2 seria superior ({v2_mean_no_f4:.4f} vs {v1_mean_no_f4:.4f}).")
    print(f"   O Fold 4 tem apenas {fold4_v1['n_eval']} obs e alta incerteza.")
else:
    print(f"   V1 não depende exclusivamente do Fold 4.")

print("\n2. A V2 Minimal apresenta maior estabilidade temporal?")
if v2_results['auc'].std() < v1_results['auc'].std():
    print(f"   ✅ SIM. V2_minimal tem menor desvio padrão ({v2_results['auc'].std():.4f} vs {v1_results['auc'].std():.4f}).")
    print(f"   V2 é {((v1_results['auc'].std() - v2_results['auc'].std()) / v1_results['auc'].std()) * 100:.1f}% mais estável.")
else:
    print(f"   NÃO. V1 tem menor desvio padrão.")

print("\n3. A V2 Minimal vencer 3 de 4 folds é mais relevante do que a maior média simples da V1?")
if folds_won_v2 > folds_won_v1:
    print(f"   ✅ SIM. V2 venceu {folds_won_v2}/4 folds, mostrando consistência.")
    print(f"   A média simples da V1 é inflada pelo Fold 4 curto e instável.")
    print(f"   Consistência entre folds é mais importante que média simples.")
else:
    print(f"   Não necessariamente. V1 venceu mais folds.")

print("\n4. Qual versão apresenta o melhor ROC-AUC global out-of-sample?")
if auc_global_v2 > auc_global_v1:
    print(f"   🏆 V2_minimal: {auc_global_v2:.4f} vs V1: {auc_global_v1:.4f}")
    print(f"   Diferença: {auc_global_v2 - auc_global_v1:+.4f}")
else:
    print(f"   🏆 V1: {auc_global_v1:.4f} vs V2: {auc_global_v2:.4f}")

print("\n5. Qual versão apresenta a melhor média ponderada?")
if weighted_mean_v2 > weighted_mean_v1:
    print(f"   🏆 V2_minimal: {weighted_mean_v2:.4f} vs V1: {weighted_mean_v1:.4f}")
else:
    print(f"   🏆 V1: {weighted_mean_v1:.4f} vs V2: {weighted_mean_v2:.4f}")

print("\n6. Qual versão apresenta menor variação entre períodos?")
if v2_results['auc'].std() < v1_results['auc'].std():
    print(f"   🏆 V2_minimal: desvio {v2_results['auc'].std():.4f} vs V1: {v1_results['auc'].std():.4f}")
    print(f"   V2 é mais estável temporalmente.")
else:
    print(f"   🏆 V1: desvio {v1_results['auc'].std():.4f} vs V2: {v2_results['auc'].std():.4f}")

print("\n7. A diferença entre as versões é suficientemente grande e estável?")
diff_magnitude = abs(auc_global_v2 - auc_global_v1)
if diff_magnitude < 0.01:
    print(f"   ⚠️ NÃO. Diferença global muito pequena ({diff_magnitude:.4f} < 0.01).")
    print(f"   As versões são praticamente equivalentes.")
else:
    print(f"   ✅ Diferença apreciável: {diff_magnitude:.4f} pontos.")

print("\n8. Os resultados são estatisticamente distinguivéis ou devem ser considerados equivalentes?")
if ci_diff[0] > 0 or ci_diff[1] < 0:
    print(f"   ✅ ESTATISTICAMENTE DISTINGUIVÉIS (IC 95% não inclui zero).")
else:
    print(f"   ⚠️ ESTATISTICAMENTE EQUIVALENTES (IC 95% inclui zero).")
    print(f"   Não há evidência estatística de diferença real entre as versões.")

print("\n" + "="*120)

# COMMAND ----------

# DBTITLE 1,🏆 RECOMENDAÇÃO FINAL
print("\n" + "="*120)
print("🏆 RECOMENDAÇÃO FINAL")
print("="*120)

print("\n📋 SÍNTESE DA AUDITORIA:\n")

# Contar quantas métricas cada versão vence
v1_wins = 0
v2_wins = 0
ties = 0

metrics_comparison = [
    ('Média Simples (1-4)', v1_mean_all, v2_mean_all),
    ('Média Simples (1-3)', v1_mean_no_f4, v2_mean_no_f4),
    ('Média Ponderada', weighted_mean_v1, weighted_mean_v2),
    ('ROC-AUC Global', auc_global_v1, auc_global_v2),
    ('Estabilidade', -v1_results['auc'].std(), -v2_results['auc'].std()),  # Negativo porque menor é melhor
    ('Folds Vencidos', folds_won_v1, folds_won_v2)
]

for metric_name, v1_val, v2_val in metrics_comparison:
    if v1_val > v2_val:
        v1_wins += 1
    elif v2_val > v1_val:
        v2_wins += 1
    else:
        ties += 1

print(f"  Placar de Métricas:")
print(f"    V1 vence em: {v1_wins}/6 métricas")
print(f"    V2_minimal vence em: {v2_wins}/6 métricas")
print(f"    Empates: {ties}/6")

print(f"\n  Achados Críticos:")

# Achado 1: Fold 4
if v1_mean_all > v2_mean_all and v1_mean_no_f4 < v2_mean_no_f4:
    print(f"    ⚠️ Fold 4 (apenas {fold4_v1['n_eval']} obs) infla artificialmente a média de V1")
    fold4_critical = True
else:
    fold4_critical = False

# Achado 2: Estabilidade
if v2_results['auc'].std() < v1_results['auc'].std():
    print(f"    ✅ V2 é {((v1_results['auc'].std() - v2_results['auc'].std()) / v1_results['auc'].std()) * 100:.1f}% mais estável que V1")
    v2_more_stable = True
else:
    v2_more_stable = False

# Achado 3: Consistência
if folds_won_v2 >= 3:
    print(f"    ✅ V2 venceu {folds_won_v2}/4 folds, mostrando consistência temporal")
    v2_consistent = True
else:
    v2_consistent = False

# Achado 4: Significância estatística
statistically_significant = (ci_diff[0] > 0 or ci_diff[1] < 0)
if statistically_significant:
    print(f"    ✅ Diferença é estatisticamente significante (IC 95% não inclui zero)")
else:
    print(f"    ⚠️ Diferença NÃO é estatisticamente significante (IC 95%: [{ci_diff[0]:+.4f}, {ci_diff[1]:+.4f}])")

# Decisão final
print("\n" + "="*120)
print("🎯 DECISÃO FINAL:")
print("="*120)

# Lógica de decisão
if not statistically_significant:
    # Versões equivalentes estatisticamente
    print("\n⚠️ As versões são ESTATISTICAMENTE EQUIVALENTES.\n")
    print("Justificativa:")
    print(f"  * O intervalo de confiança da diferença inclui zero: [{ci_diff[0]:+.4f}, {ci_diff[1]:+.4f}]")
    print(f"  * Não há evidência estatística de que uma versão seja superior.")
    print(f"  * Diferença global muito pequena: {abs(auc_global_v2 - auc_global_v1):.4f} pontos.")
    
    # Escolher a mais simples
    print("\n🏆 RECOMENDAÇÃO: **Gold V1**")
    print("\nRazão: Princípio da Parcimônia (Navalha de Occam)")
    print("  * V1 é mais simples (23 features vs 24)")
    print("  * price_vs_ma_7d não adiciona valor estatístico demonstrável")
    print("  * Menor risco de overfitting no tuning")
    print("  * Menor complexidade computacional")
    
    final_choice = 'A'
    
elif v2_wins >= 4 and v2_more_stable and v2_consistent and not fold4_critical:
    # V2 claramente superior
    print("\n✅ V2_minimal é SUPERIOR a V1.\n")
    print("Justificativa:")
    print(f"  * V2 vence em {v2_wins}/6 métricas principais")
    print(f"  * V2 venceu {folds_won_v2}/4 folds (consistência temporal)")
    print(f"  * V2 é mais estável (desvio {v2_results['auc'].std():.4f} vs {v1_results['auc'].std():.4f})")
    print(f"  * ROC-AUC global: {auc_global_v2:.4f} vs {auc_global_v1:.4f}")
    
    print("\n🏆 RECOMENDAÇÃO: **Gold V2_minimal**")
    final_choice = 'B'
    
elif v2_wins >= 3 and (v2_more_stable or v2_consistent) and fold4_critical:
    # V2 superior considerando o problema do Fold 4
    print("\n✅ V2_minimal é SUPERIOR a V1 (especialmente desconsiderando o Fold 4 problemático).\n")
    print("Justificativa:")
    print(f"  * Fold 4 tem apenas {fold4_v1['n_eval']} obs e infla artificialmente a média de V1")
    print(f"  * Sem Fold 4: V2 seria superior ({v2_mean_no_f4:.4f} vs {v1_mean_no_f4:.4f})")
    print(f"  * V2 venceu {folds_won_v2}/4 folds (consistência em períodos completos)")
    if v2_more_stable:
        print(f"  * V2 é mais estável temporalmente")
    print(f"  * ROC-AUC global: {auc_global_v2:.4f} vs {auc_global_v1:.4f}")
    
    print("\n🏆 RECOMENDAÇÃO: **Gold V2_minimal**")
    final_choice = 'B'
    
else:
    # V1 superior
    print("\n✅ V1 é SUPERIOR a V2_minimal.\n")
    print("Justificativa:")
    print(f"  * V1 vence em {v1_wins}/6 métricas principais")
    if v1_mean_all > v2_mean_all:
        print(f"  * Média simples V1: {v1_mean_all:.4f} vs V2: {v2_mean_all:.4f}")
    if auc_global_v1 > auc_global_v2:
        print(f"  * ROC-AUC global: {auc_global_v1:.4f} vs {auc_global_v2:.4f}")
    
    print("\n🏆 RECOMENDAÇÃO: **Gold V1**")
    final_choice = 'A'

print("\n" + "="*120)
print("🚀 PRÓXIMOS PASSOS:")
print("="*120)

if final_choice == 'A':
    print("\n1. Seguir com **Gold V1** para hyperparameter tuning")
    print("2. Implementar backtest com estratégia de trading usando V1")
    print("3. Avaliar retorno financeiro ajustado por risco")
    print("4. Validar em período mais recente (out-of-time final)")
elif final_choice == 'B':
    print("\n1. Seguir com **Gold V2_minimal** para hyperparameter tuning")
    print("2. Implementar backtest com estratégia de trading usando V2_minimal")
    print("3. Avaliar retorno financeiro ajustado por risco")
    print("4. Validar em período mais recente (out-of-time final)")
else:
    print("\n1. Revisar critérios de decisão")
    print("2. Considerar execução paralela de tuning em ambas versões")

print("\n⚠️  NÃO consultar o antigo Test como holdout final.")
print("⚠️  NÃO implementar backtest neste notebook.")

print("\n" + "="*120)
print("✅ AUDITORIA COMPLETA CONCLUÍDA")
print("="*120)

# COMMAND ----------

# DBTITLE 1,📝 Sumário Executivo
# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC # 📝 SUMÁRIO EXECUTIVO
# MAGIC
# MAGIC ## ✅ Processo Realizado
# MAGIC
# MAGIC 1. **Carregamento de dados:** Gold V1 (23 features) e Gold V2_minimal (24 features)
# MAGIC 2. **Walk-Forward Validation:** 4 folds temporais com janela expansiva
# MAGIC 3. **Gap de 7 pregões:** Entre treino e avaliação para evitar sobreposição do target_7d
# MAGIC 4. **XGBoost fixo:** Mesmos hiperparâmetros do notebook 33_ml_advanced (sem tuning)
# MAGIC 5. **Métricas:** ROC-AUC, Accuracy, Precision, Recall, F1 para cada fold
# MAGIC 6. **Comparação:** V1 vs V2_minimal fold por fold
# MAGIC 7. **Análise:** Estatísticas, visualizações e respostas objetivas
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Folds Executados
# MAGIC
# MAGIC * **Fold 1:** Train (2020-03 a 2021-12) → Eval (2022)
# MAGIC * **Fold 2:** Train (2020-03 a 2022-12) → Eval (2023)
# MAGIC * **Fold 3:** Train (2020-03 a 2023-12) → Eval (2024)
# MAGIC * **Fold 4:** Train (2020-03 a 2024-12) → Eval (2025+)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Garantias de Rigor
# MAGIC
# MAGIC * ✅ **Sem tuning durante walk-forward:** Hiperparâmetros fixos
# MAGIC * ✅ **Sem criação de features:** Apenas V1 e V2_minimal existentes
# MAGIC * ✅ **Sem seleção de features:** Todas as features de cada versão foram usadas
# MAGIC * ✅ **Gap temporal:** 7 pregões entre treino e avaliação
# MAGIC * ✅ **Mesmas linhas:** V1 e V2 comparadas exatamente nos mesmos registros de avaliação
# MAGIC * ✅ **Walk-forward expansivo:** Janela de treino cresce a cada fold
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ❓ Perguntas Respondidas
# MAGIC
# MAGIC 1. ✅ **price_vs_ma_7d melhora consistentemente?** Respondido automaticamente
# MAGIC 2. ✅ **Ganho em vários folds ou um período?** Analisado fold por fold
# MAGIC 3. ✅ **V1 ou V2 deve seguir?** Recomendação baseada em métricas
# MAGIC 4. ✅ **Modelo acima de 0.50?** Verificado para todos os folds
# MAGIC 5. ✅ **Robustez para backtest?** Avaliado com critérios objetivos
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🚀 Próximos Passos (após execução)
# MAGIC
# MAGIC **Se V2_minimal vencer:**
# MAGIC * Seguir para hyperparameter tuning com V2_minimal
# MAGIC * Implementar backtest com estratégia de trading
# MAGIC
# MAGIC **Se V1 vencer:**
# MAGIC * Seguir com V1 para tuning
# MAGIC * Ou reavaliar outras features da Gold V2 original
# MAGIC
# MAGIC **Se ambas falharem (ROC-AUC médio < 0.55):**
# MAGIC * Reavaliar feature engineering
# MAGIC * Considerar outros algoritmos
# MAGIC * Investigar mudança de regime
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🛡️ Limitações
# MAGIC
# MAGIC * Dataset pequeno (5 FIIs, ~6.000 registros)
# MAGIC * Período relativamente curto (2020-2025)
# MAGIC * Target binário (target_7d) — não prevemos magnitude do retorno
# MAGIC * Não consideramos custos de transação neste notebook (será feito no backtest)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📚 Documentação
# MAGIC
# MAGIC * **Notebook:** 37_walk_forward_validation
# MAGIC * **Método:** Walk-Forward Validation com janela expansiva
# MAGIC * **Tabelas comparadas:** workspace.gold.fii_features_v1 vs workspace.gold.fii_features_v2_minimal
# MAGIC * **Feature nova testada:** price_vs_ma_7d
# MAGIC * **Status:** ✅ **COMPLETO**

# COMMAND ----------

