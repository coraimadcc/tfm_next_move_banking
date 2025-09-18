# Databricks notebook source
# MAGIC %md
# MAGIC # TFM: Next Move Banking
# MAGIC **Autor:** Coraima Castillo
# MAGIC
# MAGIC En este notebook se lleva a cabo el proceso de Feature Selection para obtener las variables que mayor impacto tienen sobre el target.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importando librerías

# COMMAND ----------

import time
import json

# COMMAND ----------

# MAGIC %run /Workspace/Users/coraimac@ucm.es/TFM/utils_functions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importando datos

# COMMAND ----------

container = "datos"
account = spark.conf.get("adls.account.name")
folder = "tfm"
path_features = f"abfss://{container}@{account}.dfs.core.windows.net/{folder}/data_preparation/features/"
path_target = f"abfss://{container}@{account}.dfs.core.windows.net/{folder}/data_preparation/target/"

# COMMAND ----------

# leyendo features generadas
feats_trx = spark.read.parquet(path_features + 'feats_trx')
feats_geo = spark.read.parquet(path_features + 'feats_geo')
feats_dialog = spark.read.parquet(path_features + 'feats_dialog')
feats_purchase = spark.read.parquet(path_features + 'feats_purchase')

# leyendo target (variable dependiente)
target_1 = spark.read.parquet(path_target + 'product_1/')
target_3 = spark.read.parquet(path_target + 'product_3/')
target_4 = spark.read.parquet(path_target + 'product_4/')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Construcción master table

# COMMAND ----------

# MAGIC %md
# MAGIC **Nota:** Denominamos master table a la tabla que consolida la variable dependiente y variables predictoras.

# COMMAND ----------

master_table_1 = target_1.join(
    feats_dialog, how='left', on=['client_id', 'mon']
).join(
    feats_trx, how='left', on=['client_id', 'mon']
).join(
    feats_geo, how='left', on=['client_id', 'mon']
).join(
    feats_purchase, how='left', on=['client_id', 'mon']
)

master_table_1 = master_table_1.cache()

# COMMAND ----------

master_table_3 = target_3.join(
    feats_dialog, how='left', on=['client_id', 'mon']
).join(
    feats_trx, how='left', on=['client_id', 'mon']
).join(
    feats_geo, how='left', on=['client_id', 'mon']
).join(
    feats_purchase, how='left', on=['client_id', 'mon']
)

master_table_3 = master_table_3.cache()

# COMMAND ----------

master_table_4 = target_4.join(
    feats_dialog, how='left', on=['client_id', 'mon']
).join(
    feats_trx, how='left', on=['client_id', 'mon']
).join(
    feats_geo, how='left', on=['client_id', 'mon']
).join(
    feats_purchase, how='left', on=['client_id', 'mon']
)

master_table_4 = master_table_4.cache()

# COMMAND ----------

# convirtiendo df a pandas
master_table_pd_1 = master_table_1.toPandas()
master_table_pd_3 = master_table_3.toPandas()
master_table_pd_4 = master_table_4.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Features continuas / categóricas

# COMMAND ----------

# obtencion de features numericas y categoricas
numerical_cols, categorical_cols = _identify_type_columns_pandas(
    df=master_table_pd_1,
    params={
        'id_columns': ['mon', 'client_id'],
        'target': ['target_1', 'target_3', 'target_4'],
        'features_exclude': {
            'numerical': ['trans_dst_type_mode_m', 'trans_event_type_mode_m'],
            'categorical': ['dialog_avg_per_day_m'],
        },
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Descarte de features

# COMMAND ----------

# MAGIC %md
# MAGIC ### Alta cantidad nulos/ceros

# COMMAND ----------

# obteniendo lista de features que tienen mas del 98% 
# de sus valores en nulos o zeros, dado que no aportan informacion
features_zeros = features_lot_of_zeros(
    df=master_table_pd_1,
    umbral= 0.98,
    params={
        'id_columns': ['client_id', 'mon'],
        'target': ['target_1', 'target_3', 'target_4'],
    }
)

# COMMAND ----------

# dropeando features con muchos ceros
master_table_pd_1 = master_table_pd_1.drop(columns=features_zeros)

# COMMAND ----------

# recalculando features numericas y categoricas
numerical_cols = [col for col in numerical_cols if col in master_table_pd_1.columns]
categorical_cols = [col for col in categorical_cols if col in master_table_pd_1.columns]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baja/Nula varianza

# COMMAND ----------

# obteniendo lista de features numéricas con varianza cero
# dado que no aportaran informacion al modelo
non_zero_var_columns, zero_var_columns = identify_zero_variances(
    df=master_table_pd_1,
    params={
        'id_columns': ['client_id', 'mon'],
        'target': ['target_1', 'target_2', 'target_3'],
        'numeric_columns': numerical_cols,
    }
)

# COMMAND ----------

# dropeando features con varianza cero
master_table_pd_1 = master_table_pd_1.drop(columns=zero_var_columns)

# COMMAND ----------

# recalculando features numericas y categoricas
numerical_cols = [col for col in numerical_cols if col in master_table_pd_1.columns]
categorical_cols = [col for col in categorical_cols if col in master_table_pd_1.columns]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Multicolinealidad

# COMMAND ----------

# MAGIC %md
# MAGIC Descartando features altamente correlacionadas usando VIF (Variance Inflation Factor): herramienta estadística utilizada en el análisis de regresión que mide en qué medida dos o más variables independientes están altamente correlacionadas entre sí.
# MAGIC
# MAGIC - VIF = 1 — No hay correlación entre la variable predictora y otras variables.
# MAGIC - 1 < VIF < 5 — Correlación moderada; generalmente aceptable.
# MAGIC - VIF ≥ 5 — Indica multicolinealidad potencialmente problemática.
# MAGIC - VIF ≥ 10 — Indica multicolinealidad grave que puede requerir más investigación.

# COMMAND ----------

display(master_table_pd_1.select_dtypes(include='number').corr(method='spearman').reset_index())

# COMMAND ----------

display(master_table_pd_1.select_dtypes(include='number').corr().reset_index())

# COMMAND ----------

import mlflow
mlflow.autolog(disable=True)

# COMMAND ----------

id_columns = ['client_id', 'mon']
target = ['target_1', 'target_3', 'target_4']
features_exclude = ['trans_event_type_mode_m', 'trans_dst_type_mode_m']
features_vif = [
    col for col in master_table_pd_1.columns if col not in id_columns + target + features_exclude
]

params = {
    'id_columns': id_columns,
    'target': target,
    'exclude_cols': features_exclude,
    'fillna_default': 0,
}

vif_df = calcular_vif(
    df=master_table_pd_1,
    params=params,
)

vif_high = vif_df[vif_df['VIF'] > 5]
vif_bool = True if vif_high.shape[0] > 0 else False

while vif_bool:
    top1_vif_high = vif_high['feature'].values[0]
    value_vif_high = vif_high['VIF'].values[0]
    features_vif.remove(top1_vif_high)
    print(f'Feature eliminada: {top1_vif_high} con VIF: {value_vif_high:.5f}')

    vif_df = calcular_vif(df=master_table_pd_1[features_vif], params=params)
    vif_high = vif_df[vif_df['VIF'] > 5]
    vif_bool = True if vif_high.shape[0] > 0 else False

# COMMAND ----------

print(
    f'Cantidad features seleccionadas tras descarte por VIF: {len(features_vif)}\n{features_vif}', 
)

# COMMAND ----------

# calculando VIF de features numericas resultantes
vif_df = calcular_vif(master_table_pd_1[features_vif])

display(vif_df)

# COMMAND ----------

# master table tras descarte de features por:
# - nulidad
# - varianza cero
# - multicolinealidad
master_table_pd_1 = master_table_pd_1[id_columns + ['target_1'] + features_vif + features_exclude]
master_table_pd_3 = master_table_pd_3[id_columns + ['target_3'] + features_vif + features_exclude]
master_table_pd_4 = master_table_pd_4[id_columns + ['target_4'] + features_vif + features_exclude]

print(
    f'Cantidad de columnas tras descarte de features: {master_table_pd_1.shape[1]}.'
)

# COMMAND ----------

# recalculando features numericas y categoricas
numerical_cols = [col for col in numerical_cols if col in master_table_pd_1.columns]
categorical_cols = [col for col in categorical_cols if col in master_table_pd_1.columns]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Selection: Producto 1

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Anova
# MAGIC La prueba ANOVA (Análisis de Varianza) es una prueba estadística utilizada para comparar las medias de dos o más grupos. Su objetivo principal es determinar si existen diferencias significativas entre las medias de estos grupos, o si las diferencias observadas son simplemente resultado de la variabilidad dentro de cada grupo. 

# COMMAND ----------

anova_df = anova_test(
    master_table_pd_1, 
    target_col='target_1', 
    feature_cols=numerical_cols
)

anova_df = anova_df.reset_index()
anova_df.columns = ['features', 'anova_f', 'p_value']

# COMMAND ----------

display(anova_df)

# COMMAND ----------

# considerando solo variables con un nivel de significacion del 95%
anova_nums = anova_df[anova_df['p_value'] < 0.05]

anova_nums

# COMMAND ----------

# MAGIC %md
# MAGIC ### V-Cramer
# MAGIC Referencial de la interpretación del V-Cramer: La V de Cramer mide la fuerza de asociación entre dos variables categóricas. Sus valores oscilan entre 0 y 1, donde 0 indica ausencia de asociación y 1, asociación perfecta. La interpretación de la magnitud del valor V proporciona información sobre la fuerza de la relación.
# MAGIC
# MAGIC | v cramer | interpretación |
# MAGIC |------------|--------------|
# MAGIC | < 0.01 | insignificante |
# MAGIC | 0.01 - 0.09 | muy débil |
# MAGIC | 0.10 - 0.29 | débil |
# MAGIC | 0.30 - 0.49 | moderada |
# MAGIC | 0.50 - 0.69 | fuerte |
# MAGIC | > 0.70 | muy fuerte |

# COMMAND ----------

v_cramer_df = v_cramer_cats(
    df=master_table_pd_1,
    params={
        'target': 'target_1',
        'categorical_features': categorical_cols,
    }
)

# COMMAND ----------

display(v_cramer_df)

# COMMAND ----------

# considerando solo features con v_cramer > 0.10 (debil)
v_cramer_cat = v_cramer_df[
    v_cramer_df['v_cramer'] > 0.01
]

v_cramer_cat

# COMMAND ----------

# MAGIC %md
# MAGIC ### Information Value
# MAGIC
# MAGIC Determina el poder predictivo de una variable.
# MAGIC
# MAGIC | IV | interpretación |
# MAGIC |------------|--------------|
# MAGIC | < 0.02 | sin poder predictivo |
# MAGIC | [0.02, 0.10) | débil poder predictivo |
# MAGIC | [0.10, 0.30) | medio poder predictivo |
# MAGIC | [0.30, 0.50) | fuerte poder predictivo |
# MAGIC | > 0.50 | sospechoso |

# COMMAND ----------

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

iv_df = get_iv_dataframe(
  df=master_table_pd_1,
  target_col='target_1',
  feature_cols=numerical_cols+categorical_cols,
  categorical_cols=categorical_cols,
  bins=25
)

# COMMAND ----------

display(iv_df)

# COMMAND ----------

# tomando variables con poder predictivo > 0.02 (debil)
iv_feats = iv_df[
    iv_df['IV'] > 0.02
]

iv_feats

# COMMAND ----------

# MAGIC %md
# MAGIC ### Selección

# COMMAND ----------

feats_product_1 = list(set(anova_nums['features'].values) \
                  .union(set(v_cramer_cat['categorical_features'].values)) \
                  .union(set(iv_feats['features'].values)))

print(
    f'Features seleccionadas para modelado en Porducto 1: {len(feats_product_1)}.\n'
    f'{feats_product_1}'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Selection: Producto 3

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Anova

# COMMAND ----------

anova_df = anova_test(
    master_table_pd_3, 
    target_col='target_3', 
    feature_cols=numerical_cols
)

anova_df = anova_df.reset_index()
anova_df.columns = ['features', 'anova_f', 'p_value']

# COMMAND ----------

display(anova_df)

# COMMAND ----------

# considerando solo variables con un nivel de significacion del 95%
anova_nums = anova_df[anova_df['p_value'] < 0.05]

anova_nums

# COMMAND ----------

# MAGIC %md
# MAGIC ### V-Cramer

# COMMAND ----------

v_cramer_df = v_cramer_cats(
    df=master_table_pd_3,
    params={
        'target': 'target_3',
        'categorical_features': categorical_cols,
    }
)

# COMMAND ----------

display(v_cramer_df)

# COMMAND ----------

# considerando solo features con v_cramer > 0.10 (debil)
v_cramer_cat = v_cramer_df[
    v_cramer_df['v_cramer'] > 0.01
]

v_cramer_cat

# COMMAND ----------

# MAGIC %md
# MAGIC ### Information Value

# COMMAND ----------

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

iv_df = get_iv_dataframe(
  df=master_table_pd_3,
  target_col='target_3',
  feature_cols=numerical_cols+categorical_cols,
  categorical_cols=categorical_cols,
  bins=25
)

# COMMAND ----------

display(iv_df)

# COMMAND ----------

# tomando variables con poder predictivo > 0.02 (debil)
iv_feats = iv_df[
    iv_df['IV'] > 0.02
]

iv_feats

# COMMAND ----------

# MAGIC %md
# MAGIC ### Selección

# COMMAND ----------

feats_product_3 = list(set(anova_nums['features'].values) \
                  .union(set(v_cramer_cat['categorical_features'].values)) \
                  .union(set(iv_feats['features'].values)))

print(
    f'Features seleccionadas para modelado en Porducto 3: {len(feats_product_3)}.\n'
    f'{feats_product_3}'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Selection: Producto 4

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Anova

# COMMAND ----------

anova_df = anova_test(
    master_table_pd_4, 
    target_col='target_4', 
    feature_cols=numerical_cols
)

anova_df = anova_df.reset_index()
anova_df.columns = ['features', 'anova_f', 'p_value']

# COMMAND ----------

display(anova_df)

# COMMAND ----------

# considerando solo variables con un nivel de significacion del 95%
anova_nums = anova_df[anova_df['p_value'] < 0.05]

anova_nums

# COMMAND ----------

# MAGIC %md
# MAGIC ### V-Cramer

# COMMAND ----------

v_cramer_df = v_cramer_cats(
    df=master_table_pd_4,
    params={
        'target': 'target_4',
        'categorical_features': categorical_cols,
    }
)

# COMMAND ----------

display(v_cramer_df)

# COMMAND ----------

# considerando solo features con v_cramer > 0.10 (debil)
v_cramer_cat = v_cramer_df[
    v_cramer_df['v_cramer'] > 0.01
]

v_cramer_cat

# COMMAND ----------

# MAGIC %md
# MAGIC ### Information Value

# COMMAND ----------

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

iv_df = get_iv_dataframe(
  df=master_table_pd_4,
  target_col='target_4',
  feature_cols=numerical_cols+categorical_cols,
  categorical_cols=categorical_cols,
  bins=25
)

# COMMAND ----------

display(iv_df)

# COMMAND ----------

# tomando variables con poder predictivo > 0.02 (debil)
iv_feats = iv_df[
    iv_df['IV'] > 0.02
]

iv_feats

# COMMAND ----------

# MAGIC %md
# MAGIC ### Selección

# COMMAND ----------

feats_product_4 = list(set(anova_nums['features'].values) \
                  .union(set(v_cramer_cat['categorical_features'].values)) \
                  .union(set(iv_feats['features'].values)))

print(
    f'Features seleccionadas para modelado en Porducto 4: {len(feats_product_4)}.\n'
    f'{feats_product_4}'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Guardado features

# COMMAND ----------

# features seleccionadas por producto
features_seleccionadas = {
    'product_1': feats_product_1,
    'product_3': feats_product_3,
    'product_4': feats_product_4,
}

# COMMAND ----------

# guardando archivo json en nube
dbutils.fs.put(
    f"{path_features}features_by_product.json",
    json.dumps(features_seleccionadas, indent=2, sort_keys=True),
    overwrite=True
)