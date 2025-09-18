# Databricks notebook source
# MAGIC %md
# MAGIC # TFM: Next Move Banking
# MAGIC **Autor:** Coraima Castillo
# MAGIC
# MAGIC En este notebook se lleva a cabo el proceso de inferencia, el cual es simulado y ejemplificado con la data disponible en el último corte.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importando librerías

# COMMAND ----------

import pyspark.sql.functions as f
import json
import time

import mlflow

# COMMAND ----------

# MAGIC %run /Workspace/Users/coraimac@ucm.es/TFM/utils_functions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importando datos

# COMMAND ----------

container = "datos"
account = spark.conf.get("adls.account.name")
folder = "tfm"
path = f"abfss://{container}@{account}.dfs.core.windows.net/{folder}/"

# COMMAND ----------

customers_inference = spark.read.parquet(path + 'raw_data/purchase').withColumn(
    'mon', f.to_date(f.date_format(f.col('mon'), 'yyyy-MM-01'), 'yyyy-MM-dd')
)
last_mon = customers_inference.select(f.max('mon').alias('mon'))

customers_inference = customers_inference.join(last_mon, on=['mon'], how='inner')

print(
    f'Cantidad clientes para inferencia: {customers_inference.select("client_id").distinct().count():,}'
)

del last_mon

# COMMAND ----------

feats_trx = spark.read.parquet(path + 'data_preparation/features/feats_trx')
feats_geo = spark.read.parquet(path + 'data_preparation/features/feats_geo')
feats_dialog = spark.read.parquet(path + 'data_preparation/features/feats_dialog')
feats_purchase = spark.read.parquet(path + 'data_preparation/features/feats_purchase')

# COMMAND ----------

# leeyendo el archivo completo de features seleccionadas
path_features_selected = f"{path}/data_preparation/features/features_by_product.json"
raw = dbutils.fs.head(path_features_selected)
features_by_product = json.loads(raw)


feats_product_1 = features_by_product['product_1']
feats_product_3 = features_by_product['product_3']
feats_product_4 = features_by_product['product_4']

features_selected = list(set(feats_product_1 + feats_product_3 + feats_product_4))

print(
    f'Total features entre todos los productos: {len(features_selected)}.'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creando master table de inferencia

# COMMAND ----------

master_table_inference = customers_inference.join(
    feats_dialog, how='left', on=['client_id', 'mon']
).join(
    feats_trx, how='left', on=['client_id', 'mon']
).join(
    feats_geo, how='left', on=['client_id', 'mon']
).join(
    feats_purchase, how='left', on=['client_id', 'mon']
)

id_columns = ['client_id', 'mon']
master_table_inference = master_table_inference.select(
    *id_columns,
    *features_selected,
)

master_table_inference = master_table_inference.cache()

# COMMAND ----------

# guardando master table de inferencia
start_time = time.time()

master_table_inference.write.format('parquet').mode('overwrite').save(path + 'inference/master_table/')

end_time = time.time()
final_time = end_time - start_time

print(
    f'Tiempo de guardado: {final_time / 60:.2f} min.'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importando modelos

# COMMAND ----------

latest_version = get_latest_model_version("model_product_1")
model_uri = f"models:/model_product_1/{latest_version}"  
model_product_1 = mlflow.sklearn.load_model(model_uri)

print(
    f'Versión del modelo utilizada para Producto 1: {latest_version}'
)

# COMMAND ----------

latest_version = get_latest_model_version("model_product_3")
model_uri = f"models:/model_product_3/{latest_version}"   
model_product_3 = mlflow.sklearn.load_model(model_uri)

print(
    f'Versión del modelo utilizada para Producto 3: {latest_version}'
)

# COMMAND ----------

latest_version = get_latest_model_version("model_product_4")
model_uri = f"models:/model_product_4/{latest_version}"  
model_product_4 = mlflow.sklearn.load_model(model_uri)

print(
    f'Versión del modelo utilizada para Producto 4: {latest_version}'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predicciones

# COMMAND ----------

# leyendo master table de inferencia
master_table_inference = spark.read.parquet(path + 'inference/master_table/').toPandas()

# COMMAND ----------

y_pred_1, y_score_1 = model_product_1.predict(master_table_inference)
y_pred_3, y_score_3 = model_product_3.predict(master_table_inference)
y_pred_4, y_score_4 = model_product_4.predict(master_table_inference)

# COMMAND ----------

df_predictions_1 = pd.DataFrame({
    'mon': master_table_inference['mon'],
    'client_id': master_table_inference['client_id'],
    'product_id': 1,
    'name_product': 'product_1',
    'prob': y_score_1,
    'pred': y_pred_1,
})

df_predictions_3 = pd.DataFrame({
    'mon': master_table_inference['mon'],
    'client_id': master_table_inference['client_id'],
    'product_id': 3,
    'name_product': 'product_3',
    'prob': y_score_3,
    'pred': y_pred_3,
})

df_predictions_4 = pd.DataFrame({
    'mon': master_table_inference['mon'],
    'client_id': master_table_inference['client_id'],
    'product_id': 4,
    'name_product': 'product_4',
    'prob': y_score_4,
    'pred': y_pred_4,
})

df_predictions_1 = spark.createDataFrame(df_predictions_1)
df_predictions_3 = spark.createDataFrame(df_predictions_3)
df_predictions_4 = spark.createDataFrame(df_predictions_4)

df_predictions = df_predictions_1.unionByName(df_predictions_3)\
                                 .unionByName(df_predictions_4)

df_predictions = df_predictions.orderBy(['client_id', 'mon'], ascending=True)

# COMMAND ----------

display(df_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Guardado

# COMMAND ----------

# guardando pr
start_time = time.time()

df_predictions.write.format('parquet').mode('overwrite').save(path + 'inference/predictions/')

end_time = time.time()
final_time = end_time - start_time

print(
    f'Tiempo de guardado: {final_time / 60:.2f} min.'
)