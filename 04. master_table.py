# Databricks notebook source
# MAGIC %md
# MAGIC # TFM: Next Move Banking
# MAGIC **Autor:** Coraima Castillo
# MAGIC
# MAGIC En este notebook se lleva a cabo el proceso de construcción de Master Table con las features seleccionadas.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importando librerías

# COMMAND ----------

import time
import json

from pyspark.sql.window import Window
import pyspark.sql.functions as f

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

# leeyendo el archivo completo de features seleccionadas
path_features_selected = f"{path_features}features_by_product.json"
raw = dbutils.fs.head(path_features_selected)
features_by_product = json.loads(raw)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Construcción master table general

# COMMAND ----------

# meses de train y test
total_meses = target_1.select('mon').distinct().count()
meses_test = int(total_meses * 0.30)
meses_train = total_meses - meses_test

print(
    f'Para modelado y validación temporal se tienen {total_meses} meses.\n'
    f'Meses para training (70% meses): {meses_train}.\n'
    f'Meses para testing (30% meses): {meses_test}.'
)



# COMMAND ----------

# mes donde empieza test
last_month = target_1.select(f.max('mon').alias('last_month')) \
                   .withColumn('start_test', f.add_months('last_month', -meses_test))
start_test = last_month.select('start_test').first()[0]

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

master_table_3 = target_3.join(
    feats_dialog, how='left', on=['client_id', 'mon']
).join(
    feats_trx, how='left', on=['client_id', 'mon']
).join(
    feats_geo, how='left', on=['client_id', 'mon']
).join(
    feats_purchase, how='left', on=['client_id', 'mon']
)

master_table_4 = target_4.join(
    feats_dialog, how='left', on=['client_id', 'mon']
).join(
    feats_trx, how='left', on=['client_id', 'mon']
).join(
    feats_geo, how='left', on=['client_id', 'mon']
).join(
    feats_purchase, how='left', on=['client_id', 'mon']
)

master_table_1 = master_table_1.cache()
master_table_3 = master_table_3.cache()
master_table_4 = master_table_4.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Master table: Producto 1

# COMMAND ----------

# features para producto 1
feats_product_1 = features_by_product['product_1']

id_columns = ['client_id', 'mon']
target = 'target_1'

master_table_1 = master_table_1.select(
    *id_columns,
    *feats_product_1,
    target
)

master_table_1 = master_table_1.cache()

# COMMAND ----------

# division entre train-test
train_1 = master_table_1.filter(f.col('mon') <= start_test)
test_1 = master_table_1.filter(f.col('mon') > start_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Master table: Producto 3

# COMMAND ----------

# features para producto 3
feats_product_3 = features_by_product['product_3']

id_columns = ['client_id', 'mon']
target = 'target_3'

master_table_3 = master_table_3.select(
    *id_columns,
    *feats_product_3,
    target
)

master_table_3 = master_table_3.cache()

# COMMAND ----------

# division entre train-test
train_3 = master_table_3.filter(f.col('mon') <= start_test)
test_3 = master_table_3.filter(f.col('mon') > start_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Master table: Producto 4

# COMMAND ----------

# features para producto 4
feats_product_4 = features_by_product['product_4']

id_columns = ['client_id', 'mon']
target = 'target_4'

master_table_4 = master_table_4.select(
    *id_columns,
    *feats_product_4,
    target
)

master_table_4 = master_table_4.cache()

# COMMAND ----------

# division entre train-test
train_4 = master_table_4.filter(f.col('mon') <= start_test)
test_4 = master_table_4.filter(f.col('mon') > start_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Guardado

# COMMAND ----------

# distribucion de train-test de las tablas
total = master_table_4.count()
test_size = test_4.count()
train_size = train_4.count()

print(
    f'Total registros: {total:,}\n'
    f'Tamaño train: {train_size:,} ({train_size / total * 100:.2f}%)\n'
    f'Tamaño test: {test_size:,} ({test_size / total * 100:.2f}%)\n'
)

# COMMAND ----------

container = "datos"
account = spark.conf.get("adls.account.name")
folder = "tfm"
path_table = f"abfss://{container}@{account}.dfs.core.windows.net/{folder}/data_preparation/master_table/"

print(
    f'Path de almacenamiento features: {path_table}.'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Producto 1

# COMMAND ----------

# guardando train y test table producto 1
start_time = time.time()

product = 'product_1'
train_1.write.format('parquet').mode('overwrite').save(path_table + product + '/train_table')
test_1.write.format('parquet').mode('overwrite').save(path_table + product + '/test_table')

end_time = time.time()
final_time = end_time - start_time

print(
    f'Tiempo de guardado: {final_time / 60:.2f} min.'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Producto 3

# COMMAND ----------

# guardando train y test table producto 3
start_time = time.time()

product = 'product_3'
train_3.write.format('parquet').mode('overwrite').save(path_table + product + '/train_table')
test_3.write.format('parquet').mode('overwrite').save(path_table + product + '/test_table')

end_time = time.time()
final_time = end_time - start_time

print(
    f'Tiempo de guardado: {final_time / 60:.2f} min.'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Producto 4

# COMMAND ----------

# guardando train y test table producto 4
start_time = time.time()

product = 'product_4'
train_4.write.format('parquet').mode('overwrite').save(path_table + product + '/train_table')
test_4.write.format('parquet').mode('overwrite').save(path_table + product + '/test_table')

end_time = time.time()
final_time = end_time - start_time

print(
    f'Tiempo de guardado: {final_time / 60:.2f} min.'
)