# Databricks notebook source
# MAGIC %md
# MAGIC # TFM: Next Move Banking
# MAGIC **Autor:** Coraima Castillo
# MAGIC
# MAGIC En este notebook se desarrollan diferentes features para la etapa del modelado.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importando librerías

# COMMAND ----------

# librerias basicas
from pyspark.sql.window import Window
import pyspark.sql.functions as f
import time

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importando datos

# COMMAND ----------

container = "datos"
account = spark.conf.get("adls.account.name")
folder = "tfm"
path = f"abfss://{container}@{account}.dfs.core.windows.net/{folder}/raw_data/"

# COMMAND ----------

# df de transacciones
df_trx = spark.read.parquet(path + 'transactions')

# df de geolocalizaciones
df_geo = spark.read.parquet(path + 'geolocation')

# df de dialogos
df_dialog = spark.read.parquet(path + 'dialog')

# df de compras / movimientos transaccionales
df_purchase = spark.read.parquet(path + 'purchase')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC ### Features transaccionales

# COMMAND ----------

# MAGIC %md
# MAGIC Acorde al EDA realizado, el top 3 de monedas en donde frecuentan las transacciones en monto / número son las siguientes: 11, 15, 3.

# COMMAND ----------

# normalizando fechas de transacciones para agregaciones mensuales
df_trx = df_trx.withColumns({
    'mon': f.to_date(f.date_format(f.col('event_time'), 'yyyy-MM-01'), 'yyyy-MM-dd'),
    'event_time': f.to_date(f.col('event_time'), 'yyyy-MM-dd')
})

# COMMAND ----------

# agregando nivel cliente - mes
feats_trx = df_trx.withColumns({
    'id_currency': f.when(f.col('currency').isin([11, 15, 3]), f.col('currency')).otherwise(0),
    'cnt_trx': f.lit(1),
})

feats_trx = feats_trx.groupBy('client_id', 'mon').agg(
    f.count_distinct(f.col('event_time')).alias('trans_days_active_m'),
    f.sum(f.col('cnt_trx')).alias('trans_trx_cnt_m'),
    f.count_distinct('event_type').alias('trans_type_trx_cnt_m'),
    f.count_distinct(f.col('currency')).alias('trans_type_currency_cnt_m'),
    f.count_distinct(f.col('src_type11')).alias('trans_type_src_account_cnt_m'),    
    f.count_distinct(f.col('dst_type11')).alias('trans_type_dst_account_cnt_m'),
    f.sum(
        f.when(f.col('id_currency') == 11, f.col('cnt_trx')
    ).otherwise(0)).alias('trans_trx_currency_11_cnt_m'),
    f.sum(
        f.when(f.col('id_currency') == 15, f.col('cnt_trx')
    ).otherwise(0)).alias('trans_trx_currency_15_cnt_m'),
    f.sum(
        f.when(f.col('id_currency') == 3, f.col('cnt_trx')
    ).otherwise(0)).alias('trans_trx_currency_3_cnt_m'),
    f.sum(
        f.when(f.col('id_currency') == 0, f.col('cnt_trx')
    ).otherwise(0)).alias('trans_trx_others_currency_cnt_m'),
    f.sum(
        f.when(f.col('id_currency') == 11, f.col('amount')
    ).otherwise(0)).alias('trans_amt_trx_currency_11_sum_m'),
    f.sum(
        f.when(f.col('id_currency') == 15, f.col('amount')
    ).otherwise(0)).alias('trans_amt_trx_currency_15_sum_m'),
    f.sum(
        f.when(f.col('id_currency') == 3, f.col('amount')
    ).otherwise(0)).alias('trans_amt_trx_currency_3_sum_m'),
    f.avg(
        f.when(f.col('id_currency') == 11, f.col('amount'))
    ).alias('trans_amt_currency_11_avg_m'),
    f.std(
        f.when(f.col('id_currency') == 11, f.col('amount'))
    ).alias('trans_amt_currency_11_std_m')
)

# COMMAND ----------

# calculando moda/entropia por tipo de evento
df_event = df_trx.groupBy('client_id', 'mon', 'event_type').agg(
    f.count('*').alias('event_cnt')
).withColumn(
    'event_total', f.sum(f.col('event_cnt')).over(Window.partitionBy('client_id', 'mon'))
).withColumn(
    'prop', f.col('event_cnt') / f.col('event_total')
)

# generando ranking para determinar el tipo de evento más frecuentado
window = Window.partitionBy('client_id', 'mon').orderBy(f.desc('event_cnt'), f.asc('event_type'))
df_event = df_event.withColumn('row_number', f.row_number().over(window))

# df con el tipo de evento más frecuente: moda
df_event_mode = df_event.filter(f.col('row_number') == 1).select(
    'client_id', 'mon',
    f.col('event_type').alias('trans_event_type_mode_m'),
)

# df con la entropia del tipo de evento
df_event_entropy = df_event.withColumn('term', -f.col('prop') * f.log(f.col('prop'))) \
                       .groupBy('client_id', 'mon') \
                       .agg(f.sum('term').alias('trans_event_type_entropy_m'))

# consolidacion dfs
df_mode_entropy_event = df_event_mode.join(
    df_event_entropy, on=['client_id', 'mon'], how='left'
)

del df_event, df_event_mode, df_event_entropy

# COMMAND ----------

# calculando moda/entropia por tipo de destinatario
df_dst = df_trx.groupBy('client_id', 'mon', 'dst_type11').agg(
    f.count('*').alias('dst_cnt')
).withColumn(
    'dst_total', f.sum(f.col('dst_cnt')).over(Window.partitionBy('client_id', 'mon'))
).withColumn(
    'prop', f.col('dst_cnt') / f.col('dst_total')
)

# generando ranking para determinar el tipo de destinatario más frecuentado
window = Window.partitionBy('client_id', 'mon').orderBy(f.desc('dst_cnt'), f.asc('dst_type11'))
df_dst = df_dst.withColumn('row_number', f.row_number().over(window))

# df con el tipo de dsto más frecuente: moda
df_dst_mode = df_dst.filter(f.col('row_number') == 1).select(
    'client_id', 'mon',
    f.col('dst_type11').alias('trans_dst_type_mode_m'),
)

# df con la entropia del tipo de dsto
df_dst_entropy = df_dst.withColumn('term', -f.col('prop') * f.log(f.col('prop'))) \
                       .groupBy('client_id', 'mon') \
                       .agg(f.sum('term').alias('trans_dst_type_entropy_m'))

# consolidacion dfs
df_mode_entropy_dst = df_dst_mode.join(
    df_dst_entropy, on=['client_id', 'mon'], how='left'
)

del df_dst, df_dst_mode, df_dst_entropy, df_trx

# COMMAND ----------

# consolidando features agregadas y featues de moda/entropia
feats_trx = feats_trx.join(df_mode_entropy_event, on=['client_id', 'mon'], how='left')\
                     .join(df_mode_entropy_dst, on=['client_id', 'mon'], how='left')

del df_mode_entropy_dst, df_mode_entropy_event

# COMMAND ----------

window = Window.partitionBy('client_id').orderBy('mon')

# calculando diferentes indicadores
feats_trx = feats_trx.withColumns({
    'trans_trx_per_day_m': f.col('trans_trx_cnt_m') / f.col('trans_days_active_m'),
    'trans_dst_type_entropy_ravg_3m': f.avg('trans_dst_type_entropy_m').over(window.rowsBetween(-3 + 1, 0)),
    'trans_event_type_entropy_ravg_3m': f.avg('trans_event_type_entropy_m').over(window.rowsBetween(-3 + 1, 0)),
    'trans_trx_cnt_roc_1m': (f.col('trans_trx_cnt_m') / f.lag(f.col('trans_trx_cnt_m'), 1).over(window)) - f.lit(1),
    'trans_trx_cnt_roc_3m': (f.col('trans_trx_cnt_m') / f.lag(f.col('trans_trx_cnt_m'), 3).over(window)) - f.lit(1),
    'trans_trx_cnt_roc_6m': (f.col('trans_trx_cnt_m') / f.lag(f.col('trans_trx_cnt_m'), 6).over(window)) - f.lit(1),
    'trans_trx_cnt_roc_12m': (f.col('trans_trx_cnt_m') / f.lag(f.col('trans_trx_cnt_m'), 12).over(window)) - f.lit(1),
    'trans_days_active_rstd_6m': f.std(f.col('trans_days_active_m')).over(window.rowsBetween(-6 + 1, 0)),
    'trans_ind_trx_multicurrency': f.when(f.col('trans_type_currency_cnt_m') > 1, 1).otherwise(0),
})

# calculando indicador de crecimiento en 3 meses consecutivos
feats_trx = feats_trx.withColumn('trx_cnt_delta', f.col('trans_trx_cnt_m') - f.lag('trans_trx_cnt_m', 1).over(window))
feats_trx = feats_trx.withColumn('trans_growth_flag', f.when(f.col('trx_cnt_delta') > 0, 1).otherwise(0))

feats_trx = feats_trx.withColumn('trans_growth_streak3', f.sum('trans_growth_flag').over(window.rowsBetween(-3 + 1, 0)))\
                     .withColumn('trx_growth_streak_3m', f.when(f.col('trans_growth_streak3') == 3, 1).otherwise(0))

feats_trx = feats_trx.drop(
    'trx_cnt_delta',
    'trans_growth_flag',
    'trans_growth_streak3',
)

# COMMAND ----------

# desplazando las features un mes para evitar data leakeage
# de este modo se usarán feautures solo del mes cerrado (simulando producción)
feats_trx = feats_trx.withColumn('mon', f.add_months(f.col('mon'), 1))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Features geolocalización

# COMMAND ----------

# normalizando fechas de usos de app para agregaciones mensuales
df_geo = df_geo.withColumns({
    'mon': f.to_date(f.date_format(f.col('event_time'), 'yyyy-MM-01'), 'yyyy-MM-dd'),
    'event_time': f.to_date(f.col('event_time'), 'yyyy-MM-dd'),
    'is_night': f.when(
        (f.hour(f.col('event_time')) >= 18) | (f.hour(f.col('event_time')) < 6), 1
    ).otherwise(0),
    'is_day': f.when(
        (f.hour(f.col('event_time')) >= 6) & (f.hour(f.col('event_time')) < 18), 1
    ).otherwise(0),
})

# COMMAND ----------

feats_geo = df_geo.groupBy('client_id', 'mon').agg(
    f.count('*').alias('geo_ping_cnt_m'),
    f.count_distinct('event_time').alias('geo_days_active_m'),
    f.sum(f.col('is_day')).alias('geo_day_cnt_m'),
    f.sum(f.col('is_night')).alias('geo_night_cnt_m'),
    f.count_distinct('geohash_4').alias('geo_geohash_4_cnt_m'),
    f.count_distinct('geohash_5').alias('geo_geohash_5_cnt_m'),
    f.count_distinct('geohash_6').alias('geo_geohash_6_cnt_m'),
)

# COMMAND ----------

# calculando entropia por geohash 4
df_geohash_4 = df_geo.groupBy('client_id', 'mon', 'geohash_4').agg(
    f.count('*').alias('geohash_4_cnt')
).withColumn(
    'geohash_4_total', f.sum(f.col('geohash_4_cnt')).over(Window.partitionBy('client_id', 'mon'))
).withColumn(
    'prop', f.col('geohash_4_cnt') / f.col('geohash_4_total')
)

# generando ranking para determinar el tipo de destinatario más frecuentado
window = Window.partitionBy('client_id', 'mon').orderBy(f.desc('geohash_4_cnt'), f.asc('geohash_4'))
df_geohash_4 = df_geohash_4.withColumn('row_number', f.row_number().over(window))

# df con la entropia del tipo de dsto
df_geohash_4_entropy = df_geohash_4.withColumn('term', -f.col('prop') * f.log(f.col('prop'))) \
                       .groupBy('client_id', 'mon') \
                       .agg(f.sum('term').alias('geo_geohash_4_entropy_m'))

del df_geohash_4

# COMMAND ----------

# calculando entropia por geohash 5
df_geohash_5 = df_geo.groupBy('client_id', 'mon', 'geohash_5').agg(
    f.count('*').alias('geohash_5_cnt')
).withColumn(
    'geohash_5_total', f.sum(f.col('geohash_5_cnt')).over(Window.partitionBy('client_id', 'mon'))
).withColumn(
    'prop', f.col('geohash_5_cnt') / f.col('geohash_5_total')
)

# generando ranking para determinar el tipo de destinatario más frecuentado
window = Window.partitionBy('client_id', 'mon').orderBy(f.desc('geohash_5_cnt'), f.asc('geohash_5'))
df_geohash_5 = df_geohash_5.withColumn('row_number', f.row_number().over(window))

# df con la entropia del tipo de dsto
df_geohash_5_entropy = df_geohash_5.withColumn('term', -f.col('prop') * f.log(f.col('prop'))) \
                       .groupBy('client_id', 'mon') \
                       .agg(f.sum('term').alias('geo_geohash_5_entropy_m'))

del df_geohash_5

# COMMAND ----------

# calculando moda/entropia por geohash 6
df_geohash_6 = df_geo.groupBy('client_id', 'mon', 'geohash_6').agg(
    f.count('*').alias('geohash_6_cnt')
).withColumn(
    'geohash_6_total', f.sum(f.col('geohash_6_cnt')).over(Window.partitionBy('client_id', 'mon'))
).withColumn(
    'prop', f.col('geohash_6_cnt') / f.col('geohash_6_total')
)

# generando ranking para determinar el tipo de destinatario más frecuentado
window = Window.partitionBy('client_id', 'mon').orderBy(f.desc('geohash_6_cnt'), f.asc('geohash_6'))
df_geohash_6 = df_geohash_6.withColumn('row_number', f.row_number().over(window))

# df con la entropia del tipo de dsto
df_geohash_6_entropy = df_geohash_6.withColumn('term', -f.col('prop') * f.log(f.col('prop'))) \
                       .groupBy('client_id', 'mon') \
                       .agg(f.sum('term').alias('geo_geohash_6_entropy_m'))

del df_geohash_6, df_geo

# COMMAND ----------

# integrando features con entropia
feats_geo = feats_geo.join(
    df_geohash_4_entropy, on=['client_id', 'mon'], how='left'
).join(
    df_geohash_5_entropy, on=['client_id', 'mon'], how='left'
).join(
    df_geohash_6_entropy, on=['client_id', 'mon'], how='left'
)

del df_geohash_4_entropy, df_geohash_5_entropy, df_geohash_6_entropy

# COMMAND ----------

window = Window.partitionBy('client_id').orderBy('mon')
feats_geo = feats_geo.withColumns({
    'geo_days_active_avg_6m': f.avg(f.col('geo_days_active_m')).over(window.rowsBetween(-6 + 1, 0)),
    'geo_geohash_4_rstd_6m': f.std(f.col('geo_geohash_4_entropy_m')).over(window.rowsBetween(-6 + 1, 0)),
    'geo_geohash_5_rstd_6m': f.std(f.col('geo_geohash_5_entropy_m')).over(window.rowsBetween(-6 + 1, 0)),
    'geo_geohash_6_rstd_6m': f.std(f.col('geo_geohash_6_entropy_m')).over(window.rowsBetween(-6 + 1, 0)),
    'geo_geohash_6_roc_1m': (f.col('geo_geohash_6_cnt_m') / f.lag(f.col('geo_geohash_6_cnt_m'), 1).over(window)) - f.lit(1), 
    'geo_geohash_6_roc_3m': (f.col('geo_geohash_6_cnt_m') / f.lag(f.col('geo_geohash_6_cnt_m'), 3).over(window)) - f.lit(1), 
    'geo_geohash_6_roc_6m': (f.col('geo_geohash_6_cnt_m') / f.lag(f.col('geo_geohash_6_cnt_m'), 6).over(window)) - f.lit(1),
    'geo_usage_per_day_m': f.col('geo_ping_cnt_m') / f.col('geo_days_active_m'),
})

# COMMAND ----------

# desplazando las features un mes para evitar data leakeage
# de este modo se usarán features solo del mes cerrado (simulando producción)
feats_geo = feats_geo.withColumn('mon', f.add_months(f.col('mon'), 1))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Features diálogos

# COMMAND ----------

# normalizando fechas de dialogos para agregaciones mensuales
df_dialog = df_dialog.withColumns({
    'mon': f.to_date(f.date_format(f.col('event_time'), 'yyyy-MM-01'), 'yyyy-MM-dd'),
    'event_time_dt': f.to_date(f.col('event_time'), 'yyyy-MM-dd'),
})

# COMMAND ----------

feats_dialog = df_dialog.groupBy('client_id', 'mon').agg(
    f.count("*").alias("dialog_cnt_m"),
    f.count_distinct(f.col('event_time')).alias('dialog_days_active_m'),
)

# COMMAND ----------

# fetures globales de embeddings
df_embeddings = df_dialog.withColumn(
    'v_sqsum',  f.aggregate(
        f.transform(f.col('embedding'), lambda x: x * x),
        f.lit(0.0),
        lambda acc, x: acc + x
    )
).withColumn(
    'v_norm', f.sqrt(f.col('v_sqsum'))
)

df_norm = df_embeddings.groupBy('client_id', 'mon').agg(
    f.avg(f.col('v_norm')).alias('dialog_norm_mean_m'),
    f.std(f.col('v_norm')).alias('dialog_norm_std_m'),
    f.median(f.col('v_norm')).alias('dialog_norm_median_m'),
)

# COMMAND ----------

# feature de drift temporal (distancia euclidiana entre primer y último embedding por mes)
window_1 = Window.partitionBy('client_id', 'mon').orderBy(f.col('event_time').asc())
window_2 = Window.partitionBy('client_id', 'mon').orderBy(f.col('event_time').desc())

df_first = df_dialog.withColumn('nv1', f.row_number().over(window_1))\
                    .filter(f.col('nv1') == 1)\
                    .select(
                        'client_id', 'mon', f.col('embedding').alias('first_emb')
                    )
df_last = df_dialog.withColumn('nv1', f.row_number().over(window_2))\
                    .filter(f.col('nv1') == 1)\
                    .select(
                        'client_id', 'mon', f.col('embedding').alias('last_emb')
                    )

df_drift = df_first.join(df_last, on=['client_id','mon'], how='inner')\
    .withColumn(
        'diff_sqsum',
        f.aggregate(
            f.transform(
                f.zip_with('first_emb', 'last_emb', lambda a, b: a - b),
                lambda x: x * x
            ),
            f.lit(0.0),
            lambda acc, x: acc + x
        )
    ).withColumn(
        'dialog_drift_m', f.sqrt(f.col('diff_sqsum')),
    ).select(
        'client_id', 'mon', 'dialog_drift_m',
    )

del df_first, df_last, df_dialog, window_1, window_2

# COMMAND ----------

# MAGIC %md
# MAGIC **Nota:** La distancia coseno mide qué tanto cambian el contenido de las conversaciones de un cliente mes a mes. Si la distancia cos tiende a 0, entonces son idénticos los vectores (temas de conversación), si tiende a 1 los vectores son muy diferentes (los vectores son ortogonales), y si es 2 entonces los vectores son totalmente opuestos (temas muy diferentes, 2 es la máxima disimilitud direccional posible).

# COMMAND ----------

# distancias cosenos entre vectores del mismo mes
# (para determinar similitud entre conversaciones)
window = Window.partitionBy('client_id', 'mon').orderBy('event_time_dt')
df_cosen = df_embeddings.withColumns({
    'embedding_lag_1m': f.lag(f.col('embedding'), 1).over(window),
    'v_norm_lag_1m': f.lag(f.col('v_norm'), 1).over(window),
}).filter(f.col('embedding_lag_1m').isNotNull())

df_cosen = df_cosen.withColumn(
    'dot_u_v_lag_1m',
    f.aggregate(
        f.zip_with('embedding', 'embedding_lag_1m', lambda x, y: x * y),
        f.lit(0.0),
        lambda acc, x: acc + x,
    )
)

df_cosen = df_cosen.withColumn(
    'cos_sim', f.col('dot_u_v_lag_1m') / (f.col('v_norm') * f.col('v_norm_lag_1m'))
).withColumn(
    'cos_dist', f.lit(1.0) - f.col('cos_sim')
)

df_cosen = df_cosen.groupBy('client_id', 'mon').agg(
    f.avg(f.col('cos_dist')).alias('dialog_cos_dist_mean_m'),
    f.std(f.col('cos_dist')).alias('dialog_cos_dist_std_m'),
    f.median(f.col('cos_dist')).alias('dialog_cos_dist_median_m'),
)

del df_embeddings

# COMMAND ----------

# consolidacion features generadas
feats_dialog = feats_dialog.join(
    df_norm, how='left', on=['client_id', 'mon']
).join(
    df_drift, how='left', on=['client_id', 'mon']
).join(
    df_cosen, how='left', on=['client_id', 'mon']
)

del df_drift, df_norm, df_cosen

# COMMAND ----------

# features de comportamiento historico adicionales
window = Window.partitionBy('client_id').orderBy('mon')

feats_dialog = feats_dialog.withColumns({
    'dialog_avg_per_day_m': f.col('dialog_cnt_m') / f.col('dialog_days_active_m'),
    'dialog_drift_rstd_3m': f.std(f.col('dialog_drift_m')).over(window.rowsBetween(-3 + 1, 0)),
    'dialog_drift_rstd_6m': f.std(f.col('dialog_drift_m')).over(window.rowsBetween(-6 + 1, 0)),
    'dialog_recency_m': f.months_between(f.col('mon'), f.lag(f.col('mon'), 1).over(window)),
})

# COMMAND ----------

# desplazando las features un mes para evitar data leakeage
# de este modo se usarán feautures solo del mes cerrado (simulando producción)
feats_dialog = feats_dialog.withColumn('mon', f.add_months(f.col('mon'), 1))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Features compras

# COMMAND ----------

# normalizando fechas de compras para agregaciones mensuales
df_purchase = df_purchase.withColumns({
    'mon': f.to_date(f.date_format(f.col('mon'), 'yyyy-MM-01'), 'yyyy-MM-dd'),
})

# COMMAND ----------

# creando features adicionales
# el df de compras por producto se encuentra nivel cliente-mes (no necesita agregacion)
window = Window.partitionBy('client_id').orderBy('mon')

feats_purchase = df_purchase.withColumns({
  'pur_product_1_rsum_m': f.sum(f.col('target_1')).over(window.rowsBetween(Window.unboundedPreceding, 0)),
  'pur_product_2_rsum_m': f.sum(f.col('target_2')).over(window.rowsBetween(Window.unboundedPreceding, 0)),
  'pur_product_3_rsum_m': f.sum(f.col('target_3')).over(window.rowsBetween(Window.unboundedPreceding, 0)),
  'pur_product_4_rsum_m': f.sum(f.col('target_4')).over(window.rowsBetween(Window.unboundedPreceding, 0)),
  'pur_any_purchase': f.when(
    (f.col('target_1') + f.col('target_2') + f.col('target_3') + f.col('target_4')) > 0, 1
  ).otherwise(0),
  'pur_trans_count_avg_3m': f.avg(f.col('trans_count')).over(window.rowsBetween(-3 + 1, 0)),
  'pur_trans_count_avg_6m': f.avg(f.col('trans_count')).over(window.rowsBetween(-6 + 1, 0)),
  'pur_trans_count_std_3m': f.std(f.col('trans_count')).over(window.rowsBetween(-3 + 1, 0)),
  'pur_trans_count_std_6m': f.std(f.col('trans_count')).over(window.rowsBetween(-6 + 1, 0)),
}).withColumnsRenamed({
  'diff_trans_date': 'pur_diff_trans_date_m',
  'trans_count': 'pur_trans_count_m'
})

del df_purchase

# COMMAND ----------

# feature de recencia de la ultima compra y numero de compras por ventanas
feats_purchase = feats_purchase.withColumns({
  'pur_recency_purchase': f.months_between(
      f.col('mon'),
      f.last(
        f.when(f.col('pur_any_purchase') > 0, f.col('mon')), ignorenulls=True
      ).over(window.rowsBetween(Window.unboundedPreceding, -1))
  ),
  'pur_any_purchase_sum_3m': f.sum(f.col('pur_any_purchase')).over(window.rowsBetween(-4 + 1, 0)),
  'pur_any_purchase_sum_6m': f.sum(f.col('pur_any_purchase')).over(window.rowsBetween(-6 + 1, 0)),
})

del window

# COMMAND ----------

# desplazando las features un mes para evitar data leakeage
# de este modo se usarán feautures solo del mes cerrado (simulando producción)
feats_purchase = feats_purchase.withColumn('mon', f.add_months(f.col('mon'), 1))
feats_purchase = feats_purchase.drop(
    'target_1',
    'target_2',
    'target_3',
    'target_4',
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Guardado features generadas

# COMMAND ----------

container = "datos"
account = spark.conf.get("adls.account.name")
folder = "tfm"
path_features = f"abfss://{container}@{account}.dfs.core.windows.net/{folder}/data_preparation/features/"

print(
    f'Path de almacenamiento features: {path_features}.'
)

# COMMAND ----------

# features transaccionales
start_time = time.time()

feats_trx.write.format('parquet').mode('overwrite').save(path_features + 'feats_trx')

end_time = time.time()
final_time = end_time - start_time

print(
    f'Tiempo de guardado: {final_time / 60:.2f} min.'
)

# COMMAND ----------

# features geolocalización
start_time = time.time()

feats_geo.write.format('parquet').mode('overwrite').save(path_features + 'feats_geo')

end_time = time.time()
final_time = end_time - start_time

print(
    f'Tiempo de guardado: {final_time / 60:.2f} min.'
)

# COMMAND ----------

# features diálogos
start_time = time.time()

feats_dialog.write.format('parquet').mode('overwrite').save(path_features + 'feats_dialog')

end_time = time.time()
final_time = end_time - start_time

print(
    f'Tiempo de guardado: {final_time / 60:.2f} min.'
)

# COMMAND ----------

# features compras de productos
start_time = time.time()

feats_purchase.write.format('parquet').mode('overwrite').save(path_features + 'feats_purchase')

end_time = time.time()
final_time = end_time - start_time

print(
    f'Tiempo de guardado: {final_time / 60:.2f} min.'
)