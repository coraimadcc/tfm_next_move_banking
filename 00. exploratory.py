# Databricks notebook source
# MAGIC %md
# MAGIC # TFM: Next Move Banking
# MAGIC **Autor:** Coraima Castillo
# MAGIC
# MAGIC En este notebook se desarrolla el EDA para comprender el comportamiento de la variable a predecir (adquisición/compra de productos) y definir decisiones que impactan al objetivo del modelo.

# COMMAND ----------

from pyspark.sql.window import Window
import pyspark.sql.functions as f
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importación datos

# COMMAND ----------

container = "datos"
account = spark.conf.get("adls.account.name")
folder = "tfm"
path = f"abfss://{container}@{account}.dfs.core.windows.net/{folder}/raw_data/"

# COMMAND ----------

# MAGIC %md
# MAGIC Los datos corresponden a una muestra del Multimodal Bank Dataset (MBD, 2023), los cuales corresponden a datos reales anonimizados de clientes corporativos del banco Sberbank (banco de Rusia). Los datos fueron liberados por el equipo de Sber AI Lab con fines de investigación en modelos multimodales.
# MAGIC - trx → historial de transacciones financieras de los clientes.
# MAGIC - geo → registros de localización asociados al uso de la aplicación bancaria.
# MAGIC - dialog → interacciones de los clientes con soporte o gestores, representadas como embeddings. 
# MAGIC - target → compras mensuales de cuatro productos bancarios.

# COMMAND ----------

# df de transacciones
df_trx = spark.read.parquet(path + "transactions")

# df de ubicaciones
df_geo = spark.read.parquet(path + "geolocation")

# df de interacciones (dialogos)
df_dialog = spark.read.parquet(path + "dialog")

# df de indicadores de adquisicion nuevos productos
df_purchases = spark.read.parquet(path + "purchase")

# COMMAND ----------

display(df_dialog)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratorio: Variable dependiente

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clientes que adquieren nuevo producto c/mes

# COMMAND ----------

# cantidad de clientes
print(
    f'Cantidad clientes únicos: {df_purchases.select("client_id").distinct().count():,}\n'
    f'Cantidad de registros: {df_purchases.count():,}'
)


# COMMAND ----------

df_purchases = df_purchases.withColumns({
  'mon': f.to_date(f.col('mon'), 'yyyy-MM-dd'),
  'ind_new_product': f.when(
    (f.col('target_1') + f.col('target_2') + f.col('target_3') + f.col('target_4')) > 0,
    1
  ).otherwise(0)
})

# COMMAND ----------

# determinando si las compras corresponden a la primera compra por cliente
# o el 1 solo indica que ese mes adquirió el producto renovado (ejemplo: cdp)
display(
    df_purchases.groupBy("client_id").agg(
        f.sum('target_1').alias('target_1'),
        f.sum('target_2').alias('target_2'),
        f.sum('target_3').alias('target_3'),
        f.sum('target_4').alias('target_4'),
        f.sum('ind_new_product').alias('new_product'),
    ).filter(
        (f.col('target_1') > 1) | (f.col('target_2') > 1) | (f.col('target_3') > 1) | (f.col('target_4') > 1)
    )
)

# COMMAND ----------

# distribucion por target
display(df_purchases.groupBy('target_1').agg(
    f.count(f.col('target_1'))
))
display(df_purchases.groupBy('target_2').agg(
    f.count(f.col('target_2'))
))
display(df_purchases.groupBy('target_3').agg(
    f.count(f.col('target_3'))
))
display(df_purchases.groupBy('target_4').agg(
    f.count(f.col('target_4'))
))

# COMMAND ----------

display(df_purchases)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Frecuencia compra por producto

# COMMAND ----------

window = Window.partitionBy('client_id').orderBy('mon')
product_1 = df_purchases.filter("target_1 = 1") \
                        .withColumn('mon_lag_1m', f.lag(f.col('mon'), 1).over(window)) \
                        .withColumn('diff_months', f.months_between(f.col('mon'), f.col('mon_lag_1m'))) \
                        .filter("mon_lag_1m is not null") \
                        .select(f.lit('product_1').alias('product'), 'diff_months').toPandas()

product_2 = df_purchases.filter("target_2 = 1") \
                        .withColumn('mon_lag_1m', f.lag(f.col('mon'), 1).over(window)) \
                        .withColumn('diff_months', f.months_between(f.col('mon'), f.col('mon_lag_1m'))) \
                        .filter("mon_lag_1m is not null") \
                        .select(f.lit('product_2').alias('product'), 'diff_months').toPandas()

product_3 = df_purchases.filter("target_3 = 1") \
                        .withColumn('mon_lag_1m', f.lag(f.col('mon'), 1).over(window)) \
                        .withColumn('diff_months', f.months_between(f.col('mon'), f.col('mon_lag_1m'))) \
                        .filter("mon_lag_1m is not null") \
                        .select(f.lit('product_3').alias('product'), 'diff_months').toPandas()

product_4 = df_purchases.filter("target_4 = 1") \
                        .withColumn('mon_lag_1m', f.lag(f.col('mon'), 1).over(window)) \
                        .withColumn('diff_months', f.months_between(f.col('mon'), f.col('mon_lag_1m'))) \
                        .filter("mon_lag_1m is not null") \
                        .select(f.lit('product_4').alias('product'), 'diff_months').toPandas()

new_product = df_purchases.filter("ind_new_product = 1") \
                          .withColumn('mon_lag_1m', f.lag(f.col('mon'), 1).over(window)) \
                          .withColumn('diff_months', f.months_between(f.col('mon'), f.col('mon_lag_1m'))) \
                          .filter("mon_lag_1m is not null") \
                          .select(f.lit('any_product').alias('product'), 'diff_months').toPandas()

# COMMAND ----------

# consolidacion de dfs por producto
df_adquisiciones = pd.concat([product_1, product_2, product_3, product_4, new_product])

# df de la distribucion de los meses transcurridos entre adquisiciones
dist_adquisiciones = (df_adquisiciones
          .groupby(["product", "diff_months"])
          .size()
          .reset_index(name="count"))

dist_adquisiciones["percent"] = dist_adquisiciones.groupby("product")["count"].transform(lambda x: x / x.sum() * 100)

# df de distribucion acumulada
mins = dist_adquisiciones['diff_months'].min()
maxs = dist_adquisiciones['diff_months'].max()
grid = (pd.MultiIndex
        .from_product([dist_adquisiciones['product'].unique(), np.arange(mins, maxs+1)],
                      names=['product','diff_months'])
        .to_frame(index=False))
dist_adquisiciones = grid.merge(dist_adquisiciones, how='left').fillna({'count': 0})

dist_adquisiciones['percent'] = dist_adquisiciones.groupby('product')['count'].transform(lambda x: 100 * x / x.sum())
dist_adquisiciones['cum_percent'] = dist_adquisiciones.groupby('product')['percent'].cumsum().clip(upper=100)


# COMMAND ----------

fig = px.line(dist_adquisiciones,
              x="diff_months",
              y="percent",
              color="product",
              markers=True,
              title="Distribución de meses entre adquisiciones por producto")

fig.update_xaxes(title="Meses entre adquisiciones")
fig.update_yaxes(title="% de adquisiciones", ticksuffix='%')
fig.show()


# COMMAND ----------

fig = px.line(dist_adquisiciones,
              x='diff_months',
              y='cum_percent',
              color='product',
              markers=True,
              title='Distribución acumulada de meses entre adquisiciones por producto')
fig.update_xaxes(title='Meses entre adquisiciones')
fig.update_yaxes(title='Proporción acumulada\n', range=[0, 105], ticksuffix='%')
fig.show()

# COMMAND ----------

fig = px.histogram(
    df_adquisiciones,
    x="diff_months",
    color="product",
    facet_col="product",
    facet_col_wrap=2,
    nbins=30,
    title="Distribución de tiempo entre adquisiciones (meses)",
    facet_col_spacing=0.08,   # ← separación horizontal entre paneles
    facet_row_spacing=0.25     # ← separación vertical entre filas
)

fig.for_each_yaxis(lambda yaxis: yaxis.update(matches=None, showticklabels=True, title_text="Frecuencia"))

fig.update_layout(bargap=0.15, margin=dict(l=20, r=20, t=60, b=20))
fig.update_xaxes(title="Meses entre adquisiciones")
fig.update_yaxes(title="Frecuencia")

fig.show()


# COMMAND ----------

# MAGIC %md
# MAGIC A partir del análisis de frecuencia de compra, se observa que ~80% de las compras realizadas se realizaron 4 meses después de una compra ya ejecutada; es decir, la siguiente compra de cualquier producto suele generarse en su mayoría en un lapso de a lo mucho 4 meses.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Comportamiento compra

# COMMAND ----------

window = Window.partitionBy('client_id').orderBy('mon').rowsBetween(0, 4)

df_behavior = df_purchases.withColumns({
    't1_4w': f.max(f.col('target_1')).over(window),
    't2_4w': f.max(f.col('target_2')).over(window),
    't3_4w': f.max(f.col('target_3')).over(window),
    't4_4w': f.max(f.col('target_4')).over(window)
})

productos_adq = [
    f.when(f.col('t1_4w') == 1, f.lit('product_1')).otherwise(None),
    f.when(f.col('t2_4w') == 1, f.lit('product_2')).otherwise(None),
    f.when(f.col('t3_4w') == 1, f.lit('product_3')).otherwise(None),
    f.when(f.col('t4_4w') == 1, f.lit('product_4')).otherwise(None),
]

df_behavior = df_behavior.withColumn('prods', f.array(*productos_adq)) \
                         .withColumn("prods", f.expr("filter(prods, x -> x is not null)")) \
                         .withColumn('number_prods', f.size(f.col('prods')))

# COMMAND ----------

display(df_behavior)

# COMMAND ----------

df_pairs = df_behavior.filter(f.col("number_prods") >= 2) \
                      .withColumn("id", f.monotonically_increasing_id())

a = df_pairs.select("id", "client_id", "mon", f.posexplode("prods").alias("i", "p1"))
b = df_pairs.select("id", f.posexplode("prods").alias("j","p2"))

df_pairs = (a.join(b, "id")
              .where(f.col("i") < f.col("j"))
              .select("client_id", "mon", "p1", "p2"))

df_cooc = df_pairs.groupBy("p1","p2").count()

# COMMAND ----------

display(df_cooc)

# COMMAND ----------

tot_p1 = df_pairs.groupBy("p1").count().withColumnRenamed("count","tot_p1")
tot_p2 = df_pairs.groupBy("p2").count().withColumnRenamed("count","tot_p2")

# obteniendo probabilidades condicionales de compra de productos
df_cond = (df_cooc
    .join(tot_p1, "p1")
    .join(tot_p2, "p2")
    .withColumn("p(p2|p1)", (f.col("count")/f.col("tot_p1")).cast("double"))
    .withColumn("p(p1|p2)", (f.col("count")/f.col("tot_p2")).cast("double"))
)

print("Pairs rows:", df_pairs.count())
df_cooc.orderBy(f.desc("count")).show(truncate=False)
df_cond.orderBy(f.desc("p(p2|p1)")).show(truncate=False)

# COMMAND ----------

cooc_pd = df_cooc.toPandas().pivot(index="p1", columns="p2", values="count").fillna(0)
cooc_full = cooc_pd.add(cooc_pd.T, fill_value=0)

fig = px.imshow(cooc_full, text_auto=True, color_continuous_scale="Viridis",
                title="Co-ocurrencia de productos (ventana 4 meses ≙ 4 filas)")

fig.update_xaxes(title="Producto")
fig.update_yaxes(title="Producto")

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC El análisis de coocurrencia evidencia que los productos no suelen comportarse de manera independiente entre sí, sino que presentan relaciones de compra conjunta en la ventana de observación de 4 meses. En particular, el producto 1 suele funcionar como producto "base" (es decir, P(1|X) con frecuencia es elevada), y esto sugiere que el producto 1 refleja un patrón de adopción temprana o masiva. 
# MAGIC
# MAGIC Por otro lado, el producto 2 no solo presenta un frecuencia de compra baja, este también parece ser un producto accesorio o secuendario (esto se evidencia en que las P(X|2) son altas o moderadas), dado que casi siempre que se ha comprado el producto 2, este se acompaña de otros productos (generalmente de 1 o 3). Entonces, el producto 2 con su baja frecuencia es poco relevante para manejar como target independiente, sin embargo, la adquirencia del mismo puede funcionar como una señal para determinar las probabilidades de compra de otros productos. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Análisis cadena de productos

# COMMAND ----------

window = Window.partitionBy('client_id').orderBy('mon').rowsBetween(0, 4)

df_flags = df_behavior
for k in range(1, 5):
    df_flags = df_flags.withColumn(f"t{k}_next4", f.max(f.col(f"target_{k}")).over(window))

# COMMAND ----------

# Array de structs (pair, flag) para las 12 combinaciones i->j (i != j)
pairs_cols = []
for i in range(1,5):
    for j in range(1,5):
        if i == j:
            continue
        src_flag = (f.col(f"target_{i}") == 1)
        dst_flag = (f.col(f"t{j}_next4") == 1)
        flag_ij  = (src_flag & dst_flag).cast("int")
        pair_ij  = f.lit(f"product_{i}|product_{j}")
        pairs_cols.append(f.struct(pair_ij.alias("pair"), flag_ij.alias("flag")))

df_pairs_long = (df_flags
    .select("client_id", "mon", f.array(*pairs_cols).alias("pairs"))
    .withColumn("kv", f.explode("pairs"))
    .select("client_id", "mon", f.col("kv.pair").alias("pair"), f.col("kv.flag").alias("flag"))
    .filter(f.col("flag") == 1)
)

df_trans = (df_pairs_long
    .select(
        "client_id", "mon",
        f.split("pair", "\\|").getItem(0).alias("p_i"),
        f.split("pair", "\\|").getItem(1).alias("p_j")
    ))

df_counts = df_trans.groupBy("p_i","p_j").count()

# COMMAND ----------

flows_pd = df_counts.orderBy(f.desc("count")).toPandas()
flows_pd = flows_pd.loc[flows_pd['count'] > 0]

labels = sorted(set(flows_pd["p_i"]).union(set(flows_pd["p_j"])))
idx = {p: i for i, p in enumerate(labels)}

fig = go.Figure(go.Sankey(
    node=dict(
        label=labels,
        pad=20, thickness=20
    ),
    link=dict(
        source=[idx[s] for s in flows_pd["p_i"]],
        target=[idx[t] for t in flows_pd["p_j"]],
        value=flows_pd["count"],
        hovertemplate="%{source.label} → %{target.label}<br>Transiciones: %{value:,.0f}<extra></extra>"
    )
))

fig.update_layout(
    title_text=f"Transiciones de producto (i → j) en ventana de 4 meses",
    font=dict(size=12),
    margin=dict(l=20, r=20, t=60, b=20)
)

fig.show()


# COMMAND ----------

# MAGIC %md
# MAGIC El análisis de secuencia mediante Sankey muestra que el producto 1 suele ser la puerta de entrada al portafolio, mientras que los productos 3 y 4 forman un bloque fuertemente asociado de coadquisición. El producto 2, por su baja frecuencia y dependencia, no configura trayectorias propias, sino que se presenta como complemento. 
# MAGIC
# MAGIC Estos hallazgos permiten concluir que resulta factible el diseño de tres modelos predictivos independientes para productos 1, 3 y 4, tratando al producto 2 como variable explicativa en lugar de target.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Relación transaccionalidad y adquisición

# COMMAND ----------

# convirtiendo df de compras a pandas
df_purchases_pd = df_purchases.toPandas()

# COMMAND ----------


# diagrama de cajas trans_count
fig1 = px.box(df_purchases_pd, x="ind_new_product", y="trans_count", color="ind_new_product",
              title="Distribución de trans_count según adquisición",
              labels={"ind_new_product":"Adquisición","trans_count":"Nº transacciones"})
fig1.show()

# diagrama de cajas diff_trans_date
fig2 = px.box(df_purchases_pd, x="ind_new_product", y="diff_trans_date", color="ind_new_product",
              title="Distribución de diff_trans_date según adquisición",
              labels={"ind_new_product":"Adquisición","diff_trans_date":"Días desde última transacción"})
fig2.show()

# COMMAND ----------

# MAGIC %md
# MAGIC A partir de los diagramas de cajas se puede concluir que los clientes que han adquirido al menos un producto en el banco suelen tener mayor movimiento transaccional, puesto que el 50% de estos clientes tienen hasta 67 transacciones (mediana), mientras que los que no adquieren productos con el banco, el 50% tiene a lo mucho 33 transacciones (mediana).
# MAGIC
# MAGIC Del mismo modo, esta diferencia en comportamiento transaccional se aprecia en la recencia de transacciones; el 50% de los clientes que compraron un producto en el banco tenían una recencia entre 0-1 día, y el 50% de los clientes que no adquieren productos con el banco tiene una recencia de hasta 3 días.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratorio: Transaccionalidad

# COMMAND ----------

print(
    f'Cantidad tipos de eventos: {df_trx.select("event_type").distinct().count()}.\n'
    f'Cantidad tipos de sub-eventos: {df_trx.select("event_subtype").distinct().count()}.'
)

# COMMAND ----------

display(df_trx.groupBy('event_type').count())

# COMMAND ----------

display(
    df_trx.groupBy('event_type', 'event_subtype').count()\
          .orderBy('event_type', 'event_subtype')
)

# COMMAND ----------

print(
    f'Cantidad de tipos de moneda: {df_trx.select("currency").distinct().count()}.'
)

display(df_trx.groupBy('currency').count())

# COMMAND ----------

display(df_trx.groupBy('event_type', 'currency').count())

# COMMAND ----------

display(
    df_trx.filter(f.col('currency').isNull())
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Nota:** La monedas en donde frecuentan las transacciones son las siguientes: 11, 15, 3 (tanto en conteo como en monto).

# COMMAND ----------

# distribucion de transacciones por moneda
display(
    df_trx.groupBy('currency').agg(
        f.count('*').alias('conteo'),
        f.sum(f.col('amount')).alias('amount')
    ).withColumns({
        'pct_conteo': f.col('conteo') / f.sum(f.col('conteo')).over(Window.partitionBy()),
        'pct_amount': f.col('amount') / f.sum(f.col('amount')).over(Window.partitionBy())
    })
)

# COMMAND ----------

display(
    df_trx.groupBy('dst_type11').count()
)

# COMMAND ----------

display(
    df_trx.filter(f.col('dst_type11').isNull())
)

# COMMAND ----------

display(
    df_trx.groupBy('dst_type11', 'dst_type12').count()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratorio: Geolocalizaciones

# COMMAND ----------

print(df_geo.select('geohash_4').dropDuplicates().count())
print(df_geo.select('geohash_5').dropDuplicates().count())
print(df_geo.select('geohash_6').dropDuplicates().count())

# COMMAND ----------

display(df_geo)

# COMMAND ----------

