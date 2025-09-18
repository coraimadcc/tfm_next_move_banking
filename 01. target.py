# Databricks notebook source
# MAGIC %md
# MAGIC # TFM: Next Move Banking
# MAGIC **Autor:** Coraima Castillo
# MAGIC
# MAGIC En este notebook se desarrolla el TARGET acorde a las definiciones establecidas según el EDA.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Instalando dependencias

# COMMAND ----------

!pip install umap-learn==0.5.7

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importación librerías / datos

# COMMAND ----------

# librerias basicas
from pyspark.sql.window import Window
import pyspark.sql.functions as f
import pandas as pd
import time
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# librerias para estratificacion
from sklearn.pipeline import Pipeline
import umap
from sklearn.cluster import HDBSCAN
from sklearn.cluster import KMeans

# COMMAND ----------

import mlflow
mlflow.autolog(disable=True)

# COMMAND ----------

# MAGIC %run /Workspace/Users/coraimac@ucm.es/TFM/utils_functions

# COMMAND ----------

container = "datos"
account = spark.conf.get("adls.account.name")
folder = "tfm"
path = f"abfss://{container}@{account}.dfs.core.windows.net/{folder}/raw_data/"

# COMMAND ----------

# df de indicadores de adquisición de nuevos productos
df_purchases = spark.read.parquet(path + "purchase")

# realizando renombre de columnas para no confundir target y compras
df_purchases = df_purchases.withColumnsRenamed({
    'target_1': 'product_1',
    'target_2': 'product_2',
    'target_3': 'product_3',
    'target_4': 'product_4',
})

# COMMAND ----------

# MAGIC %md
# MAGIC # Muestreo estratificado

# COMMAND ----------

# MAGIC %md
# MAGIC A continuación, se realiza una estratificación de clientes que NO tienen ninguna compra, dado que el desbalance de la data total liberada por Sber AI Lab es tal que el 81% de los clientes no ha realizado ninguna compra; . De este modo, se construye un df con clientes que obdezca un comportamiento similar. Para la estratificación, se realizan clusters acorde al comportamiento transaccional del clientes en el 2022.
# MAGIC
# MAGIC **Fuente:** Mollaev, D., Kireev, I., Orlov, M., Kostin, A., Karpukhin, I., Postnova, M., Gusev, G., & Savchenko, A. (2025). Multimodal banking dataset: Understanding client needs through event sequences. Sber AI Lab. En Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Toronto, ON, Canada. https://arxiv.org/pdf/2409.17587

# COMMAND ----------

# MAGIC %md
# MAGIC ## Producto 1

# COMMAND ----------

# MAGIC %md
# MAGIC ### Features para estratificación

# COMMAND ----------

df_stratified = df_purchases.withColumn(
  'ind_purchase', f.when(
    f.col('product_1') > 0,
    1
  ).otherwise(0)
)

# segun el EDA los clientes suelen hacer una nueva compra 
# en un lapso de 4 meses tras su ultima compra realizada
window = Window.partitionBy('client_id').orderBy('mon').rowsBetween(0, 4)
w_6m = Window.partitionBy('client_id').orderBy('mon').rowsBetween(-6+1, 0)
w_3m = Window.partitionBy('client_id').orderBy('mon').rowsBetween(-6+1, 0)
df_stratified = df_stratified.withColumns({
    'ind_purchase': f.max(f.col('ind_purchase')).over(window),
})

df_stratified = df_stratified.groupBy('client_id', 'mon').agg(
  f.max('ind_purchase').alias('ind_any_purchase'),
  f.sum('trans_count').alias('total_trans'),
  f.max('trans_count').alias('diff_trans_date'),
).withColumns({
  'avg_trans_3m': f.avg(f.col('total_trans')).over(w_3m),
  'avg_trans_6m': f.avg(f.col('total_trans')).over(w_6m),
  'std_trans_6m': f.std(f.col('total_trans')).over(w_6m),
  'min_trans_date_6m': f.min('diff_trans_date').over(w_6m),
  'max_trans_date_6m': f.max('diff_trans_date').over(w_6m),
  'avg_trans_date_3m': f.avg('diff_trans_date').over(w_3m),
  'avg_trans_date_6m': f.avg('diff_trans_date').over(w_6m),
})

# COMMAND ----------

# segmentando dfs para la estratificacion
df_stratified0 = df_stratified.filter(f.col('ind_any_purchase') == 0)
df_stratified1 = df_stratified.filter(f.col('ind_any_purchase') == 1)

df_stratified0_pd = df_stratified0.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Identificación naturaleza features

# COMMAND ----------

# separando entre X e y
key = ['client_id', 'mon']
target = ['ind_any_purchase']
features = [
  col for col in df_stratified0_pd if col not in key + target
]

X, y = df_stratified0_pd[features], df_stratified0_pd[target]

# COMMAND ----------

numerical_cols, categorical_cols = _identify_type_columns_pandas(
    df=df_stratified0_pd,
    params={
        'id_columns': key,
        'target': target,
    }
)

# COMMAND ----------

# determinando si existen nulos en el df
X.isna().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pipeline preprocesamiento

# COMMAND ----------

# pipeline de preprocesamiento cprt
preprocessing = Pipeline(steps=[
    ('NumericalTransformer', NumericalLogTransformer(
                numerical_cols=numerical_cols,
                threshold_coef=1.5,
        )),
    ('Scaler', ContinuousFeatureScaler(scaler='standard',
                            numerical_cols=numerical_cols))
  ])

preprocessing.fit(X, y)

# COMMAND ----------

# preprocesando df para la estratificación
X_preprocessed = preprocessing.transform(X).fillna(0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reducción dimensionalidad

# COMMAND ----------

# entrenando umap con una muestra de 5%
X_sampled = X_preprocessed.sample(frac=0.05, random_state=42)
umap_reducer = umap.UMAP(
    n_neighbors=100,
    min_dist=0.085,
    metric='euclidean',
    n_components=3,
    n_jobs=-1
).fit(X_sampled)

# COMMAND ----------

# aplicando transformacion umap para todos los datos
df_umap = umap_reducer.transform(X_preprocessed) 
X_umap = pd.DataFrame(df_umap, columns=['umap_1', 'umap_2', 'umap_3'])
X_umap = X_umap.set_index(X_preprocessed.index)

# COMMAND ----------

# asignando cliente y mon al df con dimensionalidad reducida
X_umap['client_id'] = df_stratified0_pd['client_id']
X_umap['mon'] = df_stratified0_pd['mon']

# COMMAND ----------

X_umap_spark = spark.createDataFrame(X_umap)

path_target = 'abfss://datos@masterccc002sta.dfs.core.windows.net/tfm/data_preparation/target/stratification_class_0/target_1/'
X_umap_spark.write.format('parquet').mode('overwrite').save(path_target)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clusters

# COMMAND ----------

X_umap = spark.read.parquet('abfss://datos@masterccc002sta.dfs.core.windows.net/tfm/data_preparation/target/stratification_class_0/target_1/').toPandas()

# COMMAND ----------

# entrenando KMeans
kmeans = KMeans(n_clusters=3, 
                random_state=42, 
                n_init='auto')
labels_kmeans = kmeans.fit_predict(X_umap[['umap_1', 'umap_2', 'umap_3']])

# asignando etiquetas al DataFrame
X_umap['cluster'] = labels_kmeans
df_stratified0_pd['cluster'] = labels_kmeans

# COMMAND ----------

# distribución por cluster
X_umap.groupby(['cluster']).agg(
    obs=('cluster', 'count')
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gráfico

# COMMAND ----------

X_sampled = X_umap.sample(frac=0.25, random_state=42)
fig = px.scatter_3d(X_sampled, 
                    x='umap_1', y='umap_2', z='umap_3', 
                    color='cluster', 
                    opacity=0.7,
)

fig.update_traces(marker=dict(size=4)) # changing the size of each point

fig.update_layout(
    legend_title_text='clusters',
    legend=dict(
        x=1,
        y=1,
        traceorder='normal',
        font=dict(
            family='Arial',
            size=11,
            color='black'        
        ),
        bgcolor='LightSteelBlue',
        bordercolor='Black',
        borderwidth=2,
    ),
    title={
        'text': 'gráfico de dispersión: UMAP == > HDBSCAN',
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
    },
    width=1000,
    height=600,
)

# show the plot
html_str = fig.to_html()
displayHTML(html_str)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Estratificación

# COMMAND ----------

customers_0 = spark.createDataFrame(df_stratified0_pd[['mon', 'client_id', 'cluster']])
customers_1 = df_stratified1.select('mon', 'client_id')

# COMMAND ----------

min_ratio, max_ratio = 0.60, 0.65
random = np.random.uniform(0, 1)

count_1 = customers_1.count()
dist_cluster = customers_0.groupBy('cluster').agg(f.count('*').alias('cnt_cluster'))\
                          .withColumn('pct_cluster', f.col('cnt_cluster') / f.sum('cnt_cluster').over(Window.partitionBy()))
customers_0 = customers_0.withColumn(
    'ratio_0', f.lit(min_ratio) + (f.lit(max_ratio - min_ratio) * f.lit(random)),
).withColumn(
    'stratified_0', f.round((f.col('ratio_0') / (1 - f.col('ratio_0'))) * f.lit(count_1))
)

customers_0 = customers_0.join(dist_cluster.select('cluster', 'pct_cluster'), on='cluster', how='left')

window = Window.partitionBy
customers_0 = customers_0.withColumns({
    'quota_cluster': f.round(f.col('stratified_0') * f.col('pct_cluster')),
    'rand_int': f.round(f.rand(42) * 100)
})

window = Window.partitionBy('cluster').orderBy(f.col('rand_int'))
customers_0 = customers_0.withColumn('row_number', f.row_number().over(window))
customers_0 = customers_0.filter(f.col('row_number') <= f.col('quota_cluster'))


# COMMAND ----------

# consolidando clientes estratificados (sin compra) y clientes con compras
customers_stratified = customers_1.withColumn('class', f.lit(1)).unionByName(
    customers_0.select('mon', 'client_id').withColumn('class', f.lit(0))
)

customers_stratified.groupBy('class').agg(
    f.count('*').alias('count')
).withColumn(
    'percentage', f.col('count') / f.sum('count').over(Window.partitionBy())
).display()

# COMMAND ----------

# obteniendo distribucion final
df_stratified_0 = df_stratified.join(
    customers_stratified.select('mon', 'client_id'), on=['client_id', 'mon'], how='inner'
)

display(
    df_stratified_0.groupBy('ind_any_purchase').agg(
        f.count('*').alias('count')
    ).withColumn(
        'percentage', f.col('count') / f.sum('count').over(Window.partitionBy())
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Definición target

# COMMAND ----------

# obteniendo solo clientes resultantes de la estratificacion
df_target = df_purchases.join(
    customers_stratified, on=['client_id', 'mon'], how='inner'
)

# COMMAND ----------

# tras el EDA se definió enfocarse en la prediccion de pts 1,3,4
# dado que son los que tienen mayor participación, el pt 2 se utiliza para
# construccion de features (feauture engineering)}
window = Window.partitionBy('client_id').orderBy('mon').rowsBetween(0, 4)

df_target = df_target.withColumns({
    'target_1': f.max(f.col('product_1')).over(window),
})


# COMMAND ----------

# obteniendo la ultima fecha del df para filtrar hasta la fecha
# que permita visualizar la ventana de observacion completa (4 meses)
last_date = df_target.select(f.max('mon').alias('max_mon'))
last_date = last_date.withColumn(
    'max_mon', f.add_months(f.col('max_mon'), -4)
).first()[0]

print(
    f'Fecha hasta donde se puede visualizar la ventana de observacion completa: {last_date}'
)

# COMMAND ----------

# filtrando registros hasta la fecha donde se puede visualizar 
# la ventana de observacion completa
df_target = df_target.filter(
    f.col('mon') <= last_date
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Desbalance por producto

# COMMAND ----------

# desbalance producto 1
df_target_1 = df_target.groupBy('target_1').count()
df_target_1 = df_target_1.withColumn(
  'percentage', f.col('count') / f.sum('count').over(Window.partitionBy())
)
display(df_target_1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Guardado

# COMMAND ----------

target = df_target.select(
    'client_id',
    'mon',
    'target_1',
)

target = target.withColumn(
    'mon', 
    f.to_date(f.date_format(f.col('mon'), 'yyyy-MM-01'), 'yyyy-MM-dd')
)

# COMMAND ----------

path_target = 'abfss://datos@masterccc002sta.dfs.core.windows.net/tfm/data_preparation/target/product_1'
target.write.format('parquet').mode('overwrite').save(path_target)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Producto 3

# COMMAND ----------

# MAGIC %md
# MAGIC ### Features para estratificación

# COMMAND ----------

df_stratified = df_purchases.withColumn(
  'ind_purchase', f.when(
    f.col('product_3') > 0,
    1
  ).otherwise(0)
)

# segun el EDA los clientes suelen hacer una nueva compra 
# en un lapso de 4 meses tras su ultima compra realizada
window = Window.partitionBy('client_id').orderBy('mon').rowsBetween(0, 4)
w_6m = Window.partitionBy('client_id').orderBy('mon').rowsBetween(-6+1, 0)
w_3m = Window.partitionBy('client_id').orderBy('mon').rowsBetween(-6+1, 0)
df_stratified = df_stratified.withColumns({
    'ind_purchase': f.max(f.col('ind_purchase')).over(window),
})

df_stratified = df_stratified.groupBy('client_id', 'mon').agg(
  f.max('ind_purchase').alias('ind_any_purchase'),
  f.sum('trans_count').alias('total_trans'),
  f.max('trans_count').alias('diff_trans_date'),
).withColumns({
  'avg_trans_3m': f.avg(f.col('total_trans')).over(w_3m),
  'avg_trans_6m': f.avg(f.col('total_trans')).over(w_6m),
  'std_trans_6m': f.std(f.col('total_trans')).over(w_6m),
  'min_trans_date_6m': f.min('diff_trans_date').over(w_6m),
  'max_trans_date_6m': f.max('diff_trans_date').over(w_6m),
  'avg_trans_date_3m': f.avg('diff_trans_date').over(w_3m),
  'avg_trans_date_6m': f.avg('diff_trans_date').over(w_6m),
})

# COMMAND ----------

# segmentando dfs para la estratificacion
df_stratified0 = df_stratified.filter(f.col('ind_any_purchase') == 0)
df_stratified1 = df_stratified.filter(f.col('ind_any_purchase') == 1)

df_stratified0_pd = df_stratified0.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Identificación naturaleza features

# COMMAND ----------

# separando entre X e y
key = ['client_id', 'mon']
target = ['ind_any_purchase']
features = [
  col for col in df_stratified0_pd if col not in key + target
]

X, y = df_stratified0_pd[features], df_stratified0_pd[target]

# COMMAND ----------

numerical_cols, categorical_cols = _identify_type_columns_pandas(
    df=df_stratified0_pd,
    params={
        'id_columns': key,
        'target': target,
    }
)

# COMMAND ----------

# determinando si existen nulos en el df
X.isna().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pipeline preprocesamiento

# COMMAND ----------

# pipeline de preprocesamiento cprt
preprocessing = Pipeline(steps=[
    ('NumericalTransformer', NumericalLogTransformer(
                numerical_cols=numerical_cols,
                threshold_coef=1.5,
        )),
    ('Scaler', ContinuousFeatureScaler(scaler='standard',
                            numerical_cols=numerical_cols))
  ])

preprocessing.fit(X, y)

# COMMAND ----------

# preprocesando df para la estratificación
X_preprocessed = preprocessing.transform(X).fillna(0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reducción dimensionalidad

# COMMAND ----------

# entrenando umap con una muestra de 5%
X_sampled = X_preprocessed.sample(frac=0.05, random_state=42)
umap_reducer = umap.UMAP(
    n_neighbors=100,
    min_dist=0.085,
    metric='euclidean',
    n_components=3,
    n_jobs=-1
).fit(X_sampled)

# COMMAND ----------

# aplicando transformacion umap para todos los datos
df_umap = umap_reducer.transform(X_preprocessed) 
X_umap = pd.DataFrame(df_umap, columns=['umap_1', 'umap_2', 'umap_3'])
X_umap = X_umap.set_index(X_preprocessed.index)

# COMMAND ----------

# asignando cliente y mon al df con dimensionalidad reducida
X_umap['client_id'] = df_stratified0_pd['client_id']
X_umap['mon'] = df_stratified0_pd['mon']

# COMMAND ----------

X_umap_spark = spark.createDataFrame(X_umap)

path_target = 'abfss://datos@masterccc002sta.dfs.core.windows.net/tfm/data_preparation/target/stratification_class_0/target_3/'
X_umap_spark.write.format('parquet').mode('overwrite').save(path_target)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clusters

# COMMAND ----------

X_umap = spark.read.parquet('abfss://datos@masterccc002sta.dfs.core.windows.net/tfm/data_preparation/target/stratification_class_0/target_3/').toPandas()

# COMMAND ----------

# entrenando KMeans
kmeans = KMeans(n_clusters=3, 
                random_state=42, 
                n_init='auto')
labels_kmeans = kmeans.fit_predict(X_umap[['umap_1', 'umap_2', 'umap_3']])

# asignando etiquetas al DataFrame
X_umap['cluster'] = labels_kmeans
df_stratified0_pd['cluster'] = labels_kmeans

# COMMAND ----------

# distribución por cluster
X_umap.groupby(['cluster']).agg(
    obs=('cluster', 'count')
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gráfico

# COMMAND ----------

X_sampled = X_umap.sample(frac=0.25, random_state=42)
fig = px.scatter_3d(X_sampled, 
                    x='umap_1', y='umap_2', z='umap_3', 
                    color='cluster', 
                    opacity=0.7,
)

fig.update_traces(marker=dict(size=4)) # changing the size of each point

fig.update_layout(
    legend_title_text='clusters',
    legend=dict(
        x=1,
        y=1,
        traceorder='normal',
        font=dict(
            family='Arial',
            size=11,
            color='black'        
        ),
        bgcolor='LightSteelBlue',
        bordercolor='Black',
        borderwidth=2,
    ),
    title={
        'text': 'gráfico de dispersión: UMAP == > HDBSCAN',
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
    },
    width=1000,
    height=600,
)

# show the plot
html_str = fig.to_html()
displayHTML(html_str)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Estratificación

# COMMAND ----------

customers_0 = spark.createDataFrame(df_stratified0_pd[['mon', 'client_id', 'cluster']])
customers_1 = df_stratified1.select('mon', 'client_id')

# COMMAND ----------

min_ratio, max_ratio = 0.60, 0.65
random = np.random.uniform(0, 1)

count_1 = customers_1.count()
dist_cluster = customers_0.groupBy('cluster').agg(f.count('*').alias('cnt_cluster'))\
                          .withColumn('pct_cluster', f.col('cnt_cluster') / f.sum('cnt_cluster').over(Window.partitionBy()))
customers_0 = customers_0.withColumn(
    'ratio_0', f.lit(min_ratio) + (f.lit(max_ratio - min_ratio) * f.lit(random)),
).withColumn(
    'stratified_0', f.round((f.col('ratio_0') / (1 - f.col('ratio_0'))) * f.lit(count_1))
)

customers_0 = customers_0.join(dist_cluster.select('cluster', 'pct_cluster'), on='cluster', how='left')

window = Window.partitionBy
customers_0 = customers_0.withColumns({
    'quota_cluster': f.round(f.col('stratified_0') * f.col('pct_cluster')),
    'rand_int': f.round(f.rand(42) * 100)
})

window = Window.partitionBy('cluster').orderBy(f.col('rand_int'))
customers_0 = customers_0.withColumn('row_number', f.row_number().over(window))
customers_0 = customers_0.filter(f.col('row_number') <= f.col('quota_cluster'))


# COMMAND ----------

# consolidando clientes estratificados (sin compra) y clientes con compras
customers_stratified = customers_1.withColumn('class', f.lit(1)).unionByName(
    customers_0.select('mon', 'client_id').withColumn('class', f.lit(0))
)

customers_stratified.groupBy('class').agg(
    f.count('*').alias('count')
).withColumn(
    'percentage', f.col('count') / f.sum('count').over(Window.partitionBy())
).display()

# COMMAND ----------

# obteniendo distribucion final
df_stratified_0 = df_stratified.join(
    customers_stratified.select('mon', 'client_id'), on=['client_id', 'mon'], how='inner'
)

display(
    df_stratified_0.groupBy('ind_any_purchase').agg(
        f.count('*').alias('count')
    ).withColumn(
        'percentage', f.col('count') / f.sum('count').over(Window.partitionBy())
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Definición target

# COMMAND ----------

# obteniendo solo clientes resultantes de la estratificacion
df_target = df_purchases.join(
    customers_stratified, on=['client_id', 'mon'], how='inner'
)

# COMMAND ----------

# tras el EDA se definió enfocarse en la prediccion de pts 1,3,4
# dado que son los que tienen mayor participación, el pt 2 se utiliza para
# construccion de features (feauture engineering)}
window = Window.partitionBy('client_id').orderBy('mon').rowsBetween(0, 4)

df_target = df_target.withColumns({
    'target_3': f.max(f.col('product_3')).over(window),
})


# COMMAND ----------

# obteniendo la ultima fecha del df para filtrar hasta la fecha
# que permita visualizar la ventana de observacion completa (4 meses)
last_date = df_target.select(f.max('mon').alias('max_mon'))
last_date = last_date.withColumn(
    'max_mon', f.add_months(f.col('max_mon'), -4)
).first()[0]

print(
    f'Fecha hasta donde se puede visualizar la ventana de observacion completa: {last_date}'
)

# COMMAND ----------

# filtrando registros hasta la fecha donde se puede visualizar 
# la ventana de observacion completa
df_target = df_target.filter(
    f.col('mon') <= last_date
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Desbalance por producto

# COMMAND ----------

# desbalance producto 3
df_target_3 = df_target.groupBy('target_3').count()
df_target_3 = df_target_3.withColumn(
  'percentage', f.col('count') / f.sum('count').over(Window.partitionBy())
)
display(df_target_3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Guardado

# COMMAND ----------

target = df_target.select(
    'client_id',
    'mon',
    'target_3',
)

target = target.withColumn(
    'mon', 
    f.to_date(f.date_format(f.col('mon'), 'yyyy-MM-01'), 'yyyy-MM-dd')
)

# COMMAND ----------

path_target = 'abfss://datos@masterccc002sta.dfs.core.windows.net/tfm/data_preparation/target/product_3'
target.write.format('parquet').mode('overwrite').save(path_target)


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Producto 4

# COMMAND ----------

# MAGIC %md
# MAGIC ### Features para estratificación

# COMMAND ----------

df_stratified = df_purchases.withColumn(
  'ind_purchase', f.when(
    f.col('product_4') > 0,
    1
  ).otherwise(0)
)

# segun el EDA los clientes suelen hacer una nueva compra 
# en un lapso de 4 meses tras su ultima compra realizada
window = Window.partitionBy('client_id').orderBy('mon').rowsBetween(0, 4)
w_6m = Window.partitionBy('client_id').orderBy('mon').rowsBetween(-6+1, 0)
w_3m = Window.partitionBy('client_id').orderBy('mon').rowsBetween(-6+1, 0)
df_stratified = df_stratified.withColumns({
    'ind_purchase': f.max(f.col('ind_purchase')).over(window),
})

df_stratified = df_stratified.groupBy('client_id', 'mon').agg(
  f.max('ind_purchase').alias('ind_any_purchase'),
  f.sum('trans_count').alias('total_trans'),
  f.max('trans_count').alias('diff_trans_date'),
).withColumns({
  'avg_trans_3m': f.avg(f.col('total_trans')).over(w_3m),
  'avg_trans_6m': f.avg(f.col('total_trans')).over(w_6m),
  'std_trans_6m': f.std(f.col('total_trans')).over(w_6m),
  'min_trans_date_6m': f.min('diff_trans_date').over(w_6m),
  'max_trans_date_6m': f.max('diff_trans_date').over(w_6m),
  'avg_trans_date_3m': f.avg('diff_trans_date').over(w_3m),
  'avg_trans_date_6m': f.avg('diff_trans_date').over(w_6m),
})

# COMMAND ----------

# segmentando dfs para la estratificacion
df_stratified0 = df_stratified.filter(f.col('ind_any_purchase') == 0)
df_stratified1 = df_stratified.filter(f.col('ind_any_purchase') == 1)

df_stratified0_pd = df_stratified0.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Identificación naturaleza features

# COMMAND ----------

# separando entre X e y
key = ['client_id', 'mon']
target = ['ind_any_purchase']
features = [
  col for col in df_stratified0_pd if col not in key + target
]

X, y = df_stratified0_pd[features], df_stratified0_pd[target]

# COMMAND ----------

numerical_cols, categorical_cols = _identify_type_columns_pandas(
    df=df_stratified0_pd,
    params={
        'id_columns': key,
        'target': target,
    }
)

# COMMAND ----------

# determinando si existen nulos en el df
X.isna().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pipeline preprocesamiento

# COMMAND ----------

# pipeline de preprocesamiento cprt
preprocessing = Pipeline(steps=[
    ('NumericalTransformer', NumericalLogTransformer(
                numerical_cols=numerical_cols,
                threshold_coef=1.5,
        )),
    ('Scaler', ContinuousFeatureScaler(scaler='standard',
                            numerical_cols=numerical_cols))
  ])

preprocessing.fit(X, y)

# COMMAND ----------

# preprocesando df para la estratificación
X_preprocessed = preprocessing.transform(X).fillna(0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reducción dimensionalidad

# COMMAND ----------

# entrenando umap con una muestra de 5%
X_sampled = X_preprocessed.sample(frac=0.05, random_state=42)
umap_reducer = umap.UMAP(
    n_neighbors=100,
    min_dist=0.085,
    metric='euclidean',
    n_components=3,
    n_jobs=-1
).fit(X_sampled)

# COMMAND ----------

# aplicando transformacion umap para todos los datos
df_umap = umap_reducer.transform(X_preprocessed) 
X_umap = pd.DataFrame(df_umap, columns=['umap_1', 'umap_2', 'umap_3'])
X_umap = X_umap.set_index(X_preprocessed.index)

# COMMAND ----------

# asignando cliente y mon al df con dimensionalidad reducida
X_umap['client_id'] = df_stratified0_pd['client_id']
X_umap['mon'] = df_stratified0_pd['mon']

# COMMAND ----------

X_umap_spark = spark.createDataFrame(X_umap)

path_target = 'abfss://datos@masterccc002sta.dfs.core.windows.net/tfm/data_preparation/target/stratification_class_0/target_4/'
X_umap_spark.write.format('parquet').mode('overwrite').save(path_target)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clusters

# COMMAND ----------

X_umap = spark.read.parquet('abfss://datos@masterccc002sta.dfs.core.windows.net/tfm/data_preparation/target/stratification_class_0/target_4/').toPandas()

# COMMAND ----------

# entrenando KMeans
kmeans = KMeans(n_clusters=3, 
                random_state=42, 
                n_init='auto')
labels_kmeans = kmeans.fit_predict(X_umap[['umap_1', 'umap_2', 'umap_3']])

# asignando etiquetas al DataFrame
X_umap['cluster'] = labels_kmeans
df_stratified0_pd['cluster'] = labels_kmeans

# COMMAND ----------

# distribución por cluster
X_umap.groupby(['cluster']).agg(
    obs=('cluster', 'count')
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gráfico

# COMMAND ----------

X_sampled = X_umap.sample(frac=0.25, random_state=42)
fig = px.scatter_3d(X_sampled, 
                    x='umap_1', y='umap_2', z='umap_3', 
                    color='cluster', 
                    opacity=0.7,
)

fig.update_traces(marker=dict(size=4)) # changing the size of each point

fig.update_layout(
    legend_title_text='clusters',
    legend=dict(
        x=1,
        y=1,
        traceorder='normal',
        font=dict(
            family='Arial',
            size=11,
            color='black'        
        ),
        bgcolor='LightSteelBlue',
        bordercolor='Black',
        borderwidth=2,
    ),
    title={
        'text': 'gráfico de dispersión: UMAP == > HDBSCAN',
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
    },
    width=1000,
    height=600,
)

# show the plot
html_str = fig.to_html()
displayHTML(html_str)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Estratificación

# COMMAND ----------

customers_0 = spark.createDataFrame(df_stratified0_pd[['mon', 'client_id', 'cluster']])
customers_1 = df_stratified1.select('mon', 'client_id')

# COMMAND ----------

min_ratio, max_ratio = 0.60, 0.65
random = np.random.uniform(0, 1)

count_1 = customers_1.count()
dist_cluster = customers_0.groupBy('cluster').agg(f.count('*').alias('cnt_cluster'))\
                          .withColumn('pct_cluster', f.col('cnt_cluster') / f.sum('cnt_cluster').over(Window.partitionBy()))
customers_0 = customers_0.withColumn(
    'ratio_0', f.lit(min_ratio) + (f.lit(max_ratio - min_ratio) * f.lit(random)),
).withColumn(
    'stratified_0', f.round((f.col('ratio_0') / (1 - f.col('ratio_0'))) * f.lit(count_1))
)

customers_0 = customers_0.join(dist_cluster.select('cluster', 'pct_cluster'), on='cluster', how='left')

window = Window.partitionBy
customers_0 = customers_0.withColumns({
    'quota_cluster': f.round(f.col('stratified_0') * f.col('pct_cluster')),
    'rand_int': f.round(f.rand(42) * 100)
})

window = Window.partitionBy('cluster').orderBy(f.col('rand_int'))
customers_0 = customers_0.withColumn('row_number', f.row_number().over(window))
customers_0 = customers_0.filter(f.col('row_number') <= f.col('quota_cluster'))


# COMMAND ----------

# consolidando clientes estratificados (sin compra) y clientes con compras
customers_stratified = customers_1.withColumn('class', f.lit(1)).unionByName(
    customers_0.select('mon', 'client_id').withColumn('class', f.lit(0))
)

customers_stratified.groupBy('class').agg(
    f.count('*').alias('count')
).withColumn(
    'percentage', f.col('count') / f.sum('count').over(Window.partitionBy())
).display()

# COMMAND ----------

# obteniendo distribucion final
df_stratified_0 = df_stratified.join(
    customers_stratified.select('mon', 'client_id'), on=['client_id', 'mon'], how='inner'
)

display(
    df_stratified_0.groupBy('ind_any_purchase').agg(
        f.count('*').alias('count')
    ).withColumn(
        'percentage', f.col('count') / f.sum('count').over(Window.partitionBy())
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Definición target

# COMMAND ----------

# obteniendo solo clientes resultantes de la estratificacion
df_target = df_purchases.join(
    customers_stratified, on=['client_id', 'mon'], how='inner'
)

# COMMAND ----------

# tras el EDA se definió enfocarse en la prediccion de pts 1,3,4
# dado que son los que tienen mayor participación, el pt 2 se utiliza para
# construccion de features (feauture engineering)}
window = Window.partitionBy('client_id').orderBy('mon').rowsBetween(0, 4)

df_target = df_target.withColumns({
    'target_4': f.max(f.col('product_4')).over(window),
})


# COMMAND ----------

# obteniendo la ultima fecha del df para filtrar hasta la fecha
# que permita visualizar la ventana de observacion completa (4 meses)
last_date = df_target.select(f.max('mon').alias('max_mon'))
last_date = last_date.withColumn(
    'max_mon', f.add_months(f.col('max_mon'), -4)
).first()[0]

print(
    f'Fecha hasta donde se puede visualizar la ventana de observacion completa: {last_date}'
)

# COMMAND ----------

# filtrando registros hasta la fecha donde se puede visualizar 
# la ventana de observacion completa
df_target = df_target.filter(
    f.col('mon') <= last_date
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Desbalance por producto

# COMMAND ----------

# desbalance producto 4
df_target_4 = df_target.groupBy('target_4').count()
df_target_4 = df_target_4.withColumn(
  'percentage', f.col('count') / f.sum('count').over(Window.partitionBy())
)
display(df_target_4)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Guardado

# COMMAND ----------

target = df_target.select(
    'client_id',
    'mon',
    'target_4',
)

target = target.withColumn(
    'mon', 
    f.to_date(f.date_format(f.col('mon'), 'yyyy-MM-01'), 'yyyy-MM-dd')
)

# COMMAND ----------

path_target = 'abfss://datos@masterccc002sta.dfs.core.windows.net/tfm/data_preparation/target/product_4'
target.write.format('parquet').mode('overwrite').save(path_target)
