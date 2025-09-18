# Databricks notebook source
# MAGIC %md
# MAGIC # TFM: Next Move Banking
# MAGIC **Autor:** Coraima Castillo
# MAGIC
# MAGIC En este notebook se lleva a cabo la evaluación de cada modelo.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importando librerías

# COMMAND ----------

import mlflow.sklearn

# COMMAND ----------

# MAGIC %run /Workspace/Users/coraimac@ucm.es/TFM/utils_functions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importando datos

# COMMAND ----------

container = "datos"
account = spark.conf.get("adls.account.name")
folder = "tfm"
path_table = f"abfss://{container}@{account}.dfs.core.windows.net/{folder}/data_preparation/master_table/"

# COMMAND ----------

# product 1
train_1 = spark.read.parquet(path_table + 'product_1/train_table/').toPandas()
test_1 = spark.read.parquet(path_table + 'product_1/test_table/').toPandas()
X_train_1, y_train_1 = train_1.copy().drop(columns=['client_id', 'target_1']), train_1[['target_1']]
X_test_1, y_test_1 = test_1.copy().drop(columns=['client_id', 'target_1']), test_1[['target_1']]

# product 3
train_3 = spark.read.parquet(path_table + 'product_3/train_table/').toPandas()
test_3 = spark.read.parquet(path_table + 'product_3/test_table/').toPandas()
X_train_3, y_train_3 = train_3.copy().drop(columns=['client_id', 'target_3']), train_3[['target_3']]
X_test_3, y_test_3 = test_3.copy().drop(columns=['client_id', 'target_3']), test_3[['target_3']]

# product 4
train_4 = spark.read.parquet(path_table + 'product_4/train_table/').toPandas()
test_4 = spark.read.parquet(path_table + 'product_4/test_table/').toPandas()
X_train_4, y_train_4 = train_4.copy().drop(columns=['client_id', 'target_4']), train_4[['target_4']]
X_test_4, y_test_4 = test_4.copy().drop(columns=['client_id', 'target_4']), test_4[['target_4']]

# COMMAND ----------

# MAGIC %md
# MAGIC # Product 1

# COMMAND ----------

latest_version = get_latest_model_version("model_product_1")
model_uri = f"models:/model_product_1/{latest_version}"  
model_product_1 = mlflow.sklearn.load_model(model_uri)

print(
    f'Versión del modelo utilizada para Producto 1: {latest_version}'
)

# COMMAND ----------

model_1 = model_product_1['ClassifierProduct1'].model

preprocessing_1 = model_product_1[:-1]
X_test_preprocessed_1 = preprocessing_1.transform(X_test_1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluación

# COMMAND ----------

# MAGIC %md
# MAGIC ### Métricas

# COMMAND ----------

# metrics: training data
print(f"Número de registros en train set: {y_train_1.shape[0]:,.0f}")

y_pred_train_1, y_score_train_1 = model_product_1.predict(X_train_1)

metrics = compute_binary_classification_metrics(
    y_true=y_train_1, y_pred=y_pred_train_1, y_score=y_score_train_1
)
metrics_df_train = pd.DataFrame(list(metrics.items()), columns=['metricas', 'valores']).set_index('metricas')

print(f"Métricas con train set: \n{metrics_df_train}")

# COMMAND ----------

# metrics: testing data
print(f"Número de registros en test set: {y_test_1.shape[0]:,.0f}")

y_pred_test_1, y_score_test_1 = model_product_1.predict(X_test_1)

metrics = compute_binary_classification_metrics(
    y_true=y_test_1, y_pred=y_pred_test_1, y_score=y_score_test_1
)
metrics_df_test = pd.DataFrame(list(metrics.items()), columns=['metricas', 'valores']).set_index('metricas')

print(f"Métricas con test set: \n{metrics_df_test}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gráficos

# COMMAND ----------

# MAGIC %md
# MAGIC #### Curva precisión-recall

# COMMAND ----------

fig_curve_precision = plot_precision_recall_curves(
    y_test_1, y_score_test_1
)

fig_curve_precision

# COMMAND ----------

# MAGIC %md
# MAGIC #### Curva KS (Kolmogorov-Smirnov)

# COMMAND ----------

fig_ks_curve, ks_threshold = plot_ks_curve(y_test_1, y_score_test_1)

fig_ks_curve

# COMMAND ----------

# MAGIC %md
# MAGIC #### Curva Lorenz (Gini)

# COMMAND ----------

fig_lorenz_curve = plot_gini_curve(y_test_1, y_score_test_1)

fig_lorenz_curve

# COMMAND ----------

# MAGIC %md
# MAGIC #### Curva calibración

# COMMAND ----------

fig_calibration_curve = plot_calibration_curve(y_test_1, y_score_test_1)

fig_calibration_curve

# COMMAND ----------

# MAGIC %md
# MAGIC #### Matriz confusión

# COMMAND ----------

fig_confusion_matrix = plot_confusion_matrix(
    y_test_1, y_pred_test_1,
    labels=('no-recomendado', 'recomendado')
)

fig_confusion_matrix

# COMMAND ----------

# MAGIC %md
# MAGIC #### Curva ROC

# COMMAND ----------

fig_roc_curve = plot_roc_and_optimal_thresholds(y_test_1, y_score_test_1)

fig_roc_curve

# COMMAND ----------

# MAGIC %md
# MAGIC #### Shap values

# COMMAND ----------

fig_shap_values, shap_values = shap_values_classifier(
    model=model_1,
    X_preprocessed=X_test_preprocessed_1.sample(frac=0.5, random_state=42),
    max_features=50,
)

fig_shap_values

# COMMAND ----------

# MAGIC %md
# MAGIC # Product 3

# COMMAND ----------

latest_version = get_latest_model_version("model_product_3")
model_uri = f"models:/model_product_3/{latest_version}"   
model_product_3 = mlflow.sklearn.load_model(model_uri)

print(
    f'Versión del modelo utilizada para Producto 3: {latest_version}'
)

# COMMAND ----------

model_3 = model_product_3['ClassifierProduct3'].model

preprocessing_3 = model_product_3[:-1]
X_test_preprocessed_3 = preprocessing_3.transform(X_test_3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluación

# COMMAND ----------

# MAGIC %md
# MAGIC ### Métricas

# COMMAND ----------

# metrics: training data
print(f"Número de registros en train set: {y_train_3.shape[0]:,.0f}")

y_pred_train_3, y_score_train_3 = model_product_3.predict(X_train_3)

metrics = compute_binary_classification_metrics(
    y_true=y_train_3, y_pred=y_pred_train_3, y_score=y_score_train_3
)
metrics_df_train = pd.DataFrame(list(metrics.items()), columns=['metricas', 'valores']).set_index('metricas')

print(f"Métricas con train set: \n{metrics_df_train}")

# COMMAND ----------

# metrics: testing data
print(f"Número de registros en test set: {y_test_3.shape[0]:,.0f}")

y_pred_test_3, y_score_test_3 = model_product_3.predict(X_test_3)

metrics = compute_binary_classification_metrics(
    y_true=y_test_3, y_pred=y_pred_test_3, y_score=y_score_test_3
)
metrics_df_test = pd.DataFrame(list(metrics.items()), columns=['metricas', 'valores']).set_index('metricas')

print(f"Métricas con test set: \n{metrics_df_test}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gráficos

# COMMAND ----------

# MAGIC %md
# MAGIC #### Curva precisión-recall

# COMMAND ----------

fig_curve_precision = plot_precision_recall_curves(
    y_test_3, y_score_test_3
)

fig_curve_precision

# COMMAND ----------

# MAGIC %md
# MAGIC #### Curva KS (Kolmogorov-Smirnov)

# COMMAND ----------

fig_ks_curve, ks_threshold = plot_ks_curve(y_test_3, y_score_test_3)

fig_ks_curve

# COMMAND ----------

# MAGIC %md
# MAGIC #### Curva Lorenz (Gini)

# COMMAND ----------

fig_lorenz_curve = plot_gini_curve(y_test_3, y_score_test_3)

fig_lorenz_curve

# COMMAND ----------

# MAGIC %md
# MAGIC #### Curva calibración

# COMMAND ----------

fig_calibration_curve = plot_calibration_curve(y_test_3, y_score_test_3)

fig_calibration_curve

# COMMAND ----------

# MAGIC %md
# MAGIC #### Matriz confusión

# COMMAND ----------

fig_confusion_matrix = plot_confusion_matrix(
    y_test_3, y_pred_test_3,
    labels=('no-recomendado', 'recomendado')
)

fig_confusion_matrix

# COMMAND ----------

# MAGIC %md
# MAGIC #### Curva ROC

# COMMAND ----------

fig_roc_curve = plot_roc_and_optimal_thresholds(y_test_3, y_score_test_3)

fig_roc_curve

# COMMAND ----------

# MAGIC %md
# MAGIC #### Shap values

# COMMAND ----------

fig_shap_values, shap_values = shap_values_classifier(
    model=model_3,
    X_preprocessed=X_test_preprocessed_3.sample(frac=0.5, random_state=42),
    max_features=50,
)

fig_shap_values

# COMMAND ----------

# MAGIC %md
# MAGIC # Product 4

# COMMAND ----------

latest_version = get_latest_model_version("model_product_4")
model_uri = f"models:/model_product_4/{latest_version}"  
model_product_4 = mlflow.sklearn.load_model(model_uri)

print(
    f'Versión del modelo utilizada para Producto 4: {latest_version}'
)

# COMMAND ----------

model_4 = model_product_4['ClassifierProduct4'].model

preprocessing_4 = model_product_4[:-1]
X_test_preprocessed_4 = preprocessing_4.transform(X_test_4)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluación

# COMMAND ----------

# MAGIC %md
# MAGIC ### Métricas

# COMMAND ----------

# metrics: training data
print(f"Número de registros en train set: {y_train_4.shape[0]:,.0f}")

y_pred_train_4, y_score_train_4 = model_product_4.predict(X_train_4)

metrics = compute_binary_classification_metrics(
    y_true=y_train_4, y_pred=y_pred_train_4, y_score=y_score_train_4
)
metrics_df_train = pd.DataFrame(list(metrics.items()), columns=['metricas', 'valores']).set_index('metricas')

print(f"Métricas con train set: \n{metrics_df_train}")

# COMMAND ----------

# metrics: testing data
print(f"Número de registros en test set: {y_test_4.shape[0]:,.0f}")

y_pred_test_4, y_score_test_4 = model_product_4.predict(X_test_4)

metrics = compute_binary_classification_metrics(
    y_true=y_test_4, y_pred=y_pred_test_4, y_score=y_score_test_4
)
metrics_df_test = pd.DataFrame(list(metrics.items()), columns=['metricas', 'valores']).set_index('metricas')

print(f"Métricas con test set: \n{metrics_df_test}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gráficos

# COMMAND ----------

# MAGIC %md
# MAGIC #### Curva precisión-recall

# COMMAND ----------

fig_curve_precision = plot_precision_recall_curves(
    y_test_4, y_score_test_4
)

fig_curve_precision

# COMMAND ----------

# MAGIC %md
# MAGIC #### Curva KS (Kolmogorov-Smirnov)

# COMMAND ----------

fig_ks_curve, ks_threshold = plot_ks_curve(y_test_4, y_score_test_4)

fig_ks_curve

# COMMAND ----------

# MAGIC %md
# MAGIC #### Curva Lorenz (Gini)

# COMMAND ----------

fig_lorenz_curve = plot_gini_curve(y_test_4, y_score_test_4)

fig_lorenz_curve

# COMMAND ----------

# MAGIC %md
# MAGIC #### Curva calibración

# COMMAND ----------

fig_calibration_curve = plot_calibration_curve(y_test_4, y_score_test_4)

fig_calibration_curve

# COMMAND ----------

# MAGIC %md
# MAGIC #### Matriz confusión

# COMMAND ----------

fig_confusion_matrix = plot_confusion_matrix(
    y_test_4, y_pred_test_4,
    labels=('no-recomendado', 'recomendado')
)

fig_confusion_matrix

# COMMAND ----------

# MAGIC %md
# MAGIC #### Curva ROC

# COMMAND ----------

fig_roc_curve = plot_roc_and_optimal_thresholds(y_test_4, y_score_test_4)

fig_roc_curve

# COMMAND ----------

# MAGIC %md
# MAGIC #### Shap values

# COMMAND ----------

fig_shap_values, shap_values = shap_values_classifier(
    model=model_4,
    X_preprocessed=X_test_preprocessed_4.sample(frac=0.5, random_state=42),
    max_features=50,
)

fig_shap_values