# Databricks notebook source
# MAGIC %md
# MAGIC # TFM: Next Move Banking
# MAGIC **Autor:** Coraima Castillo
# MAGIC
# MAGIC En este notebook se lleva a cabo el modelado por cada producto. Para cada producto se realiza optimización de hiperparámetros y se escoge un modelo final acorde a su rendimiento por fold temporal.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Instalando dependencias

# COMMAND ----------

!pip install optuna

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importando librerías

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

from sklearn.pipeline import Pipeline
import optuna

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# COMMAND ----------

# MAGIC %run /Workspace/Users/coraimac@ucm.es/TFM/utils_functions

# COMMAND ----------

import mlflow
mlflow.autolog(disable=True)

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

# product 3
train_3 = spark.read.parquet(path_table + 'product_3/train_table/').toPandas()
test_3 = spark.read.parquet(path_table + 'product_3/test_table/').toPandas()


# product 4
train_4 = spark.read.parquet(path_table + 'product_4/train_table/').toPandas()
test_4 = spark.read.parquet(path_table + 'product_4/test_table/').toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC # Product 1

# COMMAND ----------

# seteando como indice mon
train_1['mon'] = pd.to_datetime(train_1['mon'])
train_1 = train_1.set_index('mon')

test_1['mon'] = pd.to_datetime(test_1['mon'])
test_1 = test_1.set_index('mon')

# COMMAND ----------

# dividiedo en X-y train and test
target = 'target_1'

X_train_1, y_train_1 = train_1.copy().drop(columns=['client_id', 'target_1']), train_1[[target]]
X_test_1, y_test_1 = test_1.copy().drop(columns=['client_id', 'target_1']), test_1[[target]]

# COMMAND ----------

fig, ax = plt.subplots(1,2,figsize=(9, 3.5))

sns.barplot(y_train_1.value_counts(normalize=True).reset_index(), y=0, x='target_1', color='#00968B', ax=ax[0], width=0.6)
sns.barplot(y_test_1.value_counts(normalize=True).reset_index(), y=0, x='target_1', color='#FED700', ax=ax[1], width=0.6)

ax[0].set_title('desbalance en y_train', color='#00968B')
ax[1].set_title('desbalance en y_test', color='orange')

ax[0].set_xlabel('target: target_1')
ax[1].set_xlabel('target: target_1')

for p in ax[0].patches:
    height = p.get_height()
    ax[0].annotate(
        f'{height:.2%}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', xytext=(0, 5), textcoords='offset points'
    )

for p in ax[1].patches:
    height = p.get_height()
    ax[1].annotate(
        f'{height:.2%}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', xytext=(0, 5), textcoords='offset points'
    )

plt.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Features categóricas & numéricas

# COMMAND ----------

# obtencion de features numericas y categoricas
numerical_cols, categorical_cols = _identify_type_columns_pandas(
    df=X_train_1,
    params={
        'id_columns': ['mon', 'client_id'],
        'target': ['target_1', 'target_3', 'target_4'],
        'features_exclude': {
            'numerical': ['trans_dst_type_mode_m', 'trans_event_type_mode_m'],
            'categorical': ['dialog_avg_per_day_m'],
        },
    }
)

features_product_1 = numerical_cols + categorical_cols

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline preprocesamiento

# COMMAND ----------

categorical_cols

# COMMAND ----------

preprocessing_1 = Pipeline(steps=[
    ('FeatureSelector', ColumnsSelector(columns=features_product_1)),
    ('OutliersHandler', OutliersHandler(
                            numeric_cols=numerical_cols,
                            exclude_cols=['trans_type_currency_cnt_m'])),
    ('CategoricalNullImputer', CategoricalNullImputer(
                                    categorical_cols=categorical_cols,
                                    exclude_cols={
                                        'pur_any_purchase': 0,
                                    })),
    ('NumericalNullImputer', NumericalNullImputer(
                                     numerical_cols=numerical_cols,
                                     tipo='aleatorio',
                                     exclude_cols={
                                         'trans_type_currency_cnt_m': 1,
                                     })),
    ('WoEEncoder', WoEEncoder(
                            woe_sim_threshold=0.1,
                            min_bin_pct=0.1,
                            categorical_cols=categorical_cols,
                            exclude_cols=['pur_any_purchase']        
    )),
    ('NumericalTransformer', NumericalLogTransformer(
                                     numerical_cols=numerical_cols,
                                     threshold_coef=1)),
    ('Scaler', ContinuousFeatureScaler(scaler='standard',
                            numerical_cols=numerical_cols))
])

preprocessing_1.fit(X_train_1, y_train_1)

# COMMAND ----------

# preprocessing de data train producto 1
X_preprocessed_1 = preprocessing_1.transform(X_train_1)

# COMMAND ----------

display(X_preprocessed_1.nunique().reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimización hiperparámetros

# COMMAND ----------

# MAGIC %md
# MAGIC ### LightGBM

# COMMAND ----------

objective_function = 'binary'
eval_metric = 'binary_logloss'

def objective(trial):
    # generando particiones de fechas para el backtesting
    start_date = y_train_1.index.min() + pd.DateOffset(months=1)
    end_date = y_train_1.index.max()
    freq = 'monthly'

    backtesting_dates = _generate_backtesting_dates(
        freq=freq, 
        start_date=start_date, 
        end_date=end_date
    )

    # seteando los parametros a optimizar
    params = {
        'objective': objective_function,
        'metric': eval_metric,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 8, 12),
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.01, 0.95),
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42,
    }

    threshold_prob = 0.5
    n=trial.suggest_categorical('n', [None] + list(range(1, 10)))

    # almacenamiento metricas
    f1_scores = []
    precisions = []
    recalls = []
    kss = []

    for date in backtesting_dates:
        # definiendo particiones - backtesting
        next_date = _sum_backtesting_frequency(date=date, frequency=freq)
        string_date = date.strftime("%Y-%m-%d")
        string_next_date = next_date.strftime("%Y-%m-%d")

        X_train_bt = X_preprocessed_1[X_preprocessed_1.index < date]
        X_test_bt = X_preprocessed_1[(X_preprocessed_1.index >= date) \
                                    & (X_preprocessed_1.index < next_date)]

        y_train_1_bt = y_train_1[y_train_1.index < date]
        y_test_bt = y_train_1[(y_train_1.index >= date) & (y_train_1.index < next_date)]

        if (X_train_bt.index.max() > date) or (y_train_1_bt.index.max() > date):
            msg = (
                "Fuga de información al método fit "
                f"Max date on X_train is {X_train_bt.index.max()} "
                f"Max date on y_train_1 {y_train_1_bt.index.max()} "
                f"than backtesting date: {date}"
            )
            raise ValueError(msg)

        # seteado y entrenando el modelo
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train_bt, y_train_1_bt.to_numpy().ravel(),
            eval_set=[(X_test_bt, y_test_bt)],
            sample_weight=get_sample_weight(y_train_1_bt, n=n) if n is not None else None
        )

        # evaluando el modelo entrenado
        y_score = None

        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test_bt)
            y_probs = y_score[:, 1]
            y_pred = np.where(y_probs > threshold_prob, 1, 0)

        else:
            y_pred = model.predict(X_test_bt)

        # metrica f1
        f1 = f1_score(y_test_bt, y_pred)

        # metrica precision
        if len(np.unique(y_pred)) < 2 or len(np.unique(y_test_bt)) < 2:
            precision = 0  # evita el error en precision_score()
        else:
            precision = precision_score(y_test_bt, y_pred)

        # metrica recall
        recall = recall_score(y_test_bt, y_pred)

        # metrica ks
        ks = _compute_ks_statistic(y_true=y_test_bt['target_1'], y_score=y_probs)

        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        kss.append(ks)


    mean_f1 = np.mean(f1_scores)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    ks = np.mean(kss)

    print(
        f">>> F1 mean: {mean_f1:.5f} | Precision mean: {mean_precision:.5f} | Recall mean: {mean_recall:.5f} | KS mean: {ks:.5f}\n"
        f"F1 - scores: {f1_scores}\n"
        f"Precision - scores: {precisions}\n"
        f"Recall - scores: {recalls}\n"
        f"KS - scores: {kss}"
    )

    return mean_f1

# COMMAND ----------

# optimization
name_model = 'lightgbm'
study = optuna.create_study(
    study_name=f'class_product_1_{name_model}_optimization',
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5, interval_steps=1),
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(objective, n_trials=15)

best_params = study.best_params

# COMMAND ----------

print(
  f'Mejores hiperparámetros usando LightGBM: {best_params}.'
)

# Mejores hiperparámetros usando LightGBM: {'learning_rate': 0.09678610700324558, 'num_leaves': 139, 'max_depth': 11, 'n_estimators': 2470, 'subsample': 0.5895665919053817, 'colsample_bytree': 0.538197213946161, 'reg_lambda': 0.009581019913794751, 'reg_alpha': 0.8330203547567471, 'min_gain_to_split': 0.7831162075649054, 'n': 2}.


# COMMAND ----------

# MAGIC %md
# MAGIC ### RandomForest

# COMMAND ----------

def objective(trial):
    # particiones de fechas para el backtesting
    start_date = y_train_1.index.min() + pd.DateOffset(months=1)
    end_date = y_train_1.index.max()
    freq = 'monthly'

    backtesting_dates = _generate_backtesting_dates(
        freq=freq, 
        start_date=start_date, 
        end_date=end_date
    )

    # parámetros a optimizar para Random Forest
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 100),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'bootstrap': True,
        'n_jobs': -1,
        'random_state': 42,
    }

    threshold_prob = 0.5
    n=trial.suggest_categorical('n', [None] + list(range(1, 10)))

    f1_scores = []
    precisions = []
    recalls = []
    kss = []

    for date in backtesting_dates:
        next_date = _sum_backtesting_frequency(date=date, frequency=freq)

        X_train_bt = X_preprocessed_1[X_preprocessed_1.index < date]
        X_test_bt = X_preprocessed_1[(X_preprocessed_1.index >= date) & (X_preprocessed_1.index < next_date)]

        y_train_1_bt = y_train_1[y_train_1.index < date]
        y_test_bt = y_train_1[(y_train_1.index >= date) & (y_train_1.index < next_date)]

        # entrenando modelo Random Forest
        model = RandomForestClassifier(**params)
        model.fit(
            X_train_bt, y_train_1_bt.to_numpy().ravel(), 
            sample_weight=get_sample_weight(y_train_1_bt, n=n) if n is not None else None
        )

        # obtener probabilidades y ajustar umbral
        y_probs = model.predict_proba(X_test_bt)[:, 1]
        y_pred = (y_probs > threshold_prob).astype(int)

        f1 = f1_score(y_test_bt, y_pred)

        if len(np.unique(y_pred)) < 2 or len(np.unique(y_test_bt)) < 2:
            precision = 0
        else:
            precision = precision_score(y_test_bt, y_pred)

        recall = recall_score(y_test_bt, y_pred)

        ks = _compute_ks_statistic(y_true=y_test_bt['target_1'], y_score=y_probs)


        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        kss.append(ks)

    mean_f1 = np.mean(f1_scores)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_ks = np.mean(kss)

    print(
        f">>> F1 mean: {mean_f1:.5f} | Precision mean: {mean_precision:.5f} | Recall mean: {mean_recall:.5f} | KS mean: {mean_ks:.5f}\n"
        f"F1 - scores: {f1_scores}\n"
        f"Precision - scores: {precisions}\n"
        f"Recall - scores: {recalls}\n"
        f"KS: {kss}"
    )

    return mean_f1

# COMMAND ----------

# optimization
name_model = 'random_forest'
study = optuna.create_study(
    study_name=f'class_product_1_{name_model}_optimization',
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5, interval_steps=1),
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(objective, n_trials=15)

best_params = study.best_params

# COMMAND ----------

print(
  f'Mejores hiperparámetros usando RandomForest: {best_params}.'
)

# Mejores hiperparámetros usando RandomForest: {'n_estimators': 1534, 'max_depth': 11, 'min_samples_split': 92, 'min_samples_leaf': 54, 'max_features': 'sqrt', 'criterion': 'gini', 'n': 2}.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Regresión Logística

# COMMAND ----------

def objective(trial):
    # particiones de fechas para el backtesting
    start_date = y_train_1.index.min() + pd.DateOffset(months=1)
    end_date = y_train_1.index.max()
    freq = 'monthly'

    backtesting_dates = _generate_backtesting_dates(
        freq=freq, 
        start_date=start_date, 
        end_date=end_date
    )

    solver_penalty = trial.suggest_categorical('solver_penalty', [('saga', 'l1'), ('lbfgs', 'l2'), ('saga', 'l2'), ('liblinear', 'l1'), ('liblinear', 'l2')])
    solver, penalty = solver_penalty

    # parametros a optimizar en la regresion logistica
    params = {
        'C': trial.suggest_float('C', 0.001, 0.1),  # regularizacion
        'penalty': penalty,
        'solver': solver,
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'max_iter': trial.suggest_int('max_iter', 2450, 3500),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'random_state': 42,
        'n_jobs': -1,
    }

    threshold_prob = 0.5 # trial.suggest_float('threshold_prob', 0.35, 0.65)
    n=trial.suggest_categorical('n', [None] + list(range(1, 10)))

    f1_scores = []
    precisions = []
    recalls = []
    kss = []


    for date in backtesting_dates:
        # definiendo particiones para el backtesting
        next_date = _sum_backtesting_frequency(date=date, frequency=freq)

        X_train_bt = X_preprocessed_1[X_preprocessed_1.index < date]
        X_test_bt = X_preprocessed_1[(X_preprocessed_1.index >= date) & (X_preprocessed_1.index < next_date)]

        y_train_1_bt = y_train_1[y_train_1.index < date]
        y_test_bt = y_train_1[(y_train_1.index >= date) & (y_train_1.index < next_date)]

        # entrenando modelo
        model = LogisticRegression(**params)
        model.fit(
            X_train_bt, y_train_1_bt.to_numpy().ravel(),
            sample_weight=get_sample_weight(y_train_1_bt, n=n) if n is not None else None
        )

        # obtener probabilidades y ajustar umbral
        y_probs = model.predict_proba(X_test_bt)[:, 1]
        y_pred = (y_probs > threshold_prob).astype(int)
        
        # metrica f1-score
        f1 = f1_score(y_test_bt, y_pred)
        # metrica precision
        if len(np.unique(y_pred)) < 2 or len(np.unique(y_test_bt)) < 2:
            precision = 0  # evita el error en precision_score()
        else:
            precision = precision_score(y_test_bt, y_pred)
        # metrica recall
        recall = recall_score(y_test_bt, y_pred)

        ks = _compute_ks_statistic(y_true=y_test_bt['target_1'], y_score=y_probs)


        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        kss.append(ks)

    mean_f1 = np.mean(f1_scores)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)

    mean_ks = np.mean(kss)

    print(
        f">>> F1 mean: {mean_f1:.5f} | Precision mean: {mean_precision:.5f} | Recall mean: {mean_recall:.5f}\n"
        f"F1 - scores: {f1_scores}\n"
        f"Precision - scores: {precisions}\n"
        f"Recall - scores: {recalls}\n"
        f"KS: {kss}"
    )

    return mean_f1

# COMMAND ----------

# optimization
name_model = 'reg_log'
study = optuna.create_study(
    study_name=f'class_product_1_{name_model}_optimization',
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5, interval_steps=1),
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(objective, n_trials=15)

best_params = study.best_params

# COMMAND ----------

print(
  f'Mejores hiperparámetros usando Regresión Logística: {best_params}.'
)

# Mejores hiperparámetros usando Regresión Logística: {'solver_penalty': ('saga', 'l1'), 'C': 0.07556921500862075, 'class_weight': 'balanced', 'max_iter': 3031, 'fit_intercept': True, 'n': 8}.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Selección de modelos

# COMMAND ----------

# modelos con los mejores hiperparámetros obtenidos
model_lgbm_1 = lgb.LGBMClassifier(**{
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.09678610700324558, 
    'num_leaves': 139, 
    'max_depth': 11, 
    'n_estimators': 2470, 
    'subsample': 0.5895665919053817, 
    'colsample_bytree': 0.538197213946161, 
    'reg_lambda': 0.009581019913794751, 
    'reg_alpha': 0.8330203547567471, 
    'min_gain_to_split': 0.7831162075649054,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
})

model_rf_1 = RandomForestClassifier(**{
    'n_estimators': 1534, 
    'max_depth': 11, 
    'min_samples_split': 92, 
    'min_samples_leaf': 54, 
    'max_features': 'sqrt', 
    'criterion': 'gini', 
    'bootstrap': True,
    'n_jobs': -1,
    'random_state': 42,
})

model_reglog_1 = LogisticRegression(**{
    'solver': 'saga',
    'penalty': 'l1',
    'C': 0.07556921500862075, 
    'class_weight': 'balanced', 
    'max_iter': 3031, 
    'fit_intercept': True, 
    'random_state': 42,
    'n_jobs': -1,
})

# COMMAND ----------

start_date = y_train_1.index.min() + pd.DateOffset(months=1)
end_date = y_train_1.index.max()

backtesting_dates = _generate_backtesting_dates(
    freq='monthly', 
    start_date=start_date, 
    end_date=end_date
)

i=1
threshold_prob = 0.5

metrics_fold = {
    'model': [],
    'pr_auc': [],
    'roc_auc': [],
    'brier_score': [],
} 
for date in backtesting_dates:
        before_date = date - pd.DateOffset(months=1)
        next_date = _sum_backtesting_frequency(date=date, frequency='monthly')

        X_train_kf = X_preprocessed_1[(X_preprocessed_1.index >= before_date) & (X_preprocessed_1.index < date)]
        X_test_kf = X_preprocessed_1[(X_preprocessed_1.index >= date) & (X_preprocessed_1.index < next_date)]

        y_train_kf = y_train_1[(y_train_1.index >= before_date) & (y_train_1.index < date)]
        y_test_kf = y_train_1[(y_train_1.index >= date) & (y_train_1.index < next_date)]

        print(
            f'Fechas para el fold {i}: {y_train_kf.index.min()} - {y_test_kf.index.max()}\n'
            f'Training set del fold: {y_train_kf.index.min()} - {y_train_kf.index.max()}\n'
            f'Testing set del fold: {y_test_kf.index.min()} - {y_test_kf.index.max()}\n'
        )

        # entrenamiento de los modelos
        model_lgbm_1.fit(
            X_train_kf, y_train_kf.to_numpy().ravel(),
            eval_set=[(X_test_kf, y_test_kf)],
            sample_weight=get_sample_weight(y_train_kf, n=2),
            callbacks=[lgb.early_stopping(stopping_rounds=50)],
        )

        model_rf_1.fit(
            X_train_kf, y_train_kf.to_numpy().ravel(),
            sample_weight=get_sample_weight(y_train_kf, n=2),
        )

        model_reglog_1.fit(
            X_train_kf, y_train_kf.to_numpy().ravel(),
            sample_weight=get_sample_weight(y_train_kf, n=8),
        )

        # testeando los modelos
        # realizando predicciones
        y_score_lgbm = model_lgbm_1.predict_proba(X_test_kf)[:, 1]
        y_pred_lgbm = (y_score_lgbm > threshold_prob).astype(int)

        y_score_rf = model_rf_1.predict_proba(X_test_kf)[:, 1]
        y_pred_rf = (y_score_rf > threshold_prob).astype(int)

        y_score_reglog = model_reglog_1.predict_proba(X_test_kf)[:, 1]
        y_pred_reglog = (y_score_reglog > threshold_prob).astype(int)

        # calculando metricas
        roc_auc_lgbm = roc_auc_score(y_test_kf, y_score_lgbm)
        brier_score_lgbm = brier_score_loss(y_test_kf, y_score_lgbm)
        precision, recall, _ = precision_recall_curve(y_test_kf, y_score_lgbm)
        pr_auc_lgbm = auc(recall, precision)     

        roc_auc_rf = roc_auc_score(y_test_kf, y_score_rf)
        brier_score_rf = brier_score_loss(y_test_kf, y_score_rf)
        precision, recall, _ = precision_recall_curve(y_test_kf, y_score_rf)
        pr_auc_rf = auc(recall, precision)

        roc_auc_reglog = roc_auc_score(y_test_kf, y_score_reglog)
        brier_score_reglog = brier_score_loss(y_test_kf, y_score_reglog)
        precision, recall, _ = precision_recall_curve(y_test_kf, y_score_reglog)
        pr_auc_reglog = auc(recall, precision)

        metrics_fold['model']+=['lightgbm', 'random_forest', 'reg_log']
        metrics_fold['pr_auc']+=[pr_auc_lgbm, pr_auc_rf, pr_auc_reglog]
        metrics_fold['roc_auc']+=[roc_auc_lgbm, roc_auc_rf, roc_auc_reglog]
        metrics_fold['brier_score']+=[brier_score_lgbm, brier_score_rf, brier_score_reglog]

        i+=1

# COMMAND ----------

# creando df con las metricas obtenidas de c/modelo
eval_models_1 = pd.DataFrame(metrics_fold)

# describiendo las metricas
display(eval_models_1.groupby('model').describe().T.reset_index())

# COMMAND ----------

palette = {
    'lightgbm': '#FB4F4F',      
    'random_forest': '#57F384', 
    'reg_log': '#16DBEA'       
}

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
sns.set_style('darkgrid')

sns.violinplot(data=eval_models_1, x='model', y='pr_auc', ax=ax[0], palette=palette)
ax[0].set_xlabel('modelos')
ax[0].set_ylabel('pr-auc')
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

sns.violinplot(data=eval_models_1, x='model', y='roc_auc', ax=ax[1], palette=palette)
ax[1].set_xlabel('modelos')
ax[1].set_ylabel('roc-auc')
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

sns.violinplot(data=eval_models_1, x='model', y='brier_score', ax=ax[2], palette=palette)
ax[2].set_xlabel('modelos')
ax[2].set_ylabel('brier-score')
ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.suptitle('Métricas de evaluación de los modelos')

plt.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC En la evaluación se utilizaron folds temporales con el fin de respetar la naturaleza secuencial de los datos, evitando fugas de información y simulando escenarios reales donde el modelo debe predecir el comportamiento futuro a partir de información pasada. 
# MAGIC
# MAGIC Se consideraron tres métricas: PR-AUC, ROC-AUC y Brier Score. El PR-AUC es la métrica más relevante en este contexto porque mide la capacidad de identificar correctamente a los clientes compradores en un escenario altamente desbalanceado, priorizando la clase positiva que es la de interés para el banco. El ROC-AUC complementa el análisis al mostrar la discriminación global del modelo en todos los umbrales, mientras que el Brier Score permite evaluar la calibración de las probabilidades, aspecto útil si se usan directamente como insumo en decisiones de negocio. Además de los valores medios, se analizó la varianza de las métricas entre folds, ya que una baja dispersión indica mayor estabilidad en el tiempo y por tanto mayor confiabilidad para entornos productivos. 
# MAGIC
# MAGIC Bajo estos criterios, para el producto 1, Random Forest fue el más adecuado al presentar los mejores resultados en PR-AUC y ROC-AUC, junto con una varianza aceptable, lo que lo convierte en la mejor opción para maximizar la efectividad de las campañas y asegurar robustez frente a cambios temporales en el comportamiento de los clientes.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training modelo final

# COMMAND ----------

params_model_1 = {
    'n_estimators': 1534, 
    'max_depth': 11, 
    'min_samples_split': 92, 
    'min_samples_leaf': 54, 
    'max_features': 'sqrt', 
    'criterion': 'gini', 
    'bootstrap': True,
    'n_jobs': -1,
    'random_state': 42,
}

fit_params_1 = {
    'sample_weight': get_sample_weight(y_train_1, n=2)
}

# COMMAND ----------

model_product_1 = Pipeline(steps=[
    ('FeatureSelector', ColumnsSelector(columns=features_product_1)),
    ('OutliersHandler', OutliersHandler(
                            numeric_cols=numerical_cols,
                            exclude_cols=['trans_type_currency_cnt_m'])),
    ('CategoricalNullImputer', CategoricalNullImputer(
                                    categorical_cols=categorical_cols,
                                    exclude_cols={
                                        'pur_any_purchase': 0,
                                    })),
    ('NumericalNullImputer', NumericalNullImputer(
                                     numerical_cols=numerical_cols,
                                     tipo='aleatorio',
                                     exclude_cols={
                                         'trans_type_currency_cnt_m': 1,
                                     })),
    ('NumericalTransformer', NumericalLogTransformer(
                                     numerical_cols=numerical_cols,
                                     threshold_coef=1)),
    ('Scaler', ContinuousFeatureScaler(scaler='standard',
                            numerical_cols=numerical_cols)),
    ('ClassifierProduct1', ClassifierByProduct(
                                        model_cls=RandomForestClassifier,
                                        model_params=params_model_1,
                                        fit_params=fit_params_1,
                                        threshold=0.45454545454545454545,
                                    ))
])

model_product_1.fit(X_train_1, y_train_1)

# COMMAND ----------

# guardando modelo en MLFLow
with mlflow.start_run(run_name='model_product_1'):
    mlflow.sklearn.log_model(
        model_product_1,
        'model_product_1',
        registered_model_name='model_product_1',
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # Product 3

# COMMAND ----------

# seteando como indice mon
train_3['mon'] = pd.to_datetime(train_3['mon'])
train_3 = train_3.set_index('mon')

test_3['mon'] = pd.to_datetime(test_3['mon'])
test_3 = test_3.set_index('mon')

# COMMAND ----------

# dividiedo en X-y train and test
target = 'target_3'

X_train_3, y_train_3 = train_3.copy().drop(columns=['client_id', 'target_3']), train_3[[target]]
X_test_3, y_test_3 = test_3.copy().drop(columns=['client_id', 'target_3']), test_3[[target]]

# COMMAND ----------

fig, ax = plt.subplots(1,2,figsize=(9, 3.5))

sns.barplot(y_train_3.value_counts(normalize=True).reset_index(), y=0, x='target_3', color='#00968B', ax=ax[0], width=0.6)
sns.barplot(y_test_3.value_counts(normalize=True).reset_index(), y=0, x='target_3', color='#FED700', ax=ax[1], width=0.6)

ax[0].set_title('desbalance en y_train', color='#00968B')
ax[1].set_title('desbalance en y_test', color='orange')

ax[0].set_xlabel('target: target_3')
ax[1].set_xlabel('target: target_3')

for p in ax[0].patches:
    height = p.get_height()
    ax[0].annotate(
        f'{height:.2%}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', xytext=(0, 5), textcoords='offset points'
    )

for p in ax[1].patches:
    height = p.get_height()
    ax[1].annotate(
        f'{height:.2%}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', xytext=(0, 5), textcoords='offset points'
    )

plt.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Features categóricas & numéricas

# COMMAND ----------

# obtencion de features numericas y categoricas
numerical_cols, categorical_cols = _identify_type_columns_pandas(
    df=X_train_3,
    params={
        'id_columns': ['mon', 'client_id'],
        'target': ['target_1', 'target_3', 'target_4'],
        'features_exclude': {
            'numerical': ['trans_dst_type_mode_m', 'trans_event_type_mode_m'],
            'categorical': ['dialog_avg_per_day_m', 'pur_product_3_rsum_m'],
        },
    }
)

features_product_3 = numerical_cols + categorical_cols

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline preprocesamiento

# COMMAND ----------

categorical_cols

# COMMAND ----------

preprocessing_3 = Pipeline(steps=[
    ('FeatureSelector', ColumnsSelector(columns=features_product_3)),
    ('OutliersHandler', OutliersHandler(
                            numeric_cols=numerical_cols,
                            exclude_cols=['trans_type_currency_cnt_m'])),
    ('CategoricalNullImputer', CategoricalNullImputer(
                                    categorical_cols=categorical_cols,
                                    exclude_cols={
                                        'pur_any_purchase': 0,
                                        'trx_growth_streak_3m': 0,
                                    })),
    ('NumericalNullImputer', NumericalNullImputer(
                                     numerical_cols=numerical_cols,
                                     tipo='aleatorio',
                                     exclude_cols={
                                         'trans_type_currency_cnt_m': 1,
                                     })),
    ('WoEEncoder', WoEEncoder(
                            woe_sim_threshold=0.1,
                            min_bin_pct=0.1,
                            categorical_cols=categorical_cols,
                            exclude_cols=['pur_any_purchase', 'trx_growth_streak_3m']        
    )),
    ('NumericalTransformer', NumericalLogTransformer(
                                     numerical_cols=numerical_cols,
                                     threshold_coef=1)),
    ('Scaler', ContinuousFeatureScaler(scaler='standard',
                            numerical_cols=numerical_cols))
])

preprocessing_3.fit(X_train_3, y_train_3)

# COMMAND ----------

# preprocessing de data train producto 3
X_preprocessed_3 = preprocessing_3.transform(X_train_3)

# COMMAND ----------

display(X_preprocessed_3.nunique().reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimización hiperparámetros

# COMMAND ----------

# MAGIC %md
# MAGIC ### LightGBM

# COMMAND ----------

objective_function = 'binary'
eval_metric = 'binary_logloss'

def objective(trial):
    # generando particiones de fechas para el backtesting
    start_date = y_train_3.index.min() + pd.DateOffset(months=1)
    end_date = y_train_3.index.max()
    freq = 'monthly'

    backtesting_dates = _generate_backtesting_dates(
        freq=freq, 
        start_date=start_date, 
        end_date=end_date
    )

    # seteando los parametros a optimizar
    params = {
        'objective': objective_function,
        'metric': eval_metric,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 8, 12),
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.01, 0.95),
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42,
    }

    threshold_prob = 0.5
    n=trial.suggest_categorical('n', [None] + list(range(1, 10)))

    # almacenamiento metricas
    f1_scores = []
    precisions = []
    recalls = []
    kss = []

    for date in backtesting_dates:
        # definiendo particiones - backtesting
        next_date = _sum_backtesting_frequency(date=date, frequency=freq)
        string_date = date.strftime("%Y-%m-%d")
        string_next_date = next_date.strftime("%Y-%m-%d")

        X_train_bt = X_preprocessed_3[X_preprocessed_3.index < date]
        X_test_bt = X_preprocessed_3[(X_preprocessed_3.index >= date) \
                                    & (X_preprocessed_3.index < next_date)]

        y_train_3_bt = y_train_3[y_train_3.index < date]
        y_test_bt = y_train_3[(y_train_3.index >= date) & (y_train_3.index < next_date)]

        if (X_train_bt.index.max() > date) or (y_train_3_bt.index.max() > date):
            msg = (
                "Fuga de información al método fit "
                f"Max date on X_train is {X_train_bt.index.max()} "
                f"Max date on y_train_3 {y_train_3_bt.index.max()} "
                f"than backtesting date: {date}"
            )
            raise ValueError(msg)

        # seteado y entrenando el modelo
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train_bt, y_train_3_bt.to_numpy().ravel(),
            eval_set=[(X_test_bt, y_test_bt)],
            sample_weight=get_sample_weight(y_train_3_bt, n=n) if n is not None else None
        )

        # evaluando el modelo entrenado
        y_score = None

        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test_bt)
            y_probs = y_score[:, 1]
            y_pred = np.where(y_probs > threshold_prob, 1, 0)

        else:
            y_pred = model.predict(X_test_bt)

        # metrica f1
        f1 = f1_score(y_test_bt, y_pred)

        # metrica precision
        if len(np.unique(y_pred)) < 2 or len(np.unique(y_test_bt)) < 2:
            precision = 0  # evita el error en precision_score()
        else:
            precision = precision_score(y_test_bt, y_pred)

        # metrica recall
        recall = recall_score(y_test_bt, y_pred)

        # metrica ks
        ks = _compute_ks_statistic(y_true=y_test_bt['target_3'], y_score=y_probs)

        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        kss.append(ks)


    mean_f1 = np.mean(f1_scores)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    ks = np.mean(kss)

    print(
        f">>> F1 mean: {mean_f1:.5f} | Precision mean: {mean_precision:.5f} | Recall mean: {mean_recall:.5f} | KS mean: {ks:.5f}\n"
        f"F1 - scores: {f1_scores}\n"
        f"Precision - scores: {precisions}\n"
        f"Recall - scores: {recalls}\n"
        f"KS - scores: {kss}"
    )

    return mean_f1

# COMMAND ----------

# optimization
name_model = 'lightgbm'
study = optuna.create_study(
    study_name=f'class_product_3_{name_model}_optimization',
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5, interval_steps=1),
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(objective, n_trials=15)

best_params = study.best_params

# COMMAND ----------

print(
  f'Mejores hiperparámetros usando LightGBM: {best_params}.'
)

# Mejores hiperparámetros usando LightGBM: {'learning_rate': 0.0017898177733695411, 'num_leaves': 120, 'max_depth': 11, 'n_estimators': 2550, 'subsample': 0.6906299486280936, 'colsample_bytree': 0.8084770736580205, 'reg_lambda': 0.15836265442745406, 'reg_alpha': 0.5020816011346719, 'min_gain_to_split': 0.9407596893629844, 'n': 2}.


# COMMAND ----------

# MAGIC %md
# MAGIC ### RandomForest

# COMMAND ----------

def objective(trial):
    # particiones de fechas para el backtesting
    start_date = y_train_3.index.min() + pd.DateOffset(months=1)
    end_date = y_train_3.index.max()
    freq = 'monthly'

    backtesting_dates = _generate_backtesting_dates(
        freq=freq, 
        start_date=start_date, 
        end_date=end_date
    )

    # parámetros a optimizar para Random Forest
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 100),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'bootstrap': True,
        'n_jobs': -1,
        'random_state': 42,
    }

    threshold_prob = 0.5
    n=trial.suggest_categorical('n', [None] + list(range(1, 10)))

    f1_scores = []
    precisions = []
    recalls = []
    kss = []

    for date in backtesting_dates:
        next_date = _sum_backtesting_frequency(date=date, frequency=freq)

        X_train_bt = X_preprocessed_3[X_preprocessed_3.index < date]
        X_test_bt = X_preprocessed_3[(X_preprocessed_3.index >= date) & (X_preprocessed_3.index < next_date)]

        y_train_3_bt = y_train_3[y_train_3.index < date]
        y_test_bt = y_train_3[(y_train_3.index >= date) & (y_train_3.index < next_date)]

        # entrenando modelo Random Forest
        model = RandomForestClassifier(**params)
        model.fit(
            X_train_bt, y_train_3_bt.to_numpy().ravel(), 
            sample_weight=get_sample_weight(y_train_3_bt, n=n) if n is not None else None
        )

        # obtener probabilidades y ajustar umbral
        y_probs = model.predict_proba(X_test_bt)[:, 1]
        y_pred = (y_probs > threshold_prob).astype(int)

        f1 = f1_score(y_test_bt, y_pred)

        if len(np.unique(y_pred)) < 2 or len(np.unique(y_test_bt)) < 2:
            precision = 0
        else:
            precision = precision_score(y_test_bt, y_pred)

        recall = recall_score(y_test_bt, y_pred)

        ks = _compute_ks_statistic(y_true=y_test_bt['target_3'], y_score=y_probs)


        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        kss.append(ks)

    mean_f1 = np.mean(f1_scores)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_ks = np.mean(kss)

    print(
        f">>> F1 mean: {mean_f1:.5f} | Precision mean: {mean_precision:.5f} | Recall mean: {mean_recall:.5f} | KS mean: {mean_ks:.5f}\n"
        f"F1 - scores: {f1_scores}\n"
        f"Precision - scores: {precisions}\n"
        f"Recall - scores: {recalls}\n"
        f"KS: {kss}"
    )

    return mean_f1

# COMMAND ----------

# optimization
name_model = 'random_forest'
study = optuna.create_study(
    study_name=f'class_product_3_{name_model}_optimization',
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5, interval_steps=1),
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(objective, n_trials=15)

best_params = study.best_params

# COMMAND ----------

print(
  f'Mejores hiperparámetros usando RandomForest: {best_params}.'
)



# COMMAND ----------

# MAGIC %md
# MAGIC ### Regresión Logística

# COMMAND ----------

def objective(trial):
    # particiones de fechas para el backtesting
    start_date = y_train_3.index.min() + pd.DateOffset(months=1)
    end_date = y_train_3.index.max()
    freq = 'monthly'

    backtesting_dates = _generate_backtesting_dates(
        freq=freq, 
        start_date=start_date, 
        end_date=end_date
    )

    solver_penalty = trial.suggest_categorical('solver_penalty', [('saga', 'l1'), ('lbfgs', 'l2'), ('saga', 'l2'), ('liblinear', 'l1'), ('liblinear', 'l2')])
    solver, penalty = solver_penalty

    # parametros a optimizar en la regresion logistica
    params = {
        'C': trial.suggest_float('C', 0.001, 0.1),  # regularizacion
        'penalty': penalty,
        'solver': solver,
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'max_iter': trial.suggest_int('max_iter', 2450, 3500),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'random_state': 42,
        'n_jobs': -1,
    }

    threshold_prob = 0.5 # trial.suggest_float('threshold_prob', 0.35, 0.65)
    n=trial.suggest_categorical('n', [None] + list(range(1, 10)))

    f1_scores = []
    precisions = []
    recalls = []
    kss = []


    for date in backtesting_dates:
        # definiendo particiones para el backtesting
        next_date = _sum_backtesting_frequency(date=date, frequency=freq)

        X_train_bt = X_preprocessed_3[X_preprocessed_3.index < date]
        X_test_bt = X_preprocessed_3[(X_preprocessed_3.index >= date) & (X_preprocessed_3.index < next_date)]

        y_train_3_bt = y_train_3[y_train_3.index < date]
        y_test_bt = y_train_3[(y_train_3.index >= date) & (y_train_3.index < next_date)]

        # entrenando modelo
        model = LogisticRegression(**params)
        model.fit(
            X_train_bt, y_train_3_bt.to_numpy().ravel(),
            sample_weight=get_sample_weight(y_train_3_bt, n=n) if n is not None else None
        )

        # obtener probabilidades y ajustar umbral
        y_probs = model.predict_proba(X_test_bt)[:, 1]
        y_pred = (y_probs > threshold_prob).astype(int)
        
        # metrica f1-score
        f1 = f1_score(y_test_bt, y_pred)
        # metrica precision
        if len(np.unique(y_pred)) < 2 or len(np.unique(y_test_bt)) < 2:
            precision = 0  # evita el error en precision_score()
        else:
            precision = precision_score(y_test_bt, y_pred)
        # metrica recall
        recall = recall_score(y_test_bt, y_pred)

        ks = _compute_ks_statistic(y_true=y_test_bt['target_3'], y_score=y_probs)


        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        kss.append(ks)

    mean_f1 = np.mean(f1_scores)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)

    mean_ks = np.mean(kss)

    print(
        f">>> F1 mean: {mean_f1:.5f} | Precision mean: {mean_precision:.5f} | Recall mean: {mean_recall:.5f}\n"
        f"F1 - scores: {f1_scores}\n"
        f"Precision - scores: {precisions}\n"
        f"Recall - scores: {recalls}\n"
        f"KS: {kss}"
    )

    return mean_f1

# COMMAND ----------

# optimization
name_model = 'reg_log'
study = optuna.create_study(
    study_name=f'class_product_3_{name_model}_optimization',
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5, interval_steps=1),
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(objective, n_trials=15)

best_params = study.best_params

# COMMAND ----------

print(
  f'Mejores hiperparámetros usando Regresión Logística: {best_params}.'
)



# COMMAND ----------

# MAGIC %md
# MAGIC ## Selección de modelos

# COMMAND ----------

# modelos con los mejores hiperparámetros obtenidos
model_lgbm_3 = lgb.LGBMClassifier(**{
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.0017898177733695411, 
    'num_leaves': 120, 
    'max_depth': 11, 
    'n_estimators': 2550, 
    'subsample': 0.6906299486280936, 
    'colsample_bytree': 0.8084770736580205, 
    'reg_lambda': 0.15836265442745406, 
    'reg_alpha': 0.5020816011346719, 
    'min_gain_to_split': 0.9407596893629844,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
})

model_rf_3 = RandomForestClassifier(**{
    'n_estimators': 1970, 
    'max_depth': 7, 
    'min_samples_split': 59, 
    'min_samples_leaf': 41, 
    'max_features': 'log2', 
    'criterion': 'gini', 
    'bootstrap': True,
    'n_jobs': -1,
    'random_state': 42,
})

model_reglog_3 = LogisticRegression(**{
    'solver': 'saga',
    'penalty': 'l1',
    'C': 0.003780521093327556, 
    'class_weight': 'balanced', 
    'max_iter': 2839, 
    'fit_intercept': False, 
    'random_state': 42,
    'n_jobs': -1,
})


# COMMAND ----------

start_date = y_train_3.index.min() + pd.DateOffset(months=1)
end_date = y_train_3.index.max()

backtesting_dates = _generate_backtesting_dates(
    freq='monthly', 
    start_date=start_date, 
    end_date=end_date
)

i=1
threshold_prob = 0.5

metrics_fold = {
    'model': [],
    'pr_auc': [],
    'roc_auc': [],
    'brier_score': [],
} 
for date in backtesting_dates:
        before_date = date - pd.DateOffset(months=1)
        next_date = _sum_backtesting_frequency(date=date, frequency='monthly')

        X_train_kf = X_preprocessed_3[(X_preprocessed_3.index >= before_date) & (X_preprocessed_3.index < date)]
        X_test_kf = X_preprocessed_3[(X_preprocessed_3.index >= date) & (X_preprocessed_3.index < next_date)]

        y_train_kf = y_train_3[(y_train_3.index >= before_date) & (y_train_3.index < date)]
        y_test_kf = y_train_3[(y_train_3.index >= date) & (y_train_3.index < next_date)]

        print(
            f'Fechas para el fold {i}: {y_train_kf.index.min()} - {y_test_kf.index.max()}\n'
            f'Training set del fold: {y_train_kf.index.min()} - {y_train_kf.index.max()}\n'
            f'Testing set del fold: {y_test_kf.index.min()} - {y_test_kf.index.max()}\n'
        )

        # entrenamiento de los modelos
        model_lgbm_3.fit(
            X_train_kf, y_train_kf.to_numpy().ravel(),
            eval_set=[(X_test_kf, y_test_kf)],
            sample_weight=get_sample_weight(y_train_kf, n=2),
            callbacks=[lgb.early_stopping(stopping_rounds=50)],
        )

        model_rf_3.fit(
            X_train_kf, y_train_kf.to_numpy().ravel(),
            sample_weight=get_sample_weight(y_train_kf, n=2),
        )

        model_reglog_3.fit(
            X_train_kf, y_train_kf.to_numpy().ravel(),
            sample_weight=get_sample_weight(y_train_kf, n=8),
        )

        # testeando los modelos
        # realizando predicciones
        y_score_lgbm = model_lgbm_3.predict_proba(X_test_kf)[:, 1]
        y_pred_lgbm = (y_score_lgbm > threshold_prob).astype(int)

        y_score_rf = model_rf_3.predict_proba(X_test_kf)[:, 1]
        y_pred_rf = (y_score_rf > threshold_prob).astype(int)

        y_score_reglog = model_reglog_3.predict_proba(X_test_kf)[:, 1]
        y_pred_reglog = (y_score_reglog > threshold_prob).astype(int)

        # calculando metricas
        roc_auc_lgbm = roc_auc_score(y_test_kf, y_score_lgbm)
        brier_score_lgbm = brier_score_loss(y_test_kf, y_score_lgbm)
        precision, recall, _ = precision_recall_curve(y_test_kf, y_score_lgbm)
        pr_auc_lgbm = auc(recall, precision)     

        roc_auc_rf = roc_auc_score(y_test_kf, y_score_rf)
        brier_score_rf = brier_score_loss(y_test_kf, y_score_rf)
        precision, recall, _ = precision_recall_curve(y_test_kf, y_score_rf)
        pr_auc_rf = auc(recall, precision)

        roc_auc_reglog = roc_auc_score(y_test_kf, y_score_reglog)
        brier_score_reglog = brier_score_loss(y_test_kf, y_score_reglog)
        precision, recall, _ = precision_recall_curve(y_test_kf, y_score_reglog)
        pr_auc_reglog = auc(recall, precision)

        metrics_fold['model']+=['lightgbm', 'random_forest', 'reg_log']
        metrics_fold['pr_auc']+=[pr_auc_lgbm, pr_auc_rf, pr_auc_reglog]
        metrics_fold['roc_auc']+=[roc_auc_lgbm, roc_auc_rf, roc_auc_reglog]
        metrics_fold['brier_score']+=[brier_score_lgbm, brier_score_rf, brier_score_reglog]

        i+=1

# COMMAND ----------

# creando df con las metricas obtenidas de c/modelo
eval_models_3 = pd.DataFrame(metrics_fold)

# describiendo las metricas
display(eval_models_3.groupby('model').describe().T.reset_index())

# COMMAND ----------

palette = {
    'lightgbm': '#FB4F4F',      
    'random_forest': '#57F384', 
    'reg_log': '#16DBEA'       
}

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
sns.set_style('darkgrid')

sns.violinplot(data=eval_models_3, x='model', y='pr_auc', ax=ax[0], palette=palette)
ax[0].set_xlabel('modelos')
ax[0].set_ylabel('pr-auc')
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

sns.violinplot(data=eval_models_3, x='model', y='roc_auc', ax=ax[1], palette=palette)
ax[1].set_xlabel('modelos')
ax[1].set_ylabel('roc-auc')
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

sns.violinplot(data=eval_models_3, x='model', y='brier_score', ax=ax[2], palette=palette)
ax[2].set_xlabel('modelos')
ax[2].set_ylabel('brier-score')
ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.suptitle('Métricas de evaluación de los modelos')

plt.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC En el producto 3, aunque Random Forest obtiene un PR-AUC y un ROC-AUC levemente superiores a los de LightGBM, la diferencia es mínima y se compensa con la mejor calibración y menor Brier Score que muestra LightGBM, además de una estabilidad ligeramente mayor entre los folds temporales. Considerando que en un contexto bancario la consistencia y la calidad de las probabilidades son esenciales para planificar campañas y tomar decisiones confiables, el modelo seleccionado es LightGBM, ya que ofrece un equilibrio más sólido entre discriminación, estabilidad y calibración.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training modelo final

# COMMAND ----------

params_model_3 = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.0017898177733695411, 
    'num_leaves': 120, 
    'max_depth': 11, 
    'n_estimators': 2550, 
    'subsample': 0.6906299486280936, 
    'colsample_bytree': 0.8084770736580205, 
    'reg_lambda': 0.15836265442745406, 
    'reg_alpha': 0.5020816011346719, 
    'min_gain_to_split': 0.9407596893629844,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
}

fit_params_3 = {
    'sample_weight': get_sample_weight(y_train_3, n=2),
   # 'callbacks': [lgb.early_stopping(stopping_rounds=50)]
}

# COMMAND ----------

model_product_3 = Pipeline(steps=[
    ('FeatureSelector', ColumnsSelector(columns=features_product_3)),
    ('OutliersHandler', OutliersHandler(
                            numeric_cols=numerical_cols,
                            exclude_cols=['trans_type_currency_cnt_m'])),
    ('CategoricalNullImputer', CategoricalNullImputer(
                                    categorical_cols=categorical_cols,
                                    exclude_cols={
                                        'pur_any_purchase': 0,
                                        'trx_growth_streak_3m': 0,
                                    })),
    ('NumericalNullImputer', NumericalNullImputer(
                                     numerical_cols=numerical_cols,
                                     tipo='aleatorio',
                                     exclude_cols={
                                         'trans_type_currency_cnt_m': 1,
                                     })),
    ('NumericalTransformer', NumericalLogTransformer(
                                     numerical_cols=numerical_cols,
                                     threshold_coef=1)),
    ('Scaler', ContinuousFeatureScaler(scaler='standard',
                            numerical_cols=numerical_cols)),
    ('ClassifierProduct3', ClassifierByProduct(
                                        model_cls=lgb.LGBMClassifier,
                                        model_params=params_model_3,
                                        fit_params=fit_params_3,
                                        threshold=0.43434343434343434343,
                                    ))
])

model_product_3.fit(X_train_3, y_train_3)

# COMMAND ----------

# guardando modelo en MLFLow
with mlflow.start_run(run_name='model_product_3'):
    mlflow.sklearn.log_model(
        model_product_3,
        'model_product_3',
        registered_model_name='model_product_3',
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # Product 4

# COMMAND ----------

# seteando como indice mon
train_4['mon'] = pd.to_datetime(train_4['mon'])
train_4 = train_4.set_index('mon')

test_4['mon'] = pd.to_datetime(test_4['mon'])
test_4 = test_4.set_index('mon')

# COMMAND ----------

# dividiedo en X-y train and test
target = 'target_4'

X_train_4, y_train_4 = train_4.copy().drop(columns=['client_id', 'target_4']), train_4[[target]]
X_test_4, y_test_4 = test_4.copy().drop(columns=['client_id', 'target_4']), test_4[[target]]

# COMMAND ----------

fig, ax = plt.subplots(1,2,figsize=(9, 3.5))

sns.barplot(y_train_4.value_counts(normalize=True).reset_index(), y=0, x='target_4', color='#00968B', ax=ax[0], width=0.6)
sns.barplot(y_test_4.value_counts(normalize=True).reset_index(), y=0, x='target_4', color='#FED700', ax=ax[1], width=0.6)

ax[0].set_title('desbalance en y_train', color='#00968B')
ax[1].set_title('desbalance en y_test', color='orange')

ax[0].set_xlabel('target: target_4')
ax[1].set_xlabel('target: target_4')

for p in ax[0].patches:
    height = p.get_height()
    ax[0].annotate(
        f'{height:.2%}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', xytext=(0, 5), textcoords='offset points'
    )

for p in ax[1].patches:
    height = p.get_height()
    ax[1].annotate(
        f'{height:.2%}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', xytext=(0, 5), textcoords='offset points'
    )

plt.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Features categóricas & numéricas

# COMMAND ----------

# obtencion de features numericas y categoricas
numerical_cols, categorical_cols = _identify_type_columns_pandas(
    df=X_train_4,
    params={
        'id_columns': ['mon', 'client_id'],
        'target': ['target_1', 'target_3', 'target_4'],
        'features_exclude': {
            'numerical': ['trans_dst_type_mode_m', 'trans_event_type_mode_m'],
            'categorical': ['dialog_avg_per_day_m', 'pur_product_3_rsum_m'],
        },
    }
)

features_product_4 = numerical_cols + categorical_cols

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline preprocesamiento

# COMMAND ----------

categorical_cols

# COMMAND ----------

preprocessing_4 = Pipeline(steps=[
    ('FeatureSelector', ColumnsSelector(columns=features_product_4)),
    ('OutliersHandler', OutliersHandler(
                            numeric_cols=numerical_cols,
                            exclude_cols=['trans_type_currency_cnt_m'])),
    ('CategoricalNullImputer', CategoricalNullImputer(
                                    categorical_cols=categorical_cols,
                                    exclude_cols={
                                        'trx_growth_streak_3m': 0,
                                    })),
    ('NumericalNullImputer', NumericalNullImputer(
                                     numerical_cols=numerical_cols,
                                     tipo='aleatorio',
                                     exclude_cols={
                                         'trans_type_currency_cnt_m': 1,
                                     })),
    ('WoEEncoder', WoEEncoder(
                            woe_sim_threshold=0.1,
                            min_bin_pct=0.1,
                            categorical_cols=categorical_cols,
                            exclude_cols=['trx_growth_streak_3m']        
    )),
    ('NumericalTransformer', NumericalLogTransformer(
                                     numerical_cols=numerical_cols,
                                     threshold_coef=1)),
    ('Scaler', ContinuousFeatureScaler(scaler='standard',
                            numerical_cols=numerical_cols))
])

preprocessing_4.fit(X_train_4, y_train_4)

# COMMAND ----------

# preprocessing de data train producto 4
X_preprocessed_4 = preprocessing_4.transform(X_train_4)

# COMMAND ----------

display(X_preprocessed_4.nunique().reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimización hiperparámetros

# COMMAND ----------

# MAGIC %md
# MAGIC ### LightGBM

# COMMAND ----------

objective_function = 'binary'
eval_metric = 'binary_logloss'

def objective(trial):
    # generando particiones de fechas para el backtesting
    start_date = y_train_4.index.min() + pd.DateOffset(months=1)
    end_date = y_train_4.index.max()
    freq = 'monthly'

    backtesting_dates = _generate_backtesting_dates(
        freq=freq, 
        start_date=start_date, 
        end_date=end_date
    )

    # seteando los parametros a optimizar
    params = {
        'objective': objective_function,
        'metric': eval_metric,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 8, 12),
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.01, 0.95),
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42,
    }

    threshold_prob = 0.5
    n=trial.suggest_categorical('n', [None] + list(range(1, 10)))

    # almacenamiento metricas
    f1_scores = []
    precisions = []
    recalls = []
    kss = []

    for date in backtesting_dates:
        # definiendo particiones - backtesting
        next_date = _sum_backtesting_frequency(date=date, frequency=freq)
        string_date = date.strftime("%Y-%m-%d")
        string_next_date = next_date.strftime("%Y-%m-%d")

        X_train_bt = X_preprocessed_4[X_preprocessed_4.index < date]
        X_test_bt = X_preprocessed_4[(X_preprocessed_4.index >= date) \
                                    & (X_preprocessed_4.index < next_date)]

        y_train_4_bt = y_train_4[y_train_4.index < date]
        y_test_bt = y_train_4[(y_train_4.index >= date) & (y_train_4.index < next_date)]

        if (X_train_bt.index.max() > date) or (y_train_4_bt.index.max() > date):
            msg = (
                "Fuga de información al método fit "
                f"Max date on X_train is {X_train_bt.index.max()} "
                f"Max date on y_train_4 {y_train_4_bt.index.max()} "
                f"than backtesting date: {date}"
            )
            raise ValueError(msg)

        # seteado y entrenando el modelo
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train_bt, y_train_4_bt.to_numpy().ravel(),
            eval_set=[(X_test_bt, y_test_bt)],
            sample_weight=get_sample_weight(y_train_4_bt, n=n) if n is not None else None
        )

        # evaluando el modelo entrenado
        y_score = None

        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test_bt)
            y_probs = y_score[:, 1]
            y_pred = np.where(y_probs > threshold_prob, 1, 0)

        else:
            y_pred = model.predict(X_test_bt)

        # metrica f1
        f1 = f1_score(y_test_bt, y_pred)

        # metrica precision
        if len(np.unique(y_pred)) < 2 or len(np.unique(y_test_bt)) < 2:
            precision = 0  # evita el error en precision_score()
        else:
            precision = precision_score(y_test_bt, y_pred)

        # metrica recall
        recall = recall_score(y_test_bt, y_pred)

        # metrica ks
        ks = _compute_ks_statistic(y_true=y_test_bt['target_4'], y_score=y_probs)

        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        kss.append(ks)


    mean_f1 = np.mean(f1_scores)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    ks = np.mean(kss)

    print(
        f">>> F1 mean: {mean_f1:.5f} | Precision mean: {mean_precision:.5f} | Recall mean: {mean_recall:.5f} | KS mean: {ks:.5f}\n"
        f"F1 - scores: {f1_scores}\n"
        f"Precision - scores: {precisions}\n"
        f"Recall - scores: {recalls}\n"
        f"KS - scores: {kss}"
    )

    return mean_f1

# COMMAND ----------

# optimization
name_model = 'lightgbm'
study = optuna.create_study(
    study_name=f'class_product_4_{name_model}_optimization',
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5, interval_steps=1),
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(objective, n_trials=15)

best_params = study.best_params

# COMMAND ----------

print(
  f'Mejores hiperparámetros usando LightGBM: {best_params}.'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### RandomForest

# COMMAND ----------

def objective(trial):
    # particiones de fechas para el backtesting
    start_date = y_train_4.index.min() + pd.DateOffset(months=1)
    end_date = y_train_4.index.max()
    freq = 'monthly'

    backtesting_dates = _generate_backtesting_dates(
        freq=freq, 
        start_date=start_date, 
        end_date=end_date
    )

    # parámetros a optimizar para Random Forest
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 100),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'bootstrap': True,
        'n_jobs': -1,
        'random_state': 42,
    }

    threshold_prob = 0.5
    n=trial.suggest_categorical('n', [None] + list(range(1, 10)))

    f1_scores = []
    precisions = []
    recalls = []
    kss = []

    for date in backtesting_dates:
        next_date = _sum_backtesting_frequency(date=date, frequency=freq)

        X_train_bt = X_preprocessed_4[X_preprocessed_4.index < date]
        X_test_bt = X_preprocessed_4[(X_preprocessed_4.index >= date) & (X_preprocessed_4.index < next_date)]

        y_train_4_bt = y_train_4[y_train_4.index < date]
        y_test_bt = y_train_4[(y_train_4.index >= date) & (y_train_4.index < next_date)]

        # entrenando modelo Random Forest
        model = RandomForestClassifier(**params)
        model.fit(
            X_train_bt, y_train_4_bt.to_numpy().ravel(), 
            sample_weight=get_sample_weight(y_train_4_bt, n=n) if n is not None else None
        )

        # obtener probabilidades y ajustar umbral
        y_probs = model.predict_proba(X_test_bt)[:, 1]
        y_pred = (y_probs > threshold_prob).astype(int)

        f1 = f1_score(y_test_bt, y_pred)

        if len(np.unique(y_pred)) < 2 or len(np.unique(y_test_bt)) < 2:
            precision = 0
        else:
            precision = precision_score(y_test_bt, y_pred)

        recall = recall_score(y_test_bt, y_pred)

        ks = _compute_ks_statistic(y_true=y_test_bt['target_4'], y_score=y_probs)


        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        kss.append(ks)

    mean_f1 = np.mean(f1_scores)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_ks = np.mean(kss)

    print(
        f">>> F1 mean: {mean_f1:.5f} | Precision mean: {mean_precision:.5f} | Recall mean: {mean_recall:.5f} | KS mean: {mean_ks:.5f}\n"
        f"F1 - scores: {f1_scores}\n"
        f"Precision - scores: {precisions}\n"
        f"Recall - scores: {recalls}\n"
        f"KS: {kss}"
    )

    return mean_f1

# COMMAND ----------

# optimization
name_model = 'random_forest'
study = optuna.create_study(
    study_name=f'class_product_4_{name_model}_optimization',
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5, interval_steps=1),
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(objective, n_trials=15)

best_params = study.best_params

# COMMAND ----------

print(
  f'Mejores hiperparámetros usando RandomForest: {best_params}.'
)



# COMMAND ----------

# MAGIC %md
# MAGIC ### Regresión Logística

# COMMAND ----------

def objective(trial):
    # particiones de fechas para el backtesting
    start_date = y_train_4.index.min() + pd.DateOffset(months=1)
    end_date = y_train_4.index.max()
    freq = 'monthly'

    backtesting_dates = _generate_backtesting_dates(
        freq=freq, 
        start_date=start_date, 
        end_date=end_date
    )

    solver_penalty = trial.suggest_categorical('solver_penalty', [('saga', 'l1'), ('lbfgs', 'l2'), ('saga', 'l2'), ('liblinear', 'l1'), ('liblinear', 'l2')])
    solver, penalty = solver_penalty

    # parametros a optimizar en la regresion logistica
    params = {
        'C': trial.suggest_float('C', 0.001, 0.1),  # regularizacion
        'penalty': penalty,
        'solver': solver,
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'max_iter': trial.suggest_int('max_iter', 2450, 3500),
        'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
        'random_state': 42,
        'n_jobs': -1,
    }

    threshold_prob = 0.5 # trial.suggest_float('threshold_prob', 0.35, 0.65)
    n=trial.suggest_categorical('n', [None] + list(range(1, 10)))

    f1_scores = []
    precisions = []
    recalls = []
    kss = []


    for date in backtesting_dates:
        # definiendo particiones para el backtesting
        next_date = _sum_backtesting_frequency(date=date, frequency=freq)

        X_train_bt = X_preprocessed_4[X_preprocessed_4.index < date]
        X_test_bt = X_preprocessed_4[(X_preprocessed_4.index >= date) & (X_preprocessed_4.index < next_date)]

        y_train_4_bt = y_train_4[y_train_4.index < date]
        y_test_bt = y_train_4[(y_train_4.index >= date) & (y_train_4.index < next_date)]

        # entrenando modelo
        model = LogisticRegression(**params)
        model.fit(
            X_train_bt, y_train_4_bt.to_numpy().ravel(),
            sample_weight=get_sample_weight(y_train_4_bt, n=n) if n is not None else None
        )

        # obtener probabilidades y ajustar umbral
        y_probs = model.predict_proba(X_test_bt)[:, 1]
        y_pred = (y_probs > threshold_prob).astype(int)
        
        # metrica f1-score
        f1 = f1_score(y_test_bt, y_pred)
        # metrica precision
        if len(np.unique(y_pred)) < 2 or len(np.unique(y_test_bt)) < 2:
            precision = 0  # evita el error en precision_score()
        else:
            precision = precision_score(y_test_bt, y_pred)
        # metrica recall
        recall = recall_score(y_test_bt, y_pred)

        ks = _compute_ks_statistic(y_true=y_test_bt['target_4'], y_score=y_probs)


        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        kss.append(ks)

    mean_f1 = np.mean(f1_scores)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)

    mean_ks = np.mean(kss)

    print(
        f">>> F1 mean: {mean_f1:.5f} | Precision mean: {mean_precision:.5f} | Recall mean: {mean_recall:.5f}\n"
        f"F1 - scores: {f1_scores}\n"
        f"Precision - scores: {precisions}\n"
        f"Recall - scores: {recalls}\n"
        f"KS: {kss}"
    )

    return mean_f1

# COMMAND ----------

# optimization
name_model = 'reg_log'
study = optuna.create_study(
    study_name=f'class_product_4_{name_model}_optimization',
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5, interval_steps=1),
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(objective, n_trials=15)

best_params = study.best_params

# COMMAND ----------

print(
  f'Mejores hiperparámetros usando Regresión Logística: {best_params}.'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Selección de modelos

# COMMAND ----------

# modelos con los mejores hiperparámetros obtenidos
model_lgbm_4 = lgb.LGBMClassifier(**{
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.0026421950638577593, 
    'num_leaves': 178, 
    'max_depth': 9, 
    'n_estimators': 2290, 
    'subsample': 0.5784648930522461, 
    'colsample_bytree': 0.8109219821461097, 
    'reg_lambda': 0.3873486109542369, 
    'reg_alpha': 0.9367932587479978, 
    'min_gain_to_split': 0.13926968749723365,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
})

model_rf_4 = RandomForestClassifier(**{
    'n_estimators': 1534, 
    'max_depth': 11, 
    'min_samples_split': 67, 
    'min_samples_leaf': 54, 
    'max_features': 'sqrt', 
    'criterion': 'gini', 
    'bootstrap': True,
    'n_jobs': -1,
    'random_state': 42,
})


model_reglog_4 = LogisticRegression(**{
    'solver': 'saga',
    'penalty': 'l1',
    'C': 0.06493978762962088, 
    'class_weight': 'balanced', 
    'max_iter': 2813, 
    'fit_intercept': True, 
    'random_state': 42,
    'n_jobs': -1,
})

# COMMAND ----------

start_date = y_train_4.index.min() + pd.DateOffset(months=1)
end_date = y_train_4.index.max()

backtesting_dates = _generate_backtesting_dates(
    freq='monthly', 
    start_date=start_date, 
    end_date=end_date
)

i=1
threshold_prob = 0.5

metrics_fold = {
    'model': [],
    'pr_auc': [],
    'roc_auc': [],
    'brier_score': [],
} 
for date in backtesting_dates:
        before_date = date - pd.DateOffset(months=1)
        next_date = _sum_backtesting_frequency(date=date, frequency='monthly')

        X_train_kf = X_preprocessed_4[(X_preprocessed_4.index >= before_date) & (X_preprocessed_4.index < date)]
        X_test_kf = X_preprocessed_4[(X_preprocessed_4.index >= date) & (X_preprocessed_4.index < next_date)]

        y_train_kf = y_train_4[(y_train_4.index >= before_date) & (y_train_4.index < date)]
        y_test_kf = y_train_4[(y_train_4.index >= date) & (y_train_4.index < next_date)]

        print(
            f'Fechas para el fold {i}: {y_train_kf.index.min()} - {y_test_kf.index.max()}\n'
            f'Training set del fold: {y_train_kf.index.min()} - {y_train_kf.index.max()}\n'
            f'Testing set del fold: {y_test_kf.index.min()} - {y_test_kf.index.max()}\n'
        )

        # entrenamiento de los modelos
        model_lgbm_4.fit(
            X_train_kf, y_train_kf.to_numpy().ravel(),
            eval_set=[(X_test_kf, y_test_kf)],
            sample_weight=get_sample_weight(y_train_kf, n=2),
            callbacks=[lgb.early_stopping(stopping_rounds=50)],
        )

        model_rf_4.fit(
            X_train_kf, y_train_kf.to_numpy().ravel(),
            sample_weight=get_sample_weight(y_train_kf, n=2),
        )

        model_reglog_4.fit(
            X_train_kf, y_train_kf.to_numpy().ravel(),
            sample_weight=get_sample_weight(y_train_kf, n=3),
        )

        # testeando los modelos
        # realizando predicciones
        y_score_lgbm = model_lgbm_4.predict_proba(X_test_kf)[:, 1]
        y_pred_lgbm = (y_score_lgbm > threshold_prob).astype(int)

        y_score_rf = model_rf_4.predict_proba(X_test_kf)[:, 1]
        y_pred_rf = (y_score_rf > threshold_prob).astype(int)

        y_score_reglog = model_reglog_4.predict_proba(X_test_kf)[:, 1]
        y_pred_reglog = (y_score_reglog > threshold_prob).astype(int)

        # calculando metricas
        roc_auc_lgbm = roc_auc_score(y_test_kf, y_score_lgbm)
        brier_score_lgbm = brier_score_loss(y_test_kf, y_score_lgbm)
        precision, recall, _ = precision_recall_curve(y_test_kf, y_score_lgbm)
        pr_auc_lgbm = auc(recall, precision)     

        roc_auc_rf = roc_auc_score(y_test_kf, y_score_rf)
        brier_score_rf = brier_score_loss(y_test_kf, y_score_rf)
        precision, recall, _ = precision_recall_curve(y_test_kf, y_score_rf)
        pr_auc_rf = auc(recall, precision)

        roc_auc_reglog = roc_auc_score(y_test_kf, y_score_reglog)
        brier_score_reglog = brier_score_loss(y_test_kf, y_score_reglog)
        precision, recall, _ = precision_recall_curve(y_test_kf, y_score_reglog)
        pr_auc_reglog = auc(recall, precision)

        metrics_fold['model']+=['lightgbm', 'random_forest', 'reg_log']
        metrics_fold['pr_auc']+=[pr_auc_lgbm, pr_auc_rf, pr_auc_reglog]
        metrics_fold['roc_auc']+=[roc_auc_lgbm, roc_auc_rf, roc_auc_reglog]
        metrics_fold['brier_score']+=[brier_score_lgbm, brier_score_rf, brier_score_reglog]

        i+=1

# COMMAND ----------

# creando df con las metricas obtenidas de c/modelo
eval_models_4 = pd.DataFrame(metrics_fold)

# describiendo las metricas
display(eval_models_4.groupby('model').describe().T.reset_index())

# COMMAND ----------

palette = {
    'lightgbm': '#FB4F4F',      
    'random_forest': '#57F384', 
    'reg_log': '#16DBEA'       
}

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
sns.set_style('darkgrid')

sns.violinplot(data=eval_models_4, x='model', y='pr_auc', ax=ax[0], palette=palette)
ax[0].set_xlabel('modelos')
ax[0].set_ylabel('pr-auc')
ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

sns.violinplot(data=eval_models_4, x='model', y='roc_auc', ax=ax[1], palette=palette)
ax[1].set_xlabel('modelos')
ax[1].set_ylabel('roc-auc')
ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

sns.violinplot(data=eval_models_4, x='model', y='brier_score', ax=ax[2], palette=palette)
ax[2].set_xlabel('modelos')
ax[2].set_ylabel('brier-score')
ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.suptitle('Métricas de evaluación de los modelos')

plt.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC En el producto 4, los resultados muestran diferencias claras: LightGBM obtiene el mejor PR-AUC (0.652 vs 0.644 en RF y 0.631 en RegLog) y mantiene un ROC-AUC prácticamente idéntico al de Random Forest (0.715 vs 0.714), superando ampliamente a la regresión logística. Además, logra el menor Brier Score (0.212), lo que indica mejor calibración de las probabilidades. Aunque la varianza entre folds es parecida en los dos modelos de árboles, LightGBM ofrece un balance superior entre discriminación, calibración y estabilidad. Por ende, para este producto se escoge el modelo LightGBM.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training modelo final

# COMMAND ----------

params_model_4 = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.0026421950638577593, 
    'num_leaves': 178, 
    'max_depth': 9, 
    'n_estimators': 2290, 
    'subsample': 0.5784648930522461, 
    'colsample_bytree': 0.8109219821461097, 
    'reg_lambda': 0.3873486109542369, 
    'reg_alpha': 0.9367932587479978, 
    'min_gain_to_split': 0.13926968749723365,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
}

fit_params_4 = {
    'sample_weight': get_sample_weight(y_train_4, n=2),
   # 'callbacks': [lgb.early_stopping(stopping_rounds=50)]
}

# COMMAND ----------

model_product_4 = Pipeline(steps=[
    ('FeatureSelector', ColumnsSelector(columns=features_product_4)),
    ('OutliersHandler', OutliersHandler(
                            numeric_cols=numerical_cols,
                            exclude_cols=['trans_type_currency_cnt_m'])),
    ('CategoricalNullImputer', CategoricalNullImputer(
                                    categorical_cols=categorical_cols,
                                    exclude_cols={
                                        'trx_growth_streak_3m': 0,
                                    })),
    ('NumericalNullImputer', NumericalNullImputer(
                                     numerical_cols=numerical_cols,
                                     tipo='aleatorio',
                                     exclude_cols={
                                         'trans_type_currency_cnt_m': 1,
                                     })),
    ('NumericalTransformer', NumericalLogTransformer(
                                     numerical_cols=numerical_cols,
                                     threshold_coef=1)),
    ('Scaler', ContinuousFeatureScaler(scaler='standard',
                            numerical_cols=numerical_cols)),
    ('ClassifierProduct4', ClassifierByProduct(
                                        model_cls=lgb.LGBMClassifier,
                                        model_params=params_model_4,
                                        fit_params=fit_params_4,
                                        threshold=0.5,
                                    ))
])

model_product_4.fit(X_train_4, y_train_4)

# COMMAND ----------

# guardando modelo en MLFLow
with mlflow.start_run(run_name='model_product_4'):
    mlflow.sklearn.log_model(
        model_product_4,
        'model_product_4',
        registered_model_name='model_product_4',
    )