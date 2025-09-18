# Databricks notebook source
# MAGIC %md
# MAGIC # TFM: Next Move Banking
# MAGIC **Autor:** Coraima Castillo
# MAGIC
# MAGIC En este notebook se almacenan las funciones auxiliares que facilitan los procesos realizados en los notebooks del proyecto.

# COMMAND ----------

# librerias basicas
import pandas as pd
import numpy as np
import pandas.api.types as ptypes
import matplotlib.pyplot as plt
import seaborn as sns

# librerias para transformadores
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler

# librerias estadisticas
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# librerias para metricas
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score,
    fbeta_score,
    brier_score_loss,
    auc
)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve 
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
import shap

# librerias para docstring
import typing as tp
import warnings

# librerais para logear modelos
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# COMMAND ----------

def get_latest_model_version(
    model_name: str
) -> int:
    """
    Obtiene la versión más reciente registrada en el registro de modelos de MLflow para un modelo dado.

    Parámetros:
        model_name : nombre del modelo tal como está registrado en el Model Registry de MLflow.

    Retorna:
        int: número entero que representa la versión más alta encontrada. 
             Si no existen versiones, se devuelve 1 como valor por defecto.
    """
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

# MAGIC %md
# MAGIC ## Funciones: Identificación numéricas/categóricas

# COMMAND ----------

def _identify_type_columns_pandas(
    df: pd.DataFrame,
    params: dict = dict(),
) -> tp.Tuple[tp.List[str], tp.List[str]]:
    """
    Identifica columnas numéricas y categóricas en un DataFrame de Pandas.

    Parámetros:
        - df: DataFrame de Pandas a analizar.
        - params: Diccionario que contiene parámetros opcionales:
            - id_columns: Lista de keys a excluir del análisis.
            - target: Lista de columnas objetivo a excluir.
            - features_exclude: Lista de columnas adicionales a excluir debido 
              a que son indicadores que corresponden a features categoricas.
            - threshold_values: Cantidad mínima de valores únicos para considerar una columna numérica.
    
    Retorna:
        tuple: Tupla que contiene dos listas:
            - Lista de nombres de columnas numéricas.
            - Lista de nombres de columnas categóricas.
    """
    # extrayendo features a NO considerar entre features numericas
    id_columns = params.get('id_columns', ['pk_customer', 'tpk_release_dt'])
    target = params.get('target', ['default'])
    feats_exclude_num = params.get('features_exclude', {}).get('numerical', [])
    feats_exclude_cat = params.get('features_exclude', {}).get('categorical', [])

    threshold_values = params.get('threshold_values', 3)
    
    # obteniendo features numericas
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    num_cols = [
        col for col in num_cols if ((col not in id_columns + target + feats_exclude_num) \
            and (df[col].nunique() > threshold_values)) or (col in feats_exclude_cat)
    ]
    
    cat_cols = list(
        set(df.columns) - set(num_cols) - set(target) - set(id_columns) - set(feats_exclude_cat)
    )

    return num_cols, cat_cols

# COMMAND ----------

# MAGIC %md
# MAGIC ## Funciones: Feature Selection

# COMMAND ----------

# MAGIC %md
# MAGIC ### Descarte features

# COMMAND ----------

def features_lot_of_zeros(
    df: pd.DataFrame,
    umbral: float = 0.80,
    params: tp.Dict[str, tp.List[str]] = dict(),
) -> tp.List[str]:
    """
    Identifica columnas en un DataFrame de Pandas donde el porcentaje de ceros supera un umbral.

    Parámetros:
        df : pd.DataFrame
            DataFrame de Pandas a analizar.
        umbral : float, default=0.80
            Umbral (entre 0 y 1) para considerar que una columna tiene demasiados ceros.
        params : dict, optional
            Diccionario con listas de columnas que no se deben evaluar.
            - 'id_columns': columnas identificadoras.
            - 'target': columna(s) objetivo.

    Retorna:
        list: Lista de nombres de columnas donde el porcentaje de ceros supera el umbral.
    """
    df = df.fillna(0)

    id_columns = params.get('id_columns', ['pk_customer', 'tpk_release_dt'])
    target = params.get('target', ['default'])

    features_zeros = []
    total = len(df)

    for col in df.columns:
        if col not in id_columns and col not in target:
            zero_percentage = (df[col] == 0).sum() / total
            print(f"Feature: {col}, Porcentaje de ceros: {zero_percentage * 100:.2f}%")

            if zero_percentage > umbral:
                features_zeros.append(col)

    return features_zeros


# COMMAND ----------

def identify_zero_variances(
    df: pd.DataFrame,
    params: tp.Dict[str, tp.List[str]] = dict(),
) -> tp.Tuple[tp.List[str], tp.List[str]]:
    """
    Identifica columnas numéricas con varianza cero en un DataFrame de Pandas.

    Parámetros:
        df : pd.DataFrame
            DataFrame de Pandas a analizar.
        params : dict, optional
            Diccionario con parámetros opcionales:
            - 'id_columns': lista de columnas identificadoras a excluir.
            - 'target': lista de columnas objetivo a excluir.
            - 'numeric_columns': lista de columnas numéricas a evaluar. 
              Si está vacío, se usan todas las numéricas del DataFrame.

    Retorna:
        tuple: Tupla con dos elementos:
            - Lista de columnas con varianza no cero.
            - Lista de columnas con varianza cero.
    """
    # imputar nulos con 0 para cálculo de varianza
    df = df.copy()

    id_columns = params.get('id_columns', ['pk_customer', 'tpk_release_dt'])
    target = params.get('target', ['default'])

    # determinar columnas numéricas
    numeric_columns = params.get('numeric_columns', [])
    if not numeric_columns:
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    # excluir id y target
    numeric_columns = [c for c in numeric_columns if c not in id_columns and c not in target]

    # calcular varianza
    variances = df[numeric_columns].var().to_dict()

    non_zero_var_columns = [col for col, var in variances.items() if var > 0]
    zero_var_columns = [col for col, var in variances.items() if var <= 0 or pd.isna(var)]

    print(
        f"Features numéricas originales: {len(numeric_columns)}\n"
        f"Features numéricas con varianza no nula: {len(non_zero_var_columns)}"
    )

    return non_zero_var_columns, zero_var_columns


# COMMAND ----------

def calcular_vif(
    df: pd.DataFrame,
    params: tp.Dict[str, tp.Any] = dict(),
) -> pd.DataFrame:
    """
    Calcula el Factor de Inflación de la Varianza (VIF) para cada característica en un DataFrame.

    Parámetros:
        df : pd.DataFrame
            DataFrame que contiene las características para las cuales se calculará el VIF.
        params : dict, opcional
            Diccionario con parámetros opcionales:
            - 'exclude_cols': lista de columnas a excluir del análisis.
            - 'id_columns': lista de columnas identificadoras a excluir.
            - 'target': lista de columnas objetivo a excluir.
            - 'fillna_default': valor por defecto para imputar nulos en variables numéricas.

    Retorna:
        pd.DataFrame: DataFrame con dos columnas:
            - 'feature': nombre de la característica.
            - 'VIF': valor del Factor de Inflación de la Varianza para la característica.
    """
    # obteniendo variables a excluir
    exclude_cols = params.get('exclude_cols', [])
    id_columns = params.get('id_columns', ['pk_customer', 'tpk_release_dt'])
    target = params.get('target', ['default'])

    # realizando calculo del VIF
    features = [col for col in df.columns if col not in exclude_cols + target + id_columns]
    fillna_value = params.get('fillna_default', 0)
    
    X = df.copy()[features]
    X = X.fillna(value=fillna_value)

    X_const = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i+1)
                       for i in range(len(X.columns))]
    
    return vif_data.sort_values(by='VIF', ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Anova

# COMMAND ----------

def anova_test(
    df: pd.DataFrame, 
    target_col: str, 
    feature_cols: tp.List[str],
) -> pd.DataFrame:
    """
    Realiza la prueba ANOVA entre una columna objetivo y múltiples columnas de características.

    Parámetros:
        df (pd.DataFrame): DataFrame de pandas que contiene los datos.
        target_col (str): Nombre de la columna objetivo.
        feature_cols (list): Lista de nombres de columnas de características.

    Retorna:
        pd.DataFrame: DataFrame con los resultados de la prueba ANOVA, ordenado por valor p.
    """
    results = {}
    for col in feature_cols:
        group_0 = df[df[target_col] == 0][col].dropna()
        group_1 = df[df[target_col] == 1][col].dropna()
        stat, pval = stats.f_oneway(group_0, group_1)
        results[col] = {'ANOVA_F': stat, 'p_value': pval}
    return pd.DataFrame(results).T.sort_values(by='p_value')

# COMMAND ----------

# MAGIC %md
# MAGIC ### V-Cramer

# COMMAND ----------

def v_cramer_cats(
    df: pd.DataFrame, 
    params: tp.Dict[str, tp.Any],
) -> pd.DataFrame:
    """
    Calcula el coeficiente V de Cramer entre una variable objetivo 
    y un conjunto de características categóricas.

    Parámetros:
        df (pd.DataFrame): El DataFrame que contiene los datos.
        params (Dict[str, Any]): Un diccionario con los parámetros necesarios.
            - target (str): El nombre de la columna objetivo.
            - categorical_features (List[str])

    Retorna:
        pd.DataFrame: Un DataFrame con los coeficientes V de Cramer para cada 
        característica categórica, ordenados de mayor a menor.
    """
    
    resultados = {} 
    target = params.get('target', 'default')
    features_categoricas = params.get('categorical_features', [])
    
    for feat in features_categoricas:
        # tabla de contingencia
        tabla = pd.crosstab(df[feat], df[target])

        # calculando V de Cramer
        chi2 = stats.chi2_contingency(tabla)[0]
        n = tabla.sum().sum()
        min_dim = min(tabla.shape) - 1
        v = np.sqrt(chi2 / (n * min_dim))
        
        resultados[feat] = v
    
    v_cramer_corr = pd.DataFrame({
        'categorical_features': resultados.keys(),
        'v_cramer': resultados.values()
    }).sort_values(by='v_cramer', ascending=False)
    
    return v_cramer_corr

# COMMAND ----------

# MAGIC %md
# MAGIC ### Information Value

# COMMAND ----------

def calculate_iv(
    df: pd.DataFrame, 
    feature: str, 
    target: str, 
    bins: int=10,
    categorical_cols: tp.List[str]=[]
) -> float:
    """
    Calcula el Information Value (IV) para una feature continua o categórica.
    Si es continua, primero aplica binning con quantiles.

    Parámetros:
        df (pd.DataFrame): DataFrame de pandas que contiene los datos.
        feature (str): Nombre de la columna de la feature a evaluar.
        target (str): Nombre de la columna objetivo binaria (0/1).
        bins (int, opcional): Número de bins a usar para variables continuas (default=10).
        categorical_cols (list, opcional): Lista de nombres de columnas categóricas.

    Retorna:
        float: Valor del Information Value (IV) para la feature especificada.
    """
    # condicion aplicada si la funcion es continua
    if pd.api.types.is_numeric_dtype(df[feature]) and (feature not in categorical_cols):
        try:
            df['_bin'] = pd.qcut(df[feature], q=bins, duplicates='drop')
        except ValueError:
            return np.nan
    else:
        df['_bin'] = df[feature]
    
    # calcular distribución por bin
    grouped = df.groupby('_bin')[target].agg(['count', 'sum'])
    grouped.columns = ['total', 'bads']
    grouped['goods'] = grouped['total'] - grouped['bads']

    grouped['proportion'] = grouped['total'] / grouped['total'].sum()
    grouped = grouped[(grouped['bads'] > 0) & (grouped['goods'] > 0)]
    
    # distribución proporcional
    total_goods = grouped['goods'].sum()
    total_bads = grouped['bads'].sum()
    grouped['dist_good'] = grouped['goods'] / total_goods
    grouped['dist_bad'] = grouped['bads'] / total_bads

    # calcular WoE y IV
    grouped['woe'] = np.log(grouped['dist_good'] / grouped['dist_bad'])
    grouped['iv'] = (grouped['dist_good'] - grouped['dist_bad']) * grouped['woe']

    print(
        f'{grouped}\n'
    )    
    iv = grouped['iv'].sum()

    return iv

# COMMAND ----------

def get_iv_dataframe(
    df: pd.DataFrame, 
    target_col: str, 
    feature_cols: tp.List[str], 
    bins: int=10, 
    categorical_cols: tp.List[str]=[],
) -> pd.DataFrame:
    """
    Calcula el Information Value (IV) para un conjunto de variables y retorna un DataFrame ordenado por IV.

    Parámetros:
        df (pd.DataFrame): DataFrame de pandas que contiene los datos.
        target_col (str): Nombre de la columna objetivo binaria (0/1).
        feature_cols (list): Lista de nombres de columnas de características a evaluar.
        bins (int, opcional): Número de bins a usar para variables continuas (default=10).
        categorical_cols (list, opcional): Lista de nombres de columnas categóricas.

    Retorna:
        pd.DataFrame: DataFrame con dos columnas: 'features' y 'IV', ordenado de mayor a menor IV.
    """
    iv_dict = {}
    for col in feature_cols:
        try:
            print(f"Feature: {col}")
            iv = calculate_iv(df[[col, target_col]].dropna(), col, target_col, bins=bins, categorical_cols=categorical_cols)
            iv_dict[col] = iv

        except:
            iv_dict[col] = np.nan
    iv_df = pd.DataFrame.from_dict(iv_dict, orient='index', columns=['IV'])
    iv_df = iv_df.sort_values(by='IV', ascending=False).reset_index()
    iv_df.columns = ['features', 'IV']
    
    return iv_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Funciones: Transformadores

# COMMAND ----------

# MAGIC %md
# MAGIC ### Columns Selector

# COMMAND ----------

class ColumnsSelector(BaseEstimator, TransformerMixin):
    """
    Clase para seleccionar columnas de un DataFrame.

    Atributos:
        columns (list of str): La lista de nombres de columnas a seleccionar.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.is_fitted = True
        return self

    def transform(self, X):        
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "X debe ser un DataFrame de pandas para preservar la información de columnas e índices"
            )

        X_transformed = X[self.columns]

        return X_transformed

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tratamiento Outliers

# COMMAND ----------

class OutliersHandler(BaseEstimator, TransformerMixin):
    """
    Clase para identificar y tratar valores atípicos en variables cuantitativas.

    Atributos:
    - numeric_cols: Lista de nombres de variables cuantitativas a analizar.
    - exclude_cols: Lista de columnas a excluir del tratamiento de outliers.
    """

    def __init__(
        self, 
        numeric_cols: tp.List[str],
        exclude_cols: tp.List[str],
    ):
        self.numeric_cols = numeric_cols
        self.exclude_cols = exclude_cols
        self.imputed_cols = None
        self._parametros = {}
        self.is_fitted = False

    def fit(
        self, 
        X: pd.DataFrame, 
        y=None
    ):
        self._parametros = {}
        self.imputed_cols = set(self.numeric_cols) - set(self.exclude_cols)
        
        for var in self.imputed_cols:
            varaux = X[var].dropna()
            params = {}
            
            # Calcular parámetros según distribución
            skewness = varaux.skew()
            params['skewness'] = skewness
            
            # Para distribución simétrica (|skewness| < 1)
            if abs(skewness) < 1:
                params['mean'] = varaux.mean()
                params['std'] = varaux.std()
                params['distribution'] = 'symmetric'
            # Para distribución asimétrica
            else:
                params['median'] = varaux.median()
                params['mad'] = sm.robust.mad(varaux, axis=0)
                params['distribution'] = 'asymmetric'
            
            # Calcular límites por RI
            Q1 = varaux.quantile(0.25)
            Q3 = varaux.quantile(0.75)
            IQR = Q3 - Q1
            params.update({
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'lower_bound': varaux.quantile(0.01) if (Q1 - 3 * IQR) < varaux.min() else (Q1 - 3 * IQR),
                'upper_bound': varaux.quantile(0.99) if (3 * IQR + Q3) > varaux.max() else (3 * IQR + Q3),
            })
            
            self._parametros[var] = params

        self.is_fitted = True
        return self

    def transform(
        self, 
        X: pd.DataFrame,
    ):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un DataFrame de pandas")

        X_transformed = X.copy()

        for var in self.imputed_cols:
            varaux = X[var]
            params = self._parametros[var]
            
            # Detectar outliers según la distribución
            if params['distribution'] == 'symmetric':
                z_scores = abs((varaux - params['mean']) / params['std'])
                z_theshold = 2 if abs(params['skewness']) > 0.7 else 3 
                outliers_dist = z_scores > z_theshold
            else:
                if params['median'] != 0:
                    modified_z = abs((varaux - params['median']) / params['mad'])
                    outliers_dist = modified_z > 8
                else:
                    outliers_dist = pd.Series(False, index=varaux.index)
            
            # Detectar outliers por RI
            outliers_iqr = (varaux < params['lower_bound']) | (varaux > params['upper_bound'])
            
            # Combinar criterios
            outliers = outliers_dist & outliers_iqr
            
            # Tratar outliers
            if outliers.any():
                # Valores inferiores al límite inferior
                mask_lower = (varaux < params['lower_bound']) & outliers
                X_transformed.loc[mask_lower, var] = params['lower_bound']
                
                # Valores superiores al límite superior
                mask_upper = (varaux > params['upper_bound']) & outliers
                X_transformed.loc[mask_upper, var] = params['upper_bound']
                
                # Reportar resultados
                n_outliers = outliers.sum()
                pct_outliers = 100 * n_outliers / len(varaux)
                print(
                    f'Variable {var} - Distribución: {params["distribution"]}\n'
                    f'Outliers tratados: {n_outliers:,.0f} ({pct_outliers:.2f}%)\n'
                    f'Límites aplicados: [{params["lower_bound"]:.2f}, {params["upper_bound"]:.2f}]\n'
                    f'Valores de imputación: lower = {params["lower_bound"]:.2f}, upper = {params["upper_bound"]:.2f}\n'
                )
            
            else:
                print(
                    f'Variable {var} - Distribución: {params["distribution"]}\n'
                    f'Sin outliers encontrados.\n'
                    f'Límites: [{params["lower_bound"]:.2f}, {params["upper_bound"]:.2f}]\n'
                    f'Posibles valores de imputación: lower = {params["lower_bound"]:.2f}, upper = {params["upper_bound"]:.2f}\n'
                )

        return X_transformed

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imputador nulos

# COMMAND ----------

class NumericalNullImputer:
    def __init__(self, 
                 numerical_cols: tp.List[str], 
                 tipo: str = 'media',
                 exclude_cols: tp.Dict[str, tp.Any] = None,
                 dummies_nulls: bool = True,
    ):
        """
        Clase para la imputación de valores faltantes en variables cuantitativas.

        Atributos:
        - numerical_cols: Lista de nombres de variables cuantitativas a imputar.
        - tipo: Tipo de imputación ('media', 'mediana' o 'aleatorio').
        - exclude_cols: Diccionario con el nombre de la feature y el valor a imputar como excepción.
        """

        self.numerical_cols = numerical_cols
        self.tipo = tipo
        self.exclude_cols = exclude_cols if exclude_cols is not None else {}
        self.parametros = {}  # alamacenando parámetros de imputación
        self.dummies_nulls = dummies_nulls
        self.variables_dummy = set()  # variables que tendrán dummy por nulos

    def fit(self, X, y=None):
        for var in self.numerical_cols:
            if var in self.exclude_cols:
                self.parametros[var] = self.exclude_cols[var]
                continue

            vv = X[var].copy()
            
            if (vv.isnull().any()) and (self.dummies_nulls == True):
                self.variables_dummy.add(var + '_null')
            
            if self.tipo == 'media':
                self.parametros[var] = np.nanmean(vv)
            
            elif self.tipo == 'mediana':
                self.parametros[var] = np.nanmedian(vv)
            
            elif self.tipo == 'aleatorio':
                x = vv.dropna()
                frec = x.value_counts(normalize=True).reset_index()
                frec.columns = ['Valor', 'Frec']
                frec = frec.sort_values(by='Valor')
                frec['FrecAcum'] = frec['Frec'].cumsum()
                self.parametros[var] = frec  # almacena la distribución de valores existentes
            else:
                raise ValueError("El tipo de imputación debe ser 'media', 'mediana' o 'aleatorio'")
        
        return self

    def transform(self, X):
        if not self.parametros:
            raise ValueError("El modelo debe ser ajustado con fit() antes de llamar a transform().")
        
        X_transformed = X.copy()
        n_rows = X.shape[0]

        for var in self.numerical_cols:
            # asegurando columnas dummy solo si hubo nulos en el entrenamiento
            if (self.dummies_nulls == True) and (var + '_null' in self.variables_dummy):
                X_transformed[var + '_null'] = X_transformed[var].isnull().astype(int)

            if var in self.exclude_cols:
                
                print(
                    f'Variable {var} - Valor de imputación: {self.parametros[var]}\n'
                    f'Cantidad de nulos imputados: {X_transformed[var].isnull().sum():,.0f} ({100 * X_transformed[var].isnull().sum()/n_rows:.2f}%)\n'
                )
                X_transformed[var].fillna(self.parametros[var], inplace=True)

            elif self.tipo in ['media', 'mediana']:

                print(
                    f'Variable {var} - Valor de imputación: {self.parametros[var]:,.2f}\n'
                    f'Cantidad de nulos imputados: {X_transformed[var].isnull().sum():,.0f} ({100 * X_transformed[var].isnull().sum()/n_rows:.2f}%)\n'
                )

                X_transformed[var].fillna(self.parametros[var], inplace=True)

            elif self.tipo == 'aleatorio':

                print(
                    f'Variable {var}\n'
                    f'Cantidad de nulos imputados: {X_transformed[var].isnull().sum():,.0f} ({100 * X_transformed[var].isnull().sum()/n_rows:.2f}%)\n'
                )

                frec = self.parametros[var]
                
                nan_indices = X_transformed[var].isnull()
                num_missing = nan_indices.sum()
                
                if num_missing > 0:
                    # generando valores aleatorios de forma eficiente
                    random_values = np.random.uniform(frec['FrecAcum'].iloc[0], 1, num_missing)
                    # vectorización: encontrar los valores imputados de una sola vez
                    imputed_values = np.interp(random_values, frec['FrecAcum'].values, frec['Valor'].values)
                    # asignando valores imputados a los nulos
                    X_transformed.loc[nan_indices, var] = imputed_values
    
        return X_transformed

# COMMAND ----------

class CategoricalNullImputer(BaseEstimator, TransformerMixin):
    """
    Esta clase imputa valores nulos en columnas categóricas utilizando la moda
    de cada columna. También crea columnas adicionales que indican si los valores
    originales eran nulos.

    Atributos:
        categorical_cols (list): Lista de nombres de columnas categóricas.
        exclude_cols (dict): Diccionario con el nombre de la feature y el valor a imputar como excepción.
    """

    def __init__(self, 
                 categorical_cols: tp.List[str],
                 exclude_cols: tp.Dict[str, tp.Any],
    ):
        self.categorical_modes = {}
        self.exclude_cols = exclude_cols
        self.null_columns = []
        self.imputed_cols = None
        self.categorical_cols = categorical_cols

    def fit(self, X, y=None):
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un DataFrame de pandas")
        
        self.imputed_cols = set(self.categorical_cols) - set(self.exclude_cols)
        for col in self.imputed_cols:
            self.categorical_modes[col] = X[col].mode()[0]
        
        self.null_columns = [
            col + '_null' for col in self.imputed_cols if X[col].isnull().any()
        ]

        return self

    def transform(self, X):
        X_transformed = X.copy()
        n_rows = X.shape[0]

        for col, mode in self.categorical_modes.items():
            if col + '_null' in self.null_columns:
                X_transformed[col + '_null'] = X_transformed[col].isnull().astype(int)
            
            print(
                f'Variable {col} - Moda de imputación: {mode}\n'
                f'Cantidad de nulos imputados: {X_transformed[col].isnull().sum():,.0f} ({100 * X_transformed[col].isnull().sum()/n_rows:.2f}%)\n'
            )

            X_transformed[col] = X_transformed[col].fillna(mode)

        # asegurar que todas las columnas '_null' estén presentes
        for null_col in self.null_columns:
            if null_col not in X_transformed.columns:
                X_transformed[null_col] = 0

        for col in self.exclude_cols:
            value = self.exclude_cols[col]

            print(
                f'Variable {col} - Valor de imputación: {value}\n'
                f'Cantidad de nulos imputados: {X_transformed[col].isnull().sum():,.0f} ({100 * X_transformed[col].isnull().sum()/n_rows:.2f}%)\n'
            )

            X_transformed[col] = X_transformed[col].fillna(value)
                
        return X_transformed

# COMMAND ----------

# MAGIC %md
# MAGIC ### Trannsformador Log continuas

# COMMAND ----------

class NumericalLogTransformer(BaseEstimator, TransformerMixin):
    """
    Clase para transformar columnas numéricas aplicando una transformación logarítmica
    a aquellas que tienen una distribución asimétrica.

    Atributos:
        numerical_cols (list of str): Lista de nombres de columnas numéricas.
        threshold_coef (float): Coeficiente de asimetría para considerar una distribución asimétrica.
                                Por default el umbral para el coeficiente de asimetría es 1.5.
    """

    def __init__(self,
                 numerical_cols: tp.List[str],
                 exclude_cols: tp.List[str] = [],
                 threshold_coef: float = 1.5,
    ):
        self.numerical_cols = numerical_cols
        self.exclude_cols = exclude_cols or []
        self.threshold_coef = threshold_coef
        self.asymmetric_coef = {}
        self.transformed_cols = []
        self.is_fitted = False
        
    def fit(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X debe ser un DataFrame de pandas")

        self.features = set(self.numerical_cols) - set(self.exclude_cols)

        for col in self.features:
            skew_coef = X[col].dropna().skew()
            self.asymmetric_coef[col] = skew_coef

            if abs(skew_coef) > self.threshold_coef:
                self.transformed_cols.append(col)

        self.is_fitted = True
        return self

    def transform(self, X):
        X_transformed = X.copy()

        for col in self.transformed_cols:
            X_transformed[col] = np.sign(X_transformed[col]) * np.log1p(np.abs(X_transformed[col]))
            print(
                f'Variable {col} con skew: {self.asymmetric_coef[col]:,.2f}'
            )
        
        return X_transformed


# COMMAND ----------

# MAGIC %md
# MAGIC ### Escalador

# COMMAND ----------

class ContinuousFeatureScaler(BaseEstimator, TransformerMixin):
    """
    Clase para escalar las características continuas utilizando StandardScaler o RobustScaler.

    Atributos:
        scaler (str): Tipo de escalador a utilizar. Puede ser 'standard' o 'robust'. 
        n_cats (int, opcional): Máximo número de categorías que debe tener una variable para ser considerada categórica.
        numerical_cols (list, opcional): Lista de nombres de columnas continuas.
        categorical_cols (list, opcional): Lista de nombres de columnas categóricas.
    """
    def __init__(
        self,
        scaler: str = 'robust',
        n_cats: int = 5,
        numerical_cols: tp.List[str] = None,
        categorical_cols: tp.List[str] = None,
    ):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.scaler = StandardScaler() if scaler == 'standard' else RobustScaler()
        self.n_cats = 5
    
    def fit(self, X, y=None):

        # identificando features continuas y categoricas (binarias)
        if self.categorical_cols is None:
            self.categorical_cols = X.columns[X.nunique() < self.n_cats].tolist()
        
        if self.numerical_cols is None:
            self.numerical_cols = X.columns.difference(self.categorical_cols).tolist()
        
        # ajustar el StandardScaler solo en las features continuas
        self.scaler.fit(X[self.numerical_cols])

        return self

    def transform(self, X):

        X_transformed = X.copy()
        
        # escalar las features continuas
        X_transformed[self.numerical_cols] = self.scaler.transform(
          X_transformed[self.numerical_cols]
        )
        
        return X_transformed

# COMMAND ----------

# MAGIC %md
# MAGIC ### Codficador WoE

# COMMAND ----------

class WoEEncoder:
    """
    Clase para codificar variables numéricas y categóricas utilizando Weight of Evidence (WoE).

    Atributos:
        bins (int): Número de bins para variables numéricas.
        min_bin_pct (float): Porcentaje mínimo de frecuencia para agrupar categorías poco frecuentes.
        woe_sim_threshold (float): Umbral de similitud para agrupar categorías por WoE.
        method (str): Método de binning para variables numéricas ('quantile' o 'uniform').
        numeric_cols (list, opcional): Lista de nombres de columnas numéricas.
        categorical_cols (list, opcional): Lista de nombres de columnas categóricas.
        exclude_cols (list, opcional): Lista de columnas a excluir del encoding.
    """
    def __init__(
        self, 
        bins: int=5, 
        min_bin_pct: float=0.05, 
        woe_sim_threshold: float=0.1, 
        method: str='quantile',
        numeric_cols: tp.List[str]=None, 
        categorical_cols: tp.List[str]=None, 
        exclude_cols: tp.List[str]=None
    ):
        self.bins = bins
        self.min_bin_pct = min_bin_pct
        self.woe_sim_threshold = woe_sim_threshold
        self.method = method
        self.numeric_cols = numeric_cols or []
        self.categorical_cols = categorical_cols or []
        self.exclude_cols = exclude_cols or []
        self.iv_dict = {}
        self.woe_dict = {}
        self.grouping_log = {}

    

    def fit(self, X, y):

        # consolidando features y target en un df
        df = X.copy()
        df['__target__'] = y
        self.iv_dict = {}
        self.woe_dict = {}
        self.grouping_log = {}

        self.numeric_cols = [col for col in self.numeric_cols if col not in self.exclude_cols]
        self.categorical_cols = [col for col in self.categorical_cols if col not in self.exclude_cols]

        for col in self.numeric_cols:
            print(f"\n[INFO] Procesando variable numérica: '{col}'")
            if self.method == 'quantile':
                df[col + '_binned'] = pd.qcut(df[col], q=self.bins, duplicates='drop')
            else:
                df[col + '_binned'] = pd.cut(df[col], bins=self.bins)
            woe_map, iv = self._calculate_woe_iv(df, col + '_binned', '__target__')
            self.woe_dict[col + '_binned'] = woe_map
            self.iv_dict[col + '_binned'] = iv
            print(f" - IV de '{col}': {iv:.4f}")

        for col in self.categorical_cols:
            df[col] = df[col].astype('str')
            print(f"\n[INFO] Procesando variable categórica: '{col}'")
            df, merged_low = self._group_low_freq_categories(df, col)
            self.grouping_log[col] = {'low_freq_groups': merged_low}
            woe_map, iv = self._calculate_woe_iv(df, col, '__target__')
            merged_woe = self._group_by_woe_similarity(woe_map, self.woe_sim_threshold)
            df[col] = df[col].replace(merged_woe)
            self.grouping_log[col]['woe_groups'] = merged_woe
            final_woe_map, iv = self._calculate_woe_iv(df, col, '__target__')
            self.woe_dict[col] = final_woe_map
            self.iv_dict[col] = iv
            print(f" - IV final de '{col}': {iv:.4f}")
        
        return self

    def transform(self, X):
        df = X.copy()
        for col in self.numeric_cols:
            print(f"\n[INFO] Transformando variable numérica: '{col}'")
            binned = pd.qcut(df[col], q=self.bins, duplicates='drop') if self.method == 'quantile' else pd.cut(df[col], bins=self.bins)
            df[col + '_binned'] = binned
            df[col] = df[col + '_binned'].map(self.woe_dict.get(col + '_binned', {})).fillna(0)
            df.drop(columns=[col + '_binned'], inplace=True)
        for col in self.categorical_cols:
            df[col] = df[col].astype('str')
            print(f"\n[INFO] Transformando variable categórica: '{col}'")
            if col in self.grouping_log:
                low_merge = self.grouping_log[col].get('low_freq_groups', {})
                woe_merge = self.grouping_log[col].get('woe_groups', {})
                df[col] = df[col].replace(low_merge)
                df[col] = df[col].replace(woe_merge)
            df[col] = df[col].map(self.woe_dict.get(col, {})).fillna(0)
        return df
    
    def _calculate_woe_iv(self, df, feature, target):
        eps = 0.0001
        df_agg = df.groupby(feature)[target].agg(['sum', 'count'])
        df_agg.columns = ['bad', 'total']
        df_agg['good'] = df_agg['total'] - df_agg['bad']
        total_good = df_agg['good'].sum()
        total_bad = df_agg['bad'].sum()
        df_agg['dist_good'] = df_agg['good'] / (total_good + eps)
        df_agg['dist_bad'] = df_agg['bad'] / (total_bad + eps)
        df_agg['woe'] = np.log((df_agg['dist_good'] + eps) / (df_agg['dist_bad'] + eps))
        df_agg['iv'] = (df_agg['dist_good'] - df_agg['dist_bad']) * df_agg['woe']
        iv = df_agg['iv'].sum()
        return df_agg['woe'].to_dict(), iv

    def _group_low_freq_categories(self, df, col):
        freq = df[col].value_counts(normalize=True)
        low_freq = freq[freq < self.min_bin_pct].index.tolist()
        if not low_freq:
            return df, {}
        print(f"\n[INFO] Agrupando categorías de '{col}' por baja distribución (<{self.min_bin_pct*100:.1f}%)")
        remaining = freq[~freq.index.isin(low_freq)].sort_values()
        merged = {}
        current_group = []
        current_sum = 0.0
        for cat in low_freq:
            current_group.append(cat)
            current_sum += freq[cat]
            if current_sum >= self.min_bin_pct:
                new_cat = "_".join(current_group)
                for g in current_group:
                    merged[g] = new_cat
                print(f" - Agrupadas por frecuencia: {current_group} → '{new_cat}' ({current_sum*100:.2f}%)")
                current_group = []
                current_sum = 0.0
        if current_group:
            new_cat = "_".join(current_group)
            for g in current_group:
                merged[g] = new_cat
            print(f" - Agrupadas por frecuencia (resto): {current_group} → '{new_cat}' ({current_sum*100:.2f}%)")
        df[col] = df[col].replace(merged)
        return df, merged

    def _group_by_woe_similarity(self, woe_map, threshold):
        sorted_items = sorted(woe_map.items(), key=lambda x: x[1])
        groups = []
        current_group = [sorted_items[0][0]]
        current_woe = sorted_items[0][1]
        for cat, woe in sorted_items[1:]:
            if abs(woe - current_woe) <= threshold:
                current_group.append(cat)
            else:
                groups.append(current_group)
                current_group = [cat]
                current_woe = woe
        groups.append(current_group)
        new_map = {}
        print(f"\n[INFO] Agrupando categorías por similitud de WoE (threshold = {threshold})")
        for group in groups:
            new_cat = "_".join([str(g) for g in group])
            for old_cat in group:
                new_map[old_cat] = new_cat
            if len(group) > 1:
                print(f" - Agrupadas por WoE similar: {group} → '{new_cat}'")
        return new_map

    def get_iv(self):
        return self.iv_dict

    def get_category_woe_mapping(self):
        return self.woe_dict

    def get_grouping_log(self):
        return self.grouping_log

# COMMAND ----------

# MAGIC %md
# MAGIC ### Entrenamiento

# COMMAND ----------

class ClassifierByProduct:
    def __init__(self, model_cls, model_params=None, fit_params=None, threshold=0.5):
        """
        Clase para clasificación binaria simple con ajuste de umbral.
        
        Atributos:
            - model_cls: clase del modelo (ej: RandomForestClassifier, lgb.LGBMClassifier, LogisticRegression)
            - model_params: diccionario con hiperparámetros del modelo
            - fit_params: diccionario con parámetros opcionales que se pasarán al fit
            - threshold: umbral para convertir scores de probabilidad en clases
        """
        self.model = model_cls(**(model_params or {}))
        self.fit_params = fit_params or {}
        self.threshold = float(threshold)

    def fit(self, X, y):
        self.model.fit(X, y, **self.fit_params)
        return self

    def predict(self, X):
        proba_pos = self.model.predict_proba(X)[:, 1]
        y_pred = (proba_pos > self.threshold).astype(int)
        return y_pred, proba_pos

# COMMAND ----------

def get_sample_weight(
    y_train: pd.DataFrame,
    n: int = 1,
) -> np.array:
    """
    Calcula los pesos de muestra para manejar el desbalance de clases.

    Parámetros:
        y_train (pd.DataFrame): El conjunto de etiquetas de entrenamiento.

    Retorna:
        np.array: Un array con los pesos de muestra.
    """
    y_train = np.array(y_train).ravel()
    class_weights = {0: 1.0, 1: (sum(y_train == 0) / sum(y_train == 1)) ** (1 / n)}
    sample_weight = np.array([class_weights[i] for i in y_train])

    return sample_weight

# COMMAND ----------

# MAGIC %md
# MAGIC ## Funciones: Evaluación

# COMMAND ----------

# MAGIC %md
# MAGIC ## Backtesting

# COMMAND ----------

def _generate_backtesting_dates(
    freq: str, start_date: str, end_date: str
) -> list[pd.Timestamp]:
    """
    Función auxiliar para generar fechas de backtesting con parámetros más detallados.
    Toma una frecuencia y una fecha de inicio, y devuelve una lista de fechas que están
    espaciadas según la frecuencia

    Parámetros:
      freq (str): La frecuencia del backtesting. Elija 'monthly', 'quarterly', 'yearly', 'bi_monthly', 'weekly' o 'semi_annually'.
      start_date (str): Fecha de inicio del backtest.
      end_date (str): Fecha de finalización del backtest.

    Retorna:
      Una lista de fechas
    """
    today = pd.Timestamp.today().floor("D")
    start_date = pd.to_datetime(start_date)
    if freq == "monthly":
        dates = pd.date_range(start_date, today, freq="MS").tolist()
    elif freq == "quarterly":
        dates = pd.date_range(start_date, today, freq="QS").tolist()
    elif freq == "yearly":
        dates = pd.date_range(start_date, today, freq="AS").tolist()
    elif freq == "bi_monthly":
        dates = pd.date_range(start_date, today, freq="BMS").tolist()[::2]
    elif freq == "weekly":
        dates = pd.date_range(start_date, end_date, freq="W-MON").tolist()
    elif freq == "semi_annually":
        dates = pd.date_range(start_date, today, freq="6MS").tolist()
    else:
        raise WrongIngestedParameter(
            "Frecuencia inválida. Elija 'monthly', 'quarterly', 'yearly', 'bi_monthly', 'weekly' o 'semi_annually'"
        )
    dates = [date for date in dates if pd.to_datetime(date) <= pd.to_datetime(end_date)]

    return dates

# COMMAND ----------

def _sum_backtesting_frequency(date: pd.Timestamp, frequency: str) -> pd.Timestamp:
    """Agregar período según la frecuencia de backtesting.

    Agrega a la fecha de backtesting la frecuencia de backtesting.
    Toma una fecha y una frecuencia y devuelve la siguiente fecha en la secuencia.

    Parámetros:
      date (pd.Timestamp): La fecha a la que queremos agregar la frecuencia.
      frequency (str): La frecuencia del backtest. Elija 'monthly', 'quarterly', 'yearly', 'bi_monthly', 'weekly' o 'semi_annually'.
      
    Retorna:
      Una nueva fecha con la frecuencia agregada.
    """
    if frequency == "monthly":
        new_date = date + pd.DateOffset(months=1)
    elif frequency == "quarterly":
        new_date = date + pd.DateOffset(months=3)
    elif frequency == "yearly":
        new_date = date + pd.DateOffset(months=12)
    elif frequency == "bi_monthly":
        new_date = date + pd.DateOffset(months=2)
    elif frequency == "weekly":
        new_date = date + pd.DateOffset(weeks=1)
    elif frequency == "semi_annually":
        new_date = date + pd.DateOffset(months=6)
    else:
        msg = "Parámetro de frecuencia no permitido, use monthly, bi_monthly, quarterly o yearly"
        raise ValueError(msg)

    return pd.to_datetime(new_date)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Métricas

# COMMAND ----------

def compute_binary_classification_metrics(
    y_true, y_pred, y_score
) -> tp.Dict[str, float]:
    """Calcula métricas de clasificación.

    Calcula varias métricas de clasificación para un problema de clasificación.

    Parámetros:
        y_true (np.array): Array de etiquetas verdaderas, con forma (n_samples,).
        y_pred (np.array): Array de etiquetas predichas, con forma (n_samples,).
        y_score (np.array): Array de puntuaciones o probabilidades predichas, con forma (n_samples,).

    Retorna:
        Diccionario con las siguientes claves:
            - 'accuracy': Puntuación de precisión
            - 'balanced_accuracy': Puntuación de precisión balanceada
            - 'f1': Puntuación F1
            - 'f1_micro': Puntuación F1 micro-promediada
            - 'f1_macro': Puntuación F1 macro-promediada
            - 'f1_weighted': Puntuación F1 ponderada
            - 'precision': Puntuación de precisión
            - 'precision_micro': Precisión micro-promediada
            - 'precision_macro': Precisión macro-promediada
            - 'precision_weighted': Precisión ponderada
            - 'recall': Puntuación de recall
            - 'recall_micro': Recall micro-promediado
            - 'recall_macro': Recall macro-promediado
            - 'recall_weighted': Recall ponderado
            - 'roc_auc': Puntuación ROC AUC
            - 'roc_auc_ovr': ROC AUC One-vs-Rest (OvR)
            - 'roc_auc_ovo': ROC AUC One-vs-One (OvO)
            - 'roc_auc_ovr_weighted': ROC AUC ponderado OvR
            - 'roc_auc_ovo_weighted': ROC AUC ponderado OvO
            - 'matthews_corrcoef': Coeficiente de correlación de Matthews
            - 'fpr': Tasa de faltos positivos
            - 'fnr': Tasa de falsos negativos

    """
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision": precision_score(y_true, y_pred),
        "precision_micro": precision_score(y_true, y_pred, average="micro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred),
        "recall_micro": recall_score(y_true, y_pred, average="micro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "f1_1score": fbeta_score(y_true, y_pred, beta=1.1),
        "f1_2score": fbeta_score(y_true, y_pred, beta=1.2),
        "f1_3score": fbeta_score(y_true, y_pred, beta=1.3),
        "f1_4score": fbeta_score(y_true, y_pred, beta=1.4),
        "f1_5score": fbeta_score(y_true, y_pred, beta=1.5),
        "f1_6score": fbeta_score(y_true, y_pred, beta=1.6),
        "f2_score": fbeta_score(y_true, y_pred, beta=2),
        "f3_score": fbeta_score(y_true, y_pred, beta=3),
        "f4_score": fbeta_score(y_true, y_pred, beta=4),
        "f2_score_weighted": fbeta_score(y_true, y_pred, beta=2, average="weighted"),
        "f3_score_weighted": fbeta_score(y_true, y_pred, beta=3, average="weighted"),
        "f4_score_weighted": fbeta_score(y_true, y_pred, beta=4, average="weighted"),
    }
    # si las probabilidades están disponibles en el modelo
    if y_score is not None:
        # Calcular Brier Score Loss
        brier = brier_score_loss(y_true, y_score)

        # Calcular la puntuación AUC
        auc_value = roc_auc_score(y_true, y_score)

        # Calcular el coeficiente de Gini
        gini_coefficient = 2 * auc_value - 1

        # Calcular la estadística KS
        ks_statistic = _compute_ks_statistic(y_true=y_true, y_score=y_score)

        # Calcular PR-AUC
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)     

        prob_scores = {
            "brier_score_loss": brier,
            "roc_auc": auc_value,
            "pr_auc": pr_auc,
            "gini_coefficient": gini_coefficient,
            "ks_statistic": ks_statistic,
            "roc_auc_ovr": roc_auc_score(y_true, y_score, multi_class="ovr"),
            "roc_auc_ovo": roc_auc_score(y_true, y_score, multi_class="ovo"),
            "roc_auc_ovr_weighted": roc_auc_score(
                y_true, y_score, multi_class="ovr", average="weighted"
            ),
            "roc_auc_ovo_weighted": roc_auc_score(
                y_true, y_score, multi_class="ovo", average="weighted"
            ),
        }
        metrics.update(prob_scores)

    return metrics


def _compute_ks_statistic(y_true: np.array, y_score: np.array) -> np.array:
    """
    Calcula la estadística de Kolmogorov-Smirnov (KS) para las etiquetas verdaderas
    y las puntuaciones predichas dadas.

    Parámetros:
        y_true (np.array): Array de etiquetas binarias verdaderas (0 o 1).
        y_score (np.array): Array de puntuaciones o probabilidades predichas para
            la clase positiva.

    Retorna:
        np.array: La estadística KS, una medida de la diferencia máxima
            entre los CDF empíricos de las clases positiva y negativa.
    """
    # Separar las probabilidades predichas para las clases positiva y negativa
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]

    # Calcular los CDF empíricos para las clases positiva y negativa
    pos_cdf = np.sort(pos_scores)
    neg_cdf = np.sort(neg_scores)

    # Calcular la estadística KS
    ks_statistic, _ = stats.ks_2samp(pos_cdf, neg_cdf)
    return ks_statistic

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluación

# COMMAND ----------

# MAGIC %md
# MAGIC ### Curva precision-recall

# COMMAND ----------

def plot_precision_recall_curves(
    y_true: tp.Any, 
    y_score: tp.Any, 
    reference: float=0.5
):
    """
    Grafica las curvas de precisión-recall y precisión-recall vs umbrales.

    Parámetros:
        y_true (np.array): Array de etiquetas verdaderas, con forma (n_samples,).
        y_score (np.array): Array de puntuaciones o probabilidades predichas, con forma (n_samples,).
        reference (float): Valor de referencia para la línea horizontal en la gráfica de precisión-recall vs umbrales.

    Retorna:
        fig: La figura de la gráfica.
    """
    sns.set_style("darkgrid")
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    best_threshold_index = np.nanargmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    best_f1_score = f1_scores[best_threshold_index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # precisions-recalls curve
    ax1.plot(recalls, precisions, marker='.', color='#00968B')
    ax1.axvline(x=recalls[best_threshold_index], color='r', linestyle='--', label='best recall')
    ax1.axhline(y=precisions[best_threshold_index], color='b', linestyle='--', label='best precision')
    ax1.set_xlabel('recalls')
    ax1.set_ylabel('precisions')
    ax1.set_title('precisions-recalls curve')
    ax1.legend()

    # precision-Recall vs Thresholds
    ax3 = ax2.twinx()
    ax2.plot(thresholds, precisions[:-1], 'b-')
    ax2.set_xlabel('thresholds')
    ax2.set_ylabel('precision', color='b')
    ax2.set_ylim([0, 1])
    ax2.tick_params(axis='y', labelcolor='b')

    ax3.plot(thresholds, recalls[:-1], 'r-')
    ax3.set_ylabel('recall', color='r')
    ax3.set_ylim([0, 1])
    ax3.tick_params(axis='y', labelcolor='r')

    intersection_threshold = thresholds[np.argmin(np.abs(precisions[:-1] - recalls[:-1]))]

    ax2.axvline(x=best_threshold, 
                color='g', 
                linestyle='--', 
                label=f'best threshold: {best_threshold:.3f} (F1: {best_f1_score:.3f})')

    ax2.axhline(y=reference, 
                color='k', 
                linestyle='--', 
                linewidth=0.75,
                label=f'precision-recall: {reference:.3f}')

    ax2.set_title('precisions-recalls-thresholds')

    fig.tight_layout()
    ax2.legend(loc='upper right')

    plt.close(fig)
    return fig

# COMMAND ----------

# MAGIC %md
# MAGIC ### Curva KS

# COMMAND ----------

def plot_ks_curve(
  y_true: tp.Any, 
  y_score: tp.Any, 
  figsize: tp.Tuple[int, int]=(8, 5), 
  title: str='Curva KS'
):
    """
    Genera un gráfico de la curva KS (Kolmogorov-Smirnov) para un modelo de clasificación.
    
    Parámetros:
        y_true: Array con las etiquetas verdaderas (0 y 1)
        y_score: Array con las probabilidades predichas para la clase positiva
        figsize: Tamaño de la figura (ancho, alto) en pulgadas
        title: Título del gráfico
        
    Retorna:
        fig: Objeto figura de matplotlib
    """
    sns.set_style("darkgrid")

    # Comprobar si y_true es un array, y si no lo es convertirlo a array
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true).ravel()
    
    # Separar los scores de los casos positivos y negativos
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    
    # Calcular distribuciones acumulativas
    thresholds = np.linspace(0, 1, 100)
    cum_pos = np.array([np.sum(pos_scores <= threshold) / len(pos_scores) for threshold in thresholds])
    cum_neg = np.array([np.sum(neg_scores <= threshold) / len(neg_scores) for threshold in thresholds])
    
    # Calcular estadístico KS
    ks_diffs = cum_neg - cum_pos
    ks_statistic = np.max(ks_diffs)
    ks_threshold = thresholds[np.argmax(ks_diffs)]
    
    # Crear figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Graficar curvas acumulativas
    ax.plot(thresholds, cum_neg, 'r-', label='negativos (no default)')
    ax.plot(thresholds, cum_pos, 'g-', label='positivos (default)')
    
    # Graficar diferencia KS
    ax.plot(thresholds, ks_diffs, 'b--', label='diferencia (KS)')
    
    # Marcar el punto máximo (estadístico KS)
    ax.plot([ks_threshold, ks_threshold], [cum_pos[np.argmax(ks_diffs)], cum_neg[np.argmax(ks_diffs)]], 
            'k-', linewidth=2)
    ax.scatter([ks_threshold], [cum_pos[np.argmax(ks_diffs)]], color='g', s=50)
    ax.scatter([ks_threshold], [cum_neg[np.argmax(ks_diffs)]], color='r', s=50)
    
    # Añadir texto con el valor KS
    ax.text(ks_threshold + 0.02, cum_pos[np.argmax(ks_diffs)] + (ks_statistic/2), 
            f'KS = {ks_statistic:.4f}\nthreshold = {ks_threshold:.4f}',
            fontsize=11, bbox=dict(facecolor='white', alpha=0.7))
    
    # Configurar ejes y leyenda
    ax.set_xlabel('umbral de probabilidad')
    ax.set_ylabel('proporción acumulada')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Añadir líneas de referencia
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=1, color='gray', linestyle='-', alpha=0.3)
    
    plt.tight_layout()

    plt.close(fig)    
    return fig, ks_threshold

# COMMAND ----------

# MAGIC %md
# MAGIC ### Curva Lorenz (Gini)

# COMMAND ----------

def plot_gini_curve(
    y_true: tp.Any, 
    y_score: tp.Any, 
    figsize: tp.Tuple[int, int]=(8, 5), 
    title: str='Curva de Lorenz - Coeficiente de Gini'
) -> tp.Tuple[plt.Figure, float]:
    """
    Genera un gráfico de la Curva de Lorenz y calcula el coeficiente de Gini (usando AUC ROC).
    
    Parámetros:
        y_true: Array con las etiquetas verdaderas (0 y 1)
        y_score: Array con las probabilidades predichas para la clase positiva
        figsize: Tamaño de la figura (ancho, alto) en pulgadas
        title: Título del gráfico
        
    Retorna:
        fig: Objeto figura de matplotlib
        gini: Valor del coeficiente de Gini (2*AUC - 1)
    """

    sns.set_style("darkgrid")

    # Asegurarse de que sean arrays
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    # Ordenar por las predicciones descendentes
    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order]

    # Curvas acumulativas para la Lorenz
    cum_bad = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
    cum_pop = np.arange(1, len(y_true_sorted) + 1) / len(y_true_sorted)

    # Calcular Gini a partir del AUC ROC
    auc_roc = roc_auc_score(y_true, y_score)
    gini = 2 * auc_roc - 1

    # Gráfico
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(cum_pop, cum_bad, color='blue', label='Curva de Lorenz')
    ax.plot([0, 1], [0, 1], color='black', linestyle='--', label='Línea de Equidad')

    # Anotación con el Gini correcto
    ax.text(0.6, 0.2, f'Gini (2*AUC-1)= {gini:.0%}', fontsize=12, 
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

    ax.set_xlabel('% acumulado de población')
    ax.set_ylabel('% acumulado de incumplidos')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)

    # Líneas de referencia
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=1, color='gray', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.close(fig)

    return fig

# COMMAND ----------

# MAGIC %md
# MAGIC ### Curva calibración

# COMMAND ----------

def plot_calibration_curve(
    y_test: tp.Any, 
    y_score: tp.Any, 
    n_bins=15
):
    """
    Genera un gráfico de la curva de calibración para un modelo clasificador.

    Parámetros:
    y_test (array-like): Valores reales de la variable objetivo.
    y_score (array-like): Probabilidades predichas por el modelo.
    n_bins (int): Número de contenedores para la curva de calibración.

    Retorna:
    fig: Objeto figura de matplotlib
    """
    sns.set_style("darkgrid")
    x, y = calibration_curve(y_test, y_score, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([0, 1], [0, 1], linestyle='--', label='idealmente calibrado')
    
    # curva de calibración del modelo
    ax.plot(y, x, marker='.', label='modelo clasificador')
    
    leg = ax.legend(loc='upper left')
    ax.set_xlabel('\nprobabilidad promedio predicha en cada contenedor')
    ax.set_ylabel('proporción de positivos')
    
    plt.close(fig)
    return fig

# COMMAND ----------

# MAGIC %md
# MAGIC ### Matriz confusión

# COMMAND ----------

# confusion matrix
def plot_confusion_matrix(
    y_true: tp.Any, 
    y_pred: tp.Any,
    labels: tp.Tuple[str, str]=('non-default', 'default'),
):
    """
    Grafica la matriz de confusión y la matriz de confusión en porcentajes.

    Parámetros:
        y_true (np.array): Array de etiquetas verdaderas, con forma (n_samples,).
        y_pred (np.array): Array de etiquetas predichas, con forma (n_samples,).
        labels (tuple): Etiquetas para las clases 0 y 1, respectivamente.

    Retorna:
        fig: Figura matplotlib generada
    """
    matrix_confusion = confusion_matrix(y_true, y_pred)
    
    # calculate percentages
    cm_per = matrix_confusion / matrix_confusion.sum(axis=1)[:, np.newaxis] * 100

    # create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # original confusion matrix heatmap
    sns.heatmap(matrix_confusion, annot=True, cmap='YlGnBu', fmt=',.0f', ax=ax1, annot_kws={"size": 16})
    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Prediction")
    ax1.set_ylabel("Real")
    ax1.set_title("matriz de confusion\n")

    # percentage confusion matrix heatmap
    sns.heatmap(cm_per, annot=True, cmap='YlGnBu', fmt='.0f', vmin=0, vmax=100, cbar_kws={'format': '%.0f%%'}, ax=ax2, annot_kws={"size": 16})
    ax2.set_xticklabels(labels)
    ax2.set_yticklabels(labels)
    for text in ax2.texts:
        text.set_text(f'{text.get_text()}%')
    ax2.set_ylabel("Real\n")
    ax2.set_xlabel("\nPrediction")
    ax2.set_title("matriz de confusion: porcentajes\n")

    # classification report
    report = classification_report(y_true, y_pred, output_dict=True, target_names=labels)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop(columns=['support'])

    # add extra space between individual class metrics and overall metrics
    report_str = report_df.to_string(float_format='%.2f')
    report_str = report_str.replace('\naccuracy', '\n\naccuracy')

    # add classification report text with more space and enclosed in a box
    fig.text(0.20, -0.15, report_str, ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='black'))

    plt.tight_layout()
    
    plt.close(fig)
    return fig

# COMMAND ----------

# MAGIC %md
# MAGIC ### Curva ROC

# COMMAND ----------

def plot_roc_and_optimal_thresholds(y_true, y_score):
    """
    Esta función traza la curva ROC y calcula los umbrales óptimos para un clasificador binario.

    Parámetros:
    y_true (array-like): Valores verdaderos de la variable objetivo.
    y_score (array-like): Puntajes predichos por el modelo.

    La función muestra una gráfica con la curva ROC y una tabla con los 5 umbrales óptimos
    basados en la mínima distancia entre sensibilidad y especificidad.
    """
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    sensibilidad = tpr
    especificidad = 1 - fpr

    optimal_threshold = pd.DataFrame({"sensibilidad": sensibilidad,
                                      "especificidad": especificidad,
                                      "punto_corte": thresholds})

    optimal_threshold['distancia'] = np.abs(optimal_threshold['sensibilidad'] - optimal_threshold['especificidad'])
    optimal_threshold.sort_values('distancia', ascending=True, inplace=True)
    optimal_threshold = optimal_threshold.head(5).round(3)

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]})

    # ROC curve
    ax[0].plot(fpr, tpr)
    ax[0].plot(fpr, fpr, linestyle='--', color='k')
    ax[0].set_xlabel('false positive rate')
    ax[0].set_ylabel('true positive rate')
    ax[0].set_title('ROC curve')

    # Optimal thresholds DataFrame
    ax[1].axis('off')
    tbl = ax[1].table(cellText=optimal_threshold.values, colLabels=optimal_threshold.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.2)

    plt.tight_layout()
    plt.close(fig)
    
    return fig

# COMMAND ----------

# MAGIC %md
# MAGIC ### Shap values

# COMMAND ----------

def shap_values_classifier(
    model, 
    X_preprocessed: pd.DataFrame, 
    max_features=40,
    title="SHAP values (clase positiva)"
):
   """
   Genera un gráfico de SHAP values para un modelo de clasificación binaria.
   
   Parámetros:
       model: Modelo entrenado (CatBoost u otro modelo compatible con TreeExplainer)
       X_preprocessed: Datos preprocesados para generar los SHAP values
       max_features: Número máximo de características a mostrar en el gráfico
       title: Título del gráfico
       
   Retorna:
       fig: Figura matplotlib generada
   """
   sns.set_style('white')
   
   # Crear el explicador y calcular los SHAP values
   explainer = shap.TreeExplainer(model=model)
   shap_values = explainer(X_preprocessed)
   
   # Manejar diferentes formatos de salida SHAP (binario vs multiclase)
   if hasattr(shap_values, 'shape') and len(shap_values.shape) > 2:
       shap_values_display = shap_values[:, :, 1]  # Clase positiva (índice 1)
   else:
       shap_values_display = shap_values
   
   # Crear el gráfico de beeswarm
   fig, _ = plt.subplots()
   shap.plots.beeswarm(
       shap_values_display,
       max_display=max_features,
       color=plt.get_cmap("coolwarm"),
       show=False,
   )
   
   # Ajustar el tamaño y formato del gráfico
   original_size = fig.get_size_inches()
   fig.set_size_inches(1.75 * original_size[0], 2 * original_size[0] * 3 / 4)
   plt.tight_layout()
   
   # Añadir título
   plt.title(title, fontdict={"fontsize": 15})

   plt.close(fig)
   
   return fig, shap_values