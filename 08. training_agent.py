# Databricks notebook source
# MAGIC %md
# MAGIC # TFM: Next Move Banking
# MAGIC **Autor:** Coraima Castillo
# MAGIC
# MAGIC En este notebook se realiza el entrenamiento del agente que interactúa con los ejecutivos del banco.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importando librerías

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as f
import mlflow
from collections import Counter, defaultdict
import shap

import re, os, requests, json
import time, traceback
import typing as tp
from openai import OpenAI

# COMMAND ----------

# MAGIC %run /Workspace/Users/coraimac@ucm.es/TFM/utils_functions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Keys

# COMMAND ----------

OPENAI_API_KEY = dbutils.secrets.get(scope="tfm_next_move_banking", key="openai_api")
TELEGRAM_TOKEN = dbutils.secrets.get(scope="tfm_next_move_banking", key="telegram")
CHAT_ID = dbutils.secrets.get(scope="tfm_next_move_banking", key="chat_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importando datos

# COMMAND ----------

container = "datos"
account = spark.conf.get("adls.account.name")
folder = "tfm"
path = f"abfss://{container}@{account}.dfs.core.windows.net/{folder}/inference/"

# COMMAND ----------

predictions = spark.read.parquet(path + 'predictions')
master_table = spark.read.parquet(path + 'master_table')
portfolio = spark.createDataFrame(
    [
        ('de86d38cf0f7de68e4061094f1c80627371a83f59e53aeb9d63cf0f3d82c17f3', CHAT_ID), 
        ('c0ecc85f085ff4616ce6db55dafc9b160eaaa962f3c95514ad45cf00f7d15137', CHAT_ID),
        ('bf00eaf8dbda79ae1c6992c5a78d233b515c1312ed7b131877bd868625b502c0', CHAT_ID),
        ('d7f2de43e7503f33d713c22a20963a581f7704297c429355594f5298293d6bbb', CHAT_ID),
        ('beb4a828cf2cc85b27c51af10a6c0167ddaa3ff3262660ba60ea49a1f4bfa57b', CHAT_ID),        
    ],
    ['client_id', 'chat_id']
)

# COMMAND ----------

max_date = predictions.select(f.max('mon')).first()[0]

predictions =  predictions.filter(f.col('mon') == max_date).toPandas()
master_table =  master_table.filter(f.col('mon') == max_date).toPandas()
portfolio = portfolio.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelos

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

# COMMAND ----------

# obteniendo features disponibles en el modelado
latest_version = get_latest_model_version("model_product_3")
model_uri = f"models:/model_product_3/{latest_version}"   
model_product_3 = mlflow.sklearn.load_model(model_uri)

print(
    f'Versión del modelo utilizada para Producto 3: {latest_version}'
)

# COMMAND ----------

model_3 = model_product_3['ClassifierProduct3'].model
preprocessing_3 = model_product_3[:-1]

# COMMAND ----------

# obteniendo features disponibles en el modelado
latest_version = get_latest_model_version("model_product_4")
model_uri = f"models:/model_product_4/{latest_version}"   
model_product_4 = mlflow.sklearn.load_model(model_uri)

print(
    f'Versión del modelo utilizada para Producto 4: {latest_version}'
)

# COMMAND ----------

model_4 = model_product_4['ClassifierProduct4'].model
preprocessing_4 = model_product_4[:-1]

# COMMAND ----------

model_by_product = {
    1: {'model': model_1, 'type': 'randomforest', 'transformer': preprocessing_1},
    3: {'model': model_3, 'type': 'lightgbm', 'transformer': preprocessing_3},
    4: {'model': model_4, 'type': 'lightgbm', 'transformer': preprocessing_4},
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ppst-procesamiento datos

# COMMAND ----------

# obteniendo solo recomendaciones
df_tidy = predictions.loc[predictions['pred'] == 1].copy()
df_tidy = df_tidy[['mon', 'client_id', 'product_id', 'name_product', 'prob']]

df_tidy['mon'] = pd.to_datetime(df_tidy['mon'], errors='coerce')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agente

# COMMAND ----------

# MAGIC %md
# MAGIC ### Formateadores

# COMMAND ----------

def format_recs(df_sel: pd.DataFrame) -> str:
    """
    Formatea las recomendaciones de productos para cada cliente en un DataFrame.

    Parámetros:
        df_sel (pd.DataFrame): DataFrame con columnas 'client_id', 'product_id', 'name_product', 'prob'.

    Retorna:
        str: Texto formateado con las recomendaciones por cliente.
    """
    if df_sel.empty:
        return "No tengo recomendaciones para tu cartera en este corte."
    df = df_sel.sort_values(["client_id","prob"], ascending=[True, False])
    lines = ["Oportunidades — Resumen"]
    for cid, grp in df.groupby("client_id"):
        lines.append(f"\nCliente: {cid}")
        for _, r in grp.iterrows():
            pname = r.get("name_product") or f"Producto {int(r['product_id'])}"
            lines.append(f"- {pname} — Prob {r['prob']:.0%}")
    return "\n".join(lines)

def format_reasons(df_sel: pd.DataFrame, reasons_lookup: dict) -> str:
    """
    Formatea las recomendaciones de productos junto con las razones (SHAP) para cada cliente.

    Parámetros:
        df_sel (pd.DataFrame): DataFrame con columnas 'client_id', 'product_id', 'name_product', 'prob'.
        reasons_lookup (dict): Diccionario con claves (client_id, product_id) y valores lista de razones.

    Retorna:
        str: Texto formateado con las recomendaciones y razones por cliente.
    """
    if df_sel.empty:
        return "No tengo recomendaciones para tu cartera en este corte."
    df = df_sel.sort_values(["client_id","prob"], ascending=[True, False])
    lines = ["Oportunidades — Con razones (SHAP)"]
    for cid, grp in df.groupby("client_id"):
        lines.append(f"\nCliente: {cid}")
        for _, r in grp.iterrows():
            pid = int(r["product_id"])
            pname = r.get("name_product") or f"Producto {pid}"
            rs = reasons_lookup.get((cid, pid), ["(razones SHAP se añadirán al conectar el modelo)"])
            lines.append(f"- {pname} — Prob {r['prob']:.0%}")
            for reason in rs[:2]:
                lines.append(f"  • {reason}")
    return "\n".join(lines)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Intérprete

# COMMAND ----------

client = OpenAI(api_key=OPENAI_API_KEY)

def llm_parse_intent(
    text: str
) -> tp.Dict:
    """
    Usa GPT-4o mini para extraer intención/es y posibles client_ids mencionados.

    Retorna SIEMPRE:
      {
        "intents": ["recs"] | ["razones"] | ["recs","razones"],
        "client_ids": [<str>...]
      }
    """

    SYSTEM = (
        "Devuelve EXCLUSIVAMENTE un JSON con el esquema:\n"
        '{"intents":["recs" o "razones"],"client_ids":[string,...]}\n'
        "- 'intents' puede ser ['recs'], ['razones'] o ['recs','razones'] si el usuario pide ambos; 'recs' son recomendaciones de productos para los clientes y 'razones' son razones.\n"
        "- 'client_ids' es la lista de IDs mencionados; si no hay, []."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=120,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text}
        ]
    )
    raw = resp.choices[0].message.content

    try:
        data = json.loads(raw) if raw else {}
        raw_intents = data.get("intents", data.get("intent", []))  # acepta 'intent' o 'intents'
        if isinstance(raw_intents, str):
            intents = [raw_intents]
        elif isinstance(raw_intents, list):
            intents = raw_intents
        else:
            intents = []

        valid = {"recs", "razones"}
        seen = set()
        intents = [i for i in intents if i in valid and not (i in seen or seen.add(i))]
        if not intents:
            intents = ["recs"]

        client_ids = data.get("client_ids", [])
        if not isinstance(client_ids, list):
            client_ids = []

    except Exception:
        intents = ["recs"]
        client_ids = []

    return {"intents": intents, "client_ids": client_ids}

CLIENT_ID_PAT = re.compile(r"[0-9a-f]{16,}", re.I)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Alcance

# COMMAND ----------

def resolve_client_scope(
    text: str, 
    llm_out: dict, 
    df_tidy: pd.DataFrame, 
    df_portfolio: pd.DataFrame, 
    chat_id: int
):
    """
    Devuelve:
      - df_user: recomendaciones de la cartera del chat (o subset si se pidieron clientes)
      - used_client_ids: lista de client_ids usados (o None si toda la cartera)
      - aviso: mensaje si hubo clientes solicitados fuera de la cartera
    """
    # Normaliza
    try:
        chat_id = int(chat_id)
    except Exception:
        pass

    df_user = df_tidy.merge(
        df_portfolio[['client_id', 'chat_id']], 
        on='client_id', how='inner'
    )

    # Tipos/normalización robusta
    try:
        df_user['chat_id'] = df_user['chat_id'].astype('int64')
    except Exception:
        pass

    # Normaliza IDs (asegúrate de haber normalizado df_tidy/portfolio antes)
    df_user['client_id'] = df_user['client_id'].astype(str).str.strip().str.lower()
    cartera_df = df_user[df_user['chat_id'] == chat_id]
    if cartera_df.empty:
        return cartera_df, None, "No encuentro clientes asociados a tu usuario."

    cartera = set(cartera_df['client_id'])

    # Candidatos: regex + LLM (pueden ser varios)
    CLIENT_ID_PAT = re.compile(r"[0-9a-f]{16,}", re.I)
    found_regex = {m.group(0).strip().lower() for m in CLIENT_ID_PAT.finditer(text)}
    from_llm = {str(x).strip().lower() for x in llm_out.get('client_ids', [])}
    candidates = list(found_regex | from_llm)

    if not candidates:
        # Sin cliente → toda la cartera
        return cartera_df, None, None

    # Separa pedidas que están en cartera vs. fuera
    in_cartera = [cid for cid in candidates if cid in cartera]
    out_cartera = [cid for cid in candidates if cid not in cartera]

    if in_cartera:
        subset = cartera_df[cartera_df['client_id'].isin(in_cartera)].copy()
        aviso = None
        if out_cartera:
            aviso = ("Algunos clientes no están en tu portafolio: "
                     + ", ".join(out_cartera)
                     + ". Te muestro los que sí pertenecen a tu cartera.")
        return subset, in_cartera, aviso

    # Ninguno pertenece → fallback a toda cartera con aviso
    aviso = ("El/los clientes que proporcionas no se encuentran en tu portafolio; "
             "te comparto las recomendaciones para tus clientes.")
    return cartera_df, None, aviso


# COMMAND ----------

def format_recs_by_client(df_sel: pd.DataFrame) -> str:
    """
    Genera bloques: "Cliente <id>\n - product_x (p=0.83)\n ..."
    Asume df_sel solo tiene clientes de interés (ya filtrados).
    """
    parts = []
    for cid, g in df_sel.groupby('client_id'):
        parts.append(format_recs(g))
    return "\n\n".join(parts)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Respuesta

# COMMAND ----------

def _normalize_intents(llm_out: tp.Dict) -> tp.List[str]:
    raw = llm_out.get("intents", llm_out.get("intent", ["recs"]))
    if isinstance(raw, str):
        raw = [raw]
    valid = {"recs","razones"}
    seen, out = set(), []
    for it in raw or ["recs"]:
        if it in valid and it not in seen:
            out.append(it); seen.add(it)
    return out or ["recs"]

# COMMAND ----------

def format_general_reasons(df_sel: pd.DataFrame, general_ideas_by_pid: dict) -> str:
    if df_sel.empty:
        return "No tengo recomendaciones para tu cartera en este corte."
    df = df_sel.sort_values(["client_id","prob"], ascending=[True, False])
    lines = ["Oportunidades — Razones generales (muestra SHAP)"]
    # resumimos por producto, no por cliente
    for pid, grp in df.groupby("product_id"):
        pname = grp.iloc[0].get("name_product") or f"Producto {int(pid)}"
        lines.append(f"\nProducto: {pname}")
        ideas = general_ideas_by_pid.get(int(pid), ["• Sin evidencias suficientes en la muestra."])
        lines.extend(ideas)
    return "\n".join(lines)

# COMMAND ----------

def answer_free_text(
    text: str,
    chat_id: int,
    telegram_token: str,
    tidy_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    master_table: pd.DataFrame,
    reasons_lookup: tp.Dict = None,   # cache precalculado (tu 15%)
    llm_parse_intent_fn = None,
    product_names = {},
):
    # 1) intención
    if llm_parse_intent_fn is None:
        llm_out = {"intents": ["recs"], "client_ids": []}
        intents = ["recs"]
    else:
        llm_out = llm_parse_intent_fn(text)
        raw = llm_out.get("intents", llm_out.get("intent", ["recs"]))
        intents = [raw] if isinstance(raw, str) else (raw or ["recs"])
        intents = [i for i in intents if i in ("recs","razones")] or ["recs"]

    # 2) alcance: cliente vs cartera
    df_sel, used_cids, aviso = resolve_client_scope(text, llm_out, tidy_df, portfolio_df, chat_id)

    # 3) construir respuesta
    msg_parts = []
    if "recs" in intents:
        if used_cids is None:
            # toda la cartera
            msg_parts.append(format_recs(df_sel))
        else:
            # por cliente
            msg_parts.append(format_recs_by_client(df_sel))

    if "razones" in intents:
        if used_cids is None:
            # razones generales (cartera)
            msg_parts.append(format_general_reasons_with_llm(
                reasons_lookup, product_names or {}, k=3
            ))
        else:
            # razones por cada cliente solicitado
            msg_parts.append(format_personal_reasons_multi_clients(
                df_sel=df_sel,
                client_ids=used_cids,
                reasons_lookup=reasons_lookup,
                product_names=product_names or {},
                master_features=master_table,
                model_by_product=model_by_product,
                id_col="client_id",
                topk=2
            ))

    body = "\n\n".join([m for m in msg_parts if m]).strip()
    msg = f"{aviso}\n\n{body}" if aviso else body

    # 4) enviar a Telegram
    resp = requests.get(
        f"https://api.telegram.org/bot{telegram_token}/sendMessage",
        params={"chat_id": chat_id, "text": msg}
    )

    return {
        "ok": resp.ok,
        "status_code": resp.status_code,
        "intents": intents,
        "used_client_ids": used_cids,   # <— ahora es lista o None
        "aviso": aviso,
        "rows": len(df_sel),
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Razones

# COMMAND ----------

def _feature_names_from_transformer(
    transformer,
    X_tr,                      # matriz transformada (para validar dim)
    input_cols,                # columnas de master_features usadas como input
    sample_df: pd.DataFrame,   # una fila de master para 'transform' si hace falta
    fallback_prefix: str = "f"
):
    """
    Intenta obtener nombres de columnas salientes del transformer.

    Orden de intentos:
    1) get_feature_names_out(input_features=input_cols)
    2) get_feature_names_out() sin args
    3) transformer.set_output(transform="pandas").transform(sample_df) y tomar .columns
    4) fallback f0..f{n-1}
    """
    n_out = X_tr.shape[1]
    names = None

    # (1) Intento con input_features
    if hasattr(transformer, "get_feature_names_out"):
        try:
            names = list(transformer.get_feature_names_out(input_features=list(input_cols)))
        except TypeError:
            # algunos estimadores no aceptan input_features
            try:
                names = list(transformer.get_feature_names_out())
            except Exception:
                names = None
        except Exception:
            names = None

    # (2) set_output("pandas") y transformar una fila
    if (not names) or (len(names) != n_out):
        try:
            # algunos transformadores permiten set_output aun estando fit
            tr = transformer
            if hasattr(transformer, "set_output"):
                tr = transformer.set_output(transform="pandas")
            Xt_sample = tr.transform(sample_df.iloc[:1])  # 1 fila basta
            if hasattr(Xt_sample, "columns"):
                names2 = list(Xt_sample.columns)
                if len(names2) == n_out:
                    names = names2
        except Exception:
            pass

    # (3) fallback
    if (not names) or (len(names) != n_out):
        names = [f"{fallback_prefix}{i}" for i in range(n_out)]

    return names

def _ensure_2d_array(X):
    """Convierte sparse a denso si hace falta; asegura matriz 2D np.ndarray."""
    if hasattr(X, "toarray"):  # scipy sparse
        X = X.toarray()
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X

def _tree_shap_topk(model, X_df: pd.DataFrame, k: int = 2) -> tp.List[tp.List[str]]:
    """
    SHAP con TreeExplainer (RandomForest, LightGBM, XGBoost, CatBoost).
    Retorna lista de listas de razones (top-k) por fila.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)
    # Para binario, SHAP puede devolver lista por clase: tomamos clase positiva
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values  # (n, d)
    abs_sv = np.abs(sv)
    topk_idx = np.argsort(-abs_sv, axis=1)[:, :k]

    reasons_all = []
    feats = X_df.columns
    for i in range(X_df.shape[0]):
        row = []
        for j in topk_idx[i]:
            sign = "↑" if sv[i, j] > 0 else "↓"
            row.append(f"{sign} {feats[j]} = {X_df.iloc[i, j]}")
        reasons_all.append(row)
    return reasons_all


# COMMAND ----------

def build_reasons_lookup(
    tidy: pd.DataFrame,                 # pred==1; cols: client_id, product_id, prob, name_product, mon
    master_features: pd.DataFrame,      # master cruda; cols: client_id + features origen
    model_by_product: dict,             # por producto: { 'model', 'type', 'transformer' }
    id_col: str = "client_id",
    topk: int = 2,
    only_client_ids: tp.Optional[tp.Set[str]] = None,
    frac_clients: float = 1.0,          # ← porcentaje de clientes a explicar por producto (0<frac≤1)
    random_state: int = 42              # ← semilla para muestreo reproducible
) -> dict:
    """
    Calcula razones SHAP (top-k) por (client_id, product_id) usando el transformer y modelo
    específicos de CADA producto. Soporta muestreo por porcentaje para acelerar.

    Retorna: dict[(client_id, product_id)] = [razón1, razón2]
    """
    assert id_col in master_features.columns, f"master_features debe incluir '{id_col}'"
    assert 0 < frac_clients <= 1.0, "frac_clients debe estar en (0, 1]"

    # si queremos acotar a cierto set de clientes, filtramos desde ya
    if only_client_ids is not None:
        tidy = tidy[tidy[id_col].isin(only_client_ids)].copy()

    # index para joins rápidos
    mf = master_features.set_index(id_col)

    reasons_lookup = {}

    # por cada producto con modelo + transformer
    for pid, info in model_by_product.items():
        model = info.get("model")
        mtype = (info.get("type", "") or "").lower()
        transformer = info.get("transformer")

        if model is None or transformer is None:
            continue
        if mtype not in {"randomforest", "lightgbm", "xgboost", "catboost"}:
            # Mantengo el scope simple (árboles). Si necesitas genéricos, añade KernelExplainer aparte.
            continue

        # clientes con pred==1 para este producto
        sub = tidy.loc[tidy["product_id"] == pid, [id_col, "product_id"]].drop_duplicates()
        if sub.empty:
            continue

        # quedan solo los que sí están en la master
        cids_all = [cid for cid in sub[id_col].tolist() if cid in mf.index]
        if not cids_all:
            continue

        # === muestreo por porcentaje ===
        if frac_clients < 1.0:
            n_total = len(cids_all)
            n_take = max(1, int(np.floor(n_total * frac_clients)))
            rng = np.random.default_rng(random_state)
            idx = rng.choice(n_total, size=n_take, replace=False)
            cids = [cids_all[i] for i in idx]
        else:
            cids = cids_all

        # master filtrada y en el MISMO orden que cids
        mf_sub = mf.loc[cids].copy()

        # aplica transformer del producto
        X_tr = transformer.transform(mf_sub)
        feat_names = X_tr.columns
        X_tr = _ensure_2d_array(X_tr)
        
        # _feature_names_from_transformer(transformer, X_tr.shape[1])

        # DataFrame transformado (lo que “ve” el modelo)
        X_df = pd.DataFrame(X_tr, index=cids, columns=feat_names)

        # SHAP TreeExplainer (rápido para árboles)
        reasons_batch = _tree_shap_topk(model, X_df, k=topk)

        # volcar a dict
        for cid, rs in zip(cids, reasons_batch):
            reasons_lookup[(cid, pid)] = rs

    return reasons_lookup

# COMMAND ----------

reasons_lookup = build_reasons_lookup(
    tidy=df_tidy, 
    master_features=master_table, 
    model_by_product=model_by_product, 
    id_col='client_id',
    topk=2,
    frac_clients=0.05,
)

# COMMAND ----------

reasons_lookup

# COMMAND ----------

# guardando predicciones json
dbutils.fs.put(
    f"{path}reasons_recommendations.json",
    json.dumps({str(k): v for k, v in reasons_lookup.items()}, indent=2, sort_keys=True),
    overwrite=True
)

# COMMAND ----------

# leyendo las razones de las recomendaciones
path_reasons = f"{path}reasons_recommendations.json"
raw = dbutils.fs.head(path_reasons, 1000000)

reasons_lookup_saved = {eval(k): v for k, v in json.loads(raw).items()}

# COMMAND ----------

def _feat_names_from_transformer_or_df(X_tr, transformer, n_features_out: int, fallback_prefix="f"):
    """
    Devuelve nombres de columnas para X_tr.
    - Si X_tr ya es DataFrame, usa sus columnas.
    - Si el transformer tiene get_feature_names_out con largo correcto, úsalo.
    - Si no, usa f0..f{n-1}.
    """
    import numpy as np
    import pandas as pd

    if isinstance(X_tr, pd.DataFrame):
        cols = list(X_tr.columns)
        if len(cols) == n_features_out:
            return cols

    names = None
    if hasattr(transformer, "get_feature_names_out"):
        try:
            names = list(transformer.get_feature_names_out())
        except Exception:
            names = None
    if not names or len(names) != n_features_out:
        names = [f"{fallback_prefix}{i}" for i in range(n_features_out)]
    return names


def compute_personal_reasons_on_the_fly(
    client_id: str,
    product_id: int,
    master_features: pd.DataFrame,
    model_by_product: dict,
    id_col: str = "client_id",
    topk: int = 2,
):
    """
    Calcula razones SHAP (top-k) para un cliente/producto específico en el momento.
    Devuelve lista de strings como ["↑ feature = valor", ...] o None si no se puede.
    """
    import numpy as np
    import pandas as pd

    # normaliza client_id para evitar mismatches
    cid = str(client_id).strip().lower()
    assert id_col in master_features.columns, f"master_features debe incluir '{id_col}'"

    if product_id not in model_by_product:
        return None

    info = model_by_product[product_id]
    model = info.get("model")
    transformer = info.get("transformer")
    mtype = (info.get("type", "") or "").lower()

    if model is None or transformer is None:
        return None
    if mtype not in {"randomforest", "lightgbm", "xgboost", "catboost"}:
        # mantenemos simple: solo modelos tipo árbol
        return None

    # toma fila de features crudas
    mf = master_features.set_index(id_col)
    if cid not in mf.index:
        return None
    row = mf.loc[[cid]].copy()   # DataFrame 1xD

    # transforma con el transformer del producto
    X_tr = transformer.transform(row)

    # vuelve 2D numpy si es sparse
    if hasattr(X_tr, "toarray"):
        X_tr = X_tr.toarray()

    # si es Series, rehace a 2D
    import numpy as np
    if isinstance(X_tr, np.ndarray) and X_tr.ndim == 1:
        X_tr = X_tr.reshape(1, -1)

    # nombres de columnas post-transformación
    feat_names = _feat_names_from_transformer_or_df(X_tr, transformer, X_tr.shape[1])

    # arma DataFrame que "ve" el modelo
    X_df = pd.DataFrame(X_tr, index=[cid], columns=feat_names)

    # SHAP rápido para árboles
    rlist = _tree_shap_topk(model, X_df, k=topk)  # devuelve [[...]]
    return rlist[0] if rlist else None


# COMMAND ----------

# MAGIC %md
# MAGIC ### Mensajes personalizados

# COMMAND ----------

reasons_lookup_saved

# COMMAND ----------

# agrupa razones (↑/↓ feature) por producto para carta general
def aggregate_global_reasons(
    reasons_lookup: dict,            
    product_names: dict[int, str],   
) -> dict:
    agg = {pid: {"up": Counter(), "down": Counter(), "n": 0, "name": product_names.get(pid, f"product_{pid}")} 
           for pid in set(pid for _, pid in reasons_lookup.keys())}
    for (cid, pid), reasons in reasons_lookup.items():
        agg[pid]["n"] += 1
        for r in reasons:
            sign = r[0]
            try:
                feat = r[2:].split(" = ")[0].strip()
            except Exception:
                feat = r
            if sign == "↑":
                agg[pid]["up"][feat] += 1
            else:
                agg[pid]["down"][feat] += 1
    return agg

# para razones puntuales (cliente): devuelve lista corta "feat: valor (↑/↓)"
def pretty_personal_reasons(reason_list: tp.List[str]) -> tp.List[str]:
    out = []
    for r in reason_list:
        # "↑ ingreso = 1200.5"
        sign = "al alza" if r.startswith("↑") else "a la baja"
        body = r[1:].strip()
        out.append(f"{body} ({sign})")
    return out


# COMMAND ----------

def llm_general_reason_for_product(
    product_name: str,
    top_up: tp.List[tp.Tuple[str,int]],     
    top_down: tp.List[tp.Tuple[str,int]],  
    n_clients: int,
    max_feats: int = 3,
    model: str = "gpt-4o-mini"
) -> str:
    """
    Genera 1-2 frases explicando por qué se recomienda este producto en la cartera,
    usando las tendencias más frecuentes (↑/↓) entre clientes con pred==1.
    """
    # quedarnos con top-k
    top_up = top_up[:max_feats]
    top_down = top_down[:max_feats]

    # prompt compacto
    sys = (
        "Eres un asistente bancario conciso. Explica en 1-2 frases por qué se recomienda "
        "un producto basándote solo en los indicadores dados. No inventes nada. "
        "Cuando se trata de un indicador de entropía a la alza debes indicar que se debe a una alta variedad de (nombre relacionado a la feature). Por ejemplo si el indicador es ↑ trans_event_type_entropy_m, la razón es que se evidenció alta variedad en tipo de transacciones realizadas."
        "Habla en segunda persona plural ('tus clientes')."
    )
    up_txt = ", ".join([f"{f} (frec {c})" for f,c in top_up]) if top_up else "—"
    dn_txt = ", ".join([f"{f} (frec {c})" for f,c in top_down]) if top_down else "—"
    user = (
        f"Producto: {product_name}\n"
        f"Clientes con recomendación: {n_clients}\n"
        f"Tendencias al alza: {up_txt}\n"
        f"Tendencias a la baja: {dn_txt}\n"
        "Redacta 1–2 frases."
    )

    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.2, max_tokens=120,
            messages=[{"role":"system","content":sys},
                      {"role":"user","content":user}]
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # Fallback simple si el LLM falla
        if up_txt != "—":
            return (f"{product_name} se recomienda porque en tu cartera destacan al alza: "
                    f"{', '.join([f for f,_ in top_up])}.")
        return (f"{product_name} se recomienda por el patrón observado en tus clientes "
                f"(sin suficientes señales específicas).")


# COMMAND ----------

def llm_personal_reason_for_client(
    product_name: str,
    client_id: str,
    reasons_for_client: tp.List[str], 
    model: str = "gpt-4o-mini"
) -> str:
    sys = (
        "Eres un asistente bancario conciso. Explica en 1-2 frases por qué se recomienda "
        "un producto a un cliente específico, basándote SOLO en los indicadores dados. No inventes nada."
        "Cuando se trata de un indicador de entropía a la alza debes indicar que se debe a una alta variedad de (nombre relacionado a la feature). Por ejemplo si el indicador es ↑ trans_event_type_entropy_m, la razón es que se evidenció alta variedad en tipo de transacciones realizadas."
    )
    bullets = "\n".join(f"- {r}" for r in reasons_for_client)
    user = (
        f"Cliente: {client_id}\n"
        f"Producto: {product_name}\n"
        f"Indicadores clave:\n{bullets}\n"
        "Redacta 1–2 frases, en tono profesional y claro."
    )
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.2, max_tokens=120,
            messages=[{"role":"system","content":sys},
                      {"role":"user","content":user}]
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # Fallback por si falla
        flat = pretty_personal_reasons(reasons_for_client)
        return (f"Para este cliente se recomienda {product_name} por señales como: "
                f"{'; '.join(flat)}.")


# COMMAND ----------

def format_general_reasons_with_llm(
    reasons_lookup: dict,           
    product_names: dict[int, str],
    k: int = 3
) -> str:
    agg = aggregate_global_reasons(reasons_lookup, product_names)
    parts = []
    for pid, info in agg.items():
        up = info["up"].most_common(k)
        dn = info["down"].most_common(k)
        txt = llm_general_reason_for_product(
            product_name=info["name"],
            top_up=up,
            top_down=dn,
            n_clients=info["n"],
            max_feats=k
        )
        parts.append(f"{info['name']}: {txt}")
    return "Oportunidades — Razones generales\n\n" + "\n\n".join(parts)


# COMMAND ----------

reasons_lookup_saved[('3b7b86a2361704ba3baed9eccf51b129762817f8348ba4986e9c96af61e11773',
  1)]

# COMMAND ----------

def format_personal_reasons_with_llm(
    df_sel_cliente: pd.DataFrame,    
    reasons_lookup: dict,           
    product_names: dict[int, str],
    client_id: str,
    master_features: pd.DataFrame = None,          # <=== NUEVO (requerido p/ fallback)
    model_by_product: dict = None,                 # <=== NUEVO (requerido p/ fallback)
    id_col: str = "client_id",
    topk: int = 2,
    cache_fallback: bool = True                    # cachear resultados on-the-fly
) -> str:
    parts = []
    prods = df_sel_cliente["product_id"].unique().tolist()

    # normaliza el client_id para que coincida con tus keys del lookup
    cid_norm = str(client_id).strip().lower()

    for pid in prods:
        key = (cid_norm, pid)
        rlist = reasons_lookup.get(key)

        # Fallback: si no está en el lookup (porque era una muestra), calcula al vuelo
        if rlist is None and master_features is not None and model_by_product is not None:
            try:
                rlist = compute_personal_reasons_on_the_fly(
                    client_id=cid_norm,
                    product_id=int(pid),
                    master_features=master_features,
                    model_by_product=model_by_product,
                    id_col=id_col,
                    topk=topk,
                )
                if cache_fallback and rlist:
                    reasons_lookup[key] = rlist  # cachea para siguientes consultas
            except Exception:
                rlist = None

        if not rlist:
            # si aún no hay razones para ese producto, pasa al siguiente
            continue

        txt = llm_personal_reason_for_client(
            product_name=product_names.get(pid, f"product_{pid}"),
            client_id=cid_norm,
            reasons_for_client=rlist
        )
        parts.append(f"{product_names.get(pid, f'product_{pid}')}: {txt}")

    if not parts:
        return "No hay razones disponibles para este cliente en este corte."
    return "Razones — Cliente específico\n\n" + "\n\n".join(parts)


# COMMAND ----------

def format_personal_reasons_multi_clients(
    df_sel: pd.DataFrame,
    client_ids: list[str],
    reasons_lookup: dict,
    product_names: dict[int, str],
    master_features: pd.DataFrame,
    model_by_product: dict,
    id_col: str = "client_id",
    topk: int = 2
) -> str:
    blocks = []
    for cid in client_ids:
        df_c = df_sel[df_sel[id_col] == cid]
        if df_c.empty:
            continue
        txt = format_personal_reasons_with_llm(
            df_sel_cliente=df_c,
            reasons_lookup=reasons_lookup,
            product_names=product_names,
            client_id=cid,
            master_features=master_features,
            model_by_product=model_by_product,
            id_col=id_col,
            topk=topk,
            cache_fallback=True
        )
        blocks.append(f"Cliente {cid}\n\n{txt}")
    if not blocks:
        return "No hay razones disponibles para los clientes solicitados en este corte."
    return "\n\n".join(blocks)


# COMMAND ----------

display(portfolio)

# COMMAND ----------

# testeando bot
cid_test = 'c0ecc85f085ff4616ce6db55dafc9b160eaaa962f3c95514ad45cf00f7d15137'
product_names = {
    1: 'product_1',
    3: 'product_3',
    4: 'product_4',
}
tests = [
    f"Recomendaciones y razones de los cliente c0ecc85f085ff4616ce6db55dafc9b160eaaa962f3c95514ad45cf00f7d15137 y beb4a828cf2cc85b27c51af10a6c0167ddaa3ff3262660ba60ea49a1f4bfa57b",
]

for t in tests:
    out = answer_free_text(
        text=t,
        chat_id=CHAT_ID,
        telegram_token=TELEGRAM_TOKEN,
        tidy_df=df_tidy,
        portfolio_df=portfolio,
        reasons_lookup=reasons_lookup_saved,
        llm_parse_intent_fn=llm_parse_intent,
        product_names=product_names,
        master_table=master_table,
    )
    print(t, "→", out["status_code"], out.get("used_client_ids"))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Automatización

# COMMAND ----------

BASE = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
CHAT_WHITELIST = None # [int(CHAT_ID)]

def process_update(update: dict):
    """
    Lee el mensaje entrante, llama a tu pipeline y envía la respuesta.
    """
    msg = update.get("message") or {}
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    text = msg.get("text")

    # filtros básicos
    if not text or chat_id is None:
        return

    if CHAT_WHITELIST and chat_id not in CHAT_WHITELIST:
        requests.get(f"{BASE}/sendMessage", params={"chat_id": chat_id,
                                                    "text": "No estás autorizado para usar este bot."})
        return

    try:
        out = answer_free_text(
            text=text,
            chat_id=CHAT_ID,
            telegram_token=TELEGRAM_TOKEN,
            tidy_df=df_tidy,
            portfolio_df=portfolio,
            reasons_lookup=reasons_lookup_saved,
            llm_parse_intent_fn=llm_parse_intent,
            product_names=product_names,
            master_table=master_table,
          #  model_by_product=model_by_product,
        )

        print(f"[OK] {chat_id} → intents={out['intents']} used={out.get('used_client_ids') or out.get('used_client_id')}")
    except Exception as e:
        traceback.print_exc()
        
        # falla segura hacia el usuario
        requests.get(f"{BASE}/sendMessage", params={"chat_id": chat_id,
                                                    "text": "Ocurrió un error procesando tu solicitud."})


# COMMAND ----------

def run_long_polling(sleep_secs: float = 0.5):
    """
    Loop sencillo con offset para no procesar dos veces los mismos updates.
    """
    offset = None
    print("Escuchando mensajes de Telegram… (Ctrl+C para detener)")
    while True:
        try:
            r = requests.get(f"{BASE}/getUpdates", params={
                "timeout": 30,
                "offset": offset,
                "allowed_updates": ["message"]  # ignoramos stickers, etc.
            }, timeout=35)
            data = r.json()
            for upd in data.get("result", []):
                offset = upd["update_id"] + 1
                process_update(upd)
        except KeyboardInterrupt:
            print("Detenido por usuario.")
            break
        except Exception:
            traceback.print_exc()
            time.sleep(sleep_secs) 

# COMMAND ----------

# interactuando con el usuario
run_long_polling()