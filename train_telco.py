import os
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

import shap


RANDOM_STATE = 42

# ========= 1) CARGAR DATASET ===============
df = pd.read_csv("data/telco_churn.csv")

# Limpiar y preparar columnas numéricas
df["TotalCharges" ] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

# Etiqueta binaria
y = df["Churn"].replace({"Yes": 1, "No": 0})
X = df.drop(columns=["Churn", "customerID"])

# Columnas numéricas y categóricas
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

feature_names = list(X.columns)

# ========= 2) TRAIN/TEST SPLIT =============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ========= 3) PREPROCESADO =================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ========= 4) BASELINE: LOGISTIC REGRESSION =========
logreg_pipeline = Pipeline(
    steps=[
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000)),
    ]
)

with mlflow.start_run(run_name="telco_logreg"):
    logreg_pipeline.fit(X_train, y_train)
    y_pred = logreg_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_param("model_type", "logreg")
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(logreg_pipeline, "model")


# ========= 5) MODELO PRINCIPAL: RANDOM FOREST =========
rf_pipeline = Pipeline(
    steps=[
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
    ]
)

param_grid = {
    "clf__n_estimators": [200, 300],
    "clf__max_depth": [None, 10],
    "clf__min_samples_leaf": [1, 2],
}

grid = GridSearchCV(
    rf_pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
)

with mlflow.start_run(run_name="telco_rf") as run:
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_  # Pipeline(pre, RF)
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    mlflow.log_param("model_type", "random_forest")
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", auc)

    # Guardar modelo de despliegue
    joblib.dump(best_model, "model_telco.pkl")

    # Guardar métricas como JSON (para DVC)
    with open("telco_metrics.json", "w") as f:
        json.dump({"accuracy": acc, "roc_auc": auc}, f)


# ========= 6) DIRECTORIO DE ARTEFACTOS XAI =========
os.makedirs("artifacts_telco", exist_ok=True)

# Guardar nombres de features originales
with open("artifacts_telco/telco_feature_names.json", "w") as f:
    json.dump(feature_names, f)

# ========= 7) BACKGROUND PARA SHAP =========
# Muestra pequeña de X_train para usar como background
background = X_train.sample(
    n=min(200, len(X_train)),
    random_state=RANDOM_STATE
)

# Guardamos el background en disco (valores + nombres de columnas)
background.to_csv("artifacts_telco/telco_background.csv", index=False)

# ========= 8) PERMUTATION FEATURE IMPORTANCE (GLOBAL) =========
perm_result = permutation_importance(
    best_model,
    X_test,
    y_test,
    n_repeats=20,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

perm_importance = {
    feature_names[i]: float(perm_result.importances_mean[i])
    for i in range(len(feature_names))
}

with open("artifacts_telco/telco_perm_importance.json", "w") as f:
    json.dump(perm_importance, f)

# ========= 9) SHAP GLOBAL (USANDO EL RANDOM FOREST DIRECTO) =========
# Usamos TreeExplainer sobre el RandomForest ya entrenado,
# y pasamos los datos YA TRANSFORMADOS por el preprocesador.

pre = best_model.named_steps["pre"]
rf = best_model.named_steps["clf"]

# Subconjunto de test para SHAP (en crudo)
X_test_sample = X_test.sample(
    n=min(500, len(X_test)),
    random_state=RANDOM_STATE
)

# Transformamos background y test sample con el preprocesador
background_trans = pre.transform(background)
X_test_sample_trans = pre.transform(X_test_sample)

# Si son matrices dispersas (sparse), las convertimos a densas
if hasattr(background_trans, "toarray"):
    background_trans = background_trans.toarray()
if hasattr(X_test_sample_trans, "toarray"):
    X_test_sample_trans = X_test_sample_trans.toarray()

# Feature names después del preprocesado (incluye one-hot de categóricas)
try:
    transformed_feature_names = pre.get_feature_names_out()
except AttributeError:
    # Por si hubiese una versión muy vieja de sklearn (no es tu caso),
    # usamos nombres genéricos.
    transformed_feature_names = [f"feature_{i}" for i in range(X_test_sample_trans.shape[1])]

# Creamos el explainer para el RandomForest
explainer = shap.TreeExplainer(rf)

# Calculamos valores SHAP sobre el test transformado
shap_values = explainer(X_test_sample_trans)

# shap_values.values puede ser 2D (n_samples, n_features)
# o 3D (n_samples, n_features, n_outputs). Lo manejamos genéricamente.
vals = shap_values.values
if vals.ndim == 3:
    # Promediar sobre muestras y salidas
    mean_abs_shap = np.mean(np.abs(vals), axis=(0, 2))
else:
    # Promediar solo sobre muestras
    mean_abs_shap = np.mean(np.abs(vals), axis=0)

global_shap_importance = {
    transformed_feature_names[i]: float(mean_abs_shap[i])
    for i in range(len(transformed_feature_names))
}

with open("artifacts_telco/telco_shap_global.json", "w") as f:
    json.dump(global_shap_importance, f)

print("Entrenamiento Telco + artefactos XAI completados.")
