import json
import joblib
import numpy as np
import pandas as pd

import shap
from flask import Flask, request, jsonify

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    balanced_accuracy_score,
    average_precision_score,
)

# ================== CARGA DE MODELO Y ARTEFACTOS XAI ==================

# Modelo entrenado (Pipeline: preprocesador + RandomForest)
try:
    model_telco = joblib.load("model_telco.pkl")
except FileNotFoundError:
    print("ERROR: 'model_telco.pkl' no encontrado. Ejecuta dvc repro / train_telco.py")
    model_telco = None

# Componentes del pipeline
pre = model_telco.named_steps["pre"] if model_telco is not None else None
rf = model_telco.named_steps["clf"] if model_telco is not None else None

# Nombres de features originales (antes del preprocesado)
with open("artifacts_telco/telco_feature_names.json") as f:
    TELCO_FEATURE_NAMES = json.load(f)

# Background crudo para SHAP (los mismos features que TELCO_FEATURE_NAMES)
background_df = pd.read_csv("artifacts_telco/telco_background.csv")

# Transformamos el background con el preprocesador
background_trans = pre.transform(background_df)
if hasattr(background_trans, "toarray"):
    background_trans = background_trans.toarray()

# Nombres de features después del preprocesado (incluye one-hot de categóricas)
try:
    TRANSFORMED_FEATURE_NAMES = pre.get_feature_names_out().tolist()
except AttributeError:
    TRANSFORMED_FEATURE_NAMES = [f"feature_{i}" for i in range(background_trans.shape[1])]

# Cargamos importancias globales calculadas en entrenamiento
with open("artifacts_telco/telco_perm_importance.json") as f:
    PERM_IMPORTANCE = json.load(f)

with open("artifacts_telco/telco_shap_global.json") as f:
    SHAP_GLOBAL_IMPORTANCE = json.load(f)

# Creamos el explainer de SHAP sobre el RandomForest y el background transformado
explainer = shap.TreeExplainer(rf)

# ================== INICIALIZAR APP FLASK ==================

app = Flask(__name__)


@app.get("/telco/health")
def telco_health():
    """Healthcheck simple para la API Telco."""
    return jsonify(status="ok", model_loaded=model_telco is not None), 200


# ================== ENDPOINT: PREDICCIÓN ==================

@app.post("/telco/predict")
def telco_predict():
    """
    Espera un JSON con un diccionario de features Telco.

    Ejemplo de body:
    {
      "features": {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.5,
        "TotalCharges": 1000.0
      }
    }
    """
    if model_telco is None:
        return jsonify({"error": "Modelo Telco no cargado"}), 500

    try:
        data = request.get_json(force=True)
        features_dict = data["features"]

        # Aseguramos el orden de columnas usando TELCO_FEATURE_NAMES
        x_df = pd.DataFrame([features_dict], columns=TELCO_FEATURE_NAMES)

        proba = model_telco.predict_proba(x_df)[0, 1]
        pred = int(proba >= 0.5)

        return jsonify(
            {
                "churn_probability": float(proba),
                "churn_pred": pred,  # 1 = se va, 0 = se queda
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ================== ENDPOINT: EXPLICACIÓN LOCAL ==================

@app.post("/telco/explain")
def telco_explain():
    """
    Igual que /telco/predict pero devuelve explicaciones SHAP locales.

    Body:
    {
      "features": { ... mismo formato que /telco/predict ... }
    }

    Respuesta:
    {
      "churn_probability": 0.83,
      "churn_pred": 1,
      "shap_values": [
        {"feature": "cat__Contract_Month-to-month", "shap_value": 0.25},
        {"feature": "num__tenure", "shap_value": -0.10},
        ...
      ]
    }
    """
    if model_telco is None:
        return jsonify({"error": "Modelo Telco no cargado"}), 500

    try:
        data = request.get_json(force=True)
        features_dict = data["features"]

        # DataFrame crudo en orden
        x_df = pd.DataFrame([features_dict], columns=TELCO_FEATURE_NAMES)

        # Predicción
        proba = model_telco.predict_proba(x_df)[0, 1]
        pred = int(proba >= 0.5)

        # Transformación con el preprocesador
        x_trans = pre.transform(x_df)
        if hasattr(x_trans, "toarray"):
            x_trans = x_trans.toarray()

        # Valores SHAP
        shap_values = explainer(x_trans)

        vals = shap_values.values
        # Para binaria, TreeExplainer suele devolver (n_samples, n_features)
        # o (n_samples, n_features, n_classes). Lo manejamos:
        if vals.ndim == 3:
            # Nos quedamos con la contribución de la clase "churn=1"
            contrib = vals[0, :, 1]
        else:
            contrib = vals[0, :]

        shap_response = [
            {
                "feature": TRANSFORMED_FEATURE_NAMES[i],
                "shap_value": float(contrib[i]),
            }
            for i in range(len(TRANSFORMED_FEATURE_NAMES))
        ]

        return jsonify(
            {
                "churn_probability": float(proba),
                "churn_pred": pred,
                "shap_values": shap_response,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ================== ENDPOINT: EXPLICABILIDAD GLOBAL ==================

@app.get("/telco/xai/global")
def telco_xai_global():
    """
    Devuelve las métricas de explicabilidad global:
    - Permutation feature importance (en el espacio original de features)
    - SHAP global (en el espacio transformado: num__..., cat__...)
    """
    return jsonify(
        {
            "permutation_importance": PERM_IMPORTANCE,
            "shap_global_importance": SHAP_GLOBAL_IMPORTANCE,
        }
    )


# ================== ENDPOINT: REENTRENAMIENTO RÁPIDO ==================
@app.post("/telco/retrain")
def telco_retrain():
    """
    Reentrena un modelo rápido eliminando ciertas features y compara
    con un modelo baseline que usa todas las variables.
    Ahora:
      - Baseline y reducido se entrenan con la MISMA receta que train_telco.py:
        * Pipeline: preprocesado (StandardScaler + OneHotEncoder) + RandomForest
        * class_weight="balanced"
        * GridSearchCV con el mismo param_grid
        * scoring="average_precision"
    """
    try:
        data = request.get_json(force=True)
        drop_features = data.get("drop_features", [])

        # ---------- 1) Cargar dataset completo ----------
        df = pd.read_csv("data/telco_churn.csv")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df = df.dropna(subset=["TotalCharges"])

        y = df["Churn"].replace({"Yes": 1, "No": 0})
        X_full = df.drop(columns=["Churn", "customerID"])

        n_features_original = X_full.shape[1]

        # ---------- 2) Split común para baseline y reducido ----------
        # Mismas condiciones que en train_telco.py
        X_train_full, X_test_full, y_train, y_test = train_test_split(
            X_full,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        # Param grid y receta de RF idénticos a train_telco.py
        param_grid = {
            "clf__n_estimators": [200, 300],
            "clf__max_depth": [None, 10],
            "clf__min_samples_leaf": [1, 2],
        }

        # ---------- 3) Modelo BASELINE (todas las features) ----------
        cat_full = X_full.select_dtypes(include=["object"]).columns.tolist()
        num_full = X_full.select_dtypes(exclude=["object"]).columns.tolist()

        pre_full = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_full),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_full),
            ]
        )

        rf_full = RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
        )

        pipe_full = Pipeline(
            steps=[
                ("pre", pre_full),
                ("clf", rf_full),
            ]
        )

        grid_full = GridSearchCV(
            pipe_full,
            param_grid=param_grid,
            cv=3,
            scoring="average_precision",
            n_jobs=-1,
        )

        grid_full.fit(X_train_full, y_train)
        baseline_model = grid_full.best_estimator_

        y_pred_base = baseline_model.predict(X_test_full)
        y_proba_base = baseline_model.predict_proba(X_test_full)[:, 1]

        acc_base = accuracy_score(y_test, y_pred_base)
        auc_base = roc_auc_score(y_test, y_proba_base)
        prec_base = precision_score(y_test, y_pred_base, pos_label=1, zero_division=0)
        rec_base = recall_score(y_test, y_pred_base, pos_label=1, zero_division=0)
        f1_base = f1_score(y_test, y_pred_base, pos_label=1, zero_division=0)
        bal_acc_base = balanced_accuracy_score(y_test, y_pred_base)
        auc_pr_base = average_precision_score(y_test, y_proba_base)

        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, y_pred_base).ravel()
        fpr_b, tpr_b, _ = roc_curve(y_test, y_proba_base)

        baseline_stats = {
            "accuracy": float(acc_base),
            "balanced_accuracy": float(bal_acc_base),
            "roc_auc": float(auc_base),
            "auc_pr": float(auc_pr_base),
            "precision_pos": float(prec_base),
            "recall_pos": float(rec_base),
            "f1_pos": float(f1_base),
            "confusion": {
                "tn": int(tn_b),
                "fp": int(fp_b),
                "fn": int(fn_b),
                "tp": int(tp_b),
            },
            "roc_curve": {
                "fpr": fpr_b.tolist(),
                "tpr": tpr_b.tolist(),
            },
        }

        # ---------- 4) Modelo REDUCIDO (sin algunas features) ----------
        X_reduced = X_full.copy()
        for f in drop_features:
            if f in X_reduced.columns:
                X_reduced = X_reduced.drop(columns=[f])

        n_features_final = X_reduced.shape[1]

        cat_red = X_reduced.select_dtypes(include=["object"]).columns.tolist()
        num_red = X_reduced.select_dtypes(exclude=["object"]).columns.tolist()

        pre_red = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_red),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_red),
            ]
        )

        rf_red = RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
        )

        pipe_red = Pipeline(
            steps=[
                ("pre", pre_red),
                ("clf", rf_red),
            ]
        )

        # Reusamos el MISMO split (mismas filas) para el reducido
        X_train_red, X_test_red, _, _ = train_test_split(
            X_reduced,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        grid_red = GridSearchCV(
            pipe_red,
            param_grid=param_grid,
            cv=3,
            scoring="average_precision",
            n_jobs=-1,
        )

        grid_red.fit(X_train_red, y_train)
        reduced_model = grid_red.best_estimator_

        y_pred_red = reduced_model.predict(X_test_red)
        y_proba_red = reduced_model.predict_proba(X_test_red)[:, 1]

        acc_red = accuracy_score(y_test, y_pred_red)
        auc_red = roc_auc_score(y_test, y_proba_red)
        prec_red = precision_score(y_test, y_pred_red, pos_label=1, zero_division=0)
        rec_red = recall_score(y_test, y_pred_red, pos_label=1, zero_division=0)
        f1_red = f1_score(y_test, y_pred_red, pos_label=1, zero_division=0)
        bal_acc_red = balanced_accuracy_score(y_test, y_pred_red)
        auc_pr_red = average_precision_score(y_test, y_proba_red)

        tn_r, fp_r, fn_r, tp_r = confusion_matrix(y_test, y_pred_red).ravel()
        fpr_r, tpr_r, _ = roc_curve(y_test, y_proba_red)

        retrained_stats = {
            "accuracy": float(acc_red),
            "balanced_accuracy": float(bal_acc_red),
            "roc_auc": float(auc_red),
            "auc_pr": float(auc_pr_red),
            "precision_pos": float(prec_red),
            "recall_pos": float(rec_red),
            "f1_pos": float(f1_red),
            "confusion": {
                "tn": int(tn_r),
                "fp": int(fp_r),
                "fn": int(fn_r),
                "tp": int(tp_r),
            },
            "roc_curve": {
                "fpr": fpr_r.tolist(),
                "tpr": tpr_r.tolist(),
            },
        }

        # ---------- 5) Impacto XAI: suma de importancia global eliminada ----------
        importance_removed_sum = float(
            sum(PERM_IMPORTANCE.get(f, 0.0) for f in drop_features)
        )

        return jsonify(
            {
                "removed": drop_features,
                "n_features_original": int(n_features_original),
                "n_features_final": int(n_features_final),
                "importance_removed_sum": importance_removed_sum,
                "baseline": baseline_stats,
                "retrained": retrained_stats,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    print("Iniciando API Telco en puerto 5001...")
    app.run(host="0.0.0.0", port=5001)