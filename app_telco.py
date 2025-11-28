import json
import joblib
import numpy as np
import pandas as pd

import shap
from flask import Flask, request, jsonify

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


if __name__ == "__main__":
    print("Iniciando API Telco en puerto 5001...")
    app.run(host="0.0.0.0", port=5001)
