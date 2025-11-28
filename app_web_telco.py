import os
import requests
import streamlit as st
import pandas as pd

# ================== CONFIG ==================

# URL de la API Telco (puedes cambiarla si corres en Docker/K8s)
API_URL = os.getenv("TELCO_API_URL", "http://localhost:5001")

PREDICT_ENDPOINT = f"{API_URL}/telco/predict"
EXPLAIN_ENDPOINT = f"{API_URL}/telco/explain"
GLOBAL_XAI_ENDPOINT = f"{API_URL}/telco/xai/global"


# ================== OPCIONES TELCO ==================

CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

NUMERIC_DEFAULTS = {
    "SeniorCitizen": 0,
    "tenure": 12,
    "MonthlyCharges": 70.0,
    "TotalCharges": 1000.0,
}


# ================== HELPERS ==================


def build_features_dict():
    """Construye el diccionario de features a partir de los widgets de Streamlit."""
    st.subheader("Datos del cliente")

    # Categóricas
    gender = st.selectbox("Género", CATEGORICAL_OPTIONS["gender"])
    partner = st.selectbox("¿Tiene pareja?", CATEGORICAL_OPTIONS["Partner"])
    dependents = st.selectbox("¿Tiene dependientes?", CATEGORICAL_OPTIONS["Dependents"])
    phone_service = st.selectbox("Servicio de teléfono", CATEGORICAL_OPTIONS["PhoneService"])
    multiple_lines = st.selectbox("Líneas múltiples", CATEGORICAL_OPTIONS["MultipleLines"])
    internet_service = st.selectbox("Servicio de internet", CATEGORICAL_OPTIONS["InternetService"])
    online_security = st.selectbox("Seguridad online", CATEGORICAL_OPTIONS["OnlineSecurity"])
    online_backup = st.selectbox("Copia de seguridad online", CATEGORICAL_OPTIONS["OnlineBackup"])
    device_protection = st.selectbox("Protección de dispositivo", CATEGORICAL_OPTIONS["DeviceProtection"])
    tech_support = st.selectbox("Soporte técnico", CATEGORICAL_OPTIONS["TechSupport"])
    streaming_tv = st.selectbox("Streaming TV", CATEGORICAL_OPTIONS["StreamingTV"])
    streaming_movies = st.selectbox("Streaming de películas", CATEGORICAL_OPTIONS["StreamingMovies"])
    contract = st.selectbox("Tipo de contrato", CATEGORICAL_OPTIONS["Contract"])
    paperless_billing = st.selectbox("Factura electrónica", CATEGORICAL_OPTIONS["PaperlessBilling"])
    payment_method = st.selectbox("Método de pago", CATEGORICAL_OPTIONS["PaymentMethod"])

    # Numéricas
    senior_citizen = st.selectbox("¿Es senior (>=65)?", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
    tenure = st.slider("Meses de antigüedad (tenure)", min_value=0, max_value=72, value=NUMERIC_DEFAULTS["tenure"])
    monthly_charges = st.number_input(
        "Cuota mensual (MonthlyCharges)", min_value=0.0, max_value=300.0, value=NUMERIC_DEFAULTS["MonthlyCharges"]
    )
    total_charges = st.number_input(
        "Total facturado (TotalCharges)", min_value=0.0, max_value=20000.0, value=NUMERIC_DEFAULTS["TotalCharges"]
    )

    features = {
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": float(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    return features


def call_explain(features: dict):
    """Llama a /telco/explain y devuelve el JSON o un error."""
    body = {"features": features}
    resp = requests.post(EXPLAIN_ENDPOINT, json=body)
    resp.raise_for_status()
    return resp.json()


def call_global_xai():
    resp = requests.get(GLOBAL_XAI_ENDPOINT)
    resp.raise_for_status()
    return resp.json()


# ================== UI STREAMLIT ==================

st.set_page_config(page_title="Telco Churn XAI", layout="wide")
st.title("📡 Telco Churn – Predicción y Explicabilidad")

st.sidebar.header("Configuración")
mode = st.sidebar.radio(
    "Modo",
    ["Predicción + explicación local", "Explicabilidad global"],
)

st.sidebar.markdown(f"**API URL:** `{API_URL}`")

# ============ MODO 1: PREDICCIÓN + EXPLICACIÓN LOCAL ============

if mode == "Predicción + explicación local":
    st.markdown(
        """
        En esta sección puedes introducir los datos de un cliente y obtener:
        - La **probabilidad de churn** (que se marche).
        - La **predicción binaria** (1 = se va, 0 = se queda).
        - Una **explicación local** con SHAP que muestra qué variables empujan la predicción hacia irse o quedarse.
        """
    )

    with st.form("telco_form"):
        features = build_features_dict()
        submitted = st.form_submit_button("🔮 Predecir y explicar")

    if submitted:
        try:
            result = call_explain(features)
        except Exception as e:
            st.error(f"Error llamando a la API: {e}")
        else:
            proba = result.get("churn_probability", None)
            pred = result.get("churn_pred", None)
            shap_values = result.get("shap_values", [])

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Resultado de la predicción")
                if proba is not None:
                    st.metric(
                        "Probabilidad de churn",
                        f"{proba*100:.1f} %",
                        help="Probabilidad estimada de que el cliente se dé de baja",
                    )
                if pred is not None:
                    label = "Se va (churn=1)" if pred == 1 else "Se queda (churn=0)"
                    st.write(f"**Predicción:** {label}")

                st.write("### Datos de entrada")
                st.json(features)

            with col2:
                st.subheader("Explicación local (SHAP)")

                if shap_values:
                    df_shap = pd.DataFrame(shap_values)
                    # Añadimos columna con el valor absoluto para ordenar
                    df_shap["abs_value"] = df_shap["shap_value"].abs()
                    df_top = df_shap.sort_values("abs_value", ascending=False).head(10)

                    st.write("Mostrando las **10 variables** con mayor impacto (|SHAP|):")
                    st.dataframe(df_top[["feature", "shap_value"]])

                    st.bar_chart(
                        df_top.set_index("feature")["shap_value"],
                        use_container_width=True,
                    )
                else:
                    st.info("No se recibieron valores SHAP en la respuesta de la API.")


# ============ MODO 2: EXPLICABILIDAD GLOBAL ============

else:
    st.markdown(
        """
        En esta sección puedes explorar la **explicabilidad global** del modelo de churn:
        - Importancia global por **Permutation Feature Importance** en el espacio original.
        - Importancia global por **SHAP** en el espacio transformado (features numéricas y one-hot).
        """
    )

    try:
        global_xai = call_global_xai()
    except Exception as e:
        st.error(f"Error llamando a la API: {e}")
    else:
        perm = global_xai.get("permutation_importance", {})
        shap_global = global_xai.get("shap_global_importance", {})

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Permutation Feature Importance (features originales)")
            if perm:
                df_perm = (
                    pd.DataFrame(list(perm.items()), columns=["feature", "importance"])
                    .sort_values("importance", ascending=False)
                )
                st.dataframe(df_perm)
                st.bar_chart(
                    df_perm.set_index("feature")["importance"],
                    use_container_width=True,
                )
            else:
                st.info("No se recibió `permutation_importance` de la API.")

        with col2:
            st.subheader("SHAP Global (features transformadas)")
            if shap_global:
                df_shap_g = (
                    pd.DataFrame(list(shap_global.items()), columns=["feature", "importance"])
                    .sort_values("importance", ascending=False)
                )
                st.dataframe(df_shap_g.head(25))  # primeras 25 por claridad
                st.bar_chart(
                    df_shap_g.set_index("feature")["importance"].head(25),
                    use_container_width=True,
                )
            else:
                st.info("No se recibió `shap_global_importance` de la API.")
