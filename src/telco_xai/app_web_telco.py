import os
import requests
import streamlit as st
import pandas as pd
import altair as alt

# ================== CONFIG ==================

# URL de la API Telco (puedes cambiarla si corres en Docker/K8s)
API_URL = os.getenv("TELCO_API_URL", "http://localhost:5001")

PREDICT_ENDPOINT = f"{API_URL}/telco/predict"
EXPLAIN_ENDPOINT = f"{API_URL}/telco/explain"
GLOBAL_XAI_ENDPOINT = f"{API_URL}/telco/xai/global"
RETRAIN_ENDPOINT = f"{API_URL}/telco/retrain"
SHUFFLE_SANITY_ENDPOINT = f"{API_URL}/telco/sanity/shuffle_labels"

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


def call_retrain(drop_list):
    body = {"drop_features": drop_list}
    resp = requests.post(RETRAIN_ENDPOINT, json=body)
    resp.raise_for_status()
    return resp.json()

def call_shuffle_sanity():
    """Llama al sanity-check de etiquetas barajadas."""
    resp = requests.post(SHUFFLE_SANITY_ENDPOINT, json={})
    resp.raise_for_status()
    return resp.json()


# ================== UI STREAMLIT ==================

st.set_page_config(page_title="Telco Churn XAI", layout="wide")
st.title("📡 Telco Churn – Predicción y Explicabilidad")

st.sidebar.header("Configuración")
mode = st.sidebar.radio(
    "Modo",
    [
        "Predicción + explicación local",
        "Explicabilidad global",
        "Sanity-check: reentrenar sin algunas features",
        "Sanity-check: barajar etiquetas (modelo sin señal)",
    ],
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
                    )
                else:
                    st.info("No se recibieron valores SHAP en la respuesta de la API.")


# ============ MODO 2: EXPLICABILIDAD GLOBAL ============

elif mode == "Explicabilidad global":
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
                )
            else:
                st.info("No se recibió `shap_global_importance` de la API.")


# ============ MODO 3: SANITY-CHECK REENTRENAMIENTO ============

elif mode == "Sanity-check: reentrenar sin algunas features":
    st.markdown(
        """
    ### 🔍 Sanity check: ¿Qué pasa si elimino algunas features?

    Este experimento sirve para validar si las importancias globales son razonables.
    Podrás eliminar cualquier combinación de features *originales* y ver cómo cambia el rendimiento
    respecto al modelo baseline (todas las variables).
    """
    )
        # Guía de métricas para interpretar el sanity check
    with st.expander("ℹ️ Guía rápida: ¿qué significa cada métrica?"):
        st.markdown(
            """
            **Accuracy**  
            - Porcentaje total de aciertos (tanto churn=0 como churn=1).  
            - En datasets desbalanceados puede ser engañosa (un modelo que siempre predice *No churn* puede tener ~73% de accuracy).

            **Balanced accuracy**  
            - Media de la tasa de aciertos en cada clase:  
              \\(Balanced accuracy = (TPR_churn + TNR_no_churn) / 2\\).  
            - Vale 0.5 para un modelo “aleatorio” con dos clases, aunque el dataset esté desbalanceado.  
            - Si es cercana a 0.5 pero la accuracy es alta, es señal de que el modelo solo está aprendiendo bien la clase mayoritaria.

            **ROC AUC**  
            - Mide la capacidad del modelo para separar churn vs no churn para todos los posibles umbrales.  
            - 0.5 ≈ azar, 1.0 ≈ separación perfecta.  
            - Es relativamente robusta al desbalanceo, pero puede ocultar problemas si la clase minoritaria es muy pequeña.

            **AUC-PR (Área bajo la curva Precision–Recall)**  
            - Se centra en el rendimiento sobre la clase positiva (churn=1).  
            - Muy útil en datasets desbalanceados.  
            - Si el AUC-PR baja mucho al eliminar features, el modelo está perdiendo capacidad para detectar bien el churn.

            **Precision (precision_pos)**  
            - Entre todos los que el modelo predice como churn=1, ¿qué porcentaje lo es de verdad?  
            - Alta precision significa pocos falsos positivos.

            **Recall (recall_pos)**  
            - Entre todos los clientes que realmente hacen churn=1, ¿qué porcentaje detecta el modelo?  
            - Alta recall significa pocos falsos negativos.  
            - En muchos problemas de churn interesa no tener un recall demasiado bajo.

            **F1 (f1_pos)**  
            - Media armónica entre precision y recall para churn=1.  
            - Penaliza si una de las dos es muy baja.

            👉 **Interpretación en el sanity check**  
            - Si al eliminar features la accuracy sigue bien pero **balanced accuracy, AUC-PR o recall bajan mucho**, el modelo está colapsando hacia la clase mayoritaria.  
            - Si todas las métricas cambian muy poco (Δ pequeños), probablemente has quitado variables con poca importancia global.
            """
        )
    st.markdown("---")

    # --- Lista de features originales (las mismas columnas de X) ---
    all_features = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
        "TotalCharges",
    ]

    drop_list = st.multiselect(
        "Selecciona las features a eliminar del entrenamiento:",
        options=all_features,
        default=[],
    )

    if st.button("🚀 Reentrenar modelo con estas features eliminadas"):
        try:
            result = call_retrain(drop_list)
            st.success("Reentrenamiento completado.")

            removed = result.get("removed", [])
            baseline = result.get("baseline", {})
            retrained = result.get("retrained", {})

            # ================== BLOQUE 1: RESUMEN NUMÉRICO ==================
            st.write("## 1️⃣ Resumen numérico (baseline vs reducido)")

            metrics = [
                "accuracy",
                "balanced_accuracy",
                "roc_auc",
                "auc_pr",
                "precision_pos",
                "recall_pos",
                "f1_pos",
            ]
            rows = []
            for m in metrics:
                base_val = baseline.get(m, 0.0)
                red_val = retrained.get(m, 0.0)
                delta = red_val - base_val
                rows.append(
                    {
                        "metric": m,
                        "baseline": base_val,
                        "retrained": red_val,
                        "delta_retrained_minus_baseline": delta,
                    }
                )

            df_metrics = pd.DataFrame(rows).set_index("metric")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Modelo baseline (todas las features)**")
                st.metric("Accuracy baseline", f"{baseline.get('accuracy', 0)*100:.2f} %")
                st.metric("Balanced accuracy baseline", f"{baseline.get('balanced_accuracy', 0):.3f}")
                st.metric("ROC AUC baseline", f"{baseline.get('roc_auc', 0):.3f}")
                st.metric("AUC-PR baseline", f"{baseline.get('auc_pr', 0):.3f}")
                st.metric("Recall churn=1 baseline", f"{baseline.get('recall_pos', 0):.3f}")

            with col2:
                st.markdown("**Modelo reducido (sin algunas features)**")
                st.metric("Accuracy reducido", f"{retrained.get('accuracy', 0)*100:.2f} %")
                st.metric("Balanced accuracy reducido", f"{retrained.get('balanced_accuracy', 0):.3f}")
                st.metric("ROC AUC reducido", f"{retrained.get('roc_auc', 0):.3f}")
                st.metric("AUC-PR reducido", f"{retrained.get('auc_pr', 0):.3f}")
                st.metric("Recall churn=1 reducido", f"{retrained.get('recall_pos', 0):.3f}")
                st.write(f"**Features eliminadas:** {', '.join(removed) if removed else 'Ninguna'}")

            st.write("### Tabla de métricas y deltas")
            st.dataframe(
                df_metrics.style.format(
                    {
                        "baseline": "{:.4f}",
                        "retrained": "{:.4f}",
                        "delta_retrained_minus_baseline": "{:+.4f}",
                    }
                )
            )

            # >>> Warnings explicativos sobre colapso / desbalanceo / cambios fuertes <<<

            acc_base = baseline.get("accuracy", 0.0)
            acc_red = retrained.get("accuracy", 0.0)
            bal_base = baseline.get("balanced_accuracy", 0.0)
            bal_red = retrained.get("balanced_accuracy", 0.0)
            auc_base = baseline.get("roc_auc", 0.0)
            auc_red = retrained.get("roc_auc", 0.0)
            aucpr_base = baseline.get("auc_pr", 0.0)
            aucpr_red = retrained.get("auc_pr", 0.0)
            rec_base = baseline.get("recall_pos", 0.0)
            rec_red = retrained.get("recall_pos", 0.0)

            conf_r = retrained.get("confusion", {})
            tp_r = conf_r.get("tp", 0)
            fp_r = conf_r.get("fp", 0)
            fn_r = conf_r.get("fn", 0)
            tn_r = conf_r.get("tn", 0)

            # 0) Sin features eliminadas
            if len(removed) == 0:
                st.info(
                    "ℹ️ No has eliminado ninguna feature, por lo que los modelos "
                    "**baseline** y **reducido** deberían ser prácticamente idénticos."
                )

            # 1) Modelo colapsado: no detecta churn
            if tp_r == 0 and rec_red == 0:
                st.warning(
                    "⚠️ El modelo reducido **no detecta ningún cliente que hace churn** (TP=0, recall=0). "
                    "En la práctica se comporta como un modelo que siempre predice **'No churn'**. "
                    "Por eso la *accuracy* puede seguir siendo relativamente alta: el dataset está desbalanceado."
                )

            # 2) Accuracy alta pero balanced accuracy ≈ 0.5 y AUC-PR bajo
            if acc_red > 0.7 and bal_red < 0.55 and aucpr_red < max(1e-6, aucpr_base * 0.7):
                st.warning(
                    "⚠️ El modelo reducido mantiene una **accuracy relativamente alta**, "
                    "pero la **balanced accuracy** está cerca de 0.5 y el **AUC-PR** cae bastante. "
                    "Esto es típico de un modelo que acierta sobre todo la clase mayoritaria "
                    "y ha dejado de aprender bien la clase de churn."
                )

            # 3) Accuracy tipo 'always no churn'
            if rec_red == 0 and 0.70 <= acc_red <= 0.76:
                st.info(
                    "ℹ️ Una accuracy alrededor del **73%** es lo que obtendríamos con un modelo muy simple "
                    "que siempre predice **'No churn'**. Si tu modelo reducido está cerca de ese valor "
                    "y además tiene recall≈0, significa que ha perdido la capacidad de detectar churn."
                )

            # 4) Caídas importantes vs cambios pequeños
            delta_auc = auc_red - auc_base
            delta_rec = rec_red - rec_base

            if delta_auc < -0.05 or delta_rec < -0.10:
                st.error(
                    "⬇️ Se observa una **caída fuerte** en AUC y/o en el recall de la clase churn=1. "
                    "Esto indica que las features eliminadas eran **muy importantes** para el modelo."
                )
            elif abs(delta_auc) < 0.01 and abs(delta_rec) < 0.02 and len(removed) > 0:
                st.success(
                    "✅ El rendimiento apenas cambia (ΔAUC y ΔRecall pequeños). "
                    "Probablemente has eliminado variables con **importancia global baja**, "
                    "lo cual es coherente con las explicaciones globales."
                )

            # ================== BLOQUE 2: GRÁFICAS DE RENDIMIENTO ==================
            st.write("## 2️⃣ Gráficas de rendimiento")

            col_roc, col_conf = st.columns(2)

            # --- Curvas ROC ---
            with col_roc:
                st.markdown("### Curva ROC (baseline vs reducido)")

                roc_b = baseline.get("roc_curve", {})
                roc_r = retrained.get("roc_curve", {})

                df_roc_base = pd.DataFrame(
                    {
                        "fpr": roc_b.get("fpr", []),
                        "tpr": roc_b.get("tpr", []),
                        "model": "baseline",
                    }
                )
                df_roc_red = pd.DataFrame(
                    {
                        "fpr": roc_r.get("fpr", []),
                        "tpr": roc_r.get("tpr", []),
                        "model": "retrained",
                    }
                )

                df_roc_all = pd.concat([df_roc_base, df_roc_red], ignore_index=True)

                if not df_roc_all.empty:
                    chart_roc = (
                        alt.Chart(df_roc_all)
                        .mark_line()
                        .encode(
                            x=alt.X("fpr", title="False Positive Rate"),
                            y=alt.Y("tpr", title="True Positive Rate"),
                            color=alt.Color("model", title="Modelo"),
                        )
                        .properties(title="Curva ROC baseline vs reducido")
                    )
                    st.altair_chart(chart_roc, use_container_width=True)
                else:
                    st.info("No se pudieron construir las curvas ROC.")

            # --- Matriz de confusión / barras TP,FP,TN,FN ---
            with col_conf:
                st.markdown("### TP / FP / TN / FN")

                conf_b = baseline.get("confusion", {})
                conf_r = retrained.get("confusion", {})

                df_conf = pd.DataFrame(
                    {
                        "baseline": [
                            conf_b.get("tn", 0),
                            conf_b.get("fp", 0),
                            conf_b.get("fn", 0),
                            conf_b.get("tp", 0),
                        ],
                        "retrained": [
                            conf_r.get("tn", 0),
                            conf_r.get("fp", 0),
                            conf_r.get("fn", 0),
                            conf_r.get("tp", 0),
                        ],
                    },
                    index=["TN", "FP", "FN", "TP"],
                )

                st.bar_chart(df_conf)

            # ================== BLOQUE 3: IMPORTANCIA ELIMINADA VS CONSERVADA ==================
            st.write("## 3️⃣ Importancia de las variables eliminadas vs las que se mantienen")

            st.markdown(
                """
                Aquí comparamos la importancia global (Permutation Feature Importance) de:
                - Las **features que has eliminado del entrenamiento**.
                - Las **features más importantes que se han mantenido** en el modelo.

                Si eliminas variables muy poco importantes, la caída de rendimiento debería ser pequeña.
                Si eliminas variables muy importantes, deberías ver una caída clara en AUC / Recall.
                """
            )

            # Nota: aquí uso try/except sin 'else' para evitar el problema de sintaxis
            perm = {}
            try:
                global_xai = call_global_xai()
                perm = global_xai.get("permutation_importance", {})
            except Exception as e:
                st.error(f"No se pudo recuperar la importancia global de la API: {e}")

            if not perm:
                st.info("No hay información de Permutation Feature Importance disponible.")
            else:
                # Importancia de las features ELIMINADAS
                removed_imp = {
                    f: perm.get(f, 0.0)
                    for f in removed
                    if f in perm
                }

                # Importancia de las features que SE QUEDAN
                kept_imp = {
                    f: imp
                    for f, imp in perm.items()
                    if f not in removed
                }

                col_removed, col_kept = st.columns(2)

                with col_removed:
                    st.markdown("### 🔻 Features eliminadas")
                    if removed_imp:
                        df_removed = (
                            pd.DataFrame(
                                list(removed_imp.items()),
                                columns=["feature", "importance"],
                            )
                            .sort_values("importance", ascending=False)
                        )

                        st.dataframe(df_removed)
                        chart_removed = (
                            alt.Chart(df_removed)
                            .mark_bar()
                            .encode(
                                x=alt.X("importance:Q", title="Importancia (Permutation FI)"),
                                y=alt.Y("feature:N", sort="-x", title="Feature"),
                                tooltip=["feature", "importance"],
                            )
                            .properties(
                                title="Importancia de las features eliminadas",
                                height=300,
                            )
                        )
                        st.altair_chart(chart_removed, use_container_width=True)
                    else:
                        st.info("No has eliminado ninguna feature o ninguna aparece en las importancias globales.")

                with col_kept:
                    st.markdown("### ✅ Top features que se mantienen")
                    if kept_imp:
                        df_kept = (
                            pd.DataFrame(
                                list(kept_imp.items()),
                                columns=["feature", "importance"],
                            )
                            .sort_values("importance", ascending=False)
                            .head(10)  # Top 10 por claridad
                        )

                        st.dataframe(df_kept)
                        chart_kept = (
                            alt.Chart(df_kept)
                            .mark_bar()
                            .encode(
                                x=alt.X("importance:Q", title="Importancia (Permutation FI)"),
                                y=alt.Y("feature:N", sort="-x", title="Feature"),
                                tooltip=["feature", "importance"],
                            )
                            .properties(
                                title="Top 10 features que se mantienen",
                                height=300,
                            )
                        )
                        st.altair_chart(chart_kept, use_container_width=True)
                    else:
                        st.info("No quedan features con importancia calculada (caso raro).")

            with st.expander("🔎 Ver JSON completo devuelto por la API"):
                st.json(result)

        except Exception as e:
            st.error(f"Error llamando al retrain: {e}")

# ============ MODO 4: SANITY-CHECK LABELS BARAJADAS ============

elif mode == "Sanity-check: barajar etiquetas (modelo sin señal)":
    st.markdown(
        """
    ### 🧪 Sanity check: entrenar con etiquetas barajadas

    Este experimento responde a la pregunta:

    > *“Si rompo la relación entre X y y (churn), ¿el modelo se derrumba a azar?”*

    Entrenamos dos modelos con la misma receta que en producción:
    - **Baseline**: con las etiquetas reales.
    - **Shuffled**: con las mismas features pero con las etiquetas barajadas aleatoriamente.

    Si el modelo está aprendiendo señal real:
    - El baseline debería tener **ROC AUC** y **AUC-PR** bastante por encima de azar.
    - El modelo con etiquetas barajadas debería tener:
      - ROC AUC ≈ 0.5
      - AUC-PR ≈ prevalencia de churn
      - Balanced accuracy ≈ 0.5
    """
    )

    if st.button("🚨 Ejecutar sanity-check de etiquetas barajadas"):
        try:
            result = call_shuffle_sanity()
        except Exception as e:
            st.error(f"Error llamando al endpoint de sanity-check: {e}")
        else:
            prevalence = result.get("prevalence", 0.0)
            baseline = result.get("baseline", {})
            shuffled = result.get("shuffled", {})

            st.markdown(f"**Prevalencia de churn en el dataset:** `{prevalence*100:.2f} %`")

            metrics = [
                "accuracy",
                "balanced_accuracy",
                "roc_auc",
                "auc_pr",
                "precision_pos",
                "recall_pos",
                "f1_pos",
            ]

            rows = []
            for m in metrics:
                base_val = baseline.get(m, 0.0)
                sh_val = shuffled.get(m, 0.0)
                delta = sh_val - base_val
                rows.append(
                    {
                        "metric": m,
                        "baseline": base_val,
                        "shuffled": sh_val,
                        "delta_shuffled_minus_baseline": delta,
                    }
                )

            df_metrics = pd.DataFrame(rows).set_index("metric")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📈 Modelo baseline (etiquetas reales)")
                st.metric("Accuracy", f"{baseline.get('accuracy', 0)*100:.2f} %")
                st.metric("Balanced accuracy", f"{baseline.get('balanced_accuracy', 0):.3f}")
                st.metric("ROC AUC", f"{baseline.get('roc_auc', 0):.3f}")
                st.metric("AUC-PR", f"{baseline.get('auc_pr', 0):.3f}")
                st.metric("Recall churn=1", f"{baseline.get('recall_pos', 0):.3f}")

            with col2:
                st.markdown("#### 🎲 Modelo con etiquetas barajadas")
                st.metric("Accuracy", f"{shuffled.get('accuracy', 0)*100:.2f} %")
                st.metric("Balanced accuracy", f"{shuffled.get('balanced_accuracy', 0):.3f}")
                st.metric("ROC AUC", f"{shuffled.get('roc_auc', 0):.3f}")
                st.metric("AUC-PR", f"{shuffled.get('auc_pr', 0):.3f}")
                st.metric("Recall churn=1", f"{shuffled.get('recall_pos', 0):.3f}")

            st.write("### Tabla de métricas (baseline vs etiquetas barajadas)")
            st.dataframe(
                df_metrics.style.format(
                    {
                        "baseline": "{:.4f}",
                        "shuffled": "{:.4f}",
                        "delta_shuffled_minus_baseline": "{:+.4f}",
                    }
                )
            )

            # Comentarios interpretativos rápidos
            auc_base = baseline.get("roc_auc", 0.0)
            auc_sh = shuffled.get("roc_auc", 0.0)
            aucpr_base = baseline.get("auc_pr", 0.0)
            aucpr_sh = shuffled.get("auc_pr", 0.0)
            bal_base = baseline.get("balanced_accuracy", 0.0)
            bal_sh = shuffled.get("balanced_accuracy", 0.0)

            if auc_sh < 0.6 and bal_sh < 0.55 and abs(aucpr_sh - prevalence) < 0.05:
                st.success(
                    "✅ El modelo con etiquetas barajadas se comporta prácticamente como azar "
                    "(ROC AUC ≈ 0.5, balanced accuracy ≈ 0.5, AUC-PR ≈ prevalencia). "
                    "Esto indica que el modelo baseline **sí está capturando señal real**."
                )

            if auc_base - auc_sh < 0.05 and aucpr_base - aucpr_sh < 0.02:
                st.warning(
                    "⚠️ Las métricas del modelo baseline y del modelo con etiquetas barajadas "
                    "son muy parecidas. Esto sugiere que el modelo baseline **no está "
                    "aprovechando bien la señal del dataset**, o que la señal es muy débil."
                )
                
            # ====== 2️⃣ Curvas ROC y Precision–Recall ======
            st.write("## 2️⃣ Curvas ROC y Precision–Recall")

            col_roc, col_pr = st.columns(2)

            # --- Curvas ROC baseline vs shuffled ---
            with col_roc:
                st.markdown("### Curva ROC (baseline vs etiquetas barajadas)")

                roc_b = baseline.get("roc_curve", {})
                roc_s = shuffled.get("roc_curve", {})

                df_roc_base = pd.DataFrame(
                    {
                        "fpr": roc_b.get("fpr", []),
                        "tpr": roc_b.get("tpr", []),
                        "model": "baseline",
                    }
                )
                df_roc_shuff = pd.DataFrame(
                    {
                        "fpr": roc_s.get("fpr", []),
                        "tpr": roc_s.get("tpr", []),
                        "model": "shuffled",
                    }
                )

                df_roc_all = pd.concat([df_roc_base, df_roc_shuff], ignore_index=True)

                if not df_roc_all.empty:
                    chart_roc = (
                        alt.Chart(df_roc_all)
                        .mark_line()
                        .encode(
                            x=alt.X("fpr", title="False Positive Rate"),
                            y=alt.Y("tpr", title="True Positive Rate"),
                            color=alt.Color("model", title="Modelo"),
                        )
                        .properties(title="Curva ROC baseline vs etiquetas barajadas")
                    )
                    st.altair_chart(chart_roc, use_container_width=True)
                else:
                    st.info("No se pudieron construir las curvas ROC.")

            # --- Curvas Precision–Recall baseline vs shuffled ---
            with col_pr:
                st.markdown("### Curva Precision–Recall (baseline vs etiquetas barajadas)")

                pr_b = baseline.get("pr_curve", {})
                pr_s = shuffled.get("pr_curve", {})

                df_pr_base = pd.DataFrame(
                    {
                        "recall": pr_b.get("recall", []),
                        "precision": pr_b.get("precision", []),
                        "model": "baseline",
                    }
                )
                df_pr_shuff = pd.DataFrame(
                    {
                        "recall": pr_s.get("recall", []),
                        "precision": pr_s.get("precision", []),
                        "model": "shuffled",
                    }
                )

                df_pr_all = pd.concat([df_pr_base, df_pr_shuff], ignore_index=True)

                if not df_pr_all.empty:
                    chart_pr = (
                        alt.Chart(df_pr_all)
                        .mark_line()
                        .encode(
                            x=alt.X("recall", title="Recall"),
                            y=alt.Y("precision", title="Precision"),
                            color=alt.Color("model", title="Modelo"),
                        )
                        .properties(title="Curva Precision–Recall baseline vs etiquetas barajadas")
                    )
                    st.altair_chart(chart_pr, use_container_width=True)
                else:
                    st.info("No se pudieron construir las curvas Precision–Recall.")

            with st.expander("🔎 Ver JSON completo devuelto por la API"):
                st.json(result)
