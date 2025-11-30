import os
import json
import requests
import streamlit as st
import numpy as np

# --- Config de API ---
# Prioridad: ENV var API_URL -> host.docker.internal (Docker) -> localhost (local)
API_URL = os.getenv("API_URL", "http://localhost:5000/predict")
if not API_URL:
    # En Docker Desktop (Windows/Mac) resuelve al host:
    API_URL = "http://host.docker.internal:5000/predict"
    # Si ejecutas Streamlit fuera de Docker, cambia a localhost:
    if os.getenv("RUN_LOCAL") == "1":
        API_URL = "http://localhost:5000/predict"

st.title("API de Predicción del Modelo Iris")
st.write("Ingresa las características de la flor de iris para obtener una predicción de su especie.")

# Sliders de entrada
sepal_length = st.slider('Longitud del sépalo (cm)', 0.0, 10.0, 5.0)
sepal_width  = st.slider('Ancho del sépalo (cm)',    0.0, 10.0, 3.0)
petal_length = st.slider('Longitud del pétalo (cm)', 0.0, 10.0, 4.0)
petal_width  = st.slider('Ancho del pétalo (cm)',    0.0, 10.0, 1.0)

species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

if st.button('Obtener Predicción'):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    payload = {'features': features}
    try:
        resp = requests.post(API_URL, data=json.dumps(payload), headers={'Content-Type': 'application/json'}, timeout=10)
        if resp.status_code == 200:
            pred = resp.json().get('prediction')
            st.success(f"La predicción es: **{species_map.get(pred, 'Desconocida')}** (clase={pred})")
        else:
            st.error(f"Error en la petición: {resp.status_code} – {resp.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"No se pudo conectar con la API en {API_URL}. Asegúrate de que está en ejecución.\n\nDetalle: {e}")
