import joblib
from flask import Flask, request, jsonify, Response
import numpy as np
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

# NUEVO: métrica de Prometheus
PREDICTION_COUNTER = Counter(
    "iris_prediction_count",
    "Contador de predicciones del modelo Iris por especie",
    ["species"],
)

# Cargar el modelo entrenado
try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    print("Error: 'model.pkl' no encontrado. Ejecuta el entrenamiento para generarlo.")
    model = None

app = Flask(__name__)


# NUEVO: endpoint de métricas para Prometheus / Managed Prometheus
@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return (
            jsonify({"error": "Modelo no cargado. Entrene el modelo primero."}),
            500,
        )
    try:
        data = request.get_json(force=True)
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        prediction_int = int(prediction[0])

        # Mapear el resultado numérico a una especie
        species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        predicted_species = species_map.get(prediction_int, "unknown")

        # NUEVO: incrementamos el contador para la especie predicha
        PREDICTION_COUNTER.labels(species=predicted_species).inc()

        # Devolvemos tanto el código numérico como la especie (tu Web puede seguir usando 'prediction')
        return jsonify({"prediction": prediction_int, "species": predicted_species})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/health")
def health():
    return jsonify(status="ok"), 200


@app.get("/ready")
def ready():
    # Aquí podrías comprobar que el modelo está cargado, etc.
    ready_flag = model is not None
    return jsonify(ready=ready_flag), 200


if __name__ == "__main__":
    print("Iniciando API en puerto 5000...")
    app.run(host="0.0.0.0", port=5000)
