import joblib
from flask import Flask, request, jsonify
import numpy as np

# Cargar el modelo entrenado
try:
    model = joblib.load('model.pkl')
except FileNotFoundError:
    print("Error: 'model.pkl' no encontrado. Ejecuta el entrenamiento para generarlo.")
    model = None

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo no cargado. Entrene el modelo primero.'}), 500
    try:
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
@app.get("/health")
def health():
    return jsonify(status="ok"), 200

@app.get("/ready")
def ready():
    # si quieres, aquí puedes chequear si el modelo está cargado, etc.
    return jsonify(ready=True), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)