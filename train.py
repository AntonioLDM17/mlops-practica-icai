import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import mlflow, mlflow.sklearn

# Fuerza backend headless en CI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar MLflow vía variable de entorno (sin dagshub.init)
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("iris-rf")

# Cargar CSV
try:
    iris = pd.read_csv("data/iris_dataset.csv")
except FileNotFoundError:
    raise SystemExit("Error: El archivo 'data/iris_dataset.csv' no fue encontrado.")

X = iris.drop("target", axis=1)
y = iris["target"]

with mlflow.start_run():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    n_estimators = 400
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Guardar y loguear modelo
    joblib.dump(model, "model.pkl")
    mlflow.sklearn.log_model(model, "random-forest-model")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)

    print(f"Modelo entrenado y precisión: {accuracy:.4f}")
    print("Experimento registrado con MLflow.")

    # --- Reporte para CML ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicciones")
    plt.ylabel("Valores Reales")
    plt.tight_layout()

    out_png = "confusion_matrix.png"
    plt.savefig(out_png)
    plt.close()
    print(f"Matriz de confusión guardada como '{out_png}'")

    # Sube el PNG como artifact (en una carpeta 'reports' dentro del run)
    mlflow.log_artifact(out_png, artifact_path="reports")

    # Útil para debug: ¿a dónde está subiendo MLflow?
    print("Artifact URI:", mlflow.get_artifact_uri())
    # --- Fin reporte ---
