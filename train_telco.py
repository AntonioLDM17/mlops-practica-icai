import pandas as pd
import numpy as np
import joblib
import json
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ========== 1) CARGAR DATASET ===========
df = pd.read_csv("data/telco_churn.csv")

# Normalizar variables:
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

# Etiqueta
y = df["Churn"].replace({"Yes": 1, "No": 0})
X = df.drop(columns=["Churn", "customerID"])

# Separar columnas
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()


# ========== 2) TRAIN/TEST SPLIT ===========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ========== 3) PREPROCESADO ===========
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)


# ========== 4) BASELINE: LOGISTIC REGRESSION ===========
logreg = Pipeline(
    steps=[("pre", preprocessor),
           ("clf", LogisticRegression(max_iter=1000))]
)

with mlflow.start_run(run_name="telco_logreg"):
    logreg.fit(X_train, y_train)
    pred = logreg.predict(X_test)
    acc = accuracy_score(y_test, pred)

    mlflow.log_param("model", "logreg")
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(logreg, "model_logreg")


# ========== 5) MODELO PRINCIPAL: RANDOM FOREST ===========
rf = Pipeline(
    steps=[("pre", preprocessor),
           ("clf", RandomForestClassifier(random_state=42))]
)

param_grid = {
    "clf__n_estimators": [200, 300],
    "clf__max_depth": [None, 10],
    "clf__min_samples_leaf": [1, 2],
}

grid = GridSearchCV(
    rf,
    param_grid=param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
)

with mlflow.start_run(run_name="telco_rf"):
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", auc)

    # Guardar modelo de despliegue:
    joblib.dump(best_model, "model_telco.pkl")

    # Guardar métricas como JSON (DVC)
    with open("telco_metrics.json", "w") as f:
        json.dump({"accuracy": acc, "roc_auc": auc}, f)
