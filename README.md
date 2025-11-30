# **MLOps + XAI – Iris + Telco Churn (XAI, CI/CD, GKE)**

Este repositorio implementa un sistema completo de **MLOps + Explicabilidad** con dos proyectos:

---

## **🌸 1. Iris – MLOps clásico**

* Entrenamiento con **RandomForest**
* Versionado con **DVC**
* Tracking con **MLflow**
* API Flask con métricas Prometheus
* Web Streamlit
* Monitorización con **Prometheus + Grafana** (local)
* Despliegue automático en **GKE**

---

## **📡 2. Telco Churn – Explicabilidad avanzada (XAI)**

Incluye:

### ✔ Explicabilidad Global

* **Permutation Feature Importance**
* **SHAP Global (post-one-hot)**

### ✔ Explicabilidad Local

* SHAP local para cada cliente

### ✔ Sanity Checks

* Reentrenamiento sin features seleccionadas
* Comparación estructural baseline vs reducido
* Entrenamiento con **etiquetas barajadas** (comprobación de señal)
* Tablas comparativas, curvas ROC & Precision–Recall

### ✔ Web XAI completa en Streamlit

Con los 4 modos:

* Predicción + explicación local
* Explicabilidad global
* Reentrenamiento sin atributos
* Sanity check de etiquetas barajadas

---

# 📁 **Estructura actual del repositorio**

```
mlops-practica-icai/
│
├── data/                    # Datos versionados por DVC (Iris + Telco)
│   ├── iris_dataset.csv
│   └── telco_churn.csv
│
├── src/
│   ├── iris/               # Código Iris (MLOps clásico)
│   │   ├── app.py
│   │   ├── app_web.py
│   │   └── train.py
│   │
│   └── telco_xai/          # Código Telco (Explicabilidad completa)
│       ├── app_telco.py
│       ├── app_web_telco.py
│       └── train_telco.py
│
├── scripts/
│   └── download_telco_data.py   # Descarga desde Kaggle (opcional)
│
├── notebooks/
│   └── telco_xai.ipynb     # Notebook completo de análisis XAI
│
├── monitoring/             # Prometheus + Grafana (local)
│   ├── prometheus.yaml
│   └── grafana-provisioning/
│
├── k8s/                    # Manifiestos Kubernetes (GKE)
│   ├── api-deployment.yaml
│   ├── api-service.yaml
│   ├── web-deployment.yaml
│   ├── web-service.yaml
│   ├── telco-api-deployment.yaml
│   ├── telco-api-service.yaml
│   ├── telco-web-deployment.yaml
│   ├── telco-web-service.yaml
│   └── pod-monitoring.yaml
│
├── Dockerfile              # Iris API
├── Dockerfile.web          # Iris Web
├── Dockerfile.telco        # Telco API
├── Dockerfile.web_telco    # Telco Web
│
├── docker-compose.yml      # Ejecución local completa
├── dvc.yaml                # Pipelines Iris + Telco
├── dvc.lock
├── requirements.txt
└── README.md
```

---

# 1. 🔧 Requisitos

Para ejecución local:

✔ Docker
✔ Docker Compose

> No necesitas instalar Python ni dependencias si usas `docker-compose`.

Para desarrollo:

✔ Python 3.11
✔ DVC
✔ MLflow

---

# 2. 📥 Clonar el repositorio

```bash
git clone https://github.com/AntonioLDM17/mlops-practica-icai.git
cd mlops-practica-icai
```

---

# 3. 📊 Dataset Telco (si no existe)

El dataset principal debe estar en:

```
data/telco_churn.csv
```

Si falta:

```bash
export PYTHONPATH=.
python -m scripts.download_telco_data
```

Después:

```bash
cp "<ruta_kaggle>/WA_Fn-UseC_-Telco-Customer-Churn.csv" data/telco_churn.csv
dvc add data/telco_churn.csv
dvc push
```

---

# 4. 🚀 Ejecución rápida con Docker Compose

```bash
docker-compose up --build
```

Se levantarán:

| Servicio          | Puerto | Descripción     |
| ----------------- | ------ | --------------- |
| `mlops-api`       | 5000   | API Iris        |
| `mlops-web`       | 8501   | Web Iris        |
| `mlops-telco-api` | 5001   | API Telco + XAI |
| `mlops-telco-web` | 8502   | Web Telco XAI   |
| `prometheus`      | 9090   | Métricas Iris   |
| `grafana`         | 3000   | Dashboards      |

---

## 4.1. 🌐 URLs locales

### Iris Web

👉 [http://localhost:8501](http://localhost:8501)

### Telco XAI Web

👉 [http://localhost:8502](http://localhost:8502)

### Prometheus

👉 [http://localhost:9090](http://localhost:9090)

### Grafana

👉 [http://localhost:3000](http://localhost:3000)
*(admin / admin)*

---

# 5. 🧪 Ejecución sin Docker (opcional)

Crear entorno:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ejecutar:

```bash
python src/iris/app.py
streamlit run src/iris/app_web.py --server.port 8501

python src/telco_xai/app_telco.py
streamlit run src/telco_xai/app_web_telco.py --server.port 8502
```

---

# 6. 📘 Notebook de explicabilidad

Ubicación:

```
notebooks/telco_xai.ipynb
```

Contiene:

✔ Entrenamiento completo
✔ Explicabilidad global (PFI + SHAP)
✔ Explicabilidad local
✔ Sanity checks
✔ Curvas ROC y PR-AUC
✔ Interpretación final

---

# 7. 🔄 Pipelines con DVC + MLflow

## Iris Pipeline

**Entrada:** `data/iris_dataset.csv`
**Salida:**

* `model.pkl`
* `confusion_matrix.png`
* `mlflow_metrics.json`

## Telco Churn Pipeline

**Salida:**

* `model_telco.pkl`
* `telco_metrics.json`
* `artifacts_telco/`

  * Background SHAP
  * Feature names
  * Permutation FI
  * SHAP Global Importance

Ejecutar:

```bash
dvc repro
```

---

# 8. ☁ Despliegue en Google Kubernetes Engine (GKE)

Los manifiestos están en:

```
k8s/
```

Despliegue manual:

```bash
kubectl apply -f k8s/
```

Incluye:

✔ API Iris
✔ Web Iris
✔ API Telco + XAI
✔ Web Telco
✔ PodMonitoring (Prometheus Iris)

Con GitHub Actions, el job `deploy-to-gke` hace todo automáticamente.

---

# 9. 🎯 Resumen rápido

### 1️⃣ Ejecutar todo en local:

```bash
docker-compose up --build
```

### 2️⃣ Acceder:

* Iris Web: [http://localhost:8501](http://localhost:8501)
* Telco XAI Web: [http://localhost:8502](http://localhost:8502)
* Prometheus: [http://localhost:9090](http://localhost:9090)
* Grafana: [http://localhost:3000](http://localhost:3000)

### 3️⃣ Reproducir pipelines:

```bash
dvc repro
```

### 4️⃣ Descargar dataset si falta:

```bash
export PYTHONPATH=.
python -m scripts.download_telco_data
```

