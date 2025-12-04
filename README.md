# **MLOps + XAI – Iris + Telco Churn (XAI, CI/CD, GKE)**

This repository implements a complete **MLOps + Explainability** system with two projects:

---

## **🌸 1. Iris – Classic Mlops**

* Trained with **RandomForest**
* Versioned with **DVC**
* Tracked with **MLflow**
* Flask API with Prometheus metrics
* Streamlit Web
* Monitoring with **Prometheus + Grafana** (local)
* Automatic deployment on **GKE**

---

## **📡 2. Telco Churn – Advanced Explainability (XAI)**

Includes:

### ✔ Global Explainability
* **Permutation Feature Importance**
* **SHAP Global (post-one-hot)**

### ✔ Local Explainability
* SHAP local for each client

### ✔ Sanity Checks

* Retraining without selected features
* Structural comparison baseline vs reduced
* Training with **shuffled labels** (signal check)
* Comparative tables, ROC & Precision–Recall curves

### ✔ Complete XAI Web in Streamlit

With the 4 modes:

* Prediction + local explanation
* Global explainability
* Retraining without selected features
* Sanity check with shuffled labels

---

# 📁 **Current repository structure**

```
mlops-practica-icai/
│
├── data/                    # Data versioned by DVC (Iris + Telco)
│   ├── iris_dataset.csv
│   └── telco_churn.csv
│
├── src/
│   ├── iris/               # Iris code (Classic MLOps)
│   │   ├── app.py
│   │   ├── app_web.py
│   │   └── train.py
│   │
│   └── telco_xai/          # Telco code (Complete Explainability)
│       ├── app_telco.py
│       ├── app_web_telco.py
│       └── train_telco.py
│
├── scripts/
│   └── download_telco_data.py   # Download from Kaggle (optional)
│
├── notebooks/
│   └── telco_xai.ipynb     # Complete XAI analysis notebook
│
├── report/
│   └── Report_XAI_Antonio_Lorenzo.pdf     # report
│
├── monitoring/             # Prometheus + Grafana (local)
│   ├── prometheus.yaml
│   └── grafana-provisioning/
│
├── k8s/                    # Kubernetes manifests (GKE)
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
├── docker-compose.yml      # Complete local execution
├── dvc.yaml                # Pipelines Iris + Telco
├── dvc.lock
├── requirements.txt
└── README.md
```

---

# 1. 🔧 Requirements

For local execution:

✔ Docker
✔ Docker Compose

> You do not need to install Python or dependencies if you use `docker-compose`.

For development:

✔ Python 3.11
✔ DVC
✔ MLflow

---

# 2. 📥 Clone the repository

```bash
git clone https://github.com/AntonioLDM17/mlops-practica-icai.git
cd mlops-practica-icai
```

---

# 3. 📊 Telco Dataset (if not exists)

The main dataset should be in:

```
data/telco_churn.csv
```

If missing:

```bash
export PYTHONPATH=.
python -m scripts.download_telco_data
```

After:

```bash
cp "<kaggle_path>/WA_Fn-UseC_-Telco-Customer-Churn.csv" data/telco_churn.csv
dvc add data/telco_churn.csv
dvc push
```

---

# 4. 🚀 Quick start with Docker Compose

```bash
docker-compose up --build
```

They will be available:

| Service          | Port | Description     |
| ----------------- | ------ | --------------- |
| `mlops-api`       | 5000   | Iris API        |
| `mlops-web`       | 8501   | Iris Web        |
| `mlops-telco-api` | 5001   | Telco + XAI API |
| `mlops-telco-web` | 8502   | Telco XAI Web   |
| `prometheus`      | 9090   | Iris Metrics    |
| `grafana`         | 3000   | Dashboards      |

---

## 4.1. 🌐 Local URLs

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

# 5. 🧪 Running without Docker (optional)

Create environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run:

```bash
python src/iris/app.py
streamlit run src/iris/app_web.py --server.port 8501

python src/telco_xai/app_telco.py
streamlit run src/telco_xai/app_web_telco.py --server.port 8502
```

---

# 6. 📘 Explainability Notebook

Location:

```
notebooks/telco_xai.ipynb
```

Contains:

✔ Complete training
✔ Global explainability (PFI + SHAP)
✔ Local explainability
✔ Sanity checks
✔ ROC and PR-AUC curves
✔ Final interpretation

---

# 7. 🔄 Pipelines with DVC + MLflow
## Iris Pipeline

**Input:** `data/iris_dataset.csv`
**Output:**
* `model.pkl`
* `confusion_matrix.png`
* `mlflow_metrics.json`

## Telco Churn Pipeline

**Output:**

* `model_telco.pkl`
* `telco_metrics.json`
* `artifacts_telco/`

  * Background SHAP
  * Feature names
  * Permutation FI
  * SHAP Global Importance

Run:

```bash
dvc repro
```

---

# 8. ☁ Deployment on Google Kubernetes Engine (GKE)

Manifests are in:

```
k8s/
```

Manual deployment:

```bash
kubectl apply -f k8s/
```

Includes:

✔ Iris API
✔ Iris Web
✔ Telco + XAI API
✔ Telco Web
✔ PodMonitoring (Prometheus Iris)

With GitHub Actions, the `deploy-to-gke` job does everything automatically.

---

# 9. 🎯 Quick summary
### 1️⃣ Run everything locally:

```bash
docker-compose up --build
```

### 2️⃣ Access:

* Iris Web: [http://localhost:8501](http://localhost:8501)
* Telco XAI Web: [http://localhost:8502](http://localhost:8502)
* Prometheus: [http://localhost:9090](http://localhost:9090)
* Grafana: [http://localhost:3000](http://localhost:3000)

### 3️⃣ Reproduce pipelines:

```bash
dvc repro
```

### 4️⃣ Download dataset if missing:

```bash
export PYTHONPATH=.
python -m scripts.download_telco_data
```

