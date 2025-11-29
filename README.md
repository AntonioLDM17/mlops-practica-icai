# MLOps + XAI – Iris + Telco Churn (XAI, CI/CD, GKE)

Este proyecto implementa un pipeline de **MLOps completo** con dos casos de uso:

1. **Clasificación Iris** (Random Forest + métricas + monitorización Prometheus/Grafana).
2. **Predicción de churn Telco** con **explicabilidad avanzada** (Permutation Importance + SHAP) y una **web de exploración XAI**, que permite:

   * Predicción con explicación local (SHAP).
   * Explicabilidad global (Permutation FI + SHAP Global).
   * Reentrenamiento rápido eliminando features seleccionadas.
   * Sanity-checks completos:

     * Comparación baseline vs modelo reducido.
     * Comparación baseline vs modelo entrenado con **etiquetas barajadas**.

Además, el repositorio incluye:

* **Notebook completo de análisis XAI:** `telco_xai.ipynb`, que reúne

  * entrenamiento,
  * todas las métricas,
  * gráficos ROC y PR-AUC,
  * explicabilidad global/local,
  * sanity checks,
  * interpretación detallada.

* **Script de descarga de dataset:** `download_telco_data.py`
  (útil si `data/telco_churn.csv` no existe).

Incluye:

* Versionado de datos/models con **DVC**
* Trazabilidad con **MLflow**
* APIs Flask (Iris + Telco)
* Frontends Streamlit
* Docker + docker-compose
* CI/CD con GitHub Actions
* Despliegue en **GKE** (opcional)

---

## 1. Requisitos

Para ejecutar todo en local:

* Docker
* Docker Compose

No hace falta instalar Python, DVC o MLflow para usar las aplicaciones web.

---

## 2. Clonar el repositorio

```bash
git clone https://github.com/AntonioLDM17/mlops-practica-icai.git
cd mlops-practica-icai
```

---

## 3. Dataset (si no está descargado)

El dataset Telco se almacena en `data/telco_churn.csv`.

Si no existe, puede descargarse automáticamente mediante:

```bash
python download_telco_data.py
```

---

## 4. Ejecución rápida con Docker (recomendado)

El `docker-compose.yml` levanta:

* `mlops-api` (Iris API)
* `mlops-web` (Iris Web)
* `mlops-telco-api` (Telco API con XAI + retrain + sanity checks)
* `mlops-telco-web` (Web Telco XAI)
* `prometheus`
* `grafana`

### 4.1. Levantar todo

```bash
docker-compose up --build
```

### 4.2. URLs principales

#### 🌸 Iris Web

**[http://localhost:8501](http://localhost:8501)**

#### 📡 Telco Web (XAI + Retrain + Sanity Checks)

**[http://localhost:8502](http://localhost:8502)**

Incluye:

* Predicción + SHAP local
* Explicabilidad global (PFI + SHAP Global)
* Reentrenamiento sin features
* Sanity check de etiquetas barajadas
* Gráficas comparativas ROC & PR-AUC

#### 📈 Prometheus

[http://localhost:9090](http://localhost:9090)

#### 📊 Grafana

[http://localhost:3000](http://localhost:3000) (admin / admin)

### 4.3. Parar

```bash
docker-compose down
```

---

## 5. Ejecución sin Docker (opcional)

### 5.1. Crear entorno

```bash
python -m venv .venv
source .venv/bin/activate
```

### 5.2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 5.3. Ejecutar APIs y Webs

API Telco:

```bash
python app_telco.py
```

Web Telco:

```bash
streamlit run app_web_telco.py --server.port 8502
```

---

## 6. Notebook completo de explicabilidad

El repositorio incluye el notebook:

### **`telco_xai.ipynb`**

Este notebook contiene:

* Carga y preprocesado del dataset
* Entrenamiento de Logistic Regression y Random Forest
* Métricas completas (Accuracy, Balanced Accuracy, ROC AUC, AUC-PR…)
* Curvas ROC & Precision–Recall
* Permutation Feature Importance
* SHAP Global y Local (gráficos completos)
* Sanity check eliminando features
* Sanity check barajando etiquetas
* Interpretación detallada de resultados

Es el documento central para la parte de **Explicabilidad**.

---


## 7. (Opcional) Entrenamiento con DVC y MLflow

El fichero `dvc.yaml` contiene dos pipelines:

---

### ✔ Pipeline Iris

Genera:

* `model.pkl`
* `confusion_matrix.png`
* `mlflow_metrics.json`

---

### ✔ Pipeline Telco

Ejecuta `train_telco.py` y crea:

* `model_telco.pkl` (baseline oficial usado en la API)
* `telco_metrics.json`
* Artefactos en `artifacts_telco/`:

  * `telco_background.csv`
  * `telco_feature_names.json`
  * `telco_perm_importance.json`
  * `telco_shap_global.json`

El baseline se utiliza como referencia para:

* Explicabilidad local/global
* Reentrenamiento comparado
* Sanity-check sin features
* Sanity-check shuffle labels

---

### Reproducir todo el pipeline

```bash
dvc repro
```

---

## 8. (Opcional) Despliegue en Google Kubernetes Engine (GKE)

Incluye:

* API Iris
* Web Iris
* API Telco Churn + XAI + sanity checks
* Web Telco
* Prometheus (monitorización Iris)

Cuando el clúster GKE está levantado y se ejecuta el job de deploy-to-gke, se aplica todo con:

```bash
kubectl apply -f api-deployment.yaml
kubectl apply -f api-service.yaml
kubectl apply -f web-deployment.yaml
kubectl apply -f web-service.yaml
kubectl apply -f telco-api-deployment.yaml
kubectl apply -f telco-api-service.yaml
kubectl apply -f telco-web-deployment.yaml
kubectl apply -f telco-web-service.yaml
kubectl apply -f pod-monitoring.yaml
```

---

## 9. Resumen rápido

1. Levantar todo:

```bash
docker-compose up --build
```

2. Entrar a:

* Iris Web → [http://localhost:8501](http://localhost:8501)
* Telco Web (XAI) → [http://localhost:8502](http://localhost:8502)
* Prometheus → [http://localhost:9090](http://localhost:9090)
* Grafana → [http://localhost:3000](http://localhost:3000)

3. Si falta el dataset:

```bash
python download_telco_data.py
```

4. El notebook `telco_xai.ipynb` contiene **todo el análisis XAI completo**.