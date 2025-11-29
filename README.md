# MLOps – Iris + Telco Churn (XAI, CI/CD, GKE)

Este proyecto implementa un pipeline de **MLOps completo** con dos casos de uso:

1. **Clasificación Iris** (Random Forest + métricas + monitorización Prometheus/Grafana).
2. **Predicción de churn Telco** con **explicabilidad** (Permutation Importance + SHAP) y una **web de exploración XAI**, que permite:

   * Predicción con explicación local (SHAP).
   * Explicabilidad global (Permutation FI + SHAP Global).
   * Reentrenamiento rápido eliminando features seleccionadas para comparar el rendimiento con respecto al modelo baseline oficial.

Incluye:

* Versionado de datos/modelos con **DVC**.
* Trazabilidad de experimentos con **MLflow** (remoto en DagsHub).
* **APIs Flask** para Iris y Telco.
* **Frontends Streamlit** para Iris y Telco XAI.
* **Docker + docker-compose** para reproducción local.
* **CI/CD con GitHub Actions + CML**.
* Despliegue en **GKE** (opcional).

---

## 1. Requisitos

Para **ejecutar el proyecto desde cero en local**, se necesita:

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

## 3. Ejecución rápida con Docker (recomendado)

El fichero `docker-compose.yml` levanta:

* `mlops-api` → API Iris
* `mlops-web` → Web Iris
* `mlops-telco-api` → API Telco Churn + XAI + retrain
* `mlops-telco-web` → Web Telco Churn + XAI + sanity-check
* `prometheus` → Servidor Prometheus
* `grafana` → Dashboard Grafana

### 3.1. Levantar todo

```bash
docker-compose up --build
```

### 3.2. URLs locales

#### 🌸 3.2.1. Web Iris

`http://localhost:8501`

Permite interactuar con el modelo Iris y visualizar métricas.

---

#### 📡 3.2.2. Web Telco Churn + XAI + Sanity Check

`http://localhost:8502`

Incluye:

### ✔ Predicción + Explicación local

* Formulario para introducir los datos del cliente.
* El backend devuelve:

  * Probabilidad de churn
  * Predicción (0/1)
  * Valores **SHAP locales**

### ✔ Explicabilidad global

* **Permutation Feature Importance** (features originales)
* **SHAP Global** (features transformadas: numéricas + one-hot)

### ✔ Sanity-check interactivo (reentrenamiento sin features)

Permite:

* Seleccionar un conjunto de features a eliminar
* Reentrenar un modelo desde cero en las mismas condiciones que el baseline
* Comparar:

| Métrica                 | Comparación                        |
| ----------------------- | ---------------------------------- |
| Accuracy                | Baseline vs reducido               |
| Balanced Accuracy       | Ideal para dataset desbalanceado   |
| ROC AUC                 | Curvas ROC comparadas              |
| AUC-PR                  | Métrica principal para churn       |
| Precision / Recall / F1 | Especialmente sobre clase positiva |
| Confusion Matrix        | TN / FP / FN / TP                  |
| Importancia eliminada   | Comprobación de explicabilidad     |

---

### 📈 3.2.3. Prometheus

`http://localhost:9090`

Recolecta métricas expuestas por la API Iris.

---

### 📊 3.2.4. Grafana

`http://localhost:3000`
Usuario: `admin`
Contraseña: `admin`

Dashboards precargados para visualizar las métricas monitorizadas.

---

### 3.3. Parar servicios

```bash
docker-compose down
```

---

## 4. (Opcional) Ejecución sin Docker

*(igual que antes, no modificado)*

---

## 5. Entrenamiento con DVC y MLflow

El fichero `dvc.yaml` define dos pipelines reproducibles:

### ✔ Pipeline Iris

Genera:

* `model.pkl`
* `confusion_matrix.png`
* `mlflow_metrics.json`

### ✔ Pipeline Telco

Ejecuta `train_telco.py`, que genera:

* `model_telco.pkl` (modelo baseline utilizado en la API)
* `telco_metrics.json`
* Carpeta `artifacts_telco/` con:

  * `telco_background.csv`
  * `telco_feature_names.json`
  * `telco_perm_importance.json`
  * `telco_shap_global.json`

Este baseline es la referencia oficial para las comparaciones dentro del modo **sanity-check** de la web de Telco.

Para ejecutar todo el pipeline:

```bash
dvc repro
```

---

## 6. (Opcional) Despliegue en Google Kubernetes Engine (GKE)

*(igual que antes)*

Se despliegan:

* API Iris
* Web Iris
* API Telco Churn + XAI + retrain
* Web Telco (frontend XAI + sanity check)
* Prometheus (vía PodMonitoring)

Y se aplican los manifiestos:

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

## 7. Resumen rápido

1. Ejecutar en local:

   ```bash
   docker-compose up --build
   ```

2. Acceder a:

   * Iris Web → `http://localhost:8501`
   * Telco XAI Web → `http://localhost:8502`

     * Predicción + explicabilidad
     * Explicabilidad global
     * **Sanity-check con reentrenamiento**
   * Prometheus → `http://localhost:9090`
   * Grafana → `http://localhost:3000`

3. No se necesita configurar Python, MLflow o DVC para usar las webs.

4. El proyecto demuestra un pipeline completo de **MLOps real**, con CI/CD, XAI, reproductibilidad, Docker, Kubernetes y monitorización.
