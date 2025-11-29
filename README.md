# MLOps + XAI – Iris + Telco Churn (XAI, CI/CD, GKE)

Este proyecto implementa un pipeline de **MLOps completo** con dos casos de uso:

1. **Clasificación Iris** (Random Forest + métricas + monitorización Prometheus/Grafana).
2. **Predicción de churn Telco** con **explicabilidad** (Permutation Importance + SHAP) y una **web de exploración XAI**, que permite:

   * Predicción con explicación local (SHAP).
   * Explicabilidad global (Permutation FI + SHAP Global).
   * Reentrenamiento rápido eliminando features seleccionadas.
   * Sanity-checks completos de modelo:

     * Comparación baseline vs modelo reentrenado sin ciertas features.
     * Comparación baseline vs modelo entrenado con **etiquetas barajadas** (shuffle labels).

Incluye:

* Versionado de datos y modelos con **DVC**.
* Trazabilidad de experimentos con **MLflow** (remoto en DagsHub).
* **APIs Flask** para Iris y Telco.
* **Frontends Streamlit** para Iris y Telco XAI.
* **Docker y docker-compose** para ejecución local.
* **CI/CD con GitHub Actions + CML**.
* **Despliegue automático en GKE** (opcional).

---

## 1. Requisitos

Para ejecutar el proyecto desde cero en local:

* Docker
* Docker Compose

No es necesario instalar Python, DVC ni MLflow para usar las aplicaciones web.

---

## 2. Clonar el repositorio

```bash
git clone https://github.com/AntonioLDM17/mlops-practica-icai.git
cd mlops-practica-icai
```

---

## 3. Ejecución rápida con Docker (recomendado)

El fichero `docker-compose.yml` levanta los siguientes servicios:

* `mlops-api` → API Iris
* `mlops-web` → Web Iris
* `mlops-telco-api` → API Telco Churn + XAI + sanity checks
* `mlops-telco-web` → Web Telco Churn + XAI + retrain + sanity checks
* `prometheus` → Recolección de métricas
* `grafana` → Dashboards provisionados

### 3.1. Levantar todo

```bash
docker-compose up --build
```

### 3.2. URLs locales

---

### 🌸 3.2.1. Web Iris

**[http://localhost:8501](http://localhost:8501)**
Interfaz Streamlit para predicción y visualización.

---

### 📡 3.2.2. Web Telco Churn + XAI + Sanity Checks

**[http://localhost:8502](http://localhost:8502)**

Incluye:

### ✔ Predicción + Explicación local

* Probabilidad de churn
* Predicción (0/1)
* Valores SHAP locales

---

### ✔ Explicabilidad global

* **Permutation Feature Importance**
* **SHAP Global**

---

### ✔ Reentrenamiento rápido eliminando features

Comparación detallada baseline vs reducido:

* Accuracy
* Balanced Accuracy
* ROC AUC
* AUC-PR
* Precision/Recall/F1
* Matriz de confusión
* Curvas ROC comparadas
* Curvas Precision–Recall comparadas
* Importancia eliminada

---

### ✔ Sanity-check: modelo entrenado con etiquetas barajadas (shuffle labels)

Analiza el comportamiento del modelo cuando **no existe señal real**:

* Entrenamiento con labels aleatorizados
* Comparación directo con baseline con:

  * Balanced accuracy ≈ 0.5
  * ROC AUC ≈ 0.5
  * AUC-PR igual a prevalencia
  * Curvas ROC diagonales
  * Curvas PR horizontales
  * Matriz de confusión de predicciones aleatorias
  * FI y SHAP sin estructura coherente

Con esto se valida:

* Ausencia de data leakage
* Coherencia de explicabilidad
* Que el baseline realmente aprende algo

---

### 📈 3.2.3. Prometheus

**[http://localhost:9090](http://localhost:9090)**

Recolecta métricas expuestas por la API Iris.

---

### 📊 3.2.4. Grafana

**[http://localhost:3000](http://localhost:3000)**
Usuario: `admin`
Contraseña: `admin`

---

### 3.3. Parar servicios

```bash
docker-compose down
```

---

## 4. (Opcional) Ejecución sin Docker – APIs y Webs

Esta sección permite ejecutar todo desde Python, aunque **Docker sigue siendo la vía recomendada**.

### Requisitos

* Python 3.11
* pip
* Opcional: conda o venv

---

### 4.1. Crear entorno virtual (ejemplo con venv)

```bash
python -m venv .venv
source .venv/bin/activate    # Linux / Mac
# o en Windows:
# .venv\Scripts\activate
```

---

### 4.2. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Nota: los datos y modelos se gestionan con **DVC**.
> Para entrenamiento reproducible necesitarías credenciales de DagsHub.
> Pero **para ejecutar las apps no es necesario**.

---

### 4.3. Lanzar a mano API + Web Telco

Terminal 1 → API Telco:

```bash
python app_telco.py
# API escuchando en http://localhost:5001
```

Terminal 2 → Web Telco:

```bash
streamlit run app_web_telco.py --server.port 8502
```

Iris tiene equivalente (`app.py` y `app_web.py`), aunque no es necesario si se usa Docker.

---

## 5. (Opcional) Entrenamiento con DVC y MLflow

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

## 6. (Opcional) Despliegue en Google Kubernetes Engine (GKE)

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

## 7. Resumen rápido

1. Ejecutar en local:

   ```bash
   docker-compose up --build
   ```

2. Abrir:

   * Iris Web → **[http://localhost:8501](http://localhost:8501)**
   * Telco XAI Web → **[http://localhost:8502](http://localhost:8502)**

     * Predicción + explicación local
     * Explicabilidad global
     * Reentrenamiento eliminando features
     * **Sanity-check: shuffle labels**
   * Prometheus → **[http://localhost:9090](http://localhost:9090)**
   * Grafana → **[http://localhost:3000](http://localhost:3000)**

3. No requiere instalación local de Python, DVC o MLflow para usar las webs.

4. Demuestra un pipeline completo de **MLOps real**: reproducibilidad, CICD, Docker, Kubernetes, XAI y monitorización.
