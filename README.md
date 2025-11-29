# MLOps – Iris + Telco Churn (XAI, CI/CD, GKE)

Este proyecto implementa un pipeline de **MLOps completo** con dos casos de uso:

1. **Clasificación Iris** (Random Forest + métricas + monitorización Prometheus/Grafana).
2. **Predicción de churn Telco** con **explicabilidad** (Permutation Importance + SHAP) y una **web de exploración XAI**.

Incluye:

* Versionado de datos/modelos con **DVC**.
* Trazabilidad de experimentos con **MLflow** (remoto en DagsHub).
* **APIs Flask** para Iris y Telco.
* **Frontends Streamlit** para Iris y Telco XAI.
* **Docker + docker-compose** para reproducir todo en local.
* **CI/CD con GitHub Actions + CML**.
* Despliegue en **Google Kubernetes Engine (GKE)** (opcional, gestionado por el autor).

---

## 1. Requisitos

Para **ejecutar el proyecto desde cero en local**, solo hace falta:

* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose](https://docs.docker.com/compose/) (en Docker Desktop ya viene incluido)

💡 **No es necesario** tener Python, DVC, MLflow ni cuenta en Google Cloud para **probar las aplicaciones web**.

---

## 2. Clonar el repositorio

```bash
git clone https://github.com/AntonioLDM17/mlops-practica-icai.git
cd mlops-practica-icai
```

---

## 3. Ejecución rápida con Docker (recomendado)

El fichero `docker-compose.yml` levanta:

* `mlops-api` → API de Iris (Flask, puerto interno 5000).
* `mlops-web` → Web de Iris (Streamlit).
* `mlops-telco-api` → API de Telco Churn + XAI (Flask, puerto interno 5001).
* `mlops-telco-web` → Web de Telco Churn + XAI (Streamlit).
* `prometheus` → Servidor Prometheus.
* `grafana` → Dashboard Grafana con Prometheus como datasource.

### 3.1. Levantar todos los servicios

Desde la carpeta raíz del proyecto:

```bash
docker-compose up --build
```

La primera vez puede tardar unos minutos porque:

* Descarga las imágenes base de Python.
* Instala dependencias (`requirements.txt`).
* Construye las imágenes de las APIs y las webs.

Cuando termine verás los logs de todos los contenedores.

> Si quieres lanzarlo en segundo plano:
>
> ```bash
> docker-compose up -d --build
> ```

---

### 3.2. URLs de acceso en local

Una vez levantado con Docker, puedes entrar a:

#### 🌸 3.2.1. Web Iris (clasificación de flores)

* **URL**: `http://localhost:8501`

Características:

* Interfaz Streamlit para interactuar con el modelo Iris.
* Llama internamente a la API `mlops-api` (Flask).
* Permite hacer predicciones y ver el comportamiento del modelo.

---

#### 📡 3.2.2. Web Telco Churn + XAI

* **URL**: `http://localhost:8502`

Características:

* Modo **Predicción + explicación local**:

  * Rellena el formulario con las características de un cliente Telco.
  * El backend devuelve:

    * Probabilidad de churn.
    * Predicción binaria (0 = se queda, 1 = se va).
    * Valores SHAP locales para esa observación.
* Modo **Explicabilidad global**:

  * Muestra:

    * Importancia global vía **Permutation Feature Importance**.
    * Importancia global vía **SHAP** en el espacio transformado (numéricas + one-hot).

---

#### 📈 3.2.3. Prometheus

* **URL**: `http://localhost:9090`

Se utiliza para:

* Recolectar métricas de la API de Iris.
* Exponerlas para su consumo desde Grafana.

---

#### 📊 3.2.4. Grafana

* **URL**: `http://localhost:3000`
* Usuario por defecto: `admin`
* Contraseña: `admin` (configurada en `docker-compose.yml`)

Hay dashboards provisionados automáticamente (vía `./grafana-provisioning`) que permiten:

* Ver métricas de la API de Iris.
* Explorar la información exportada por Prometheus.

---

### 3.3. Parar los servicios

Si lanzaste en primer plano (con logs):

* Pulsa `CTRL + C` en la terminal.

Si está en segundo plano (`-d`):

```bash
docker-compose down
```

---

## 4. (Opcional) Ejecución sin Docker – APIs y Webs

> ⚠️ Esta parte es opcional y está pensada para entornos con Python configurado.
> Para ver su funcionamiento y pruebas sencillas, **es suficiente con usar Docker**.

Requisitos:

* Python 3.11
* `pip`
* Opcionalmente `conda` o `venv` para crear un entorno virtual.

### 4.1. Crear entorno virtual (ejemplo con venv)

```bash
python -m venv .venv
source .venv/bin/activate  # Linux / Mac
# o en Windows:
# .venv\Scripts\activate
```

### 4.2. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Nota: en este proyecto los datos y modelos grandes (datasets, modelos entrenados, artifacts XAI) están versionados con **DVC** y almacenados en un remoto (DagsHub).
> Para reproducir el entrenamiento completo haría falta configurar las credenciales de DagsHub.
> Sin embargo, para pruebas de ejecución con Docker **no es necesario**.

### 4.3. Lanzar API y Web de Telco (local)

En una terminal:

```bash
python app_telco.py
# API Telco escuchando en http://localhost:5001
```

En otra terminal:

```bash
streamlit run app_web_telco.py --server.port 8502
# Web Telco XAI en http://localhost:8502
```

De forma similar se podrían lanzar la API y la web de Iris (`app.py` y `app_web.py`), pero para la práctica la vía recomendada es **Docker**.

---

## 5. (Opcional) Entrenamiento con DVC y MLflow

> Esta parte documenta la lógica de MLOps, pero **no es necesaria** para el uso normal de las webs.

El fichero `dvc.yaml` define dos pipelines:

* **Iris**:

  * `prepare` → verifica que existe `data/iris_dataset.csv`.
  * `train` → ejecuta `python train.py --n_estimators <N>`, genera:

    * `model.pkl`
    * `confusion_matrix.png`
    * `mlflow_metrics.json` (métrica de accuracy para DVC)

* **Telco**:

  * `prepare_telco` → verifica que existe `data/telco_churn.csv`.
  * `train_telco` → ejecuta `python train_telco.py`, genera:

    * `model_telco.pkl`
    * `telco_metrics.json`
    * artefactos XAI en `artifacts_telco/`:

      * `telco_background.csv` (background para SHAP)
      * `telco_feature_names.json` (features originales)
      * `telco_perm_importance.json`
      * `telco_shap_global.json`

Para reprocesar el pipeline con DVC:

```bash
dvc repro
```

MLflow está configurado para loggear experimentos (por ejemplo, en DagsHub usando las variables de entorno que se pasan en el workflow de GitHub Actions).

---

## 6. (Opcional) Despliegue en Google Kubernetes Engine (GKE)

> Esta parte **no hace falta** ejecutarla si solo quieres probar las aplicaciones web en local o con Docker.
> Está gestionada por el autor y automatizada vía GitHub Actions.

Resumen:

* Imágenes Docker se construyen y escanean con **Trivy** en GitHub Actions.
* Se publican en **Google Container Registry**:
  `gcr.io/icai2025-mlops/mlops-api`, `gcr.io/icai2025-mlops/mlops-web`,
  `gcr.io/icai2025-mlops/mlops-telco-api`, `gcr.io/icai2025-mlops/mlops-telco-web`.
* Manifiestos Kubernetes:

  * `api-deployment.yaml`, `api-service.yaml` → API Iris.
  * `web-deployment.yaml`, `web-service.yaml` → Web Iris.
  * `telco-api-deployment.yaml`, `telco-api-service.yaml` → API Telco.
  * `telco-web-deployment.yaml`, `telco-web-service.yaml` → Web Telco XAI.
  * `pod-monitoring.yaml` → integración con Prometheus.

Cuando el clúster GKE está levantado y se ejecuta el job de `deploy-to-gke`, se aplica todo con:

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

1. **Forma más simple de probar el proyecto** 👉
   Clonar repo y ejecutar:

   ```bash
   docker-compose up --build
   ```

   Luego abrir:

   * Iris Web: `http://localhost:8501`
   * Telco XAI Web: `http://localhost:8502` (predicción + explicabilidad)
   * Prometheus: `http://localhost:9090`
   * Grafana: `http://localhost:3000` (admin / admin)

2. **No necesita** cuenta en Google Cloud ni configuración de DVC para usar las aplicaciones.

3. El despliegue en GKE y la integración con MLflow/DVC están documentados y automatizados, y demuestran la parte de **MLOps avanzado**.

