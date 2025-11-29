import kagglehub
import shutil
import os

"""
Descarga el dataset Telco Customer Churn desde Kaggle
y lo copia automáticamente a data/telco_churn.csv.

No requiere API key.
"""

print("📥 Descargando dataset Telco desde Kaggle...")
path = kagglehub.dataset_download("blastchar/telco-customer-churn")

print("📁 Dataset descargado en:", path)

# Archivos que vienen en el dataset
csv_path = os.path.join(path, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError("❌ No se encontró el CSV dentro del dataset descargado.")

# Crear carpeta data si no existe
os.makedirs("data", exist_ok=True)

# Copiar archivo
dest = "data/telco_churn.csv"
shutil.copy(csv_path, dest)

print(f"✅ Archivo copiado a {dest}")
