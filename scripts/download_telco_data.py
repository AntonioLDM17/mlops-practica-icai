import kagglehub
import shutil
import os

"""
Download Telco Customer Churn Dataset from Kaggle
and copy it to data/telco_churn.csv

No API key required.
"""

print("📥 Downloading Telco dataset from Kaggle...")
path = kagglehub.dataset_download("blastchar/telco-customer-churn")

print("📁 Dataset downloaded at:", path)

# Files included in the dataset
csv_path = os.path.join(path, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError("❌ CSV not found inside the downloaded dataset.")
# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Copy file
dest = "data/telco_churn.csv"
shutil.copy(csv_path, dest)

print(f"✅ File copied to {dest}")