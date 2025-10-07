# Imagen base
FROM python:3.11-slim

# Mejora logs en tiempo real
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicación y modelo
COPY app.py .
COPY model.pkl .

# Exponer puerto
EXPOSE 5000

# Ejecutar API
CMD ["python", "app.py"]
