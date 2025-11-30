# Imagen base
FROM python:3.11-slim

# Mejora logs en tiempo real
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código y modelo
COPY src ./src
COPY model.pkl .

# Exponer puerto
EXPOSE 5000

# Ejecutar API Iris
CMD ["python", "src/iris/app.py"]
