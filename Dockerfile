# 1) Usar una imagen base oficial de Python (Linux)
FROM python:3.10-slim

# 2) Directorio de trabajo dentro del contenedor
WORKDIR /app

# 3) Instalar herramientas de sistema que algunas libs necesitan (opcional pero útil)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# 4) Copiar el archivo de dependencias para el contenedor
COPY requirements-docker.txt ./requirements-docker.txt

# 5) Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements-docker.txt

# 6) Copiar TODO el repositorio dentro del contenedor
COPY . .

# 7) Exponer el puerto donde correrá FastAPI
EXPOSE 8000

# 8) Comando que se ejecuta al arrancar el contenedor
#    IMPORTANTE: esto asume que habrá un archivo app/main.py con un objeto FastAPI llamado `app`
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
