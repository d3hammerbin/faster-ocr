FROM python:3.11-slim-bookworm

# Configurar timezone
ENV TZ=America/Mexico_City
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Instalar dependencias del sistema necesarias para PaddleOCR, OpenCV y health check
RUN apt-get update && apt-get install -y \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libfontconfig1 \
    libxrender1 \
    ffmpeg \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements y instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY . .

# Copiar y ejecutar script de inicialización de base de datos
COPY init-database.sh /app/
RUN chmod +x /app/init-database.sh

# Crear directorios para datos y base de datos
RUN mkdir -p /app/data /app/database
RUN chmod 755 /app/database

# Exponer puerto
EXPOSE 8000

# Variables de entorno
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Comando para ejecutar la aplicación
CMD ["/bin/bash", "-c", "/app/init-database.sh && python -m uvicorn main:app --host 0.0.0.0 --port 8000"]