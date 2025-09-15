#!/bin/bash

# Script de inicialización para la base de datos
# Asegurar que el directorio de la base de datos existe y crear la DB si no existe

echo "Inicializando directorio de base de datos..."

# Crear directorio si no existe
mkdir -p /app/database

# Establecer permisos correctos
chmod 755 /app/database

# Si existe una base de datos en el host, migrarla al volumen
if [ -f "/app/host/credentials.db" ]; then
    echo "Migrando base de datos del host al volumen persistente..."
    cp /app/host/credentials.db /app/database/credentials.db
    chmod 644 /app/database/credentials.db
    echo "Base de datos migrada exitosamente desde el host."
elif [ ! -f "/app/database/credentials.db" ]; then
    echo "No se encontró base de datos existente. Se creará una nueva cuando sea necesario."
fi

# Verificar que el directorio existe
if [ -d "/app/database" ]; then
    echo "Directorio de base de datos creado exitosamente: /app/database"
    ls -la /app/database/
else
    echo "Error: No se pudo crear el directorio de base de datos"
    exit 1
fi

echo "Inicialización de base de datos completada."

# Verificar que Python puede importar los módulos necesarios
echo "Verificando dependencias de Python..."
python -c "import sqlite3; import fastapi; print('Dependencias verificadas correctamente')" || {
    echo "Error: Faltan dependencias de Python"
    exit 1
}

echo "Iniciando aplicación FastAPI..."