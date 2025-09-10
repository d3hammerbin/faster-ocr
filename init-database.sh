#!/bin/bash

# Script de inicialización para la base de datos
# Asegurar que el directorio de la base de datos existe y crear la DB si no existe

echo "Inicializando directorio de base de datos..."

# Crear directorio si no existe
mkdir -p /app/database

# Establecer permisos correctos
chmod 755 /app/database

# Si existe una base de datos local, migrarla al volumen
if [ -f "/app/credentials.db" ]; then
    echo "Migrando base de datos local al volumen persistente..."
    cp /app/credentials.db /app/database/credentials.db
    chmod 644 /app/database/credentials.db
    echo "Base de datos migrada exitosamente."
fi

# Verificar si la base de datos existe en el host, si no, crearla
if [ ! -f "/app/host/credentials.db" ]; then
    echo "Base de datos no encontrada en el host. Creando nueva base de datos..."
    python -c "
import sqlite3
import sys
sys.path.append('/app')
from database import init_database

# Crear la base de datos en el host
conn = sqlite3.connect('/app/host/credentials.db')
conn.close()

# Crear un enlace simbólico en el directorio de la aplicación
import os
if os.path.exists('/app/database/credentials.db'):
    os.remove('/app/database/credentials.db')
os.symlink('/app/host/credentials.db', '/app/database/credentials.db')

# Inicializar las tablas
init_database()
print('Base de datos creada exitosamente en el host.')
"
else
    echo "Base de datos existente encontrada en el host."
    # Crear enlace simbólico si no existe
    if [ ! -f "/app/database/credentials.db" ]; then
        ln -sf /app/host/credentials.db /app/database/credentials.db
    fi
fi

echo "Inicialización de base de datos completada."