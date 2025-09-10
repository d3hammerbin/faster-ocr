#!/bin/bash

# Script para iniciar Faster OCR API con Docker

set -e

echo "🚀 Iniciando Faster OCR API con Docker..."

# Verificar si existe el archivo .env
if [ ! -f ".env" ]; then
    echo "⚠️  Archivo .env no encontrado. Creando desde .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "📝 Archivo .env creado. Por favor, configura tu OPENAI_API_KEY antes de continuar."
        echo "💡 Edita el archivo .env y agrega tu clave API de OpenAI."
        exit 1
    else
        echo "❌ Archivo .env.example no encontrado."
        exit 1
    fi
fi

# Verificar si OPENAI_API_KEY está configurada
if ! grep -q "OPENAI_API_KEY=sk-" .env 2>/dev/null; then
    echo "⚠️  OPENAI_API_KEY no está configurada correctamente en .env"
    echo "💡 Edita el archivo .env y agrega tu clave API de OpenAI."
    exit 1
fi

# Crear directorio de datos si no existe
if [ ! -d "data" ]; then
    echo "📁 Creando directorio de datos..."
    mkdir -p data
fi

# Verificar si Docker está ejecutándose
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker no está ejecutándose. Por favor, inicia Docker Desktop."
    exit 1
fi

# Verificar si Docker Compose está disponible
if ! command -v docker-compose > /dev/null 2>&1; then
    echo "❌ Docker Compose no está instalado."
    exit 1
fi

echo "🔨 Construyendo imagen Docker..."
docker-compose build

echo "🚀 Iniciando contenedor..."
docker-compose up -d

echo "⏳ Esperando que la aplicación esté lista..."
sleep 10

# Verificar si la aplicación está respondiendo
if curl -f http://localhost:8000/ > /dev/null 2>&1; then
    echo "✅ ¡Faster OCR API está ejecutándose correctamente!"
    echo ""
    echo "🌐 Accesos:"
    echo "   API: http://localhost:8000"
    echo "   Documentación: http://localhost:8000/docs"
    echo ""
    echo "📋 Comandos útiles:"
    echo "   Ver logs: docker-compose logs -f"
    echo "   Detener: docker-compose down"
    echo "   Estado: docker-compose ps"
else
    echo "❌ La aplicación no está respondiendo. Verificando logs..."
    docker-compose logs --tail=20
    exit 1
fi