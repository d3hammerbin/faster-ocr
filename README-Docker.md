# Faster OCR API - Docker

Esta guía explica cómo ejecutar la aplicación Faster OCR API usando Docker y Docker Compose v2.

## Requisitos Previos

- Docker Engine 20.10+
- Docker Compose v2.0+
- Archivo `.env` con las variables de entorno necesarias

## Configuración Inicial

1. **Clonar el repositorio y navegar al directorio:**
   ```bash
   cd faster-ocr
   ```

2. **Crear archivo de variables de entorno:**
   ```bash
   cp .env.example .env
   ```
   
   Editar el archivo `.env` y configurar:
   ```env
   OPENAI_API_KEY=tu_clave_api_de_openai
   ```

3. **Crear directorio de datos (opcional):**
   ```bash
   mkdir -p data
   ```

## Comandos Docker Compose

### Construir y ejecutar la aplicación
```bash
# Construir la imagen y ejecutar en segundo plano
docker-compose up -d --build

# Ver logs en tiempo real
docker-compose logs -f

# Ver estado de los servicios
docker-compose ps
```

### Gestión del contenedor
```bash
# Detener la aplicación
docker-compose down

# Detener y eliminar volúmenes
docker-compose down -v

# Reiniciar la aplicación
docker-compose restart

# Reconstruir sin caché
docker-compose build --no-cache
```

### Acceso y debugging
```bash
# Acceder al contenedor
docker-compose exec faster-ocr-api bash

# Ver logs específicos
docker-compose logs faster-ocr-api

# Seguir logs en tiempo real
docker-compose logs -f faster-ocr-api
```

## Acceso a la Aplicación

- **API**: http://localhost:8000
- **Documentación**: http://localhost:8000/docs

## Persistencia de Datos

La base de datos SQLite se almacena directamente en el archivo `credentials.db` en el directorio raíz del proyecto. El sistema incluye inicialización automática que garantiza:

- **Creación automática**: Si el archivo `credentials.db` no existe, se crea automáticamente al iniciar el contenedor
- **Persistencia completa**: Los datos se mantienen entre reinicios del contenedor
- **Acceso directo**: Se puede acceder directamente al archivo desde el host
- **Backup sencillo**: Es fácil hacer backup y restaurar la base de datos
- **Inicialización de tablas**: Las tablas se crean automáticamente con la estructura correcta

### Acceso a la base de datos desde el host

**Acceso directo al archivo:**
```bash
# El archivo credentials.db está disponible directamente en el directorio del proyecto
ls -la ./credentials.db

# Consultar con sqlite3 desde el host (si está instalado)
sqlite3 ./credentials.db "SELECT COUNT(*) FROM credential_operations;"
```

**Consultar datos desde el contenedor:**
```bash
# Ejecutar consultas SQL desde el contenedor
docker-compose exec faster-ocr-api python -c "
import sqlite3
conn = sqlite3.connect('/app/database/credentials.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM credential_operations')
print('Total operaciones:', cursor.fetchone()[0])
conn.close()
"
```

**Backup y restauración:**
```bash
# Backup simple (copiar archivo)
cp ./credentials.db ./backup-$(date +%Y%m%d).db

# Exportar a SQL desde el contenedor
docker-compose exec faster-ocr-api sqlite3 /app/database/credentials.db .dump > backup.sql

# Restaurar desde backup
cp ./backup-20240101.db ./credentials.db

# Importar desde SQL
cat backup.sql | docker-compose exec -T faster-ocr-api sqlite3 /app/database/credentials.db
```
- **Health Check**: http://localhost:8000/

## Configuración

### Timezone
La aplicación está configurada para usar el timezone `America/Mexico_City`.

### Volúmenes
- `./data:/app/data` - Base de datos y archivos persistentes
- `./docs:/app/docs` - Documentos y archivos de prueba

### Puertos
- Puerto 8000 expuesto para la API FastAPI

### Variables de Entorno
- `OPENAI_API_KEY`: Clave API de OpenAI (requerida)
- `TZ`: Timezone configurado a America/Mexico_City
- `PYTHONPATH`: Configurado a /app
- `PYTHONUNBUFFERED`: Habilitado para logs en tiempo real

## Troubleshooting

### Problemas comunes

1. **Error de permisos:**
   ```bash
   sudo chown -R $USER:$USER data/
   ```

2. **Puerto ocupado:**
   ```bash
   # Cambiar puerto en docker-compose.yml
   ports:
     - "8001:8000"  # Usar puerto 8001 en lugar de 8000
   ```

3. **Problemas con PaddleOCR:**
   ```bash
   # Reconstruir imagen sin caché
   docker-compose build --no-cache
   ```

4. **Ver logs detallados:**
   ```bash
   docker-compose logs --tail=100 faster-ocr-api
   ```

### Health Check
El contenedor incluye un health check que verifica cada 30 segundos si la aplicación responde correctamente.

### Backup de datos
```bash
# Crear backup de la base de datos
docker-compose exec faster-ocr-api cp /app/data/credentials_optimized.db /app/data/backup_$(date +%Y%m%d_%H%M%S).db
```

## Producción

Para entornos de producción, considerar:

1. **Usar un proxy reverso (nginx)**
2. **Configurar SSL/TLS**
3. **Implementar logging centralizado**
4. **Configurar monitoreo y alertas**
5. **Usar secrets para variables sensibles**

## Desarrollo

Para desarrollo local con hot-reload:
```bash
# Montar código fuente como volumen
docker-compose -f docker-compose.dev.yml up
```