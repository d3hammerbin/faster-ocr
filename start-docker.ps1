# Script de PowerShell para iniciar Faster OCR API con Docker

Write-Host "🚀 Iniciando Faster OCR API con Docker..." -ForegroundColor Green

# Verificar si existe el archivo .env
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  Archivo .env no encontrado. Creando desde .env.example..." -ForegroundColor Yellow
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "📝 Archivo .env creado. Por favor, configura tu OPENAI_API_KEY antes de continuar." -ForegroundColor Yellow
        Write-Host "💡 Edita el archivo .env y agrega tu clave API de OpenAI." -ForegroundColor Cyan
        exit 1
    } else {
        Write-Host "❌ Archivo .env.example no encontrado." -ForegroundColor Red
        exit 1
    }
}

# Verificar si OPENAI_API_KEY está configurada
$envContent = Get-Content ".env" -ErrorAction SilentlyContinue
if (-not ($envContent -match "OPENAI_API_KEY=sk-")) {
    Write-Host "⚠️  OPENAI_API_KEY no está configurada correctamente en .env" -ForegroundColor Yellow
    Write-Host "💡 Edita el archivo .env y agrega tu clave API de OpenAI." -ForegroundColor Cyan
    exit 1
}

# Crear directorio de datos si no existe
if (-not (Test-Path "data")) {
    Write-Host "📁 Creando directorio de datos..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Path "data" -Force | Out-Null
}

# Verificar si Docker está ejecutándose
try {
    docker info | Out-Null
} catch {
    Write-Host "❌ Docker no está ejecutándose. Por favor, inicia Docker Desktop." -ForegroundColor Red
    exit 1
}

# Verificar si Docker Compose está disponible
try {
    docker-compose --version | Out-Null
} catch {
    Write-Host "❌ Docker Compose no está instalado." -ForegroundColor Red
    exit 1
}

Write-Host "🔨 Construyendo imagen Docker..." -ForegroundColor Cyan
docker-compose build

Write-Host "🚀 Iniciando contenedor..." -ForegroundColor Cyan
docker-compose up -d

Write-Host "⏳ Esperando que la aplicación esté lista..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Verificar si la aplicación está respondiendo
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ ¡Faster OCR API está ejecutándose correctamente!" -ForegroundColor Green
        Write-Host ""
        Write-Host "🌐 Accesos:" -ForegroundColor Cyan
        Write-Host "   API: http://localhost:8000" -ForegroundColor White
        Write-Host "   Documentación: http://localhost:8000/docs" -ForegroundColor White
        Write-Host ""
        Write-Host "📋 Comandos útiles:" -ForegroundColor Cyan
        Write-Host "   Ver logs: docker-compose logs -f" -ForegroundColor White
        Write-Host "   Detener: docker-compose down" -ForegroundColor White
        Write-Host "   Estado: docker-compose ps" -ForegroundColor White
    }
} catch {
    Write-Host "❌ La aplicación no está respondiendo. Verificando logs..." -ForegroundColor Red
    docker-compose logs --tail=20
    exit 1
}