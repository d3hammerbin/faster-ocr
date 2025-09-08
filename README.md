# Faster OCR API

API desarrollada con FastAPI para procesamiento rápido de OCR.

## Instalación

1. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## Ejecución

Para ejecutar el servidor de desarrollo:

```bash
uvicorn main:app --reload
```

La API estará disponible en: http://localhost:8000

## Endpoints disponibles

- `GET /` - Endpoint raíz de bienvenida
- `GET /hola` - Endpoint de prueba que retorna un saludo
- `GET /docs` - Documentación interactiva de la API (Swagger UI)
- `GET /redoc` - Documentación alternativa de la API (ReDoc)

## Estructura del proyecto

```
faster-ocr/
├── main.py              # Aplicación principal FastAPI
├── requirements.txt     # Dependencias del proyecto
├── README.md           # Documentación del proyecto
└── docs/               # Documentación adicional
```