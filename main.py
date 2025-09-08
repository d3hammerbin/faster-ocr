from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import uuid
import os
from pathlib import Path

# Crear instancia de la aplicación FastAPI
app = FastAPI(
    title="Faster OCR API",
    description="API para procesamiento rápido de OCR",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Endpoint raíz de la API"""
    return {"mensaje": "Bienvenido a Faster OCR API"}

@app.get("/hola")
async def hola():
    """Endpoint de prueba que retorna un saludo"""
    return {"mensaje": "¡Hola! La API está funcionando correctamente"}

@app.post("/upload-credential")
async def upload_credential(
    file: UploadFile = File(...),
    side: str = Form(default="front")
):
    """Endpoint para subir imágenes de credenciales"""
    
    # Validar que el parámetro side sea válido
    if side not in ["front", "back"]:
        raise HTTPException(status_code=400, detail="El parámetro 'side' debe ser 'front' o 'back'")
    
    # Validar formato de archivo
    allowed_extensions = [".jpg", ".jpeg", ".png"]
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Formato de archivo no permitido. Solo se aceptan: {', '.join(allowed_extensions)}"
        )
    
    # Crear directorio credentials si no existe
    credentials_dir = Path("./credentials")
    credentials_dir.mkdir(exist_ok=True)
    
    # Generar nombre único con UUID-v4
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = credentials_dir / unique_filename
    
    # Guardar archivo
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        return {
            "mensaje": "Imagen subida exitosamente",
            "filename": unique_filename,
            "side": side,
            "path": str(file_path),
            "original_filename": file.filename
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar el archivo: {str(e)}")