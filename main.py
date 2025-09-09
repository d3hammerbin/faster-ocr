from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import uuid
import os
from pathlib import Path
from PIL import Image
import io

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
    
    # Guardar archivo y obtener información de la imagen
    try:
        content = await file.read()
        
        # Obtener dimensiones de la imagen
        image = Image.open(io.BytesIO(content))
        width, height = image.size
        file_size_bytes = len(content)
        file_size_kb = round(file_size_bytes / 1024, 2)
        
        # Guardar archivo
        with open(file_path, "wb") as f:
            f.write(content)
        
        return {
            "mensaje": "Imagen subida exitosamente",
            "filename": unique_filename,
            "side": side,
            "path": str(file_path),
            "original_filename": file.filename,
            "image_info": {
                "width": width,
                "height": height,
                "dimensions": f"{width}x{height}",
                "file_size_bytes": file_size_bytes,
                "file_size_kb": file_size_kb,
                "format": image.format
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")