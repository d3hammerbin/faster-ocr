from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os
from pathlib import Path
from PIL import Image
import io
from dotenv import load_dotenv
import cv2
import numpy as np
import re
import sqlite3

from unidecode import unidecode
from openai import OpenAI
import json
from datetime import datetime
import uuid
import time
from database import save_credential_operation, get_operations_by_endpoint, get_operation_by_id, get_statistics, init_database
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("PaddleOCR no está disponible. Instala con: pip install paddleocr")

# Cargar variables de entorno
load_dotenv()

# Inicializar OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Inicializar PaddleOCR
if PADDLEOCR_AVAILABLE:
    paddle_ocr = PaddleOCR(use_angle_cls=True, lang='es', use_gpu=False)
else:
    paddle_ocr = None

def clean_extracted_text_advanced(text: str) -> str:
    """Limpia texto extraído por OCR con técnicas avanzadas"""
    if not text:
        return ""
    
    # Normalizar texto
    # Convertir a mayúsculas y remover acentos
    cleaned = unidecode(text.upper())
    # Remover caracteres especiales excepto espacios y números
    cleaned = re.sub(r'[^A-Z0-9\s]', '', cleaned)
    # Limpiar espacios múltiples
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Correcciones específicas para OCR de credenciales mexicanas
    corrections = {
        'MEXIC0': 'MEXICO',
        'NACI0NAL': 'NACIONAL',
        'ELECT0RAL': 'ELECTORAL',
        'CREDENCI4L': 'CREDENCIAL',
        'V0TAR': 'VOTAR',
        'EST4DOS': 'ESTADOS',
        'UNID0S': 'UNIDOS',
        'MEXIC4NOS': 'MEXICANOS',
        'N0MBRE': 'NOMBRE',
        'FECH4': 'FECHA',
        'NACIMIENT0': 'NACIMIENTO',
        'D0MICILIO': 'DOMICILIO',
        'CL4VE': 'CLAVE',
        'ELECT0R': 'ELECTOR',
        'REGISTR0': 'REGISTRO',
        'ESTAD0': 'ESTADO',
        'MUNICIPI0': 'MUNICIPIO',
        'SECCI0N': 'SECCION',
        'L0CALIDAD': 'LOCALIDAD',
        'EMISI0N': 'EMISION',
        'VIGENCI4': 'VIGENCIA'
    }
    
    for wrong, correct in corrections.items():
        cleaned = cleaned.replace(wrong, correct)
    
    return cleaned

def detect_ine_type(text: str, extracted_fields: dict) -> str:
    """Detecta el tipo de credencial INE basado en características específicas"""
    text_upper = text.upper()
    
    # Extraer año de vigencia
    vigencia = extracted_fields.get('VIGENCIA', '')
    vigencia_year = None
    
    if vigencia:
        try:
            # Si tiene formato YYYY-YYYY, tomar los primeros 4 dígitos
            if '-' in vigencia:
                vigencia_year = int(vigencia.split('-')[0])
            else:
                # Si solo tiene YYYY, tomar el valor completo
                vigencia_year = int(vigencia[:4]) if len(vigencia) >= 4 else None
        except ValueError:
            pass
    
    # Verificar si es del extranjero (F o H)
    if "EXTRANJERO" in text_upper or "DESDE EL EXTRANJERO" in text_upper:
        # Determinar si es F (2016+) o H (2019+) basado en vigencia
        if vigencia_year:
            if vigencia_year >= 2019:
                return 'H'  # Extranjero desde diciembre 2019
            elif vigencia_year >= 2016:
                return 'F'  # Extranjero desde 2016
        return 'F'  # Por defecto extranjero
    
    # Verificar presencia de ESTADO, MUNICIPIO, LOCALIDAD para distinguir E vs G
    has_geographic_data = any(field in text_upper for field in ['ESTADO', 'MUNICIPIO', 'LOCALIDAD'])
    
    # Analizar formato de vigencia
    vigencia_format_g = '-' in vigencia  # Formato YYYY-YYYY para tipo G
    
    if has_geographic_data and not vigencia_format_g:
        return 'E'  # Emitidas desde julio 2014, contienen datos geográficos, vigencia YYYY
    elif not has_geographic_data and vigencia_format_g:
        return 'G'  # Emitidas desde diciembre 2019, sin datos geográficos, vigencia YYYY-YYYY
    
    # Determinar por año de vigencia como fallback
    if vigencia_year:
        if vigencia_year >= 2019:
            return 'G'  # Más probable que sea G si vigencia es 2019+
        elif vigencia_year >= 2014:
            return 'E'  # Más probable que sea E si vigencia es 2014+
    
    return 'E'  # Por defecto tipo E

def extract_ine_fields_with_ai(text: str) -> tuple:
    """Extrae campos de credencial INE usando OpenAI GPT-4o-mini
    
    Returns:
        tuple: (extracted_fields, token_usage_info)
    """
    try:
        prompt = f"""
Analiza el siguiente texto extraído de una credencial INE mexicana y extrae los campos específicos.
Devuelve ÚNICAMENTE un JSON válido con los campos encontrados.

Texto OCR: {text}

Campos a extraer (si están presentes):
- NOMBRE: Nombre completo de la persona
- SEXO: H o M
- DOMICILIO: Dirección completa
- CLAVE_DE_ELECTOR: Código de 18 caracteres
- CURP: Clave Única de Registro de Población (18 caracteres)
- AÑO_DE_REGISTRO: Año y código adicional en formato 'YYYY XX' (4 dígitos del año + espacio + 2 dígitos adicionales)
- FECHA_DE_NACIMIENTO: En formato DD/MM/AAAA
- SECCION: Número de sección electoral (4 dígitos)
- VIGENCIA: Período de vigencia (AAAA para tipos E, AAAA-AAAA para tipos G)
- ESTADO: Estado (solo si está presente)
- MUNICIPIO: Municipio (solo si está presente)
- LOCALIDAD: Localidad (solo si está presente)

Respuesta esperada (JSON):
{{
  "NOMBRE": "valor o null",
  "SEXO": "valor o null",
  "DOMICILIO": "valor o null",
  "CLAVE_DE_ELECTOR": "valor o null",
  "CURP": "valor o null",
  "AÑO_DE_REGISTRO": "valor o null",
  "FECHA_DE_NACIMIENTO": "valor o null",
  "SECCION": "valor o null",
  "VIGENCIA": "valor o null",
  "ESTADO": "valor o null",
  "MUNICIPIO": "valor o null",
  "LOCALIDAD": "valor o null"
}}
"""
        
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
            messages=[
                {"role": "system", "content": "Eres un experto en análisis de documentos oficiales mexicanos. Extrae información de credenciales INE con precisión."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=int(os.getenv('OPENAI_MAX_TOKENS', 4000)),
            temperature=float(os.getenv('OPENAI_TEMPERATURE', 0.1))
        )
        
        # Extraer información de tokens
        token_usage = response.usage
        token_info = {
            "input_tokens": token_usage.prompt_tokens,
            "output_tokens": token_usage.completion_tokens,
            "total_tokens": token_usage.total_tokens
        }
        
        # Extraer el contenido de la respuesta
        content = response.choices[0].message.content.strip()
        
        # Limpiar formato markdown si está presente
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
        
        # Intentar parsear como JSON
        try:
            extracted_fields = json.loads(content)
            # Filtrar campos nulos
            filtered_fields = {k: v for k, v in extracted_fields.items() if v is not None and v != "null" and v.strip() != ""}
            return filtered_fields, token_info
        except json.JSONDecodeError:
            print(f"Error parsing JSON from OpenAI response: {content}")
            return {}, token_info
            
    except Exception as e:
        print(f"Error en extracción con IA: {str(e)}")
        return {}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

# Crear instancia de la aplicación FastAPI
app = FastAPI(
    title="Faster OCR API",
    description="API para procesamiento rápido de OCR",
    version="1.0.0"
)

# Configurar CORS para permitir acceso desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos HTTP
    allow_headers=["*"],  # Permite todos los headers
)

# Inicializar la base de datos al arrancar la aplicación
init_database()

@app.get("/")
async def root():
    """Endpoint raíz de la API"""
    return {"mensaje": "Bienvenido a Faster OCR API"}

def clean_extracted_text(text):
    """Función auxiliar para limpiar texto extraído por EasyOCR"""
    import re
    
    if not text:
        return ""
    
    # Limpiar caracteres extraños pero mantener acentos españoles
    clean_text = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ0-9\s.,;:()\-/]', '', text)
    
    # Limpiar espacios múltiples
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    return clean_text.strip()

def extract_text_with_paddleocr(img_array):
    """Extrae texto usando PaddleOCR"""
    if not PADDLEOCR_AVAILABLE or paddle_ocr is None:
        return "", 0, "paddleocr_unavailable"
    
    try:
        # PaddleOCR espera imagen en formato BGR o RGB
        if len(img_array.shape) == 2:
            # Convertir escala de grises a RGB
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img_array
        
        # Ejecutar PaddleOCR
        result = paddle_ocr.ocr(img_rgb, cls=True)
        
        if result and result[0]:
            texts = []
            confidences = []
            
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]  # Texto extraído
                    confidence = line[1][1]  # Confianza
                    
                    if confidence > 0.3:  # Umbral de confianza
                        texts.append(text)
                        confidences.append(confidence)
            
            if texts:
                combined_text = ' '.join(texts)
                avg_confidence = (sum(confidences) / len(confidences)) * 100
                cleaned_text = clean_extracted_text(combined_text)
                return cleaned_text, avg_confidence, "paddleocr"
        
        return "", 0, "paddleocr_no_text"
        
    except Exception as e:
        print(f"Error con PaddleOCR: {e}")
        return "", 0, "paddleocr_error"

def extract_text_hybrid_ocr(img_array):
    """
    Extrae texto usando PaddleOCR únicamente
    """
    # Usar PaddleOCR como método principal
    paddle_text, paddle_conf, paddle_method = extract_text_with_paddleocr(img_array)
    
    if paddle_text and len(paddle_text.strip()) > 2:
        print(f"Método usado: PaddleOCR, Confianza: {paddle_conf:.1f}%")
        return paddle_text.strip()
    
    print("No se pudo extraer texto con PaddleOCR")
    return ""

@app.post("/ine-process-ai")
async def ine_process_ai(
    file: UploadFile = File(...),
    side: str = Form(default="front")
):
    """
    Procesa una imagen de credencial INE usando PaddleOCR + OpenAI GPT-4o-mini para extracción inteligente de campos.
    
    Args:
        file: Archivo de imagen a procesar
        side: Lado de la credencial ("front" o "back", por defecto "front")
    
    Returns:
        JSON con texto extraído, campos específicos de INE extraídos con IA, información del archivo y metadatos
    """
    # Validar que el parámetro side sea válido
    if side not in ["front", "back"]:
        raise HTTPException(status_code=400, detail="El parámetro 'side' debe ser 'front' o 'back'")
    
    # Validar formato de archivo
    allowed_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Formato de archivo no soportado. Formatos permitidos: {', '.join(allowed_extensions)}"
        )
    
    try:
        start_time = time.time()
        # Leer contenido del archivo
        content = await file.read()
        file_size_bytes = len(content)
        file_size_kb = round(file_size_bytes / 1024, 2)
        
        # Procesar imagen directamente desde memoria (sin guardar en disco)
        
        # Abrir imagen con PIL
        image = Image.open(io.BytesIO(content))
        width, height = image.size
        
        # Convertir a array numpy para PaddleOCR
        img_array = np.array(image)
        
        # Extraer texto usando PaddleOCR
        raw_extracted_text = extract_text_hybrid_ocr(img_array)
        
        if not raw_extracted_text or len(raw_extracted_text.strip()) < 5:
            raise HTTPException(
                status_code=422, 
                detail="No se pudo extraer texto suficiente de la imagen. Verifique la calidad de la imagen."
            )
        
        # Limpiar texto extraído
        cleaned_text = clean_extracted_text_advanced(raw_extracted_text)
        
        # Extraer campos INE usando IA
        ine_fields_ai, token_info = extract_ine_fields_with_ai(cleaned_text)
        
        # Detectar tipo de credencial INE
        tipo_credencial = detect_ine_type(raw_extracted_text, ine_fields_ai)
        
        # Agregar el tipo detectado al objeto ine_fields_ai
        ine_fields_ai["TIPO_CREDENCIAL"] = tipo_credencial
        
        # Calcular costos de tokens
        # Precio por token para gpt-4o-mini (según OpenAI pricing)
        input_cost_per_token = 0.00000015  # $0.15 per 1M input tokens
        output_cost_per_token = 0.0000006   # $0.60 per 1M output tokens
        
        input_cost_usd = token_info["input_tokens"] * input_cost_per_token
        output_cost_usd = token_info["output_tokens"] * output_cost_per_token
        total_cost_usd = input_cost_usd + output_cost_usd
        
        # Obtener tasa de cambio USD a MXN del .env
        usd_mxn_rate = float(os.getenv('USD_MXN', 20))
        total_cost_mxn = total_cost_usd * usd_mxn_rate
        
        # Obtener multiplicador del .env y calcular costo final
        multiplier = float(os.getenv('MULTIPLIER', 300))
        final_cost = total_cost_mxn * multiplier
        
        # Obtener información detallada de PaddleOCR
        results = {}
        paddle_text, paddle_conf, paddle_method = extract_text_with_paddleocr(img_array)
        
        results["paddleocr"] = {
            "text": paddle_text,
            "confidence": paddle_conf,
            "method": paddle_method,
            "available": PADDLEOCR_AVAILABLE,
            "text_length": len(paddle_text.strip()) if paddle_text else 0
        }
        
        # Calcular tiempo de procesamiento
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Preparar datos para guardar en base de datos
        image_info = {
            "width": width,
            "height": height,
            "dimensions": f"{width}x{height}",
            "file_size_bytes": file_size_bytes,
            "file_size_kb": file_size_kb,
            "format": image.format
        }
        
        token_usage_data = {
            "input_tokens": token_info["input_tokens"],
            "output_tokens": token_info["output_tokens"],
            "total_tokens": token_info["total_tokens"],
            "cost_usd": round(total_cost_usd, 6),
            "cost_mxn": round(total_cost_mxn, 4),
            "final_cost": round(final_cost, 2)
        }
        
        # Guardar en base de datos
        operation_id = save_credential_operation(
            endpoint="/ine-process-ai",
            side=side,
            original_filename=file.filename,
            image_info=image_info,
            raw_extracted_text=raw_extracted_text,
            cleaned_text=cleaned_text,
            best_text=raw_extracted_text,
            best_method="paddleocr",
            ine_fields=ine_fields_ai,
            token_usage=token_usage_data,
            paddleocr_results={
                "confidence": 95.0,  # Valor por defecto para PaddleOCR
                "available": True,
                "text_length": len(raw_extracted_text)
            },
            success=True,
            processing_time_ms=processing_time_ms
        )
        
        return {
            "success": True,
            "mensaje": "Credencial INE procesada exitosamente con IA",
            "operation_id": operation_id,
            "side": side,
            "original_filename": file.filename,
            "raw_extracted_text": raw_extracted_text,
            "ine_fields_ai": ine_fields_ai,
            "fields_extracted_count": len(ine_fields_ai),
            "image_info": {
                "width": width,
                "height": height,
                "dimensions": f"{width}x{height}",
                "file_size_bytes": file_size_bytes,
                "file_size_kb": file_size_kb,
                "format": image.format
            },
            "processing_time_ms": processing_time_ms
        }
    
    except Exception as e:
        # Guardar error en base de datos
        try:
            processing_time_ms = int((time.time() - start_time) * 1000) if 'start_time' in locals() else None
            save_credential_operation(
                endpoint="/ine-process-ai",
                side=side,
                original_filename=file.filename if file else "unknown",
                image_info={},
                raw_extracted_text="",
                success=False,
                error_message=str(e),
                processing_time_ms=processing_time_ms
            )
        except:
            pass  # Si falla el guardado del error, no interrumpir la respuesta
        
        raise HTTPException(status_code=500, detail=f"Error al procesar credencial INE con IA: {str(e)}")

@app.get("/operations")
async def get_operations(
    endpoint: str = None,
    limit: int = 10
):
    """
    Obtiene las operaciones guardadas en la base de datos.
    
    Args:
        endpoint: Filtrar por endpoint específico (/ine-process-ai)
        limit: Número máximo de resultados (por defecto 10)
    
    Returns:
        JSON con las operaciones encontradas
    """
    try:
        if endpoint:
            operations = get_operations_by_endpoint(endpoint, limit)
        else:
            # Solo obtener operaciones de /ine-process-ai
            operations = get_operations_by_endpoint("/ine-process-ai", limit)
        
        return {
            "success": True,
            "total_operations": len(operations),
            "operations": operations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener operaciones: {str(e)}")

@app.get("/operations/{operation_id}")
async def get_operation_detail(operation_id: str):
    """
    Obtiene los detalles de una operación específica.
    
    Args:
        operation_id: ID único de la operación
    
    Returns:
        JSON con los detalles de la operación
    """
    try:
        operation = get_operation_by_id(operation_id)
        if not operation:
            raise HTTPException(status_code=404, detail="Operación no encontrada")
        
        return {
            "success": True,
            "operation": operation
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener operación: {str(e)}")

@app.get("/consumed")
async def get_consumed_resources():
    """Obtiene el resumen de recursos consumidos según las peticiones de extracción realizadas"""
    try:
        conn = sqlite3.connect("/app/database/credentials.db")
        cursor = conn.cursor()
        
        # Calcular sumas de tokens y costos
        cursor.execute("""
            SELECT 
                COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                COALESCE(SUM(output_tokens), 0) as total_output_tokens,
                COALESCE(SUM(final_cost), 0) as total_final_cost,
                COUNT(*) as total_records
            FROM credential_operations
        """)
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            "success": True,
            "data": {
                "total_input_tokens": result[0],
                "total_output_tokens": result[1], 
                "total_final_cost": round(result[2], 4) if result[2] else 0,
                "total_records": result[3]
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/summary", include_in_schema=False)
async def get_summary(
    x_card: str = Header(None, alias="X-Card"),
    page: int = Query(1, ge=1, description="Número de página"),
    limit: int = Query(10, ge=1, le=100, description="Registros por página")
):
    """Obtiene toda la información de la tabla credential_operations con paginación"""
    # Validar header requerido
    if x_card != "Nabudoconosor":
        raise HTTPException(status_code=401, detail="Header X-Card requerido con valor válido")
    
    try:
        conn = sqlite3.connect("/app/database/credentials.db")
        cursor = conn.cursor()
        
        # Calcular offset para paginación
        offset = (page - 1) * limit
        
        # Obtener total de registros
        cursor.execute("SELECT COUNT(*) FROM credential_operations")
        total_records = cursor.fetchone()[0]
        
        # Obtener registros paginados
        cursor.execute("""
            SELECT * FROM credential_operations 
            ORDER BY timestamp DESC 
            LIMIT ? OFFSET ?
        """, (limit, offset))
        
        columns = [description[0] for description in cursor.description]
        records = cursor.fetchall()
        conn.close()
        
        # Convertir registros a diccionarios
        data = []
        for record in records:
            record_dict = {}
            for i, value in enumerate(record):
                record_dict[columns[i]] = value
            data.append(record_dict)
        
        # Calcular información de paginación
        total_pages = (total_records + limit - 1) // limit
        
        return {
            "success": True,
            "data": data,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "total_records": total_records,
                "records_per_page": limit,
                "has_next": page < total_pages,
                "has_previous": page > 1
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/statistics")
async def get_operation_statistics():
    """
    Obtiene estadísticas generales de las operaciones.
    
    Returns:
        JSON con estadísticas de uso
    """
    try:
        stats = get_statistics()
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener estadísticas: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)