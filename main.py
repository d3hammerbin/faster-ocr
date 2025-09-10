from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uuid
import os
from pathlib import Path
from PIL import Image
import io
from dotenv import load_dotenv
import cv2
import numpy as np
import re
from rapidfuzz import fuzz, process
from unidecode import unidecode
from openai import OpenAI
import json
from datetime import datetime
import uuid
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

# Campos específicos de credenciales INE mexicanas
INE_FIELDS = {
    'NOMBRE': ['NOMBRE'],
    'FECHA_DE_NACIMIENTO': ['FECHA DE NACIMIENTO', 'NACIMIENTO'],
    'SEXO': ['SEXO', 'H', 'M'],
    'DOMICILIO': ['DOMICILIO', 'DIRECCION'],
    'CLAVE_DE_ELECTOR': ['CLAVE DE ELECTOR', 'CLAVE ELECTOR'],
    'CURP': ['CURP'],
    'AÑO_DE_REGISTRO': ['AÑO DE REGISTRO', 'REGISTRO'],
    'ESTADO': ['ESTADO'],
    'MUNICIPIO': ['MUNICIPIO'],
    'SECCION': ['SECCION'],
    'LOCALIDAD': ['LOCALIDAD'],
    'EMISION': ['EMISION'],
    'VIGENCIA': ['VIGENCIA']
}

# Texto a ignorar en credenciales INE
IGNORE_TEXT = [
    'MEXICO',
    'INSTITUTO NACIONAL ELECTORAL',
    'INSTITUTONACIONALELECTORAL',
    'CREDENCIAL PARA VOTAR',
    'CREDENCIALPARAVOTAR',
    'ESTADOS UNIDOS MEXICANOS',
    'MEXICO',
    'INE',
    'ELECTORAL'
]

def normalize_text(text: str) -> str:
    """Normaliza texto removiendo acentos y caracteres especiales"""
    if not text:
        return ""
    # Convertir a mayúsculas y remover acentos
    normalized = unidecode(text.upper())
    # Remover caracteres especiales excepto espacios y números
    normalized = re.sub(r'[^A-Z0-9\s]', '', normalized)
    # Limpiar espacios múltiples
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

def clean_extracted_text_advanced(text: str) -> str:
    """Limpia texto extraído por OCR con técnicas avanzadas"""
    if not text:
        return ""
    
    # Normalizar texto
    cleaned = normalize_text(text)
    
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

def extract_ine_fields(ocr_text: str) -> dict:
    """Extrae campos específicos de credenciales INE del texto OCR usando patrones robustos"""
    extracted_fields = {}
    
    # Limpiar y normalizar texto
    cleaned_text = clean_extracted_text_advanced(ocr_text)
    
    # Remover texto de IGNORE_TEXT
    for ignore_text in IGNORE_TEXT:
        cleaned_text = cleaned_text.replace(ignore_text, ' ')
    
    # Normalizar espacios múltiples
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # Patrones específicos para cada campo
    patterns = {
        'NOMBRE': {
            'pattern': r'H\s+([A-ZÁÉÍÓÚÑ\s]+?)\s+DOMICILIO',
            'fallback': r'SEXO\s+H\s+([A-ZÁÉÍÓÚÑ\s]+?)\s+DOMICILIO'
        },
        'SEXO': {
            'pattern': r'SEXO\s+([HM])',
            'fallback': r'\b([HM])\b(?=.*DOMICILIO)'
        },
        'DOMICILIO': {
            'pattern': r'DOMICILIO\s+([A-ZÁÉÍÓÚÑ0-9\s,]+?)(?=\s+CLAVEDEELECTOR)',
            'fallback': r'DOMICILIO\s+([A-ZÁÉÍÓÚÑ0-9\s,]+?)(?=\s+[A-Z]{6}\d{8})'
        },
        'CLAVE_DE_ELECTOR': {
            'pattern': r'CLAVEDEELECTOR([A-Z0-9]{18})',
            'fallback': r'([A-Z]{6}\d{8}[HM]\d{3})'
        },
        'CURP': {
            'pattern': r'CURP\s+([A-Z]{4}\d{6}[HM][A-Z]{5}\d{2})',
            'fallback': r'([A-Z]{4}\d{6}[HM][A-Z]{5}\d{2})'
        },
        'AÑO_DE_REGISTRO': {
            'pattern': r'ANODEREGISTRO\s+(\d{6})',
            'fallback': r'\b(\d{6})\b(?=\s+FECHADENACIMIENTO)'
        },
        'FECHA_DE_NACIMIENTO': {
            'pattern': r'FECHADENACIMIENTO\s+SECCION\s+VIGENCIA\s+(\d{8})\s+\d{4}\s+\d{8}',
            'fallback': r'\b(\d{8})\b(?=\s+\d{4}\s+\d{8}$)'
        },
        'SECCION': {
            'pattern': r'FECHADENACIMIENTO\s+SECCION\s+VIGENCIA\s+\d{8}\s+(\d{4})\s+\d{8}',
            'fallback': r'\b\d{8}\s+(\d{4})\s+\d{8}$'
        },
        'VIGENCIA': {
            'pattern': r'FECHADENACIMIENTO\s+SECCION\s+VIGENCIA\s+\d{8}\s+\d{4}\s+(\d{8})',
            'fallback': r'\b\d{8}\s+\d{4}\s+(\d{8})$'
        }
    }
    
    # Extraer cada campo usando los patrones
    for field_name, config in patterns.items():
        field_value = None
        
        # Intentar patrón principal
        match = re.search(config['pattern'], cleaned_text, re.IGNORECASE)
        if match:
            field_value = match.group(1).strip()
        else:
            # Intentar patrón de respaldo
            if 'fallback' in config:
                match = re.search(config['fallback'], cleaned_text, re.IGNORECASE)
                if match:
                    field_value = match.group(1).strip()
        
        # Limpiar y validar el valor extraído
        if field_value:
            # Remover espacios extra y caracteres no deseados
            field_value = re.sub(r'\s+', ' ', field_value).strip()
            
            # Validaciones específicas por campo
            if field_name == 'NOMBRE' and len(field_value) > 5 and 'SEXO' not in field_value:
                extracted_fields[field_name] = field_value
            elif field_name == 'SEXO' and field_value in ['H', 'M']:
                extracted_fields[field_name] = field_value
            elif field_name == 'DOMICILIO' and len(field_value) > 10:
                extracted_fields[field_name] = field_value
            elif field_name == 'CLAVE DE ELECTOR' and len(field_value) == 18:
                extracted_fields[field_name] = field_value
            elif field_name == 'CURP' and len(field_value) == 18:
                extracted_fields[field_name] = field_value
            elif field_name == 'AÑO DE REGISTRO' and len(field_value) == 6:
                extracted_fields[field_name] = field_value
            elif field_name == 'FECHA DE NACIMIENTO' and len(field_value) == 8:
                # Formatear fecha de DDMMAAAA a DD/MM/AAAA
                formatted_date = f"{field_value[:2]}/{field_value[2:4]}/{field_value[4:]}"
                extracted_fields[field_name] = formatted_date
            elif field_name == 'SECCION' and len(field_value) == 4:
                extracted_fields[field_name] = field_value
            elif field_name == 'VIGENCIA' and len(field_value) == 8:
                # Formatear vigencia de AAAAAAAA a AAAA-AAAA
                formatted_vigencia = f"{field_value[:4]}-{field_value[4:]}"
                extracted_fields[field_name] = formatted_vigencia
    
    return extracted_fields

def fuzzy_match_correction(text: str, reference_list: list, threshold: int = 80) -> str:
    """Corrige texto usando fuzzy matching contra una lista de referencia"""
    if not text or not reference_list:
        return text
    
    match = process.extractOne(text, reference_list, scorer=fuzz.ratio)
    if match and match[1] >= threshold:
        return match[0]
    
    return text

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

def extract_ine_fields_with_ai(text: str) -> dict:
    """Extrae campos de credencial INE usando OpenAI GPT-4o-mini"""
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
  "ANO_DE_REGISTRO": "valor o null",
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
            return filtered_fields
        except json.JSONDecodeError:
            print(f"Error parsing JSON from OpenAI response: {content}")
            return {}
            
    except Exception as e:
        print(f"Error en extracción con IA: {str(e)}")
        return {}

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

@app.post("/ine-process")
async def ine_process(
    file: UploadFile = File(...),
    side: str = Form(default="front")
):
    """
    Procesa una imagen de credencial INE usando PaddleOCR con normalización y extracción de campos específicos.
    
    Args:
        file: Archivo de imagen a procesar
        side: Lado de la credencial ("front" o "back", por defecto "front")
    
    Returns:
        JSON con texto extraído, campos específicos de INE, información del archivo y metadatos
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
            detail=f"Formato de archivo no permitido. Solo se aceptan: {', '.join(allowed_extensions)}"
        )
    
    # Crear directorio credentials si no existe
    credentials_dir = Path("./credentials")
    credentials_dir.mkdir(exist_ok=True)
    
    # Generar nombre único con UUID-v4
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = credentials_dir / unique_filename
    
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        
        # Obtener información de la imagen
        width, height = image.size
        file_size_bytes = len(content)
        file_size_kb = round(file_size_bytes / 1024, 2)
        
        # Preprocesar imagen
        def preprocess_for_comparison(img):
            import cv2
            import numpy as np
            
            img_array = np.array(img)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Redimensionar y mejorar
            gray = cv2.resize(gray, (790, 500), interpolation=cv2.INTER_CUBIC)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            return gray
        
        processed_array = preprocess_for_comparison(image)
        
        # Extraer texto usando PaddleOCR
        raw_extracted_text = extract_text_hybrid_ocr(processed_array)
        
        # Aplicar normalización y limpieza avanzada
        cleaned_text = clean_extracted_text_advanced(raw_extracted_text)
        
        # Extraer campos específicos de credencial INE
        ine_fields = extract_ine_fields(raw_extracted_text)
        
        # Probar PaddleOCR únicamente
        results = {}
        
        # PaddleOCR
        paddle_text, paddle_conf, paddle_method = extract_text_with_paddleocr(processed_array)
        results["paddleocr"] = {
            "text": paddle_text,
            "confidence": paddle_conf,
            "method": paddle_method,
            "available": PADDLEOCR_AVAILABLE,
            "text_length": len(paddle_text.strip()) if paddle_text else 0
        }
        
        # Guardar archivo
        with open(file_path, "wb") as f:
            f.write(content)
        
        return {
            "success": True,
            "mensaje": "Credencial INE procesada exitosamente con normalización y extracción de campos específicos",
            "filename": unique_filename,
            "side": side,
            "path": str(file_path),
            "original_filename": file.filename,
            "raw_extracted_text": raw_extracted_text,
            "cleaned_text": cleaned_text,
            "ine_fields": ine_fields,
            "fields_extracted_count": len(ine_fields),
            "image_info": {
                "width": width,
                "height": height,
                "dimensions": f"{width}x{height}",
                "file_size_bytes": file_size_bytes,
                "file_size_kb": file_size_kb,
                "format": image.format
            },
            "results": results,
            "best_method": "paddleocr",
            "best_text": paddle_text,
            "extracted_text": raw_extracted_text,
            "comparison_summary": {
                "total_methods": 1,
                "available_methods": 1 if PADDLEOCR_AVAILABLE else 0,
                "methods_with_text": 1 if paddle_text and len(paddle_text.strip()) > 0 else 0
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar credencial INE: {str(e)}")

@app.post("/validate-ine-field")
async def validate_ine_field(
    field_name: str = Form(...),
    field_value: str = Form(...),
    reference_values: str = Form(default="")
):
    """
    Valida un campo específico de credencial INE usando fuzzy matching.
    
    Args:
        field_name: Nombre del campo a validar
        field_value: Valor del campo extraído
        reference_values: Valores de referencia separados por comas (opcional)
    
    Returns:
        JSON con resultado de validación y correcciones sugeridas
    """
    try:
        # Normalizar el valor del campo
        normalized_value = clean_extracted_text_advanced(field_value)
        
        # Verificar si el campo es válido
        if field_name not in INE_FIELDS:
            available_fields = list(INE_FIELDS.keys())
            return JSONResponse(content={
                "success": False,
                "error": f"Campo '{field_name}' no válido",
                "available_fields": available_fields
            })
        
        # Procesar valores de referencia si se proporcionan
        reference_list = []
        if reference_values:
            reference_list = [ref.strip() for ref in reference_values.split(",") if ref.strip()]
        
        # Aplicar corrección con fuzzy matching si hay referencias
        corrected_value = normalized_value
        similarity_score = 100
        
        if reference_list:
            corrected_value = fuzzy_match_correction(normalized_value, reference_list)
            # Calcular similitud con el mejor match
            best_match = process.extractOne(normalized_value, reference_list, scorer=fuzz.ratio)
            if best_match:
                similarity_score = best_match[1]
        
        return JSONResponse(content={
            "success": True,
            "field_name": field_name,
            "original_value": field_value,
            "normalized_value": normalized_value,
            "corrected_value": corrected_value,
            "similarity_score": similarity_score,
            "is_valid": similarity_score >= 80,
            "reference_values": reference_list,
            "message": "Campo validado exitosamente"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validando campo INE: {str(e)}")

@app.post("/text-similarity")
async def calculate_text_similarity(
    text1: str = Form(...),
    text2: str = Form(...),
    normalize: bool = Form(default=True)
):
    """
    Calcula la similitud entre dos textos usando fuzzy matching.
    
    Args:
        text1: Primer texto a comparar
        text2: Segundo texto a comparar
        normalize: Si aplicar normalización a los textos
    
    Returns:
        JSON con puntuaciones de similitud usando diferentes algoritmos
    """
    try:
        # Aplicar normalización si se solicita
        processed_text1 = clean_extracted_text_advanced(text1) if normalize else text1
        processed_text2 = clean_extracted_text_advanced(text2) if normalize else text2
        
        # Calcular diferentes tipos de similitud
        ratio_score = fuzz.ratio(processed_text1, processed_text2)
        partial_ratio_score = fuzz.partial_ratio(processed_text1, processed_text2)
        token_sort_score = fuzz.token_sort_ratio(processed_text1, processed_text2)
        token_set_score = fuzz.token_set_ratio(processed_text1, processed_text2)
        
        # Calcular promedio ponderado
        weighted_average = (
            ratio_score * 0.3 +
            partial_ratio_score * 0.2 +
            token_sort_score * 0.25 +
            token_set_score * 0.25
        )
        
        return JSONResponse(content={
            "success": True,
            "text1": {
                "original": text1,
                "processed": processed_text1
            },
            "text2": {
                "original": text2,
                "processed": processed_text2
            },
            "similarity_scores": {
                "ratio": ratio_score,
                "partial_ratio": partial_ratio_score,
                "token_sort_ratio": token_sort_score,
                "token_set_ratio": token_set_score,
                "weighted_average": round(weighted_average, 2)
            },
            "is_similar": weighted_average >= 80,
            "normalization_applied": normalize,
            "message": "Similitud calculada exitosamente"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculando similitud: {str(e)}")

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
        # Leer contenido del archivo
        content = await file.read()
        file_size_bytes = len(content)
        file_size_kb = round(file_size_bytes / 1024, 2)
        
        # Crear directorio credentials si no existe
        credentials_dir = Path("./credentials")
        credentials_dir.mkdir(exist_ok=True)
        
        # Crear nombre único para el archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"ine_ai_{side}_{timestamp}_{file.filename}"
        file_path = credentials_dir / unique_filename
        
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
        ine_fields_ai = extract_ine_fields_with_ai(cleaned_text)
        
        # Detectar tipo de credencial INE
        tipo_credencial = detect_ine_type(raw_extracted_text, ine_fields_ai)
        
        # Agregar el tipo detectado al objeto ine_fields_ai
        ine_fields_ai["TIPO_CREDENCIAL"] = tipo_credencial
        
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
        
        # Guardar archivo
        with open(file_path, "wb") as f:
            f.write(content)
        
        return {
            "success": True,
            "mensaje": "Credencial INE procesada exitosamente con IA (GPT-4o-mini) para extracción inteligente de campos",
            "filename": unique_filename,
            "side": side,
            "path": str(file_path),
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
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar credencial INE con IA: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)