import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Ruta de la base de datos
DB_PATH = Path("/app/database/credentials.db")

def init_database():
    """
    Inicializa la base de datos SQLite y crea las tablas necesarias.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Crear tabla principal para operaciones de credenciales (optimizada para /ine-process-ai)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS credential_operations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            
            -- Información básica de la operación
            operation_id TEXT UNIQUE NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            endpoint TEXT NOT NULL,
            side TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            
            -- Información de la imagen
            image_width INTEGER,
            image_height INTEGER,
            image_dimensions TEXT,
            file_size_bytes INTEGER,
            file_size_kb REAL,
            image_format TEXT,
            
            -- Texto extraído
            raw_extracted_text TEXT,
            cleaned_text TEXT,
            best_text TEXT,
            best_method TEXT,
            
            -- Campos INE extraídos con IA
            nombre TEXT,
            fecha_nacimiento TEXT,
            sexo TEXT,
            domicilio TEXT,
            clave_elector TEXT,
            curp TEXT,
            ano_registro TEXT,
            estado TEXT,
            municipio TEXT,
            seccion TEXT,
            localidad TEXT,
            emision TEXT,
            vigencia TEXT,
            tipo_credencial TEXT,
            fields_extracted_count INTEGER DEFAULT 0,
            
            -- Token usage
            input_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER,
            cost_usd REAL,
            cost_mxn REAL,
            final_cost REAL,
            
            -- Resultados de OCR
            paddleocr_confidence REAL,
            paddleocr_available BOOLEAN DEFAULT TRUE,
            paddleocr_text_length INTEGER,
            
            -- Metadatos adicionales
            success BOOLEAN DEFAULT TRUE,
            error_message TEXT,
            processing_time_ms INTEGER
        )
    """)
    
    # Crear índices para optimizar consultas
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON credential_operations(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_endpoint ON credential_operations(endpoint)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_operation_id ON credential_operations(operation_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp_endpoint ON credential_operations(timestamp, endpoint)")
    
    conn.commit()
    conn.close()

def get_connection():
    """
    Obtiene una conexión a la base de datos.
    """
    return sqlite3.connect(DB_PATH)

def save_credential_operation(
    endpoint: str,
    side: str,
    original_filename: str,
    image_info: Dict[str, Any],
    raw_extracted_text: str,
    cleaned_text: Optional[str] = None,
    best_text: Optional[str] = None,
    best_method: Optional[str] = None,
    ine_fields: Optional[Dict[str, str]] = None,
    token_usage: Optional[Dict[str, Any]] = None,
    paddleocr_results: Optional[Dict[str, Any]] = None,
    success: bool = True,
    error_message: Optional[str] = None,
    processing_time_ms: Optional[int] = None
) -> str:
    """
    Guarda una operación de credencial en la base de datos.
    
    Returns:
        str: El operation_id generado para esta operación
    """
    operation_id = str(uuid.uuid4())
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Preparar datos de campos INE
    ine_data = ine_fields or {}
    
    # Preparar datos de token usage
    token_data = token_usage or {}
    
    # Preparar datos de PaddleOCR
    paddle_data = paddleocr_results or {}
    
    cursor.execute("""
        INSERT INTO credential_operations (
            operation_id, endpoint, side, original_filename,
            image_width, image_height, image_dimensions,
            file_size_bytes, file_size_kb, image_format,
            raw_extracted_text, cleaned_text, best_text, best_method,
            nombre, fecha_nacimiento, sexo, domicilio,
            clave_elector, curp, ano_registro, estado,
            municipio, seccion, localidad, emision, vigencia,
            tipo_credencial, fields_extracted_count,
            input_tokens, output_tokens, total_tokens,
            cost_usd, cost_mxn, final_cost,
            paddleocr_confidence, paddleocr_available, paddleocr_text_length,
            success, error_message, processing_time_ms
        ) VALUES (
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?
        )
    """, (
        operation_id, endpoint, side, original_filename,
        image_info.get('width'), image_info.get('height'), image_info.get('dimensions'),
        image_info.get('file_size_bytes'), image_info.get('file_size_kb'), image_info.get('format'),
        raw_extracted_text, cleaned_text, best_text, best_method,
        ine_data.get('NOMBRE'), ine_data.get('FECHA_DE_NACIMIENTO'), ine_data.get('SEXO'), ine_data.get('DOMICILIO'),
        ine_data.get('CLAVE_DE_ELECTOR'), ine_data.get('CURP'), ine_data.get('AÑO_DE_REGISTRO'), ine_data.get('ESTADO'),
        ine_data.get('MUNICIPIO'), ine_data.get('SECCION'), ine_data.get('LOCALIDAD'), ine_data.get('EMISION'), ine_data.get('VIGENCIA'),
        ine_data.get('TIPO_CREDENCIAL'), len(ine_data) if ine_data else 0,
        token_data.get('input_tokens'), token_data.get('output_tokens'), token_data.get('total_tokens'),
        token_data.get('cost_usd'), token_data.get('cost_mxn'), token_data.get('final_cost'),
        paddle_data.get('confidence'), paddle_data.get('available', True), paddle_data.get('text_length'),
        success, error_message, processing_time_ms
    ))
    
    conn.commit()
    conn.close()
    
    return operation_id

def get_operations_by_endpoint(endpoint: str, limit: int = 100):
    """
    Obtiene operaciones por endpoint.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM credential_operations 
        WHERE endpoint = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (endpoint, limit))
    
    columns = [description[0] for description in cursor.description]
    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    conn.close()
    return results

def get_operation_by_id(operation_id: str):
    """
    Obtiene una operación específica por su ID.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM credential_operations 
        WHERE operation_id = ?
    """, (operation_id,))
    
    row = cursor.fetchone()
    if row:
        columns = [description[0] for description in cursor.description]
        result = dict(zip(columns, row))
    else:
        result = None
    
    conn.close()
    return result

def get_statistics():
    """
    Obtiene estadísticas generales de las operaciones.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Estadísticas generales
    cursor.execute("""
        SELECT 
            COUNT(*) as total_operations,
            COUNT(CASE WHEN endpoint = '/ine-process-ai' THEN 1 END) as ai_operations,
            COUNT(CASE WHEN success = 1 THEN 1 END) as successful_operations,
            AVG(processing_time_ms) as avg_processing_time,
            SUM(CASE WHEN cost_mxn IS NOT NULL THEN cost_mxn ELSE 0 END) as total_cost_mxn,
            SUM(CASE WHEN final_cost IS NOT NULL THEN final_cost ELSE 0 END) as total_final_cost
        FROM credential_operations
    """)
    
    row = cursor.fetchone()
    columns = [description[0] for description in cursor.description]
    stats = dict(zip(columns, row))
    
    conn.close()
    return stats

# Inicializar la base de datos al importar el módulo
init_database()