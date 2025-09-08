#!/usr/bin/env python3
"""
QR Extractor Pro - Sistema avanzado de extracción de códigos QR
Soporte para múltiples estrategias de detección y procesamiento masivo
"""

import os
import sys
import json
import argparse
import base64
from datetime import datetime
from typing import Optional, Tuple, Dict, List, Any
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pyzbar import pyzbar
import requests
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class QRExtractorPro:
    """Extractor avanzado de códigos QR con múltiples estrategias"""
    
    def __init__(self, api_key: Optional[str] = None, debug: bool = False):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.debug = debug
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'methods_used': {}
        }
        
    def log_debug(self, message: str) -> None:
        """Registra mensajes de debug si está habilitado"""
        if self.debug:
            print(f"[DEBUG] {message}")
            
    def save_debug_image(self, image: np.ndarray, filename: str) -> None:
        """Guarda imagen de debug si está habilitado"""
        if self.debug:
            debug_dir = Path("debug_regions")
            debug_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(debug_dir / filename), image)
            
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Mejora la imagen para mejor detección de QR"""
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Aplicar filtros de mejora
        enhanced = cv2.GaussianBlur(gray, (3, 3), 0)
        enhanced = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return enhanced
        
    def extract_region_full(self, image: np.ndarray) -> np.ndarray:
        """Extrae la imagen completa"""
        return image
        
    def extract_region_exact(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extrae región exacta del QR (560px-723px, altura completa)"""
        height, width = image.shape[:2]
        
        # Coordenadas exactas basadas en análisis previo
        start_x = max(0, 560)
        end_x = min(width, 723)
        
        if start_x >= end_x:
            return None
            
        region = image[:, start_x:end_x]
        self.save_debug_image(region, "region_exact.png")
        return region
        
    def extract_region_right(self, image: np.ndarray) -> np.ndarray:
        """Extrae región derecha (70% del ancho hacia la derecha)"""
        height, width = image.shape[:2]
        start_x = int(width * 0.7)
        
        region = image[:, start_x:]
        self.save_debug_image(region, "region_right.png")
        return region
        
    def extract_region_right_top(self, image: np.ndarray) -> np.ndarray:
        """Extrae región superior derecha"""
        height, width = image.shape[:2]
        start_x = int(width * 0.7)
        end_y = int(height * 0.5)
        
        region = image[:end_y, start_x:]
        self.save_debug_image(region, "region_right_top.png")
        return region
        
    def extract_region_right_bottom(self, image: np.ndarray) -> np.ndarray:
        """Extrae región inferior derecha"""
        height, width = image.shape[:2]
        start_x = int(width * 0.7)
        start_y = int(height * 0.5)
        
        region = image[start_y:, start_x:]
        self.save_debug_image(region, "region_right_bottom.png")
        return region
        
    def extract_region_center_right(self, image: np.ndarray) -> np.ndarray:
        """Extrae región centro derecha"""
        height, width = image.shape[:2]
        start_x = int(width * 0.6)
        start_y = int(height * 0.25)
        end_y = int(height * 0.75)
        
        region = image[start_y:end_y, start_x:]
        self.save_debug_image(region, "region_center_right.png")
        return region
        
    def read_qr_local(self, image: np.ndarray) -> Optional[str]:
        """Lee QR usando pyzbar localmente"""
        try:
            # Intentar con imagen original
            qr_codes = pyzbar.decode(image)
            
            if qr_codes:
                for qr in qr_codes:
                    qr_data = qr.data.decode('utf-8')
                    if self.is_valid_ine_qr(qr_data):
                        return qr_data
                        
            # Intentar con imagen mejorada
            enhanced = self.enhance_image(image)
            qr_codes = pyzbar.decode(enhanced)
            
            if qr_codes:
                for qr in qr_codes:
                    qr_data = qr.data.decode('utf-8')
                    if self.is_valid_ine_qr(qr_data):
                        return qr_data
                        
        except Exception as e:
            self.log_debug(f"Error en lectura local: {e}")
            
        return None
        
    def is_valid_ine_qr(self, qr_data: str) -> bool:
        """Valida si el QR es válido para INE"""
        return (
            qr_data.startswith('http://qr.ine.mx/') and 
            len(qr_data) > 30 and
            '/P/' in qr_data
        )
        
    def ask_api_qr(self, image: np.ndarray) -> Tuple[Optional[str], int, float]:
        """Consulta API para extraer QR"""
        if not self.api_key:
            return None, 0, 0.0
            
        try:
            # Convertir imagen a base64
            _, buffer = cv2.imencode('.png', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extrae únicamente la URL del código QR de esta imagen. Responde solo con la URL completa, sin texto adicional."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 150
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                tokens = data.get('usage', {}).get('total_tokens', 0)
                cost = tokens * 0.00000525  # Precio por token para gpt-4o-mini
                
                qr_url = data['choices'][0]['message']['content'].strip()
                
                if self.is_valid_ine_qr(qr_url):
                    return qr_url, tokens, cost
                    
            return None, 0, 0.0
            
        except Exception as e:
            self.log_debug(f"Error en API: {e}")
            return None, 0, 0.0
            
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Procesa una imagen con todas las estrategias disponibles"""
        self.log_debug(f"Procesando: {image_path}")
        
        # Cargar imagen
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("No se pudo cargar la imagen")
        except Exception as e:
            return {
                "archivo": os.path.basename(image_path),
                "status": "ERROR",
                "error": str(e),
                "qr_url": "",
                "metodo": "error",
                "tokens": 0,
                "costo": 0.0
            }
            
        # Estrategias de extracción en orden de prioridad
        strategies = [
            ("local_completa", self.extract_region_full),
            ("local_region_exacta", self.extract_region_exact),
            ("local_region_derecha", self.extract_region_right),
            ("local_region_superior_derecha", self.extract_region_right_top),
            ("local_region_inferior_derecha", self.extract_region_right_bottom),
            ("local_region_centro_derecha", self.extract_region_center_right)
        ]
        
        # Intentar cada estrategia
        for method_name, extract_func in strategies:
            self.log_debug(f"Intentando método: {method_name}")
            
            try:
                region = extract_func(image)
                if region is None:
                    continue
                    
                qr_url = self.read_qr_local(region)
                if qr_url:
                    self.log_debug(f"QR encontrado con {method_name}: {qr_url}")
                    return {
                        "archivo": os.path.basename(image_path),
                        "status": "ÉXITO",
                        "qr_url": qr_url,
                        "metodo": method_name,
                        "tokens": 0,
                        "costo": 0.0
                    }
                    
            except Exception as e:
                self.log_debug(f"Error en {method_name}: {e}")
                continue
                
        # Último recurso: API con la mejor región disponible
        self.log_debug("Métodos locales fallaron, usando API...")
        
        # Usar región exacta si está disponible, sino región derecha, sino imagen completa
        best_region = self.extract_region_exact(image)
        if best_region is None:
            best_region = self.extract_region_right(image)
        if best_region is None:
            best_region = image
            
        qr_url, tokens, cost = self.ask_api_qr(best_region)
        
        if qr_url:
            return {
                "archivo": os.path.basename(image_path),
                "status": "ÉXITO",
                "qr_url": qr_url,
                "metodo": "api_fallback",
                "tokens": tokens,
                "costo": cost
            }
        else:
            return {
                "archivo": os.path.basename(image_path),
                "status": "FALLO",
                "qr_url": "",
                "metodo": "ninguno",
                "tokens": tokens,
                "costo": cost
            }
            
    def update_stats(self, result: Dict[str, Any]) -> None:
        """Actualiza estadísticas globales"""
        self.stats['total_processed'] += 1
        
        if result['status'] == 'ÉXITO':
            self.stats['successful'] += 1
        else:
            self.stats['failed'] += 1
            
        self.stats['total_tokens'] += result.get('tokens', 0)
        self.stats['total_cost'] += result.get('costo', 0.0)
        
        method = result.get('metodo', 'unknown')
        self.stats['methods_used'][method] = self.stats['methods_used'].get(method, 0) + 1
        
    def process_directory(self, directory: str, extensions: List[str] = None) -> List[Dict[str, Any]]:
        """Procesa todas las imágenes en un directorio"""
        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg']
            
        results = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            print(f"Error: El directorio {directory} no existe")
            return results
            
        # Buscar archivos de imagen
        image_files = []
        for ext in extensions:
            image_files.extend(directory_path.glob(f"*{ext}"))
            image_files.extend(directory_path.glob(f"*{ext.upper()}"))
            
        if not image_files:
            print(f"No se encontraron imágenes en {directory}")
            return results
            
        print(f"Procesando {len(image_files)} imágenes...")
        
        for i, image_file in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] {image_file.name}")
            result = self.process_image(str(image_file))
            self.update_stats(result)
            results.append(result)
            
            # Mostrar progreso
            if result['status'] == 'ÉXITO':
                print(f"  ✅ {result['metodo']} - {result['qr_url'][:50]}...")
            else:
                print(f"  ❌ {result['metodo']}")
                
        return results
        
    def save_report(self, results: List[Dict[str, Any]], output_file: str = None) -> str:
        """Guarda reporte detallado"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"reporte_qr_pro_{timestamp}.json"
            
        report = {
            "fecha": datetime.now().isoformat(),
            "estadisticas": self.stats,
            "tasa_exito": (self.stats['successful'] / self.stats['total_processed'] * 100) if self.stats['total_processed'] > 0 else 0,
            "resultados": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        return output_file
        
    def print_summary(self) -> None:
        """Imprime resumen de resultados"""
        print("\n" + "="*80)
        print("RESUMEN FINAL")
        print("="*80)
        print(f"📊 Total procesadas: {self.stats['total_processed']}")
        print(f"✅ Exitosas: {self.stats['successful']}")
        print(f"❌ Fallidas: {self.stats['failed']}")
        
        if self.stats['total_processed'] > 0:
            success_rate = self.stats['successful'] / self.stats['total_processed'] * 100
            print(f"📈 Tasa de éxito: {success_rate:.1f}%")
            
        print(f"💰 Tokens usados: {self.stats['total_tokens']}")
        print(f"💰 Costo total: ${self.stats['total_cost']:.4f}")
        
        print("\n📋 Métodos utilizados:")
        for method, count in self.stats['methods_used'].items():
            print(f"   {method}: {count} imágenes")
            
def main():
    parser = argparse.ArgumentParser(
        description="QR Extractor Pro - Sistema avanzado de extracción de códigos QR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s imagen.png                    # Procesar una imagen
  %(prog)s --directory ./imagenes        # Procesar directorio
  %(prog)s --directory . --debug         # Procesar directorio actual con debug
  %(prog)s imagen.png --output reporte.json  # Guardar reporte personalizado
        """
    )
    
    parser.add_argument(
        'input',
        nargs='?',
        help='Archivo de imagen a procesar (opcional si se usa --directory)'
    )
    
    parser.add_argument(
        '--directory', '-d',
        help='Directorio con imágenes a procesar'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Archivo de salida para el reporte JSON'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Habilitar modo debug (guarda imágenes de regiones)'
    )
    
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=['.png', '.jpg', '.jpeg'],
        help='Extensiones de archivo a procesar (default: .png .jpg .jpeg)'
    )
    
    args = parser.parse_args()
    
    # Validar argumentos
    if not args.input and not args.directory:
        parser.error("Debe especificar una imagen o un directorio con --directory")
        
    # Crear extractor
    extractor = QRExtractorPro(debug=args.debug)
    
    results = []
    
    if args.directory:
        # Procesar directorio
        results = extractor.process_directory(args.directory, args.extensions)
    else:
        # Procesar archivo individual
        if not os.path.exists(args.input):
            print(f"Error: El archivo {args.input} no existe")
            sys.exit(1)
            
        result = extractor.process_image(args.input)
        extractor.update_stats(result)
        results = [result]
        
        # Mostrar resultado individual
        print(f"\nResultado para {result['archivo']}:")
        if result['status'] == 'ÉXITO':
            print(f"✅ QR encontrado ({result['metodo']}): {result['qr_url']}")
            print(f"💰 Costo: ${result['costo']:.4f}")
        else:
            print(f"❌ No se pudo extraer QR ({result['metodo']})")
            if result.get('error'):
                print(f"Error: {result['error']}")
                
    # Guardar reporte
    if results:
        report_file = extractor.save_report(results, args.output)
        print(f"\n📄 Reporte guardado en: {report_file}")
        
    # Mostrar resumen
    extractor.print_summary()
    
if __name__ == "__main__":
    main()