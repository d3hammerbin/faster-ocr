#!/usr/bin/env python3
"""
Script de migración para optimizar la base de datos eliminando campos no utilizados.
Este script elimina la base de datos existente y crea una nueva estructura optimizada.
"""

import os
from pathlib import Path
from database import init_database, DB_PATH

def migrate_database():
    """
    Migra la base de datos eliminando la estructura anterior y creando la nueva optimizada.
    """
    print("Iniciando migración de base de datos...")
    
    # Verificar si existe la base de datos anterior
    if DB_PATH.exists():
        backup_path = DB_PATH.with_suffix('.db.backup')
        print(f"Respaldando base de datos anterior: {DB_PATH} -> {backup_path}")
        
        # Si ya existe un backup, eliminarlo
        if backup_path.exists():
            os.remove(backup_path)
        
        # Renombrar la base de datos actual como backup
        os.rename(DB_PATH, backup_path)
        print("Base de datos anterior respaldada exitosamente.")
    else:
        print("No se encontró base de datos anterior.")
    
    # Crear nueva estructura optimizada
    print("Creando nueva estructura de base de datos optimizada...")
    init_database()
    print("Nueva base de datos creada exitosamente.")
    
    print("\n✅ Migración completada exitosamente!")
    print("\nCambios realizados:")
    print("- Eliminados campos del método tradicional (ine_*)")
    print("- Simplificados campos INE a nombres directos (nombre, sexo, curp, etc.)")
    print("- Mantenidos campos de IA y token usage")
    print("- Estructura optimizada para /ine-process-ai únicamente")

if __name__ == "__main__":
    migrate_database()