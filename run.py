#!/usr/bin/env python3
# ==============================================================================
# run.py - Script de lanzamiento del proyecto
# ==============================================================================
"""
Script principal para ejecutar el sistema de gestión energética.
Uso:
    python run.py              → Lanza la interfaz gráfica
    python run.py --cli        → Ejecuta en modo línea de comandos
    python run.py --cli --optimize  → CLI con optimización genética
    python run.py --help       → Muestra la ayuda completa
"""

import sys
import os

# Asegurar que el directorio del proyecto está en el path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.main import main

if __name__ == '__main__':
    main()
