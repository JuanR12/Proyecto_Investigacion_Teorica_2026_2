"""
config.py — Configuración centralizada de rutas del proyecto.

╔══════════════════════════════════════════════════════════════════════╗
║  INSTRUCCIONES DE USO                                                ║
║  1. Por defecto, las rutas se calculan automáticamente desde la      ║
║     ubicación de este archivo. No necesitas cambiar nada si el       ║
║     proyecto mantiene su estructura original de carpetas.            ║
║                                                                      ║
║  2. Si quieres personalizar alguna ruta (por ejemplo, guardar las    ║
║     predicciones en otro disco), descomenta la línea correspondiente ║
║     en la sección "RUTAS PERSONALIZADAS" y pon tu ruta.             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# DETECCIÓN AUTOMÁTICA
# La raíz del proyecto es la carpeta que contiene este archivo config.py
# ─────────────────────────────────────────────────────────────────────
PROYECTO_RAIZ = Path(__file__).resolve().parent

# Rutas derivadas automáticamente
RUTA_DATOS    = PROYECTO_RAIZ / "datos"                     # xlsx crudos descargados
RUTA_PARQUET  = PROYECTO_RAIZ / "outputs" / "parquet"       # archivos parquet procesados
RUTA_PREDS    = PROYECTO_RAIZ / "outputs" / "predicciones"  # CSVs de predicciones
RUTA_FIGURAS  = PROYECTO_RAIZ / "outputs" / "FIGURAS"       # imágenes generadas
RUTA_PREDS_2025_01  = PROYECTO_RAIZ / "outputs" / "predicciones" / "2025-01-01_a_2025-03-01"
RUTA_PREDS_2026    = PROYECTO_RAIZ / "outputs" / "predicciones" / "2026-01-01_a_2026-05-01_ventana_movil"


# ─────────────────────────────────────────────────────────────────────
# RUTAS PERSONALIZADAS — descomenta y ajusta si necesitas cambiar alguna
# ─────────────────────────────────────────────────────────────────────
# Ejemplos:
#   Windows:  Path(r"C:\Users\TuNombre\Desktop\MiCarpeta")
#   Mac/Linux: Path("/Users/tunombre/Desktop/MiCarpeta")

# PROYECTO_RAIZ = Path(r"C:\Users\TuNombre\Desktop\Proyecto_Investigacion_Teorica_2026_2")
# RUTA_DATOS    = Path(r"C:\ruta\personalizada\datos")
# RUTA_PARQUET  = Path(r"C:\ruta\personalizada\parquet")
# RUTA_PREDS    = Path(r"C:\ruta\personalizada\predicciones")
# RUTA_FIGURAS  = Path(r"C:\ruta\personalizada\figuras")

# ─────────────────────────────────────────────────────────────────────
# Crear carpetas de salida si no existen
# ─────────────────────────────────────────────────────────────────────
RUTA_PREDS.mkdir(parents=True, exist_ok=True)
RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
RUTA_PREDS_2026.mkdir(parents=True, exist_ok=True)
