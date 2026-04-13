import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURACIÓN
# ==========================================
archivo_entrada = Path(r"20260401.csv")   # cambia esto si el nombre o ruta es distinto

archivo_lineas = Path(r"lista_unica_lineas.csv")
archivo_estaciones = Path(r"lista_unica_estaciones_parada.csv")

tamano_bloque = 500_000

# ==========================================
# DETECTAR CODIFICACIÓN
# ==========================================
encoding_usado = "utf-8-sig"
try:
    pd.read_csv(archivo_entrada, nrows=5, encoding=encoding_usado)
except UnicodeDecodeError:
    encoding_usado = "latin-1"

print(f"Codificación usada: {encoding_usado}")

# ==========================================
# CONJUNTOS PARA VALORES ÚNICOS
# ==========================================
lineas_unicas = set()
estaciones_unicas = set()

# ==========================================
# LEER POR BLOQUES Y ACUMULAR ÚNICOS
# ==========================================
for bloque in pd.read_csv(
    archivo_entrada,
    usecols=["Linea", "Estacion_Parada"],
    chunksize=tamano_bloque,
    encoding=encoding_usado
):
    bloque.columns = [col.strip() for col in bloque.columns]

    # Limpiar texto
    bloque["Linea"] = (
        bloque["Linea"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    bloque["Estacion_Parada"] = (
        bloque["Estacion_Parada"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Quitar vacíos falsos
    bloque = bloque[
        (bloque["Linea"] != "") &
        (bloque["Estacion_Parada"] != "") &
        (bloque["Linea"].str.lower() != "nan") &
        (bloque["Estacion_Parada"].str.lower() != "nan")
    ]

    # Acumular únicos
    lineas_unicas.update(bloque["Linea"].unique())
    estaciones_unicas.update(bloque["Estacion_Parada"].unique())

# ==========================================
# ORDENAR RESULTADOS
# ==========================================
lineas_ordenadas = sorted(lineas_unicas)
estaciones_ordenadas = sorted(estaciones_unicas)

# ==========================================
# MOSTRAR EN PANTALLA
# ==========================================
print("\n=== LINEAS ÚNICAS ===")
for linea in lineas_ordenadas:
    print(linea)

print("\n=== ESTACIONES/PARADAS ÚNICAS ===")
for estacion in estaciones_ordenadas:
    print(estacion)

# ==========================================
# GUARDAR EN CSV
# ==========================================
pd.DataFrame({"Linea": lineas_ordenadas}).to_csv(
    archivo_lineas, index=False, encoding="utf-8-sig"
)

pd.DataFrame({"Estacion_Parada": estaciones_ordenadas}).to_csv(
    archivo_estaciones, index=False, encoding="utf-8-sig"
)

print("\nProceso terminado.")
print(f"Total de lineas únicas: {len(lineas_ordenadas)}")
print(f"Total de estaciones/paradas únicas: {len(estaciones_ordenadas)}")
print(f"Archivo generado: {archivo_lineas}")
print(f"Archivo generado: {archivo_estaciones}")