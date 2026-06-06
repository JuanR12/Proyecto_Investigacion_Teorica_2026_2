import pandas as pd
from pathlib import Path

# =========================
# CONFIGURACIÓN
# =========================
archivo_entrada = Path(r"20260408.csv")
archivo_salida = Path(r"20260408_filtrado_limpio_zona_c_av_suba.csv")

valor_objetivo = "(32) Zona C Av. Suba"
tamano_bloque = 500_000

# Columnas que sí quieres conservar
columnas_interes = [
    "Estacion_Parada",
    "Fecha_Transaccion",
    "Linea",
    "Numero_Tarjeta"
]

# =========================
# DETECTAR CODIFICACIÓN
# =========================
encoding_usado = "utf-8-sig"
try:
    pd.read_csv(archivo_entrada, nrows=5, encoding=encoding_usado)
except UnicodeDecodeError:
    encoding_usado = "latin-1"

print(f"Codificación usada: {encoding_usado}")

# =========================
# PROCESAMIENTO POR BLOQUES
# =========================
primera_escritura = True
total_original = 0
total_filtrado = 0

for bloque in pd.read_csv(
    archivo_entrada,
    chunksize=tamano_bloque,
    encoding=encoding_usado
):
    # Limpiar nombres de columnas
    bloque.columns = [col.strip() for col in bloque.columns]

    # Verificar que existan las columnas necesarias
    columnas_faltantes = [col for col in columnas_interes if col not in bloque.columns]
    if columnas_faltantes:
        raise ValueError(
            f"Faltan estas columnas en el archivo: {columnas_faltantes}\n"
            f"Columnas encontradas: {bloque.columns.tolist()}"
        )

    total_original += len(bloque)

    # Conservar solo columnas útiles
    bloque = bloque[columnas_interes].copy()

    # Limpiar texto en la columna Linea
    bloque["Linea"] = (
        bloque["Linea"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Opcional: limpiar también Estacion_Parada y Numero_Tarjeta
    bloque["Estacion_Parada"] = bloque["Estacion_Parada"].astype(str).str.strip()
    bloque["Numero_Tarjeta"] = bloque["Numero_Tarjeta"].astype(str).str.strip()

    # Filtrar por la línea deseada
    bloque_filtrado = bloque[bloque["Linea"] == valor_objetivo].copy()

    total_filtrado += len(bloque_filtrado)

    # Guardar resultado
    bloque_filtrado.to_csv(
        archivo_salida,
        mode="w" if primera_escritura else "a",
        header=primera_escritura,
        index=False,
        encoding="utf-8-sig"
    )

    primera_escritura = False

print("Proceso terminado.")
print(f"Registros originales procesados: {total_original:,}")
print(f"Registros filtrados guardados:   {total_filtrado:,}")
print(f"Archivo de salida: {archivo_salida}")