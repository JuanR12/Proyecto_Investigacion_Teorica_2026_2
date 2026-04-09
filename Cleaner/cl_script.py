import pandas as pd
from pathlib import Path

# =========================
# CONFIGURACIÓN
# =========================
archivo_entrada = Path(r"20260408.csv")   # si está en la misma carpeta del script
archivo_salida = Path(r"20260408_filtrado_zona_c_av_suba.csv")

columna_filtro = "Linea"
valor_objetivo = "(32) Zona C Av. Suba"
tamano_bloque = 500_000  # puedes subir o bajar este valor según tu RAM

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
    bloque.columns = bloque.columns.str.strip()

    # Verificar que exista la columna
    if columna_filtro not in bloque.columns:
        raise ValueError(
            f"No existe la columna '{columna_filtro}'. "
            f"Columnas encontradas: {list(bloque.columns)}"
        )

    # Limpiar texto en la columna Linea
    bloque[columna_filtro] = bloque[columna_filtro].astype(str).str.strip()

    # Aplicar filtro
    bloque_filtrado = bloque[bloque[columna_filtro] == valor_objetivo].copy()

    # Contadores
    total_original += len(bloque)
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

print(f"Total de registros procesados: {total_original:,}")
print(f"Total de registros filtrados: {total_filtrado:,}")
print(f"Archivo generado: {archivo_salida}")