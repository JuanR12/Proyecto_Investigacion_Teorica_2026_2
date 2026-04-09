import pandas as pd
from pathlib import Path

# =========================
# CONFIGURACIÓN
# =========================
archivo_entrada = Path(r"20260408.csv")
archivo_salida = Path(r"20260408_tarjetas_repetidas_zona_c_av_suba.csv")

valor_objetivo = "(32) Zona C Av. Suba"
tamano_bloque = 500_000

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
    encabezado = pd.read_csv(archivo_entrada, nrows=0, encoding=encoding_usado)
except UnicodeDecodeError:
    encoding_usado = "latin-1"
    encabezado = pd.read_csv(archivo_entrada, nrows=0, encoding=encoding_usado)

encabezado.columns = [col.strip() for col in encabezado.columns]

columnas_faltantes = [col for col in columnas_interes if col not in encabezado.columns]
if columnas_faltantes:
    raise ValueError(
        f"Faltan estas columnas en el archivo: {columnas_faltantes}\n"
        f"Columnas encontradas: {encabezado.columns.tolist()}"
    )

print(f"Codificación usada: {encoding_usado}")

# =========================
# PRIMER PASO:
# filtrar por Linea y conservar columnas útiles
# =========================
partes_filtradas = []

for bloque in pd.read_csv(
    archivo_entrada,
    usecols=columnas_interes,
    chunksize=tamano_bloque,
    encoding=encoding_usado
):
    bloque.columns = [col.strip() for col in bloque.columns]

    bloque["Linea"] = (
        bloque["Linea"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    bloque["Estacion_Parada"] = bloque["Estacion_Parada"].astype(str).str.strip()
    bloque["Numero_Tarjeta"] = bloque["Numero_Tarjeta"].astype(str).str.strip()

    bloque["Fecha_Transaccion"] = pd.to_datetime(
        bloque["Fecha_Transaccion"],
        errors="coerce"
    )

    bloque = bloque[bloque["Linea"] == valor_objetivo].copy()

    if not bloque.empty:
        partes_filtradas.append(bloque)

# Unir todo lo filtrado
if not partes_filtradas:
    print("No se encontraron registros para la línea objetivo.")
    raise SystemExit

df = pd.concat(partes_filtradas, ignore_index=True)

# =========================
# SEGUNDO PASO:
# ordenar por tarjeta y fecha
# =========================
df = df.sort_values(
    by=["Numero_Tarjeta", "Fecha_Transaccion"],
    ascending=[True, True]
).reset_index(drop=True)

# =========================
# TERCER PASO:
# conservar solo tarjetas repetidas
# =========================
# Esto mantiene solamente las tarjetas que aparecen 2 o más veces
df_repetidas = df[df["Numero_Tarjeta"].duplicated(keep=False)].copy()

# =========================
# GUARDAR RESULTADO
# =========================
df_repetidas.to_csv(archivo_salida, index=False, encoding="utf-8-sig")

print("Proceso terminado.")
print(f"Registros en línea objetivo: {len(df):,}")
print(f"Registros con tarjetas repetidas: {len(df_repetidas):,}")
print(f"Tarjetas únicas repetidas: {df_repetidas['Numero_Tarjeta'].nunique():,}")
print(f"Archivo guardado en: {archivo_salida}")