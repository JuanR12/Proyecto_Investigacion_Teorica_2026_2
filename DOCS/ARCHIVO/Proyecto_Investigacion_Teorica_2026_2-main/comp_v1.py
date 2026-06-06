import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# CONFIGURACIÓN
# =========================================================

# Archivo de ingresos (validaciones individuales)
archivo_ingresos = Path(r"20260408.csv")

# Archivo de salidas (agregado por intervalos de 15 minutos)
archivo_salidas = Path(r"salidas_20260408.csv")   # cambia esto si el nombre real difiere

# Salidas
archivo_comparacion = Path(r"comparacion_marsella_20260408.csv")
archivo_grafica_lineas = Path(r"comparacion_marsella_20260408_lineas.png")
archivo_grafica_barras = Path(r"comparacion_marsella_20260408_barras.png")

# Estación a analizar
estacion_objetivo_ingresos = "(05103) Marsella"
estacion_objetivo_salidas = "(05103) Marsella"

# Tamaño de bloque para el archivo grande de ingresos
tamano_bloque = 500_000


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def detectar_encoding_csv(ruta_csv):
    encoding_usado = "utf-8-sig"
    try:
        pd.read_csv(ruta_csv, nrows=5, encoding=encoding_usado)
    except UnicodeDecodeError:
        encoding_usado = "latin-1"
    return encoding_usado


def construir_intervalos_dia(fecha_base):
    inicio_dia = pd.Timestamp(fecha_base).normalize()
    intervalos = pd.date_range(
        start=inicio_dia,
        end=inicio_dia + pd.Timedelta(days=1) - pd.Timedelta(minutes=15),
        freq="15min"
    )
    return pd.DataFrame({"Intervalo_15min": intervalos})


# =========================================================
# 1) PROCESAR INGRESOS
#    Archivo de validaciones individuales
# =========================================================

encoding_ingresos = detectar_encoding_csv(archivo_ingresos)
print(f"Codificación ingresos: {encoding_ingresos}")

columnas_ingresos = ["Estacion_Parada", "Fecha_Transaccion"]

partes_ingresos = []

for bloque in pd.read_csv(
    archivo_ingresos,
    usecols=columnas_ingresos,
    chunksize=tamano_bloque,
    encoding=encoding_ingresos
):
    bloque.columns = [c.strip() for c in bloque.columns]

    bloque["Estacion_Parada"] = (
        bloque["Estacion_Parada"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    bloque = bloque[bloque["Estacion_Parada"] == estacion_objetivo_ingresos].copy()

    if bloque.empty:
        continue

    bloque["Fecha_Transaccion"] = pd.to_datetime(
        bloque["Fecha_Transaccion"],
        errors="coerce"
    )

    bloque = bloque.dropna(subset=["Fecha_Transaccion"])

    # Llevar cada validación al inicio de su intervalo de 15 minutos
    bloque["Intervalo_15min"] = bloque["Fecha_Transaccion"].dt.floor("15min")

    partes_ingresos.append(bloque[["Intervalo_15min"]])

if not partes_ingresos:
    raise ValueError("No se encontraron ingresos para la estación objetivo.")

df_ingresos = pd.concat(partes_ingresos, ignore_index=True)

# Fecha base del día
fecha_base = df_ingresos["Intervalo_15min"].min().normalize()

ingresos_15min = (
    df_ingresos
    .groupby("Intervalo_15min")
    .size()
    .reset_index(name="Ingresos")
)

print(f"Total ingresos en {estacion_objetivo_ingresos}: {ingresos_15min['Ingresos'].sum():,}")


# =========================================================
# 2) PROCESAR SALIDAS
#    Archivo agregado por cuarto de hora
# =========================================================

encoding_salidas = detectar_encoding_csv(archivo_salidas)
print(f"Codificación salidas: {encoding_salidas}")

df_salidas = pd.read_csv(archivo_salidas, encoding=encoding_salidas)
df_salidas.columns = [c.strip() for c in df_salidas.columns]

columnas_necesarias_salidas = [
    "Fecha_Transaccion",
    "Tiempo",
    "Estacion",
    "Salidas_S"
]

faltantes = [c for c in columnas_necesarias_salidas if c not in df_salidas.columns]
if faltantes:
    raise ValueError(
        f"Faltan columnas en archivo de salidas: {faltantes}\n"
        f"Columnas encontradas: {df_salidas.columns.tolist()}"
    )

# Limpiar estación
df_salidas["Estacion"] = (
    df_salidas["Estacion"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

# Filtrar Marsella
df_salidas = df_salidas[df_salidas["Estacion"] == estacion_objetivo_salidas].copy()

if df_salidas.empty:
    raise ValueError("No se encontraron salidas para la estación objetivo.")

# Convertir fecha
df_salidas["Fecha_Transaccion"] = pd.to_datetime(
    df_salidas["Fecha_Transaccion"],
    errors="coerce"
)

# Normalizar Tiempo
df_salidas["Tiempo"] = df_salidas["Tiempo"].astype(str).str.strip()

# Construir datetime del intervalo usando fecha + tiempo
df_salidas["Intervalo_15min"] = pd.to_datetime(
    df_salidas["Fecha_Transaccion"].dt.strftime("%Y-%m-%d") + " " + df_salidas["Tiempo"],
    errors="coerce"
)

df_salidas["Salidas_S"] = pd.to_numeric(df_salidas["Salidas_S"], errors="coerce").fillna(0)

df_salidas = df_salidas.dropna(subset=["Intervalo_15min"])

# Agrupar por si hubiera más de un registro por intervalo
salidas_15min = (
    df_salidas
    .groupby("Intervalo_15min", as_index=False)["Salidas_S"]
    .sum()
    .rename(columns={"Salidas_S": "Salidas"})
)

print(f"Total salidas en {estacion_objetivo_salidas}: {salidas_15min['Salidas'].sum():,}")


# =========================================================
# 3) ARMAR TABLA COMPARATIVA COMPLETA DEL DÍA
# =========================================================

intervalos_dia = construir_intervalos_dia(fecha_base)

comparacion = (
    intervalos_dia
    .merge(ingresos_15min, on="Intervalo_15min", how="left")
    .merge(salidas_15min, on="Intervalo_15min", how="left")
)

comparacion["Ingresos"] = comparacion["Ingresos"].fillna(0).astype(int)
comparacion["Salidas"] = comparacion["Salidas"].fillna(0)

comparacion["Diferencia"] = comparacion["Ingresos"] - comparacion["Salidas"]
comparacion["Fecha"] = comparacion["Intervalo_15min"].dt.date
comparacion["Hora"] = comparacion["Intervalo_15min"].dt.strftime("%H:%M")

comparacion = comparacion[
    ["Fecha", "Hora", "Intervalo_15min", "Ingresos", "Salidas", "Diferencia"]
]

comparacion.to_csv(archivo_comparacion, index=False, encoding="utf-8-sig")
print(f"Archivo comparativo exportado: {archivo_comparacion}")


# =========================================================
# 4) GRÁFICA 1: LÍNEAS COMPARATIVAS
# =========================================================

plt.figure(figsize=(14, 6))
plt.plot(comparacion["Hora"], comparacion["Ingresos"], label="Ingresos")
plt.plot(comparacion["Hora"], comparacion["Salidas"], label="Salidas")
plt.title(f"Ingresos vs Salidas por intervalo de 15 min - {estacion_objetivo_ingresos}")
plt.xlabel("Hora del día")
plt.ylabel("Validaciones")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig(archivo_grafica_lineas, dpi=300)
plt.show()

print(f"Gráfica de líneas guardada en: {archivo_grafica_lineas}")


# =========================================================
# 5) GRÁFICA 2: DIFERENCIA NETA
# =========================================================

plt.figure(figsize=(14, 6))
plt.bar(comparacion["Hora"], comparacion["Diferencia"])
plt.title(f"Diferencia neta (Ingresos - Salidas) por intervalo de 15 min - {estacion_objetivo_ingresos}")
plt.xlabel("Hora del día")
plt.ylabel("Diferencia neta")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(archivo_grafica_barras, dpi=300)
plt.show()

print(f"Gráfica de barras guardada en: {archivo_grafica_barras}")