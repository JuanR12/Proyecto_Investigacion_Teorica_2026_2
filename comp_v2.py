import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# CONFIGURACIÓN
# ==========================================
archivo_salidas = Path(r"salidas_20260224.csv")   # cambia al nombre real si difiere
archivo_salida_csv = Path(r"marsella_entradas_salidas_20260408.csv")
archivo_grafica = Path(r"marsella_entradas_salidas_20260408.png")

estacion_objetivo = "(02103)Mazurén"

# ==========================================
# DETECTAR CODIFICACIÓN
# ==========================================
encoding_usado = "utf-8-sig"
try:
    pd.read_csv(archivo_salidas, nrows=5, encoding=encoding_usado)
except UnicodeDecodeError:
    encoding_usado = "latin-1"

print(f"Codificación usada: {encoding_usado}")

# ==========================================
# LEER ARCHIVO
# ==========================================
df = pd.read_csv(archivo_salidas, encoding=encoding_usado)
df.columns = [col.strip() for col in df.columns]

# Verificar columnas requeridas
columnas_requeridas = [
    "Fecha_Transaccion",
    "Tiempo",
    "Estacion",
    "Entradas_E",
    "Salidas_S"
]

faltantes = [col for col in columnas_requeridas if col not in df.columns]
if faltantes:
    raise ValueError(
        f"Faltan estas columnas en el archivo: {faltantes}\n"
        f"Columnas encontradas: {df.columns.tolist()}"
    )

# ==========================================
# LIMPIEZA Y FILTRO
# ==========================================
df["Estacion"] = (
    df["Estacion"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

df = df[df["Estacion"] == estacion_objetivo].copy()

if df.empty:
    raise ValueError(f"No se encontraron datos para la estación {estacion_objetivo}")

# Convertir fecha
df["Fecha_Transaccion"] = pd.to_datetime(df["Fecha_Transaccion"], errors="coerce")

# Limpiar tiempo
df["Tiempo"] = df["Tiempo"].astype(str).str.strip()

# Construir datetime completo
df["FechaHora"] = pd.to_datetime(
    df["Fecha_Transaccion"].dt.strftime("%Y-%m-%d") + " " + df["Tiempo"],
    errors="coerce"
)

# Convertir columnas numéricas
df["Entradas_E"] = pd.to_numeric(df["Entradas_E"], errors="coerce").fillna(0)
df["Salidas_S"] = pd.to_numeric(df["Salidas_S"], errors="coerce").fillna(0)

# Eliminar filas inválidas
df = df.dropna(subset=["FechaHora"])

# ==========================================
# AGRUPAR POR INTERVALO
# ==========================================
# Por si hubiera más de un registro en el mismo cuarto de hora
df_resumen = (
    df.groupby("FechaHora", as_index=False)[["Entradas_E", "Salidas_S"]]
    .sum()
    .sort_values("FechaHora")
)

# Crear columna de hora para exportar/graficar
df_resumen["Hora"] = df_resumen["FechaHora"].dt.strftime("%H:%M")
df_resumen["Fecha"] = df_resumen["FechaHora"].dt.date

# Reordenar columnas
df_resumen = df_resumen[["Fecha", "Hora", "FechaHora", "Entradas_E", "Salidas_S"]]

# ==========================================
# EXPORTAR CSV
# ==========================================
df_resumen.to_csv(archivo_salida_csv, index=False, encoding="utf-8-sig")

print("Proceso terminado.")
print(f"Registros exportados: {len(df_resumen):,}")
print(f"Archivo CSV generado: {archivo_salida_csv}")

# ==========================================
# GRÁFICA COMPARATIVA EN UN SOLO PLANO
# ==========================================
plt.figure(figsize=(14, 6))
plt.plot(df_resumen["Hora"], df_resumen["Entradas_E"], label="Entradas_E")
plt.plot(df_resumen["Hora"], df_resumen["Salidas_S"], label="Salidas_S")

plt.title(f"Entradas y Salidas por intervalo de 15 min - {estacion_objetivo}")
plt.xlabel("Hora del día")
plt.ylabel("Cantidad")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig(archivo_grafica, dpi=300)
plt.show()

print(f"Gráfica guardada en: {archivo_grafica}")