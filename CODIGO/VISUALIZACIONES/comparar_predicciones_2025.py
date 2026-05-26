"""
Comparación predicciones 2025 vs datos reales
Lee los parquets de 2025 directamente de la carpeta de entrenamiento
y los contrasta con el CSV de predicciones del forecasting recursivo.
"""

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────

RUTA_PARQUET = r"C:\Users\Juanshots\Desktop\PROYECTO_INV_TEO\DATOS LIMPIOS\ENTRADA Y SALIDA MENSUAL\parquet"
RUTA_PRED    = r"C:\Users\Juanshots\Desktop\PROYECTO_INV_TEO\DATOS LIMPIOS\ENTRADA Y SALIDA MENSUAL\PREDICCIONES\predicciones_2025_prueba.csv"
RUTA_SALIDA  = r"C:\Users\Juanshots\Desktop\PROYECTO_INV_TEO\DATOS LIMPIOS\ENTRADA Y SALIDA MENSUAL\PREDICCIONES\comparacion_2025.png"

# ─────────────────────────────────────────────
# 1. DATOS REALES 2025
# ─────────────────────────────────────────────

archivos_2025 = sorted(glob.glob(f"{RUTA_PARQUET}\\2025-*-entradas.parquet"))

if not archivos_2025:
    raise FileNotFoundError("No se encontraron archivos de 2025 en la carpeta de parquets.")

print(f"Archivos 2025 encontrados: {len(archivos_2025)}")

partes = []
for archivo in archivos_2025:
    df_mes = pd.read_parquet(archivo)
    agg = df_mes.groupby("datetime", as_index=False)["entradas"].sum()
    partes.append(agg)

real = pd.concat(partes, ignore_index=True)
real["datetime"] = pd.to_datetime(real["datetime"])
real = real.sort_values("datetime").reset_index(drop=True)

print(f"Rango real:  {real['datetime'].min()} -> {real['datetime'].max()}")
print(f"Intervalos reales: {len(real):,}")

# ─────────────────────────────────────────────
# 2. PREDICCIONES
# ─────────────────────────────────────────────

pred = pd.read_csv(RUTA_PRED)
pred["datetime"] = pd.to_datetime(pred["datetime"])
pred = pred.sort_values("datetime").reset_index(drop=True)

print(f"Rango pred:  {pred['datetime'].min()} -> {pred['datetime'].max()}")
print(f"Intervalos predichos: {len(pred):,}")

# ─────────────────────────────────────────────
# 3. MERGE Y MÉTRICAS
# ─────────────────────────────────────────────

df = real.merge(pred, on="datetime", how="inner")
df = df.rename(columns={"entradas": "real", "prediccion": "pred"})

mae  = mean_absolute_error(df["real"], df["pred"])
rmse = mean_squared_error(df["real"], df["pred"]) ** 0.5
mape = (np.abs(df["real"] - df["pred"]) / df["real"].replace(0, np.nan)).mean() * 100

print(f"\nMétricas sobre intervalos comunes ({len(df):,} puntos):")
print(f"  MAE:  {mae:>10,.0f} pasajeros")
print(f"  RMSE: {rmse:>10,.0f} pasajeros")
print(f"  MAPE: {mape:>9.1f} %")

# Agrupado diario para gráficas más legibles
diario = df.set_index("datetime").resample("D").sum()

# ─────────────────────────────────────────────
# 4. GRÁFICAS
# ─────────────────────────────────────────────

fig, axes = plt.subplots(4, 1, figsize=(24, 18))
fig.suptitle("XGBoost TransMilenio: Predicción 2025 vs Real\n"
             f"MAE={mae:,.0f}  RMSE={rmse:,.0f}  MAPE={mape:.1f}%", fontsize=13)

# -- Panel 1: serie diaria completa
ax = axes[0]
ax.plot(diario.index, diario["real"], label="Real",     color="steelblue", linewidth=1.2)
ax.plot(diario.index, diario["pred"], label="Predicho", color="tomato",    linewidth=1.2, alpha=0.85)
ax.set_title("Serie diaria completa 2025 (suma de intervalos de 15 min por día)")
ax.set_ylabel("Entradas diarias")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.legend()
ax.grid(True, alpha=0.3)

# -- Panel 2: zoom dos semanas enero (granularidad 15 min)
inicio_zoom = pd.Timestamp("2025-01-01")
fin_zoom    = inicio_zoom + pd.Timedelta(weeks=2)
zoom        = df[(df["datetime"] >= inicio_zoom) & (df["datetime"] < fin_zoom)]

ax = axes[1]
ax.plot(zoom["datetime"], zoom["real"], label="Real",     color="steelblue", linewidth=1.0)
ax.plot(zoom["datetime"], zoom["pred"], label="Predicho", color="tomato",    linewidth=1.0, alpha=0.85)
ax.set_title("Zoom: primeras 2 semanas de enero 2025 (intervalos de 15 min)")
ax.set_ylabel("Entradas")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.legend()
ax.grid(True, alpha=0.3)

# -- Panel 3: error absoluto diario
error_diario = (diario["real"] - diario["pred"]).abs()

ax = axes[2]
ax.bar(error_diario.index, error_diario.values, color="slategray", width=1, alpha=0.7)
ax.axhline(error_diario.mean(), color="tomato", linewidth=1.5, linestyle="--",
           label=f"Error medio diario: {error_diario.mean():,.0f}")
ax.set_title("Error absoluto diario (|real - predicho|)")
ax.set_ylabel("Error absoluto")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.legend()
ax.grid(True, alpha=0.3)

# -- Panel 4: perfil horario promedio real vs predicho
df["hora"] = df["datetime"].dt.hour
perfil = df.groupby("hora")[["real", "pred"]].mean()

ax = axes[3]
ax.plot(perfil.index, perfil["real"], label="Real",     color="steelblue", linewidth=2)
ax.plot(perfil.index, perfil["pred"], label="Predicho", color="tomato",    linewidth=2, alpha=0.85)
ax.set_title("Perfil horario promedio anual (media por hora del día)")
ax.set_xlabel("Hora del día")
ax.set_ylabel("Entradas promedio por intervalo de 15 min")
ax.set_xticks(range(0, 24, 2))
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RUTA_SALIDA, dpi=150, bbox_inches="tight")
plt.show()
print(f"Figura guardada en {RUTA_SALIDA}")
