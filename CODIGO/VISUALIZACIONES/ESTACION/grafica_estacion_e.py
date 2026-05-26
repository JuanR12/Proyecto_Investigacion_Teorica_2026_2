"""
Comparación predicción vs real por estación - Sistema Troncal TransMilenio
Entrada:  validaciones_entrada_{STATION_ID}_pred.csv  (salida del modelo)
          *-entradas.parquet                           (datos reales)
Salida:   grafica_estacion_{STATION_ID}.png
"""

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────
# CONFIGURACIÓN  — único valor a cambiar
# ─────────────────────────────────────────────

STATION_ID   = "06000"

RUTA_PARQUET = r"C:\Users\Juanshots\Desktop\PROYECTO_INV_TEO\DATOS LIMPIOS\ENTRADA Y SALIDA MENSUAL\parquet"
RUTA_PRED    = rf"C:\Users\Juanshots\Desktop\PROYECTO_INV_TEO\DATOS LIMPIOS\ENTRADA Y SALIDA MENSUAL\PREDICCIONES\validaciones_entrada_{STATION_ID}_pred_2025.csv"
RUTA_SALIDA  = rf"C:\Users\Juanshots\Desktop\PROYECTO_INV_TEO\DATOS LIMPIOS\ENTRADA Y SALIDA MENSUAL\PREDICCIONES\grafica_estacion_{STATION_ID}.png"

# ─────────────────────────────────────────────
# 1. CARGA DE PREDICCIÓN
# ─────────────────────────────────────────────

pred = pd.read_csv(RUTA_PRED, parse_dates=["datetime"])
pred = pred.sort_values("datetime").reset_index(drop=True)

INICIO_PRED = pred["datetime"].min()
FIN_PRED    = pred["datetime"].max()

print(f"Estación:   {STATION_ID}")
print(f"Predicción: {INICIO_PRED.date()} -> {FIN_PRED.date()}  ({len(pred):,} intervalos)")

# ─────────────────────────────────────────────
# 2. CARGA DE DATOS REALES PARA EL PERÍODO PREDICHO
#    y para el año de validación 2025 (perfil horario)
# ─────────────────────────────────────────────

archivos = sorted(glob.glob(rf"{RUTA_PARQUET}\*-entradas.parquet"))
if not archivos:
    raise FileNotFoundError(f"No se encontraron parquets en {RUTA_PARQUET}")

partes = []
for archivo in archivos:
    df_mes = pd.read_parquet(archivo)
    df_est = df_mes[df_mes["station_id"] == STATION_ID]
    if df_est.empty:
        continue
    partes.append(df_est[["datetime", "entradas"]])

if not partes:
    raise ValueError(f"La estación {STATION_ID} no aparece en ningún parquet.")

real_total = (
    pd.concat(partes, ignore_index=True)
    .assign(datetime=lambda d: pd.to_datetime(d["datetime"]))
    .groupby("datetime", as_index=False)["entradas"].sum()
    .sort_values("datetime")
    .reset_index(drop=True)
)

# Subconjunto que solapa con el período predicho (puede estar vacío si aún no hay datos)
real_pred = real_total[
    (real_total["datetime"] >= INICIO_PRED) &
    (real_total["datetime"] <= FIN_PRED)
].reset_index(drop=True)

# Año de validación 2025 para el perfil horario
real_2025 = real_total[
    (real_total["datetime"] >= "2025-01-01") &
    (real_total["datetime"] <  "2026-01-01")
].reset_index(drop=True)

hay_real_pred = len(real_pred) > 0
print(f"Datos reales en período predicho: {'sí' if hay_real_pred else 'NO — solo se grafica predicción'}")
print(f"Datos reales 2025 para perfil:    {len(real_2025):,} intervalos")

# ─────────────────────────────────────────────
# 3. MÉTRICAS (solo si hay solapamiento real)
# ─────────────────────────────────────────────

if hay_real_pred:
    comp = pred.merge(real_pred, on="datetime", how="inner")
    if len(comp) > 0:
        mae  = mean_absolute_error(comp["entradas"], comp["prediccion"])
        rmse = mean_squared_error(comp["entradas"], comp["prediccion"]) ** 0.5
        mape = (np.abs(comp["entradas"] - comp["prediccion"]) /
                comp["entradas"].replace(0, np.nan)).mean() * 100
        print(f"\nMétricas sobre {len(comp):,} intervalos solapados:")
        print(f"  MAE:  {mae:>10,.1f} pasajeros")
        print(f"  RMSE: {rmse:>10,.1f} pasajeros")
        print(f"  MAPE: {mape:>10,.1f} %")
    else:
        hay_real_pred = False

# ─────────────────────────────────────────────
# 4. PERFIL HORARIO PROMEDIO
#    Pred vs real 2025 — resume la calidad del patrón diario
# ─────────────────────────────────────────────

perfil_pred  = pred.groupby(pred["datetime"].dt.hour)["prediccion"].mean()

if len(real_2025) > 0:
    perfil_real25 = real_2025.groupby(real_2025["datetime"].dt.hour)["entradas"].mean()
else:
    perfil_real25 = None

# Perfil real del período predicho si existe
if hay_real_pred and len(comp) > 0:
    perfil_real_pred = comp.groupby(comp["datetime"].dt.hour)["entradas"].mean()
else:
    perfil_real_pred = None

# ─────────────────────────────────────────────
# 5. FIGURA
# ─────────────────────────────────────────────

n_panels = 2 if hay_real_pred else 3
fig, axes = plt.subplots(n_panels, 1, figsize=(22, 5 * n_panels))
fig.suptitle(
    f"Estación {STATION_ID}  —  Predicción de entradas 2026 (XGBoost)",
    fontsize=13, y=1.01
)

COLOR_PRED = "#dc2626"   # rojo
COLOR_REAL = "#2563eb"   # azul

# ── Panel 0: serie completa del período predicho ──────────────────────────────
ax = axes[0]
ax.plot(pred["datetime"], pred["prediccion"],
        color=COLOR_PRED, linewidth=0.7, label="Predicción XGBoost", zorder=3)
if hay_real_pred:
    ax.plot(comp["datetime"], comp["entradas"],
            color=COLOR_REAL, linewidth=0.7, alpha=0.7, label="Real", zorder=2)

ax.set_title("Serie completa — período predicho")
ax.set_ylabel("Entradas (intervalos 15 min)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)

# ── Panel 1: zoom primeras 2 semanas ─────────────────────────────────────────
ax = axes[1]
corte_2s = INICIO_PRED + pd.Timedelta(weeks=2)
mask_2s  = pred["datetime"] < corte_2s

ax.plot(pred.loc[mask_2s, "datetime"], pred.loc[mask_2s, "prediccion"],
        color=COLOR_PRED, linewidth=1.2, label="Predicción XGBoost")
if hay_real_pred:
    mask_2s_real = comp["datetime"] < corte_2s
    ax.plot(comp.loc[mask_2s_real, "datetime"], comp.loc[mask_2s_real, "entradas"],
            color=COLOR_REAL, linewidth=1.2, alpha=0.8, label="Real")

ax.set_title("Zoom: primeras 2 semanas")
ax.set_ylabel("Entradas (intervalos 15 min)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)

# ── Panel 2: perfil horario promedio ─────────────────────────────────────────
#ax = axes[2]
#ax.plot(perfil_pred.index, perfil_pred.values,
#        color=COLOR_PRED, linewidth=2, marker="o", markersize=3,
#        label="Predicción 2026 (promedio)")

#if perfil_real25 is not None:
#    ax.plot(perfil_real25.index, perfil_real25.values,
#            color=COLOR_REAL, linewidth=2, linestyle="--", marker="s", markersize=3,
#            label="Real 2025 (promedio)", alpha=0.8)

#if perfil_real_pred is not None:
#    ax.plot(perfil_real_pred.index, perfil_real_pred.values,
#            color="seagreen", linewidth=1.5, linestyle=":", marker="^", markersize=3,
#            label=f"Real {INICIO_PRED.year} período predicho", alpha=0.9)

#ax.axvspan(0, 5,  alpha=0.05, color="gray")
#ax.axvspan(22, 24, alpha=0.05, color="gray")
#ax.set_title("Perfil horario promedio (validaciones por intervalo de 15 min)")
#ax.set_xlabel("Hora del día")
#ax.set_ylabel("Promedio de entradas")
#ax.set_xticks(range(0, 24, 2))
#ax.legend(fontsize=9)
#ax.grid(True, alpha=0.25)

# ── Panel 3: distribución del error (solo si hay real solapado) ───────────────
#if hay_real_pred and n_panels == 4:
#    ax = axes[3]
#    errores = comp["prediccion"] - comp["entradas"]
#
#    ax.hist(errores, bins=80, color=COLOR_PRED, alpha=0.75, edgecolor="white", linewidth=0.3)
#    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
#    ax.axvline(errores.mean(), color="darkorange", linewidth=1.5,
#               linestyle="-", label=f"Media = {errores.mean():,.1f}")

#    ax.set_title("Distribución del error (predicción − real)")
#    ax.set_xlabel("Error (pasajeros por intervalo de 15 min)")
#    ax.set_ylabel("Frecuencia")
#    ax.legend(fontsize=9)
#    ax.grid(True, alpha=0.25)
    

plt.tight_layout()
plt.savefig(RUTA_SALIDA, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nFigura guardada en {RUTA_SALIDA}")
