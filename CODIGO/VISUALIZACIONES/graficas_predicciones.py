# ── visualizar_predicciones.py ─────────────────────────────────────────────────
# Script independiente. Requiere que los parquets de 2025 estén disponibles.
# Pegar después de la sección EXTRA del xgboost_transmilenio_v1.py,
# o ejecutar por separado apuntando las rutas de abajo.

import glob
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Importar rutas desde config.py (raíz del proyecto). Ver config.py para personalizar.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RUTA_PARQUET as PARQUET_DIR, RUTA_PREDS, RUTA_FIGURAS

# ── RUTAS (ajustar) ────────────────────────────────────────────────────────────
PRED_CSV = RUTA_PREDS   / "predicciones_2025_prueba.csv"
OUT_IMG  = RUTA_FIGURAS / "resultados_visualizacion.png"

# ── 1. CARGAR REAL DESDE PARQUETS 2025 ────────────────────────────────────────
archivos_2025 = sorted(glob.glob(str(PARQUET_DIR / "2025-*-entradas.parquet")))
if not archivos_2025:
    raise FileNotFoundError("No se encontraron parquets de 2025 en PARQUET_DIR")

real = (
    pd.concat(
        [pd.read_parquet(f).groupby("datetime", as_index=False)["entradas"].sum()
         for f in archivos_2025],
        ignore_index=True,
    )
    .groupby("datetime", as_index=False)["entradas"].sum()
)
real["datetime"] = pd.to_datetime(real["datetime"])

# ── 2. CARGAR PREDICCIONES ─────────────────────────────────────────────────────
pred = pd.read_csv(PRED_CSV, parse_dates=["datetime"])
# El CSV del modelo exporta datetime,real,prediccion,error_absoluto
# El CSV reducido solo tiene datetime,prediccion -- ambos funcionan aquí
pred = pred[["datetime", "prediccion"]].copy()
pred["prediccion"] = pd.to_numeric(pred["prediccion"], errors="coerce")

# ── 3. MERGE Y MÉTRICAS ────────────────────────────────────────────────────────
df = real.merge(pred, on="datetime", how="inner").dropna()
df = df.sort_values("datetime").reset_index(drop=True)

y_real = df["entradas"].values
y_pred = df["prediccion"].values

mae  = mean_absolute_error(y_real, y_pred)
rmse = mean_squared_error(y_real, y_pred) ** 0.5
mape = np.mean(np.abs((y_real - y_pred) / np.where(y_real == 0, 1, y_real))) * 100

print(f"Puntos comparados : {len(df):,}")
print(f"MAE               : {mae:,.0f} pasajeros")
print(f"RMSE              : {rmse:,.0f} pasajeros")
print(f"MAPE (aprox.)     : {mape:.1f} %")

# ── 4. SERIES AUXILIARES ───────────────────────────────────────────────────────
df["hora"]    = df["datetime"].dt.hour
df["dow"]     = df["datetime"].dt.weekday          # 0=lunes … 6=domingo
df["error"]   = y_real - y_pred
df["abs_err"] = np.abs(df["error"])

perfil_real = df.groupby("hora")[["entradas", "prediccion"]].mean()
perfil_dow  = df.groupby("dow")[["entradas", "prediccion"]].mean()

DOS_SEMANAS = df["datetime"] < (df["datetime"].min() + pd.Timedelta(weeks=2))

# ── 5. FIGURA ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(20, 11))
fig.suptitle(
    f"XGBoost TransMilenio – Entradas 2025   |   MAE {mae:,.0f}  RMSE {rmse:,.0f}  MAPE {mape:.1f}%",
    fontsize=13, y=1.01,
)

COLOR_REAL = "#2563eb"
COLOR_PRED = "#dc2626"
ALPHA_PRED = 0.80

# Panel A: serie completa 2025
ax = axes[0, 0]
ax.plot(df["datetime"], y_real, color=COLOR_REAL, lw=0.6, label="Real")
ax.plot(df["datetime"], y_pred, color=COLOR_PRED, lw=0.6, alpha=ALPHA_PRED, label="Predicción")
ax.set_title("Serie completa 2025")
ax.set_ylabel("Entradas por intervalo de 15 min")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.legend()
ax.grid(True, alpha=0.25)

# Panel B: zoom primeras 2 semanas
ax = axes[0, 1]
sub = df[DOS_SEMANAS]
ax.plot(sub["datetime"], sub["entradas"], color=COLOR_REAL, lw=1.2, label="Real")
ax.plot(sub["datetime"], sub["prediccion"], color=COLOR_PRED, lw=1.2, alpha=ALPHA_PRED, label="Predicción")
ax.set_title("Zoom: primeras 2 semanas de 2025")
ax.set_ylabel("Entradas por intervalo de 15 min")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.legend()
ax.grid(True, alpha=0.25)

# Panel C: perfil horario promedio
ax = axes[1, 0]
ax.plot(perfil_real.index, perfil_real["entradas"],    color=COLOR_REAL, lw=2, marker="o", ms=4, label="Real")
ax.plot(perfil_real.index, perfil_real["prediccion"],  color=COLOR_PRED, lw=2, marker="o", ms=4, label="Predicción")
ax.fill_between(perfil_real.index,
                perfil_real["entradas"], perfil_real["prediccion"],
                alpha=0.12, color=COLOR_PRED)
ax.set_title("Perfil horario promedio (todo 2025)")
ax.set_xlabel("Hora del día")
ax.set_ylabel("Entradas promedio")
ax.set_xticks(range(0, 24, 2))
ax.legend()
ax.grid(True, alpha=0.25)

# Panel D: perfil por día de semana
dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
ax = axes[1, 1]
x = np.arange(7)
w = 0.35
ax.bar(x - w/2, perfil_dow["entradas"],   width=w, color=COLOR_REAL, alpha=0.85, label="Real")
ax.bar(x + w/2, perfil_dow["prediccion"], width=w, color=COLOR_PRED, alpha=0.75, label="Predicción")
ax.set_title("Promedio por día de semana (todo 2025)")
ax.set_xticks(x)
ax.set_xticklabels(dias)
ax.set_ylabel("Entradas promedio por intervalo")
ax.legend()
ax.grid(True, alpha=0.25, axis="y")

plt.tight_layout()
plt.savefig(OUT_IMG, dpi=150, bbox_inches="tight")
plt.show()
print(f"Figura guardada en {OUT_IMG}")