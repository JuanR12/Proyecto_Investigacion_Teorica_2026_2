import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RUTA_PREDICCION = r"" # Ruta del archivo de predicciones (verificar que el nombre coincida)

pred = pd.read_csv(RUTA_PREDICCION, parse_dates=["datetime"])
pred["prediccion"] = pred["prediccion"].clip(lower=0).round().astype(int)

# ─────────────────────────────────────────────
# VISUALIZACIÓN
# ─────────────────────────────────────────────

COLOR_REAL = "#38bdf8"       # azul claro
COLOR_PRED = "#ef4444"       # rojo
COLOR_FILL = "#fecaca"       # rojo muy claro para el área de error

pred_diario = pred.set_index("datetime").resample("D")[["real", "prediccion"]].sum()
zoom_fin    = pred["datetime"].min() + pd.Timedelta(weeks=2)
zoom        = pred[pred["datetime"] < zoom_fin]

# ── Figura 1: intervalos de 15 min ───────────────────────────────────────────
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 11))
fig1.suptitle("XGBoost — Predicción vs Real · Entradas 2025 (intervalos 15 min)", fontsize=13)

ax1.plot(pred["datetime"], pred["real"],
         color=COLOR_REAL, linewidth=0.7, linestyle="--", label="Real", zorder=3)
ax1.plot(pred["datetime"], pred["prediccion"],
         color=COLOR_PRED, linewidth=0.7, alpha=0.9, label="Predicción", zorder=2)
ax1.set_xlim(pred["datetime"].min(), pred["datetime"].max())
ax1.set_title("Año completo 2025")
ax1.set_ylabel("Entradas")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.2)

ax2.fill_between(zoom["datetime"], zoom["real"], zoom["prediccion"],
                 color=COLOR_FILL, label="Error", zorder=1)
ax2.plot(zoom["datetime"], zoom["real"],
         color=COLOR_REAL, linewidth=1.4, linestyle="--", label="Real", zorder=3)
ax2.plot(zoom["datetime"], zoom["prediccion"],
         color=COLOR_PRED, linewidth=1.4, alpha=0.9, label="Predicción", zorder=2)
ax2.set_xlim(zoom["datetime"].min(), zoom["datetime"].max())
ax2.set_title("Zoom: primeras 2 semanas de enero 2025")
ax2.set_ylabel("Entradas")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2)

fig1.tight_layout()
fig1.savefig("prediccion_2025_15min.png", dpi=150, bbox_inches="tight")

# ── Figura 2: totales diarios ─────────────────────────────────────────────────
def fmt_millones(x, _):
    return f"{x/1e6:.1f}M"

fig2, ax3 = plt.subplots(figsize=(22, 6))
fig2.suptitle("XGBoost — Predicción vs Real · Entradas diarias 2025", fontsize=13)

ax3.fill_between(pred_diario.index, pred_diario["real"], pred_diario["prediccion"],
                 color=COLOR_FILL, label="Error", zorder=1)
ax3.plot(pred_diario.index, pred_diario["prediccion"],
         color=COLOR_PRED, linewidth=1.8, alpha=0.9, label="Predicción", zorder=3)
ax3.plot(pred_diario.index, pred_diario["real"],
         color=COLOR_REAL, linewidth=1.8, label="Real", zorder=2)
ax3.set_xlim(pred_diario.index.min(), pred_diario.index.max())
ax3.set_ylim(bottom=0)
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_millones))
ax3.set_ylabel("Entradas diarias")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.2)

fig2.tight_layout()
fig2.savefig("prediccion_2025_diaria.png", dpi=150, bbox_inches="tight")

plt.show()
print("Guardado: prediccion_2025_15min.png  |  prediccion_2025_diaria.png")