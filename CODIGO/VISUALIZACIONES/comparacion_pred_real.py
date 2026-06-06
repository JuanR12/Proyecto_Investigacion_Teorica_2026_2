import glob
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Importar rutas desde config.py (raíz del proyecto). Ver config.py para personalizar.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RUTA_PARQUET as PARQUET_DIR, RUTA_PREDS, RUTA_FIGURAS

PRED_CSV = RUTA_PREDS / "predicciones_2025_prueba.csv"

real = pd.concat(
    [pd.read_parquet(f).groupby("datetime")["entradas"].sum()
     for f in sorted(glob.glob(str(PARQUET_DIR / "2025-*-entradas.parquet")))],
).groupby("datetime").sum().reset_index()
real["datetime"] = pd.to_datetime(real["datetime"])

pred = pd.read_csv(PRED_CSV, parse_dates=["datetime"])
df   = real.merge(pred, on="datetime", how="inner").sort_values("datetime")

fig, ax = plt.subplots(figsize=(20, 5))
ax.plot(df["datetime"], df["entradas"],   color="#2563eb", lw=0.6, label="Real")
ax.plot(df["datetime"], df["prediccion"], color="#dc2626", lw=0.6, alpha=0.8, label="Predicción")
ax.set_ylabel("Entradas por intervalo de 15 min")
ax.legend()
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "comparativa_2025_entradas.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Figura guardada en {RUTA_FIGURAS / 'comparativa_2025_entradas.png'}")