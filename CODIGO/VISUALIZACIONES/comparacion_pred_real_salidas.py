import glob
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Importar rutas desde config.py (raíz del proyecto). Ver config.py para personalizar.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RUTA_PARQUET as PARQUET_DIR, RUTA_PREDS, RUTA_FIGURAS

PRED_CSV = RUTA_PREDS / "predicciones_2025_prueba.csv"

pred = pd.read_csv(PRED_CSV, parse_dates=["datetime"])
pred = pred.sort_values("datetime").reset_index(drop=True)

INICIO = pred["datetime"].min()
FIN    = pred["datetime"].max()

# Solo carga los parquets que solapan con el rango predicho
parquets = sorted(glob.glob(str(PARQUET_DIR / "*-salidas.parquet")))
partes = []
for f in parquets:
    df_mes = pd.read_parquet(f)
    df_mes["datetime"] = pd.to_datetime(df_mes["datetime"])
    df_mes = df_mes[(df_mes["datetime"] >= INICIO) & (df_mes["datetime"] <= FIN)]
    if df_mes.empty:
        continue
    partes.append(df_mes.groupby("datetime")["salidas"].sum())

if not partes:
    raise ValueError(f"No hay datos reales entre {INICIO.date()} y {FIN.date()}")

real = pd.concat(partes).groupby("datetime").sum().reset_index()
real.columns = ["datetime", "salidas"]

df = real.merge(pred, on="datetime", how="inner").sort_values("datetime")
print(f"Rango graficado: {df['datetime'].min().date()} -> {df['datetime'].max().date()}  ({len(df):,} intervalos)")

fig, ax = plt.subplots(figsize=(20, 5))
ax.plot(df["datetime"], df["salidas"],    color="#7aa9ce", lw=0.8, label="Real")
ax.plot(df["datetime"], df["prediccion"], color="#fdc0cc", lw=0.8, alpha=0.9, label="Predicción")
ax.set_ylabel("Salidas por intervalo de 15 min")
ax.legend()
ax.grid(True, alpha=0.25, color="#d5b9e4")
plt.tight_layout()

RUTA_SALIDA = RUTA_FIGURAS / "comparativa_pred_real_salidas.png"
plt.savefig(RUTA_SALIDA, dpi=150, bbox_inches="tight")
plt.show()
print(f"Figura guardada en {RUTA_SALIDA}")