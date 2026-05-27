import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PARQUET_DIR = r"C:\Users\gordi\Desktop\Transmilenio\Proyecto_Investigacion_Teorica_2026_2\outputs\parquet"
PRED_CSV    = r"C:\Users\gordi\Desktop\Transmilenio\Proyecto_Investigacion_Teorica_2026_2\outputs\predicciones\predicciones_2025_prueba.csv"

pred = pd.read_csv(PRED_CSV, parse_dates=["datetime"])
pred = pred.sort_values("datetime").reset_index(drop=True)

INICIO = pred["datetime"].min()
FIN    = pred["datetime"].max()

# Solo carga los parquets que solapan con el rango predicho
parquets = sorted(glob.glob(f"{PARQUET_DIR}\\*-entradas.parquet"))
partes = []
for f in parquets:
    df_mes = pd.read_parquet(f)
    df_mes["datetime"] = pd.to_datetime(df_mes["datetime"])
    df_mes = df_mes[(df_mes["datetime"] >= INICIO) & (df_mes["datetime"] <= FIN)]
    if df_mes.empty:
        continue
    partes.append(df_mes.groupby("datetime")["entradas"].sum())

if not partes:
    raise ValueError(f"No hay datos reales entre {INICIO.date()} y {FIN.date()}")

real = pd.concat(partes).groupby("datetime").sum().reset_index()
real.columns = ["datetime", "entradas"]

df = real.merge(pred, on="datetime", how="inner").sort_values("datetime")
print(f"Rango graficado: {df['datetime'].min().date()} -> {df['datetime'].max().date()}  ({len(df):,} intervalos)")

fig, ax = plt.subplots(figsize=(20, 5))
ax.plot(df["datetime"], df["entradas"],   color="#7aa9ce", lw=0.8, label="Real")
ax.plot(df["datetime"], df["prediccion"], color="#fdc0cc", lw=0.8, alpha=0.9, label="Predicción")
ax.set_ylabel("Entradas por intervalo de 15 min")
ax.legend()
ax.grid(True, alpha=0.25, color="#d5b9e4")
plt.tight_layout()

RUTA_SALIDA = Path(PRED_CSV).parent / "comparativa_pred_real.png"
plt.savefig(RUTA_SALIDA, dpi=150, bbox_inches="tight")
plt.show()
print(f"Figura guardada en {RUTA_SALIDA}")