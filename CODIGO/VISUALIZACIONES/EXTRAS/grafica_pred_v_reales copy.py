import glob
import pandas as pd
import matplotlib.pyplot as plt

PARQUET_DIR = r"C:\Users\Juanshots\Desktop\PROYECTO_INV_TEO\DATOS LIMPIOS\ENTRADA Y SALIDA MENSUAL\parquet"
PRED_CSV    = r"C:\Users\Juanshots\Desktop\PROYECTO_INV_TEO\DATOS LIMPIOS\ENTRADA Y SALIDA MENSUAL\PREDICCIONES\predicciones_2026_prueba1.csv"


FECHA_INICIO = "2026-01-01"
FECHA_FIN    = "2026-04-01"

real = pd.concat(
    [pd.read_parquet(f).groupby("datetime")["entradas"].sum()
     for f in sorted(glob.glob(f"{PARQUET_DIR}\\2026-*-entradas.parquet"))],
).groupby("datetime").sum().reset_index()
real["datetime"] = pd.to_datetime(real["datetime"])

pred = pd.read_csv(PRED_CSV, parse_dates=["datetime"])
df   = real.merge(pred, on="datetime").sort_values("datetime")
df   = df[(df["datetime"] >= FECHA_INICIO) & (df["datetime"] <= FECHA_FIN)]

fig, ax = plt.subplots(figsize=(20, 5))
ax.plot(df["datetime"], df["entradas"],   color="#2563eb", lw=0.6, label="Real")
ax.plot(df["datetime"], df["prediccion"], color="#dc2626", lw=0.6, alpha=0.8, label="Predicción")
ax.set_ylabel("Entradas por intervalo de 15 min")
ax.legend()
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("comparativa_2026.png", dpi=150, bbox_inches="tight")
plt.show()