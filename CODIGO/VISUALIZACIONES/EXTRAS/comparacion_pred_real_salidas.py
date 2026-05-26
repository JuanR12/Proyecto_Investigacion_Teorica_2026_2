import glob
import pandas as pd
import matplotlib.pyplot as plt

PARQUET_DIR = r"C:\Users\Juanshots\Desktop\PROYECTO_INV_TEO\DATOS LIMPIOS\ENTRADA Y SALIDA MENSUAL\parquet"
PRED_CSV    = r"C:\Users\Juanshots\Desktop\PROYECTO_INV_TEO\DATOS LIMPIOS\ENTRADA Y SALIDA MENSUAL\PREDICCIONES\validaciones_salidas_pred_2025.csv"

real = pd.concat(
    [pd.read_parquet(f).groupby("datetime")["salidas"].sum()
     for f in sorted(glob.glob(f"{PARQUET_DIR}\\2025-*-salidas.parquet"))],
).groupby("datetime").sum().reset_index()
real["datetime"] = pd.to_datetime(real["datetime"])

pred = pd.read_csv(PRED_CSV, parse_dates=["datetime"])
df   = real.merge(pred, on="datetime", how="inner").sort_values("datetime")

fig, ax = plt.subplots(figsize=(20, 5))
ax.plot(df["datetime"], df["salidas"],   color="#2563eb", lw=0.6, label="Real")
ax.plot(df["datetime"], df["prediccion"], color="#dc2626", lw=0.6, alpha=0.8, label="Predicción")
ax.set_ylabel("Entradas por intervalo de 15 min")
ax.legend()
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("comparativa_2025.png", dpi=150, bbox_inches="tight")
plt.show()