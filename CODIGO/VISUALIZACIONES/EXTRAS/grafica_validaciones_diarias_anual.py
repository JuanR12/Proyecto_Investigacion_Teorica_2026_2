import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RUTA = r"C:\Users\Juanshots\Desktop\PROYECTO_INV_TEO\DATOS LIMPIOS\ENTRADA Y SALIDA MENSUAL\parquet"

def cargar_serie(tipo):
    archivos = sorted(glob.glob(f"{RUTA}\\*-{tipo}.parquet"))
    partes = []
    for f in archivos:
        df = pd.read_parquet(f)
        agg = df.groupby("datetime")[tipo].sum()
        partes.append(agg)
    serie = pd.concat(partes).sort_index()
    # Resample a diario para que la gráfica sea legible
    return serie.resample("D").sum()

print("Cargando entradas...")
entradas = cargar_serie("entradas")
print("Cargando salidas...")
salidas  = cargar_serie("salidas")

años = sorted(entradas.index.year.unique())
fig, axes = plt.subplots(len(años), 1, figsize=(20, 3 * len(años)), sharex=False)
fig.suptitle("Sistema Troncal TransMilenio — Entradas y Salidas diarias", fontsize=14, y=1.01)

for ax, año in zip(axes, años):
    e = entradas[entradas.index.year == año]
    s = salidas[salidas.index.year == año]
    ax.plot(e.index, e.values, color="#2563eb", linewidth=0.9, label="Entradas")
    ax.plot(s.index, s.values, color="#dc2626", linewidth=0.9, label="Salidas")
    ax.set_xlim(e.index.min(), e.index.max())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.set_ylabel(str(año), fontsize=11, fontweight="bold", rotation=0, labelpad=35, va="center")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="x", labelsize=8)

plt.tight_layout()
plt.savefig("serie_anual_transmilenio.png", dpi=150, bbox_inches="tight")
plt.show()
print("Guardado en serie_anual_transmilenio.png")