import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# CONFIGURACIÓN
# =========================================================
ruta_archivos = Path("./datos/")
fecha = pd.Timestamp("2026-03-10")

# =========================================================
# CARGAR DATOS
# =========================================================
fecha_str = fecha.strftime("%Y%m%d")

# Entradas por troncal
df_ent = pd.read_csv(ruta_archivos / f"{fecha_str}.csv", usecols=["Linea", "Fecha_Transaccion"])
df_ent["Linea"] = df_ent["Linea"].str.strip()
df_ent["Fecha_Transaccion"] = pd.to_datetime(df_ent["Fecha_Transaccion"], errors="coerce")
df_ent = df_ent[
    (df_ent["Fecha_Transaccion"].dt.date == fecha.date()) &
    (df_ent["Fecha_Transaccion"].dt.hour >= 3)
]
df_ent["Intervalo"] = df_ent["Fecha_Transaccion"].dt.floor("15min")

# Salidas totales del sistema
df_sal = pd.read_csv(ruta_archivos / f"S{fecha_str}.csv", usecols=["Fecha_Transaccion", "Tiempo", "Salidas_S"])
df_sal["Fecha_Transaccion"] = pd.to_datetime(
    df_sal["Fecha_Transaccion"].astype(str) + " " + df_sal["Tiempo"].astype(str), errors="coerce"
)
df_sal = df_sal[
    (df_sal["Fecha_Transaccion"].dt.date == fecha.date()) &
    (df_sal["Fecha_Transaccion"].dt.hour >= 3)
]
df_sal["Intervalo"] = df_sal["Fecha_Transaccion"].dt.floor("15min")

# =========================================================
# ÍNDICE COMPLETO
# =========================================================
indice = pd.date_range(
    start=fecha.replace(hour=3, minute=0, second=0),
    end=fecha.replace(hour=23, minute=45, second=0),
    freq="15min"
)

# =========================================================
# PIVOT ENTRADAS POR TRONCAL
# =========================================================
conteo_ent = (
    df_ent.groupby(["Intervalo", "Linea"])
    .size()
    .reset_index(name="Validaciones")
)
pivot_ent = conteo_ent.pivot(index="Intervalo", columns="Linea", values="Validaciones")
pivot_ent = pivot_ent.reindex(indice).fillna(0)

# Ordenar troncales por volumen total
pivot_ent = pivot_ent[pivot_ent.sum().sort_values(ascending=True).index]

# =========================================================
# SALIDAS TOTALES
# =========================================================
sal_total = (
    df_sal.groupby("Intervalo")["Salidas_S"]
    .sum()
    .reindex(indice)
    .fillna(0)
)

# =========================================================
# HORA DECIMAL PARA EJE Y
# =========================================================
horas = indice.hour + indice.minute / 60

# =========================================================
# GRAFICAR
# =========================================================
n_troncales = len(pivot_ent.columns)
cmap = plt.get_cmap("tab20", n_troncales)
colores = [cmap(i) for i in range(n_troncales)]

fig, ax = plt.subplots(figsize=(16, 12))

# --- LADO POSITIVO: entradas apiladas por troncal ---
base = np.zeros(len(indice))
for i, col in enumerate(pivot_ent.columns):
    valores = pivot_ent[col].values
    ax.fill_betweenx(horas, base, base + valores,
                     alpha=0.85, color=colores[i], label=col)
    base += valores

# --- LADO NEGATIVO: salidas totales ---
ax.fill_betweenx(horas, 0, -sal_total.values,
                 alpha=0.75, color="#37474F", label="Salidas (sistema)")

# Línea divisoria en x=0
ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")

# --- EJES ---
ax.set_ylim(3, 24)
ax.set_yticks(range(3, 24))
ax.set_yticklabels([f"{h:02d}:00" for h in range(3, 24)], fontsize=9)
ax.invert_yaxis()  # 03:00 arriba

# Etiquetas eje X: mostrar valores absolutos
xticks = ax.get_xticks()
ax.set_xticklabels([f"{abs(int(x)):,}" for x in xticks], fontsize=8)

# Anotaciones de lado
xmax = base.max()
ax.text(xmax * 0.5,  3.3, "VALIDACIONES →", fontsize=10, color="dimgray", ha="center")
ax.text(-sal_total.max() * 0.5, 3.3, "← SALIDAS",     fontsize=10, color="dimgray", ha="center")

ax.set_xlabel("Número de transacciones")
ax.set_ylabel("Hora del día")
ax.grid(True, axis="x", alpha=0.2, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Leyenda fuera
ax.legend(
    loc="lower right",
    bbox_to_anchor=(1.28, 0),
    fontsize=7.5,
    frameon=False,
    title="Troncal / Zona",
    title_fontsize=8.5
)

fig.suptitle(
    f"Flujo del sistema troncal — {fecha.strftime('%d/%m/%Y')}",
    fontsize=14, fontweight="bold", y=0.98
)

Path("media").mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig(f"media/butterfly_{fecha_str}.png", dpi=150, bbox_inches="tight")
plt.show()