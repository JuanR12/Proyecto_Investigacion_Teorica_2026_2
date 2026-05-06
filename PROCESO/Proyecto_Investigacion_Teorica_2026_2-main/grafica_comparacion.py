import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# CONFIGURACIÓN
# =========================================================
ruta_archivos = Path("./datos/")
codigo_estacion = "09000"
fechas = pd.date_range("2026-03-10", "2026-03-14", freq="D")
dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]

# Paleta más distinguible y agradable
colores = ["#2196F3", "#E53935", "#43A047", "#FB8C00", "#8E24AA"]

# =========================================================
# OBTENER NOMBRE REAL DE LA ESTACIÓN
# =========================================================
def obtener_nombre_estacion():
    fecha_str = fechas[0].strftime("%Y%m%d")
    df = pd.read_csv(ruta_archivos / f"{fecha_str}.csv", usecols=["Estacion_Parada"])
    df["Estacion_Parada"] = df["Estacion_Parada"].str.strip()
    encontradas = df[df["Estacion_Parada"].str.contains(codigo_estacion, na=False)]["Estacion_Parada"].unique()
    return encontradas[0] if len(encontradas) > 0 else f"Estación {codigo_estacion}"

nombre_estacion = obtener_nombre_estacion()
print(f"Estación encontrada: {nombre_estacion}")

# =========================================================
# FUNCIÓN AUXILIAR
# =========================================================
def procesar_dia(fecha):
    fecha_str = fecha.strftime("%Y%m%d")
    df_ent = pd.read_csv(ruta_archivos / f"{fecha_str}.csv", usecols=["Estacion_Parada", "Fecha_Transaccion"])
    df_ent["Estacion_Parada"] = df_ent["Estacion_Parada"].str.strip()
    df_ent = df_ent[df_ent["Estacion_Parada"].str.contains(codigo_estacion, na=False)]
    df_ent["Fecha_Transaccion"] = pd.to_datetime(df_ent["Fecha_Transaccion"], errors="coerce")
    df_ent = df_ent[
        (df_ent["Fecha_Transaccion"].dt.date == fecha.date()) &
        (df_ent["Fecha_Transaccion"].dt.hour >= 3)
    ]

    df_sal = pd.read_csv(ruta_archivos / f"S{fecha_str}.csv", usecols=["Estacion", "Fecha_Transaccion", "Tiempo", "Salidas_S"])
    df_sal["Estacion"] = df_sal["Estacion"].str.strip()
    df_sal = df_sal[df_sal["Estacion"].str.contains(codigo_estacion, na=False)]
    df_sal["Fecha_Transaccion"] = pd.to_datetime(
        df_sal["Fecha_Transaccion"].astype(str) + " " + df_sal["Tiempo"].astype(str), errors="coerce"
    )
    df_sal = df_sal[
        (df_sal["Fecha_Transaccion"].dt.date == fecha.date()) &
        (df_sal["Fecha_Transaccion"].dt.hour >= 3)
    ]

    df_ent["Intervalo_15min"] = df_ent["Fecha_Transaccion"].dt.floor("15min")
    df_sal["Intervalo_15min"] = df_sal["Fecha_Transaccion"].dt.floor("15min")

    ent_15 = df_ent.groupby("Intervalo_15min").size().reset_index(name="Ingresos")
    sal_15 = df_sal.groupby("Intervalo_15min")["Salidas_S"].sum().reset_index(name="Salidas")

    indice_completo = pd.date_range(
        start=fecha.replace(hour=3, minute=0, second=0),
        end=fecha.replace(hour=23, minute=45, second=0),
        freq="15min"
    )
    df_indice = pd.DataFrame({"Intervalo_15min": indice_completo})
    ent_15 = pd.merge(df_indice, ent_15, on="Intervalo_15min", how="left").fillna(0)
    sal_15 = pd.merge(df_indice, sal_15, on="Intervalo_15min", how="left").fillna(0)

    comp = pd.merge(ent_15, sal_15, on="Intervalo_15min", how="left")
    comp["Fecha"] = fecha
    return comp

# =========================================================
# PROCESAR
# =========================================================
df_total = pd.concat([procesar_dia(f) for f in fechas]).sort_values(["Fecha", "Intervalo_15min"])
df_total["Hora"] = df_total["Intervalo_15min"].dt.strftime("%H:%M")
horas_orden = pd.date_range("03:00", "23:45", freq="15min").strftime("%H:%M").tolist()

# =========================================================
# GRAFICAR
# =========================================================
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(16, 9),
    sharex=True,
    gridspec_kw={"hspace": 0.08}
)

for i, (fecha, dia) in enumerate(zip(fechas, dias_semana)):
    df_dia = df_total[df_total["Fecha"] == fecha].copy()
    df_dia = df_dia.set_index("Hora").reindex(horas_orden).reset_index()
    color = colores[i]
    # Etiqueta con día y fecha: "Lunes 10/03"
    label = f"{dia} {fecha.strftime('%d/%m')}"

    ax1.plot(df_dia["Hora"], df_dia["Ingresos"], color=color, linewidth=1.8)
    ax2.plot(df_dia["Hora"], df_dia["Salidas"],  color=color, linewidth=1.8, label=label)

fig.suptitle(nombre_estacion, fontsize=14, fontweight="bold", y=0.98)

ax1.set_ylabel("Validaciones")
ax2.set_ylabel("Salidas")
ax2.set_xlabel("Hora")

for ax in (ax1, ax2):
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

handles, labels = ax2.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    ncol=len(fechas),
    frameon=False,
    bbox_to_anchor=(0.5, 0.95),
    fontsize=10
)

fig.subplots_adjust(top=0.88)  # deja espacio entre título+leyenda y las gráficas

ticks_mostrar = horas_orden[::4]
ax2.set_xticks(ticks_mostrar)
ax2.set_xticklabels(ticks_mostrar, rotation=90)
fig.subplots_adjust(bottom=0.1)

plt.savefig(f"media/estacion_{codigo_estacion}.png", dpi=150, bbox_inches="tight")
plt.show()