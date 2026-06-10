"""
Genera gráficas de predicción vs. real (2026) para cada CSV en
outputs/predicciones/PORTALES/, generadas por xgboost_todas_estaciones.py
(período 2026-01-01 a 2026-05-01, ventana móvil) para los 11 portales/cabeceras
con código "xx000".

Versión para documento (formato cuadrado, 2 paneles: zoom 2 semanas + perfil
horario — sin el panel de serie completa), con título grande y colores
saturados para que se distingan bien en impreso/doble columna. Métricas
mostradas: MAE, RMSE y WMAPE (MAPE omitido).

Calcula las métricas contra los datos reales de 2026 ya disponibles en
outputs/parquet/.

Guarda una figura .png por estación/dirección en outputs/FIGURAS/PORTALES/ y
un resumen de métricas en outputs/predicciones/PORTALES/resumen_visualizacion.csv
"""

import sys
import re
import glob
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import RUTA_PARQUET, RUTA_PREDS, RUTA_FIGURAS, PROYECTO_RAIZ

CARPETA_PREDICCION = "PORTALES"
RUTA_PRED_PORTALES = RUTA_PREDS / CARPETA_PREDICCION
RUTA_FIG_PORTALES = RUTA_FIGURAS / "PORTALES"
RUTA_FIG_PORTALES.mkdir(parents=True, exist_ok=True)

DIRECCIONES = ["entradas", "salidas"]

PORTALES = [
    "02000", "03000", "04000", "05000", "06000",
    "07000", "08000", "09000", "10000", "12000", "40000",
]

# Colores saturados, fáciles de diferenciar en impreso/doble columna:
# real en azul fuerte sólido, predicción en rojo punteado.
COLOR_REAL = "#1565C0"
COLOR_PRED = "#E53935"
COLOR_FDS  = "#FFE082"
COLOR_ACC  = "#BDBDBD"


def normalizar_nombre(texto: str) -> str:
    """Slug seguro para nombres de archivo — debe coincidir con
    xgboost_todas_estaciones.py para que las rutas calcen."""
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    texto = re.sub(r"[^a-zA-Z0-9]+", "_", texto).strip("_")
    return texto or "estacion"


def cargar_real(direccion, station_id, inicio, fin):
    partes = []
    for archivo in sorted(glob.glob(str(RUTA_PARQUET / f"*-{direccion}.parquet"))):
        df_m = pd.read_parquet(archivo)
        df_est = df_m[df_m["station_id"] == station_id]
        if not df_est.empty:
            partes.append(df_est[["datetime", direccion]])
    if not partes:
        return pd.DataFrame(columns=["datetime", direccion])
    real = (pd.concat(partes, ignore_index=True)
            .assign(datetime=lambda d: pd.to_datetime(d["datetime"]))
            .groupby("datetime", as_index=False)[direccion].sum()
            .sort_values("datetime").reset_index(drop=True))
    return real[(real["datetime"] >= inicio) & (real["datetime"] <= fin)].reset_index(drop=True)


def graficar_estacion(direccion, station_id, station_name):
    archivo = RUTA_PRED_PORTALES / f"{direccion}_{station_id}_{normalizar_nombre(station_name)}.csv"
    if not archivo.exists():
        print(f"  [SKIP] No encontrado: {archivo.name}")
        return None

    pred = (pd.read_csv(archivo, parse_dates=["datetime"])
            .sort_values("datetime").reset_index(drop=True))
    inicio, fin = pred["datetime"].min(), pred["datetime"].max()

    real_pred = cargar_real(direccion, station_id, inicio, fin)

    comp = None
    mae = rmse = mape = wmape = None
    if len(real_pred) > 0:
        comp = pred.merge(real_pred, on="datetime", how="inner")
        if len(comp) > 0:
            mae   = mean_absolute_error(comp[direccion], comp["prediccion"])
            rmse  = mean_squared_error(comp[direccion], comp["prediccion"]) ** 0.5
            mape  = (np.abs(comp[direccion] - comp["prediccion"]) /
                     comp[direccion].replace(0, np.nan)).mean() * 100
            wmape = (comp[direccion] - comp["prediccion"]).abs().sum() / comp[direccion].sum() * 100

    # Formato cuadrado para documento a doble columna: 2 paneles apilados
    # (zoom 2 semanas + perfil horario), sin la serie completa.
    fig, axes = plt.subplots(2, 1, figsize=(8, 8.5))

    titulo = f"{station_name} ({station_id}) — {direccion.capitalize()}"
    if mae is not None:
        titulo += f"\nMAE: {mae:,.0f}   RMSE: {rmse:,.0f}   WMAPE: {wmape:.1f}%"
    fig.suptitle(titulo, fontsize=15, fontweight="bold")

    # Panel 1: zoom primeras 2 semanas
    corte_2s = inicio + pd.Timedelta(weeks=2)
    mask_2s = pred["datetime"] < corte_2s
    ax = axes[0]
    if comp is not None:
        mask_2s_r = comp["datetime"] < corte_2s
        ax.plot(comp.loc[mask_2s_r, "datetime"], comp.loc[mask_2s_r, direccion],
                color=COLOR_REAL, lw=1.8, label="Real", zorder=2)
    ax.plot(pred.loc[mask_2s, "datetime"], pred.loc[mask_2s, "prediccion"],
            color=COLOR_PRED, lw=1.8, linestyle="--", alpha=0.9, label="Predicción", zorder=3)
    for fecha in pd.date_range(inicio, corte_2s, freq="D"):
        if fecha.weekday() >= 5:
            ax.axvspan(fecha, fecha + pd.Timedelta(days=1), alpha=0.3, color=COLOR_FDS, zorder=0)
    ax.set_title("Primeras 2 semanas (fines de semana sombreados)", fontsize=11)
    ax.set_ylabel(f"{direccion.capitalize()} (15 min)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, color=COLOR_ACC)

    # Panel 2: perfil horario promedio
    ax = axes[1]
    if comp is not None:
        perfil_real = comp.groupby(comp["datetime"].dt.hour)[direccion].mean()
        ax.plot(perfil_real.index, perfil_real.values,
                color=COLOR_REAL, lw=2.2, marker="o", ms=4, label="Real", zorder=2)
    perfil_pred = pred.groupby(pred["datetime"].dt.hour)["prediccion"].mean()
    ax.plot(perfil_pred.index, perfil_pred.values,
            color=COLOR_PRED, lw=2.2, linestyle="--", marker="s", ms=4, alpha=0.9,
            label="Predicción", zorder=3)
    ax.set_title("Perfil horario promedio", fontsize=11)
    ax.set_xlabel("Hora del día")
    ax.set_ylabel(f"{direccion.capitalize()} promedio")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, color=COLOR_ACC)

    fig.subplots_adjust(top=0.85, hspace=0.45)
    salida = RUTA_FIG_PORTALES / f"{direccion}_{station_id}_{normalizar_nombre(station_name)}.png"
    plt.savefig(salida, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "direccion": direccion,
        "station_id": station_id,
        "station_name": station_name,
        "figura": salida.name,
        "mae": round(mae, 2) if mae is not None else None,
        "rmse": round(rmse, 2) if rmse is not None else None,
        "mape": round(mape, 2) if mape is not None else None,
        "wmape": round(wmape, 2) if wmape is not None else None,
    }


def main():
    catalogo = pd.read_parquet(PROYECTO_RAIZ / "outputs" / "catalogo_estaciones.parquet")
    nombres = (catalogo[["station_id", "station_name"]]
               .drop_duplicates().set_index("station_id")["station_name"])

    filas = []
    for direccion in DIRECCIONES:
        for station_id in PORTALES:
            station_name = nombres.get(station_id, station_id)
            print(f"Graficando {direccion} {station_id} {station_name}...")
            fila = graficar_estacion(direccion, station_id, station_name)
            if fila is not None:
                if fila["mae"] is not None:
                    print(f"  MAE={fila['mae']}  RMSE={fila['rmse']}  "
                          f"MAPE={fila['mape']}%  WMAPE={fila['wmape']}%")
                filas.append(fila)

    resumen = pd.DataFrame(filas)
    ruta_resumen = RUTA_PRED_PORTALES / "resumen_visualizacion.csv"
    resumen.to_csv(ruta_resumen, index=False)
    print(f"\nFiguras guardadas en: {RUTA_FIG_PORTALES}")
    print(f"Resumen de métricas (real 2026 vs predicción) en: {ruta_resumen}")


if __name__ == "__main__":
    main()
