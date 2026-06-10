#Antes de correr el codigo no olviden descargar los paquetes y librerias:
#pip install xgboost scikit-learn pandas numpy matplotlib pyarrow

"""
Predicción de flujos de entradas por estación - TransMilenio
XGBoost con features de calendario y rezagos

Copia de xgboost_transmilenio_v1_mejorado.py adaptada para entrenar y predecir
estación por estación (en vez de a nivel de sistema agregado), con la misma
retroalimentación: train 2021-2023, validación 2024, test 2025 (simulación de
despliegue en vivo).

Para cada estación de ESTACIONES se genera una figura independiente con el
mismo formato cuadrado para documento a doble columna:
  - Panel 1: promedio diario, segundo semestre 2025 (simulación en vivo)
  - Panel 2: zoom segunda semana de abril 2025
  - Métricas MAE, RMSE y WMAPE (sin MAPE), colores azul/rojo saturados

Estructura esperada de archivos:
    outputs/parquet/
        2021-01-entradas.parquet
        2021-02-entradas.parquet
        ...
        2025-12-entradas.parquet

Cada parquet tiene columnas: linea_id, station_id, datetime, entradas
"""

import glob
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Importar rutas desde config.py (raíz del proyecto). Ver config.py para personalizar.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import RUTA_PARQUET, RUTA_FIGURAS

# ─────────────────────────────────────────────
# ESTACIONES A PROCESAR
# ─────────────────────────────────────────────

ESTACIONES = [
    ("07106", "EL_CAMPIN"),
    ("07107", "U_NACIONAL"),
    ("06107", "Ciudad_Universitaria"),
    ("06106", "Recinto_Ferial"),
    ("02101", "Toberin"),
]

RUTA_FIG_ESTACIONES = RUTA_FIGURAS / "ESTACIONES_MEJORADO"
RUTA_FIG_ESTACIONES.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "hour_sin", "hour_cos",
    "min_sin", "min_cos",
    "dow_sin", "dow_cos",
    "week_sin", "week_cos",
    "month", "year",
    "es_fin_de_semana",
    "lag_15m", "lag_30m", "lag_1h", "lag_24h", "lag_1w",
    "rolling_1h", "rolling_24h",
]

COLOR_REAL = "#1976D2"   # azul — datos reales
COLOR_PRED = "#E53935"   # rojo punteado — predicción XGBoost


def calcular_wmape(y_true, y_pred):
    """WMAPE (MAPE ponderado por volumen): sum(|real-pred|) / sum(real) * 100.
    Más robusto que el MAPE simple porque los intervalos de baja demanda
    no distorsionan el promedio."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100


def cargar_serie_estacion(archivos, station_id):
    """Carga y agrega la serie temporal de 15 min de entradas para una estación."""
    partes = []
    for archivo in archivos:
        df_mes = pd.read_parquet(archivo)
        df_est = df_mes[df_mes["station_id"] == station_id]
        if df_est.empty:
            continue
        partes.append(df_est.groupby("datetime", as_index=False)["entradas"].sum())

    if not partes:
        raise ValueError(f"sin datos para la estación {station_id}")

    df = pd.concat(partes, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def construir_features(df):
    """Rellena huecos temporales y agrega calendario, rezagos y promedios móviles."""
    rango_completo = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="15min")
    huecos = rango_completo.difference(df["datetime"])
    if len(huecos) > 0:
        df = df.set_index("datetime").reindex(rango_completo, fill_value=0).reset_index()
        df.columns = ["datetime", "entradas"]

    df["hour_sin"]  = np.sin(2 * np.pi * df["datetime"].dt.hour / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["datetime"].dt.hour / 24)
    df["min_sin"]   = np.sin(2 * np.pi * df["datetime"].dt.minute / 60)
    df["min_cos"]   = np.cos(2 * np.pi * df["datetime"].dt.minute / 60)
    df["dow_sin"]   = np.sin(2 * np.pi * df["datetime"].dt.weekday / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["datetime"].dt.weekday / 7)
    df["week_sin"]  = np.sin(2 * np.pi * df["datetime"].dt.isocalendar().week.astype(int) / 52)
    df["week_cos"]  = np.cos(2 * np.pi * df["datetime"].dt.isocalendar().week.astype(int) / 52)
    df["month"]     = df["datetime"].dt.month
    df["year"]      = df["datetime"].dt.year
    df["es_fin_de_semana"] = (df["datetime"].dt.weekday >= 5).astype(int)

    df["lag_15m"]   = df["entradas"].shift(1)
    df["lag_30m"]   = df["entradas"].shift(2)
    df["lag_1h"]    = df["entradas"].shift(4)
    df["lag_24h"]   = df["entradas"].shift(96)
    df["lag_1w"]    = df["entradas"].shift(672)
    df["rolling_1h"]  = df["entradas"].shift(1).rolling(4).mean()
    df["rolling_24h"] = df["entradas"].shift(1).rolling(96).mean()

    return df.dropna().reset_index(drop=True)


def procesar_estacion(archivos, station_id, station_name):
    print(f"\n{'='*65}")
    print(f"Estación {station_id} - {station_name}")
    print(f"{'='*65}")

    df = cargar_serie_estacion(archivos, station_id)
    df = construir_features(df)

    print(f"Rango temporal: {df['datetime'].min()} -> {df['datetime'].max()}  ({len(df):,} filas)")

    mask_train = df["datetime"] < "2024-01-01"
    mask_val   = (df["datetime"] >= "2024-01-01") & (df["datetime"] < "2025-01-01")
    mask_test  = (df["datetime"] >= "2025-01-01") & (df["datetime"] < "2026-01-01")

    train = df[mask_train]
    val   = df[mask_val]
    test  = df[mask_test]

    if len(train) < 1000 or len(val) < 100 or len(test) < 100:
        print(f"  [SKIP] datos insuficientes (train={len(train)}, val={len(val)}, test={len(test)})")
        return

    X_train, y_train = train[FEATURES], train["entradas"]
    X_val,   y_val   = val[FEATURES],   val["entradas"]
    X_test,  y_test  = test[FEATURES],  test["entradas"]

    print(f"Train:      {train['datetime'].min().date()} -> {train['datetime'].max().date()} ({len(train):,} filas)")
    print(f"Validación: {val['datetime'].min().date()}   -> {val['datetime'].max().date()}   ({len(val):,} filas)")
    print(f"Test:       {test['datetime'].min().date()}  -> {test['datetime'].max().date()}  ({len(test):,} filas)")

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="mae",
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    pred_test = np.clip(model.predict(X_test), 0, None)

    mae_test   = mean_absolute_error(y_test, pred_test)
    rmse_test  = mean_squared_error(y_test, pred_test) ** 0.5
    wmape_test = calcular_wmape(y_test, pred_test)
    print(f"Test 2025  |  MAE={mae_test:,.1f}  RMSE={rmse_test:,.1f}  WMAPE={wmape_test:.1f}%")

    # ─────────────────────────────────────────────
    # VISUALIZACIÓN
    # ─────────────────────────────────────────────
    # Formato cuadrado para documento a doble columna: 2 paneles apilados,
    # colores saturados (azul sólido = real, rojo punteado = predicción),
    # título grande y corto, sin MAPE.

    fig, axes = plt.subplots(2, 1, figsize=(8, 8.5))
    fig.suptitle(
        f"XGBoost TransMilenio — {station_name.replace('_', ' ')} (Entradas)\n"
        f"MAE: {mae_test:,.0f}   RMSE: {rmse_test:,.0f}   WMAPE: {wmape_test:.1f}%",
        fontsize=15, fontweight="bold"
    )

    # -- Panel 1: promedio diario, segundo semestre 2025 (simulación en vivo)
    diario = (
        pd.DataFrame({"datetime": test["datetime"].values, "real": y_test.values, "prediccion": pred_test})
        .assign(fecha=lambda d: d["datetime"].dt.floor("D"))
        .groupby("fecha", as_index=False)[["real", "prediccion"]].mean()
    )
    diario = diario[(diario["fecha"] >= "2025-07-01") & (diario["fecha"] < "2026-01-01")]

    ax = axes[0]
    ax.plot(diario["fecha"], diario["real"], label="Real",
            color=COLOR_REAL, linewidth=1.6, alpha=0.9, zorder=2)
    ax.plot(diario["fecha"], diario["prediccion"], label="Predicción",
            color=COLOR_PRED, linewidth=1.6, linestyle="--", alpha=0.9, zorder=3)
    ax.set_title("Promedio diario — segundo semestre 2025 (simulación en vivo)", fontsize=11)
    ax.set_ylabel("Entradas promedio (15 min)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # -- Panel 2: zoom segunda semana de abril 2025
    zoom_inicio = pd.Timestamp("2025-04-07")  # lunes, segunda semana de abril 2025
    zoom_fin    = pd.Timestamp("2025-04-21")  # +2 semanas
    mask_zoom = (test["datetime"] >= zoom_inicio) & (test["datetime"] < zoom_fin)

    zoom_real = y_test[mask_zoom].values
    zoom_pred = pred_test[mask_zoom]
    mae_zoom   = mean_absolute_error(zoom_real, zoom_pred) if mask_zoom.any() else np.nan
    rmse_zoom  = mean_squared_error(zoom_real, zoom_pred) ** 0.5 if mask_zoom.any() else np.nan
    wmape_zoom = calcular_wmape(zoom_real, zoom_pred) if mask_zoom.any() else np.nan

    ax = axes[1]
    ax.plot(test.loc[mask_zoom, "datetime"], zoom_real,
            label="Real", color=COLOR_REAL, linewidth=1.6, alpha=0.9, zorder=2)
    ax.plot(test.loc[mask_zoom, "datetime"], zoom_pred,
            label="Predicción", color=COLOR_PRED, linewidth=1.6,
            linestyle="--", alpha=0.85, zorder=3)
    ax.set_title(
        f"Zoom: segunda semana de abril 2025   |   "
        f"MAE: {mae_zoom:,.0f}   RMSE: {rmse_zoom:,.0f}   WMAPE: {wmape_zoom:.1f}%",
        fontsize=11
    )
    ax.set_ylabel("Entradas (15 min)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Ajuste manual de márgenes en lugar de tight_layout: evita que el suptitle
    # (dos líneas) se solape con el título del panel superior, y deja espacio
    # para las etiquetas de fecha rotadas en la parte inferior.
    fig.subplots_adjust(top=0.87, hspace=0.45)

    salida = RUTA_FIG_ESTACIONES / f"resultados_xgboost_entradas_{station_id}_{station_name}.png"
    plt.savefig(salida, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura guardada en {salida}")


def main():
    archivos = sorted(glob.glob(str(RUTA_PARQUET / "*-entradas.parquet")))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos en {RUTA_PARQUET}")

    for station_id, station_name in ESTACIONES:
        procesar_estacion(archivos, station_id, station_name)

    print(f"\nTodas las estaciones procesadas. Figuras en: {RUTA_FIG_ESTACIONES}")


if __name__ == "__main__":
    main()
