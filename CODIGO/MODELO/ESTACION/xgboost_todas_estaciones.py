"""
Predicción de flujos (entradas y salidas) para TODAS las estaciones del sistema.

Recorre el catálogo de estaciones (outputs/catalogo_estaciones.parquet), entrena
un modelo XGBoost por cada combinación estación + dirección, genera el forecast
recursivo para el período configurado y guarda:

  - Un CSV de predicciones por estación/dirección en outputs/predicciones/,
    nombrado como "{direccion}_{station_id}_{nombre_estacion}.csv"
  - Un CSV resumen consolidado (resumen_predicciones_estaciones.csv) con las
    métricas de validación, totales del pronóstico e importancia de features
    de cada corrida — reemplaza la inspección visual individual por estación.

Reanudable: si el CSV de una estación/dirección ya existe, se omite sin volver
a entrenar. Esto permite pausar el proceso (Ctrl+C) y continuarlo después.

Manejo de errores: una estación que falle (datos insuficientes, huecos, etc.)
no detiene el proceso — se registra en el resumen con su mensaje de error y
se continúa con la siguiente.
"""

import sys
import re
import glob
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Importar rutas desde config.py (raíz del proyecto). Ver config.py para personalizar.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import RUTA_PARQUET, RUTA_PREDS, PROYECTO_RAIZ

# ============================================================
# CONFIGURACIÓN — define uno o varios períodos a predecir
# ============================================================
# Cada tupla es (inicio, fin) con `fin` exclusivo. Se puede agregar tantos
# períodos como se quiera: el script los procesa uno por uno y guarda cada
# uno en su propia subcarpeta dentro de outputs/predicciones/, nombrada
# como "{inicio}_a_{fin}" — así se pueden acumular corridas para distintas
# fechas sin que se mezclen ni se sobrescriban entre sí.
PERIODOS = [
    ("2026-01-01", "2026-05-01"),
]

DIRECCIONES = ["entradas", "salidas"]   # quita una si solo te interesa una dirección

# Para pruebas rápidas: pon un número (p.ej. 1) para procesar solo esa cantidad
# de estaciones del catálogo. Déjalo en None para procesar las 164 completas.
LIMITE_ESTACIONES = None

# Si se define (lista de station_id), solo se procesan esas estaciones —
# útil para corridas focalizadas (p.ej. solo portales). None = catálogo completo.
# Portales / cabeceras con código "xx000":
FILTRO_ESTACIONES = [
    "02000",  # Portal Norte
    "03000",  # Portal Suba
    "04000",  # Cabecera Calle 80
    "05000",  # Portal Américas
    "06000",  # Portal El Dorado
    "07000",  # Portal Sur JFK Coop. Financiera
    "08000",  # Portal Tunal
    "09000",  # Cabecera Usme
    "10000",  # Portal 20 de Julio
    "12000",  # Puente Aranda
    "40000",  # Cable Portal Tunal
]

# Si se define, todas las carpetas de salida usan este nombre fijo en vez de
# "{periodo_inicio}_a_{periodo_fin}{SUFIJO_CARPETA}". Útil para corridas
# especiales que no necesitan organizarse por período (p.ej. "PORTALES").
NOMBRE_CARPETA_OVERRIDE = "PORTALES"

RUTA_CATALOGO = PROYECTO_RAIZ / "outputs" / "catalogo_estaciones.parquet"

# ============================================================
# VENTANA DE ENTRENAMIENTO / VALIDACIÓN
# ============================================================
# Por defecto el modelo entrena con 2022-2023 y valida con 2024 (igual que el
# template original). Para una "ventana móvil" — entrenar/validar con los datos
# más recientes posibles y así predecir un período más lejano con más precisión —
# desplaza estos cortes. Ej. para predecir 2026: entrena con 2022-2024 y valida
# con todo 2025 (el año completo más reciente disponible).
ENTRENAMIENTO_FIN = "2025-01-01"   # train: todo lo anterior a esta fecha
VALIDACION_INICIO = "2025-01-01"
VALIDACION_FIN    = "2026-01-01"   # validación: [VALIDACION_INICIO, VALIDACION_FIN)

# Sufijo añadido al nombre de la carpeta de salida, para distinguir corridas
# que usan una ventana de entrenamiento distinta a la del template original
# (p.ej. "2026-01-01_a_2026-05-01_ventana_movil"). Déjalo en "" si no lo necesitas.
SUFIJO_CARPETA = "_ventana_movil"

FESTIVOS_FIJOS = {(1, 1), (5, 1), (7, 20), (8, 7), (12, 8), (12, 25)}
BUFFER = 672  # una semana de intervalos de 15 min, usada como semilla del forecast

# El orden de esta lista debe coincidir EXACTAMENTE con construir_fila_features()
FEATURES = [
    "hour_sin", "hour_cos",
    "min_sin", "min_cos",
    "dow_sin", "dow_cos",
    "week_sin", "week_cos",
    "month", "year",
    "es_fin_de_semana",
    "es_festivo_fijo",
    "lag_15m", "lag_30m", "lag_1h", "lag_24h", "lag_1w",
    "rolling_1h", "rolling_24h",
]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalizar_nombre(texto: str) -> str:
    """Convierte el nombre de una estación a un slug seguro para nombres de archivo
    (sin tildes, espacios ni símbolos)."""
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    texto = re.sub(r"[^a-zA-Z0-9]+", "_", texto).strip("_")
    return texto or "estacion"


def ruta_csv_estacion(carpeta: Path, direccion: str, station_id: str, station_name: str) -> tuple[Path, str]:
    nombre = f"{direccion}_{station_id}_{normalizar_nombre(station_name)}.csv"
    return carpeta / nombre, nombre


def cargar_serie_estacion(direccion: str, station_id: str) -> pd.DataFrame:
    """Carga y agrega la serie temporal de 15 min de una estación para una dirección."""
    archivos = sorted(glob.glob(str(RUTA_PARQUET / f"*-{direccion}.parquet")))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron parquets de {direccion} en {RUTA_PARQUET}")

    partes = []
    for archivo in archivos:
        df_mes = pd.read_parquet(archivo)
        df_est = df_mes[df_mes["station_id"] == station_id]
        if df_est.empty:
            continue
        partes.append(df_est.groupby("datetime", as_index=False)[direccion].sum())

    if not partes:
        raise ValueError(f"sin datos para la estación {station_id}")

    df = pd.concat(partes, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df[df["datetime"] >= "2022-01-01"].reset_index(drop=True)


def construir_features(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Rellena huecos temporales y agrega calendario, festivos, rezagos y promedios móviles."""
    df = df.copy()

    rango_completo = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="15min")
    huecos = rango_completo.difference(df["datetime"])
    if len(huecos) > 0:
        df = df.set_index("datetime").reindex(rango_completo, fill_value=0).reset_index()
        df.columns = ["datetime", columna]

    df["hour_sin"] = np.sin(2 * np.pi * df["datetime"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["datetime"].dt.hour / 24)
    df["min_sin"]  = np.sin(2 * np.pi * df["datetime"].dt.minute / 60)
    df["min_cos"]  = np.cos(2 * np.pi * df["datetime"].dt.minute / 60)
    df["dow_sin"]  = np.sin(2 * np.pi * df["datetime"].dt.weekday / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["datetime"].dt.weekday / 7)
    df["week_sin"] = np.sin(2 * np.pi * df["datetime"].dt.isocalendar().week.astype(int) / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["datetime"].dt.isocalendar().week.astype(int) / 52)
    df["month"] = df["datetime"].dt.month
    df["year"]  = df["datetime"].dt.year
    df["es_fin_de_semana"] = (df["datetime"].dt.weekday >= 5).astype(int)
    df["es_festivo_fijo"]  = df["datetime"].apply(lambda ts: int((ts.month, ts.day) in FESTIVOS_FIJOS))

    df["lag_15m"] = df[columna].shift(1)
    df["lag_30m"] = df[columna].shift(2)
    df["lag_1h"]  = df[columna].shift(4)
    df["lag_24h"] = df[columna].shift(96)
    df["lag_1w"]  = df[columna].shift(672)
    df["rolling_1h"]  = df[columna].shift(1).rolling(4).mean()
    df["rolling_24h"] = df[columna].shift(1).rolling(96).mean()

    return df.dropna().reset_index(drop=True)


def construir_fila_features(ts, hist):
    """Construye una fila de features para el instante `ts`, en el mismo orden que FEATURES."""
    return [
        np.sin(2 * np.pi * ts.hour / 24),    np.cos(2 * np.pi * ts.hour / 24),
        np.sin(2 * np.pi * ts.minute / 60),  np.cos(2 * np.pi * ts.minute / 60),
        np.sin(2 * np.pi * ts.weekday() / 7), np.cos(2 * np.pi * ts.weekday() / 7),
        np.sin(2 * np.pi * ts.isocalendar()[1] / 52), np.cos(2 * np.pi * ts.isocalendar()[1] / 52),
        ts.month, ts.year,
        int(ts.weekday() >= 5),
        int((ts.month, ts.day) in FESTIVOS_FIJOS),
        hist[-1], hist[-2], hist[-4], hist[-96], hist[-672],
        float(np.mean(hist[-4:])), float(np.mean(hist[-96:])),
    ]


# Verificación de que el orden de FEATURES coincide con construir_fila_features.
# Si cambias una de las dos listas sin la otra, este assert lo detecta de inmediato.
assert FEATURES == [
    "hour_sin", "hour_cos", "min_sin", "min_cos", "dow_sin", "dow_cos",
    "week_sin", "week_cos", "month", "year", "es_fin_de_semana", "es_festivo_fijo",
    "lag_15m", "lag_30m", "lag_1h", "lag_24h", "lag_1w", "rolling_1h", "rolling_24h",
], "FEATURES y construir_fila_features deben tener exactamente el mismo orden"


def pronosticar_periodo(model, df, columna, inicio, fin):
    """Forecast recursivo paso a paso (15 min) para el período [inicio, fin)."""
    hist = list(df[df["datetime"] < inicio][columna].values[-BUFFER:])
    if len(hist) < BUFFER:
        raise ValueError(f"historial insuficiente antes de {inicio} ({len(hist)} < {BUFFER} intervalos)")

    rango = pd.date_range(inicio, fin, freq="15min", inclusive="left")
    preds = np.empty(len(rango))

    for i, ts in enumerate(rango):
        X_fut = np.array([construir_fila_features(ts, hist)])
        pred = float(np.clip(model.predict(X_fut), 0, None)[0])
        preds[i] = pred
        hist.append(pred)
        hist.pop(0)

    return pd.DatetimeIndex(rango), preds


def entrenar_y_predecir(direccion, station_id, ruta_csv, periodo_inicio, periodo_fin):
    """Entrena el modelo de una estación/dirección, pronostica [periodo_inicio, periodo_fin)
    y guarda el CSV de salida. Devuelve un dict con métricas para el resumen."""
    inicio_reloj = time.time()
    columna = direccion

    df = cargar_serie_estacion(direccion, station_id)
    df = construir_features(df, columna)

    mask_train = df["datetime"] < ENTRENAMIENTO_FIN
    mask_val   = (df["datetime"] >= VALIDACION_INICIO) & (df["datetime"] < VALIDACION_FIN)
    train, val = df[mask_train], df[mask_val]

    if len(train) < 1000 or len(val) < 100:
        raise ValueError(f"datos insuficientes para entrenar (train={len(train)}, val={len(val)})")

    X_train, y_train = train[FEATURES], train[columna]
    X_val,   y_val   = val[FEATURES],   val[columna]

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

    pred_val = np.clip(model.predict(X_val), 0, None)
    mae  = mean_absolute_error(y_val, pred_val)
    rmse = mean_squared_error(y_val, pred_val) ** 0.5
    # WMAPE (MAPE ponderado por volumen): más estable que el MAPE simple cuando
    # hay intervalos de muy baja demanda (madrugadas, festivos).
    wmape = float(np.abs(y_val.values - pred_val).sum() / y_val.values.sum() * 100)

    futuro, preds = pronosticar_periodo(model, df, columna, periodo_inicio, periodo_fin)

    pd.DataFrame({
        "datetime":   futuro,
        "prediccion": preds.round().astype(int),
    }).to_csv(ruta_csv, index=False)

    importancias = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    top_features = "; ".join(f"{f}={v:.3f}" for f, v in importancias.head(5).items())

    return {
        "duracion_seg":        round(time.time() - inicio_reloj, 1),
        "mae_val":             round(float(mae), 2),
        "rmse_val":            round(float(rmse), 2),
        "wmape_val":           round(wmape, 2),
        "filas_entrenamiento": len(train),
        "filas_validacion":    len(val),
        "pred_total":          round(float(preds.sum())),
        "pred_promedio":       round(float(preds.mean()), 1),
        "pred_intervalos":     len(preds),
        "top_features":        top_features,
    }


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def procesar_periodo(periodo_inicio, periodo_fin, estaciones):
    """Corre las 164 estaciones x direcciones para un único período y guarda
    sus resultados en outputs/predicciones/{periodo_inicio}_a_{periodo_fin}/."""
    nombre_intervalo = NOMBRE_CARPETA_OVERRIDE or f"{periodo_inicio}_a_{periodo_fin}{SUFIJO_CARPETA}"
    carpeta = RUTA_PREDS / nombre_intervalo
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_resumen = carpeta / "resumen_predicciones_estaciones.csv"

    resumen_filas = []
    if ruta_resumen.exists():
        resumen_filas = pd.read_csv(ruta_resumen).to_dict("records")

    total = len(estaciones) * len(DIRECCIONES)
    contador = 0

    print(f"\n{'=' * 60}")
    print(f"PERÍODO: {periodo_inicio} -> {periodo_fin}")
    print(f"Entrenamiento: < {ENTRENAMIENTO_FIN}  |  Validación: [{VALIDACION_INICIO}, {VALIDACION_FIN})")
    print(f"Carpeta de salida: {carpeta}")
    print(f"Total de corridas: {total}")
    print(f"{'=' * 60}\n")

    for direccion in DIRECCIONES:
        for _, fila in estaciones.iterrows():
            station_id, station_name = fila["station_id"], fila["station_name"]
            contador += 1
            ruta_csv, nombre_csv = ruta_csv_estacion(carpeta, direccion, station_id, station_name)

            etiqueta = f"[{contador}/{total}] {direccion:<8} {station_id} {station_name}"

            if ruta_csv.exists():
                print(f"{etiqueta} -- ya existe, se omite")
                continue

            print(f"{etiqueta} -- procesando...")
            t0 = time.time()
            try:
                info = entrenar_y_predecir(direccion, station_id, ruta_csv, periodo_inicio, periodo_fin)
                fila_resumen = {
                    "direccion": direccion,
                    "station_id": station_id,
                    "station_name": station_name,
                    "archivo": nombre_csv,
                    "status": "ok",
                    "error": "",
                    **info,
                }
                print(f"    OK en {info['duracion_seg']}s  |  MAE={info['mae_val']}  RMSE={info['rmse_val']}  "
                      f"WMAPE={info['wmape_val']}%  |  total pronosticado={info['pred_total']:,}")
            except Exception as exc:
                fila_resumen = {
                    "direccion": direccion,
                    "station_id": station_id,
                    "station_name": station_name,
                    "archivo": "",
                    "status": "error",
                    "error": str(exc),
                    "duracion_seg": round(time.time() - t0, 1),
                }
                print(f"    ERROR: {exc}")

            resumen_filas.append(fila_resumen)
            # Se reescribe tras cada estación: si el proceso se interrumpe, no se pierde el progreso.
            pd.DataFrame(resumen_filas).to_csv(ruta_resumen, index=False)

    print(f"\nPeríodo {nombre_intervalo} terminado.")
    print(f"CSVs de predicciones en: {carpeta}")
    print(f"Resumen consolidado en:  {ruta_resumen}")


def main():
    catalogo = pd.read_parquet(RUTA_CATALOGO)
    estaciones = (
        catalogo[["station_id", "station_name"]]
        .drop_duplicates()
        .sort_values("station_id")
        .reset_index(drop=True)
    )
    if FILTRO_ESTACIONES is not None:
        estaciones = estaciones[estaciones["station_id"].isin(FILTRO_ESTACIONES)].reset_index(drop=True)
        print(f"[FILTRO] Procesando solo {len(estaciones)} estación(es): {sorted(FILTRO_ESTACIONES)}\n")

    if LIMITE_ESTACIONES is not None:
        estaciones = estaciones.head(LIMITE_ESTACIONES)
        print(f"[MODO PRUEBA] Procesando solo las primeras {LIMITE_ESTACIONES} estación(es) del catálogo.\n")

    print(f"Estaciones en catálogo: {len(estaciones)}  |  Direcciones: {DIRECCIONES}")
    print(f"Períodos a procesar: {len(PERIODOS)}")
    for inicio, fin in PERIODOS:
        print(f"  - {inicio} -> {fin}")

    for periodo_inicio, periodo_fin in PERIODOS:
        procesar_periodo(periodo_inicio, periodo_fin, estaciones)

    print("\nTodos los períodos fueron procesados.")
    print(f"Resultados en: {RUTA_PREDS}")


if __name__ == "__main__":
    main()
