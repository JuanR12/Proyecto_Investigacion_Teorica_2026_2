#Antes de correr el codigo no olviden descargar los paquetes y librerias:
#pip install xgboost scikit-learn pandas numpy matplotlib pyarrow

"""
Predicción de flujos de entradas - Sistema Troncal TransMilenio
XGBoost con features de calendario y rezagos

Copia de xgboost_transmilenio_v1.py con visualizaciones mejoradas:
  - Métricas MAE, RMSE y MAPE para baseline, validación y test
  - Gráficas sin solapamiento de texto (rotación de fechas, formateadores,
    ajuste de márgenes en vez de tight_layout)
  - Real vs. predicción con colores y estilos de línea claramente distinguibles
  - Títulos representativos: el test 2025 simula un despliegue en vivo del
    modelo (entrenado con 2021-2023, validado con 2024) prediciendo todo 2025
  - Zoom centrado en la segunda semana de abril de 2025

Estructura esperada de archivos:
    datos/
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
from config import RUTA_PARQUET, RUTA_PREDS, RUTA_FIGURAS

# ─────────────────────────────────────────────
# 1. CARGA Y AGREGACIÓN
# ─────────────────────────────────────────────

archivos = sorted(glob.glob(str(RUTA_PARQUET / "*-entradas.parquet")))

if not archivos:
    raise FileNotFoundError(f"No se encontraron archivos en {RUTA_PARQUET}")

print(f"Archivos encontrados: {len(archivos)}")

partes = []
for archivo in archivos:
    df_mes = pd.read_parquet(archivo)
    # Agregado a nivel sistema: suma de entradas por intervalo de 15 min
    # Si se quiere trabajar por troncal, agregar linea_id al groupby
    agg = df_mes.groupby("datetime", as_index=False)["entradas"].sum()
    partes.append(agg)

df = pd.concat(partes, ignore_index=True)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

print(f"Rango temporal: {df['datetime'].min()} -> {df['datetime'].max()}")
print(f"Total filas: {len(df):,}")

# ─────────────────────────────────────────────
# 2. VERIFICACIÓN DE HUECOS EN LOS DATOS
# ─────────────────────────────────────────────

rango_completo = pd.date_range(
    start=df["datetime"].min(),
    end=df["datetime"].max(),
    freq="15min"
)
huecos = rango_completo.difference(df["datetime"])

if len(huecos) > 0:
    print(f"\nATENCIÓN: {len(huecos)} intervalos de 15 min faltantes.")
    print("Los lags calculados sobre filas (shift) serán incorrectos en esas posiciones.")
    # Rellenar con 0 para mantener la secuencia temporal intacta
    df = df.set_index("datetime").reindex(rango_completo, fill_value=0).reset_index()
    df.columns = ["datetime", "entradas"]
    print(f"Huecos rellenados con 0. Filas totales ahora: {len(df):,}")
else:
    print("Sin huecos temporales. Serie completa.")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

# Codificación cíclica: evita que hora 23 y hora 0 sean "lejanas" para el modelo
df["hour_sin"]  = np.sin(2 * np.pi * df["datetime"].dt.hour / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["datetime"].dt.hour / 24)

# Minuto dentro de la hora (0, 15, 30, 45) también cíclico
df["min_sin"]   = np.sin(2 * np.pi * df["datetime"].dt.minute / 60)
df["min_cos"]   = np.cos(2 * np.pi * df["datetime"].dt.minute / 60)

# Día de semana cíclico
df["dow_sin"]   = np.sin(2 * np.pi * df["datetime"].dt.weekday / 7)
df["dow_cos"]   = np.cos(2 * np.pi * df["datetime"].dt.weekday / 7)

# Semana del año cíclica (captura estacionalidad anual)
df["week_sin"]  = np.sin(2 * np.pi * df["datetime"].dt.isocalendar().week.astype(int) / 52)
df["week_cos"]  = np.cos(2 * np.pi * df["datetime"].dt.isocalendar().week.astype(int) / 52)

# Mes y año como numéricos (el año ayuda a capturar tendencia de recuperación post-pandemia)
df["month"]     = df["datetime"].dt.month
df["year"]      = df["datetime"].dt.year

# Indicadores binarios
df["es_fin_de_semana"] = (df["datetime"].dt.weekday >= 5).astype(int)

# Rezagos temporales (shift sobre filas, válido solo si la serie no tiene huecos)
df["lag_15m"]   = df["entradas"].shift(1)    # 15 min atrás
df["lag_30m"]   = df["entradas"].shift(2)    # 30 min atrás
df["lag_1h"]    = df["entradas"].shift(4)    # 1 hora atrás
df["lag_24h"]   = df["entradas"].shift(96)   # mismo intervalo ayer
df["lag_1w"]    = df["entradas"].shift(672)  # mismo intervalo semana pasada

# Promedios móviles (ventanas sobre filas pasadas)
df["rolling_1h"]  = df["entradas"].shift(1).rolling(4).mean()    # promedio última hora
df["rolling_24h"] = df["entradas"].shift(1).rolling(96).mean()   # promedio último día

df = df.dropna().reset_index(drop=True)

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

# ─────────────────────────────────────────────
# 4. PARTICIÓN TEMPORAL
# ─────────────────────────────────────────────
# Train: 2021-2023  |  Validación: 2024  |  Test: 2025


mask_train = df["datetime"] < "2024-01-01"
mask_val   = (df["datetime"] >= "2024-01-01") & (df["datetime"] < "2025-01-01")
mask_test  = df["datetime"] >= "2025-01-01"

train = df[mask_train]
val   = df[mask_val]
test  = df[mask_test]

X_train, y_train = train[FEATURES], train["entradas"]
X_val,   y_val   = val[FEATURES],   val["entradas"]
X_test,  y_test  = test[FEATURES],  test["entradas"]

print(f"\nTrain:      {train['datetime'].min().date()} -> {train['datetime'].max().date()} ({len(train):,} filas)")
print(f"Validación: {val['datetime'].min().date()}   -> {val['datetime'].max().date()}   ({len(val):,} filas)")
print(f"Test:       {test['datetime'].min().date()}  -> {test['datetime'].max().date()}  ({len(test):,} filas)")

# ─────────────────────────────────────────────
# 5. BASELINES (umbrales mínimos a superar)
# ─────────────────────────────────────────────


def calcular_mape(y_true, y_pred):
    """MAPE en porcentaje, ignorando intervalos con valor real = 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calcular_wmape(y_true, y_pred):
    """WMAPE (MAPE ponderado por volumen): sum(|real-pred|) / sum(real) * 100.
    Más robusto que el MAPE simple porque los intervalos de baja demanda
    no distorsionan el promedio."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100


# Baseline 1: promedio histórico por (hour, minute, weekday)
# Calculado solo sobre train para no filtrar información del futuro
perfil = (
    train.groupby(["hour_sin", "hour_cos", "dow_sin", "dow_cos"])["entradas"]
    .mean()
    .reset_index()
    .rename(columns={"entradas": "baseline_perfil"})
)

# Para el test se usa el mismo agrupador
test_base = test.merge(perfil, on=["hour_sin", "hour_cos", "dow_sin", "dow_cos"], how="left")
baseline_pred = test_base["baseline_perfil"].fillna(train["entradas"].mean()).values

mae_base  = mean_absolute_error(y_test, baseline_pred)
rmse_base = mean_squared_error(y_test, baseline_pred) ** 0.5
mape_base = calcular_mape(y_test, baseline_pred)
wmape_base = calcular_wmape(y_test, baseline_pred)
print(f"\nBaseline (perfil histórico) sobre test 2025:")
print(f"  MAE:   {mae_base:>10,.0f} pasajeros")
print(f"  RMSE:  {rmse_base:>10,.0f} pasajeros")
print(f"  MAPE:  {mape_base:>10,.1f} %")
print(f"  WMAPE: {wmape_base:>10,.1f} %")

# ─────────────────────────────────────────────
# 6. APLICACIÓN DEL MODELO XGBOOST
# ─────────────────────────────────────────────

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    # early_stopping_rounds requiere eval_set
    early_stopping_rounds=30,
    eval_metric="mae",
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,   # imprime cada 50 iteraciones
)

pred_val  = model.predict(X_val)
pred_val = np.clip(pred_val, 0, None)
pred_test = model.predict(X_test)
pred_test = np.clip(pred_test, 0, None)

# ─────────────────────────────────────────────
# 7. MÉTRICAS
# ─────────────────────────────────────────────

mae_val   = mean_absolute_error(y_val, pred_val)
rmse_val  = mean_squared_error(y_val, pred_val) ** 0.5
mape_val  = calcular_mape(y_val, pred_val)
wmape_val = calcular_wmape(y_val, pred_val)

mae_test  = mean_absolute_error(y_test, pred_test)
rmse_test = mean_squared_error(y_test, pred_test) ** 0.5
mape_test = calcular_mape(y_test, pred_test)
wmape_test = calcular_wmape(y_test, pred_test)

print("\n" + "="*65)
print(f"{'':30} {'MAE':>8}  {'RMSE':>8}  {'MAPE':>7}  {'WMAPE':>7}")
print("="*65)
print(f"{'Baseline (perfil histórico) test':30} {mae_base:>8,.0f}  {rmse_base:>8,.0f}  {mape_base:>6.1f}%  {wmape_base:>6.1f}%")
print(f"{'XGBoost validación 2024':30} {mae_val:>8,.0f}  {rmse_val:>8,.0f}  {mape_val:>6.1f}%  {wmape_val:>6.1f}%")
print(f"{'XGBoost test 2025':30} {mae_test:>8,.0f}  {rmse_test:>8,.0f}  {mape_test:>6.1f}%  {wmape_test:>6.1f}%")
print("="*65)
print("Si XGBoost no supera el baseline, revisar features o datos.")

# ─────────────────────────────────────────────
# EXTRA. EXPORTAR PREDICCIÓN
# ─────────────────────────────────────────────

resultados = pd.DataFrame({
    "datetime": test["datetime"].values,
    "real":     y_test.values,
    "prediccion": pred_test,
    "error_absoluto": np.abs(y_test.values - pred_test)
})

resultados["prediccion"] = pred_test.round().astype(int)
resultados["error_absoluto"] = (resultados["real"] - resultados["prediccion"]).abs()

resultados.to_csv(RUTA_PREDS / "predicciones_2025.csv", index=False)
print("Predicciones exportadas en predicciones_2025.csv")

# ─────────────────────────────────────────────
# 8. VISUALIZACIONES
# ─────────────────────────────────────────────
# Formato cuadrado para documento a doble columna: 2 paneles apilados,
# colores saturados fáciles de diferenciar en impreso (azul sólido = real,
# rojo punteado = predicción), título grande y corto, sin MAPE.

COLOR_REAL = "#1976D2"   # azul — datos reales
COLOR_PRED = "#E53935"   # rojo punteado — predicción XGBoost

fig, axes = plt.subplots(2, 1, figsize=(8, 8.5))
fig.suptitle(
    "XGBoost TransMilenio — Sistema retroalimentado (Entradas)\n"
    f"MAE: {mae_test:,.0f}   RMSE: {rmse_test:,.0f}   WMAPE: {wmape_test:.1f}%",
    fontsize=15, fontweight="bold"
)

# -- Panel 1: promedio diario, segundo semestre 2025 (simulación en vivo)
diario = (
    pd.DataFrame({"datetime": test["datetime"].values, "real": y_test.values, "prediccion": pred_test})
    .assign(fecha=lambda d: d["datetime"].dt.floor("D"))
    .groupby("fecha", as_index=False)[["real", "prediccion"]].mean()
)
diario = diario[diario["fecha"] >= "2025-07-01"]

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
mae_zoom  = mean_absolute_error(zoom_real, zoom_pred) if mask_zoom.any() else np.nan
rmse_zoom = mean_squared_error(zoom_real, zoom_pred) ** 0.5 if mask_zoom.any() else np.nan
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

plt.savefig(RUTA_FIGURAS / "resultados_xgboost_entradas_mejorado.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Figura guardada en {RUTA_FIGURAS / 'resultados_xgboost_entradas_mejorado.png'}")
