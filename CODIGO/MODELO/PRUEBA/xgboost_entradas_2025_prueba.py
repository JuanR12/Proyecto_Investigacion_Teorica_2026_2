"""
Predicción de flujos de entradas - Sistema Troncal TransMilenio
XGBoost con features de calendario y rezagos

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
#Filtro de datos desde 2021
df = df[df["datetime"] >= "2022-01-01"].reset_index(drop=True)

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
# Train: 2021-2023  |  Validación: 2024


mask_train = df["datetime"] < "2024-01-01"
mask_val   = (df["datetime"] >= "2024-01-01") & (df["datetime"] < "2025-01-01")

train = df[mask_train]
val   = df[mask_val]

X_train, y_train = train[FEATURES], train["entradas"]
X_val,   y_val   = val[FEATURES],   val["entradas"]

print(f"\nTrain:      {train['datetime'].min().date()} -> {train['datetime'].max().date()} ({len(train):,} filas)")
print(f"Validación: {val['datetime'].min().date()}   -> {val['datetime'].max().date()}   ({len(val):,} filas)")

# ─────────────────────────────────────────────
# 5. BASELINES (umbrales mínimos a superar)
# ─────────────────────────────────────────────
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

pred_val = np.clip(model.predict(X_val), 0, None)

# ── Forecasting recursivo 2025 ─────────────────────────────────────────────
BUFFER = 672  # lag máximo necesario
hist = list(df[df["datetime"] < "2025-01-01"]["entradas"].values[-BUFFER:])  # últimos valores reales de 2024

futuro = pd.date_range("2025-01-01", "2025-12-31 23:45", freq="15min")
preds_2025 = []

for ts in futuro:
    lag15   = hist[-1]
    lag30   = hist[-2]
    lag1h   = hist[-4]
    lag24h  = hist[-96]
    lag1w   = hist[-672]
    roll1h  = np.mean(hist[-4:])
    roll24h = np.mean(hist[-96:])

    fila = {
        "hour_sin": np.sin(2*np.pi*ts.hour/24),
        "hour_cos": np.cos(2*np.pi*ts.hour/24),
        "min_sin":  np.sin(2*np.pi*ts.minute/60),
        "min_cos":  np.cos(2*np.pi*ts.minute/60),
        "dow_sin":  np.sin(2*np.pi*ts.weekday()/7),
        "dow_cos":  np.cos(2*np.pi*ts.weekday()/7),
        "week_sin": np.sin(2*np.pi*ts.isocalendar()[1]/52),
        "week_cos": np.cos(2*np.pi*ts.isocalendar()[1]/52),
        "month": ts.month, "year": ts.year,
        "es_fin_de_semana": int(ts.weekday() >= 5),
        "lag_15m": lag15, "lag_30m": lag30, "lag_1h": lag1h,
        "lag_24h": lag24h, "lag_1w": lag1w,
        "rolling_1h": roll1h, "rolling_24h": roll24h,
    }

    X_fut = pd.DataFrame([fila])[FEATURES]
    pred  = float(np.clip(model.predict(X_fut), 0, None)[0])
    preds_2025.append(pred)
    hist.append(pred)
    hist.pop(0)  # mantiene el buffer en tamaño fijo

# ─────────────────────────────────────────────
# 7. MÉTRICAS
# ─────────────────────────────────────────────

mae_val   = mean_absolute_error(y_val, pred_val)
rmse_val  = mean_squared_error(y_val, pred_val) ** 0.5


print("\n" + "="*45)
print(f"{'':30} {'MAE':>6}  {'RMSE':>8}")
print("="*45)
print(f"{'XGBoost validación 2024':30} {mae_val:>8,.0f}  {rmse_val:>8,.0f}")
print("="*45)
print("Si XGBoost no supera el baseline, revisar features o datos.")

# ─────────────────────────────────────────────
# EXTRA. EXPORTAR PREDICCIÓN
# ─────────────────────────────────────────────

resultados_2025 = pd.DataFrame({
    "datetime":    futuro,
    "prediccion":  np.array(preds_2025).round().astype(int),
})
resultados_2025.to_csv(RUTA_PREDS / "predicciones_2025_prueba.csv", index=False)
print("Predicciones 2025 exportadas.")

# ─────────────────────────────────────────────
# 8. VISUALIZACIONES
# ─────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(24, 14))
fig.suptitle("XGBoost TransMilenio: Entradas sistema agregado", fontsize=14)

# -- Panel 1
ax = axes[0]
ax.plot(futuro, preds_2025, label="XGBoost 2025", color="tomato", linewidth=0.8)
ax.set_title("Predicción 2025: serie completa")
ax.set_ylabel("Entradas")
ax.legend()
ax.grid(True, alpha=0.3)

# -- Panel 2
ax = axes[1]
dos_semanas = futuro < (futuro.min() + pd.Timedelta(weeks=2))
ax.plot(futuro[dos_semanas], np.array(preds_2025)[dos_semanas],
        label="XGBoost 2025", color="tomato")
ax.set_title("Zoom: primeras 2 semanas de predicción 2025")
ax.set_ylabel("Entradas")
ax.legend()
ax.grid(True, alpha=0.3)

# -- Panel 3: importancia de features

# Esta gráfica muestra cuánto aportó cada feature a las decisiones del modelo
# durante el entrenamiento. XGBoost mide cuántas veces usó cada feature para
# hacer una división en sus árboles, ponderado por cuánto redujo el error esa
# división. El resultado se normaliza para que todo sume 1.
#
# Interpretación esperada de los valores:
#   - lag_15m, lag_1h, rolling_1h: importancia alta. El flujo de los últimos
#     intervalos es el predictor más fuerte del flujo actual.
#   - hour_sin, hour_cos: importancia media-alta. La hora del día estructura
#     todo el patrón de demanda del sistema.
#   - dow_sin, dow_cos: importancia media. Diferencia días laborables de fines
#     de semana y festivos.
#   - lag_1w: importancia notable. Captura patrones semanales repetidos como
#     Semana Santa, cuya caída se refleja en el valor real de 7 días atrás.
#   - min_sin, min_cos: importancia esperada baja o casi cero. El minuto dentro
#     de la hora aporta poca información adicional sobre la que ya da la hora.
#
# Si los lags dominan completamente sobre las features de calendario, el modelo
# depende de valores pasados reales para funcionar bien, lo que limita su
# capacidad predictiva a largo plazo sin datos reales disponibles.
# Si una feature esperada aparece con importancia casi cero, puede estar mal
# construida o ser redundante con otra.

ax = axes[2]
importancias = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
importancias.plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Importancia de features (XGBoost)")
ax.set_xlabel("Importancia relativa")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RUTA_FIGURAS / "prediccion_2025_prueba.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Figura guardada en {RUTA_FIGURAS / 'prediccion_2025_prueba.png'}")
