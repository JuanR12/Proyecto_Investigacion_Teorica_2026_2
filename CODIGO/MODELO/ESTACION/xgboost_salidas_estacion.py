"""
Predicción de flujos de salidas - Sistema Troncal TransMilenio
XGBoost con features de calendario y rezagos

Estructura esperada de archivos:
    datos/
        2021-01-salidas.parquet
        2021-02-salidas.parquet
        ...
        2025-12-salidas.parquet

Cada parquet tiene columnas: linea_id, station_id, datetime, salidas
"""

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── #0 Estación objetivo ──────────────────────────────────────────────────────────
# Cambiar este valor para predecir otra estación.
# Los IDs válidos están en catalogo_estaciones.parquet (columna station_id).
STATION_ID = "02000"

# ─────────────────────────────────────────────
# 1. CARGA Y AGREGACIÓN
# ─────────────────────────────────────────────

RUTA_DATOS = r"C:\Users\Juanshots\Desktop\PROYECTO_INV_TEO\DATOS LIMPIOS\ENTRADA Y SALIDA MENSUAL\parquet"
archivos = sorted(glob.glob(f"{RUTA_DATOS}\\*-salidas.parquet"))

if not archivos:
    raise FileNotFoundError(f"No se encontraron archivos en {RUTA_DATOS}")

print(f"Archivos encontrados: {len(archivos)}")

partes = []
for archivo in archivos:
    df_mes = pd.read_parquet(archivo)
    # Agregado a nivel sistema: suma de salidas por intervalo de 15 min
    # Si se quiere trabajar por troncal, agregar linea_id al groupby
    df_est = df_mes[df_mes["station_id"] == STATION_ID]
    if df_est.empty:
        continue
    agg = df_est.groupby("datetime", as_index=False)["salidas"].sum()
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
    df.columns = ["datetime", "salidas"]
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

# Festivos de fecha fija en Colombia
FESTIVOS_FIJOS = {(1, 1), (5, 1), (7, 20), (8, 7), (12, 8), (12, 25)}
df["es_festivo_fijo"] = df["datetime"].apply(
    lambda ts: int((ts.month, ts.day) in FESTIVOS_FIJOS)
)

# Rezagos temporales (shift sobre filas, válido solo si la serie no tiene huecos)
df["lag_15m"]   = df["salidas"].shift(1)    # 15 min atrás
df["lag_30m"]   = df["salidas"].shift(2)    # 30 min atrás
df["lag_1h"]    = df["salidas"].shift(4)    # 1 hora atrás
df["lag_24h"]   = df["salidas"].shift(96)   # mismo intervalo ayer
df["lag_1w"]    = df["salidas"].shift(672)  # mismo intervalo semana pasada

# Promedios móviles (ventanas sobre filas pasadas)
df["rolling_1h"]  = df["salidas"].shift(1).rolling(4).mean()    # promedio última hora
df["rolling_24h"] = df["salidas"].shift(1).rolling(96).mean()   # promedio último día

df = df.dropna().reset_index(drop=True)

FEATURES = [
    "hour_sin", "hour_cos",
    "min_sin", "min_cos",
    "dow_sin", "dow_cos",
    "week_sin", "week_cos",
    "month", "year",
    "es_fin_de_semana",
    "es_festivo_fijo",        # <-- línea nueva
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

X_train, y_train = train[FEATURES], train["salidas"]
X_val,   y_val   = val[FEATURES],   val["salidas"]

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

# ── Forecasting recursivo 2025 por trimestres ─────────────────────────────
# Cada trimestre se siembra con valores reales del período anterior,
# lo que evita la acumulación de error a lo largo del año completo.

BUFFER = 672
trimestres = [
    ("2025-01-01", "2025-04-01")
]

preds_2025 = []
futuro_list = []

for inicio, fin in trimestres:
    # Semilla: últimos 672 valores reales anteriores al inicio del trimestre
    hist = list(
        df[df["datetime"] < inicio]["salidas"].values[-BUFFER:]
    )

    rango = pd.date_range(inicio, fin, freq="15min", inclusive="left")

    for ts in rango:
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
            "es_festivo_fijo": int((ts.month, ts.day) in FESTIVOS_FIJOS),
            "lag_15m": lag15, "lag_30m": lag30, "lag_1h": lag1h,
            "lag_24h": lag24h, "lag_1w": lag1w,
            "rolling_1h": roll1h, "rolling_24h": roll24h,
        }

        X_fut = pd.DataFrame([fila])[FEATURES]
        pred  = float(np.clip(model.predict(X_fut), 0, None)[0])
        preds_2025.append(pred)
        futuro_list.append(ts)
        hist.append(pred)
        hist.pop(0)

    print(f"Trimestre {inicio[:7]} -> {fin[:7]} completado ({len(rango):,} intervalos)")

futuro = pd.DatetimeIndex(futuro_list)

# ─────────────────────────────────────────────
# 7. MÉTRICAS
# ─────────────────────────────────────────────

mae_val   = mean_absolute_error(y_val, pred_val)
rmse_val  = mean_squared_error(y_val, pred_val) ** 0.5


print("\n" + "="*45)
print(f"{'':30} {'MAE':>6}  {'RMSE':>8}")
print("="*45)
print(f"{'XGBoost validación 2025':30} {mae_val:>8,.0f}  {rmse_val:>8,.0f}")
print("="*45)
print("Si XGBoost no supera el baseline, revisar features o datos.")

# ─────────────────────────────────────────────
# EXTRA. EXPORTAR PREDICCIÓN
# ─────────────────────────────────────────────

resultados_2025 = pd.DataFrame({
    "datetime":    futuro,
    "prediccion":  np.array(preds_2025).round().astype(int),
})
resultados_2025.to_csv(
    rf"C:\Users\Juanshots\Desktop\PROYECTO_INV_TEO\DATOS LIMPIOS\ENTRADA Y SALIDA MENSUAL\PREDICCIONES\validaciones_salida_{STATION_ID}_pred_2025.csv",
    index=False
)
print("Predicciones estacion exportadas.")

# ─────────────────────────────────────────────
# 8. VISUALIZACIONES
# ─────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(24, 14))
fig.suptitle("XGBoost TransMilenio: Entradas sistema agregado", fontsize=14)

# -- Panel 1
ax = axes[0]
ax.plot(futuro, preds_2025, label="XGBoost 2025", color="pink", linewidth=0.8)
ax.set_title("Predicción 2026: serie completa")
ax.set_ylabel("Entradas")
ax.legend()
ax.grid(True, alpha=0.3)

# -- Panel 2
ax = axes[1]
dos_semanas = futuro < (futuro.min() + pd.Timedelta(weeks=2))
ax.plot(futuro[dos_semanas], np.array(preds_2025)[dos_semanas],
        label="XGBoost 2025", color="pink")
ax.set_title("Zoom: primeras 2 semanas de predicción 2026")
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
plt.savefig("prediccion_2026_prueba.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figura guardada en prediccion_2026_prueba.png")
