#Antes de correr el cadigo no olviden descargar los paquetes y librerias:
#pip install xgboost scikit-learn pandas numpy matplotlib pyarrow

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────
# 1. CARGA Y AGREGACIÓN
# ─────────────────────────────────────────────

RUTA_DATOS = r"" # Ubiquen la ruta donde se encuentren los archivos de la base de datos aquí
archivos = sorted(glob.glob(f"{RUTA_DATOS}\\*-entradas.parquet"))

if not archivos:
    raise FileNotFoundError(f"No se encontraron archivos en {RUTA_DATOS}")

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
print(f"\nBaseline (perfil histórico) sobre test 2025:")
print(f"  MAE:  {mae_base:>10,.0f} pasajeros")
print(f"  RMSE: {rmse_base:>10,.0f} pasajeros")

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
mae_test  = mean_absolute_error(y_test, pred_test)
rmse_test = mean_squared_error(y_test, pred_test) ** 0.5

print("\n" + "="*45)
print(f"{'':30} {'MAE':>6}  {'RMSE':>8}")
print("="*45)
print(f"{'Baseline (perfil histórico) test':30} {mae_base:>8,.0f}  {rmse_base:>8,.0f}")
print(f"{'XGBoost validación 2024':30} {mae_val:>8,.0f}  {rmse_val:>8,.0f}")
print(f"{'XGBoost test 2025':30} {mae_test:>8,.0f}  {rmse_test:>8,.0f}")
print("="*45)
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

resultados.to_csv(r"predicciones_2025.csv", index=False) # Pueden poner una ruta antes de predicciones_2025.csv para guardar ahí los datos de las predicciones
print("Predicciones exportadas en predicciones_2025.csv")

# ─────────────────────────────────────────────
# 8. VISUALIZACIONES
# ─────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(24, 14))
fig.suptitle("XGBoost TransMilenio: Entradas sistema agregado", fontsize=14)

# -- Panel 1: test completo 2025
ax = axes[0]
ax.plot(test["datetime"], y_test.values, label="Real", color="steelblue", linewidth=0.8)
ax.plot(test["datetime"], pred_test, label="XGBoost", color="tomato", linewidth=0.8, alpha=0.85)
ax.set_title("Test 2025: serie completa")
ax.set_ylabel("Entradas")
ax.legend()
ax.grid(True, alpha=0.3)

# -- Panel 2: zoom dos semanas de test
dos_semanas = test["datetime"] < (test["datetime"].min() + pd.Timedelta(weeks=2))
ax = axes[1]
ax.plot(test.loc[dos_semanas, "datetime"], y_test[dos_semanas].values,
        label="Real", color="steelblue")
ax.plot(test.loc[dos_semanas, "datetime"], pred_test[dos_semanas],
        label="XGBoost", color="tomato", alpha=0.85)
ax.set_title("Zoom: primeras 2 semanas de test 2025")
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
plt.savefig("resultados_xgboost.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figura guardada en resultados_xgboost.png")
