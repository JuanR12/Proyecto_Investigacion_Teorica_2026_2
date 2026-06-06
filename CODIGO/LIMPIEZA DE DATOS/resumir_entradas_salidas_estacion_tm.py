from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
# Carpeta donde están los archivos de entrada.
CARPETA_DATOS = Path(r"C:\ruta\a\tu\carpeta\de\datos")

# Carpeta donde se guardarán los resultados.
CARPETA_SALIDA = Path(r"C:\ruta\a\tu\carpeta\de\salida")

# Nombre exacto del archivo a procesar dentro de CARPETA_DATOS.
NOMBRE_ARCHIVO_ENTRADA = "salidas_20260224.csv"

# Estación a analizar.
ESTACION_OBJETIVO = "(02103)Mazurén"

# Columnas mínimas necesarias.
COLUMNAS_REQUERIDAS = [
    "Fecha_Transaccion",
    "Tiempo",
    "Estacion",
    "Entradas_E",
    "Salidas_S",
]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalizar_nombre_base(texto: str) -> str:
    """Convierte un nombre a formato seguro para usarlo en nombres de archivo."""
    texto = Path(texto).stem.strip().lower()
    texto = re.sub(r"[^a-zA-Z0-9_-]+", "_", texto)
    texto = re.sub(r"_+", "_", texto).strip("_")
    return texto or "archivo"



def buscar_archivo_por_nombre(nombre_archivo: str, carpeta_datos: Path) -> Path:
    """Busca el archivo dentro de la carpeta de datos y sus subcarpetas."""
    coincidencias = list(carpeta_datos.rglob(nombre_archivo))

    if not coincidencias:
        raise FileNotFoundError(
            f"No se encontró '{nombre_archivo}' dentro de '{carpeta_datos.resolve()}'."
        )

    if len(coincidencias) > 1:
        rutas = "\n".join(str(r.resolve()) for r in coincidencias)
        raise FileExistsError(
            f"Se encontraron varias coincidencias para '{nombre_archivo}'.\n"
            f"Especifica mejor el nombre o deja solo una copia:\n{rutas}"
        )

    return coincidencias[0]



def detectar_encoding(ruta_archivo: Path) -> str:
    """Prueba primero UTF-8 con BOM y, si falla, usa latin-1."""
    encoding = "utf-8-sig"
    try:
        pd.read_csv(ruta_archivo, nrows=5, encoding=encoding)
    except UnicodeDecodeError:
        encoding = "latin-1"
    return encoding



def generar_ruta_salida(
    ruta_entrada: Path,
    carpeta_salida: Path,
    prefijo: str,
    sufijo: str,
    evitar_sobrescritura: bool = True,
) -> Path:
    """Genera un nombre automático basado en el archivo de entrada."""
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    nombre_base = normalizar_nombre_base(ruta_entrada.name)
    ruta_salida = carpeta_salida / f"{prefijo}_{nombre_base}{sufijo}"

    if not evitar_sobrescritura or not ruta_salida.exists():
        return ruta_salida

    version = 2
    while True:
        candidata = carpeta_salida / f"{prefijo}_{nombre_base}_v{version}{sufijo}"
        if not candidata.exists():
            return candidata
        version += 1


# ============================================================
# PROCESO PRINCIPAL
# ============================================================

def main() -> None:
    ruta_entrada = buscar_archivo_por_nombre(NOMBRE_ARCHIVO_ENTRADA, CARPETA_DATOS)
    ruta_salida_csv = generar_ruta_salida(
        ruta_entrada=ruta_entrada,
        carpeta_salida=CARPETA_SALIDA,
        prefijo="resumen_entradas_salidas",
        sufijo=".csv",
    )
    ruta_salida_png = generar_ruta_salida(
        ruta_entrada=ruta_entrada,
        carpeta_salida=CARPETA_SALIDA,
        prefijo="grafica_entradas_salidas",
        sufijo=".png",
    )

    encoding_usado = detectar_encoding(ruta_entrada)

    print(f"Archivo encontrado: {ruta_entrada.resolve()}")
    print(f"Codificación usada: {encoding_usado}")

    # ------------------------------------------------------------
    # PASO 1: leer archivo y validar estructura
    # ------------------------------------------------------------
    df = pd.read_csv(ruta_entrada, encoding=encoding_usado)
    df.columns = [col.strip() for col in df.columns]

    faltantes = [col for col in COLUMNAS_REQUERIDAS if col not in df.columns]
    if faltantes:
        raise ValueError(
            f"Faltan estas columnas en el archivo: {faltantes}\n"
            f"Columnas encontradas: {df.columns.tolist()}"
        )

    # ------------------------------------------------------------
    # PASO 2: limpiar texto, fechas y variables numéricas
    # ------------------------------------------------------------
    df["Estacion"] = (
        df["Estacion"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    )

    df = df[df["Estacion"] == ESTACION_OBJETIVO].copy()
    if df.empty:
        raise ValueError(f"No se encontraron datos para la estación {ESTACION_OBJETIVO}")

    df["Fecha_Transaccion"] = pd.to_datetime(df["Fecha_Transaccion"], errors="coerce")
    df["Tiempo"] = df["Tiempo"].astype(str).str.strip()

    # Construye una marca de tiempo completa uniendo fecha y hora.
    df["FechaHora"] = pd.to_datetime(
        df["Fecha_Transaccion"].dt.strftime("%Y-%m-%d") + " " + df["Tiempo"],
        errors="coerce",
    )

    df["Entradas_E"] = pd.to_numeric(df["Entradas_E"], errors="coerce").fillna(0)
    df["Salidas_S"] = pd.to_numeric(df["Salidas_S"], errors="coerce").fillna(0)

    df = df.dropna(subset=["FechaHora"])

    # ------------------------------------------------------------
    # PASO 3: agrupar por intervalo de tiempo
    # ------------------------------------------------------------
    # Esto evita duplicados si existe más de un registro en el mismo cuarto de hora.
    df_resumen = (
        df.groupby("FechaHora", as_index=False)[["Entradas_E", "Salidas_S"]]
        .sum()
        .sort_values("FechaHora")
    )

    df_resumen["Hora"] = df_resumen["FechaHora"].dt.strftime("%H:%M")
    df_resumen["Fecha"] = df_resumen["FechaHora"].dt.date
    df_resumen = df_resumen[["Fecha", "Hora", "FechaHora", "Entradas_E", "Salidas_S"]]

    # ------------------------------------------------------------
    # PASO 4: exportar CSV sin sobrescribir resultados previos
    # ------------------------------------------------------------
    df_resumen.to_csv(ruta_salida_csv, index=False, encoding="utf-8-sig")

    print("Proceso terminado.")
    print(f"Registros exportados: {len(df_resumen):,}")
    print(f"Archivo CSV generado: {ruta_salida_csv.resolve()}")

    # ------------------------------------------------------------
    # PASO 5: graficar y guardar imagen
    # ------------------------------------------------------------
    plt.figure(figsize=(14, 6))
    plt.plot(df_resumen["Hora"], df_resumen["Entradas_E"], label="Entradas_E")
    plt.plot(df_resumen["Hora"], df_resumen["Salidas_S"], label="Salidas_S")

    plt.title(f"Entradas y salidas por intervalo de 15 min - {ESTACION_OBJETIVO}")
    plt.xlabel("Hora del día")
    plt.ylabel("Cantidad")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta_salida_png, dpi=300)
    plt.show()

    print(f"Gráfica guardada en: {ruta_salida_png.resolve()}")


if __name__ == "__main__":
    main()
