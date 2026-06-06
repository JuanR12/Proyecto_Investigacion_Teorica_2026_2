from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# ============================================================
# CONFIGURACIÓN GENERAL — ajusta las rutas antes de ejecutar
# ============================================================
# Carpeta donde están los archivos CSV de entrada (datos crudos transaccionales).
# Nota: estos archivos NO son los parquet del pipeline principal.
CARPETA_DATOS = Path(r"C:\ruta\a\tu\carpeta\de\datos")

# Carpeta donde se guardarán los resultados filtrados.
# Se crea automáticamente si no existe.
CARPETA_SALIDA = Path(r"C:\ruta\a\tu\carpeta\de\salida")

# Nombre exacto del archivo a procesar dentro de CARPETA_DATOS.
# Ejemplo: "20260408.csv"
NOMBRE_ARCHIVO_ENTRADA = "20260401.csv"

# Filtro por zona en la columna "Linea"
# Ver lista de nombres correctos en la carpeta "INFORMACIÓN_BASES_DE_DATOS"
USAR_FILTRO_LINEA = True
VALOR_LINEA = "(32) Zona C Av. Suba"

# Filtro por estación/parada troncal en la columna "Estacion_Parada"
# Ver lista de nombres correctos en la carpeta "INFORMACIÓN_BASES_DE_DATOS"
USAR_FILTRO_ESTACION = False
VALOR_ESTACION = "(05103) Marsella"

# Filtro por tipo de tarjeta en la columna "Tipo_Tarjeta"
USAR_FILTRO_TIPO_TARJETA = True
VALOR_TIPO_TARJETA = "tullave Plus"

# Tamaño de lectura por bloques. Útil para archivos grandes.
TAMANO_BLOQUE = 500_000

# Columnas mínimas necesarias para el proceso.
COLUMNAS_INTERES = [
    "Estacion_Parada",
    "Fecha_Transaccion",
    "Linea",
    "Numero_Tarjeta",
    "Tipo_Tarjeta",
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
    """
    Busca el archivo dentro de la carpeta de datos y sus subcarpetas.
    Falla con error claro si no existe o si aparece más de una coincidencia.
    """
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
    sufijo: str = ".csv",
    evitar_sobrescritura: bool = True,
) -> Path:
    """
    Genera automáticamente el nombre del archivo de salida usando el nombre original.

    Ejemplo:
    datos_filtrados_20260408.csv

    Si el archivo ya existe, agrega _v2, _v3, etc. para no sobrescribir.
    """
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
    ruta_salida = generar_ruta_salida(
        ruta_entrada=ruta_entrada,
        carpeta_salida=CARPETA_SALIDA,
        prefijo="datos_filtrados_tarjetas_repetidas",
    )

    encoding_usado = detectar_encoding(ruta_entrada)

    # Leer solo encabezados para validar columnas antes de procesar el archivo completo.
    encabezado = pd.read_csv(ruta_entrada, nrows=0, encoding=encoding_usado)
    encabezado.columns = [col.strip() for col in encabezado.columns]

    columnas_faltantes = [col for col in COLUMNAS_INTERES if col not in encabezado.columns]
    if columnas_faltantes:
        raise ValueError(
            f"Faltan estas columnas en el archivo: {columnas_faltantes}\n"
            f"Columnas encontradas: {encabezado.columns.tolist()}"
        )

    print(f"Archivo encontrado: {ruta_entrada.resolve()}")
    print(f"Codificación usada: {encoding_usado}")

    # ------------------------------------------------------------
    # PASO 1: leer por bloques, limpiar y aplicar filtros básicos
    # ------------------------------------------------------------
    partes_filtradas: list[pd.DataFrame] = []

    for bloque in pd.read_csv(
        ruta_entrada,
        usecols=COLUMNAS_INTERES,
        chunksize=TAMANO_BLOQUE,
        encoding=encoding_usado,
    ):
        bloque.columns = [col.strip() for col in bloque.columns]

        bloque["Linea"] = (
        bloque["Linea"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        )
        bloque["Estacion_Parada"] = (
            bloque["Estacion_Parada"]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
        bloque["Numero_Tarjeta"] = bloque["Numero_Tarjeta"].astype(str).str.strip()
        bloque["Tipo_Tarjeta"] = (
            bloque["Tipo_Tarjeta"]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
        bloque["Fecha_Transaccion"] = pd.to_datetime(
                bloque["Fecha_Transaccion"], errors="coerce"
        )

        if USAR_FILTRO_LINEA:
            bloque = bloque[bloque["Linea"] == VALOR_LINEA]

        if USAR_FILTRO_ESTACION:
            bloque = bloque[bloque["Estacion_Parada"] == VALOR_ESTACION]

        if USAR_FILTRO_TIPO_TARJETA:
            bloque = bloque[bloque["Tipo_Tarjeta"] == VALOR_TIPO_TARJETA]

        if not bloque.empty:
            partes_filtradas.append(bloque)
            
        if not partes_filtradas:
            raise SystemExit("No se encontraron registros con los filtros aplicados.")

    df = pd.concat(partes_filtradas, ignore_index=True)

    # ------------------------------------------------------------
    # PASO 2: ordenar por tarjeta y fecha para dejar la trazabilidad
    # ------------------------------------------------------------
    df = df.sort_values(
        by=["Numero_Tarjeta", "Fecha_Transaccion"],
        ascending=[True, True],
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # PASO 3: conservar solo tarjetas con más de una fecha/hora distinta
    # ------------------------------------------------------------
    conteo_fechas = df.groupby("Numero_Tarjeta")["Fecha_Transaccion"].nunique()
    tarjetas_validas = conteo_fechas[conteo_fechas > 1].index
    df_repetidas = df[df["Numero_Tarjeta"].isin(tarjetas_validas)].copy()

    # ------------------------------------------------------------
    # PASO 4: exportar resultado sin sobrescribir automáticamente
    # ------------------------------------------------------------
    df_repetidas.to_csv(ruta_salida, index=False, encoding="utf-8-sig")

    print("Proceso terminado.")
    print(f"Registros tras filtros: {len(df):,}")
    print(f"Registros con tarjetas repetidas: {len(df_repetidas):,}")
    print(f"Tarjetas repetidas únicas: {df_repetidas['Numero_Tarjeta'].nunique():,}")
    print(f"Archivo guardado en: {ruta_salida.resolve()}")


if __name__ == "__main__":
    main()
