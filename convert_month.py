"""
convert_month.py  —  Convierte un mes de datos xlsx a Parquet.
Uso:    python src/convert_month.py DICIEMBRE25
Salida: outputs/25_12_E.parquet  y  outputs/25_12_S.parquet

Schema:
    linea_id    string     ID de la troncal (5 dígitos)
    station_id  string     ID de la estación (5 dígitos)
    datetime    timestamp  fecha + intervalo de 15 min
    entradas/salidas int32 conteo agregado de todos los accesos

Deps: pip install pandas openpyxl pyarrow
"""

import re, sys, time
import pandas as pd
from pathlib import Path
from datetime import timedelta

DATA_DIR   = Path("datos")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MESES = {
    "ENERO":"01","FEBRERO":"02","MARZO":"03","ABRIL":"04","MAYO":"05","JUNIO":"06",
    "JULIO":"07","AGOSTO":"08","SEPTIEMBRE":"09","OCTUBRE":"10","NOVIEMBRE":"11","DICIEMBRE":"12",
}

def parse_id(raw: str) -> str | None:
    m = re.match(r"\((\d+)\)", str(raw).strip())
    return m.group(1).zfill(5) if m else None

def find_header_row(fp: Path) -> int:
    preview = pd.read_excel(fp, header=None, nrows=15, dtype=str)
    for i, row in preview.iterrows():
        vals = [str(v).strip() for v in row if pd.notna(v)]
        if any(v in ("Estación", "ESTACION", "Estacion") for v in vals):
            return i
    return 6

def convert(fp: Path, tipo: str, yy: str, mm: str) -> None:
    t0 = time.time()
    print(f"  Leyendo {fp.name}...", flush=True)

    header = find_header_row(fp)
    df = pd.read_excel(fp, skiprows=header, header=0, dtype=str)

    # Eliminar columna Total general
    df = df.loc[:, ~df.columns.astype(str).str.contains("Total", case=False)]

    # Renombrar columnas según formato
    # Salidas:  Línea(0) | Estación(1) | Acceso(2) | Intervalo(3) | días...
    # Entradas: vacía(0) | Fase(1) | Línea(2) | Estación(3) | Acceso(4) | Intervalo(5) | días...
    cols = list(df.columns)
    if tipo == "entradas":
        cols[:6] = ["_vacia", "fase", "linea_raw", "estacion_raw", "acceso", "intervalo"]
        df.columns = cols
        df = df[~df["fase"].str.upper().str.contains("DUAL", na=False)]
        df = df.drop(columns=["_vacia", "fase", "acceso"])
    else:
        cols[:4] = ["linea_raw", "estacion_raw", "acceso", "intervalo"]
        df.columns = cols
        df = df.drop(columns=["acceso"])

    # Parsear IDs y descartar filas sin datos válidos
    df["linea_id"]   = df["linea_raw"].map(parse_id)
    df["station_id"] = df["estacion_raw"].map(parse_id)
    df = df.dropna(subset=["station_id", "linea_id", "intervalo"])
    df = df.drop(columns=["linea_raw", "estacion_raw"])

    # Columnas de días = todo lo que no sea las de identidad
    day_cols = [c for c in df.columns if c not in ("linea_id", "station_id", "intervalo")]
    print(f"  {len(df):,} filas  |  {len(day_cols)} días  —  melt...", flush=True)

    # Formato largo: una fila por (estación, datetime)
    df = df.melt(
        id_vars=["linea_id", "station_id", "intervalo"],
        value_vars=day_cols,
        var_name="fecha",
        value_name="conteo",
    )

    df["conteo"] = pd.to_numeric(df["conteo"], errors="coerce")
    df = df.dropna(subset=["conteo"])
    df = df[df["conteo"] >= 0]

    # Construir datetime completo: fecha (del header de columna) + intervalo HH:MM
    df["fecha"]    = pd.to_datetime(df["fecha"]).dt.date.astype(str)
    df["datetime"] = pd.to_datetime(df["fecha"] + " " + df["intervalo"].str.zfill(5))
    df = df.drop(columns=["fecha", "intervalo"])

    # Agregar todos los accesos de una misma estación en el mismo intervalo
    print(f"  Agregando por estación...", flush=True)
    df = (df.groupby(["linea_id", "station_id", "datetime"], as_index=False)["conteo"]
            .sum()
            .rename(columns={"conteo": tipo}))

    df["linea_id"]   = df["linea_id"].astype("string")
    df["station_id"] = df["station_id"].astype("string")
    df[tipo]         = df[tipo].astype("int32")

    suffix = "E" if tipo == "entradas" else "S"
    out    = OUTPUT_DIR / f"{yy}_{mm}_{suffix}.parquet"
    df.to_parquet(out, index=False)

    elapsed = timedelta(seconds=int(time.time() - t0))
    print(f"  Guardado: {out.name}  |  {len(df):,} filas  |  {elapsed}")

def main():
    if len(sys.argv) < 2:
        print("Uso: python src/convert_month.py DICIEMBRE25")
        sys.exit(1)

    raw = sys.argv[1].upper().replace(".XLSX", "").replace(".XLS", "")
    if raw.endswith("S"): raw = raw[:-1]

    mes_str = next((m for m in sorted(MESES, key=len, reverse=True) if raw.startswith(m)), None)
    if not mes_str:
        print(f"[ERROR] Mes no reconocido en '{sys.argv[1]}'"); sys.exit(1)

    yr = raw[len(mes_str):].strip()
    if not yr.isdigit():
        print(f"[ERROR] Año no reconocido: '{yr}'"); sys.exit(1)

    yy4 = ("20" + yr) if len(yr) == 2 else yr
    yy2 = yy4[2:]
    mm  = MESES[mes_str]

    print(f"\n{mes_str} {yy4}  →  Parquet\n")

    for sufijo, tipo in [("S", "salidas"), ("", "entradas")]:
        fp = DATA_DIR / f"{mes_str}{yy2}{sufijo}.xlsx"
        if not fp.exists():
            fp = DATA_DIR / f"{mes_str}{yy4}{sufijo}.xlsx"
        if not fp.exists():
            print(f"  [WARN] No encontrado: {mes_str}{yy2}{sufijo}.xlsx")
            continue
        convert(fp, tipo, yy2, mm)

    print("\nListo.")

if __name__ == "__main__":
    main()
