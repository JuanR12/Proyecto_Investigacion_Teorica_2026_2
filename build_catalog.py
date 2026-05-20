"""
build_catalog.py  —  Construye el catálogo de estaciones troncales.
Uso: python src/build_catalog.py
Deps: pip install pandas openpyxl pyarrow
"""

import re, sys, time
import pandas as pd
from pathlib import Path
from datetime import timedelta

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("datos")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MESES = {
    "ENERO":1,"FEBRERO":2,"MARZO":3,"ABRIL":4,"MAYO":5,"JUNIO":6,
    "JULIO":7,"AGOSTO":8,"SEPTIEMBRE":9,"OCTUBRE":10,"NOVIEMBRE":11,"DICIEMBRE":12,
}
TARGET_YEARS = {2019, 2021, 2022, 2023, 2024, 2025}

# ── Helpers ────────────────────────────────────────────────────────────────────
def parse_filename(name: str) -> dict | None:
    name = name.upper().replace(".XLSX","").replace(".XLS","")
    tipo = "salidas" if name.endswith("S") else "entradas"
    if tipo == "salidas": name = name[:-1]
    for mes, num in MESES.items():
        if name.startswith(mes):
            yr = name[len(mes):].strip()
            if not yr.isdigit(): return None
            yr = int(yr); yr = 2000+yr if yr < 100 else yr
            return {"mes":num,"mes_nombre":mes,"año":yr,"tipo":tipo} if yr in TARGET_YEARS else None
    return None

def find_header_row(filepath: Path) -> int:
    preview = pd.read_excel(filepath, header=None, nrows=15, dtype=str)
    for i, row in preview.iterrows():
        vals = [str(v).strip() for v in row if pd.notna(v)]
        if any(v in ("Estación","ESTACION","Estacion") for v in vals):
            return i
    return 6

def parse_id_name(raw: str) -> tuple[str, str]:
    m = re.match(r"\((\d+)\)\s*(.*)", str(raw).strip())
    return (m.group(1).zfill(5), m.group(2).strip()) if m else ("UNKNOWN", raw)

def extract_stations(filepath: Path, meta: dict) -> pd.DataFrame:
    header = find_header_row(filepath)
    tipo   = meta["tipo"]

    # Entradas: col 0 vacía | col 1 Fase | col 2 Línea | col 3 Estación
    # Salidas:  col 0 Línea | col 1 Estación
    if tipo == "entradas":
        cols = [1, 2, 3]
        col_names = ["fase_raw", "linea_raw", "estacion_raw"]
    else:
        cols = [0, 1]
        col_names = ["linea_raw", "estacion_raw"]

    try:
        df = pd.read_excel(filepath, skiprows=header, usecols=cols, dtype=str)
    except Exception as e:
        print(f"\n  [ERROR] {filepath.name}: {e}")
        return pd.DataFrame()

    df.columns = col_names

    # Entradas: descartar filas del sistema Dual
    if tipo == "entradas":
        df = df[~df["fase_raw"].str.upper().str.contains("DUAL", na=False)]

    df = df.dropna(subset=["estacion_raw"])
    df = df[df["estacion_raw"].str.match(r"\(\d+\)", na=False)]

    if df.empty:
        print(f"\n  [WARN] {filepath.name}: sin estaciones troncales")
        return pd.DataFrame()

    df["station_id"], df["station_name"] = zip(*df["estacion_raw"].map(parse_id_name))
    df["linea_id"],   df["linea_name"]   = zip(*df["linea_raw"].map(parse_id_name))
    df = df[df["station_id"] != "UNKNOWN"]
    df["año"], df["mes"] = meta["año"], meta["mes"]

    return df[["station_id","station_name","linea_id","linea_name","año","mes"]].drop_duplicates()

def build_catalog(df: pd.DataFrame) -> pd.DataFrame:
    nombre = (df.sort_values(["año","mes"], ascending=False)
                .drop_duplicates("station_id")[["station_id","station_name"]])
    lineas = (df.dropna(subset=["linea_name"])
                .groupby("station_id")["linea_name"]
                .apply(lambda x: ", ".join(sorted(x.unique())))
                .reset_index().rename(columns={"linea_name":"lineas"}))
    aparicion = (df.assign(fecha=pd.to_datetime(df["año"].astype(str)+"-"+df["mes"].astype(str).str.zfill(2)+"-01"))
                   .groupby("station_id")["fecha"]
                   .agg(primera_aparicion="min", ultima_aparicion="max").reset_index())
    historial = (df.drop_duplicates(["station_id","station_name"])
                   .groupby("station_id")["station_name"]
                   .apply(lambda x: " | ".join(sorted(x.unique())))
                   .reset_index().rename(columns={"station_name":"nombres_historicos"}))
    return (nombre.merge(lineas,"left","station_id")
                  .merge(aparicion,"left","station_id")
                  .merge(historial,"left","station_id")
                  .sort_values("station_id").reset_index(drop=True))

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    files = sorted(DATA_DIR.glob("*.xlsx")) + sorted(DATA_DIR.glob("*.xls"))
    total = len(files)
    print(f"{total} archivos encontrados. Años objetivo: {sorted(TARGET_YEARS)}\n")

    all_records, errors, skipped = [], [], []
    last_pct = -1

    for i, fp in enumerate(files, 1):
        meta = parse_filename(fp.name)
        if meta is None:
            skipped.append(fp.name)
        else:
            rec = extract_stations(fp, meta)
            (all_records if not rec.empty else errors).append(fp.name if rec.empty else rec)

        pct = int(i / total * 100)
        milestone = pct // 10
        if milestone > last_pct // 10:
            elapsed = timedelta(seconds=int(time.time()-t0))
            print(f"  {pct:3d}%  —  {i}/{total} archivos  —  {elapsed} transcurrido")
            last_pct = pct

    print(f"\nOK: {len(all_records)}  |  Errores: {len(errors)}  |  Ignorados: {len(skipped)}")
    if errors:
        print("  Errores:", ", ".join(errors))
    if not all_records:
        print("[ERROR] Sin datos válidos."); return

    df = pd.concat(all_records, ignore_index=True)
    print(f"Registros: {len(df):,}  |  Estaciones únicas: {df['station_id'].nunique()}")

    cat = build_catalog(df)
    print(f"Catálogo: {len(cat)} estaciones")

    cambios = cat[cat["nombres_historicos"].str.contains(r"\|", na=False)]
    if not cambios.empty:
        print(f"\nCambios de nombre detectados ({len(cambios)}):")
        print(cambios[["station_id","nombres_historicos"]].to_string(index=False))

    cat.to_csv(OUTPUT_DIR/"catalogo_estaciones.csv", index=False, encoding="utf-8-sig")
    try:
        cat.to_parquet(OUTPUT_DIR/"catalogo_estaciones.parquet", index=False)
        print("\nGuardado: CSV + Parquet")
    except ImportError:
        print("\nGuardado: solo CSV (instalar pyarrow para Parquet)")

    print(f"Tiempo total: {timedelta(seconds=int(time.time()-t0))}")

if __name__ == "__main__":
    main()
