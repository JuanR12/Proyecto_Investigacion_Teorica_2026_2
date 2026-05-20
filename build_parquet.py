"""
build_parquet.py  —  Convierte todos los xlsx de datos/ a Parquet.
Omite archivos que ya tienen su Parquet generado.

Uso:    python src/build_parquet.py
Salida: outputs/parquet/YYYY-MM-entradas.parquet
                        YYYY-MM-salidas.parquet

Schema:
    linea_id    string     ID de la troncal
    station_id  string     ID de la estación
    datetime    timestamp  fecha + intervalo 15 min
    entradas /
    salidas     int32      conteo agregado por estación

Deps: pip install pandas openpyxl pyarrow
"""

import re, sys, time
import pandas as pd
from pathlib import Path
from datetime import timedelta

DATA_DIR    = Path("datos")
OUTPUT_DIR  = Path("outputs/parquet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MESES = {
    "ENERO":1,"FEBRERO":2,"MARZO":3,"ABRIL":4,"MAYO":5,"JUNIO":6,
    "JULIO":7,"AGOSTO":8,"SEPTIEMBRE":9,"OCTUBRE":10,"NOVIEMBRE":11,"DICIEMBRE":12,
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_filename(name: str) -> dict | None:
    """Extrae año, mes y tipo de un nombre de archivo xlsx."""
    stem = name.upper().replace(".XLSX","").replace(".XLS","")
    tipo = "salidas" if stem.endswith("S") else "entradas"
    if tipo == "salidas": stem = stem[:-1]
    for mes, num in MESES.items():
        if stem.startswith(mes):
            yr = stem[len(mes):].strip()
            if not yr.isdigit(): return None
            yr = int(yr)
            yr = 2000 + yr if yr < 100 else yr
            return {"año": yr, "mes": num, "tipo": tipo}
    return None

def parquet_name(meta: dict) -> str:
    """YYYY-MM-tipo.parquet  →  2024-03-entradas.parquet"""
    return f"{meta['año']}-{meta['mes']:02d}-{meta['tipo']}.parquet"

def parse_id(raw: str) -> str | None:
    m = re.match(r"\((\d+)\)", str(raw).strip())
    return m.group(1).zfill(5) if m else None

def find_header_row(fp: Path) -> int:
    preview = pd.read_excel(fp, header=None, nrows=15, dtype=str)
    for i, row in preview.iterrows():
        vals = [str(v).strip() for v in row if pd.notna(v)]
        if any(v in ("Estación","ESTACION","Estacion") for v in vals):
            return i
    return 6

def convert(fp: Path, meta: dict) -> tuple[bool, str]:
    """
    Convierte un xlsx a Parquet. Retorna (éxito, mensaje).
    """
    try:
        header = find_header_row(fp)
        df = pd.read_excel(fp, skiprows=header, header=0, dtype=str)
        df = df.loc[:, ~df.columns.astype(str).str.contains("Total", case=False)]

        tipo = meta["tipo"]
        cols = list(df.columns)
        if tipo == "entradas":
            cols[:6] = ["_vacia","fase","linea_raw","estacion_raw","acceso","intervalo"]
            df.columns = cols
            df = df[~df["fase"].str.upper().str.contains("DUAL", na=False)]
            df = df.drop(columns=["_vacia","fase","acceso"])
        else:
            cols[:4] = ["linea_raw","estacion_raw","acceso","intervalo"]
            df.columns = cols
            df = df.drop(columns=["acceso"])

        df["linea_id"]   = df["linea_raw"].map(parse_id)
        df["station_id"] = df["estacion_raw"].map(parse_id)
        df = df.dropna(subset=["station_id","linea_id","intervalo"])
        df = df.drop(columns=["linea_raw","estacion_raw"])

        day_cols = [c for c in df.columns if c not in ("linea_id","station_id","intervalo")]
        df = df.melt(id_vars=["linea_id","station_id","intervalo"],
                     value_vars=day_cols, var_name="fecha", value_name="conteo")

        df["conteo"] = pd.to_numeric(df["conteo"], errors="coerce")
        df = df.dropna(subset=["conteo"])
        df = df[df["conteo"] >= 0]

        df["fecha"]    = pd.to_datetime(df["fecha"]).dt.date.astype(str)
        df["datetime"] = pd.to_datetime(df["fecha"] + " " + df["intervalo"].str.zfill(5))
        df = df.drop(columns=["fecha","intervalo"])

        df = (df.groupby(["linea_id","station_id","datetime"], as_index=False)["conteo"]
                .sum().rename(columns={"conteo": tipo}))

        df["linea_id"]   = df["linea_id"].astype("string")
        df["station_id"] = df["station_id"].astype("string")
        df[tipo]         = df[tipo].astype("int32")

        out = OUTPUT_DIR / parquet_name(meta)
        df.to_parquet(out, index=False)
        return True, f"{len(df):,} filas"

    except Exception as e:
        return False, str(e)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # Recolectar todos los xlsx con metadata válida
    all_files = []
    for fp in sorted(DATA_DIR.glob("*.xlsx")):
        meta = parse_filename(fp.name)
        if meta:
            all_files.append((fp, meta))
        else:
            print(f"[IGNORADO] {fp.name} — nombre no reconocido")

    total     = len(all_files)
    skipped   = 0
    converted = 0
    errors    = []
    last_pct  = -1

    print(f"\n{total} archivos encontrados en '{DATA_DIR}'\n")

    for i, (fp, meta) in enumerate(all_files, 1):
        dest = OUTPUT_DIR / parquet_name(meta)

        # Saltar si el parquet ya existe y es más reciente que el xlsx
        if dest.exists() and dest.stat().st_mtime >= fp.stat().st_mtime:
            skipped += 1
        else:
            ok, msg = convert(fp, meta)
            if ok:
                converted += 1
            else:
                errors.append((fp.name, msg))

        pct = int(i / total * 100)
        if pct // 10 > last_pct // 10:
            elapsed = timedelta(seconds=int(time.time() - t0))
            # Estimar tiempo restante
            eta = timedelta(seconds=int((time.time()-t0) / i * (total-i)))
            print(f"  {pct:3d}%  —  {i}/{total} archivos  —  {elapsed} transcurrido  —  ETA {eta}")
            last_pct = pct

    elapsed = timedelta(seconds=int(time.time() - t0))
    print(f"\n── Resumen ──────────────────────────────────────────")
    print(f"  Convertidos: {converted}")
    print(f"  Omitidos (ya existían): {skipped}")
    print(f"  Errores:     {len(errors)}")
    if errors:
        print(f"\n  Archivos con error:")
        for name, msg in errors:
            print(f"    {name}: {msg}")
    print(f"\n  Tiempo total: {elapsed}")
    print(f"  Parquets en: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
