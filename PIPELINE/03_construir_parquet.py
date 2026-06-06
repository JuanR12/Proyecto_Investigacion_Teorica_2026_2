"""
build_parquet.py  —  Convierte todos los xlsx de datos/ a Parquet.
Reconvierte entradas desde cero (filtro corregido).
Omite salidas que ya existen y están verificadas.

Uso:    python src/build_parquet.py
Salida: outputs/parquet/YYYY-MM-entradas.parquet
                        YYYY-MM-salidas.parquet

Deps: pip install pandas openpyxl pyarrow
"""

import re, sys, time, random
import pandas as pd
from pathlib import Path
from datetime import timedelta

PROYECTO_RAIZ = Path(__file__).resolve().parent.parent
DATA_DIR    = PROYECTO_RAIZ / "datos"
PARQUET_DIR = PROYECTO_RAIZ / "outputs" / "parquet"
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

MESES = {
    "ENERO":1,"FEBRERO":2,"MARZO":3,"ABRIL":4,"MAYO":5,"JUNIO":6,
    "JULIO":7,"AGOSTO":8,"SEPTIEMBRE":9,"OCTUBRE":10,"NOVIEMBRE":11,"DICIEMBRE":12,
}
TARGET_YEARS = {2019, 2022, 2023, 2024, 2025, 2026}

# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_filename(name: str) -> dict | None:
    stem = name.upper().replace(".XLSX","").replace(".XLS","")
    tipo = "salidas" if stem.endswith("S") else "entradas"
    if tipo == "salidas": stem = stem[:-1]
    for mes, num in MESES.items():
        if stem.startswith(mes):
            yr = stem[len(mes):].strip()
            if not yr.isdigit(): return None
            yr = int(yr)
            yr = 2000 + yr if yr < 100 else yr
            return {"año": yr, "mes": num, "tipo": tipo} if yr in TARGET_YEARS else None
    return None

def parquet_name(meta: dict) -> str:
    return f"{meta['año']}-{meta['mes']:02d}-{meta['tipo']}.parquet"

def parse_id(raw: str) -> str | None:
    m = re.match(r"\((\d+)\)", str(raw).strip())
    return m.group(1).zfill(5) if m else None

def find_header(fp: Path) -> int:
    preview = pd.read_excel(fp, header=None, nrows=15, dtype=str)
    for i, row in preview.iterrows():
        if any(str(v).strip() in ("Estación","Estacion","ESTACION") for v in row if pd.notna(v)):
            return i
    return 6

def is_new_salidas_fmt(first_col: str) -> bool:
    """
    Formato B (2024 parcial en adelante): primera columna es 'Linea' sin tilde.
    Formato A (2019-2023 y algunos 2024): primera columna es 'Línea' con tilde.
    """
    return str(first_col).strip() in ("Linea", "LINEA")

def parse_day_col(col, new_fmt: bool) -> str | None:
    s = str(col).strip()
    if new_fmt:
        clean = s.lstrip("_").replace("_", "-")
        return clean if re.match(r"\d{4}-\d{2}-\d{2}", clean) else None
    try:
        return pd.to_datetime(s).strftime("%Y-%m-%d")
    except:
        return None

# ── Conversión ─────────────────────────────────────────────────────────────────

def convert(fp: Path, meta: dict) -> tuple[bool, str]:
    try:
        tipo  = meta["tipo"]
        año   = meta["año"]
        mes   = meta["mes"]

        header = find_header(fp)
        df = pd.read_excel(fp, skiprows=header, header=0, dtype=str)

        # Eliminar columna Total general
        df = df.loc[:, ~df.columns.astype(str).str.contains("Total", case=False)]

        cols = list(df.columns)

        if tipo == "entradas":
            # Formato único todos los años:
            # Unnamed(0) | Fase(1) | Línea(2) | Estación(3) | Acceso(4) | Intervalo(5) | días...
            cols[:6] = ["_vacia","fase","linea_raw","estacion_raw","acceso","intervalo"]
            df.columns = cols
            # Filtro correcto: solo filas donde Línea contiene "Zona" (troncal real)
            df = df[df["linea_raw"].str.contains("Zona", case=False, na=False)]
            df = df.drop(columns=["_vacia","fase","acceso"])

        else:
            new_fmt = is_new_salidas_fmt(cols[0])
            # Formato A: Línea(0) | Estación(1) | Acceso(2) | INTERVALO(3) | días datetime...
            # Formato B: Linea(0) | Estacion(1) | Acceso_Estacion(2) | Intervalo(3) | días _YYYY_MM_DD...
            cols[:4] = ["linea_raw","estacion_raw","acceso","intervalo"]
            df.columns = cols
            df = df.drop(columns=["acceso"])

        # Filtrar solo filas con estación en formato (NNNNN)
        df = df[df["estacion_raw"].str.match(r"\(\d+\)", na=False)]

        # Parsear IDs
        df["linea_id"]   = df["linea_raw"].map(parse_id)
        df["station_id"] = df["estacion_raw"].map(parse_id)
        df = df.dropna(subset=["station_id","linea_id","intervalo"])
        df = df.drop(columns=["linea_raw","estacion_raw"])

        # Detectar formato de columnas de días
        new_day_fmt = tipo == "salidas" and is_new_salidas_fmt(
            # re-leer primera columna del original para detectar
            pd.read_excel(fp, skiprows=header, header=0, nrows=0).columns[0]
        )

        non_id = {"linea_id","station_id","intervalo"}
        day_cols = {c: parse_day_col(c, new_day_fmt)
                    for c in df.columns if c not in non_id}
        # Solo días del mes correcto
        day_cols = {k: v for k, v in day_cols.items()
                    if v and pd.Timestamp(v).year == año
                    and pd.Timestamp(v).month == mes}

        if not day_cols:
            return False, "Sin columnas de días válidas para el mes"

        df = df.melt(
            id_vars=["linea_id","station_id","intervalo"],
            value_vars=list(day_cols.keys()),
            var_name="fecha_col",
            value_name="conteo",
        )
        df["fecha"]   = df["fecha_col"].map(day_cols)
        df["conteo"]  = pd.to_numeric(df["conteo"], errors="coerce")
        df = df.dropna(subset=["conteo"])
        df = df[df["conteo"] >= 0]

        # Normalizar intervalo HH:MM (acepta HH:MM y HH:MM:SS)
        df["intervalo"] = df["intervalo"].str.strip().str[:5]
        df["datetime"]  = pd.to_datetime(
            df["fecha"] + " " + df["intervalo"], format="%Y-%m-%d %H:%M"
        )
        df = df.drop(columns=["fecha_col","fecha","intervalo"])

        # Agregar todos los accesos por estación
        df = (df.groupby(["linea_id","station_id","datetime"], as_index=False)["conteo"]
                .sum()
                .rename(columns={"conteo": tipo}))

        df["linea_id"]   = df["linea_id"].astype("string")
        df["station_id"] = df["station_id"].astype("string")
        df[tipo]         = df[tipo].astype("int32")

        out = PARQUET_DIR / parquet_name(meta)
        df.to_parquet(out, index=False)
        return True, f"{len(df):,} filas | {df['station_id'].nunique()} estaciones"

    except Exception as e:
        return False, str(e)

# ── Verificación ───────────────────────────────────────────────────────────────

def verify(path: Path, tipo: str) -> tuple[bool, str]:
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return False, "archivo vacío"
        required = {"linea_id","station_id","datetime",tipo}
        missing  = required - set(df.columns)
        if missing:
            return False, f"columnas faltantes: {missing}"
        df_op = df[df["datetime"].dt.hour.between(6, 22)]
        if df_op.empty:
            return False, "sin datos en horario operativo"
        linea   = random.choice(df_op["linea_id"].unique())
        muestra = df_op[df_op["linea_id"] == linea]
        nans    = muestra[tipo].isna().sum()
        negs    = (muestra[tipo] < 0).sum()
        if nans > 0: return False, f"{nans} NaN en muestra"
        if negs > 0: return False, f"{negs} negativos en muestra"
        return True, f"OK ({len(df):,} filas | linea muestra: {linea})"
    except Exception as e:
        return False, str(e)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # Reconvertir entradas desde cero (filtro corregido)
    entradas_existentes = list(PARQUET_DIR.glob("*-entradas.parquet"))
    if entradas_existentes:
        print(f"Eliminando {len(entradas_existentes)} parquets de entradas para reconversión...")
        for fp in entradas_existentes:
            fp.unlink()

    all_files = []
    for fp in sorted(DATA_DIR.glob("*.xlsx")):
        meta = parse_filename(fp.name)
        if meta:
            all_files.append((fp, meta))
        else:
            print(f"[IGNORADO] {fp.name}")

    total    = len(all_files)
    converted, skipped, errors = 0, 0, []
    last_pct = -1

    print(f"\n{total} archivos a procesar\n")

    for i, (fp, meta) in enumerate(all_files, 1):
        dest = PARQUET_DIR / parquet_name(meta)
        tipo = meta["tipo"]

        if dest.exists() and dest.stat().st_mtime >= fp.stat().st_mtime:
            ok, msg = verify(dest, tipo)
            if ok:
                skipped += 1
            else:
                print(f"\n  [REVERIF FALLO] {dest.name}: {msg} — reconvirtiendo")
                ok, msg = convert(fp, meta)
                if ok: converted += 1
                else:  errors.append((fp.name, msg))
        else:
            ok, msg = convert(fp, meta)
            if ok:
                converted += 1
                v_ok, v_msg = verify(dest, tipo)
                if not v_ok:
                    errors.append((fp.name, f"conversión OK pero verificación falló: {v_msg}"))
            else:
                errors.append((fp.name, msg))

        pct = int(i / total * 100)
        if pct // 10 > last_pct // 10:
            elapsed = timedelta(seconds=int(time.time()-t0))
            eta     = timedelta(seconds=int((time.time()-t0)/i*(total-i)))
            print(f"  {pct:3d}%  —  {i}/{total}  —  {elapsed} transcurrido  —  ETA {eta}")
            last_pct = pct

    elapsed = timedelta(seconds=int(time.time()-t0))
    print(f"\n── Resumen ──────────────────────────────────────────")
    print(f"  Convertidos:            {converted}")
    print(f"  Omitidos (verificados): {skipped}")
    print(f"  Errores:                {len(errors)}")
    if errors:
        print(f"\n  Archivos con error:")
        for name, msg in errors:
            print(f"    {name}: {msg}")
    print(f"\n  Tiempo total: {elapsed}")

if __name__ == "__main__":
    main()
