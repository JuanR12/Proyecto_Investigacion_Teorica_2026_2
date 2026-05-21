"""
build_parquet.py  —  Convierte todos los xlsx de datos/ a Parquet.
Omite archivos que ya tienen su Parquet generado y verificado.

Uso:    python src/build_parquet.py
Salida: outputs/parquet/YYYY-MM-entradas.parquet
                        YYYY-MM-salidas.parquet

Deps: pip install pandas openpyxl pyarrow
"""

import re, sys, time, random
import pandas as pd
from pathlib import Path
from datetime import timedelta

DATA_DIR   = Path("datos")
OUTPUT_DIR = Path("outputs/parquet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MESES = {
    "ENERO":1,"FEBRERO":2,"MARZO":3,"ABRIL":4,"MAYO":5,"JUNIO":6,
    "JULIO":7,"AGOSTO":8,"SEPTIEMBRE":9,"OCTUBRE":10,"NOVIEMBRE":11,"DICIEMBRE":12,
}

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
            return {"año": yr, "mes": num, "tipo": tipo}
    return None

def parquet_name(meta: dict) -> str:
    return f"{meta['año']}-{meta['mes']:02d}-{meta['tipo']}.parquet"

def parse_id(raw: str) -> str | None:
    m = re.match(r"\((\d+)\)", str(raw).strip())
    return m.group(1).zfill(5) if m else None

def find_header_row(fp: Path) -> int:
    preview = pd.read_excel(fp, header=None, nrows=15, dtype=str)
    for i, row in preview.iterrows():
        vals = [str(v).strip() for v in row if pd.notna(v)]
        if any(v in ("Estación","ESTACION","Estacion","Estacion") for v in vals):
            return i
    return 6

def is_new_salidas_format(cols: list) -> bool:
    """
    Detecta si el archivo de salidas usa el formato nuevo (2024+).
    Nuevo: columnas de días como '_2024_04_01' (string con guión bajo).
    Viejo: columnas de días como datetime objects '2024-04-01 00:00:00'.
    """
    day_cols = [str(c) for c in cols if str(c) not in
                ("Línea","Linea","Estación","Estacion","Acceso de Estación",
                 "Acceso_Estacion","INTERVALO","Intervalo","Total general","Unnamed: 0")]
    return any(str(c).startswith("_") for c in day_cols)

def parse_day_col(col, new_format: bool) -> str | None:
    """
    Convierte una columna de día a string de fecha 'YYYY-MM-DD'.
    Nuevo formato: '_2024_04_01' → '2024-04-01'
    Viejo formato: datetime object o '2024-04-01 00:00:00' → '2024-04-01'
    """
    s = str(col).strip()
    if new_format:
        # '_2024_04_01' → strip '_' → '2024_04_01' → reemplazar '_' → '2024-04-01'
        clean = s.lstrip("_").replace("_", "-")
        return clean if re.match(r"\d{4}-\d{2}-\d{2}", clean) else None
    else:
        try:
            return pd.to_datetime(s).strftime("%Y-%m-%d")
        except Exception:
            return None

# ── Conversión ─────────────────────────────────────────────────────────────────

def convert(fp: Path, meta: dict) -> tuple[bool, str]:
    try:
        tipo   = meta["tipo"]
        año    = meta["año"]
        mes    = meta["mes"]
        header = find_header_row(fp)

        df = pd.read_excel(fp, skiprows=header, header=0, dtype=str)

        # Eliminar Total general
        df = df.loc[:, ~df.columns.astype(str).str.contains("Total", case=False)]

        cols = list(df.columns)

        if tipo == "entradas":
            # Formato único en todos los años:
            # vacía(0) | Fase(1) | Línea(2) | Estación(3) | Acceso(4) | Intervalo(5) | días...
            cols[:6] = ["_vacia","fase","linea_raw","estacion_raw","acceso","intervalo"]
            df.columns = cols
            df = df[~df["fase"].str.upper().str.contains("DUAL", na=False)]
            df = df.drop(columns=["_vacia","fase","acceso"])
        else:
            new_fmt = is_new_salidas_format(cols)
            if new_fmt:
                # Nuevo formato 2024+: Linea | Estacion | Acceso_Estacion | Intervalo | _YYYY_MM_DD...
                cols[:4] = ["linea_raw","estacion_raw","acceso","intervalo"]
            else:
                # Viejo formato 2019-2023: Línea | Estación | Acceso de Estación | INTERVALO | días...
                cols[:4] = ["linea_raw","estacion_raw","acceso","intervalo"]
            df.columns = cols
            df = df.drop(columns=["acceso"])

        # Parsear IDs
        df["linea_id"]   = df["linea_raw"].map(parse_id)
        df["station_id"] = df["estacion_raw"].map(parse_id)
        df = df.dropna(subset=["station_id","linea_id","intervalo"])
        df = df.drop(columns=["linea_raw","estacion_raw"])

        # Identificar columnas de días y filtrar solo las del mes correcto
        new_fmt = tipo == "salidas" and is_new_salidas_format(list(df.columns))
        non_day = {"linea_id","station_id","intervalo"}
        day_cols_raw = [c for c in df.columns if c not in non_day]

        # Construir mapa col_original → fecha_str, filtrando días fuera del mes
        valid_day_cols = {}
        for c in day_cols_raw:
            fecha_str = parse_day_col(c, new_fmt)
            if fecha_str is None:
                continue
            try:
                dt = pd.Timestamp(fecha_str)
                if dt.year == año and dt.month == mes:
                    valid_day_cols[c] = fecha_str
            except Exception:
                continue

        if not valid_day_cols:
            return False, "No se encontraron columnas de días válidas para el mes"

        df = df.melt(
            id_vars=["linea_id","station_id","intervalo"],
            value_vars=list(valid_day_cols.keys()),
            var_name="fecha_col",
            value_name="conteo",
        )

        # Mapear nombre de columna a fecha string
        df["fecha"] = df["fecha_col"].map(valid_day_cols)
        df = df.drop(columns=["fecha_col"])

        df["conteo"] = pd.to_numeric(df["conteo"], errors="coerce")
        df = df.dropna(subset=["conteo"])
        df = df[df["conteo"] >= 0]

        # Normalizar intervalo: acepta HH:MM y HH:MM:SS
        df["intervalo"] = df["intervalo"].str.strip().str[:5]  # tomar solo HH:MM
        df["datetime"]  = pd.to_datetime(df["fecha"] + " " + df["intervalo"],
                                         format="%Y-%m-%d %H:%M")
        df = df.drop(columns=["fecha","intervalo"])

        # Agregar accesos por estación
        df = (df.groupby(["linea_id","station_id","datetime"], as_index=False)["conteo"]
                .sum().rename(columns={"conteo": tipo}))

        df["linea_id"]   = df["linea_id"].astype("string")
        df["station_id"] = df["station_id"].astype("string")
        df[tipo]         = df[tipo].astype("int32")

        out = OUTPUT_DIR / parquet_name(meta)
        df.to_parquet(out, index=False)
        return True, f"{len(df):,} filas | {df['station_id'].nunique()} estaciones"

    except Exception as e:
        return False, str(e)

# ── Verificación ───────────────────────────────────────────────────────────────

def verify(path: Path, tipo: str) -> tuple[bool, str]:
    """
    Verifica que el Parquet sea legible y tenga datos válidos.
    Toma una muestra aleatoria en horario de operación (06:00-22:00)
    y verifica que no haya NaN, negativos o valores incoherentes.
    """
    try:
        df = pd.read_parquet(path)

        if df.empty:
            return False, "archivo vacío"

        required = {"linea_id","station_id","datetime", tipo}
        missing  = required - set(df.columns)
        if missing:
            return False, f"columnas faltantes: {missing}"

        # Filtrar horario operativo 06:00-22:00
        df_op = df[df["datetime"].dt.hour.between(6, 22)]
        if df_op.empty:
            return False, "sin datos en horario operativo"

        # Muestra aleatoria de una línea al azar
        linea = random.choice(df_op["linea_id"].unique())
        muestra = df_op[df_op["linea_id"] == linea]

        nans = muestra[tipo].isna().sum()
        negs = (muestra[tipo] < 0).sum()
        if nans > 0: return False, f"{nans} NaN en muestra"
        if negs > 0: return False, f"{negs} valores negativos en muestra"

        return True, f"OK ({len(df):,} filas, linea muestra: {linea})"

    except Exception as e:
        return False, str(e)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    all_files = []
    for fp in sorted(DATA_DIR.glob("*.xlsx")):
        meta = parse_filename(fp.name)
        if meta:
            all_files.append((fp, meta))
        else:
            print(f"[IGNORADO] {fp.name}")

    total    = len(all_files)
    converted, verified_ok, skipped, errors = 0, 0, 0, []
    last_pct = -1

    print(f"\n{total} archivos encontrados\n")

    for i, (fp, meta) in enumerate(all_files, 1):
        dest = OUTPUT_DIR / parquet_name(meta)
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
    print(f"  Convertidos:  {converted}")
    print(f"  Omitidos (ya existían y verificados): {skipped}")
    print(f"  Errores:      {len(errors)}")
    if errors:
        print(f"\n  Archivos con error:")
        for name, msg in errors:
            print(f"    {name}: {msg}")
    print(f"\n  Tiempo total: {elapsed}")

if __name__ == "__main__":
    main()
