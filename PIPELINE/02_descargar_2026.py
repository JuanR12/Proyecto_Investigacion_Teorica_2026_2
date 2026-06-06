"""
update_2026.py  —  Descarga, convierte y verifica los archivos de 2026.
Uso:    python src/update_2026.py
Deps:   pip install requests pandas openpyxl pyarrow
"""

import re, time, random, requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from urllib.parse import quote

PROYECTO_RAIZ = Path(__file__).resolve().parent.parent
DATA_DIR    = PROYECTO_RAIZ / "datos"
PARQUET_DIR = PROYECTO_RAIZ / "outputs" / "parquet"
DATA_DIR.mkdir(exist_ok=True)
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

BASE_E = "https://storage.googleapis.com/validaciones_tmsa/ValidacionTroncal/2026"
BASE_S = "https://storage.googleapis.com/validaciones_tmsa/Salidas/2026"

MESES = {
    1:("Enero","Ene",31), 2:("Febrero","Feb",28), 3:("Marzo","Mar",31),
    4:("Abril","Abr",30), 5:("Mayo","May",31),    6:("Junio","Jun",30),
    7:("Julio","Jul",31), 8:("Agosto","Ago",31),  9:("Septiembre","Sep",30),
    10:("Octubre","Oct",31), 11:("Noviembre","Nov",30), 12:("Diciembre","Dic",31),
}

import calendar

# ── URLs ───────────────────────────────────────────────────────────────────────

def urls_entradas(mes: int) -> list[str]:
    dd      = calendar.monthrange(2026, mes)[1]
    nombre, abrev, _ = MESES[mes]
    mm      = f"{mes:02d}"
    # 2026: "al DD de Mes del 2026" y "al DD de Abrev del 2026"
    variantes = [
        f"{mm} TM Resumen de Validaciones Troncales al {dd} de {nombre} del 2026 Intervalo 15 Mint.xlsx",
        f"{mm} TM Resumen de Validaciones Troncales al {dd} de {abrev} del 2026 Intervalo 15 Mint.xlsx",
        f"{mm} TM Resumen de Validaciones Troncales al {dd} de {nombre} 2026 Intervalo 15 Mint.xlsx",
    ]
    return [f"{BASE_E}/{quote(v)}" for v in variantes]

def url_salidas(mes: int) -> str:
    nombre = MESES[mes][0]
    return f"{BASE_S}/Resumen_Salidas_Cada_15_minutos_{nombre}_2026.xlsx"

# ── Descarga ───────────────────────────────────────────────────────────────────

def descargar(urls: list[str], dest: Path) -> bool:
    if dest.exists():
        print(f"    [SKIP] {dest.name}")
        return True
    for url in urls:
        try:
            r = requests.get(url, timeout=60, stream=True)
            if r.status_code == 200:
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                print(f"    [OK]   {dest.name}  ({dest.stat().st_size//1024} KB)")
                return True
        except requests.RequestException as e:
            print(f"    [ERR]  {e}")
    return False

# ── Conversión ─────────────────────────────────────────────────────────────────

def parse_id(raw: str) -> str | None:
    m = re.match(r"\((\d+)\)", str(raw).strip())
    return m.group(1).zfill(5) if m else None

def find_header(fp: Path) -> int:
    raw = pd.read_excel(fp, header=None, nrows=15, dtype=str)
    for i, row in raw.iterrows():
        if any(str(v).strip() in ("Estación","Estacion","ESTACION") for v in row if pd.notna(v)):
            return i
    return 6

def is_new_fmt(cols: list) -> bool:
    return any(str(c).startswith("_") for c in cols
               if str(c) not in ("Linea","Estacion","Acceso_Estacion","Intervalo","Total general"))

def parse_day(col, new_fmt: bool) -> str | None:
    s = str(col).strip()
    if new_fmt:
        clean = s.lstrip("_").replace("_", "-")
        return clean if re.match(r"\d{4}-\d{2}-\d{2}", clean) else None
    try: return pd.to_datetime(s).strftime("%Y-%m-%d")
    except: return None

def convertir(fp: Path, tipo: str, año: int, mes: int) -> tuple[bool, str]:
    try:
        header = find_header(fp)
        df = pd.read_excel(fp, skiprows=header, header=0, dtype=str)
        df = df.loc[:, ~df.columns.astype(str).str.contains("Total", case=False)]

        cols = list(df.columns)
        if tipo == "entradas":
            cols[:6] = ["_vacia","fase","linea_raw","estacion_raw","acceso","intervalo"]
            df.columns = cols
            df = df[~df["fase"].str.upper().str.contains("DUAL", na=False)]
            df = df.drop(columns=["_vacia","fase","acceso"])
        else:
            new_fmt = is_new_fmt(cols)
            cols[:4] = ["linea_raw","estacion_raw","acceso","intervalo"]
            df.columns = cols
            df = df.drop(columns=["acceso"])

        df["linea_id"]   = df["linea_raw"].map(parse_id)
        df["station_id"] = df["estacion_raw"].map(parse_id)
        df = df.dropna(subset=["station_id","linea_id","intervalo"])
        df = df.drop(columns=["linea_raw","estacion_raw"])

        new_fmt  = tipo == "salidas" and is_new_fmt([c for c in df.columns if c not in ("linea_id","station_id","intervalo")])
        day_cols = {c: parse_day(c, new_fmt) for c in df.columns if c not in ("linea_id","station_id","intervalo")}
        day_cols = {k:v for k,v in day_cols.items()
                    if v and pd.Timestamp(v).year == año and pd.Timestamp(v).month == mes}

        df = df.melt(id_vars=["linea_id","station_id","intervalo"],
                     value_vars=list(day_cols.keys()),
                     var_name="fecha_col", value_name="conteo")
        df["fecha"]   = df["fecha_col"].map(day_cols)
        df["conteo"]  = pd.to_numeric(df["conteo"], errors="coerce")
        df = df.dropna(subset=["conteo"])
        df = df[df["conteo"] >= 0]
        df["intervalo"] = df["intervalo"].str.strip().str[:5]
        df["datetime"]  = pd.to_datetime(df["fecha"] + " " + df["intervalo"], format="%Y-%m-%d %H:%M")
        df = df.drop(columns=["fecha_col","fecha","intervalo"])

        df = (df.groupby(["linea_id","station_id","datetime"], as_index=False)["conteo"]
                .sum().rename(columns={"conteo": tipo}))
        df["linea_id"]   = df["linea_id"].astype("string")
        df["station_id"] = df["station_id"].astype("string")
        df[tipo]         = df[tipo].astype("int32")

        out = PARQUET_DIR / f"2026-{mes:02d}-{tipo}.parquet"
        df.to_parquet(out, index=False)
        return True, f"{len(df):,} filas | {df['station_id'].nunique()} estaciones"
    except Exception as e:
        return False, str(e)

# ── Verificación ───────────────────────────────────────────────────────────────

def verificar(fp_xlsx: Path, fp_pq: Path, tipo: str, mes: int) -> tuple[bool, str]:
    """
    Compara el total mensual del xlsx contra el parquet
    y verifica una muestra aleatoria en horario operativo.
    """
    try:
        df_pq = pd.read_parquet(fp_pq)

        # Total parquet
        total_pq = int(df_pq[tipo].sum())

        # Total xlsx: re-leer solo columnas de días del mes
        header = find_header(fp_xlsx)
        df_xl  = pd.read_excel(fp_xlsx, skiprows=header, header=0, dtype=str)
        df_xl  = df_xl.loc[:, ~df_xl.columns.astype(str).str.contains("Total", case=False)]
        cols   = list(df_xl.columns)

        if tipo == "entradas":
            cols[:6] = ["_vacia","fase","linea_raw","estacion_raw","acceso","intervalo"]
            df_xl.columns = cols
            df_xl = df_xl[~df_xl["fase"].str.upper().str.contains("DUAL", na=False)]
            df_xl = df_xl.drop(columns=["_vacia","fase","acceso","linea_raw","estacion_raw","intervalo"], errors="ignore")
        else:
            cols[:4] = ["linea_raw","estacion_raw","acceso","intervalo"]
            df_xl.columns = cols
            df_xl = df_xl.drop(columns=["acceso","linea_raw","estacion_raw","intervalo"], errors="ignore")

        new_fmt  = is_new_fmt(list(df_xl.columns))
        day_cols = {c: parse_day(c, new_fmt) for c in df_xl.columns}
        day_cols = {k:v for k,v in day_cols.items()
                    if v and pd.Timestamp(v).month == mes}
        total_xl = pd.to_numeric(df_xl[list(day_cols.keys())].stack(), errors="coerce").sum()
        total_xl = int(total_xl)

        diff_pct = abs(total_pq - total_xl) / total_xl * 100 if total_xl else 0
        if diff_pct > 1:
            return False, f"Diferencia de totales: xlsx={total_xl:,}  parquet={total_pq:,}  ({diff_pct:.2f}%)"

        # Muestra aleatoria en horario operativo
        df_op = df_pq[df_pq["datetime"].dt.hour.between(6, 22)]
        if df_op.empty:
            return False, "Sin datos en horario operativo"

        linea  = random.choice(df_op["linea_id"].unique())
        muestra = df_op[df_op["linea_id"] == linea]
        nans   = muestra[tipo].isna().sum()
        negs   = (muestra[tipo] < 0).sum()
        if nans > 0: return False, f"{nans} NaN en muestra"
        if negs > 0: return False, f"{negs} negativos en muestra"

        return True, f"OK | xlsx={total_xl:,}  parquet={total_pq:,}  diff={diff_pct:.3f}%  linea_muestra={linea}"

    except Exception as e:
        return False, str(e)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Detectar qué meses están disponibles en el servidor
    # Por ahora: enero-abril 2026 confirmados
    MESES_2026 = [1, 2, 3, 4]

    fallos = []
    print(f"\n── Descarga 2026 ────────────────────────────────────────\n")

    for mes in MESES_2026:
        nombre = MESES[mes][0]
        print(f"  {nombre} 2026")

        dest_e = DATA_DIR / f"{nombre.upper()}26.xlsx"
        dest_s = DATA_DIR / f"{nombre.upper()}26S.xlsx"

        ok_e = descargar(urls_entradas(mes), dest_e)
        ok_s = descargar([url_salidas(mes)], dest_s)

        if not ok_e: fallos.append(f"entradas {nombre}")
        if not ok_s: fallos.append(f"salidas  {nombre}")

        time.sleep(0.3)

    print(f"\n── Conversión y verificación ────────────────────────────\n")

    for mes in MESES_2026:
        nombre = MESES[mes][0]
        for tipo, dest in [("entradas", DATA_DIR / f"{nombre.upper()}26.xlsx"),
                           ("salidas",  DATA_DIR / f"{nombre.upper()}26S.xlsx")]:
            pq = PARQUET_DIR / f"2026-{mes:02d}-{tipo}.parquet"
            if not dest.exists():
                print(f"  [SKIP] {dest.name} no descargado")
                continue

            print(f"  Convirtiendo {dest.name}...", end=" ", flush=True)
            ok, msg = convertir(dest, tipo, 2026, mes)
            if not ok:
                print(f"ERROR: {msg}")
                fallos.append(f"conversión {dest.name}: {msg}")
                continue
            print(msg)

            print(f"  Verificando...", end=" ", flush=True)
            ok, msg = verificar(dest, pq, tipo, mes)
            print(("✓ " if ok else "✗ ") + msg)
            if not ok:
                fallos.append(f"verificación {dest.name}: {msg}")

    print(f"\n── Resumen ──────────────────────────────────────────────")
    if fallos:
        print(f"  Problemas ({len(fallos)}):")
        for f in fallos: print(f"    - {f}")
    else:
        print("  Todo correcto.")

if __name__ == "__main__":
    main()