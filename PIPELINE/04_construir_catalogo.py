"""
build_catalog.py  —  Construye el catálogo de estaciones a partir de los parquets.
1. Extrae todos los station_id presentes en los parquets por año/mes.
2. Busca el nombre de cada estación en el xlsx correspondiente.
3. Detecta estaciones nuevas o eliminadas entre períodos.

Años:   2019 (control) | 2022-2026 (análisis)
Salida: outputs/catalogo_estaciones.parquet + .csv

Uso:    python src/build_catalog.py
Deps:   pip install pandas openpyxl pyarrow
"""

import re, time
import pandas as pd
from pathlib import Path
from datetime import timedelta

PROYECTO_RAIZ = Path(__file__).resolve().parent.parent
DATA_DIR    = PROYECTO_RAIZ / "datos"
PARQUET_DIR = PROYECTO_RAIZ / "outputs" / "parquet"
OUTPUT_DIR  = PROYECTO_RAIZ / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_YEARS = {2019, 2022, 2023, 2024, 2025, 2026}

MESES_INV = {
    1:"ENERO",2:"FEBRERO",3:"MARZO",4:"ABRIL",5:"MAYO",6:"JUNIO",
    7:"JULIO",8:"AGOSTO",9:"SEPTIEMBRE",10:"OCTUBRE",11:"NOVIEMBRE",12:"DICIEMBRE",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_id_nombre(raw: str) -> tuple:
    m = re.match(r"\((\d+)\)\s*(.*)", str(raw).strip())
    if m:
        return m.group(1).zfill(5), m.group(2).strip()
    return None, None

def limpiar_nombre(nombre: str) -> str:
    if pd.isna(nombre): return nombre
    for sep in [" - ", " – ", " — "]:
        if sep in nombre:
            return nombre.split(sep)[0].strip()
    return nombre.strip()

def find_header(fp: Path) -> int:
    preview = pd.read_excel(fp, header=None, nrows=15, dtype=str)
    for i, row in preview.iterrows():
        if any(str(v).strip() in ("Estación","Estacion","ESTACION") for v in row if pd.notna(v)):
            return i
    return 6

def xlsx_para_mes(año: int, mes: int, tipo: str) -> Path:
    yy     = str(año)[2:]
    nombre = MESES_INV[mes]
    sufijo = "S" if tipo == "salidas" else ""
    for path in [DATA_DIR / f"{nombre}{yy}{sufijo}.xlsx",
                 DATA_DIR / f"{nombre}{año}{sufijo}.xlsx"]:
        if path.exists():
            return path
    return None

def extraer_nombres_xlsx(fp: Path, tipo: str) -> dict:
    try:
        header = find_header(fp)
        df = pd.read_excel(fp, skiprows=header, usecols=[0, 1, 2], header=0, dtype=str)
        df = df.dropna(how="all")
        cols = list(df.columns)

        if tipo == "entradas":
            cols[:3] = ["_vacia","linea_raw","estacion_raw"]
            df.columns = cols
            df = df[df["linea_raw"].str.contains("Zona", case=False, na=False)]
        else:
            cols[:3] = ["linea_raw","estacion_raw","_acceso"]
            df.columns = cols
            df = df[df["linea_raw"].str.contains("Zona", case=False, na=False)]

        df = df[df["estacion_raw"].str.match(r"\(\d+\)", na=False)]

        result = {}
        for _, row in df.iterrows():
            sid, nombre = parse_id_nombre(row["estacion_raw"])
            if sid:
                result[sid] = limpiar_nombre(nombre)
        return result
    except Exception as e:
        print(f"  [ERROR] {fp.name}: {e}")
        return {}

# ── Paso 1: Recolectar station_ids de los parquets ────────────────────────────

def recolectar_ids_parquet() -> pd.DataFrame:
    records  = []
    archivos = sorted(PARQUET_DIR.glob("*.parquet"))
    total    = len(archivos)
    print(f"Leyendo {total} parquets...")

    for i, fp in enumerate(archivos, 1):
        parts = fp.stem.split("-")
        if len(parts) != 3: continue
        año, mes, tipo = int(parts[0]), int(parts[1]), parts[2]
        if año not in TARGET_YEARS: continue

        df = pd.read_parquet(fp, columns=["linea_id","station_id"]).drop_duplicates()
        df["año"]  = año
        df["mes"]  = mes
        df["tipo"] = tipo
        records.append(df)

        pct = int(i / total * 100)
        if pct % 20 == 0 or i == total:
            print(f"  {pct}%  —  {i}/{total}")

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()

# ── Paso 2: Buscar nombres en xlsx ────────────────────────────────────────────

def buscar_nombres(df_ids: pd.DataFrame) -> pd.DataFrame:
    ids_por_año    = df_ids.groupby("año")["station_id"].apply(set).to_dict()
    nombre_records = []
    meses_pref     = [6, 7, 3, 4, 8, 1, 2, 5, 9, 10, 11, 12]

    for año, ids_año in sorted(ids_por_año.items()):
        print(f"\n  {año}: {len(ids_año)} estaciones únicas")
        nombres_encontrados = {}
        lineas_encontradas  = {}

        for mes in meses_pref:
            ids_faltantes = ids_año - set(nombres_encontrados.keys())
            if not ids_faltantes:
                break

            for tipo in ["entradas", "salidas"]:
                fp = xlsx_para_mes(año, mes, tipo)
                if not fp:
                    continue

                nombres = extraer_nombres_xlsx(fp, tipo)

                # Lineas desde el parquet del mismo mes
                pq = PARQUET_DIR / f"{año}-{mes:02d}-{tipo}.parquet"
                lineas = {}
                if pq.exists():
                    df_pq = pd.read_parquet(pq, columns=["linea_id","station_id"]).drop_duplicates()
                    lineas = df_pq.set_index("station_id")["linea_id"].to_dict()

                nuevos = 0
                for sid in ids_faltantes:
                    if sid in nombres:
                        nombres_encontrados[sid] = nombres[sid]
                        if sid in lineas:
                            lineas_encontradas[sid] = lineas[sid]
                        nuevos += 1

                if nuevos:
                    print(f"    {mes:02d}/{año} ({tipo}): +{nuevos} nombres")
                if nuevos > 0:
                    break  # con entradas fue suficiente para este mes

        for sid in ids_año:
            nombre_records.append({
                "station_id":   sid,
                "station_name": nombres_encontrados.get(sid),
                "linea_id":     lineas_encontradas.get(sid),
                "año":          año,
            })

        sin_nombre = ids_año - set(nombres_encontrados.keys())
        if sin_nombre:
            print(f"    [WARN] Sin nombre: {sin_nombre}")

    return pd.DataFrame(nombre_records)

# ── Paso 3: Construir catálogo ────────────────────────────────────────────────

def build_catalog(df_ids: pd.DataFrame, df_nombres: pd.DataFrame) -> pd.DataFrame:
    nombre_reciente = (df_nombres.dropna(subset=["station_name"])
                                 .sort_values("año", ascending=False)
                                 .drop_duplicates("station_id")
                                 [["station_id","station_name"]])

    historial = (df_nombres.dropna(subset=["station_name"])
                           .drop_duplicates(["station_id","station_name"])
                           .groupby("station_id")["station_name"]
                           .apply(lambda x: " | ".join(sorted(x.unique())))
                           .reset_index()
                           .rename(columns={"station_name":"nombres_historicos"}))

    lineas = (df_nombres.dropna(subset=["linea_id"])
                        .groupby("station_id")["linea_id"]
                        .apply(lambda x: ", ".join(sorted(x.unique())))
                        .reset_index()
                        .rename(columns={"linea_id":"lineas_id"}))

    aparicion = (df_ids.assign(
                    fecha=pd.to_datetime(
                        df_ids["año"].astype(str)+"-"+df_ids["mes"].astype(str).str.zfill(2)+"-01"
                    ))
                 .groupby("station_id")["fecha"]
                 .agg(primera_aparicion="min", ultima_aparicion="max")
                 .reset_index())

    años_presencia = (df_ids.groupby("station_id")["año"]
                            .apply(lambda x: ", ".join(str(a) for a in sorted(x.unique())))
                            .reset_index()
                            .rename(columns={"año":"años_presencia"}))

    return (df_ids[["station_id"]].drop_duplicates()
            .merge(nombre_reciente, on="station_id", how="left")
            .merge(lineas,          on="station_id", how="left")
            .merge(aparicion,       on="station_id", how="left")
            .merge(historial,       on="station_id", how="left")
            .merge(años_presencia,  on="station_id", how="left")
            .sort_values("station_id")
            .reset_index(drop=True))

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"Años objetivo: {sorted(TARGET_YEARS)}\n")

    print("── Paso 1: IDs desde parquets ───────────────────────────")
    df_ids = recolectar_ids_parquet()
    print(f"  Estaciones únicas: {df_ids['station_id'].nunique()}")
    print(f"  Por año:")
    print(df_ids.groupby("año")["station_id"].nunique().to_string())

    print("\n── Paso 2: Nombres desde xlsx ───────────────────────────")
    df_nombres = buscar_nombres(df_ids)

    print("\n── Paso 3: Catálogo ──────────────────────────────────────")
    catalog = build_catalog(df_ids, df_nombres)
    print(f"  Total estaciones:   {len(catalog)}")
    print(f"  Sin nombre:         {catalog['station_name'].isna().sum()}")

    cambios = catalog[catalog["nombres_historicos"].str.contains(r"\|", na=False)]
    if not cambios.empty:
        print(f"\n  Cambios de nombre ({len(cambios)}):")
        print(cambios[["station_id","nombres_historicos"]].to_string(index=False))

    todos = ", ".join(str(a) for a in sorted(TARGET_YEARS))
    parciales = catalog[catalog["años_presencia"] != todos]
    if not parciales.empty:
        print(f"\n  Estaciones con presencia parcial en los años ({len(parciales)}):")
        print(parciales[["station_id","station_name","años_presencia"]].to_string(index=False))

    catalog.to_parquet(OUTPUT_DIR / "catalogo_estaciones.parquet", index=False)
    catalog.to_csv(OUTPUT_DIR / "catalogo_estaciones.csv", index=False, encoding="utf-8-sig")

    print(f"\nGuardado en {OUTPUT_DIR}")
    print(f"Tiempo total: {timedelta(seconds=int(time.time()-t0))}")

if __name__ == "__main__":
    main()
