import pandas as pd
import re
from pathlib import Path

PARQUET_DIR = Path(r'C:\Users\gordi\Desktop\Transmilenio\Proyecto_Investigacion_Teorica_2026_2\outputs\parquet')
OUTPUT_DIR  = PARQUET_DIR.parent
OUTPUT_DIR.mkdir(exist_ok=True)

AÑOS = [2019, 2021, 2022, 2023, 2024, 2025]

def parse_id_nombre(raw: str) -> tuple[str, str]:
    m = re.match(r"\((\d+)\)\s*(.*)", str(raw).strip())
    return (m.group(1).zfill(5), m.group(2).strip()) if m else (None, raw)

def limpiar_nombre(nombre: str) -> str:
    if pd.isna(nombre): return nombre
    for sep in [' - ', ' – ', ' — ', '-']:
        if sep in nombre:
            return nombre.split(sep)[0].strip()
    return nombre.strip()

# ── 1. Extraer station_id y linea_id de un archivo por año ────────────────────
print("Leyendo estaciones por año...")
records = []

for año in AÑOS:
    candidatos = sorted(PARQUET_DIR.glob(f"{año}-*-salidas.parquet"))
    if not candidatos:
        print(f"  [WARN] Sin archivos para {año}")
        continue
    fp = next((f for f in candidatos if '-06-' in f.name), candidatos[len(candidatos)//2])
    df = pd.read_parquet(fp, columns=['linea_id','station_id']).drop_duplicates()
    df['año'] = año
    records.append(df)
    print(f"  {año}: {fp.name}  →  {df['station_id'].nunique()} estaciones")

df_all = pd.concat(records, ignore_index=True).drop_duplicates()

# ── 2. Primera y última aparición ─────────────────────────────────────────────
print("\nCalculando apariciones...")
aparicion = []
for año in AÑOS:
    archivos = sorted(PARQUET_DIR.glob(f"{año}-*-salidas.parquet"))
    if not archivos: continue
    for fp in [archivos[0], archivos[-1]]:
        mes = int(fp.name.split('-')[1])
        dt  = pd.Timestamp(year=año, month=mes, day=1)
        ids = pd.read_parquet(fp, columns=['station_id'])['station_id'].unique()
        for sid in ids:
            aparicion.append({'station_id': sid, 'fecha': dt})

df_ap   = pd.DataFrame(aparicion)
primera = df_ap.groupby('station_id')['fecha'].min().rename('primera_aparicion')
ultima  = df_ap.groupby('station_id')['fecha'].max().rename('ultima_aparicion')

# ── 3. Reconstruir nombres desde los xlsx vía parquet no es posible ───────────
# Los parquet no tienen nombres, así que los extraemos de un xlsx por año.
# Leemos solo las columnas de Línea y Estación.
print("\nExtrayendo nombres desde xlsx...")

DATA_DIR = PARQUET_DIR.parent.parent / 'datos'
MESES_INV = {
    1:"ENERO",2:"FEBRERO",3:"MARZO",4:"ABRIL",5:"MAYO",6:"JUNIO",
    7:"JULIO",8:"AGOSTO",9:"SEPTIEMBRE",10:"OCTUBRE",11:"NOVIEMBRE",12:"DICIEMBRE"
}

nombre_records = []
for año in AÑOS:
    # Usar el mismo mes central que usamos para parquet
    candidatos = sorted(PARQUET_DIR.glob(f"{año}-*-salidas.parquet"))
    if not candidatos: continue
    fp_pq = next((f for f in candidatos if '-06-' in f.name), candidatos[len(candidatos)//2])
    mes   = int(fp_pq.name.split('-')[1])
    yy    = str(año)[2:]
    xlsx  = DATA_DIR / f"{MESES_INV[mes]}{yy}S.xlsx"
    if not xlsx.exists():
        xlsx = DATA_DIR / f"{MESES_INV[mes]}{año}S.xlsx"
    if not xlsx.exists():
        print(f"  [WARN] No encontrado xlsx para {año}-{mes:02d}")
        continue

    try:
        raw = pd.read_excel(xlsx, header=None, nrows=15, dtype=str)
        header_row = 6
        for i, row in raw.iterrows():
            if any(str(v).strip() in ('Estación','Estacion','ESTACION') for v in row if pd.notna(v)):
                header_row = i
                break

        df = pd.read_excel(xlsx, skiprows=header_row, usecols=[0,1], dtype=str)
        df.columns = ['linea_raw','estacion_raw']
        df = df.dropna(subset=['estacion_raw'])
        df = df[df['estacion_raw'].str.match(r'\(\d+\)', na=False)]

        df['station_id']   = df['estacion_raw'].apply(lambda x: parse_id_nombre(x)[0])
        df['station_name'] = df['estacion_raw'].apply(lambda x: parse_id_nombre(x)[1])
        df['station_name'] = df['station_name'].map(limpiar_nombre)
        df['año']          = año

        nombre_records.append(df[['station_id','station_name','año']].dropna())
        print(f"  {año}: {xlsx.name}  →  {df['station_id'].nunique()} nombres")
    except Exception as e:
        print(f"  [ERROR] {xlsx.name}: {e}")

# ── 4. Nombre más reciente por estación ───────────────────────────────────────
df_nombres = pd.concat(nombre_records, ignore_index=True)
nombre_reciente = (df_nombres.sort_values('año', ascending=False)
                              .drop_duplicates('station_id')
                              [['station_id','station_name']])

# Historial de nombres limpios (sin patrocinio y sin duplicados)
historial = (df_nombres.drop_duplicates(['station_id','station_name'])
                        .groupby('station_id')['station_name']
                        .apply(lambda x: ' | '.join(sorted(x.unique())))
                        .reset_index()
                        .rename(columns={'station_name':'nombres_historicos'}))

# Colapsar historiales donde el único cambio era el patrocinio
historial['nombres_historicos'] = historial['nombres_historicos'].apply(
    lambda x: x if ' | ' in x else x
)

# ── 5. Líneas asociadas ────────────────────────────────────────────────────────
lineas = (df_all.groupby('station_id')['linea_id']
                .apply(lambda x: ', '.join(sorted(x.unique())))
                .rename('lineas_id').reset_index())

# ── 6. Ensamblar catálogo ──────────────────────────────────────────────────────
catalog = (df_all[['station_id']].drop_duplicates()
           .merge(nombre_reciente, on='station_id', how='left')
           .merge(lineas,          on='station_id', how='left')
           .merge(primera,         on='station_id', how='left')
           .merge(ultima,          on='station_id', how='left')
           .merge(historial,       on='station_id', how='left')
           .sort_values('station_id')
           .reset_index(drop=True))

# ── 7. Reporte ─────────────────────────────────────────────────────────────────
cambios = catalog[catalog['nombres_historicos'].str.contains(r'\|', na=False)]
print(f"\nEstaciones totales:          {len(catalog)}")
print(f"Sin nombre:                  {catalog['station_name'].isna().sum()}")
print(f"Con cambio de nombre real:   {len(cambios)}")
if not cambios.empty:
    print(cambios[['station_id','nombres_historicos']].to_string(index=False))

# ── 8. Guardar ─────────────────────────────────────────────────────────────────
catalog.to_parquet(OUTPUT_DIR / 'catalogo_estaciones.parquet', index=False)
catalog.to_csv(OUTPUT_DIR / 'catalogo_estaciones.csv', index=False, encoding='utf-8-sig')
print(f"\nGuardado en {OUTPUT_DIR}")