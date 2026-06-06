import random
import sys
from pathlib import Path
import pandas as pd

# Importar rutas desde config.py (raíz del proyecto). Ver config.py para personalizar.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RUTA_DATOS as DATA_DIR

AÑOS = [2019, 2021, 2022, 2023, 2024, 2025, 2026]
MESES_INV = {
    1:"ENERO",2:"FEBRERO",3:"MARZO",4:"ABRIL",5:"MAYO",6:"JUNIO",
    7:"JULIO",8:"AGOSTO",9:"SEPTIEMBRE",10:"OCTUBRE",11:"NOVIEMBRE",12:"DICIEMBRE"
}

def find_header(fp):
    raw = pd.read_excel(fp, header=None, nrows=15, dtype=str)
    for i, row in raw.iterrows():
        if any(str(v).strip() in ('Estación','Estacion','ESTACION') for v in row if pd.notna(v)):
            return i
    return 6

def inspeccionar(fp, tipo):
    try:
        header = find_header(fp)
        df = pd.read_excel(fp, skiprows=header, header=0, nrows=5, dtype=str)
        df = df.loc[:, ~df.columns.astype(str).str.contains("Total", case=False)]
        cols = list(df.columns)
        print(f"  Header fila {header} | {len(cols)} columnas")
        print(f"  Primeras 6: {[str(c) for c in cols[:6]]}")
        print(f"  Últimas  3: {[str(c) for c in cols[-3:]]}")

        # Fila de muestra con dato real
        muestra = df.dropna(how='all').iloc[1] if len(df) > 1 else df.iloc[0]
        print(f"  Muestra fila: {[str(v)[:20] for v in muestra.values[:6]]}")
    except Exception as e:
        print(f"  [ERROR] {e}")

print("=" * 65)
for año in AÑOS:
    # Elegir 2 meses al azar disponibles para ese año
    candidatos_e = list(DATA_DIR.glob(f"*{str(año)[2:]}*.xlsx"))
    candidatos_e = [f for f in candidatos_e if not f.name.endswith("S.xlsx")]
    candidatos_s = list(DATA_DIR.glob(f"*{str(año)[2:]}*S.xlsx"))

    muestra_e = random.sample(candidatos_e, min(2, len(candidatos_e)))
    muestra_s = random.sample(candidatos_s, min(2, len(candidatos_s)))

    print(f"\n{'─'*65}")
    print(f"  {año}")
    print(f"{'─'*65}")

    for fp in muestra_e:
        print(f"\n  [ENTRADAS] {fp.name}")
        inspeccionar(fp, "entradas")

    for fp in muestra_s:
        print(f"\n  [SALIDAS]  {fp.name}")
        inspeccionar(fp, "salidas")

print("\n" + "=" * 65)