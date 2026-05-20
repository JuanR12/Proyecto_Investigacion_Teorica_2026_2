"""
download_data.py  —  Descarga archivos de entradas y salidas de Transmilenio.
Uso:    python src/download_data.py
Salida: archivos xlsx en datos/ con formato MES{YY}.xlsx y MES{YY}S.xlsx

Deps: pip install requests
"""

import calendar
import time
import requests
from pathlib import Path
from urllib.parse import quote

DATA_DIR = Path("datos")
DATA_DIR.mkdir(exist_ok=True)

BASE_ENTRADAS = "https://storage.googleapis.com/validaciones_tmsa/ValidacionTroncal"
BASE_SALIDAS  = "https://storage.googleapis.com/validaciones_tmsa/Salidas"

# Años a descargar
TARGET_YEARS = [2019, 2021, 2022, 2023, 2024]

MESES_ES = {
    1:  ("Enero",      "Ene"),
    2:  ("Febrero",    "Feb"),
    3:  ("Marzo",      "Mar"),
    4:  ("Abril",      "Abr"),
    5:  ("Mayo",       "May"),
    6:  ("Junio",      "Jun"),
    7:  ("Julio",      "Jul"),
    8:  ("Agosto",     "Ago"),
    9:  ("Septiembre", "Sep"),
    10: ("Octubre",    "Oct"),
    11: ("Noviembre",  "Nov"),
    12: ("Diciembre",  "Dic"),
}


def last_day(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def build_entradas_urls(year: int, month: int) -> list[str]:
    """
    Construye variantes posibles de la URL de entradas.
    El patrón tiene una inconsistencia: algunos meses incluyen 'de' antes
    de la abreviación y otros no. Se generan ambas variantes.
    Ej: '12 TM ... al 31 de Dic 2024 ...' vs '11 TM ... al 30 Nov 2024 ...'
    """
    mm        = f"{month:02d}"
    dd        = last_day(year, month)
    mes_abrev = MESES_ES[month][1]
    base      = f"{BASE_ENTRADAS}/{year}"

    # Variante con "de": "al 31 de Dic 2024"
    nombre_con_de    = f"{mm} TM Resumen de Validaciones Troncales al {dd} de {mes_abrev} {year} Intervalo 15 Mint.xlsx"
    # Variante sin "de": "al 30 Nov 2024"
    nombre_sin_de    = f"{mm} TM Resumen de Validaciones Troncales al {dd} {mes_abrev} {year} Intervalo 15 Mint.xlsx"

    return [
        f"{base}/{quote(nombre_con_de)}",
        f"{base}/{quote(nombre_sin_de)}",
    ]


def build_salidas_urls(year: int, month: int) -> list[str]:
    """2023+: con año en nombre. 2022: algunos sin año, se intentan ambas variantes."""
    mes_nombre = MESES_ES[month][0]
    base       = f"{BASE_SALIDAS}/{year}"
    con_año    = f"{base}/Resumen_Salidas_Cada_15_minutos_{mes_nombre}_{year}.xlsx"
    sin_año    = f"{base}/Resumen_Salidas_Cada_15_minutos_{mes_nombre}.xlsx"
    return [con_año, sin_año] if year == 2022 else [con_año]


def local_name(year: int, month: int, tipo: str) -> str:
    """Nombre local: DICIEMBRE25.xlsx o DICIEMBRE25S.xlsx"""
    mes_upper = MESES_ES[month][0].upper()
    yy        = str(year)[2:]
    suffix    = "S" if tipo == "salidas" else ""
    return f"{mes_upper}{yy}{suffix}.xlsx"


def download_file(urls: list[str], dest: Path) -> bool:
    """
    Intenta descargar desde cada URL en orden.
    Retorna True si tuvo éxito, False si todas fallaron.
    """
    if dest.exists():
        print(f"    [SKIP] Ya existe: {dest.name}")
        return True

    for url in urls:
        try:
            r = requests.get(url, timeout=60, stream=True)
            if r.status_code == 200:
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                size_kb = dest.stat().st_size // 1024
                print(f"    [OK]   {dest.name}  ({size_kb} KB)")
                return True
            # 404 es esperado si el archivo no existe, no imprimir error
        except requests.RequestException as e:
            print(f"    [ERR]  {url}: {e}")

    return False


def main():
    t0       = time.time()
    total    = len(TARGET_YEARS) * 12 * 2  # entradas + salidas por mes/año
    done     = 0
    failures = []
    last_pct = -1

    print(f"Descargando {total} archivos ({len(TARGET_YEARS)} años × 12 meses × 2 tipos)\n")

    for year in TARGET_YEARS:
        print(f"\n── {year} ────────────────────────────────────────")
        for month in range(1, 13):
            mes_nombre = MESES_ES[month][0]

            # Entradas
            dest_e = DATA_DIR / local_name(year, month, "entradas")
            urls_e = build_entradas_urls(year, month)
            ok_e   = download_file(urls_e, dest_e)
            if not ok_e:
                failures.append({"tipo": "entradas", "año": year, "mes": mes_nombre,
                                  "archivo": dest_e.name, "urls": urls_e})
                print(f"    [FAIL] {dest_e.name}")

            # Salidas
            dest_s = DATA_DIR / local_name(year, month, "salidas")
            urls_s = build_salidas_urls(year, month)
            ok_s   = download_file(urls_s, dest_s)
            if not ok_s:
                failures.append({"tipo": "salidas", "año": year, "mes": mes_nombre,
                                  "archivo": dest_s.name, "urls": urls_s})
                print(f"    [FAIL] {dest_s.name}")

            done += 2
            pct  = int(done / total * 100)
            if pct // 10 > last_pct // 10:
                from datetime import timedelta
                elapsed = timedelta(seconds=int(time.time() - t0))
                print(f"\n  {pct:3d}%  —  {done}/{total}  —  {elapsed} transcurrido\n")
                last_pct = pct

            time.sleep(0.3)  # pausa leve para no saturar el servidor

    # ── Reporte final ──────────────────────────────────────────────────────────
    from datetime import timedelta
    elapsed = timedelta(seconds=int(time.time() - t0))
    descargados = total - len(failures)

    print(f"\n{'─'*55}")
    print(f"Descargados: {descargados}/{total}  |  Tiempo: {elapsed}")

    if failures:
        print(f"\nNo encontrados ({len(failures)}) — descargar manualmente:")
        print(f"{'Archivo':<30s}  {'Tipo':<10s}  URL intentada")
        print("─" * 90)
        for f in failures:
            print(f"  {f['archivo']:<28s}  {f['tipo']:<10s}  {f['urls'][0]}")
    else:
        print("Todos los archivos descargados correctamente.")


if __name__ == "__main__":
    main()
