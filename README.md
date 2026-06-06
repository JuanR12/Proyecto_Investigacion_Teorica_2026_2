# Proyecto de Investigación Teórica 2026 — XGBoost TransMilenio

Predicción de flujos de entradas y salidas del sistema troncal TransMilenio (Bogotá)
usando XGBoost con features de calendario, codificación cíclica y rezagos temporales.

---

## Estructura del proyecto

```
Proyecto_Investigacion_Teorica_2026_2/
│
├── config.py                   ← Configuración de rutas (modifica aquí si es necesario)
├── requirements.txt            ← Dependencias Python
│
├── PIPELINE/                   ← Pipeline de datos (ejecutar en orden numérico)
│   ├── 01_descargar_datos.py   → Descarga xlsx históricos de Google Cloud Storage
│   ├── 02_descargar_2026.py    → Descarga datos del año en curso (2026)
│   ├── 03_construir_parquet.py → Convierte xlsx a formato parquet
│   └── 04_construir_catalogo.py→ Genera el catálogo de estaciones
│
├── CODIGO/
│   ├── LIMPIEZA DE DATOS/      ← Limpieza y agregación de registros
│   ├── LISTA DE ESTACIONES Y LINEAS/ ← Listados de referencia
│   ├── MODELO/
│   │   ├── SISTEMA/            ← Modelos a nivel de sistema completo
│   │   ├── ESTACION/           ← Modelos por estación individual
│   │   └── PRUEBA/             ← Scripts de validación del modelo
│   ├── VISUALIZACIONES/        ← Gráficas de predicciones y comparaciones
│   │   └── ESTACION/           ← Gráficas por estación
│   └── UTILIDADES/             ← Herramientas auxiliares y diagnóstico
│
├── NOTEBOOKS/
│   ├── visualizaciones.ipynb   ← Notebook centralizado con todas las gráficas
│   └── pruebas.ipynb           ← Exploración interactiva
│
├── outputs/
│   ├── parquet/                ← Datos procesados (2019, 2022–2026)
│   ├── predicciones/           ← CSVs exportados por los modelos
│   └── FIGURAS/                ← Imágenes generadas
│
└── DOCS/
    ├── Propuesta_Proyecto_G9.pdf
    ├── INFORMACION_BASES_DE_DATOS/  ← Listados CSV de referencia
    └── ARCHIVO/                ← Versiones históricas del código
```

---

## Flujo de trabajo

```
1. DESCARGA          2. CONVERSIÓN         3. LIMPIEZA
PIPELINE/            PIPELINE/             CODIGO/
01_descargar_datos   03_construir_parquet  LIMPIEZA DE DATOS/
        │                    │                    │
        ▼                    ▼                    ▼
   datos/*.xlsx      outputs/parquet/      datos agregados
                     *.parquet

4. MODELADO                     5. VISUALIZACIÓN
CODIGO/MODELO/                  NOTEBOOKS/visualizaciones.ipynb  ← recomendado
SISTEMA/  → sistema completo    — o —
ESTACION/ → por estación        CODIGO/VISUALIZACIONES/          ← scripts individuales
                                         │
                                         ▼
                                outputs/FIGURAS/
```

---

## Configuración rápida

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Descargar datos históricos (funciona desde cualquier directorio)
python PIPELINE/01_descargar_datos.py

# 3. Convertir a parquet
python PIPELINE/03_construir_parquet.py

# 4. Ejecutar modelo del sistema
python CODIGO/MODELO/SISTEMA/xgboost_transmilenio_v1.py

# 5. Generar todas las gráficas (opción recomendada)
#    Abrir NOTEBOOKS/visualizaciones.ipynb en Jupyter y ejecutar celda a celda
```

> **Nota:** Si el proyecto está en una ruta distinta, edita `config.py`.
> Todos los scripts del pipeline, modelos y visualizaciones calculan sus rutas
> automáticamente desde la ubicación del archivo — no hace falta cambiar nada.

---

## Scripts con rutas de datos externos

Los siguientes scripts trabajan con archivos de datos crudos que **no están en el
repositorio** (archivos transaccionales descargados manualmente). Antes de
ejecutarlos hay que editar la variable de ruta en la sección `CONFIGURACIÓN` al
inicio de cada archivo:

| Script | Variable a ajustar | Descripción |
|--------|--------------------|-------------|
| `CODIGO/LIMPIEZA DE DATOS/filtrar_tarjetas_repetidas_tm.py` | `CARPETA_DATOS`, `CARPETA_SALIDA` | Filtrado de tarjetas por línea/estación |
| `CODIGO/LIMPIEZA DE DATOS/resumir_entradas_salidas_estacion_tm.py` | `CARPETA_DATOS`, `CARPETA_SALIDA` | Resumen por estación |
| `CODIGO/LISTA DE ESTACIONES Y LINEAS/Listado_Estaciones_Lineas.py` | `archivo_entrada` | Extrae listas únicas de líneas y estaciones; guarda en `DOCS/INFORMACION_BASES_DE_DATOS/` |
| `CODIGO/UTILIDADES/grafica_red_ponderada.py` | `archivo_salidas` | Pondera el grafo de rutas por flujo de salidas |

> `CODIGO/UTILIDADES/grafica_red_rutas.py` lee `DOCS/INFORMACION_BASES_DE_DATOS/Servicios.csv`
> automáticamente — no requiere configuración manual.

> `CODIGO/VISUALIZACIONES/grafica_predicciones_2025.py` requiere un CSV con columnas
> `datetime`, `real` y `prediccion` (formato diferente al de salida de los modelos).

---

## Datos

| Fuente | Google Cloud Storage (`validaciones_tmsa`) |
|--------|---------------------------------------------|
| Cobertura | 2019, 2022–2026 (entradas y salidas) |
| Granularidad | Intervalos de 15 minutos, por línea y estación |
| Formato crudo | xlsx (descargado) → parquet (procesado) |
| Archivos parquet | 129 archivos en `outputs/parquet/` |

---

## Propuesta del proyecto

Ver [`DOCS/Propuesta_Proyecto_G9.pdf`](DOCS/Propuesta_Proyecto_G9.pdf).
