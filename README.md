# ECG Foto → Señal

Digitalización de electrocardiogramas fotografiados o en PDF hacia señales con escala temporal y
de voltaje explícitas, con máscaras, trazabilidad y mediciones sólo cuando exista evidencia.

**Misión, objetivos medibles y estado: `docs/mission.md`.** Estado (2026-09-26): dos motores
integrados (Ahus, ECG-Digitiser), eje temporal por evidencia de imagen, control de calidad sin
verdad y banco con imágenes reales (Kaggle) y fotos de un GE MAC2000 (`docs/evaluation.md`).
No hay validación clínica ni diagnósticos habilitados. Inventario del entorno, decisiones y
bloqueos: `docs/environment.md`.

Uso rápido (una foto → señal exportada + informe PDF de una página + `overview.png`):
```bash
bash benchmarks/setup_engines.sh          # motores y configs/engines.local.yml
ecg-photo process foto.jpg --out salida --speed 25 --gain 10 \
    --author NOMBRE --reason "valores impresos en la hoja" [--printed-rr-ms 742] \
    [--printed-pr-ms 160 --printed-qrs-ms 92 --printed-qt-ms 380 --printed-qtc-ms 418] \
    [--printed-axis-deg 45]
```
Los valores `--printed-*` son los de la cabecera que imprime el electrocardiógrafo: el programa
compara con ellos su RR y sus intervalos y marca las discrepancias (F11 en `docs/evaluation.md`).

## Instalación

**En un paso** (Linux / macOS; necesita `python3.12`, `git`, `curl` y
[`uv`](https://docs.astral.sh/uv/); descarga ~2.5 GB la primera vez):

```bash
./install.sh                  # programa + motor Ahus (CPU)
./install.sh --with-digitiser # además el motor secundario ECG-Digitiser
```

Crea `.venv` con las versiones de `requirements.lock`, instala los motores en `external/` con
sus versiones fijadas (`configs/engines/*-requirements.lock`, torch sólo CPU; otra compilación
con `TORCH_INDEX_URL`), verifica el sha256 de los pesos y termina con `ecg-photo doctor`, que
comprueba en cualquier momento que la instalación puede digitalizar. Es idempotente.

Sólo el programa, sin motores (desarrollo, pruebas):

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -c requirements.lock -e .[dev]
```
`requirements.lock` fija las versiones exactas con las que se probaron el código y los
bancos (también las usa CI). Para actualizarlas: `uv pip compile pyproject.toml --extra dev
--python-version 3.12 --no-header --no-annotate -o requirements.lock`, instalar, y volver a
pasar las pruebas y los bancos afectados antes de fusionar.

## Verificación
```bash
ruff check src tests && ruff format --check src tests && mypy src && pytest -q
```

## Documentación
- `docs/protocolo_validacion.md` — protocolo para validar con fotos del propio electrocardiógrafo
  (`ecg-photo concordance`: tabla por foto y Bland-Altman frente a lo impreso).
- `docs/guia_uso.md` — **guía de uso para el médico**: tomar la foto, procesarla, leer calidad
  y mediciones, comparar con lo impreso, límites y privacidad.
- `docs/mission.md` — misión, principios, objetivos medibles y su estado.
- `docs/architecture.md` — capas, flujo de datos, dónde vive cada cosa, cómo extender.
- `docs/pipeline.md` — las dos corridas (`digitize`, `confirm-scale`) y `ecg-photo process`.
- `docs/geometry.md` — marcos de coordenadas, transformaciones, calibración.
- `docs/api.md` — API local, worker, lotes, interfaz de revisión.
- `docs/engines.md` — motores integrados y su evaluación inicial.
- `docs/evaluation.md` — todos los bancos, con resumen e índice al inicio.
- `docs/environment.md` — inventario del entorno, decisiones y bloqueos.

## Estructura
- `src/ecg_photo/contracts.py` — contrato canónico v1 (pydantic, `extra="forbid"`), validación
  semántica del directorio de revisión, JSON estricto (`null`, nunca `NaN`).
- `src/ecg_photo/ingest.py` — admisión por bytes mágicos y límites, EXIF, raster o PDF,
  ampliación de fotos pequeñas, rejilla de la página.
- `src/ecg_photo/pipeline.py` — `digitize_page` (motor → corrida sin escala) y
  `confirm_scale` (corrida nueva en mV y s, eje temporal por evidencia o supuesto del motor).
- `src/ecg_photo/digitizers/` — adaptadores de motores (Ahus, ECG-Digitiser) tras un protocolo común.
- `src/ecg_photo/grid.py`, `aligned.py`, `calibration.py`, `transforms.py`, `geometry.py`,
  `signal.py` — rejilla, marco corregido de perspectiva, resolución de escalas, cadenas de
  transformaciones, px ↔ s/mV, rejilla temporal y remuestreo sin puentear huecos.
- `src/ecg_photo/qc.py` — control de calidad sin verdad; `intervals.py` — PR/QRS/QT/QTc medidos
  en la señal; `measure.py` — aritmética de RR/FC/QTc.
- `src/ecg_photo/export.py`, `render.py` — CSV/WFDB/JSON y PNG/PDF.
- `src/ecg_photo/process.py` — cadena completa en un comando y `overview.png`; `report.py` —
  informe PDF de una página.
- `src/ecg_photo/store.py` (+ `store_models.py`), `worker.py`, `api.py` (+ `api_results.py`,
  `api_common.py`), `batch.py`, `ui/` — estudios y revisiones, worker único, API local, lotes
  reanudables, interfaz de revisión.
- `src/ecg_photo/paths.py` — ubicación de `configs/` (independiente del directorio actual).
- `configs/` — configuración, entradas admitidas, inventario de motores/pesos, formatos de Ahus.
- `benchmarks/` — bancos reproducibles; `benchmarks/results/` — sus métricas.
- `external/` (ignorado) — motores fijados por commit; `patches/` — parches mínimos aplicados.

## Comandos
```bash
ecg-photo process foto.jpg --out salida --speed 25 --gain 10 --author NOMBRE --reason "..." \
    [--engine ahus|ecg-digitiser] [--time-source auto|evidence|engine] \
    [--printed-rr-ms 742 | --printed-hr 81]     # todo en un comando (ver docs/pipeline.md)
ecg-photo fixture --kind calibrated --out rev
ecg-photo validate rev
ecg-photo export rev --out exp
ecg-photo run-demo --out demo
ecg-photo ingest foto.png --out estudio [--render-dpi 300]
ecg-photo calibrate estudio --page page-1 --speed 25 --gain 10 --author NOMBRE --reason "..."
ecg-photo digitize estudio --page page-1 --engine ahus --duration 10 \
    --ahus-root ... --ahus-python ... --ahus-config ... [--runs-root DIR]
ecg-photo digitize estudio --page page-1 --engine ecg-digitiser --duration 10 \
    --digitiser-root ... --digitiser-python ... [--digitiser-model models/M3]
ecg-photo confirm-scale estudio/runs/run-XXX --gain 10 \
    --author NOMBRE --reason "..." [--speed 25] [--fs 500] \
    [--time-source evidence|engine]
ecg-photo export estudio/runs/run-YYY --out export_dir
ecg-photo serve --store STORE_DIR [--port 8000]   # API local 127.0.0.1 + worker único
# luego abrir http://127.0.0.1:8000/ui para la revisión local
ecg-photo batch DIR --store STORE_DIR --report lote.json --engine ahus --duration 10 \
    [--gain 10 --speed 25 --author NOMBRE --reason "..."] [--resume]
```
Un solo proceso (`serve` o `batch`) por almacén; al arrancar, `serve` recupera los
trabajos que quedaron a medias (ver `docs/api.md`, «Reinicio y lotes»).

Benchmarks: `docs/evaluation.md` (F6-a: PTB-XL real -> imágenes sintéticas -> pipeline; señal real, imagen sintética — no fotos, no validación clínica).

## Reglas no negociables
Sin escala temporal no hay señal temporal; sin ganancia no hay mV; los huecos no se rellenan;
`representative_beat` no sirve para FC/RR; ningún valor por defecto de velocidad/ganancia.
