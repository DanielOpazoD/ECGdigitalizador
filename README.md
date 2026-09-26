# ECG Foto → Señal

Digitalización de electrocardiogramas fotografiados o en PDF hacia señales con escala temporal y
de voltaje explícitas, con máscaras, trazabilidad y mediciones sólo cuando exista evidencia.

**Misión, objetivos medibles y estado: `docs/mission.md`.** Estado (2026-09-26): dos motores
integrados (Ahus, ECG-Digitiser), eje temporal por evidencia de imagen, control de calidad sin
verdad y banco con imágenes reales (Kaggle) y fotos de un GE MAC2000 (`docs/evaluation.md`).
No hay validación clínica ni diagnósticos habilitados. Inventario del entorno, decisiones y
bloqueos: `docs/environment.md`.

Uso rápido (una foto → señal exportada + informe de calidad + `overview.png`):
```bash
bash benchmarks/setup_engines.sh          # motores y configs/engines.local.yml
ecg-photo process foto.jpg --out salida --speed 25 --gain 10 \
    --author NOMBRE --reason "valores impresos en la hoja" [--printed-rr-ms 742]
```

## Instalación
```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e .[dev]
```

## Verificación
```bash
ruff check src tests && ruff format --check src tests && mypy src && pytest -q
```

## Estructura
- `src/ecg_photo/contracts.py` — contrato canónico v1 (pydantic, `extra="forbid"`), validación
  semántica del directorio de revisión, JSON estricto (`null`, nunca `NaN`).
- `src/ecg_photo/geometry.py` — px ↔ s/mV; toda escala debe ser positiva y finita.
- `src/ecg_photo/signal.py` — rejilla semiabierta `[0, n/fs)`, remuestreo sin puentear huecos.
- `src/ecg_photo/measure.py` — RR/FC/PR/QRS/QT/QTc con nombres sufijados por unidad.
- `src/ecg_photo/fixtures.py` — revisiones sintéticas deterministas (calibrada, hueco, ganancia
  desconocida, tiempo desconocido, cola 2.503 s).
- `src/ecg_photo/transforms.py` — cadenas ordenadas de transformaciones (EXIF, crop,
  rotate90, affine, homografía) aplicables e invertibles punto a punto.
- `src/ecg_photo/ingest.py` — admisión por bytes mágicos y límites (`supported_inputs.yml`),
  EXIF, raster embebido o render PDF, `study_id` seguro.
- `src/ecg_photo/grid.py` — estimador determinista del paso de rejilla por autocorrelación.
- `src/ecg_photo/calibration.py` — resolución de escala por evidencias (manual > concordancia).
- `src/ecg_photo/process.py` — cadena completa en un comando (`ecg-photo process`).
- `src/ecg_photo/qc.py` — control de calidad sin verdad; `aligned.py` — rejilla en el marco
  corregido de perspectiva del motor.
- `src/ecg_photo/store.py`, `worker.py`, `api.py` — revisiones, worker único, API local,
  recuperación tras reinicio; `batch.py` — lotes con informe reanudable.
- `configs/` — configuración propuesta, entradas admitidas, inventario de motores/pesos,
  perfiles de formato (`configs/profiles/`).
- `external/` (ignorado) — checkouts de motores candidatos fijados por commit.
- `patches/` — parches mínimos aplicados a `external/` (ver `patches/README.md`).

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
