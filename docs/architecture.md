# Arquitectura

Mapa del código para quien lo mantiene: capas, flujo de datos, dónde se
guarda cada cosa y cómo se extiende. Qué persigue el programa y cómo se
prioriza: `docs/mission.md`. Detalle de cada etapa: `docs/pipeline.md`,
`docs/geometry.md`, `docs/api.md`, `docs/engines.md`; resultados medidos:
`docs/evaluation.md`.

## Capas (dependencias sólo hacia abajo, sin ciclos)

```
Entrada     cli.py  ·  api.py (FastAPI) + ui/index.html  ·  batch.py
                │            │                                │
Orquestación    process.py (un comando)   worker.py (cola, 1 trabajo)   store.py (estudios, revisiones, corridas)
                │                               │
Dominio     ingest.py → pipeline.py (digitize_page, confirm_scale) → qc.py, intervals.py → export.py / render.py
            grid.py · aligned.py · calibration.py · transforms.py · signal.py · geometry.py · measure.py
Motores     digitizers/base.py (Protocol Digitizer, EngineOutput) · ahus.py · ecg_digitiser.py
Contrato    contracts.py (pydantic, extra="forbid")        paths.py (ubicación de configs/)
```

- `contracts.py` no depende de nada del paquete; todo lo demás depende de él.
- Los motores son subprocesos externos (`external/`, no versionado) detrás de
  `Digitizer.run(image, work_dir) -> EngineOutput`; el resto del código no
  conoce sus formatos.
- `qc.py` sólo lee una corrida confirmada (manifiesto + `.npy`): se puede
  ejecutar sobre corridas antiguas sin re-ejecutar motores.

## Flujo de datos

```
foto/PDF ──ingest──► revisión (manifest.json, pages/*.png, *.grid.json, transformación archivo→página)
         ──digitize_page(motor)──► corrida 1: trazas del motor, unidades arbitrarias,
                                   velocidad/ganancia "unknown", geometría por derivación
         ──confirm_scale──► corrida 2 (nueva, la 1 queda intacta): señal en mV y s,
                            eje temporal por evidencia de imagen o supuesto del motor,
                            CalibrationEvidence y ProcessingStep por derivación
         ──run_qc──► qc.json (good / acceptable / insufficient, sin verdad)
         ──intervals──► intervals.json (PR, QRS, QT, QTc; ok / doubtful / unavailable)
         ──export / render──► CSV, WFDB, PNG/PDF por derivación, overview.png
```

Principios que el código hace cumplir (no sólo documenta):

- Sin escala no hay señal: `export` se niega con `TIME_SCALE_UNKNOWN` /
  `GAIN_UNKNOWN`; ningún valor por defecto de velocidad o ganancia.
- Nada se sobrescribe: cada paso crea una corrida nueva con su procedencia;
  una confirmación rechazada no deja corridas a medias.
- Los huecos quedan como no observados (máscaras), nunca interpolados.
- Cada escala lleva su evidencia (`CalibrationEvidence`) y cada
  transformación su paso (`TransformChain`), validados al escribir
  (`validate_revision_dir_report`).

## Dónde vive cada cosa

| Qué | Dónde |
|---|---|
| Configuración versionada | `configs/` (resuelta desde el árbol de código por `ecg_photo.paths`; `ECG_PHOTO_CONFIG_DIR` la sustituye) |
| Rutas locales de motores | `configs/engines.local.yml` (no versionado; lo escribe `benchmarks/setup_engines.sh`) |
| Checkouts y pesos de motores | `external/` (no versionado; commits y sha256 en `configs/checkpoints.json`) |
| Parches mínimos a motores | `patches/` |
| Almacén de `serve` / `batch` | directorio `--store` (un proceso por almacén, con bloqueo) |
| Corridas de bancos | `runs/` (no versionado); resultados resumidos en `benchmarks/results/` |
| Imágenes reales de pacientes | nunca en el repositorio |

## Cómo extender

- **Otro motor**: una clase con `spec: EngineSpec` y
  `run(image_path, work_dir) -> EngineOutput` en `digitizers/`; si el motor
  puede informar la geometría (columna de página por muestra), rellenar
  `EngineOutput.geometry` para habilitar el eje temporal por evidencia;
  registrarlo en `worker.default_engines`; entrada en `configs/checkpoints.json`.
  Probarlo con `benchmarks/f6b_kaggle_bench.py` antes de ofrecerlo.
- **Otro formato de hoja**: añadirlo al conjunto de formatos del motor (Ahus:
  `configs/ahus_lead_layouts.yml`) y medirlo con `benchmarks/f8_layout_eval.py`
  en impresiones 3×4 y en las imágenes Kaggle reales (un formato de más puede
  empeorar los existentes, ver F8). `qc.short_lead_s` deduce la duración de
  cada derivación del nombre `FxC`.
- **Otro control de calidad**: en `qc.run_qc`, con una prueba que reproduzca
  el caso y la re-evaluación de `benchmarks/f7_qc_eval.py` (separación de
  etiquetas en Kaggle) antes de cambiar umbrales.

## Pruebas y calidad

- `ruff check`, `ruff format --check`, `mypy src`, `pytest` (CI en cada PR,
  `.github/workflows/ci.yml`). Las pruebas no necesitan motores ni red: usan
  fixtures sintéticos (`fixtures.py`) y motores falsos.
- Cada corrección de un fallo observado lleva una prueba que lo reproduce.
- Los bancos (`benchmarks/`) miden sobre datos reales (PTB-XL, Kaggle) y
  guardan sólo métricas y sha256 en `benchmarks/results/`; se ejecutan a mano
  (CPU, minutos a horas), no en CI.

## Deuda técnica conocida

- `api.create_app` (≈340 líneas) y `cli.build_parser` concentran todas las
  rutas / subcomandos en una función cada uno.
- `store.py` (≈870 líneas) mezcla estudios, revisiones, corridas, bloqueo y
  recuperación tras reinicio.
- Dependencias de ejecución sin versión fijada salvo FastAPI/uvicorn/pdfium;
  no hay archivo de bloqueo.
- Instalar como wheel no incluye `configs/` (hay que fijar
  `ECG_PHOTO_CONFIG_DIR`).
