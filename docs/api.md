# API local y coordinador (F5-a)

Una aplicación Python: `ecg-photo serve` levanta la API FastAPI en `127.0.0.1`
(`public_network_binding: false`; `--allow-non-loopback` desactivado por defecto, F9)
y un único hilo worker con cola acotada (`execution.max_pending_jobs` de
`configs/default.yml`). El worker carga los motores una vez; la API nunca ejecuta
inferencia en su ciclo.

## Almacén (`src/ecg_photo/store.py`)

`STORE/<study_id>/`:
- `state.json` — `{study_id, active_revision, config_hash, selected_run_id,
  published_run_id, deleted, revisions}`; siempre tmp+`os.replace`.
- `rev<N>/` — revisión inmutable (rev1 = salida de ingest); contiene `config.json`
  (`RunConfig` congelado: motor, `engine_spec` commit+sha256 de pesos, duración,
  `time_source`, gain/speed/fs, `layout_profile`, `page_id`) y `calibration.json`.
- `runs/<run_id>/job.json` + `runs/<run_id>/result/` — salida aislada por run.
- `artifacts.json` — artefactos de exportación registrados con sha256 y MIME.

`config_hash = sha256(json canónico sort_keys del RunConfig)`. `create_run` fija
`selected_run_id`: sólo el último run seleccionado de una revisión puede publicar.
`publish` verifica en sección crítica (lock por estudio): existe, no borrado, sin
cancelación, `active_revision == input_revision`, `config_hash` igual,
`selected_run_id == run_id` y `validate_revision_dir(result)` sin problemas;
si no, `completed_unpublished` (resultado conservado, nunca promovido).
`delete` marca `deleted` y borra revs/runs/artefactos; una salida tardía del
worker se descarta y nunca recrea el estudio.

## Rutas

| Ruta | Comportamiento |
|---|---|
| `POST /studies` | multipart `file` (+ `options` JSON validado como StudyPatch). Streaming con corte en `max_file_mib` → 413; UNSUPPORTED_TYPE → 415; PIXEL_LIMIT/PAGE_LIMIT → 413. Nombre original sólo metadato escapado (T44). → 201 `{study_id, revision, config_hash}` |
| `GET /studies/{id}` | estado + páginas + segmentos del run publicado + `published_result_stale` |
| `POST /studies/{id}/runs` | `{expected_revision, config_hash}` → 202 `{run_id}`; 409 conflicto; 422 engine None; 503 cola llena (sin `Retry-After`: no hay estimación razonable) |
| `GET /runs/{id}` | job.json sin rutas absolutas |
| `POST /runs/{id}/cancel` | solicitud de cancelación → 202 |
| `PATCH /studies/{id}` | `{expected_revision, changes, author, reason}` → nueva revisión (copia rev, amplía `calibration.json` con evidencia manual para speed/gain, nuevo config+hash, `selected_run_id=None`); 409; 422 claves desconocidas |
| `POST /studies/{id}/exports` | `{revision, run_id, formats[csv,json,png,pdf,wfdb]}` → 201 artefactos; 409 si el run no es de esa revisión o no está completed |
| `GET /studies/{id}/artifacts/{artifact_id}` | FileResponse con `filename_safe` y MIME; 404 si no registrado |
| `DELETE /studies/{id}` | 204 |
| `GET /studies/{id}/pages/{page_id}/raster` | PNG del ráster de la revisión activa, `Cache-Control: no-store`; 404 si la página no existe |
| `GET /studies/{id}/pages/{page_id}/grid` | contenido de `page-N.grid.json`; 404 si no hay estimación |
| `GET /studies/{id}/runs` | lista de `job.json` del estudio (sin rutas), orden por `created_at` |
| `GET /studies/{id}/runs/{run_id}/segments` | resumen de segmentos del result del run (has_geometry, source_region, statuses); 404 si el run no pertenece o no tiene result |
| `GET /studies/{id}/runs/{run_id}/segments/{segment_id}/trace` | traza JSON: `x_px` sólo si `frame=="file"`, `values` con `null` en huecos (nunca 0), `t_s` sólo si hay `signal_path`; si n>20000 devuelve cada k-ésima muestra con `"decimation": k` (sólo presentación, no altera la señal) |
| `GET /health` | `{"status":"ok"}` únicamente |

`PATCH` acepta además `lead_labels: {segment_id: etiqueta}` con etiquetas
`I,II,III,aVR,aVL,aVF,V1..V6,unknown` (otras → 422). Crea la revisión nueva y
guarda/mergea `corrections.json` con `previous_label` del run publicado; el
worker aplica esas correcciones al manifest del run (`lead_status=confirmed`,
evidencia "manual correction … engine proposed …") antes de `confirm_scale`.
`GET /studies/{id}` incluye `corrections` de la revisión activa.

IDs con `^[a-z0-9-]{1,64}$` → 404 si no. Errores `{"detail": {"code", "message"}}`.

## Motores del worker

`configs/engines.local.yml` (NO versionado; rutas a `external/`):

```yaml
ahus:
  root: external/ahus
  python: external/.venv-ahus/bin/python
  config: external/ahus/src/config/inference_wrapper_george-moody-2024.yml
ecg_digitiser:
  root: external/ecg-digitiser
  python: external/.venv-digitiser/bin/python
  model_dir: models/M3
```

El motor `fake` sólo existe con `ECG_PHOTO_ENABLE_FAKE_ENGINE=1` (tests).

## Lo que NO hace todavía

- Sin autenticación multiusuario (T46, pendiente F9).
- Cancelación best-effort: se marca `cancel_requested` y se comprueba antes de
  empezar y antes de publicar; no interrumpe el subproceso del motor.
- Sin interfaz de usuario.
- Multi-página: `page_id` fijable en config vía PATCH pero por defecto `page-1`.

## Interfaz de revisión

`ecg-photo serve` levanta también una UI estática mínima en `GET /ui`
(`GET /` redirige con 307; `Cache-Control: no-store`). Muestra el original de la
página con overlay de la región fuente de cada segmento, la reconstrucción de la
traza con los huecos como `null` (nunca 0), y permite: corregir etiquetas de
derivación vía `PATCH` con `expected_revision`, lanzar/cancelar runs, y exportar
sólo el run publicado. `GET /studies/{id}` devuelve además `config`
(RunConfig congelado de la revisión activa) y `corrections`.
