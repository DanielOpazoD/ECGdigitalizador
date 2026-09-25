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

## Reinicio y lotes (F5-c, T47–T49)

**Un proceso por almacén.** `serve` y `batch` toman un `flock` exclusivo sobre
`STORE/.lock` (se libera al salir o si el proceso muere). Un segundo proceso
sobre el mismo almacén termina con código 2 y `STORE_LOCKED`; así
`max_active_jobs = 1` se cumple también entre procesos.

**Recuperación al arrancar** (`Store.recover`, `worker.resume_after_restart`),
antes de atender peticiones; el informe se imprime como JSON (`{"recovery": …}`):

| Estado encontrado | Acción |
|---|---|
| `running` | `failed`, stage `interrupted`, error `INTERRUPTED: … during <stage>`; `work/` y `result/` parciales borrados. **No se reintenta** (el motor puede ser la causa de la caída); el usuario relanza. |
| `queued`, run seleccionado de la revisión activa con el mismo `config_hash` | re-encolado en orden de `created_at`; si la cola no alcanza → `failed` `QUEUE_FULL` |
| `queued` no vigente (otro run seleccionado, o revisión/hash cambiados) | `cancelled`, stage `superseded` (en marcha normal acabaría `completed_unpublished`) |
| `queued` con `cancel_requested` | `cancelled` |
| `completed` (parada entre fin y publicación) con `result/` | pasa por `publish()` con todas sus comprobaciones → publicado o `completed_unpublished` |
| `completed` sin `result/` | `failed`, stage `interrupted` |
| estudio `deleted` con restos | borrado completado; nunca se recrea |
| carpeta de estudio sin `state.json` (subida cortada antes del 201) | se borra si sólo contiene `rev1`; si contiene otra cosa se conserva y se informa (`orphans_kept`) |
| `upload-*`, `*.json.tmp` | borrados |

**Lotes** (`ecg-photo batch DIR --store S --report R.json --engine E …`):
cada archivo regular no oculto de `DIR` (orden por nombre) → un estudio con las
opciones como `rev1` (speed/gain son evidencia manual: exigen `--author` y
`--reason`, sin valores por defecto) → un run procesado en el mismo proceso con
el mismo código del worker (`Worker.process`) → publicación con las mismas
comprobaciones. Un archivo que falla no detiene el lote: `rejected` (con
`reason_code` de ingest, p. ej. `UNSUPPORTED_TYPE`), `failed` (error del motor
o del pipeline), `error` (otro fallo), `published` o `completed_unpublished`.
El informe (`schema: ecg-photo-batch/1`, `options_hash`, `entries`, `summary`)
se reescribe de forma atómica tras cada paso; un archivo en curso queda como
`processing`. `--resume` continúa el mismo informe: exige las mismas opciones,
salta los archivos con el mismo sha256 ya en estado final (`published`,
`completed_unpublished`, `rejected`, `cancelled`) y reintenta el resto sobre el
estudio ya creado (y sobre su run si seguía en cola) en lugar de duplicarlo
(`attempts` se incrementa). Sin `--resume` un informe existente no se
sobrescribe. Código de salida 0 sólo si todo quedó `published`, 1 si no.
Los estudios de un lote son estudios normales del almacén: `serve` los muestra
en la UI y admiten correcciones y nuevos runs.

## Lo que NO hace todavía

- Sin autenticación multiusuario (T46, pendiente F9).
- Cancelación best-effort: se marca `cancel_requested` y se comprueba antes de
  empezar y antes de publicar; no interrumpe el subproceso del motor.
- Lotes sólo por CLI (no hay endpoint de lotes); secuenciales, un motor y unas
  opciones por lote. Un run interrumpido no se reintenta solo.
- Interfaz de revisión mínima (`/ui`); sin rotación coordinada ni multiusuario.
- Multi-página: `page_id` fijable en config vía PATCH pero por defecto `page-1`.

## Interfaz de revisión

`ecg-photo serve` levanta también una UI estática mínima en `GET /ui`
(`GET /` redirige con 307; `Cache-Control: no-store`). Muestra el original de la
página con overlay de la región fuente de cada segmento, la reconstrucción de la
traza con los huecos como `null` (nunca 0), y permite: corregir etiquetas de
derivación vía `PATCH` con `expected_revision`, lanzar/cancelar runs, y exportar
sólo el run publicado. `GET /studies/{id}` devuelve además `config`
(RunConfig congelado de la revisión activa) y `corrections`.
