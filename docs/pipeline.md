# Pipeline de digitalización: dos corridas (F4-a)

> **Atajo (objetivo O6 de `docs/mission.md`):** `ecg-photo process foto.jpg --out DIR --speed 25
> --gain 10 --author ... --reason ...` ejecuta en un directorio nuevo `ingest` → `digitize` →
> `confirm-scale` → `qc` → `export` y escribe `DIR/summary.json` y `DIR/overview.png` (las 12
> derivaciones redibujadas en formato 3×4 + II a 25 mm/s y 10 mm/mV para compararlas a ojo con
> la hoja). Eje temporal `--time-source auto` (defecto): primero la evidencia de la imagen; si se
> rechaza (sin geometría o sin rejilla medible), el supuesto del motor, y el motivo del rechazo
> queda en `summary.json` (`evidence_axis_refused`). `--time-source evidence` no admite ese
> retroceso. Velocidad y ganancia son obligatorias: son lo impreso en la hoja, confirmado por
> quien ejecuta. Un intento rechazado no deja corridas a medias. El detalle de cada paso sigue
> abajo.

El flujo completo de una foto de ECG a señal exportable pasa por **dos corridas
separadas** sobre el mismo estudio, cada una con su propio directorio de corrida
(`run-*`). Nada escala la señal del motor hasta que un humano confirma
explícitamente las escalas.

## Corrida 1: `digitize`

```bash
ecg-photo ingest foto.png --out study_dir
ecg-photo digitize study_dir --page page-1 --engine ahus --duration 10 \
    --ahus-root external/ahus --ahus-python external/.venv-ahus/bin/python \
    --ahus-config external/ahus/src/config/inference_wrapper_george-moody-2024.yml
```

`digitize` ejecuta el motor (ahus o ecg-digitiser) y escribe una corrida
inmutable en `study_dir/runs/run-*/` con, por derivación detectada:

- `traces/<seg>.npy` — traza del motor en el marco canónico del motor
  (`raw_coordinate_frame = engine-canonical:<engine_id>`), **unidades
  `arbitrary`**, NaN donde el motor no extrajo señal.
- `masks/<seg>-raw-support.npz` — soporte observado (con enmascaramiento de
  rachas de ceros exactos ≥ 50 muestras cuando el motor codifica los huecos
  como 0.0, p. ej. ecg-digitiser).
- Manifiesto con `speed_status=gain_status=unknown`, `units=None`,
  `signal_path=None`, `lead_status=proposed`, `time_relation=simultaneous` y un
  `sync_group_id` compartido por página.
- `run_report.json` con `leads_written`, `skipped_leads` (sin soporte →
  `NO_SUPPORT`), wall time y layout detectado.

## ¿Por qué `unknown`?

F2 mostró que la fs/duración canónicas de los motores son **supuestos del
motor**, no evidencia: ahus re-muestrea a una rejilla fija
(`target_num_samples`); ecg-digitiser emite 500 Hz × 10 s y rellena los huecos
con ceros exactos. Sus escalas presentan errores medidos de ~0.8–1.6 % sobre la
verdad sintética (rr_est 1.008–1.016 s frente a 1.000 s; lag best 58–130 ms).
Por eso la primera corrida declara todo desconocido y `export` se niega con
`TIME_SCALE_UNKNOWN`/`GAIN_UNKNOWN`.

## Corrida 2: `confirm-scale`

```bash
ecg-photo confirm-scale study_dir/runs/run-XXX \
    --speed 25 --gain 10 --author <quien> --reason "<motivo>" [--fs 500]
```

Crea una **corrida nueva** (la anterior queda intacta) donde:

- Los valores del motor se toman **como mV tal cual los emitió** el motor;
  `units="mV"`, `signal_path` escrito con NaN en los huecos.
- `speed_mm_s`/`gain_mm_mV` quedan con estado `manual` y dos
  `CalibrationEvidence` con autor y razón, `previous_value=None`, método
  "manual confirmation of engine assumption".
- `observed_duration_s` = duración asumida del motor; rejilla vía
  `grid_from_observed_duration`; re-muestreo por bloques contiguos soportados
  solo si `--fs` difiere de la fs asumida del motor.
- `effective_dt_ms/effective_dv_mV = None` con `resolution_evidence`
  `method="none"`.

### Qué significa (y qué NO) la confirmación manual

Significa: "asumo que el motor corrió sobre una imagen a 25 mm/s, 10 mm/mV" —
una afirmación de escala, no una medición. **No** valida que el papel fuera
realmente 25 mm/s, no corrige el sesgo temporal del motor (~1 %), ni aporta
evidencia geométrica. La evidencia real vendría de `px_per_mm` + pulso de
calibración (F3) o geometría.

## Limitaciones registradas

- La traza del motor en corrida 1 NO es evidencia temporal (`arbitrary`).
- Los ceros exactos enmascarados se marcan `support=False`, no se interpolan.
- `confirm-scale` no modifica `state.json` ni `selected_run_id`.
- `export` acepta el directorio de corrida como si fuera una revisión.

## Smoke test real (imgkit seed7, ahus, 200 dpi)

Ejecutado sobre `runs/f2/imgkit_seed7/tiled12-0.png` (verdad: 12 derivaciones,
1 mV, RR 1.000 s):

- `ingest` → estudio con estimación de rejilla 7.874 px/mm tras el
  refinamiento por líneas de F4-b (verdad 7.874).
- `digitize --engine ahus --duration 10` → 12 derivaciones, sin saltadas,
  `layout=3x4+1R`, 3m31s en CPU; export de la corrida 1 se niega con
  `TIME_SCALE_UNKNOWN`/`GAIN_UNKNOWN` como se espera.
- `confirm-scale --speed 25 --gain 10` → corrida nueva; `export` produce
  CSV/PNG/PDF/WFDB por derivación (WFDB multilead con NaN preservados).
- Relectura WFDB `seg-II-page-1` vs verdad (`metrics_for_lead` de
  `benchmarks/f2_synthetic_bench.py`): coverage 0.998, best_lag 39 ms,
  rr_est 1.008 s (verdad 1.000), r_peak_amp_ratio 0.848,
  amplitude_ratio 0.98.

### Reconciliación B-07 (r@lag vs la ventana de comparación)

La discrepancia con el r@lag=0.954 del bench F2 se resolvió: la verdad de
`metrics_for_lead` en F2 era el fixture de 3 s, así que r@lag solo cubría
t ≤ 3 s. Comparando la exportación WFDB de 10 s contra el registro
periódico de 10 s: r@lag = **0.957 (ventana 3 s), 0.882 (5 s), 0.654
(10 s)**. La caída es el error de escala temporal acumulado del motor
(~0.8 % → ~80 ms de deriva a 10 s), no un defecto del pipeline — confirma
que la duración debe derivarse de evidencia propia, no de la rejilla del
motor. F4-c derivará la duración desde evidencia de imagen (px/mm de
rejilla × velocidad confirmada × extensión medida de la traza).
**Seguimiento:** no usar r@lag como gate hasta fijar la ventana en el bench
(pendiente; el bench no se modifica en F4-b).

## F4-c: eje temporal desde evidencia de imagen

Los parches `ahus-0001`/`ecg-digitiser-0002` hacen que cada motor escriba
`*_geometry.json`, lo que permite mapear índice de muestra canónica → píxel de
página por derivación (`LeadGeometry`, `raw_coordinate_frame="file"`,
`raw_paths/seg-*-geometry.json` con la cadena afín/homografía). Con
`confirm-scale --time-source evidence` (defecto) el eje temporal se deriva de
`t = (x_px − x_first)/(px_per_mm_x · speed)` usando nuestra rejilla refinada y
la velocidad resuelta — nunca la rejilla del motor. Exige geometría presente,
`px_per_mm_x` (grid json) y velocidad (evidencia o `--speed`); si falta algo
→ `ValueError` con `TIME_SCALE_UNKNOWN`/`CALIBRATION_MISSING`, sin fallback
silencioso. `--time-source engine` conserva el comportamiento anterior para
comparar.

Resultado sobre imgkit seed7 con ahus (sintético, una sola imagen, no es
validación; geometría guardada en
`benchmarks/results/f4c_ahus_seed7_geometry.json`):

| time-source | r@lag 10 s | best_lag | rr_est | observed_duration_s | coverage |
|---|---|---|---|---|---|
| evidence | **0.998** | 90 ms | 1.001 s | 9.913 | 1.000 |
| engine | 0.654 | 39 ms | 1.008 s | 10.000 (asumida) | 0.998 |

El eje temporal por evidencia elimina casi toda la deriva de escala temporal
del motor (~0.8 %): la duración medida (9.913 s desde la extensión de la traza
en píxeles) queda a <1 % de la verdad en rr_est.
