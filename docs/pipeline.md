# Pipeline de digitalización: dos corridas (F4-a)

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

- `ingest` → estudio con estimación de rejilla 7.936 px/mm (verdad 7.874).
- `digitize --engine ahus --duration 10` → 12 derivaciones, sin saltadas,
  `layout=3x4+1R`, 3m31s en CPU; export de la corrida 1 se niega con
  `TIME_SCALE_UNKNOWN`/`GAIN_UNKNOWN` como se espera.
- `confirm-scale --speed 25 --gain 10` → corrida nueva; `export` produce
  CSV/PNG/PDF/WFDB por derivación (WFDB multilead con NaN preservados).
- Relectura WFDB `seg-II-page-1` vs verdad (`metrics_for_lead` de
  `benchmarks/f2_synthetic_bench.py`): coverage 0.998, best_lag 39 ms,
  r@lag 0.654, rr_est 1.008 s (verdad 1.000), r_peak_amp_ratio 0.848,
  amplitude_ratio 0.98. RR, amplitud y cobertura coinciden con el bench
  F2; `r@lag` no (F2 informó 0.954 para la misma semilla). La señal
  exportada y el CSV canónico guardado en `runs/f2/ahus_seed7` difieren
  como máximo 0.07 mV y ambos dan r@lag 0.654 con el mismo procedimiento,
  así que la discrepancia está en la comparación (o en una corrida distinta
  del motor), no en el pipeline. Pendiente de reconciliar (B-07) antes de
  usar r@lag como métrica de gate.
