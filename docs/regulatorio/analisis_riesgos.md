# Análisis de riesgos inicial (ISO 14971, borrador)

Primera identificación de peligros del uso previsto (`uso_previsto.md`), con
los controles que ya existen en el código y la prueba o banco que los
verifica. Las probabilidades son **estimaciones cualitativas del equipo de
desarrollo sin datos clínicos**; se revisarán con la validación de
`docs/protocolo_validacion.md`. Las pruebas citadas se comprueban
automáticamente (`tests/test_traceability.py`): si se renombra o borra una, el
CI falla.

## Escalas

- **Severidad**: S1 menor (molestia, repetir la foto); S2 seria (medición o
  alerta errónea que podría influir en una decisión si no se contrasta con el
  original); S3 crítica (daño grave o muerte). Por el uso previsto (el
  original siempre a la vista y revisado por un profesional), ningún peligro se
  ha estimado en S3; esa premisa es en sí un control (C0) que depende del
  usuario y es la razón principal de que la validación clínica sea necesaria.
- **Probabilidad** (de que el error llegue al usuario sin aviso): P1 remota,
  P2 ocasional, P3 frecuente.
- **Aceptable**: S1 con cualquier P; S2 sólo con P1 tras los controles.

## Peligros, controles y verificación

| ID | Peligro → daño posible | S | P inicial | Controles (código) | Verificación | P residual | ¿Aceptable? |
|---|---|---|---|---|---|---|---|
| R01 | Velocidad supuesta o errónea → FC e intervalos falsos | S2 | P3 | nunca se supone la velocidad (sin ella no hay eje temporal); eje por evidencia de la rejilla; contraste del RR con lo impreso | `test_fixtures.py::test_T57_time_unknown`, `test_cli.py::test_cli_time_unknown_export`, `test_export.py::test_csv_time_unknown_rejected`, `test_geometry.py::test_T13_speed_doubled_times_halved`, `test_concordance.py::test_cli_concordance`, `test_qc.py::test_rr_compared_with_printed_values`; bancos F6-b, F7 paso 3 | P1 | Sí |
| R02 | Ganancia supuesta o errónea → amplitudes y Sokolow-Lyon falsos | S2 | P3 | sin ganancia la señal queda en píxeles, nunca en mV | `test_fixtures.py::test_T56_gain_unknown`, `test_export.py::test_csv_px_units_for_gain_unknown`, `test_contracts.py::test_units_mV_without_signal_path_fails`, `test_geometry.py::test_T14_gain_doubled_amplitudes_halved` | P1 | Sí |
| R03 | Digitización mala presentada como buena → mediciones falsas | S2 | P2 | informe de calidad sin verdad; `insufficient` degrada las mediciones `ok` | `test_qc.py::test_flat_and_missing_leads_are_insufficient`, `test_qc.py::test_consistent_page_is_good`, `test_intervals.py::test_insufficient_quality_demotes_ok`; banco F7 paso 2 y 7 | P2 | **No aún**: en Kaggle ninguna inservible como `good`, pero sin fotos propias suficientes (F13: 30 % `insufficient`). Pendiente de validación |
| R04 | Huecos rellenados → señal inventada | S2 | P2 | lo no observado queda como NaN / sin trazar; nunca se interpola a través de un hueco | `test_signal.py::test_gap_no_interpolation`, `test_render.py::test_T36_gap_not_bridged_in_png`, `test_export.py::test_T37_wfdb_gap_nans_preserved`, `test_fixtures.py::test_gap_fixture_masks` | P1 | Sí |
| R05 | Derivaciones intercambiadas o invertidas → eje o morfología falsos | S2 | P2 | consistencia de Einthoven en el QC; corrección manual de etiquetas con autor | `test_qc.py::test_inverted_limb_lead_is_inconsistent`, `test_api.py::test_lead_label_corrections` | P1 | Sí |
| R06 | Intervalo erróneo presentado como `ok` | S2 | P2 | estado `ok`/`doubtful`/`unavailable`; sin onda P no hay PR; pocas derivaciones → `doubtful`; contraste con lo impreso | `test_intervals.py::test_no_p_wave_means_no_pr`, `test_intervals.py::test_few_leads_are_doubtful`, `test_intervals.py::test_compare_with_printed_values`, `test_concordance.py::test_ok_subset_uses_the_status_before_the_printed_comparison`; bancos F9–F11, F13 | P2 | **No aún**: PR con DE mayor que la tolerancia CSE (F9) y QTc −16 ms en el ensayo F13. Pendiente de validación |
| R07 | Alerta de ritmo: FA no detectada (falsa tranquilidad) o falsa alarma | S2 | P2 | texto «no es un diagnóstico; mire el trazado»; extrasístole aislada no dispara | `test_rhythm.py::test_irregular_without_p_is_flagged_as_consistent_with_af`, `test_rhythm.py::test_isolated_premature_beat_stays_regular`, `test_rhythm.py::test_too_short_is_unavailable_and_measure_reports_it`; banco F14 (FA 95 %, 6 % falsas alarmas) | P2 | **No aún**: 5 % de FA sin aviso en PTB-XL; sin datos propios |
| R08 | Sokolow-Lyon falso → sospecha de HVI errónea | S1 | P2 | convención del 12SL; sólo con ganancia conocida | `test_intervals.py::test_sokolow_lyon`, `test_intervals.py::test_r_s_amplitudes_follow_the_12sl_convention`; banco F15 (97 % de acuerdo en ≥ 3.5 mV) | P2 | Sí (S1) |
| R09 | Segmento ST informado con error → isquemia mal valorada | S2 | P3 | **no se informa** el ST (F15: DE 0.098 mV > 0.05) | criterio fijado de antemano en `benchmarks/f15_amplitudes_12sl.py` | — | Sí (eliminado) |
| R10 | Resultado de otra configuración o revisión publicado → datos de otro análisis | S2 | P2 | publicación con verificación de revisión, `config_hash` y run seleccionado bajo lock | `test_api.py::test_retry_only_last_selected_publishes`, `test_api.py::test_t40_patch_during_run_keeps_old_result`, `test_store.py::test_run_complete_and_publish` | P1 | Sí |
| R11 | Caída a mitad de proceso → resultado parcial usado | S2 | P2 | recuperación al arrancar: lo interrumpido se marca `failed`, nunca se publica un resultado inválido | `test_recovery.py::test_running_job_is_interrupted_not_retried`, `test_recovery.py::test_invalid_result_is_not_published_on_recovery`, `test_recovery.py::test_process_lock_is_exclusive` | P1 | Sí |
| R12 | Estudio de otro paciente (confusión) | S2 | P2 | identificadores seguros y nombre original conservado; el programa **no** gestiona la identidad del paciente | `test_ingest.py::test_t44_safe_study_id_and_filename_escape`, `test_api.py::test_artifact_filename_safe_is_basename` | P2 | **Depende del usuario** (C0); fuera del alcance técnico actual |
| R13 | Filtración de imágenes o datos del paciente | S2 | P2 | sólo `127.0.0.1` por defecto; fuera de él exige token; borrado sin rastro y auditado; nada sale a internet | `test_security.py::test_access_token`, `test_security.py::test_serving_beyond_this_computer_needs_a_token`, `test_security.py::test_delete_leaves_no_trace_and_is_audited` | P1 | Sí |
| R14 | Archivo malicioso o enorme → caída o bloqueo | S1 | P2 | límites de tamaño, páginas y píxeles antes de decodificar | `test_ingest.py::test_t43_pixel_limit_before_decode`, `test_ingest.py::test_admit_file_size_limit`, `test_ingest.py::test_admit_pdf_page_limit` | P1 | Sí |
| R15 | Pesos del motor alterados o motor roto → salida distinta de la evaluada | S2 | P1 | sha256 de pesos verificado; `doctor` informa motores rotos; ejecución determinista (A3) | `test_digitizers.py::test_verify_weights`, `test_doctor.py::test_broken_engine_is_reported`; A3 | P1 | Sí |
| R16 | Formato no evaluado procesado como si lo fuera | S2 | P2 | 3×4+1R y 6×2 con perfil explícito | `test_process.py::test_process_6x2_layout`, `test_qc.py::test_6x2_layout_leads_are_5s_and_rr_uses_them` | P2 | **No aún**: otros formatos no se detectan ni rechazan de forma fiable (O8) |
| R17 | Unidades mal interpretadas en el archivo exportado | S2 | P2 | unidades explícitas en CSV/WFDB; manifiesto estricto | `test_contracts.py::test_signal_path_with_gain_unknown_fails`, `test_export.py::test_csv_roundtrip_and_T38`, `test_export.py::test_T37_wfdb_roundtrip` | P1 | Sí |

## Riesgos no aceptables aún

R03, R06, R07, R12 y R16. Todos dependen de datos que hoy no existen en el
repositorio (fotos propias con los valores impresos) o de funciones fuera del
alcance actual. Por eso el estado es «herramienta de investigación»: el
beneficio-riesgo global no puede declararse aceptable antes de ejecutar
`docs/protocolo_validacion.md`.

## Pendiente

- Revisión del análisis por alguien distinto de quien escribió el código.
- Estudio de usabilidad (IEC 62366-1): errores de uso al declarar velocidad y
  ganancia, lectura del estado `doubtful`.
- Vigilancia post-uso: cómo recoger y analizar los casos en que la medición
  no coincidió con el original.
