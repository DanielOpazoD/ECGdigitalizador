# Requisitos y trazabilidad (IEC 62304, borrador)

Requisitos del software derivados de `docs/mission.md` (principios y
objetivos) y del análisis de riesgos (`analisis_riesgos.md`), cada uno con su
implementación y la prueba automática o el banco que lo verifica.
`tests/test_traceability.py` comprueba que cada prueba citada aquí y en el
análisis de riesgos existe, y que cada riesgo con controles tiene al menos una
prueba.

## Proceso (resumen)

| Actividad IEC 62304 | Cómo se hace hoy |
|---|---|
| Plan de desarrollo | `docs/mission.md` (objetivos y prioridades); un PR por cambio |
| Requisitos | esta tabla |
| Arquitectura | `docs/architecture.md`, `docs/pipeline.md`, `docs/api.md` |
| Verificación de unidades e integración | `pytest` en CI en cada PR (`.github/workflows/`), más ruff y mypy |
| Verificación del sistema | bancos de `benchmarks/` documentados en `docs/evaluation.md` |
| Gestión de configuración | git; dependencias fijadas (`requirements.lock`, `configs/engines/*-requirements.lock`); pesos con sha256 |
| SOUP (software de terceros) | motores Ahus y ECG-Digitiser en `external/` a commit fijo (`docs/engines.md`); PyTorch CPU y demás en los lock |
| Gestión de problemas | issues y PR de GitHub; fallos documentados en `docs/evaluation.md` |
| Liberación | pendiente: no hay versiones liberadas para uso clínico |

## Requisitos

| ID | Requisito | Implementación | Verificación | Riesgos |
|---|---|---|---|---|
| REQ-01 | Sin velocidad declarada o medida no se produce eje temporal ni mediciones en ms | `geometry.py`, `signal.py`, `cli.py` | `test_fixtures.py::test_T57_time_unknown`, `test_cli.py::test_cli_time_unknown_export`, `test_signal.py::test_raw_path_validation` | R01 |
| REQ-02 | Sin ganancia la señal queda en píxeles, nunca en mV | `signal.py`, `export.py`, `contracts.py` | `test_fixtures.py::test_T56_gain_unknown`, `test_signal.py::test_px_units_when_gain_none`, `test_contracts.py::test_signal_path_with_gain_unknown_fails` | R02, R17 |
| REQ-03 | Conversión px → ms / mV exacta para escalas conocidas | `geometry.py` | `test_geometry.py::test_T01_dt_25mm_s_10px_mm`, `test_geometry.py::test_T03_rise_100px_is_1mV`, `test_geometry.py::test_T13_speed_doubled_times_halved`, `test_geometry.py::test_T14_gain_doubled_amplitudes_halved` | R01, R02 |
| REQ-04 | La escala sale de evidencia (rejilla, pulso) o de confirmación humana con autor; los conflictos se marcan | `calibration.py`, `grid.py` | `test_calibration.py::test_conflicting_evidence_review`, `test_calibration.py::test_manual_without_author_not_privileged`, `test_grid.py::test_single_period_is_ambiguous_not_guessed` | R01 |
| REQ-05 | Los huecos no se rellenan en ninguna salida | `signal.py`, `render.py`, `export.py` | `test_signal.py::test_gap_no_interpolation`, `test_render.py::test_T36_gap_not_bridged_in_png`, `test_export.py::test_T37_wfdb_gap_nans_preserved` | R04 |
| REQ-06 | Cada corrida lleva un informe de calidad calculado sin la verdad | `qc.py` | `test_qc.py::test_consistent_page_is_good`, `test_qc.py::test_flat_and_missing_leads_are_insufficient`, `test_qc.py::test_inverted_limb_lead_is_inconsistent` | R03, R05 |
| REQ-07 | Intervalos con estado `ok`/`doubtful`/`unavailable`; `insufficient` degrada `ok` | `intervals.py` | `test_intervals.py::test_intervals_of_known_beats`, `test_intervals.py::test_few_leads_are_doubtful`, `test_intervals.py::test_insufficient_quality_demotes_ok`, `test_intervals.py::test_report_never_raises` | R06 |
| REQ-08 | Contraste con los valores impresos con tolerancias fijadas de antemano | `intervals.py`, `concordance.py` | `test_intervals.py::test_compare_with_printed_values`, `test_concordance.py::test_concordance_end_to_end_and_resume` | R01, R06 |
| REQ-09 | Alerta de ritmo irregular, nunca como diagnóstico | `rhythm.py` | `test_rhythm.py::test_regular_strip`, `test_rhythm.py::test_irregular_without_p_is_flagged_as_consistent_with_af` | R07 |
| REQ-10 | Amplitudes R/S y Sokolow-Lyon con la convención del 12SL; ST no informado | `intervals.py` | `test_intervals.py::test_r_s_amplitudes_follow_the_12sl_convention`, `test_intervals.py::test_sokolow_lyon` | R08, R09 |
| REQ-11 | Sólo el último run seleccionado de la revisión activa puede publicarse | `store.py` | `test_store.py::test_run_complete_and_publish`, `test_api.py::test_retry_only_last_selected_publishes`, `test_api.py::test_t40_patch_during_run_keeps_old_result` | R10 |
| REQ-12 | Recuperación tras caída sin publicar resultados parciales; un proceso por almacén | `worker.py`, `store.py` | `test_recovery.py::test_running_job_is_interrupted_not_retried`, `test_recovery.py::test_invalid_result_is_not_published_on_recovery`, `test_recovery.py::test_process_lock_is_exclusive` | R11 |
| REQ-13 | Límites de entrada antes de decodificar | `ingest.py` | `test_ingest.py::test_t43_pixel_limit_before_decode`, `test_ingest.py::test_admit_file_size_limit`, `test_ingest.py::test_admit_pdf_page_limit` | R14 |
| REQ-14 | Sólo `127.0.0.1` por defecto; fuera de él, token obligatorio | `api.py`, `cli.py` | `test_security.py::test_access_token`, `test_security.py::test_serving_beyond_this_computer_needs_a_token` | R13 |
| REQ-15 | Borrado sin rastro de datos del paciente y registro de auditoría | `store.py` | `test_security.py::test_delete_leaves_no_trace_and_is_audited`, `test_store.py::test_delete_marks_state`, `test_api.py::test_t50_delete_during_run` | R13 |
| REQ-16 | Pesos de los motores verificados; motores rotos informados | `digitizers/`, `doctor.py` | `test_digitizers.py::test_verify_weights`, `test_doctor.py::test_broken_engine_is_reported` | R15 |
| REQ-17 | Formatos 3×4+1R y 6×2 con perfil explícito | `process.py`, `qc.py` | `test_process.py::test_process_6x2_layout`, `test_qc.py::test_6x2_layout_leads_are_5s_and_rr_uses_them` | R16 |
| REQ-18 | Exportación con unidades explícitas y manifiesto estricto | `export.py`, `contracts.py` | `test_export.py::test_T37_wfdb_roundtrip`, `test_export.py::test_csv_roundtrip_and_T38`, `test_contracts.py::test_extra_fields_rejected` | R17 |
| REQ-19 | Nombres de archivo de usuario nunca usados como rutas | `ingest.py`, `api.py` | `test_ingest.py::test_t44_safe_study_id_and_filename_escape`, `test_api.py::test_artifact_filename_safe_is_basename` | R12, R13 |

## Verificación del sistema (bancos, no pruebas automáticas)

| Objetivo | Banco | Resultado resumido |
|---|---|---|
| Fidelidad de la señal (O1, O2) | F6-a, F6-b | `docs/evaluation.md` |
| Eje temporal por evidencia (O3) | F7 | ídem |
| QC que separe (O4) | F7 pasos 2 y 7 | ídem |
| Intervalos y eje (O7) | F9–F12 | ídem |
| Ritmo, amplitudes | F14, F15 | ídem |
| Reproducibilidad | A3 | ídem |
| Protocolo con el equipo propio | F13 (ensayo sin fotos) | **pendiente con fotos reales** |
