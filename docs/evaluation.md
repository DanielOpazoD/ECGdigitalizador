# Evaluación F6-a: PTB-XL → ECG-Image-Kit → pipeline

Alcance: **señales 12 derivaciones reales (PTB-XL), imágenes sintéticas
(ECG-Image-Kit) — NO fotos reales, NO validación clínica.**

## Datos

- **PTB-XL 1.0.3** (PhysioNet, `records500/00000`, 500 Hz, WFDB): registros
  `00001_hr`–`00010_hr`. Licencia **CC BY 4.0**. Cita obligatoria:
  Wagner P. et al., "PTB-XL, a large publicly available electrocardiography
  dataset 1.0.3", PhysioNet, doi:10.13026/kfzx-aw45.
  Los sha256 de cada fichero `.hea`/`.dat`/`LICENSE` están en
  `benchmarks/results/f6_ptbxl_two_engines_<fecha>.json` (bloque `ptbxl.files`).
- **Imágenes**: generadas con `gen_ecg_image_from_data.py` (`--lead_bbox
  --store_config`, semilla = nº de registro) — render ECG-Image-Kit estándar,
  25 mm/s y 10 mm/mV (la calibración de rejilla de `ecg_plot` es fija; el json
  de `store_config` no expone claves `paper_speed`/ganancia, queda anotado en
  el resultado). Formato 3x4 + tira de ritmo completa (II por defecto, leída
  del json de imgkit por registro, no hardcodeada).

## Protocolo

Por cada registro y motor: PNG → `ingest` → `estimate_page_grids` →
`digitize_page(engine_duration_s=10)` → `confirm_scale(gain=10, speed=25,
time_source="evidence")`. Métricas por derivación calculadas **sólo sobre
muestras observadas** (`metrics_common.segment_metrics`): duración observada
vs ventana esperada (10 s tira de ritmo, 2.5 s segmentos), r@lag (Pearson en
lag óptimo sobre el registro completo), mejor lag, error de offset vs la
casilla {0,2.5,5,7.5} s, RMSE en mV, razón de amplitud p95–p5, cobertura.

Comando exacto:

```bash
python benchmarks/f6_ptbxl_bench.py \
  --ahus-root external/ahus --ahus-python external/.venv-ahus/bin/python \
  --imgkit-root external/ecg-image-kit/codes/ecg-image-generator \
  --imgkit-python external/.venv-imgkit/bin/python \
  --base-config external/ahus/src/config/inference_wrapper_george-moody-2024.yml \
  --digitiser-root external/ecg-digitiser \
  --digitiser-python external/.venv-digitiser/bin/python \
  --digitiser-model models/M3 \
  --engines ahus ecg-digitiser --work runs/f6 --records 1 2 3 4 5 6 7 8 9 10
```

## Agregados (10 registros x 12 derivaciones; medianas)

| motor | n_ok/n | mediana r@lag (IQR) | mediana rmse_mV | mediana amp_ratio | mediana err duración % | mediana |offset| s | frac r≥0.9 |
|---|---|---|---|---|---|---|---|---|
| ahus | 120/120 | 0.994 (0.988–0.996) | 0.028 | 1.001 | −0.284 | 0.005 | 0.983 |
| ecg-digitiser | 120/120 | 0.988 (0.968–0.995) | 0.024 | 1.000 | −0.301 | 0.004 | 0.942 |

Tiempo de ejecución ≈ 4–5 min por imagen y motor en CPU (máquina del banco:
macOS arm64). Rejilla estimada: px/mm_x = 7.8739 en los 10 registros (imgkit
39.37 px/5 mm → 7.874 exacto).

### Por derivación (mediana r@lag; n = 10 por derivación)

| derivación | ahus | ecg-digitiser |
|---|---|---|
| I | 0.993 | 0.986 |
| II (ritmo) | 0.991 | 0.953 |
| III | 0.990 | 0.977 |
| aVR | 0.994 | 0.995 |
| aVL | 0.987 | 0.989 |
| aVF | 0.996 | 0.993 |
| V1 | 0.997 | 0.995 |
| V2 | 0.996 | 0.994 |
| V3 | 0.994 | 0.985 |
| V4 | 0.990 | 0.981 |
| V5 | 0.991 | 0.974 |
| V6 | 0.995 | 0.988 |

## Fallos observados

Ningún fallo de motor ni de registro (`failures` vacío en el JSON). Las 120
derivaciones por motor quedaron `ok`; no hubo `no_signal` ni `engine_error`.

## Nota sobre el mapeo de geometría de ECG-Digitiser

La corrida de humo (registro 00001) expuso un bug propio: el motor remuestrea
las columnas de cada máscara a su rejilla canónica de 500 Hz con su
`sec_per_pixel` asumido, de modo que la muestra i ↔ columna
`x1 + (i/fs)/sec_per_pixel` — no `x1+i` como asumía el parche F4-c. Corregido
en `parse_digitiser_geometry` (la verificación previa a la corrección daba
r@lag ≈ −0.01 en la tira de ritmo frente a 0.96 del motor crudo; tras la
corrección, mediana r@lag 0.986 en el humo).

## Limitaciones

- Imágenes sintéticas limpias; no fotos/escaneos reales.
- Sin huecos ni ruido impuesto: las métricas miden el camino feliz.
- `x_px` de cada motor se evalúa en una fila de referencia (centro de
  máscara/alineado); residuo pequeño con rotación.
- La ventana esperada 2.5 s/10 s proviene del json de imgkit, no del motor.

## Necesidades F6-b

- Repetir con fotos/escaneos reales (p. ej. PTB-Image, ECG-Image-Database si
  el acceso lo permite) y con degradación controlada.
- Métricas de medición clínica (FC/RR/intervalos) sobre LUDB u ondas
  delineadas.
- Cubrir el caso `no_signal` con más detalle (por qué el motor no produjo
  traza), y aVR/aVL/aVF en otros layouts.
