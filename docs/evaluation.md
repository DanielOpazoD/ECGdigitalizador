# Evaluaciones: resumen e índice

Cada fase mide sobre datos reales y guarda las métricas en
`benchmarks/results/`. Nada de esto es validación clínica. Resultado principal
de cada una (r@lag: correlación por derivación con la verdad tras alinear
±0.2 s; mediana):

| Fase | Datos | Qué se midió | Resultado principal |
|---|---|---|---|
| [F6-a](#evaluación-f6-a-ptb-xl--ecg-image-kit--pipeline) | PTB-XL real → imágenes sintéticas | dos motores, pipeline completo | línea base sintética |
| [F6-b](#evaluación-f6-b-imágenes-reales-kaggle-physionet-ecg-image-digitization) | Kaggle: impresiones, escaneos y fotos reales (10 × 9 tipos) | Ahus y ECG-Digitiser | Ahus 0.922; ECG-Digitiser falla en escaneos y fotos |
| [F7 paso 1](#f7-rejilla-ambigua-1-mm-o-5-mm-resuelta-con-el-tamaño-de-página) | Kaggle | rejilla 1 mm / 5 mm por tamaño de página | escala propia en 10/10 escaneos |
| [F7 paso 2–3](#f7-paso-2-control-de-calidad-sin-verdad-ecg_photoqc-ecg-photo-qc) | Kaggle + fotos MAC2000 | control de calidad sin verdad; RR impreso | `insufficient` 0.80 / 0.12 frente a `good` 0.93 / 0.97 |
| [F7 paso 4](#f7-paso-4-resolución-de-la-foto-y-ampliación-al-ingresar) | Kaggle fotos reducidas | resolución y ampliación al ingresar | 1000 px: 0.740 → 0.847 |
| [F7 paso 5](#f7-paso-5-perspectiva--rejilla-en-el-marco-alineado-del-motor) | Kaggle | rejilla en el marco corregido de perspectiva | eje propio 0.941 frente a 0.922 del motor |
| [F7 paso 6–7](#f7-paso-6-detector-de-qrs-del-contraste-con-lo-impreso) | fotos MAC2000 + Kaggle | detector de QRS, RR medio recortado | RR: 1/104 tiras fuera de ±5 % |
| [F8](#f8-formatos-distintos-de-34--ii-objetivo-o8-de-docsmissionmd) | PTB-XL impreso 3×4 y 6×2 + Kaggle | formatos de hoja | 6×2: 0.70 → 0.993 sin empeorar 3×4 |
| [F9](#f9-intervalos-pr-qrs-qt-objetivo-o7-de-docsmissionmd) | LUDB (anotaciones de cardiólogos) | PR, QRS, QT | hoja impresa, valores `ok`: QRS +1.6 ± 9.8 ms, QT −13.7 ± 13.2 ms (CSE sí), PR −3.6 ± 11.5 ms |
| [F10](#f10-intervalos-en-imágenes-reales-kaggle) | Kaggle escaneos y fotos reales | error que añade la digitalización a PR/QRS/QT | PR y QT \|dif.\| ≈ 5 ms; QRS +8 ms más ancho |
| [F11](#f11-coinciden-nuestros-intervalos-con-los-que-imprime-el-electrocardiógrafo) | PTB-XL + medidas GE 12SL (PTB-XL+) | acuerdo con lo que imprime un GE | QT corregido (−20 → ≈ 0 ms de sesgo); tolerancias para contrastar con la cabecera impresa |
| [F12](#f12-eje-eléctrico-del-qrs) | PTB-XL + 12SL + etiquetas de cardiólogos; Kaggle | eje del QRS | \|dif.\| mediana 5.6° frente a 12SL; categoría = cardiólogos 87 % (12SL 88 %) |

Las secciones siguen en orden cronológico; cuando una fase posterior cambia
una conclusión anterior, la posterior lo dice.

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

# Evaluación F6-b: imágenes reales (Kaggle PhysioNet ECG Image Digitization)

Alcance previsto: **impresiones, escaneos y fotos reales de señales reales**
(split `train` de la competición Kaggle `physionet-ecg-image-digitization`).
Sigue sin ser validación clínica ni datos de los equipos locales (B-01/B-03).

## Estado

**Ejecutado (2026-09-25)** en CPU x86 de 4 núcleos y 16 GB, sin GPU:

- Fase 1: 10 registros × `0001`/`0003`/`0005` × 2 motores × 2 ejes.
- Fase 2: Ahus en las 9 variantes de los 10 registros; ECG-Digitiser en
  `0004` (10 registros) y `0011`/`0012` (3 registros). ECG-Digitiser **no** se
  corrió en las fotos `0006`/`0009`/`0010`: en `0005` ya daba r@lag 0.13 y
  cuesta ≈17 min por foto; se juzgó que no aportaba información nueva.

Resultados finales (sólo métricas, sha256 y metadatos; ni imágenes ni
señales): `benchmarks/results/f6b_kaggle_two_engines_2026-09-25.json`. El
JSON `_phase1` queda como registro intermedio (mismas cifras en sus tipos).

## Requisitos

1. Credenciales en variables de entorno del entorno de ejecución (nunca en el
   repo): `KAGGLE_USERNAME` + `KAGGLE_KEY`, o `KAGGLE_API_TOKEN`. Una clave
   nueva `KGAT_…` en `KAGGLE_KEY` se reenvía como `KAGGLE_API_TOKEN`.
2. Red permitida hacia `www.kaggle.com`, `api.kaggle.com`, `kaggle.com`
   (las descargas redirigen a `storage.googleapis.com`).
3. Reglas de la competición aceptadas en kaggle.com (si no, 403).

## Protocolo

```bash
bash benchmarks/setup_engines.sh          # motores fijados + parches + pesos verificados
pip install kaggle
python benchmarks/f6b_fetch_kaggle.py --list          # verificar estructura remota
python benchmarks/f6b_fetch_kaggle.py --dest runs/f6b/kaggle --n-records 10
python benchmarks/f6b_kaggle_bench.py --data runs/f6b/kaggle --work runs/f6b/bench \
    --types 0001 0003 0005                            # subconjunto; reanudable
# re-puntuar las corridas guardadas con el puntuador actual (sin re-ejecutar motores)
python benchmarks/f6b_rescore.py --data runs/f6b/kaggle --work runs/f6b/bench \
    --out runs/f6b/cases.rescored.jsonl
python benchmarks/f6b_report.py --cases runs/f6b/cases.rescored.jsonl \
    --meta runs/f6b/bench/f6b_results.json --types 0001 0003 0005 \
    --out benchmarks/results/f6b_kaggle_two_engines_<fecha>_phase1.json
```

- Selección determinista de registros (semilla fija sobre `train.csv`);
  sha256 de cada archivo en `fetch_manifest.json`. Nada se versiona.
- Por registro, variante de imagen (`<id>-<NNNN>.png`) y motor: `ingest →
  estimate_page_grids → digitize_page(10 s) → confirm_scale` en **dos ejes
  temporales** sobre la misma pasada del motor: `evidence` (rejilla propia) y
  `engine` (rejilla canónica del motor). Velocidad 25 mm/s y ganancia
  10 mm/mV entran como evidencia manual; `fs`/`sig_len` del registro sólo se
  usan para puntuar.
- Métricas por derivación sólo sobre muestras observadas: r@lag, RMSE, razón
  de amplitud, error de duración y **SNR estilo competición** (lag óptimo,
  medias restadas; no es la métrica oficial exacta: esa suma potencias de las
  12 derivaciones por registro, aquí se da la mediana por derivación). La
  verdad de Kaggle trae NaN fuera del tramo mostrado de cada derivación
  (2.5 s; II completa 10 s): se recorta a su tramo finito.
- **Alineación**: la estimación en el eje `engine` vive en el lienzo de 10 s
  del motor y cada motor coloca las derivaciones cortas distinto (Ahus en su
  hueco de la página, ECG-Digitiser desde t=0; `engine_placement_s` lo
  registra), así que se recorta a su tramo observado. Se busca el lag en
  **±0.2 s** alrededor del rango natural (`lag_slack_s`, como la alineación
  de la métrica oficial) y sólo se puntúan las muestras solapadas.
- Los fallos son casos, no se descartan: `ingest_rejected`, `engine_error`,
  `time_scale_unknown` (sin evidencia de rejilla el eje `evidence` se niega),
  `no_signal`, `no_valid_lag` y **`lead_missing`** (derivación que el motor
  no devolvió; cuenta en el denominador). Agregados por motor × eje × variante.
- Cada (registro, variante, motor) terminado se añade a `cases.jsonl`: un
  banco interrumpido se reanuda sin repetir trabajo.

## Tipos de imagen (verificado)

Tomado de la página *data description* de la competición (vía API de Kaggle,
2026-09-25). Los códigos `0002`, `0007` y `0008` no existen en `train`
(9 variantes por registro).

| código | tipo |
|---|---|
| 0001 | imagen original en color generada con ECG-image-kit |
| 0003 | impresa en color, escaneada en color |
| 0004 | impresa en color, escaneada en blanco y negro |
| 0005 | foto de móvil de la impresión en color |
| 0006 | foto de móvil del ECG en la pantalla de un portátil |
| 0009 | foto de móvil de impresiones manchadas y empapadas |
| 0010 | foto de móvil de impresiones con daño extenso |
| 0011 | escaneo en color de impresiones con moho |
| 0012 | escaneo en blanco y negro de impresiones con moho |

## Datos (fase 1)

10 registros `train` elegidos con semilla fija (`20260925`): 1512936796,
1561472702, 2338722083, 2806926781, 3203822582, 3406869873, 4079781180,
4166392674, 946413478, 950071314. `fs` 256–1025 Hz (256, 500, 512, 1000,
1025), siempre 10 s. 101 archivos, 835 MB, sha256 en el JSON. Imágenes:
original 2200×1700; escaneos ≈2130×1650; fotos 4032×3024.

## Resultados fase 1 (medianas por derivación; r@lag con IQR)

| motor | eje | tipo | ok/filas | r@lag med [IQR] | SNR dB med | amp. med | no-ok |
|---|---|---|---|---|---|---|---|
| Ahus | evidence | 0001 | 120/120 | 0.993 [0.99–1.00] | 18.5 | 1.01 | |
| Ahus | evidence | 0003 | 0/10 | – | – | – | `time_scale_unknown` 10 |
| Ahus | evidence | 0005 | 0/10 | – | – | – | `time_scale_unknown` 9, `engine_error` 1* |
| Ahus | engine | 0001 | 120/120 | 0.993 [0.99–1.00] | 18.1 | 1.01 | |
| Ahus | engine | 0003 | 120/120 | 0.910 [0.75–0.97] | 7.5 | 1.01 | |
| Ahus | engine | 0005 | 106/109 | 0.835 [0.66–0.95] | 4.6 | 1.02 | `engine_error` 1*, `lead_missing` 2 |
| ECG-Digitiser | evidence | 0001 | 120/120 | 0.990 [0.97–0.99] | 17.0 | 1.00 | |
| ECG-Digitiser | evidence | 0003 | 0/10 | – | – | – | `engine_error` 7, `time_scale_unknown` 3 |
| ECG-Digitiser | evidence | 0005 | 0/10 | – | – | – | `time_scale_unknown` 8, `engine_error` 2* |
| ECG-Digitiser | engine | 0001 | 120/120 | 0.989 [0.97–1.00] | 16.6 | 1.00 | |
| ECG-Digitiser | engine | 0003 | 36/43 | 0.957 [0.91–0.98] | 10.7 | 1.02 | `engine_error` 7 |
| ECG-Digitiser | engine | 0005 | 89/98 | 0.119 [0.05–0.26] | −15.5 | 7.22 | `lead_missing` 7, `engine_error` 2* |

\* incluye `3203822582-0005`, fallido en **ambos** motores por una ejecución
de diagnóstico mía concurrente (ver fallos); se repite en la fase 2.

Tiempo de pared mediano por imagen: Ahus 39 s (0001), 39 s (0003), 64 s
(0005); ECG-Digitiser 332 s, 336 s y 1041 s.

Rejilla propia (eje `evidence`): 0001 → 7.874 px/mm en los 10 registros
(200 dpi exactos); **0003 y 0005 → rechazada en 20/20** (`px_per_mm = None`).
En el registro inspeccionado el periodo dominante era ambiguo (≈37.4 px en el
escaneo, ≈55.5 px en la foto): el estimador ya no acepta un periodo como 1 mm
sin estructura menor/mayor que lo confirme (commit `4eb56fc`). Antes de ese
cambio el escaneo daba 38.3 px/mm, 5× el valor real (≈7.5 px/mm).

## Fallos observados (fase 1)

1. **Eje `evidence` inutilizable fuera de la imagen original**: 0/20 en
   escaneos y fotos, en ambos motores, por la negativa de la rejilla. Es el
   comportamiento deseado frente a un error 5× silencioso, pero hoy la única
   medida en imágenes reales es el eje `engine` (duración asumida, 10 s).
2. **ECG-Digitiser no produce señal en 7/10 escaneos en color (0003)**: el
   propio motor aborta con `ValueError: Signal is empty for record page-1`
   (reproducido fuera del banco sobre la entrada guardada de 2806926781).
   En los 3 que sí procesa (1512936796, 4079781180, 950071314) la calidad es
   alta (r@lag 0.957).
3. **ECG-Digitiser en fotos (0005)**: procesa 8/10 pero la traza no se
   parece a la verdad (r@lag 0.12, amplitud ×7.2, cobertura mediana 0.35) y
   omite la derivación III en 6 de ellas (I y III en 1561472702). Es un fallo
   de calidad, no de ejecución: el pipeline no lo marca por sí solo.
4. **Ahus en escaneos/fotos**: devuelve todas las derivaciones (salvo aVL y V5
   en 950071314-0005); r@lag baja de 0.993 (original) a 0.910 (escaneo) y
   0.835 (foto), SNR de 18 dB a 7.5 y 4.6 dB; amplitud estable (1.01–1.02).
5. **Fallos causados por mí, no por los motores** (3203822582-0005): una
   ejecución manual de ECG-Digitiser para diagnosticar el punto 2 coincidió
   con el banco. (a) Ahus murió por OOM del cgroup de 16 GB (9.5 GB RSS en la
   foto de 12 MP); (b) ECG-Digitiser usa carpetas temporales fijas relativas
   a su checkout (`data/temp_nnUNet_*`, borradas con `rmtree` al empezar) y
   la ejecución manual borró las de la corrida del banco. Consecuencia
   operativa: **no ejecutar dos instancias de ECG-Digitiser sobre el mismo
   checkout**, y con fotos grandes no correr motores en paralelo con 16 GB.
6. **Errores del banco destapados por datos reales** (corregidos con pruebas
   que los reproducen): verdad recortada al hueco vs. lienzo del motor
   (`log10(0)`, abortaba el banco); colocación distinta de las derivaciones
   por motor (`trim_to_observed`); lag fijado a 0 con estimación y verdad de
   igual longitud (r@lag de Ahus en escaneos 0.459 sin holgura vs 0.910 con
   ±0.2 s, lag mediano 23 ms); mensaje de error recortado por el principio
   (sólo barras de progreso); derivaciones no devueltas fuera del
   denominador.

## Resultados fase 2 (todas las variantes; medianas por derivación)

Tras la fase 2 se repitieron los `engine_error` (con mensaje completo): los
dos de 3203822582-0005 causados por mí salen bien al repetirlos (Ahus 12/12,
ECG-Digitiser 11/12); 3406869873-0005 es `Signal is empty` de ECG-Digitiser.

Ahus, eje `engine` (10 registros por variante):

| tipo | ok/filas | r@lag med [IQR] | SNR dB med | amp. med | no-ok |
|---|---|---|---|---|---|
| 0001 original | 120/120 | 0.993 [0.99–1.00] | 18.1 | 1.01 | |
| 0003 escaneo color | 120/120 | 0.910 [0.75–0.97] | 7.5 | 1.01 | |
| 0004 escaneo B/N | 120/120 | 0.919 [0.76–0.96] | 7.9 | 1.02 | |
| 0005 foto impresión | 118/120 | 0.833 [0.66–0.95] | 4.5 | 1.02 | `lead_missing` 2 |
| 0006 foto de pantalla | 95/120 | 0.954 [0.84–0.98] | 10.4 | 1.00 | `lead_missing` 25 |
| 0009 foto manchada/empapada | 120/120 | 0.933 [0.86–0.97] | 8.7 | 1.04 | |
| 0010 foto con daño extenso | 120/120 | 0.854 [0.72–0.94] | 5.6 | 1.03 | |
| 0011 escaneo color con moho | 120/120 | 0.899 [0.74–0.97] | 7.0 | 1.03 | |
| 0012 escaneo B/N con moho | 119/120 | 0.834 [0.67–0.96] | 4.6 | 1.04 | `lead_missing` 1 |
| **todas** | 1052/1080 | 0.922 [0.78–0.98] | 8.1 | 1.02 | `lead_missing` 28 |

ECG-Digitiser, eje `engine`:

| tipo | registros | ok/filas | r@lag med [IQR] | SNR dB med | no-ok |
|---|---|---|---|---|---|
| 0001 | 10 | 120/120 | 0.989 [0.97–1.00] | 16.6 | |
| 0003 | 10 | 36/43 | 0.957 [0.91–0.98] | 10.7 | `engine_error` 7 |
| 0004 | 10 | 48/54 | 0.951 [0.90–0.98] | 10.0 | `engine_error` 6 |
| 0005 | 10 | 100/109 | 0.132 [0.05–0.29] | −14.9 | `engine_error` 1, `lead_missing` 8 |
| 0011 | 3 | 12/14 | 0.873 [0.76–0.96] | 5.8 | `engine_error` 2 |
| 0012 | 3 | 12/14 | 0.839 [0.73–0.95] | 4.7 | `engine_error` 2 |

Eje `evidence`: sólo hay rejilla propia en la original (10/10) y en **2 de 10
escaneos B/N** (`0004`: 1512936796 7.49 px/mm; 4079781180 7.47 px/mm en x,
sin y). En esos dos, Ahus r@lag 0.936 (24/24) y ECG-Digitiser 0.980 (24/24).
En las otras 7 variantes, 0/10 (`time_scale_unknown`).

Tiempo de pared mediano por imagen: Ahus 35–39 s (escaneos), 59–64 s
(fotos); ECG-Digitiser ≈290–300 s (escaneos), ≈1000 s (fotos). Parte de la
fase 2 coincidió con ≈1.5 min de otra ejecución de Ahus (fotos del usuario,
fuera del banco): efecto menor en los tiempos, ninguno en las métricas.

### Fallos observados (fase 2)

1. **ECG-Digitiser "Signal is empty" es sistemático en escaneos**: 7/10 en
   color (0003), 6/10 en B/N (0004), 2/3 con moho (0011 y 0012); también en
   una foto (3406869873-0005). Los 18 `engine_error` del motor tienen ese
   mismo mensaje. Cuando no aborta, su calidad en escaneos es la mejor
   (r@lag 0.95–0.96).
2. **Ahus en fotos de pantalla (0006)** no devuelve 25 de 120 derivaciones,
   17 de ellas en 2 registros (3203822582 devuelve 3 de 12; 4166392674, 4 de
   12); las que devuelve son buenas (r@lag 0.954).
3. **Degradaciones para Ahus**, de menor a mayor impacto en r@lag: pantalla
   (0.954, pero con derivaciones perdidas), manchas (0.933), escaneos
   (0.91–0.92), moho color (0.899), daño extenso (0.854), foto de impresión
   y moho B/N (0.833–0.834). La amplitud apenas se desvía (1.00–1.04).
4. **La tira de ritmo de 10 s es la derivación más frágil** en escaneos y
   fotos: el error temporal se acumula a lo largo de la tira (visto al
   superponer traza y verdad; mismo fenómeno que B-07).

## Limitaciones

- 10 registros; ECG-Digitiser sólo en escaneos (3 registros en los de moho) y
  sin las fotos 0006/0009/0010. Sin intervalos de confianza: medianas e IQR
  descriptivos.
- Métrica por derivación estilo competición, no la oficial por registro.
- El eje `engine` asume 10 s de página y 25 mm/s/10 mm/mV confirmados a mano;
  en esta competición es cierto por construcción, en el flujo local no se
  sabe (B-03).
- Todas las variantes parten de una página renderizada con ECG-image-kit (con
  el que se entrenó ECG-Digitiser): 0001 es ese render; el resto son su
  impresión escaneada o fotografiada. No son de los equipos locales ni de su
  formato (B-02/B-03). No es validación clínica.

## Verificación de la cadena (no es evaluación)

Imagen sintética propia con la estructura de la competición (3x4 + tira II,
25 mm/s, 10 mm/mV, 200 dpi, registro `999001`) más una variante rotada 3°,
desenfocada y con ruido (`0005`), en CPU x86 de 4 núcleos. Sirve sólo para
comprobar el script; los números reales saldrán del banco. Expuso dos fallos
propios, corregidos con pruebas que los reproducen:

1. **Rejilla 5× en la imagen desenfocada.** El desenfoque borró las líneas de
   1 mm; el estimador tomó el periodo de 5 mm (39.38 px) como 1 mm y el eje
   por evidencia salió 5× corto sin avisar (Ahus r@lag 0.997 → 0.22,
   duración −80 %). Ahora un periodo sólo se acepta como 1 mm si la
   estructura menor/mayor lo confirma (pico de autocorrelación en 5P sobre
   4P ≥ 0.15; observado +0.19…+0.46 en rejillas reales, −0.04 desenfocada,
   −0.28 líneas uniformes); si no, `px_per_mm = None` con
   `ambiguous_period_px_*` registrado, y el eje por evidencia se niega
   (`time_scale_unknown`). También cubre un pico fuerte múltiplo (2–5×) de
   uno débil (líneas menores tenues), antes tomado como 1 mm.
2. **`confirm_scale` (evidence) con una derivación sin `x_px` finito** en sus
   muestras observadas (ECG-Digitiser): `IndexError` que tumbaba la corrida;
   ahora esa derivación queda sin confirmar.

Tras las correcciones, Ahus (≈70 s por imagen):

| variante | eje | ok/filas | r@lag med | SNR dB med | err dur % |
|---|---|---|---|---|---|
| 0001 limpia | evidence | 12/12 | 0.997 | 21.3 | 0.04 |
| 0001 limpia | engine | 12/12 | 0.990 | 17.0 | 0.00 |
| 0005 degradada | evidence | 0/1 (`time_scale_unknown`) | – | – | – |
| 0005 degradada | engine | 12/12 | 0.989 | 16.1 | 0.00 |

ECG-Digitiser (≈550 s por imagen en esta CPU) recuperó sólo 7 derivaciones
con r@lag ≤ 0.74 en esta imagen propia (no generada con ECG-Image-Kit, con
el que se entrenó y con el que F6-a dio 0.988); no se interpreta hasta ver
imágenes de la competición. Coste a planificar: ≈10 min por imagen y motor
para ambos motores → 10 registros × 12 variantes ≈ 20 h; empezar por un
subconjunto de variantes.

# F7: rejilla ambigua (1 mm o 5 mm) resuelta con el tamaño de página

## Problema

Desde `4eb56fc` el estimador de rejilla se niega cuando sólo ve un periodo P
(sin estructura menor/mayor): P puede ser 1 mm o 5 mm. En F6-b eso dejó el eje
`evidence` sin escala en casi todos los escaneos y fotos.

## Cambio (`src/ecg_photo/grid.py`, aplicado en `estimate_page_grids`)

`estimate_grid` no cambia. Un paso aparte, `resolve_ambiguous_period`:

1. **Supuesto de página**: la hoja entera está en la imagen, mide 250–300 mm
   de ancho (tira de 10 s a 25 mm/s = 250 mm) y ocupa ≥ 35 % del ancho. Así
   `ancho_px / P` sólo encaja en una hipótesis: 250–857 → P = 1 mm;
   50–171 → P = 5 mm; fuera de eso, sin decidir. Observado: 56.7–82.9 para
   periodos de 5 mm (escaneos y fotos Kaggle, fotos del GE MAC2000), 264–270
   para 1 mm (renders limpios de página completa).
2. **Rejilla uniforme**: P se vuelve a medir en 5 franjas horizontales; el
   supuesto sólo se aplica si ≥ 3 franjas coinciden dentro del 3 %
   ((máx−mín)/mediana). Una escala global es falsa con perspectiva.
3. La escala fina sale de la rejilla medida (refinada por posición de
   líneas); el supuesto sólo elige ×1 o ×5 y queda en `method`/`limitations`
   (`period_step_mm`, `band_spread_rel_x`, periodos ambiguos conservados).

## Evaluación (Kaggle F6-b, eje `evidence` re-confirmado sin re-ejecutar motores)

`benchmarks/f7_grid_prior_eval.py` → `benchmarks/results/f7_grid_prior_kaggle_2026-09-26.json`.
Error de escala = duración de la tira II por eje `evidence` / por eje
`engine` (10 s exactos en esta competición) − 1, sobre la misma traza (no
depende de la cobertura). Ahus:

| tipo | con escala propia (antes → ahora) | error de escala (%) |
|---|---|---|
| 0001 original | 10/10 → 10/10 | −0.2 … 0.0 |
| 0003 escaneo color | 0/10 → 10/10 | −1.0 … +0.3 |
| 0004 escaneo B/N | 2/10 → 10/10 | −2.1 … +1.9 |
| 0011 escaneo color con moho | 0/10 → 10/10 | −0.9 … +1.5 |
| 0012 escaneo B/N con moho | 0/10 → 10/10 | −2.3 … **+6.5** (3 de 10 > 2.5 %) |
| 0005 foto de impresión | 0/10 → 2/10 | +0.2, +1.1 |
| 0006 foto de pantalla | 0/10 → 3/10 | +1.0 (otro: el motor sólo devolvió 2.5 s de II; no medible) |
| 0010 foto con daño | 0/10 → 3/10 | +0.9 … +2.4 |
| 0009 foto manchada | 0/10 → 0/10 | – (sin periodo) |

r@lag por derivación en el eje `evidence` (mediana, Ahus): escaneo color
0.938 (eje `engine` 0.910), B/N 0.910 (0.919), moho color 0.868 (0.899), moho
B/N 0.788 (0.834). ECG-Digitiser: 0.974 / 0.979 / 0.930 / 0.883.

Sin la condición de rejilla uniforme (primera versión, `c7570e9`) dos fotos
con perspectiva (dispersión entre franjas 3.9 % y 13 %) daban +24.8 % y
+10.6 % de error de escala; ahora se niegan.

## Fotos reales del GE MAC2000 (3 fotos del usuario; no versionadas)

Las tres dan periodo de 5 mm por el supuesto de página (cociente 58–76),
pero **se niegan** por la condición de uniformidad: en una la escala varía
6.54 → 7.37 px/mm de arriba abajo (perspectiva, dispersión 5.8 %); en las
otras dos no hay 3 franjas medibles. Sin la condición, su FC por eje
`evidence` se desviaba +0.4 %, +9.2 % y +6.3 % del RR impreso por el equipo;
por eje `engine` (10 s del formato `4x2.5x3_25_R1`), −0.4 %, +4.2 % y +0.3 %.

Conclusión: para el formato del MAC2000 el eje `engine` es hoy el más fiable
en fotos; el eje `evidence` es fiable en escaneos planos y sirve como control
(detecta errores gruesos como el ×5). Para fotos con perspectiva hace falta
escala local por derivación o rectificación de la página.

## Limitaciones

- Una foto de cerca de parte de la hoja rompe el supuesto de página y
  podría elegir ×5 cuando es ×1; la duración de la tira medida debe entonces
  discrepar del formato (control pendiente, F7 paso 2).
- El escaneo B/N con moho (0012) llega a +6.5 % con rejilla uniforme.
- Umbrales ajustados con 90 imágenes Kaggle + 3 fotos; no es validación
  clínica.

# F7 paso 2: control de calidad sin verdad (`ecg_photo.qc`, `ecg-photo qc`)

Informe de sólo lectura sobre una corrida confirmada:

- Por derivación: `MISSING_LEAD`, `NO_SIGNAL`, `LOW_COVERAGE` (< 90 % de su
  tramo de 2.5 s o 10 s), `FLAT_TRACE` (rango p0.5–p99.5 < 0.05 mV) y
  `RHYTHM_DURATION_MISMATCH` (tira de ritmo ≠ 10 s ± 5 %, útil en el eje
  `evidence`).
- Entre derivaciones, identidades que cumple cualquier ECG real y que en el
  formato 3×4 + II (MAC2000 `4x2.5x3_25_R1`, Kaggle) comparten columna:
  Einthoven I + III = II y Goldberger aVR + aVL + aVF = 0. Residuo = rms del
  incumplimiento / rms de la señal (alineación ±40 ms entre filas).
  ≤ 0.35 coherente; 0.35–1.0 `LIMB_LEADS_DOUBTFUL`; > 1.0
  `LIMB_LEADS_INCONSISTENT`.
- Etiqueta: `insufficient` con derivación ausente, vacía, plana o residuo
  > 1.0; `acceptable` con cualquier otra marca; si no, `good`.

Umbrales ajustados en Kaggle: residuo de Einthoven mediano 0.10 en originales,
0.14–0.35 en escaneos, 0.34–0.53 en fotos; con la verdad, las derivaciones
siguen buenas hasta 1.0 y se degradan por encima (Goldberger > 1.0: r@lag
mediana 0.76, p10 0.35).

## ¿Separa buenas de malas? (Kaggle, eje `engine`, `benchmarks/f7_qc_eval.py`)

`benchmarks/results/f7_qc_kaggle_2026-09-26.json` — r@lag contra la verdad
por imagen (mediana de sus derivaciones):

| motor | etiqueta | imágenes | r@lag mediana | p10 |
|---|---|---|---|---|
| Ahus | good | 21 | 0.934 | 0.906 |
| Ahus | acceptable | 48 | 0.902 | 0.795 |
| Ahus | insufficient | 21 | 0.803 | 0.195 |
| ECG-Digitiser | good | 12 | 0.971 | 0.923 |
| ECG-Digitiser | acceptable | 8 | 0.953 | 0.622 |
| ECG-Digitiser | insufficient | 8 | 0.122 | 0.092 |

Las fotos inservibles de ECG-Digitiser (r@lag ≈ 0.13) salen `insufficient`
sin conocer la verdad.

## Fotos reales del GE MAC2000 (usuario; no versionadas)

| imagen | Einthoven | Goldberger | etiqueta | observado a mano |
|---|---|---|---|---|
| MAC2000, 81/min | 0.08 | 0.19 | good | FC = impresa |
| MAC2000, 96/min | 0.31 | 0.75 | acceptable (dudosa) | aVR/aVL/aVF r 0.82–0.89 vs PMcardio |
| MAC2000, 48/min, 1036 px | 1.06 | – | insufficient | aVL vacía (`FLAT_TRACE`), V6 cortada (`LOW_COVERAGE`) |
| otro equipo, formato 6×2 | 1.06 | 0.54 | insufficient | formato no soportado |
| imágenes PMcardio (referencia) | 0.14–0.24 | 0.21–0.23 | good | |

Limitaciones: umbrales ajustados con 118 imágenes Kaggle y 6 del usuario;
las identidades sólo cubren derivaciones de miembros (V1–V6 sólo tienen
cobertura/planitud); un fallo que respete las identidades (p. ej. las tres de
una columna escaladas igual) no se detecta. Guía para quien lee, no
validación clínica.

## F7 paso 3: contraste con lo impreso por el equipo

`ecg-photo qc RUN_DIR --printed-rr-ms 742` (o `--printed-hr 81`): detecta los
QRS en la tira de ritmo digitalizada (deflexión dominante, ≥ 0.3 s entre
latidos), mide el RR mediano y lo compara con el valor que el usuario copia
del encabezado del electrocardiógrafo. Con más de ±5 % de diferencia marca
`RR_MISMATCH_PRINTED` (etiqueta `insufficient`: el eje temporal no sirve para
medir intervalos); sin tira medible, `RR_NOT_MEASURABLE`. Se prefiere el RR
impreso a la FC (entera, ±1 lpm ≈ 1–2 %).

Fotos del GE MAC2000 (eje `engine`):

| foto | latidos | RR medido | RR impreso | error | etiqueta |
|---|---|---|---|---|---|
| 81/min | 13 | 739 ms | 742 ms | −0.4 % | good |
| 96/min | 16 | 624 ms | 622 ms | +0.3 % | acceptable (aVR/aVL/aVF dudosas) |
| 48/min, 1036 px | 8 | 1305 ms | 1252 ms | +4.2 % | insufficient (aVL plana, V6 cortada) |

En ritmos con disociación AV el equipo imprime RR y PP distintos; el
contraste usa el RR (frecuencia de QRS). Tolerancia fijada con 3 fotos; no es
validación clínica.

# F7 paso 4: resolución de la foto y ampliación al ingresar

Las fotos que llegan por mensajería se recomprimen a ~1000–1600 px de ancho;
una foto del GE MAC2000 recibida a 1036 px perdió aVL, cortó V6 y dio +4 %
de RR. `benchmarks/f7_resolution_eval.py` reduce las 10 fotos Kaggle `0005`
(4032 px) a esos anchos (Lanczos), opcionalmente las amplía de nuevo, pasa
Ahus por el pipeline normal (eje `engine`) y puntúa contra la verdad.
Resultados: `benchmarks/results/f7_resolution_kaggle_2026-09-26.json`.

| entrada al motor | ok/filas | r@lag med [IQR] | SNR dB | QC acceptable / insufficient |
|---|---|---|---|---|
| original 4032 px | 115/120 | 0.831 [0.68–0.95] | 4.7 | 8 / 2 |
| reducida a 1600 px | 120/120 | 0.834 [0.69–0.93] | 5.1 | 9 / 1 |
| reducida a 1000 px | 119/120 | 0.740 [0.55–0.87] | 3.2 | 1 / 9 |
| 1000 px ampliada ×2 (a mano) | 120/120 | 0.845 [0.69–0.94] | 5.3 | 9 / 1 |
| 1000 px con el ingreso nuevo (amplía ×2 solo) | 120/120 | 0.847 [0.68–0.94] | 5.3 | 8 / 2 |

- Hasta 1600 px no se pierde calidad; a 1000 px sí, y el control de calidad
  lo detecta (9/10 `insufficient`).
- Ampliar antes del motor la recupera entera. Desde `ab76d68`, `ingest`
  amplía las fotos de 400–1599 px de ancho por el factor entero que llega a
  ≥ 2000 px (máx. ×4), registrado como paso afín de la transformación
  archivo → página (`f7-lanczos-upsample-v1`), así que la geometría sigue
  refiriéndose al archivo. Los PDF no cambian.
- Sin ampliar, el motor no aprovecha la resolución extra de la original
  (4032 px no mejora a 1600 px): el cuello está en ~1000 px de ancho de hoja.

Al montar el experimento apareció un fallo de los adaptadores de motor
(`0b40ae5`): con una carpeta de trabajo relativa, la configuración que recibía
el subproceso de Ahus (que corre con `cwd` en su propio directorio) no
existía, y el error llegaba como `exit status 1` sin la salida del motor.
Ahora ambos adaptadores usan rutas absolutas y Ahus devuelve la cola de su
stderr.

Limitaciones: 10 fotos de un tipo (foto de impresión); reducción sintética
con Lanczos, no la compresión JPEG real de cada aplicación.

# F7 paso 5: perspectiva — rejilla en el marco alineado del motor

Ahus endereza la hoja con una homografía antes de leer los trazos, y sus
muestras son lineales en las columnas de esa imagen alineada; el adaptador la
registra en la cadena de geometría. Desde `4184796`, `confirm_scale` con eje
`evidence` deforma la página a ese marco (`src/ecg_photo/aligned.py`),
estima allí la rejilla (supuesto de página + uniformidad) y toma el tiempo de
cada derivación de su extensión en columnas alineadas. Sin cadena de Ahus
(ECG-Digitiser, corridas antiguas) se usa la rejilla de la página como antes.

Re-confirmación del eje `evidence` sobre las corridas F6-b guardadas, sin
re-ejecutar motores (`benchmarks/f7_grid_prior_eval.py --out-dir
runs/f7/aligned` → `benchmarks/results/f7_aligned_grid_kaggle_2026-09-26.json`).
Ahus; error de escala = duración de II por `evidence` / por `engine` − 1:

| tipo | con escala propia (página → alineado) | error de escala (%) | r@lag `evidence` | r@lag `engine` |
|---|---|---|---|---|
| 0001 original | 10/10 → 10/10 | −0.1 … +0.1 | 0.993 | 0.993 |
| 0003 escaneo color | 10/10 → 10/10 | −0.4 … 0.0 | 0.931 | 0.910 |
| 0004 escaneo B/N | 10/10 → 10/10 | −0.2 … −0.1 | 0.928 | 0.919 |
| 0005 foto de impresión | 2/10 → 10/10 | −0.5 … +1.4; **+21.3** (dos hojas superpuestas) | 0.845 | 0.833 |
| 0006 foto de pantalla | 3/10 → 8/10 | −1.0 … +0.3; **−77.7** (el motor sólo leyó 2.5 s) | 0.960 | 0.954 |
| 0009 foto manchada | 0/10 → 9/10 | −0.1 … +0.9 | 0.937 | 0.933 |
| 0010 foto con daño | 3/10 → 9/10 | +0.2 … +2.7 | 0.901 | 0.854 |
| 0011 escaneo color con moho | 10/10 → 10/10 | −1.0 … +0.2 | 0.910 | 0.899 |
| 0012 escaneo B/N con moho | 10/10 → 10/10 | −0.8 … +3.5 | 0.839 | 0.834 |
| **todas** | | | **0.941** | 0.922 |

- Con Ahus, el eje por rejilla propia supera ya al del motor en todos los
  tipos salvo la imagen original, donde empatan (global 0.941 frente a
  0.922), y funciona en fotos: antes se negaba en casi todas.
- Los dos casos extremos no son errores de escala silenciosos: en la foto de
  dos hojas la rejilla alineada sale a 6.7 px/mm frente a ~10.9 en las demás;
  en la foto de pantalla el motor colocó su lienzo de 10 s sobre 2.5 s de la
  hoja. En ambos la tira no mide 10 s y el control de calidad marca
  `RHYTHM_DURATION_MISMATCH`.
- ECG-Digitiser no cambia (su geometría no trae homografía).

Limitaciones: depende de la rectificación de Ahus (si su cuadrilátero es
erróneo, la escala también); 10 registros por tipo; no es validación clínica.

# F7 paso 6: detector de QRS del contraste con lo impreso

Cuatro fotos nuevas del GE MAC2000 (usuario; no versionadas, sin datos del
paciente en el repositorio) con ritmos difíciles: sinusal 77/min con papel
curvado, taquicardia supraventricular (RR 290 ms), bloqueo AV 2:1 (RR 1392 ms,
PP 759 ms) y fibrilación auricular 114/min con extrasístoles aberrantes.
La digitalización de Ahus (eje `engine`) reproducía bien las tiras de ritmo;
el contraste con el RR impreso fallaba en las cuatro por el detector:

| foto | antes: latidos / RR / error | causa | ahora: latidos / RR / error |
|---|---|---|---|
| sinusal 77/min | 6 / 526 ms / −32.2 % | deriva de 0.9 mV por el papel curvado y T altas: umbral fijo | 13 / 793 ms / +2.2 % |
| TSV RR 290 ms | 17 / 581 ms / +100 % | refractario de 0.3 s: contaba un latido de cada dos | 34 / 292 ms / +0.6 % |
| bloqueo AV 2:1 | 9 / 1065 ms / −23.5 % | ondas T (0.23 mV frente a R 0.43) contadas como QRS | 7 / 1384 ms / −0.6 % |
| FA 114/min | 16 / 566 ms / +7.6 % | se comparaba la mediana; el equipo imprime la media | 19 / 520 ms / −1.1 % |

Detector nuevo (`ecg_photo.qc.detect_qrs`): energía de pendiente de la
banda 5–25 Hz en 100 ms (raíz, escala de amplitud); candidatos ≥ 0.2 s
aparte por encima de 0.3 × p99.5 y latidos por encima de 0.4 × la mediana de
los candidatos (una extrasístole grande no oculta las demás). El número de
latidos fue exacto en las cuatro fotos para suelos 0.25–0.35 y ventanas
80–150 ms. El RR comparado es la media (`rr_measured_ms`); la mediana queda
en `rr_median_ms`. Pruebas sintéticas reproducen los cuatro fallos (fallan
con el detector anterior).

**Pendiente — identidades de miembros en fotos de bajo voltaje.** Tres de las
cuatro fotos siguen `insufficient` por `LIMB_LEADS_INCONSISTENT` aunque las
trazas son plausibles: el residuo se normaliza por el rms de la señal, que en
estos trazados es 0.06–0.14 mV, frente a errores absolutos de 0.08–0.16 mV
(≈ 1 mm en el papel). Un filtro paso alto de 0.56 Hz (el del equipo) no lo
resuelve. Cambiar la normalización (p. ej. un suelo absoluto en mV) exige
recalibrar los umbrales con las corridas Kaggle de `f7_qc_eval.py`; no se
cambia sin esa evaluación. En la foto de FA el error de Goldberger es
0.28 mV y falta V3: ahí la marca sí parece merecida.

# F7 paso 7: re-evaluación del QC en Kaggle con el detector nuevo

`benchmarks/f7_qc_eval.py` sobre las corridas F6-b guardadas (eje `engine`,
sin re-ejecutar motores) →
`benchmarks/results/f7_qc_rr_floor_kaggle_2026-09-26.json`.

**Separación de etiquetas:** idéntica a la tabla del paso 2 (Ahus good /
acceptable / insufficient: r@lag mediana 0.934 / 0.902 / 0.803;
ECG-Digitiser 0.971 / 0.953 / 0.122). El detector sólo afecta al contraste
de RR, que en Kaggle no se usa para etiquetar sin RR impreso.

**Contraste de RR contra la verdad.** Como "RR impreso" se toma el RR medio que
`detect_qrs` mide en la II verdadera (10 s), y se compara con el de la tira
digitalizada (104 imágenes con tira medible; 14 sin tira medible, casi todas
fotos inservibles de ECG-Digitiser o fallos de Ahus en `0006`):

| agregación del RR medido | \|error\| mediana | p90 | casos > 5 % |
|---|---|---|---|
| media simple (paso 6) | 0.10 % | 0.76 % | 7 |
| **detector nuevo + media recortada (0.5–1.8 × mediana)** | **0.10 %** | **0.73 %** | **1** |

Con la media simple, un latido perdido en la traza digitalizada (intervalo
≈ 2 RR) o uno espurio (intervalo partido) sesga el RR más del 5 %. Desde este
paso `rr_measured_ms` es la media de los intervalos entre 0.5 y 1.8 veces la
mediana (`RR_KEEP`; si ninguno cae ahí, la media de todos). La prueba
sintética de FA (RR 0.36–0.72 s, extrasístole aberrante) sigue pasando: esos
intervalos quedan dentro del rango. El caso restante > 5 % (registro
3406869873, foto manchada `0009`, Ahus, −8.5 %) sale `acceptable`. Prueba
nueva: `test_rr_mean_ignores_missed_beat` (falla con la media simple).
Las fotos del GE MAC2000 de los pasos 3 y 6 no están disponibles en el
repositorio: la media recortada no se re-midió sobre ellas, sólo sobre sus
reproducciones sintéticas.

**Suelo absoluto para las identidades de miembros.** Se probó normalizar el
residuo por `max(rms_ref, suelo)` (equivale a exigir residuo relativo > 1 y
error absoluto > suelo para `LIMB_LEADS_INCONSISTENT`). Imágenes Kaggle por
r@lag de sus derivaciones de miembros:

| suelo | falsas alarmas (miembros r ≥ 0.9, marcadas inconsistentes) | aciertos (miembros r < 0.7, marcadas inconsistentes) | Ahus `good`: n / p10 r@lag |
|---|---|---|---|
| 0 (actual) | 6 | 5 | 21 / 0.906 |
| 0.1 mV | 5 | 5 | 25 / 0.864 |
| 0.15 mV | 4 | 1 | 32 / 0.875 |
| 0.2 mV | 3 | 1 | 45 / 0.824 |

Ningún suelo mejora la separación: 0.1 mV quita una falsa alarma pero mete en
`good` imágenes peores (p10 0.906 → 0.864); desde 0.15 mV se pierden 4 de 5
aciertos. **Se mantiene el suelo 0** (`IDENTITY_RMS_FLOOR_MV`; `run_qc` acepta
`identity_floor_mV` para re-evaluar). Las fotos de bajo voltaje del paso 6
siguen marcadas `LIMB_LEADS_INCONSISTENT`; la de FA (Goldberger 0.28 mV, sin
V3) también lo estaría con cualquier suelo ≤ 0.2 mV. Queda pendiente: con
Kaggle (voltajes normales) no se puede ajustar un criterio para trazados de
bajo voltaje; hacen falta más fotos reales de ese tipo.

Limitaciones: 118 imágenes Kaggle de 10 registros; RR "impreso" derivado de
la señal verdadera con el mismo detector; no es validación clínica.

# F8: formatos distintos de 3×4 + II (objetivo O8 de `docs/mission.md`)

Motivo: una foto del usuario de otro electrocardiógrafo, en formato 6×2 (seis
filas de dos columnas de 5 s), salía `insufficient` sin decir por qué. La
configuración de Ahus que usábamos (`inference_wrapper_george-moody-2024.yml`)
sólo deja elegir al identificador de formato entre tres variantes de 3×4.

**Banco sintético** (`benchmarks/f8_layout_eval.py` →
`benchmarks/results/f8_layout_ptbxl_2026-09-26.json`): las 12 derivaciones
reales de los 10 registros PTB-XL de F6-a dibujadas como impresión limpia
(25 mm/s, 10 mm/mV, rejilla mm, pulso de calibración, 200 ppp) en 3×4+1R y
6×2; Ahus con tres conjuntos de formatos. r@lag mediana por derivación contra
la parte impresa de la verdad (120 derivaciones por fila):

| impresión | formatos de Ahus | formato detectado | r@lag `evidence` | r@lag `engine` |
|---|---|---|---|---|
| 3×4+1R | George-Moody (anterior) | 3x4+1R 10/10 | 0.994 | 0.992 |
| 3×4+1R | `lead_layouts_all.yml` | standard_3x4_with_r1 10/10 | 0.994 | 0.992 |
| 3×4+1R | **`configs/ahus_lead_layouts.yml`** | 3x4+1R 10/10 | 0.994 | 0.992 |
| 6×2 | George-Moody (anterior) | 6x4 9/10, 3x4+3R 1/10 | 0.728 | 0.700 |
| 6×2 | `lead_layouts_all.yml` | standard_6x2 10/10 | 0.993 | 0.993 |
| 6×2 | **`configs/ahus_lead_layouts.yml`** | standard_6x2 10/10 | 0.993 | 0.993 |

**Imágenes reales 3×4** (Kaggle F6-b, tipos 0003 escaneo color, 0005 foto de
impresión, 0010 foto con daño; 30 imágenes; eje `engine`, pareado por
derivación con las corridas F6-b):

| formatos de Ahus | r@lag mediana 0003 / 0005 / 0010 | derivaciones que bajan / suben > 0.1 | formato detectado |
|---|---|---|---|
| George-Moody (F6-b) | 0.910 / 0.833 / 0.854 | — | — |
| `lead_layouts_all.yml` | 0.886 / 0.821 / 0.865 | 20 / 13 (sobre todo aVF, 9) | standard_3x4_with_r1/r2 |
| **`configs/ahus_lead_layouts.yml`** | 0.911 / 0.835 / 0.854 | 3 / 6 | 3x4+1R 29/30, 6x4 1/30 |

- El conjunto completo de Ahus reconoce 6×2 pero **empeora las imágenes 3×4
  reales**: detecta variantes con tira de ritmo comodín y pierde sobre todo
  aVF (`benchmarks/results/f8_kaggle_all_layouts_2026-09-26.json`; un fallo
  adicional del motor en esa corrida fue por disco lleno, no del motor).
- **Decisión:** `configs/ahus_lead_layouts.yml` = los tres formatos 3×4
  anteriores + `standard_6x2` de Ahus, por defecto en el adaptador desde este
  paso. Reconoce 6×2 como el conjunto completo y deja las imágenes 3×4 reales
  como estaban (`benchmarks/results/f8_kaggle_own_layouts_2026-09-26.json`;
  las diferencias de ±0.1 en pocas derivaciones aparecen en ambos sentidos).
- El control de calidad y `overview.png` usan el formato que informa el motor:
  cada derivación dura 10 s / columnas (6×2 → 5 s) y, sin tira de 10 s, el RR
  se mide en la derivación más larga (II primero).

Limitaciones: 6×2 sólo con impresiones sintéticas limpias (no fotos reales
de ese formato); no se evaluaron 6×2 con tira de ritmo ni 12×1; ECG-Digitiser
no cambia (no informa formato). No es validación clínica.

# F9: intervalos PR, QRS, QT (objetivo O7 de `docs/mission.md`)

`ecg_photo.intervals` (detalle del método en su docstring): latido mediano
por derivación, límites del QRS por energía de pendiente, fin de T cuando la
onda vuelve a menos del 15 % de su amplitud de la línea isoeléctrica, inicio
de P por tangente; valor = mediana entre derivaciones. Cada intervalo sale
`ok`, `doubtful` (hallado en < 3 derivaciones o derivaciones en desacuerdo:
IQR > 40 ms PR, 30 ms QRS, 40 ms QT) o `unavailable` (p. ej. sin onda P);
nunca un valor por defecto. QTc (Bazett, Fridericia) sólo con RR medido.

**Referencia:** LUDB (PhysioNet; Kalyakulina et al., IEEE Access 2020): 200
registros de 10 s, 12 derivaciones, 500 Hz, con inicio y fin de P, QRS y T
marcados por cardiólogos en cada derivación. La verdad de un registro usa la
misma agregación que la medida (mediana de latidos por derivación, mediana
entre derivaciones). Parámetros ajustados sólo con los registros impares
(`benchmarks/results/f9_intervals_ludb_odd_tuning_2026-09-26.json`); resultados
en los pares, no vistos al ajustar
(`benchmarks/results/f9_intervals_ludb_even_2026-09-26.json`,
`benchmarks/f9_intervals_eval.py`). Tolerancias de referencia: criterios CSE
para programas de medida (diferencia media / DE frente a la referencia: PR y
QRS 10 / 10 ms, QT 25 / 30 ms).

Brazos: **señal completa** (12 derivaciones de 10 s); **impreso** (lo que
muestra una hoja 3×4 + II: 2.5 s por derivación, II 10 s — lo máximo que
daría una digitalización perfecta); **digitalizado** (esa hoja dibujada como
imagen, Ahus, `confirm_scale` con eje por evidencia de imagen; 20 registros
pares).

Registros pares (diferencia medida − cardiólogos, ms; «todos» incluye los
`doubtful`):

| brazo | intervalo | todos: n · media ± DE · \|dif.\| mediana | sólo `ok`: n · media ± DE · \|dif.\| mediana | CSE con `ok` |
|---|---|---|---|---|
| impreso | PR | 87 · +3.5 ± 31.3 · 9.4 | 56 · −3.6 ± 11.5 · 8.5 | no (DE 11.5 > 10) |
| impreso | QRS | 100 · +3.8 ± 12.8 · 5.5 | 74 · +1.6 ± 9.8 · 4.5 | sí |
| impreso | QT | 100 · −13.4 ± 28.4 · 14.5 | 73 · −13.7 ± 13.2 · 14.0 | sí |
| digitalizado | PR | 16 · +0.2 ± 17.5 · 8.0 | 12 · −0.8 ± 10.7 · 7.1 | no (DE 10.7 > 10) |
| digitalizado | QRS | 20 · +9.0 ± 17.1 · 9.2 | 13 · +4.9 ± 8.2 · 7.0 | sí |
| digitalizado | QT | 20 · −5.0 ± 19.9 · 13.0 | 9 · −9.5 ± 22.4 · 20.0 | sí |
| señal completa | PR / QRS / QT | ver JSON | 70 / 83 / 77 · +0.3 ± 17.5 / −1.1 ± 17.5 / −15.0 ± 34.9 | no |

- El sesgo es pequeño en todos los brazos (|media| ≤ 15 ms); el error típico
  es de 5–15 ms. La DE la inflan unos pocos registros con errores grandes;
  la marca `doubtful` retiene la mayoría de ellos (en la hoja impresa, la DE
  baja de 31 a 11 ms en PR y de 28 a 13 ms en QT).
- Sobre la misma hoja, digitalizar con Ahus añade poco error a lo que da la
  señal impresa perfecta (mismos 20 registros, \|dif.\| mediana PR 8.0 frente
  a 11.2, QRS 9.2 frente a 6.5, QT 13.0 frente a 11.2).
- La señal completa (12 × 10 s) no mejora a la impresa: más latidos también
  suman latidos atípicos; el método no se ajustó para ese caso, que no es el
  del programa.
- QT se medía ~10–15 ms más corto que los cardiólogos con la mediana entre
  derivaciones. **Actualizado en F11:** QT pasa a ser el percentil 75 entre
  derivaciones (el QT es global); en estos mismos registros pares, hoja
  impresa, valores `ok`: **−2.7 ± 13.5 ms** (\|dif.\| mediana 8 ms) frente a
  −13.7 ± 13.2 ms; digitalizado `ok`: +2.2 ± 22.2 ms (9 registros). La tabla
  de arriba conserva los valores con la mediana; el JSON de resultados ya
  lleva los nuevos.

Limitaciones: una sola base (LUDB, un electrocardiógrafo Schiller), 100
registros de prueba y sólo 20 por el motor; la hoja digitalizada es una
impresión sintética limpia, no una foto; ritmos con marcapasos y bloqueos
incluidos pero no analizados por separado; la tolerancia CSE se aplicó a
nuestra agregación (mediana entre derivaciones), no al protocolo CSE
original. No es validación clínica.

# F10: intervalos en imágenes reales (Kaggle)

Pregunta: ¿cuánto error añade la digitalización de escaneos y fotos reales a
PR, QRS y QT? Estas imágenes no tienen anotaciones de cardiólogos; la
referencia es el mismo algoritmo (`ecg_photo.intervals`) aplicado a la señal
verdadera recortada tal como se imprimió. El error del algoritmo frente a
cardiólogos es F9 (LUDB). Corridas Ahus confirmadas con eje por evidencia de
imagen de F7 paso 5 (sin re-ejecutar motores); 88 de 90 imágenes confirmadas
(`benchmarks/f10_intervals_kaggle.py`,
`benchmarks/results/f10_intervals_kaggle_2026-09-26.json`).

Diferencia digitalizado − señal verdadera (ms; n · media ± DE · \|dif.\| mediana):

| grupo | PR | QRS | QT |
|---|---|---|---|
| 0001 imagen generada | 10 · +2 ± 14 · 4 | 10 · +5 ± 5 · 4 | 10 · +6 ± 10 · 4 |
| 0003 / 0004 escaneos | 20 · +3 ± 13 · 6 | 20 · +6 ± 6 · 5 | 20 · +1 ± 11 · 5 |
| 0005 foto de impresión | 10 · +9 ± 14 · 6 | 10 · +8 ± 7 · 8 | 9 · +16 ± 29 · 11 |
| 0006 foto de pantalla | 7 · −5 ± 25 · 9 | 7 · +8 ± 5 · 8 | 7 · +3 ± 8 · 5 |
| 0009 / 0010 fotos dañadas | 18 · −1 ± 9 · 6 | 18 · +10 ± 5 · 9 | 18 · +4 ± 10 · 7 |
| 0011 / 0012 escaneos con moho | 20 · +4 ± 16 · 6 | 20 · +9 ± 6 · 8 | 20 · −1 ± 9 · 6 |
| todas | 85 · +2 ± 14 · 5 | 85 · +8 ± 6 · 8 | 84 · +4 ± 13 · 6 |
| estado `ok` y calidad ≠ `insufficient` | 49 · −1 ± 9 · 5 | 66 · +7 ± 6 · 7 | 36 · +1 ± 7 · 5 |
| calidad `insufficient` | 17 · +3 ± 22 · 6 | 17 · +12 ± 6 · 13 | 16 · +3 ± 8 · 4 |

(Filas 0003/0004 y similares agrupan los dos tipos; valores por tipo en el JSON.)

- PR y QT: la digitalización añade poco (\|dif.\| mediana ≈ 5 ms); los valores
  `ok` con calidad aceptable quedan en −1 ± 9 ms (PR) y +1 ± 7 ms (QT).
- **QRS sale 5–10 ms más ancho** en todos los tipos, también en la imagen
  generada limpia: los dos límites se desplazan (inicio ≈ 4 ms antes, fin
  ≈ 2–4 ms después, 916 derivaciones), lo que apunta al trazo digitalizado
  más suave, no a ruido. Se probó un umbral de límites adaptado al ruido de
  la línea de base: no lo corrige de forma consistente entre las dos mitades
  de los registros y empeora LUDB (sesgo −4 a −26 ms); **descartado**. Sumado
  al sesgo propio del algoritmo en LUDB (+1.6 ms), el QRS de una foto queda
  unos +8 ms por encima de lo que marcaría un cardiólogo, dentro de la
  tolerancia CSE de media (10 ms) pero sin margen.
- **Cambio:** con calidad `insufficient`, ningún intervalo sale `ok`
  (`intervals.demote_on_qc`); el valor sigue visible como `doubtful` con el
  motivo. En esas imágenes el error de PR tenía DE 22 ms y el QRS +12 ms.

Limitaciones: 10 registros; la referencia es el algoritmo, no cardiólogos;
fs de la verdad variable (250–1025 Hz). No es validación clínica.

# F11: ¿coinciden nuestros intervalos con los que imprime el electrocardiógrafo?

El GE MAC2000 del usuario imprime FC, PR, QRS, QT y QTc calculados por el
programa GE Marquette 12SL. PTB-XL+ (PhysioNet) publica esas medidas 12SL
globales para cada registro de PTB-XL; se comparan con `ecg_photo.intervals`
sobre 300 registros PTB-XL elegidos al azar (semilla fija), recortados como una
hoja 3×4 + II (`benchmarks/f11_intervals_12sl.py`,
`benchmarks/results/f11_intervals_12sl_2026-09-27.json`).

**Hallazgo:** con la mediana entre derivaciones, nuestro QT salía **19–22 ms
más corto** que el QT impreso por 12SL (en las dos mitades de la muestra); PR
(+3 / −1 ms) y QRS (−2 / −6 ms) coincidían. El QT es un intervalo global
(inicio de QRS más temprano a fin de T más tardío); el percentil 75 entre
derivaciones no tiene sesgo frente a 12SL (−0.6 / −3.2 ms) y además acerca el
QT a los cardiólogos de LUDB (F9: −13.7 → −2.7 ms). **Cambio:** QT = percentil
75 entre derivaciones; PR y QRS siguen con la mediana. Un QT medido corto
puede ocultar un QT prolongado, por eso importa.

Nota: una referencia «global» construida con las marcas por derivación de
LUDB (inicio más temprano a fin más tardío entre las 12 anotaciones) sale
28 ms más ancha en QRS y 40 ms en QT que la mediana: toma el extremo de 12
anotaciones independientes y no sirve como objetivo.

Diferencia nuestra − 12SL, segunda mitad (150 registros, no usada para
elegir tolerancias), hoja impresa 3×4 + II, valores `ok`:

| medida | n | media ± DE (ms) | \|dif.\| mediana | tolerancia (p95 de la 1.ª mitad) | dentro |
|---|---|---|---|---|---|
| RR | 150 | −3.0 ± 41.1 | 1 | 31 ms | 96 % |
| PR | 86 | −6.2 ± 12.0 | 9 | 31 ms | 99 % |
| QRS | 128 | −4.8 ± 15.2 | 6 | 20 ms | 97 % |
| QT | 100 | −16.2 ± 33.1 | 18 | 39 ms | 91 % |
| QTc Bazett | 100 | −14.3 ± 48.2 | 21 | 46 ms | 91 % |

(Con todos los valores, no sólo `ok`, el QT queda en −5.4 ± 42.2 ms.)

**Uso:** `ecg-photo process ... --printed-pr-ms --printed-qrs-ms
--printed-qt-ms --printed-qtc-ms` compara cada intervalo con lo impreso en la
cabecera (`intervals.compare_printed`, tolerancias de la tabla). Fuera de
tolerancia, un valor `ok` pasa a `doubtful` con el motivo; `intervals.json`
(`printed`) y el informe PDF lo muestran («impreso X ms (coincide / NO
coincide)»). En la muestra de comprobación, entre el 1 % y el 9 % de los
valores `ok` quedarían marcados sin que haya error de digitalización (la
señal es la verdadera): son diferencias de método con 12SL, y una alerta pide
revisar el original, no invalida la medida.

Limitaciones: la comparación usa la señal verdadera, no fotos (el error que
añade la digitalización está en F10); 12SL no es una referencia clínica sino
la del equipo; tolerancias de 150 registros. No es validación clínica.

## PR: intentos de reducir su dispersión (descartados)

PR es el único intervalo con DE algo por encima de la tolerancia CSE
(11.5 ms frente a 10 en LUDB, hoja impresa, `ok`). Probado sobre las mitades
de ajuste (LUDB impares, primera mitad 12SL) y comprobado en las otras:

| cambio | mitades de ajuste | mitades de comprobación | decisión |
|---|---|---|---|
| amplitud mínima de P 0.03–0.08 mV × IQR máximo 25–40 ms | ninguna combinación mejora en las dos referencias | — | descartado |
| PR sólo de derivaciones con P ≥ 30–70 % de la mayor | 0.6 mejora 12SL (DE 27 → 16) pero empeora LUDB (11 → 17) | — | descartado |
| PR `ok` sólo con ≥ 5 derivaciones (antes 3) | 12SL DE 27 → 14 (quita un error de 207 ms), LUDB igual | LUDB 11.5 → 11.5; 12SL 12.0 → 12.1, con 4–6 valores `ok` menos | descartado: la mejora era un solo caso |

Con hojas impresas cada derivación tiene 2.5 s (2–4 latidos) y la onda P es
pequeña; la DE de ~11–12 ms frente a cardiólogos y frente a 12SL parece el
límite de este método. Los valores con PR poco fiable ya salen `doubtful`.

# F12: eje eléctrico del QRS

`intervals.qrs_axis`: área neta del QRS del latido mediano (sobre la línea
isoeléctrica, entre inicio y fin del QRS) en las seis derivaciones de
miembros; cada una es la proyección de un mismo vector sobre su ángulo del
sistema hexaxial (I 0°, II 60°, III 120°, aVR −150°, aVL −30°, aVF 90°) y el
vector se ajusta por mínimos cuadrados con todas las disponibles (más robusto
que I y aVF solas). `ok` con ≥ 4 derivaciones de miembros; `doubtful` con
menos o si el vector neto es casi nulo (eje indeterminado); nada con < 2.
Sin parámetros ajustados a los datos de evaluación.

**Referencias** (muestra F11: 300 registros PTB-XL, hoja 3×4 + II;
`benchmarks/f12_axis.py`, `benchmarks/results/f12_axis_ptbxl_2026-09-27.json`):
el eje frontal de GE 12SL (PTB-XL+) y la etiqueta de eje de los cardiólogos
de PTB-XL. PTB-XL usa los tipos de eje alemanes: `LAD` («Linkstyp», mediana
−13° tanto por nosotros como por 12SL) es una variante normal y `ALAD`
(«überdrehter Linkstyp», mediana −45° / −40°) la desviación izquierda; tomar
`LAD` como desviación rebajaba artificialmente el acuerdo (75 % → 87 %).

| hoja impresa | n | frente a 12SL: media · \|dif.\| mediana · p90 · ≤ 15° | categoría = 12SL | categoría = cardiólogos (12SL = cardiólogos) |
|---|---|---|---|---|
| todos | 289 | −1.6° · 5.6° · 23° · 82 % | 91 % | 87 % (88 %) |
| `ok` | 268 | −1.9° · 5.4° · 20° · 85 % | 92 % | 88 % (89 %) |

Categorías: normal −30°…+90°, izquierda −90°…−30°, derecha +90°…180° y
cuadrante noroeste. El acuerdo con los cardiólogos es el mismo que el del
propio 12SL; los desacuerdos están en el borde de −30°.

**Digitalización real** (F10 ampliado, escaneos y fotos Kaggle, referencia el
mismo algoritmo sobre la señal verdadera): el eje cambia poco (\|dif.\|
mediana 1–2° en todos los tipos; `ok` con calidad aceptable +1 ± 4°); los
`doubtful` tienen \|dif.\| mediana 35°, es decir, la marca retiene los
casos malos.

**Contraste con lo impreso:** `ecg-photo process ... --printed-axis-deg`;
tolerancia 39° (p95 de la primera mitad frente a 12SL, valores `ok`; en la
segunda mitad quedan dentro el 96 %), con diferencia angular (±180°).
`intervals.json` (`qrs_axis_deg`, estado propio), la interfaz y el informe
PDF (fila «Eje QRS» con la categoría) lo muestran.

Limitaciones: 300 registros de un solo equipo de adquisición; la categoría de
los cardiólogos existe en 173 de ellos; hoja limpia (el efecto de la foto está
en F10). No es validación clínica.
