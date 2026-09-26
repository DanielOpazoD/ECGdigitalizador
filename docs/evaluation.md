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
