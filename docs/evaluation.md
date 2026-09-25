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

**Fase 1 ejecutada (2026-09-25)**: 10 registros × variantes `0001` (original),
`0003` (escaneo color) y `0005` (foto móvil) × 2 motores × 2 ejes, en CPU
x86 de 4 núcleos y 16 GB, sin GPU. Resultados:
`benchmarks/results/f6b_kaggle_two_engines_2026-09-25_phase1.json` (sólo
métricas, sha256 y metadatos; ni imágenes ni señales). Fase 2 (resto de
variantes: Ahus en todas, ECG-Digitiser en los escaneos de los 10 registros
y en las fotos de 3) en curso; se añadirá aquí al terminar.

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

## Limitaciones

- 10 registros, 3 de 9 variantes en esta fase; ECG-Digitiser en las fotos de
  la fase 2 sólo en 3 registros (≈17 min por foto en CPU). Sin intervalos de
  confianza: medianas e IQR descriptivos.
- Métrica por derivación estilo competición, no la oficial por registro.
- El eje `engine` asume 10 s de página y 25 mm/s/10 mm/mV confirmados a mano;
  en esta competición es cierto por construcción, en el flujo local no se
  sabe (B-03).
- Todas las variantes parten de una página renderizada con ECG-image-kit (con
  el que se entrenó ECG-Digitiser): 0001 es ese render; el resto son su
  impresión escaneada o fotografiada. No son de los equipos locales ni de su
  formato (B-02/B-03). No es validación clínica.
- Un `engine_error` de ECG-Digitiser en fotos (3406869873-0005) quedó con el
  mensaje recortado (código anterior); la fase 2 lo repite con el mensaje
  completo.

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
