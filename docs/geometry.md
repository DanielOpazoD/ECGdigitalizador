# Geometría, transformaciones y calibración (F3-a)

## Marcos de coordenadas y cadena

Cuatro marcos: archivo original, imagen decodificada orientada, imagen
rectificada y señal local por segmento (origen arriba-izquierda, x →, y ↓ en
imagen; tiempo creciente y voltaje hacia arriba en señal). Cada corrección
(EXIF, recorte, rotación, homografía, deformación) se registra como
`TransformStep` dentro de un `TransformChain` con `transform_id`, marcos de
origen/destino, parámetros, tamaños de entrada/salida y
`procedure_version` — nunca se sustituye la cadena por «imagen normalizada»
(`contracts.py`).

`transforms.py` aplica/invierte la cadena punto a punto:
`apply_chain` (fuente→destino) e `invert_chain` (destino→fuente). Un paso
`exif_orientation` se expande a sub-pasos `rotate90`/`affine` equivalentes a
`PIL.ImageOps.exif_transpose` (verificado para las 8 orientaciones sobre
esquinas y píxeles, T39/T10–T12). `homography_folds` muestrea el jacobiano de
una homografía sobre la rejilla del marco y detecta pliegues (cambio de signo
del determinante): una homografía que pliega no es invertible y no puede
confiarse para reproyección.

Conversión en región rectificada uniforme (`geometry.py`):
`t = (x − x0)/(px_mm_x·s)`, `v = (y0 − y)/(px_mm_y·g)`.

## Estimador de rejilla (`grid.py`)

Determinista, numpy/scipy: en cada región (mosaico 3×3), perfil medio de
columna/fila → detrendido con media móvil → autocorrelación por FFT → primer
pico fuerte en [3, 60] px. La rejilla menor puede ser débil: si el primer pico
fuerte resulta ser el mayor (5 mm), se busca un subarmónico cerca de /5 con
umbrales relajados. Si sólo se confirma el período mayor, `px_per_mm = mayor/5`
y se declara en `limitations`. Para imágenes RGB se usa el canal de «rojez»
(R − (G+B)/2) que aísla la tinta de rejilla de la traza negra. Se agregan por
mediana entre regiones (mínimo 3 regiones con valor, si no → `None`), y
`residual_rel_*` = desviación estándar/mediana entre regiones.

Verificación no-CI sobre `runs/f2/imgkit_seed{7,8,9}/tiled12-0.png`
(verdad: 200 dpi → 7.874 px/mm, x_grid 39.37 px/5 mm):

| imagen | px/mm x | px/mm y | residual x | residual y | mayor px |
|---|---|---|---|---|---|
| seed7 | 7.936 | 7.936 | 0.0006 | 0.0014 | 39.27 |
| seed8 | 7.936 | 7.936 | 0.0006 | 0.0014 | 39.27 |
| seed9 | 7.936 | 7.936 | 0.0006 | 0.0014 | 39.27 |

Error ≈ +0.8 % frente a 7.874; residuales inter-región < 0.15 %. Sobre el PNG
propio de `render.py`: 11.81 px/mm a 300 dpi y 7.87 a 200 dpi (±2 %, en tests).

## Resolución por evidencias (`calibration.py`)

`resolve_scale(quantity, evidence)`:
- `manual` con autor+motivo gana → `ScaleStatus.manual`; si contradice otra
  evidencia > tolerancia (2 %) sigue ganando y añade `CALIBRATION_CONFLICT`.
- Sin manual: todas las evidencias elegibles deben concordar dentro de la
  tolerancia → `confirmed` (mediana); si discrepan → `review_required` +
  `CALIBRATION_CONFLICT`, valor `None`; si no hay → `unknown` +
  `CALIBRATION_MISSING`.
- `grid_period` sólo alimenta `px_per_mm_*`; `speed_mm_s`/`gain_mm_mV` exigen
  `declared_text`, `calibration_pulse` o `manual`. La rejilla da px/mm, no
  mm/s.

## Ingesta (`ingest.py`, T42–T44)

`sniff_kind` decide por bytes mágicos (JPEG `FFD8FF`, PNG `89504E47`,
PDF `%PDF-`); cualquier otro → `UNSUPPORTED_TYPE`. `admit` aplica límites de
`configs/supported_inputs.yml` (`max_file_mib`, `max_pdf_pages`,
`max_pixels_per_page`) **antes de decodificar** (apertura perezosa de PIL;
`DecompressionBombError` → `PIXEL_LIMIT`). `ingest` copia la fuente a
`source/<sha256>.<ext>`, escribe `pages/page-N.png` orientada (EXIF) y un
`TransformChain` por página; en PDF extrae el raster embebido si una sola
imagen cubre la página (`pdf_embedded_raster`, dpi declarado si x/y concuerdan
al 1 %) o renderiza con pypdfium2 a `render_dpi` (`pdf_rendered`). Multi-página
→ una `Page` por página, nunca mezcladas. `study_id` es uuid4, nunca derivado
del nombre; `original_filename` se guarda escapado (≤255, sin controles).
