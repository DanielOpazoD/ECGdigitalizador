# Protocolo de validación con el electrocardiógrafo propio

Objetivo: medir, con fotos reales del flujo clínico, cuánto concuerdan las
mediciones del programa con las que imprimió el electrocardiógrafo (GE MAC2000
u otro) en la misma hoja. Es la evidencia que falta (O2, O4, O7 en
`docs/mission.md`). No es un estudio de validación clínica: compara con el
equipo, no con un cardiólogo ni con un desenlace.

## 1. Criterios fijados antes de ver los datos

Para cada medida impresa (FC, PR, QRS, QT, QTc, eje), sobre los valores que el
programa marca `ok`:

| Criterio | Umbral de aceptación | De dónde sale |
|---|---|---|
| Dentro de la tolerancia (PR 31 ms, QRS 20, QT 39, QTc 46, eje 39°, FC ±5 %) | **≥ 90 %** | F11/F12: 91–99 % con señal perfecta; la foto añade error |
| Sesgo (media de programa − impreso) | \|sesgo\| ≤ 10 ms (PR, QRS), ≤ 15 ms (QT, QTc), ≤ 10° (eje), ≤ 3 lpm (FC) | F9–F12 |
| Fotos con calidad `insufficient` | ≤ 20 % | F7: 1–2 de cada 10 fotos Kaggle |
| Medida marcada `ok` (no dudosa) | ≥ 50 % de las impresas | F10: 57 % (QT) a 94 % (QRS) en fotos y escaneos Kaggle |

Si un criterio no se cumple, el resultado se informa igual; no se cambian
umbrales ni se excluyen fotos después de ver los datos.

## 2. Muestra

- **30 a 50 ECG consecutivos** del equipo, tal como llegan (no elegir los
  «bonitos»: la selección sesga el resultado). Si hay pocos de algún tipo,
  anotarlo; no reemplazarlos.
- Idealmente incluye, por el flujo normal: ritmo sinusal, fibrilación
  auricular, bloqueos de rama, bajo voltaje, marcapasos. Anotar el ritmo en una
  columna aparte (opcional) si se quiere analizar por subgrupo.
- Sólo hojas 3×4 + II o 6×2 (formatos evaluados).

## 3. Toma de la foto (igual que en el uso real)

Como en `docs/guia_uso.md` §1: hoja completa y plana, rejilla visible, sin
flash ni reflejos, cámara perpendicular, **archivo original** (no reenviado
por mensajería). Anotar el modelo de teléfono. Una foto por hoja; si una sale
mal, se deja igual (es un dato) y, si se quiere, se añade una segunda como
foto distinta.

## 4. Privacidad

- Todo se procesa en el computador del médico; nada sale a internet.
- Nombrar las fotos sin datos del paciente (`ecg001.jpg`, `ecg002.jpg`…) y
  guardar fuera del repositorio la correspondencia con la ficha, si hace falta.
- La cabecera suele incluir nombre y RUT: si se van a compartir resultados,
  compartir **sólo** `concordancia.csv` y `concordancia.md` (no contienen
  imágenes ni identificadores si los nombres de archivo no los tienen).

## 5. Transcribir lo impreso

```bash
ecg-photo concordance carpeta_fotos --printed valores.csv --template
```

Crea `valores.csv` con una fila por foto. Abrirlo en Excel o LibreOffice y
copiar de cada hoja: `hr_bpm` (FC), `pr_ms`, `qrs_ms`, `qt_ms`, `qtc_ms` (el QTc
que imprima el equipo; el programa compara con Bazett) y `axis_deg` (eje QRS;
0–360 o ±180, da igual). Dejar vacío lo que no está impreso. Se aceptan `;` y
coma decimal.

Recomendado: que la transcripción la haga una persona y la revise otra (o
transcribir dos veces y comparar): un error de tipeo parece un error del
programa.

## 6. Procesar

```bash
ecg-photo concordance carpeta_fotos --printed valores.csv --out resultado \
    --speed 25 --gain 10
```

Velocidad y ganancia son las impresas en las hojas (obligatorias; el programa
nunca las supone). Tarda cerca de 1 minuto por foto en un computador sin
tarjeta gráfica; si se interrumpe, al volver a ejecutarlo sigue donde quedó.

Resultados en `resultado/`:

| Archivo | Contenido |
|---|---|
| `concordancia.md` | tabla por medida: n, sesgo, DE, límites de acuerdo 95 %, \|dif.\| mediana, % dentro de tolerancia (todos y `ok`) |
| `concordancia.csv` | una fila por foto: calidad, cada medida del programa y la impresa, estado |
| `concordancia.json` | lo mismo, para análisis |
| `bland_altman.png` | gráfico de Bland-Altman por medida (× = dudosa) |
| `runs/<foto>/` | el resultado completo de cada foto (informe PDF, señal, calidad) |

## 7. Leer el resultado

- Comparar cada fila de `concordancia.md` con la tabla del §1.
- Revisar en `bland_altman.png` si el error crece con el valor (p. ej. QT
  largos) o si hay puntos aislados; abrir el `report.pdf` de esas fotos.
- Una foto con «NO coincide» puede deberse a la digitalización, a la
  transcripción o a una diferencia de método con el equipo: mirar el original.

## Ensayo del protocolo sin fotos

`runs/f13` (no versionado) reprodujo el protocolo completo con 20 registros
PTB-XL impresos como hojas 3×4 + II y, como «valores impresos», las medidas
del programa GE 12SL publicadas en PTB-XL+ (con coma decimal, como una
planilla en español). Resultado en `docs/evaluation.md`, «F13».
