# Guía de uso para el médico

Cómo pasar la foto de un ECG en papel a una señal digital, qué significa cada
resultado y qué no hacer con él. Para instalación y detalles técnicos:
`README.md` y `docs/`.

## Para qué sirve y para qué no

- **Sirve para:** obtener la señal de las 12 derivaciones (CSV, WFDB, PDF)
  desde una foto o escaneo, con FC, PR, QRS, QT, QTc y eje orientativos, y
  saber **cuánto confiar** en cada cosa.
- **No sirve para:** diagnosticar ni reemplazar la lectura del trazado
  original. No es un dispositivo médico validado. Ante cualquier valor
  «dudoso», «—» o «NO coincide», mande el original.

## 1. Tomar la foto

| Hacer | Por qué |
|---|---|
| Hoja **completa**, plana, con los cuatro bordes dentro de la foto | la escala se mide en la rejilla de la propia hoja |
| Rejilla milimetrada **visible** (sin sombras ni reflejos del flash) | sin rejilla medible el eje temporal cae al supuesto del motor |
| Cámara lo más **perpendicular** posible a la hoja | la perspectiva se corrige, pero una foto oblicua pierde detalle |
| Enviar el **archivo original** (compartir como documento, no como foto de WhatsApp) | por debajo de ~1600 px de ancho la calidad baja: a 1000 px, 9 de 10 fotos salieron `insufficient` antes de la corrección automática (F7) |

Formatos admitidos: PNG, JPEG o PDF de hasta 25 MB y 10 páginas. Hojas
**3×4 + tira de ritmo II** (p. ej. GE MAC2000) y **6×2**. Otros formatos no
están evaluados.

## 2. Procesarla

Instalación: `./install.sh` (ver `README.md`); `ecg-photo doctor` dice si
está lista.

**Interfaz web** (`ecg-photo serve --store DIRECTORIO` y abrir
`http://127.0.0.1:8000/ui`; sólo accesible desde el mismo computador):

1. **Cargar** el archivo.
2. En «Configuración»: motor `ahus`, **velocidad** y **ganancia** tal como
   están impresas en la hoja (normalmente 25 mm/s y 10 mm/mV), eje temporal
   «evidencia», su nombre en **Autor** y un **Motivo** («valores impresos en la
   hoja»). *Guardar como revisión nueva* y *Ejecutar trabajo*.
3. Esperar 1–2 minutos: el trabajo pasa a «completed» y se publica.
4. Revisar la línea de **calidad**, la de **intervalos**, el trazado
   redibujado y el **Informe PDF**.

**Un solo comando:**

```bash
ecg-photo process foto.jpg --out resultado --speed 25 --gain 10 \
    --author "Dra. X" --reason "valores impresos en la hoja"
```

Deja en `resultado/` el informe `report.pdf`, `overview.png`, la señal en
`export/` y el resumen `summary.json`.

**Velocidad y ganancia las confirma usted.** El programa nunca las supone: sin
ellas no entrega señal.

## 3. Leer la calidad

| Etiqueta | Qué hacer |
|---|---|
| `good` | la digitalización es coherente; usar con los límites de abajo |
| `acceptable` | coherente con alguna alerta; mirar cuál |
| `insufficient` | **no usar** mediciones: revisar el original o repetir la foto. Todos los intervalos pasan a «dudoso» |

| Alerta | Significa |
|---|---|
| `MISSING_LEAD` | una derivación no se digitalizó |
| `NO_SIGNAL` / `FLAT_TRACE` | derivación vacía o plana (foto cortada, trazo perdido) |
| `LOW_COVERAGE` | se perdió parte de una derivación (p. ej. superposición con otra) |
| `RHYTHM_DURATION_MISMATCH` | la tira de ritmo no dura lo esperado |
| `LIMB_LEADS_DOUBTFUL` / `LIMB_LEADS_INCONSISTENT` | I, II, III (o aVR, aVL, aVF) no cumplen las leyes de Einthoven / Goldberger: alguna derivación de miembros está mal digitalizada. **Puede ser falsa alarma en trazados de bajo voltaje** |
| `RR_MISMATCH_PRINTED` | el RR medido difiere > 5 % del impreso |
| `RR_NOT_MEASURABLE` | no hay latidos suficientes en la tira de ritmo |

## 4. Leer las mediciones

Cada medición viene con un estado:

- **ok:** hallada en suficientes derivaciones que concuerdan.
- **dudoso** (en rojo en el informe): pocas derivaciones, derivaciones en
  desacuerdo, calidad insuficiente o no coincide con lo impreso. **Revisar el
  original.**
- **—:** no medible (p. ej. sin onda P en fibrilación auricular). El programa
  no rellena con valores por defecto.

Precisión medida en valores `ok` (no es validación clínica; detalle en
`docs/evaluation.md`):

| Medición | Frente a | Error típico |
|---|---|---|
| QRS | cardiólogos (LUDB) | +1.6 ± 9.8 ms; en fotos reales sale ~8 ms más ancho |
| QT | cardiólogos (LUDB) | −2.7 ± 13.5 ms; sin sesgo frente al QT impreso por GE |
| PR | cardiólogos (LUDB) | −3.6 ± 11.5 ms (algo por encima de la tolerancia de referencia) |
| Eje QRS | GE 12SL | diferencia mediana 5.6°; misma categoría que los cardiólogos en 87 % |
| FC | señal verdadera (Kaggle) y FC impresa (7 fotos MAC2000) | RR dentro de ±5 % en 103 de 104 tiras buenas y en las 7 fotos |

Eje: normal −30° a +90°; desviado a la izquierda −30° a −90°; a la derecha
+90° a 180°; extremo, cuadrante noroeste.

## 5. Comparar con lo que imprimió el equipo

Es el control más útil: el electrocardiógrafo midió sobre la señal real.

- **Interfaz:** bloque «Valores impresos en la hoja» bajo los intervalos.
  Copie FC, PR, QRS, QT, QTc y eje (deje vacío lo que no aparece) y pulse
  *Comparar con lo impreso*.
- **Comando:** `--printed-hr 75 --printed-pr-ms 160 --printed-qrs-ms 92
  --printed-qt-ms 380 --printed-qtc-ms 418 --printed-axis-deg 45`.

Tolerancias (PR 31 ms, QRS 20, QT 39, QTc 46, eje 39°): el 91–99 % de las
mediciones buenas caen dentro. Un **«NO coincide»** puede venir de la
digitalización o de una diferencia de método con el equipo; en ambos casos,
**manda el original**.

## 6. Privacidad

- Todo corre en su computador; nada se envía a internet.
- Los estudios quedan en el directorio `--store`. Cuando ya no los necesite,
  bórrelos con la API (`DELETE /studies/{id}`; la interfaz aún no tiene botón)
  o eliminando el directorio del almacén.
- No suba imágenes con datos de pacientes a repositorios ni a servicios
  externos.

## Problemas frecuentes

| Síntoma | Causa probable y qué hacer |
|---|---|
| Una derivación vacía, `FLAT_TRACE` | foto recortada o de baja resolución: repetir con la hoja completa y el archivo original |
| `insufficient` con `LIMB_LEADS_INCONSISTENT` en un trazado de bajo voltaje | posible falsa alarma: comparar a ojo el trazado redibujado con la hoja |
| Eje temporal «supuesto del motor» en vez de «medido en la imagen» | no se pudo medir la rejilla (sombra, reflejo, rejilla muy tenue) |
| Formato no reconocido | sólo 3×4 + II y 6×2 están evaluados |
| Tarda 1–2 minutos | normal en un computador sin tarjeta gráfica |
