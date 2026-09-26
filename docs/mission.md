# Misión y objetivos

Este documento fija para qué existe el programa y con qué criterios se decide
qué hacer después. Cada fase (y cada PR) debe poder decir a qué objetivo sirve
y con qué número observado lo demuestra.

## Misión

Convertir una **foto o un PDF de un electrocardiograma en papel** —en el caso
real de partida, una foto de móvil de un trazado de un GE MAC2000 (formato
3×4 + tira de ritmo II, 25 mm/s, 10 mm/mV)— en una **señal digital de 12
derivaciones con escalas de tiempo y voltaje explícitas y trazables**, que
**diga honestamente cuándo no es fiable**, y que se ejecute **localmente**, sin
enviar imágenes de pacientes a servicios externos.

Referencia práctica: la conversión que hoy da una aplicación comercial
(PMcardio) sobre las mismas fotos. El objetivo no es ganarle en diagnóstico,
sino entregar la señal y sus escalas con evidencia y con control de calidad.

## Para quién

Un médico que tiene el trazado en papel (o sólo su foto) y quiere:

1. la señal digital para archivarla, compararla o medirla;
2. saber si esa digitalización se puede usar o si hay que repetir la foto;
3. no depender de subir la imagen (con datos del paciente) a la nube.

## Principios (no negociables)

- **Sin escala no hay señal**: sin velocidad no hay eje temporal, sin ganancia
  no hay mV; nunca un valor por defecto silencioso (ver `README.md`).
- **Evidencia antes que supuesto**: la escala sale de la rejilla medida en la
  imagen, del pulso de calibración o de una confirmación humana registrada; el
  supuesto del motor (lienzo de 10 s) sólo como alternativa explícita.
- **Los huecos no se rellenan**; lo que el motor no vio queda como no
  observado.
- **Honestidad del resultado**: cada corrida lleva un informe de calidad
  (`good` / `acceptable` / `insufficient`) calculado sin conocer la verdad; es
  preferible marcar `insufficient` a entregar una señal mala como buena.
- **Privacidad**: las imágenes reales de pacientes no se versionan, no se
  suben a servicios externos y se borran cuando el usuario lo pide.
- **Números observados, no promesas**: cada cambio se mide en un banco
  (sintético, Kaggle o fotos reales) y se documenta en `docs/evaluation.md`,
  también cuando el resultado es negativo. Nada de esto es validación clínica.

## Objetivos medibles

| # | Objetivo | Criterio | Estado (2026-09-26) |
|---|---|---|---|
| O1 | Señal fiel en escaneos e impresiones | r@lag mediana por derivación ≥ 0.90 en escaneos Kaggle | **Cumplido**: 0.91–0.93 (Ahus, eje `evidence`) |
| O2 | Señal usable en fotos de móvil | r@lag mediana ≥ 0.85 en fotos Kaggle; escala propia en ≥ 8/10 | **Parcial**: 0.85–0.96 según tipo; foto de impresión 0.845; escala propia 8–10/10 |
| O3 | Eje temporal por evidencia | error de duración de la tira ≤ 2 % en ≥ 90 % de las imágenes | **Parcial**: −1.0…+2.7 % salvo 2 fallos marcados por QC |
| O4 | Control de calidad que separe | `insufficient` con r@lag claramente menor que `good`; ninguna digitalización inservible como `good` | **Cumplido en Kaggle** (0.80/0.12 frente a 0.93/0.97); falsas alarmas en trazados de bajo voltaje |
| O5 | Contraste con lo impreso | RR medido dentro de ±5 % del RR impreso cuando la tira es buena | **Cumplido en Kaggle** (1/104 fuera) y en 7 fotos MAC2000 |
| O6 | Uso real por el médico | de la foto al CSV/PDF/WFDB con **un solo comando** o desde la interfaz, con el informe de calidad a la vista | **Cumplido**: CLI `ecg-photo process` (con `summary.json` y `overview.png`); en la interfaz web, línea de calidad y `overview` del resultado publicado |
| O7 | Mediciones básicas | RR/FC y, con evidencia suficiente, PR/QRS/QT, con su incertidumbre | **Pendiente**: sólo RR/FC en el QC |
| O8 | Formatos del flujo real | 3×4+1R (MAC2000) soportado; 6×2 y otros detectados y rechazados con motivo | **Parcial**: 3×4+1R sí; 6×2 sale `insufficient` sin decir por qué |

## Cómo se prioriza

1. Primero lo que impide el **uso real** (O6) o entrega señales malas sin
   avisar (O4).
2. Después lo que mejora la fidelidad en **fotos de móvil** (O2, O3), que es
   la entrada real.
3. Después ampliar alcance (O7, O8).

Cada paso: reproducir el problema con una prueba o un banco, cambiar lo
mínimo, medir antes/después y documentarlo con sus limitaciones.

## Fuera de alcance

- Diagnóstico automático o interpretación clínica.
- Validación clínica o regulatoria (requeriría datos y protocolo propios).
- Rendimiento en GPU (el entorno de trabajo es CPU).
