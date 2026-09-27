# Uso previsto y alcance (borrador inicial)

Documento de trabajo, no una declaración regulatoria. Fija qué hace y qué no
hace el programa para que el análisis de riesgos (`analisis_riesgos.md`) y la
trazabilidad (`trazabilidad.md`) tengan un marco estable. Cualquier uso clínico
exige antes la validación de `docs/protocolo_validacion.md` y una evaluación
regulatoria formal.

## Estado actual

**Herramienta de investigación y apoyo, no un dispositivo médico validado.**
No debe usarse como única base de una decisión clínica. Toda medición se
contrasta con el trazado original, que siempre manda.

## Uso previsto (propuesto)

Convertir la foto, el escaneo o el PDF de un electrocardiograma de 12
derivaciones impreso en papel en una señal digital con escalas de tiempo y
voltaje trazables, con un informe de calidad calculado sin conocer la verdad,
para **archivar, comparar y revisar** el trazado.

- **Usuario previsto**: médico o profesional de salud que sabe leer un ECG y
  tiene el original (o su foto) a la vista.
- **Entorno**: un computador del propio usuario; sin conexión a servicios
  externos (`serve` sólo en `127.0.0.1` salvo decisión explícita con token).
- **Entradas**: PNG/JPEG/PDF de un ECG impreso 3×4 + tira de ritmo (formato
  principal, GE MAC2000) o 6×2; velocidad y ganancia **declaradas por el
  usuario** tal como están impresas (nunca supuestas).
- **Salidas**: señal (CSV/WFDB/PDF/PNG), informe de calidad
  (`good`/`acceptable`/`insufficient`), mediciones orientativas (FC, PR, QRS,
  QT, QTc, eje, Sokolow-Lyon) con estado `ok`/`doubtful`/`unavailable`,
  alerta de ritmo irregular y contraste con los valores impresos.

## Fuera del uso previsto

- Diagnóstico o interpretación automática; la alerta de ritmo es un aviso para
  mirar el trazado, no un diagnóstico.
- Segmento ST, isquemia, arritmias distintas de «RR irregular».
- Monitorización, urgencias o decisiones con tiempo crítico.
- Trazados pediátricos, de marcapasos o formatos no evaluados
  (`docs/mission.md`, O8).
- Uso sin el original a la vista o sin que un profesional revise el resultado.

## Clasificación orientativa (a confirmar)

Un software que entrega mediciones usadas para decidir sobre un paciente
suele ser un dispositivo médico. Como referencia, en la UE (MDR, regla 11)
quedaría probablemente en clase IIa, y su ciclo de vida de software en
IEC 62304 clase B (una falla podría llevar a una lesión no grave si el
médico no contrasta con el original). En Chile corresponde consultar al
ISP. Esta clasificación es un supuesto de trabajo para dimensionar la
documentación, no una determinación.

## Normas de referencia

| Norma | Para qué | Estado en este repositorio |
|---|---|---|
| ISO 14971 | gestión de riesgos | análisis inicial en `analisis_riesgos.md` |
| IEC 62304 | ciclo de vida del software | requisitos y trazabilidad en `trazabilidad.md`; control de versiones, CI y dependencias fijadas (`requirements.lock`) |
| IEC 62366-1 | usabilidad | pendiente (guía de uso en `docs/guia_uso.md`; sin estudio formativo) |
| ISO 13485 | sistema de calidad | fuera del alcance de este repositorio |
| IEC 60601-2-25 / ISO 80601-2-86 | exactitud de mediciones de ECG | usados como referencia de tolerancias (CSE) en `docs/evaluation.md` |
