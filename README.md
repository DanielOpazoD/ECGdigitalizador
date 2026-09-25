# ECG Foto → Señal

Digitalización de electrocardiogramas fotografiados o en PDF hacia señales con escala temporal y
de voltaje explícitas, con máscaras, trazabilidad y mediciones sólo cuando exista evidencia.

**Estado: F0–F1 (contrato v1 + aritmética + fixtures sintéticos).** No hay motor de digitalización
integrado, no hay validación clínica, no hay diagnósticos habilitados. Ver `docs/environment.md`
para el inventario real, decisiones y bloqueos.

## Instalación
```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e .[dev]
```

## Verificación
```bash
ruff check src tests && ruff format --check src tests && mypy src && pytest -q
```

## Estructura
- `src/ecg_photo/contracts.py` — contrato canónico v1 (pydantic, `extra="forbid"`), validación
  semántica del directorio de revisión, JSON estricto (`null`, nunca `NaN`).
- `src/ecg_photo/geometry.py` — px ↔ s/mV; toda escala debe ser positiva y finita.
- `src/ecg_photo/signal.py` — rejilla semiabierta `[0, n/fs)`, remuestreo sin puentear huecos.
- `src/ecg_photo/measure.py` — RR/FC/PR/QRS/QT/QTc con nombres sufijados por unidad.
- `src/ecg_photo/fixtures.py` — revisiones sintéticas deterministas (calibrada, hueco, ganancia
  desconocida, tiempo desconocido, cola 2.503 s).
- `src/ecg_photo/transforms.py` — cadenas ordenadas de transformaciones (EXIF, crop,
  rotate90, affine, homografía) aplicables e invertibles punto a punto.
- `src/ecg_photo/ingest.py` — admisión por bytes mágicos y límites (`supported_inputs.yml`),
  EXIF, raster embebido o render PDF, `study_id` seguro.
- `src/ecg_photo/grid.py` — estimador determinista del paso de rejilla por autocorrelación.
- `src/ecg_photo/calibration.py` — resolución de escala por evidencias (manual > concordancia).
- `configs/` — configuración propuesta, entradas admitidas, inventario de motores/pesos,
  perfiles de formato (`configs/profiles/`).
- `external/` (ignorado) — checkouts de motores candidatos fijados por commit.
- `patches/` — parches mínimos aplicados a `external/` (ver `patches/README.md`).

## Comandos
```bash
ecg-photo fixture --kind calibrated --out rev
ecg-photo validate rev
ecg-photo export rev --out exp
ecg-photo run-demo --out demo
ecg-photo ingest foto.png --out estudio [--render-dpi 300]
ecg-photo calibrate estudio --page page-1 --speed 25 --gain 10 --author NOMBRE --reason "..."
```

## Reglas no negociables
Sin escala temporal no hay señal temporal; sin ganancia no hay mV; los huecos no se rellenan;
`representative_beat` no sirve para FC/RR; ningún valor por defecto de velocidad/ganancia.
