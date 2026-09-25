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
- `configs/` — configuración propuesta, entradas admitidas, inventario de motores/pesos.
- `external/` (ignorado) — checkouts de motores candidatos fijados por commit.

## Reglas no negociables
Sin escala temporal no hay señal temporal; sin ganancia no hay mV; los huecos no se rellenan;
`representative_beat` no sirve para FC/RR; ningún valor por defecto de velocidad/ganancia.
