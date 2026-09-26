# F0 — Entorno real (inspeccionado, no supuesto)

Fecha de inspección: 2026-09-25 (UTC).

## Hardware / SO
| Ítem | Valor observado | Comando |
|---|---|---|
| SO | macOS 26.5.2 (25F84) | `sw_vers` |
| CPU | Apple M4 Pro (Virtual), 12 núcleos | `sysctl -n machdep.cpu.brand_string`, `sysctl -n hw.ncpu` |
| RAM | 17 179 869 184 B (16 GiB) | `sysctl -n hw.memsize` |
| GPU/CUDA | **No hay CUDA.** Aceleración MPS no evaluada; `torch` no instalado. | `python -c "import torch"` → ModuleNotFoundError |
| Git LFS | git-lfs/3.8.0 | `git lfs version` |

Consecuencia: la inferencia de los motores basados en PyTorch/nnU-Net deberá ejecutarse en CPU
(o MPS si se valida) en esta máquina. Cualquier tiempo de inferencia medido aquí no es
representativo de un servidor con CUDA.

## Python y dependencias efectivas
Intérprete: `/opt/homebrew/bin/python3.12` (venv en `.venv`). El Python del sistema (3.9.6, Xcode)
no se usa. `uv` no está instalado; se usa `venv` + `pip`.

```
numpy==2.5.3  scipy==1.18.1  pillow==12.3.0  pydantic==2.13.5  jsonschema==4.26.0
wfdb==4.3.1   matplotlib==3.11.2  pypdf==6.19.0  PyYAML==6.0.3
pytest==9.1.1 ruff==0.16.9  mypy==2.3.1
```

Reproducción:
```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e .[dev]
pytest -q && ruff check src tests && mypy src
```

## Datos disponibles
| Pregunta F0 | Respuesta |
|---|---|
| ¿Señales nativas del equipo local? | **No aportadas.** Bloqueo B-01. |
| ¿Fotos / PDF reales del flujo local? | **No aportados.** Bloqueo B-02. |
| ¿Formato inicial (marca/modelo de electrocardiógrafo, layout, mm/s, mm/mV por defecto)? | **Desconocido.** Bloqueo B-03. |
| ¿Datos públicos desidentificados? | No descargados todavía (PTB-XL / PhysioNet 2024 son candidatos; requieren registro de licencia y espacio). |

Todas las pruebas de F1 usan fixtures sintéticos deterministas generados por
`ecg_photo.fixtures`. **No constituyen validación clínica.**

## Motores candidatos — inventario fijado
Checkouts en `external/` (ignorados por git; se fija commit, no contenido). Pesos LFS **no
descargados** (`GIT_LFS_SKIP_SMUDGE=1`); se registran los OIDs sha256 declarados por LFS.

| Motor | Repo | Commit fijado | Fecha commit | Licencia | Pesos (sha256 LFS, tamaño) |
|---|---|---|---|---|---|
| Ahus Open-ECG-Digitizer | github.com/Ahus-AIM/Open-ECG-Digitizer | `97a15087d4abcda843da8c58ee74b1d8f47e6f9a` | 2026-06-04 | **CC BY-SA 4.0** (no es licencia de software; revisar compatibilidad antes de redistribuir) | `unet_weights_07072025.pt` 17fe7071…d320b (90 MB); `lead_name_unet_weights_07072025.pt` 840bd6bf…175a2 (23 MB) |
| ECG-Digitiser (Krones) | github.com/felixkrones/ECG-Digitiser | `e6f62aa776f105e4c7b04f21669da4d4f0df370b` | 2025-06-18 | BSD-2-Clause | nnU-Net M1 fold_0 `checkpoint_final.pth` d0e0d4c5…e355 (475 MB); M3 fold_all best 60020c47…f255b, final 8e4bae0b…375cffb (475 MB c/u) |
| Kaggle (Đăng, dangnh0611) | github.com/dangnh0611/kaggle_ecg_digitization | `8358d5b09a5b9519275e9b54c11716da5752a9d6` | 2026-02-06 | MIT (código) | Pesos **fuera de GitHub**, en dataset Kaggle `dangnh0611/ecg-checkpoints` (requiere cuenta Kaggle; términos del dataset no verificados). Sin hash aún. |
| ECG-Image-Kit | github.com/alphanumericslab/ecg-image-kit | `27b90f56896c9fc78b05a83ca14844ea2637aa0b` | 2024-07-11 | BSD-3-Clause | n/a (generador sintético) |
| ECGMiner | github.com/adofersan/ecg-miner | `369b014eca4318dab3247d135da421b77f05bd64` | — | MIT | n/a (referencia de interfaz asistida) |

Hashes completos y comandos: `configs/checkpoints.json`.

**Estado:** ningún motor ha sido ejecutado ni evaluado. No hay motor seleccionado. La comparación
(F2) exige: `git lfs pull` con verificación de hash, instalación de torch/nnU-Net en CPU,
y el banco sintético de `ecg-image-kit`. Requisitos del Ahus README: Python ≥3.12, probado sólo en
Ubuntu/Debian con CUDA.

## Registro de decisiones (F0)
| ID | Decisión | Motivo |
|---|---|---|
| D-01 | Python 3.12 + venv/pip | Único intérprete moderno disponible; Ahus exige ≥3.12 |
| D-02 | Contrato v1 con pydantic 2 + `extra="forbid"` | Rechazo de campos desconocidos; validación semántica en `model_validator` |
| D-03 | Máscara `gap_fill` siempre `false` en v1 | La política de relleno no está evaluada; nunca se puentea |
| D-04 | Pesos fuera de git; sólo hashes en `configs/checkpoints.json` | Tamaño y licencias |
| D-05 | WFDB sólo cuando hay mV + escala temporal + simultaneidad demostrada | Contrato cap. 14 |
| D-06 | Parche `float(rot_angle)` en ECG-Digitiser (parche 0001, `patches/`) | `rotate` de torchvision rechaza `np.float32`; sin el parche `digitize.py:352` falla siempre |

## Bloqueos abiertos
| ID | Bloqueo | Qué desbloquea | Qué se hace mientras |
|---|---|---|---|
| B-01 | Sin señales nativas del equipo local | Validación fs/amplitud contra referencia real | Fixtures sintéticos |
| B-02 | Sin fotos/PDF reales **del flujo local** | Perfiles de layout (F3), banco con las imágenes reales del usuario | Parcial: F6-b usa escaneos y fotos reales de impresiones (Kaggle PhysioNet, formato ECG-image-kit), no del equipo local; resultados en docs/evaluation.md |
| B-03 | Marca/modelo de ECG desconocidos | `supported_inputs.yml` con perfil confirmado | Perfiles `unknown`; medidas bloqueadas por LAYOUT_UNSUPPORTED |
| B-04 | Sin CUDA | Tiempos de inferencia representativos | Ejecución CPU para funcionalidad, no rendimiento |
| B-05 | Pesos LFS de ECG-Digitiser: `git lfs pull` falla por presupuesto LFS del proxy | Inferencia ECG-Digitiser | Resuelto: descarga directa vía `media.githubusercontent.com`, sha256 verificados contra `configs/checkpoints.json` |
| B-06 | Licencia CC BY-SA 4.0 de Ahus (pesos y código) | Redistribución del motor | Uso local de evaluación; decisión legal pendiente del usuario |
| B-07 | Discrepancia r@lag 0.954 (bench 3 s) vs 0.654 (pipeline 10 s) | Comparabilidad de métricas | Resuelto: es la ventana — r@lag 0.957/0.882/0.654 a 3/5/10 s; refleja el error temporal acumulado del motor (docs/pipeline.md) |
| B-08 | Pesos Kaggle requieren cuenta y aceptación de términos | Evaluar candidato Đăng | Se documenta; no se descarga |
| B-09 | Red del entorno cloud sin acceso a `kaggle.com` (403 del proxy) | Banco F6-b con imágenes reales de la competición | Resuelto (2026-09-25): entorno con red completa y credenciales en variables de entorno; reglas aceptadas (sin 403). Descarga de 10 registros (835 MB) con `f6b_fetch_kaggle.py`; fase 1 ejecutada |
