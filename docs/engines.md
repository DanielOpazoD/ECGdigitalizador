# F2 — Banco de comparación de motores (sintético)

Estado: **una sola ejecución de humo** sobre imágenes sintéticas de ecg-image-kit generadas a partir de un fixture propio (todas las derivaciones idénticas). No es evaluación de rendimiento ni validación clínica. Sirve para fijar la interfaz del adaptador y comprobar reproducibilidad (commit, pesos, configuración, hash).

## Ahus Open-ECG-Digitizer
- Commit `97a15087…`, pesos verificados por sha256 (ver configs/checkpoints.json), dispositivo CPU, config efectiva hash `5313f4ef4b890a64ee82e4dff7602f4814d55becc7939a77595d478c726b255f`.
- Salida nativa: CSV canónico de 12 columnas en µV, longitud fija (`target_num_samples`), NaN donde no extrae; fs implícita = n/duración (no declarada por el motor). Layout detectado: `3x4+1R`.
- Tiempo de pared por imagen (CPU M4 Pro virtual): 252–278 s.

| semilla | derivación | cobertura | SNR dB | RMSE mV | r | ratio amplitud |
|---|---|---|---|---|---|---|
| 7 | II | 0.998 | -4.93 | 0.182 | 0.069 | 0.986 |
| 7 | III (max) | 0.247 | -1.94 | 0.211 | 0.176 | 0.950 |
| 7 | V4 (min) | 0.250 | -22.15 | 0.166 | -0.000 | 0.955 |
| 8 | II | 0.998 | -4.92 | 0.181 | 0.070 | 0.978 |
| 8 | III (max) | 0.247 | -1.93 | 0.211 | 0.175 | 0.952 |
| 8 | V4 (min) | 0.250 | -22.15 | 0.166 | 0.000 | 0.956 |
| 9 | II | 0.998 | -4.91 | 0.181 | 0.073 | 0.985 |
| 9 | III (max) | 0.247 | -1.91 | 0.211 | 0.182 | 0.953 |
| 9 | V4 (min) | 0.250 | -22.15 | 0.166 | 0.000 | 0.959 |

Observaciones: La tira de ritmo (II) alcanza ~99.8 % de cobertura; las demás derivaciones ocupan su hueco de 2.5 s (~25 %). El ratio de amplitud es ~0.95–0.99 en todos los casos, pero la correlación de Pearson es baja (~0.07 en II, ~0.18 en III, ~0 en V4) y la SNR es negativa en esta comparación: las métricas se calcularon contra la verdad tiled sin verificar alineación temporal fina ni signo por derivación. Los resultados son idénticos entre semillas dentro de ~0.01 en todas las métricas (la imagen generada es 2200×1700 a 200 dpi, x_grid=y_grid=39.37 px/cm).

## ECG-Digitiser (Krones), Kaggle (Đăng)
No ejecutados en F2-a. Pesos ECG-Digitiser 3×475 MB (LFS) pendientes; Kaggle requiere cuenta. Bloqueos B-05/B-06 vigentes.

## Cómo reproducir
```bash
# entornos (una sola vez)
python3.12 -m venv external/.venv-ahus
external/.venv-ahus/bin/pip install -r <(cut -d= -f1 external/ahus-freeze.txt)  # ver external/ahus-freeze.txt para versiones exactas
# weights: cd external/ahus && git lfs pull  (sha256 verificados en configs/checkpoints.json)
python3.12 -m venv external/.venv-imgkit
# deps imgkit: numpy==1.26.4 matplotlib pillow scipy wfdb imgaug opencv-python imageio shapely \
#   qrcode pyyaml scikit-image pandas seaborn beautifulsoup4 html5lib validators imutils \
#   joblib==1.3.2 requests tensorflow spacy

# ejecución del banco (3 semillas ≈ 15 min CPU)
. .venv/bin/activate
python benchmarks/f2_synthetic_bench.py \
  --ahus-root external/ahus \
  --ahus-python external/.venv-ahus/bin/python \
  --imgkit-root external/ecg-image-kit/codes/ecg-image-generator \
  --imgkit-python external/.venv-imgkit/bin/python \
  --base-config external/ahus/src/config/inference_wrapper_george-moody-2024.yml \
  --work runs/f2 --seeds 7 8 9
# resultado: runs/f2/f2_results.json (copiado a benchmarks/results/f2_ahus_synthetic_2026-09-25.json)
```
