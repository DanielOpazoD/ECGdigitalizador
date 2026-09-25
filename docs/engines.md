# F2 — Banco de comparación de motores (sintético)

Estado: **ejecución de humo con dos motores** sobre imágenes sintéticas de ecg-image-kit generadas a partir de un fixture propio (todas las derivaciones idénticas). No es evaluación de rendimiento ni validación clínica. Sirve para fijar la interfaz del adaptador y comprobar reproducibilidad (commit, pesos, parches, configuración, hash).

Ninguno de los dos motores queda seleccionado; ambos quedan fijados (commit, pesos, parche, config) para F3.

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

Análisis de alineación (ahora computado en el banco, derivación II): el mejor desplazamiento fijo es **+58–63 ms** con r≈0.95–0.96 (frente a r≈0.13 sin desplazar); el **RR estimado es 1.008 s frente a 1.000 s** de verdad (error de escala temporal ≈0.8 %, acumulativo a lo largo de la tira) y la amplitud de pico R ≈0.85×. Es decir, el resto de error tras el desfase se explica sobre todo por deriva de escala temporal, no por la forma de la traza. Conclusión operativa para F3/F4: la rejilla canónica de Ahus (`target_num_samples` sobre el ancho detectado) **no** puede tomarse como escala temporal confirmada; la calibración horizontal debe derivarse de nuestra propia detección de rejilla o de evidencia explícita, y el desfase inicial exige registrar `study_start_s` por segmento.

## ECG-Digitiser (Krones)
- Commit `e6f62aa7…`; 3 checkpoints verificados por sha256 (M1 fold_0 final `d0e0d4c5…`, M3 fold_all best `60020c47…`, M3 fold_all final `8e4bae0b…`; ver `configs/checkpoints.json`). LFS del proxy agotado → descarga vía `media.githubusercontent.com` + verificación.
- Requiere **parche 0001** (`patches/ecg-digitiser-0001-rotate-float-angle.patch`, `float(rot_angle)` en `src/run/digitize.py:352`): torchvision `rotate` rechaza `np.float32` y sin el parche la línea falla siempre. El adaptador comprueba el texto del parche antes de verificar pesos.
- Ejecutado con **M3 (`fold_all/checkpoint_final`)**, el modelo entrenado con todos los folds (66 600 imágenes); M1 (fold_0, 31 028) requiere además el enlace `fold_all -> fold_0` porque el código fija `-f all` (ver `patches/README.md`).
- Salida nativa: WFDB `.hea/.dat`, **fs=500 Hz, unidades mV, 5000 muestras (10 s)**, sig_names I, II, III, aVR, aVL, aVF, V1–V6. **`np.nan_to_num` antes de `wrsamp`: las zonas no extraídas quedan codificadas como 0.0 exacto** (adaptador expone `missing_encoded_as_zero` y `exact_zero_fraction_<lead>`). No devuelve layout ni metadatos.
- Tiempo de pared por imagen (CPU): 246–249 s.

## Kaggle (Đăng)
No ejecutado. Pesos fuera de GitHub (dataset Kaggle), requiere cuenta y aceptación de términos. Bloqueo B-07 vigente.

## Tabla comparativa (derivación II, verdad 1 mV / RR 1.000 s)

| semilla | motor | cobertura | frac. ceros exactos | best_lag_ms | r@best_lag | rr_est_s | r_peak_amp_ratio |
|---|---|---|---|---|---|---|---|
| 7 | ahus | 0.999 | 0.001 | 63 | 0.954 | 1.008 | 0.849 |
| 7 | ecg-digitiser | 1.000 | 0.000 | 126 | 0.830 | 1.014 | 0.972 |
| 8 | ahus | 0.998 | 0.002 | 63 | 0.957 | 1.008 | 0.848 |
| 8 | ecg-digitiser | 1.000 | 0.000 | 122 | 0.840 | 1.014 | 0.976 |
| 9 | ahus | 0.998 | 0.002 | 58 | 0.960 | 1.008 | 0.848 |
| 9 | ecg-digitiser | 1.000 | 0.000 | 130 | 0.820 | 1.016 | 0.973 |

Notas de lectura (sólo números observados): las 11 derivaciones de 2.5 s muestran en ambos motores ~75 % de hueco (Ahus: NaN → cobertura ~0.25; Digitiser: ceros exactos ~0.75). ECG-Digitiser estima RR 1.008–1.016 s en las derivaciones con picos y su pico R ≈0.96–0.98× vs 0.85× de Ahus; su mejor desplazamiento (~125 ms) es mayor que el de Ahus (~60 ms). En V4 (semilla 7) Digitiser no recupera forma (r@lag=−0.17) pese a marcar cobertura 1.0.

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

# entorno digitiser (una sola vez)
python3.12 -m venv external/.venv-digitiser
# deps: torch torchvision numpy scipy scikit-image opencv-python matplotlib tqdm wfdb pillow
#   + pip install -e external/ecg-digitiser/nnUNet   (ver external/digitiser-freeze.txt)
cd external/ecg-digitiser
# pesos: git lfs pull falla por presupuesto LFS -> descarga media.githubusercontent.com + sha256
git apply ../../patches/ecg-digitiser-0001-rotate-float-angle.patch
ln -s fold_0 models/M1/nnUNet_results/Dataset500_Signals/nnUNetTrainer__nnUNetPlans__2d/fold_all  # sólo si se usa M1
cd -

# ejecución del banco (2 motores x 3 semillas ≈ 25 min CPU)
. .venv/bin/activate
python benchmarks/f2_synthetic_bench.py \
  --ahus-root external/ahus \
  --ahus-python external/.venv-ahus/bin/python \
  --imgkit-root external/ecg-image-kit/codes/ecg-image-generator \
  --imgkit-python external/.venv-imgkit/bin/python \
  --base-config external/ahus/src/config/inference_wrapper_george-moody-2024.yml \
  --digitiser-root external/ecg-digitiser \
  --digitiser-python external/.venv-digitiser/bin/python \
  --digitiser-model models/M3 \
  --engines ahus ecg-digitiser \
  --work runs/f2 --seeds 7 8 9
# resultado: runs/f2/f2_results.json (copiado a benchmarks/results/f2_two_engines_synthetic_2026-09-25.json)
```
