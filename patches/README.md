# Parches sobre checkouts de `external/`

Parches mínimos necesarios para ejecutar los motores fijados. Se aplican sobre el
checkout (ignorado por git) y no se suben al repositorio externo.

```bash
git -C external/ecg-digitiser apply ../../patches/ecg-digitiser-0001-rotate-float-angle.patch
git -C external/ecg-digitiser apply ../../patches/ecg-digitiser-0002-write-geometry.patch
git -C external/ahus apply ../../patches/ahus-0001-write-geometry.patch
```

- `ecg-digitiser-0001`: `float(rot_angle)` para la llamada a `rotate` (np.float32
  no es aceptado por torchvision).
- `ecg-digitiser-0002`: escribe `<record>_geometry.json` (ángulo de rotación,
  shapes, por derivación `x1/y1/width/height`, escalas asumidas del motor).
- `ahus-0001`: escribe `<base>_geometry.json` (source/destination points,
  tamaños original/remuestreado/alineado, `valid_span`, `target_num_samples`,
  `apply_dewarping`); el stash de `valid_span` va en `lead_identifier.normalize`.

Nota M1: `src/run/digitize.py` llama a `nnUNetv2_predict -f all` fijo, así que para
usar `models/M1` hay que enlazar `fold_all -> fold_0` dentro de
`models/M1/nnUNet_results/Dataset500_Signals/nnUNetTrainer__nnUNetPlans__2d/`.
