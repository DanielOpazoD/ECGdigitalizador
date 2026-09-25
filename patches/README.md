# Parches sobre checkouts de `external/`

Parches mínimos necesarios para ejecutar los motores fijados. Se aplican sobre el
checkout (ignorado por git) y no se suben al repositorio externo.

```bash
git -C external/ecg-digitiser apply ../../patches/ecg-digitiser-0001-rotate-float-angle.patch
```

Nota M1: `src/run/digitize.py` llama a `nnUNetv2_predict -f all` fijo, así que para
usar `models/M1` hay que enlazar `fold_all -> fold_0` dentro de
`models/M1/nnUNet_results/Dataset500_Signals/nnUNetTrainer__nnUNetPlans__2d/`.
