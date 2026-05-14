# AGENTS.md

## Project

CNN-based garbage classification (4 classes: 厨余垃圾/可回收物/其他垃圾/有害垃圾). ResNet-34 architecture, ~21M params, 256×256 RGB input, ~900 lines across 8 Python files. No package structure.

## Pipeline (order matters)

```bash
python Merge_classes.py   # merges 265 → 4 classes, creates ../trash_division_data/ultimate_4_class/
python Train.py           # trains the model, saves best_model.pth + training_log.csv
python Finetune.py        # optional: freezes early layers, saves finetuned_model.pth + finetune_log.csv
python Evaluate.py        # plots confusion matrix / ROC / PR curves from best_model.pth
python Curve.py           # plots loss/f1/acc/lr curves from training_log.csv
```

Also usable standalone: `python Model.py` prints `torchsummary` parameter summary.

## Dependencies

No `requirements.txt` — install manually: `torch`, `torchvision`, `tqdm`, `matplotlib`, `pandas`, `Pillow`, `torchsummary`. `Evaluate.py` additionally needs `scikit-learn`.

## Data setup

Data expected **outside repo** at `../trash_division_data/` (sibling dir). `Merge_classes.py` reads `val/classname.txt` there; `Train.py` and `Finetune.py` expect `ultimate_4_class/{train,val}/` with class-numbered subdirs (`1/` to `4/`). All paths relative to repo root.

## .gitignore — whitelist pattern

`.gitignore` uses `*` (ignore everything) then un-ignores specific files with `!` patterns. **Any new file you add to the repo must be explicitly whitelisted** or it will be invisible to git. The current whitelist: `Dataloader.py`, `LICENSE`, `Merge_classes.py`, `Model.py`, `README.md`, `THIRD_PARTY_LICENSES.md`, `Train.py`, `.gitattributes`, `.gitignore`.

`best_model.pth` and `finetuned_model.pth` are **untracked** (~125 MB each) — back them up manually if needed. `Finetune.py`, `Curve.py`, `Evaluate.py`, `AGENTS.md`, `training_log*.csv`, and `finetune_log.csv` are also untracked (not in whitelist).

## Gotchas

- **Windows: set `num_workers=0`** in `create_dataloaders()` call sites (`Train.py:191`, `Finetune.py:196`, `Dataloader.py:229`)
- Device selection priority: `cuda > xpu > cpu` (`xpu` = Intel GPU)
- Training auto-resumes from `best_model.pth` if present in repo root; fine-tuning auto-loads it too
- `Dataloader.py` uses `RobustImageFolder` — scans all images, skips corrupted ones (tqdm progress), slow on first load
- Image normalization: hardcoded ImageNet stats (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- `create_dataloaders()` has a `val_split` parameter that's **never used** — the code always expects a pre-split `val/` folder

### Finetune-specific

- **BUG**: `freeze_base_layers()` references `model.stage2` and `model.stage3` but the model uses `layer2`/`layer3`. This crashes at runtime — fix to `model.layer2`/`model.layer3` (or delete the function, since it would freeze `layer2`+`layer3` while docstring says only conv1+stage2).
- `freeze_base_layers()` actually freezes **conv1, bn1, layer2, AND layer3** (despite docstring saying only conv1 + stage2). Only layer1, layer4, and fc are trainable.
- Class weights use `power=1.5` (vs `power=1.0` in Train) — amplifies minority-class weighting
- Defaults: `lr=0.0001`, `epochs=30` (vs `lr=0.001`, `epochs=20` in Train)
- Writes `finetune_log.csv` (Train writes `training_log.csv`)
- Loads `best_model.pth` then saves `finetuned_model.pth`

### Curve.py

- Hardcoded to read `training_log.csv` only — won't work for `finetune_log.csv`
- Requires `pandas`, saves `training_curves.png`

### Evaluate.py

- Hardcoded constants at top of `__main__` block: `MODEL_PATH`, `DATA_ROOT`, `BATCH_SIZE`, `NUM_WORKERS`
- Loads model from `best_model.pth` by default; handles both bare state_dict and `model_state_dict`/`model` key wrappers
- Saves `confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`
- Requires `scikit-learn`

## Model architecture reference

`Model.py` attribute names (for freezing / layer access):
- `conv1`, `bn1`, `relu`, `maxpool`
- `layer1`, `layer2`, `layer3`, `layer4`
- `avgpool`, `dropout` (nn.Dropout), `fc` (nn.Linear(512, 4))

## Testing

No test suite.
