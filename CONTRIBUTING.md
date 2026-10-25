# Contributing

Thanks for considering a contribution. This repo is a small SSD object-detection training pipeline, so contributions are best kept small and focused.

## Setup

```
git clone <repo-url> && cd ssd-pipeline
python -m pip install torch torchvision albumentations opencv-python torchmetrics matplotlib tqdm
```

Test data lives under `data/` (gitignored) and is not in the repo — fetch or label your own following the tutorial in the README.

## How to contribute

1. Open an issue describing the bug or feature before writing code.
2. For a code change, keep the diff surgical: one behavior per commit, no drive-by refactors.
3. For new code, follow the existing layout: shared modules go in the `ssd_pipeline/` package, scripts stay at the repo root, and all tunables stay in `ssd_pipeline/config.py`.
4. Run the import smoke check before committing:

```
python -m py_compile train.py eval.py inference.py inference_video.py ssd_pipeline/*.py
```

## Code style

- Plain Python, no linter config enforcement — match the surrounding file style.
- Docstrings/comments only where the code cannot show the intent.
- Paths, hyperparameters, and class lists must remain config-driven; no hardcoding dataset-specific values in scripts or the package.

## Pull requests

- Describe what changed and why in the PR body; link the issue.
- Do not commit model checkpoints (`*.pth`/`*.pt`), datasets, or output plots — they are gitignored.
- If the change affects training behavior or the README workflow, update the README in the same PR.

## Reporting issues

Include: what you ran, the exact error traceback, your `ssd_pipeline/config.py` class list (no secrets/paths needed beyond data dir layout), and versions (`torch`, `torchvision`).

## License

By contributing you agree that your contributions will be licensed under the [MIT License](LICENSE).
