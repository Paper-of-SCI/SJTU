# Repository Guidelines

## Project Structure & Module Organization
This repository collects 3D Gaussian Splatting experiments and supporting tools. Core reusable Python modules live in `modules/` (`GaussianModel`, renderers, losses, optimizers) and pure utilities live in `utils/` (COLMAP readers, image helpers, PLY I/O). Training and comparison methods are under `methods/`, including `3dgs/`, `watersplatting/`, `seasplat/`, `leo_3DGS/`, and the vendored `methods/external/gsplat/`. Dataset inputs are expected under `src/datasets/`; generated artifacts belong in `outputs/`, `results/`, or `expirements/`. Dataset preparation scripts live in `scripts/`.

## Build, Test, and Development Commands
Use the project Conda environments under `/home/leo/miniconda3/envs/`; activate the method-specific environment before running CUDA code.

```bash
cd methods/external/gsplat
python -m pip install -e . --no-build-isolation
python -m pip install -r examples/requirements.txt
```

Installs the local `gsplat` checkout and example dependencies. For a lightweight root-level syntax check:

```bash
python -m compileall modules utils scripts methods/3dgs
```

Run the custom trainer from `methods/external/gsplat/examples/`, for example `python leo_3dgs_trainer.py default --data_dir <dataset> --disable_viewer --disable_video`.

## Coding Style & Naming Conventions
Python code uses 4-space indentation, `snake_case` functions and variables, and `PascalCase` classes. Keep module boundaries clear: data parsing in `utils/`, learnable/rendering logic in `modules/`, executable workflows in `methods/` or `scripts/`. The upstream `gsplat` subtree uses Black/isort via `lint/format-code.sh`; follow that style when editing files there.

## Testing Guidelines
The vendored `gsplat` tests are under `methods/external/gsplat/tests/` and use pytest:

```bash
cd methods/external/gsplat
pytest tests/test_basic.py
pytest tests/
```

Name new tests `test_*.py` and test functions `test_*`. GPU/CUDA tests may require the correct Conda CUDA variables and can be slow; document skipped or unrun tests in PR notes.

## Commit & Pull Request Guidelines
Recent history uses short messages such as `feat:...` and `update:...`, often with Chinese descriptions. Keep commits concise, scoped, and action-oriented. Pull requests should describe the method or module changed, list run commands and results, mention dataset paths used, and include images/metrics when rendering behavior changes.

## Agent-Specific Instructions
Do not modify source code unless explicitly asked. Preserve existing English comments; when asked for Chinese explanation, add Chinese clarification without rewriting the original comment. Avoid committing generated datasets, checkpoints, build caches, or large render outputs.
