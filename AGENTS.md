# Repository Guidelines

## Project Structure & Module Organization
- `source/leisaac/leisaac/` holds the core Python package; subpackages such as `devices/`, `tasks/`, and `utils/` map directly to IsaacLab integrations, task configs, and shared helpers.
- `scripts/` contains runnable entry points (`environments/`, `mimic/`, `evaluation/`, `convert/`) used for teleoperation, data generation, and dataset conversion.
- `assets/`, `datasets/`, and `converted_datasets/` store USD scenes and demonstration data; keep large binary files out of git-lfs-tracked areas unless instructed.
- `.vscode/` captures recommended workspace settings; sync changes sparingly and only when they improve the default developer experience.

## Build, Test, and Development Commands
- `pip install -e source/leisaac` installs the package in editable mode for local development.
- `pre-commit install` (once) then `pre-commit run --all-files` mirrors CI formatting and lint checks (black, isort, flake8, codespell, pyupgrade).
- Teleoperation smoke check: `python scripts/environments/teleoperation/teleop_se3_agent.py --task LeIsaac-SO101-PickOrange-v0 --device cuda --headless` verifies runtime integration without cameras.

## Coding Style & Naming Conventions
- Follow Python 3.10 standards with 4-space indentation, Google-style docstrings, and type hints where practical.
- Format via `black --line-length 120` and `isort --profile black`; lint with `flake8` (ignores in `.flake8` are preconfigured, do not add new ones casually).
- Use `snake_case` for modules, functions, and dataset file names; reserve `PascalCase` for classes and task config objects (e.g., `LiftCubeEnvCfg`).
- Configuration files and assets should match existing patterns (`*_cfg.py`, `.usd` file names mirroring task IDs).

## Testing Guidelines
- The repository currently lacks a formal unit-test suite; add `pytest`-based tests alongside the feature (e.g., `source/leisaac/leisaac/tests/test_<feature>.py`) and keep execution under a few minutes.
- Prefer deterministic, headless checks that stub IsaacLab dependencies; when full simulation is unavoidable, guard tests with markers so CI can skip hardware-dependent cases.
- Run `pytest source/leisaac -m "not slow"` locally before requesting review, and document any new markers in the test docstring.

## Commit & Pull Request Guidelines
- Follow the existing git history: imperative subject lines under 72 chars with optional PR references, e.g., `add mimic dataset exporter (#42)`.
- Group related changes per commit, keep diffs focused, and run `pre-commit` prior to committing.
- Pull requests should include: a concise summary, instructions for reproducing validation runs (commands and dataset paths), linked issues if applicable, and screenshots or logs for UI/simulation changes.
- Request review only after ensuring assets are referenced but not committed, large datasets are shared via release artifacts, and all checks pass locally.

## Assets & Configuration Tips
- Download official USD scenes into `assets/` as outlined in `README.md`; maintain the expected directory layout (`scenes/<name>/scene.usd` plus `assets/` and `objects/`).
- When introducing new environments, update both the corresponding task config in `source/leisaac/leisaac/tasks/` and provide a runnable example script under `scripts/` to aid reviewers.
