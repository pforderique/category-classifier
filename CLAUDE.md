# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync --extra dev                    # dev + training deps
uv sync --extra server                 # FastAPI server deps
uv sync --extra dev --extra server     # all

# Tests
uv run pytest tests/
uv run pytest tests/test_api_server.py -v   # single file

# Train a model
uv run category-classifier train --data data/transactions.csv --model-name personal-v1

# Run inference server
uv run category-classifier-serve

# Deploy
./scripts/upload-model.sh --model personal-v1 --approve
./scripts/update-server.sh --approve
```

## Architecture

Two entry points defined in `pyproject.toml`:
- `category-classifier` → `category_classifier/cli.py` (train / predict / benchmark)
- `category-classifier-serve` → `app/main.py` (FastAPI inference server)

### Training pipeline (`category_classifier/`)

Data flow: `dataset.py` loads CSV → `encoder.py` embeds item names via `paraphrase-MiniLM-L3-v2` → `training.py` trains a `LinearClassifier` (`model.py`) on `[embedding; normalized_price]` → `model_pack.py` saves the model pack → `evaluate.py` writes metrics + PNG figures.

`preprocessing.py` handles price parsing and category normalization (emoji stripped for internal labels, preserved for display). `predictor.py` is the inference wrapper used both by CLI and server.

### Inference server (`app/`)

`config.py` reads `.env` → `server.py` creates the FastAPI app → `api.py` registers routes → `model_runtime.py` manages an LRU cache of up to `MAX_LOADED_MODELS` loaded predictors.

Key routes:
- `GET /healthz`
- `GET /available_models`
- `GET /models/{model_name}/prediction?item_name=X&price=Y`

### Model pack format

Each trained model saves to `models/<name>/`: `model.pt`, `manifest.json` (encoder + price stats + class order), `label_map.json`, `metrics.json`, `figures/`.

### Deployment

GCP VM (gcloud). Scripts in `scripts/` use env vars from `.env` (`DEPLOY_GCLOUD_INSTANCE`, `DEPLOY_GCLOUD_ZONE`, etc.). Server runs as a systemd service named `category-classifier`.

## Deployment Scripts

**To deploy:** Use these two scripts in order. Both are in `scripts/`:

### `./scripts/upload-model.sh --model <model-name> --approve`

Uploads a trained model to pfo-server. The model is synced via scp to `/opt/category-classifier/models/<model-name>/`. Use when you've trained a new model and want to make it available on the inference server.

Example: `./scripts/upload-model.sh --model piero-v2 --approve`

**When to use:** After training a new model locally with `uv run category-classifier train --data ... --model-name piero-v2`.

### `./scripts/update-server.sh --approve`

Pulls the latest code from `main` branch on the pfo-server VM and restarts the `category-classifier` systemd service. Use when you've committed code changes (new date encoding, API updates, etc.) and want them live.

**When to use:** After committing code changes that affect the server behavior (new features, bug fixes, training pipeline changes).

**Order:** Always run `upload-model.sh` *before* `update-server.sh` if you're deploying a new model alongside code changes. If code only, just run `update-server.sh`.
