# category-classifier

Local transaction category classifier with a train/predict/benchmark CLI.

Requires Python 3.11 or newer.

## What this version includes

- CSV/TSV ingestion with case-insensitive headers: `item`, `cost`, `date`, `category`
- Category normalization that strips leading emoji for internal labels
- PyTorch linear classifier trained on `[sentence_embedding ; normalized_price]`
- Reusable model packs under `models/<model_name>/`
- `predict` command that returns original display labels (emoji preserved)
- `benchmark` command for end-to-end latency on CPU and optionally MPS
- Optional FastAPI server for serving one model pack behind a tiny HTTP API

## Quick start

1. Get `uv` if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Sync environment with `uv`:

```bash
uv python install 3.11
uv sync --extra dev
```

3. Create a local `.env` file:

```bash
cp .env.example .env
```

Fill in required values in `.env`.

4. Train a model (first run downloads the sentence embedder):

```bash
uv run category-classifier train \
  --data data/transactions.csv \
  --model-name personal-v1
```

The output JSON shows the model pack location and evaluation metrics (accuracy, F1, confusion matrix). Model packs are saved to `models/<model_name>/` with figures/ subfolder containing confusion matrix and per-class accuracy plots.

5. Predict on a transaction:

```bash
uv run category-classifier predict \
  --model-pack personal-v1 \
  --item-name "Dinner at Nobu" \
  --price "120.50"
```

6. Benchmark end-to-end latency:

```bash
uv run category-classifier benchmark \
  --model-pack personal-v1 \
  --devices cpu,mps \
  --iterations 100
```

## FastAPI server

Install the server extra:

```bash
uv sync --extra dev --extra server
```

The serve command loads `.env` automatically, so keep server settings in your `.env`:

```bash
uv run category-classifier-serve
```

The server exposes these routes:

- `GET /healthz`
- `GET /prediction?item_name=...&price=...`
- `GET /available_models`
- `GET /model`
- `POST /switch` with JSON body `{"model_name":"<name>"}`

### API examples

Minimal route examples:

```bash
curl "http://127.0.0.1:8000/healthz"
curl "http://127.0.0.1:8000/available_models"
curl "http://127.0.0.1:8000/model"

curl -X POST "http://127.0.0.1:8000/switch" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"personal-v1"}'

curl "http://127.0.0.1:8000/prediction?item_name=Coffee%20Shop&price=6.50"
```

Environment variables:

- `MODELS_DIR` - directory containing model packs as subdirectories, e.g. `models/personal-v1/`
- Relative `MODELS_DIR` paths resolve from the project root (the directory containing `app/` and `category_classifier/`)
- `DEFAULT_MODEL` - optional model name to load at startup
- `INFERENCE_DEVICE` - `auto`, `cpu`, or `mps`
- `HOST` - bind address, default `0.0.0.0`
- `PORT` - bind port, default `8000`

## Deployment (VM)

This section is the production path.

### 1. Configure local deploy settings

Set these in your local `.env`:

- `LOCAL_MODELS_DIR` - local source directory, usually `./models`
- `DEPLOY_GCLOUD_INSTANCE` - instance name (example: `pfo-server`)
- `DEPLOY_GCLOUD_ZONE` - instance zone (example: `us-east1-b`)
- `DEPLOY_GCLOUD_PROJECT` - GCP project id (example: `orderique`)
- `DEPLOY_MODELS_DIR` - remote directory where model folders will be uploaded
- `DEPLOY_USE_SUDO` - set `1` only if `DEPLOY_MODELS_DIR` is root-owned (for example under `/opt`)

### 2. Upload models to VM

Upload one model:

```bash
./scripts/upload-model.sh --model personal-v1
```

Upload all valid models under `LOCAL_MODELS_DIR`:

```bash
./scripts/upload-model.sh --all
```

The script asks for confirmation before transfer (press Enter to continue), and if a remote model already exists it warns and asks again before overwriting.
Use `--approve` to skip prompts in non-interactive workflows:

```bash
./scripts/upload-model.sh --model personal-v1 --approve
```

For `/opt/...` deployments, set `DEPLOY_USE_SUDO=1` (this is the default in `.env.example`).

### 3. Configure runtime env on the VM

On the VM, create a `.env` in the repo root with at least:

```env
MODELS_DIR=/opt/category-classifier/models
DEFAULT_MODEL=personal-v1
INFERENCE_DEVICE=auto
HOST=0.0.0.0
PORT=8000
```

### 4. Start the API on the VM

```bash
gcloud compute ssh --zone "us-east1-b" "pfo-server" --project "orderique"
# then on the VM:
cd /path/to/category-classifier
uv sync --extra server
uv run category-classifier-serve
```

### 5. Verify and switch model

```bash
curl "http://<vm-ip>:8000/healthz"
curl "http://<vm-ip>:8000/model"
curl -X POST "http://<vm-ip>:8000/switch" \
  -H "Content-Type: application/json" \
  -d '{"model_name":"personal-v1"}'
```

### friend workflow

1. Friend trains model locally into `models/<model_name>/`.
2. Friend sends you the model folder.
3. You place it in your local `models/`.
4. You run `./scripts/upload-model.sh --model <model_name>`.

## Dataset contract

Required headers:

- `item`
- `cost`
- `date`
- `category`

Notes:

- Header matching is not capitalization-sensitive.
- Rows missing any required field are skipped in-memory with a warning log.
- `cost` accepts signed currency-like values (for example `$2,200.00`, `-$10.00`).
- `date` is validated but not used as a model feature in v1.
- `category` is cleaned internally, but prediction output uses the original display labels.

## Model pack format

Each trained model pack writes:

- `model.pt` — PyTorch model weights
- `manifest.json` — training config, schema version, class order, price normalization params
- `label_map.json` — category mappings (clean, display, ID)
- `metrics.json` — accuracy, F1, confusion matrix, class counts
- `figures/` — PNG visualizations (confusion matrix heatmap, per-class accuracy, metrics summary)

## Notes

- First `train` run downloads the sentence embedder (~200MB); subsequent runs use the cached model.
- During training, you may see matplotlib font warnings about emoji glyphs—these are harmless and do not affect PNG figure generation or model accuracy.
- Price normalization uses training set statistics; the same `price_mean` and `price_std` are applied at predict time to avoid data leakage.
- Dataset rows with missing required fields are logged and skipped.
- The API server keeps one active predictor instance per process and can switch active models at runtime.
