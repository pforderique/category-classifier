# category-classifier

Local transaction category classifier with a train/predict/benchmark CLI.

## What this version includes

- CSV/TSV ingestion with case-insensitive headers: `item`, `cost`, `date`, `category`
- Category normalization that strips leading emoji for internal labels
- PyTorch linear classifier trained on `[sentence_embedding ; normalized_price]`
- Reusable model packs under `artifacts/<model_name>/`
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
uv sync --extra dev
```

3. Train a model (first run downloads the sentence embedder):

```bash
uv run category-classifier train \
  --data data/transactions.csv \
  --model-name personal-v1
```

The output JSON shows the model pack location and evaluation metrics (accuracy, F1, confusion matrix). Model packs are saved to `artifacts/<model_name>/` with figures/ subfolder containing confusion matrix and per-class accuracy plots.

4. Predict on a transaction:

```bash
uv run category-classifier predict \
  --model-pack personal-v1 \
  --item-name "Dinner at Nobu" \
  --price "120.50"
```

5. Benchmark end-to-end latency:

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

Copy model packs under `models/<model_name>/` (or set a custom models directory with `MODELS_DIR`), then run:

```bash
export MODELS_DIR="/Users/pfo/ws/category-classifier/models"
export DEFAULT_MODEL="personal-v1"  # optional
uv run category-classifier-serve
```

The server exposes these routes:

- `GET /healthz`
- `GET /prediction/?item_name=...&price=...`
- `GET /available_models`
- `GET /model`
- `POST /switch` with JSON body `{"model_name":"<name>"}`

Example request:

```bash
curl "http://127.0.0.1:8000/prediction/?item_name=Monthly%20Rent&price=2200.00"
```

Environment variables:

- `MODELS_DIR` - directory containing model packs as subdirectories, e.g. `models/personal-v1/`
- `DEFAULT_MODEL` - optional model name to load at startup
- `INFERENCE_DEVICE` - `auto`, `cpu`, or `mps`
- `HOST` - bind address, default `0.0.0.0`
- `PORT` - bind port, default `8000`

### launchd deployment

Use a LaunchAgent plist to keep the API running in the background on macOS. A minimal example:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.pfo.category-classifier</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/pfo/ws/category-classifier/.venv/bin/category-classifier-serve</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>MODELS_DIR</key>
    <string>/Users/pfo/ws/category-classifier/models</string>
    <key>DEFAULT_MODEL</key>
    <string>personal-v1</string>
    <key>HOST</key>
    <string>0.0.0.0</string>
    <key>PORT</key>
    <string>8000</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/tmp/category-classifier.out.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/category-classifier.err.log</string>
</dict>
</plist>
```

Load it with `launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.pfo.category-classifier.plist` and unload it with `launchctl bootout gui/$UID ~/Library/LaunchAgents/com.pfo.category-classifier.plist`.

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
