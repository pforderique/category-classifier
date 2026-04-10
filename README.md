# category-classifier

Local transaction category classifier with a train/predict/benchmark CLI.

## What this version includes

- CSV/TSV ingestion with case-insensitive headers: `item`, `cost`, `date`, `category`
- Category normalization that strips leading emoji for internal labels
- PyTorch linear classifier trained on `[sentence_embedding ; normalized_price]`
- Reusable model packs under `artifacts/<model_name>/`
- `predict` command that returns original display labels (emoji preserved)
- `benchmark` command for end-to-end latency on CPU and optionally MPS

## Quick start

1. Get `uv` if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Sync environment with `uv`:

```bash
uv sync --extra dev
```

3. Train:

```bash
uv run category-classifier train \
  --data data/transactions.tsv \
  --model-name personal-v1 \
  --artifacts-dir artifacts \
  --device cpu
```

4. Predict:

```bash
uv run category-classifier predict \
  --model-pack personal-v1 \
  --artifacts-dir artifacts \
  --item-name "Amex Gold Renewal Fee" \
  --price "$250.00"
```

5. Benchmark:

```bash
uv run category-classifier benchmark \
  --model-pack personal-v1 \
  --artifacts-dir artifacts \
  --devices cpu,mps
```

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

- `model.pt`
- `manifest.json`
- `label_map.json`
- `metrics.json`
