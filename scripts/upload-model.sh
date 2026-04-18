#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/upload-model.sh --model <model_name> [--env-file <path>]
  scripts/upload-model.sh --all [--env-file <path>]

Uploads one model pack or all model packs from LOCAL_MODELS_DIR to a remote VM over SSH.
Assumes SSH auth is already configured for DEPLOY_TARGET.
The script always requires an explicit confirmation before upload.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_FILE="${REPO_ROOT}/.env"
MODE="single"
MODEL_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODE="single"
      MODEL_NAME="${2:-}"
      shift 2
      ;;
    --all)
      MODE="all"
      MODEL_NAME=""
      shift
      ;;
    --env-file)
      ENV_FILE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing .env file: ${ENV_FILE}" >&2
  exit 1
fi

set -a
# shellcheck source=/dev/null
source "${ENV_FILE}"
set +a

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Required variable '${name}' is not set in ${ENV_FILE}" >&2
    exit 1
  fi
}

require_env DEPLOY_TARGET
require_env DEPLOY_MODELS_DIR

if ! command -v ssh >/dev/null 2>&1; then
  echo "'ssh' is required but not found on PATH." >&2
  exit 1
fi
if ! command -v rsync >/dev/null 2>&1; then
  echo "'rsync' is required but not found on PATH." >&2
  exit 1
fi

LOCAL_MODELS_DIR="${LOCAL_MODELS_DIR:-${REPO_ROOT}/models}"
LOCAL_MODELS_DIR="${LOCAL_MODELS_DIR/#\~/$HOME}"

if [[ ! -d "${LOCAL_MODELS_DIR}" ]]; then
  echo "Local models directory does not exist: ${LOCAL_MODELS_DIR}" >&2
  exit 1
fi

REQUIRED_PACK_FILES=(model.pt manifest.json label_map.json metrics.json)

is_valid_pack() {
  local model_dir="$1"
  for required in "${REQUIRED_PACK_FILES[@]}"; do
    if [[ ! -f "${model_dir}/${required}" ]]; then
      return 1
    fi
  done
  return 0
}

MODELS=()

if [[ "${MODE}" == "all" ]]; then
  while IFS= read -r -d '' candidate; do
    model_name="$(basename "${candidate}")"
    if is_valid_pack "${candidate}"; then
      MODELS+=("${model_name}")
    fi
  done < <(find "${LOCAL_MODELS_DIR}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
else
  if [[ -z "${MODEL_NAME}" ]]; then
    echo "Missing required argument: --model <model_name>" >&2
    exit 2
  fi
  if [[ "${MODEL_NAME}" == *"/"* || "${MODEL_NAME}" == *"\\"* ]]; then
    echo "Model name must be a direct directory name: ${MODEL_NAME}" >&2
    exit 2
  fi
  model_path="${LOCAL_MODELS_DIR}/${MODEL_NAME}"
  if [[ ! -d "${model_path}" ]]; then
    echo "Model does not exist: ${model_path}" >&2
    exit 1
  fi
  if ! is_valid_pack "${model_path}"; then
    echo "Model pack is missing required files: ${model_path}" >&2
    exit 1
  fi
  MODELS+=("${MODEL_NAME}")
fi

if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "No valid model packs found in ${LOCAL_MODELS_DIR}" >&2
  exit 1
fi

echo "Upload target:"
echo "  SSH target: ${DEPLOY_TARGET}"
echo "  Remote models dir: ${DEPLOY_MODELS_DIR}"
echo "  Local models dir:  ${LOCAL_MODELS_DIR}"
echo "  Models:"
for model in "${MODELS[@]}"; do
  echo "    - ${model}"
done

read -r -p "Type UPLOAD to continue: " CONFIRM
if [[ "${CONFIRM}" != "UPLOAD" ]]; then
  echo "Cancelled."
  exit 1
fi

echo "Connecting to remote host..."
ssh "${DEPLOY_TARGET}" "mkdir -p '${DEPLOY_MODELS_DIR}'"

for model in "${MODELS[@]}"; do
  local_path="${LOCAL_MODELS_DIR}/${model}/"
  remote_path="${DEPLOY_MODELS_DIR}/${model}/"
  rsync -a "${local_path}" "${DEPLOY_TARGET}:${remote_path}"
  echo "Uploaded model: ${model}"
done

echo "Upload complete."
