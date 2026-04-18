#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/upload-model.sh --model <model_name> [--env-file <path>]
  scripts/upload-model.sh --all [--env-file <path>]
  scripts/upload-model.sh --model <model_name> --approve [--env-file <path>]
  scripts/upload-model.sh --all --approve [--env-file <path>]

Uploads one model pack or all model packs from LOCAL_MODELS_DIR to a GCE VM.
Uses gcloud compute ssh/scp under the hood.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_FILE="${REPO_ROOT}/.env"
MODE="single"
MODEL_NAME=""
AUTO_APPROVE=0

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
    --approve)
      AUTO_APPROVE=1
      shift
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

require_env DEPLOY_GCLOUD_INSTANCE
require_env DEPLOY_GCLOUD_ZONE
require_env DEPLOY_GCLOUD_PROJECT
require_env DEPLOY_MODELS_DIR

if ! command -v gcloud >/dev/null 2>&1; then
  echo "'gcloud' is required but not found on PATH." >&2
  exit 1
fi

LOCAL_MODELS_DIR="${LOCAL_MODELS_DIR:-${REPO_ROOT}/models}"
LOCAL_MODELS_DIR="${LOCAL_MODELS_DIR/#\~/$HOME}"
DEPLOY_MODELS_DIR="${DEPLOY_MODELS_DIR/#\~/$HOME}"
DEPLOY_USE_SUDO="${DEPLOY_USE_SUDO:-0}"
DEPLOY_STAGING_DIR="${DEPLOY_STAGING_DIR:-/tmp/category-classifier-upload-${USER}}"

if [[ "${DEPLOY_USE_SUDO}" != "0" && "${DEPLOY_USE_SUDO}" != "1" ]]; then
  echo "DEPLOY_USE_SUDO must be '0' or '1', got '${DEPLOY_USE_SUDO}'." >&2
  exit 1
fi

if [[ ! -d "${LOCAL_MODELS_DIR}" ]]; then
  echo "Local models directory does not exist: ${LOCAL_MODELS_DIR}" >&2
  exit 1
fi

GCLOUD_SSH=(gcloud compute ssh --zone "${DEPLOY_GCLOUD_ZONE}" "${DEPLOY_GCLOUD_INSTANCE}" --project "${DEPLOY_GCLOUD_PROJECT}")
GCLOUD_SCP=(gcloud compute scp --recurse --zone "${DEPLOY_GCLOUD_ZONE}" --project "${DEPLOY_GCLOUD_PROJECT}")

remote_cmd() {
  "${GCLOUD_SSH[@]}" --command "$1"
}

maybe_sudo() {
  if [[ "${DEPLOY_USE_SUDO}" == "1" ]]; then
    printf "sudo -n %s" "$1"
  else
    printf "%s" "$1"
  fi
}

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
echo "  GCE instance: ${DEPLOY_GCLOUD_INSTANCE}"
echo "  Zone:         ${DEPLOY_GCLOUD_ZONE}"
echo "  Project:      ${DEPLOY_GCLOUD_PROJECT}"
echo "  Remote models dir: ${DEPLOY_MODELS_DIR}"
echo "  Use sudo:     ${DEPLOY_USE_SUDO}"
echo "  Local models dir:  ${LOCAL_MODELS_DIR}"
echo "  Auto approve: ${AUTO_APPROVE}"
echo "  Models:"
for model in "${MODELS[@]}"; do
  echo "    - ${model}"
done

echo
if [[ "${AUTO_APPROVE}" == "1" ]]; then
  echo "Auto-approved (--approve): proceeding without prompt."
else
  read -r -p "Press Enter to continue or Ctrl+C to cancel: "
fi

echo "Checking remote paths..."
if [[ "${DEPLOY_USE_SUDO}" == "1" ]]; then
  if ! remote_cmd "sudo -n true" >/dev/null 2>&1; then
    echo "DEPLOY_USE_SUDO=1 but remote sudo is not available without a password." >&2
    echo "Configure passwordless sudo for deploy commands or set DEPLOY_USE_SUDO=0 with a writable DEPLOY_MODELS_DIR." >&2
    exit 1
  fi
fi

if ! remote_cmd "$(maybe_sudo "mkdir -p '${DEPLOY_MODELS_DIR}'")" >/dev/null; then
  echo "Failed to create remote models dir: ${DEPLOY_MODELS_DIR}" >&2
  echo "If this path needs root permissions, set DEPLOY_USE_SUDO=1 in .env." >&2
  echo "Or use a user-writable path like /home/<user>/category-classifier/models." >&2
  exit 1
fi

remote_cmd "rm -rf '${DEPLOY_STAGING_DIR}' && mkdir -p '${DEPLOY_STAGING_DIR}'" >/dev/null

EXISTING_REMOTE_MODELS=()
for model in "${MODELS[@]}"; do
  if remote_cmd "$(maybe_sudo "test -d '${DEPLOY_MODELS_DIR}/${model}'")" >/dev/null 2>&1; then
    EXISTING_REMOTE_MODELS+=("${model}")
  fi
done

if [[ "${#EXISTING_REMOTE_MODELS[@]}" -gt 0 ]]; then
  echo "Warning: these remote model directories already exist and will be overwritten:"
  for model in "${EXISTING_REMOTE_MODELS[@]}"; do
    echo "  - ${model}"
  done
  echo
  if [[ "${AUTO_APPROVE}" == "1" ]]; then
    echo "Auto-approved (--approve): proceeding with overwrite."
  else
    read -r -p "Press Enter to overwrite or Ctrl+C to cancel: "
  fi
fi

for model in "${MODELS[@]}"; do
  local_path="${LOCAL_MODELS_DIR}/${model}"
  remote_cmd "rm -rf '${DEPLOY_STAGING_DIR}/${model}'" >/dev/null
  "${GCLOUD_SCP[@]}" "${local_path}" "${DEPLOY_GCLOUD_INSTANCE}:${DEPLOY_STAGING_DIR}"
  remote_cmd "$(maybe_sudo "rm -rf '${DEPLOY_MODELS_DIR}/${model}'")" >/dev/null
  remote_cmd "$(maybe_sudo "mv '${DEPLOY_STAGING_DIR}/${model}' '${DEPLOY_MODELS_DIR}/${model}'")" >/dev/null
  echo "Uploaded model: ${model}"
done

remote_cmd "rm -rf '${DEPLOY_STAGING_DIR}'" >/dev/null || true

echo "Upload complete."
