#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/update-server.sh [--approve] [--env-file <path>]

SSHes into the GCE VM, pulls the latest code from the current git branch,
and restarts the category-classifier systemd service.

Required .env variables (shared with upload-model.sh):
  DEPLOY_GCLOUD_INSTANCE, DEPLOY_GCLOUD_ZONE, DEPLOY_GCLOUD_PROJECT

Optional .env variables:
  DEPLOY_REPO_DIR   Remote repo path (default: parent of DEPLOY_MODELS_DIR)
  DEPLOY_SERVICE    Systemd service name (default: category-classifier)
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_FILE="${REPO_ROOT}/.env"
AUTO_APPROVE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --approve)
      AUTO_APPROVE=1
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

require_env DEPLOY_GCLOUD_INSTANCE
require_env DEPLOY_GCLOUD_ZONE
require_env DEPLOY_GCLOUD_PROJECT

if ! command -v gcloud >/dev/null 2>&1; then
  echo "'gcloud' is required but not found on PATH." >&2
  exit 1
fi

# Derive repo dir from DEPLOY_MODELS_DIR if not set explicitly.
DEPLOY_MODELS_DIR="${DEPLOY_MODELS_DIR:-/opt/category-classifier/models}"
DEPLOY_REPO_DIR="${DEPLOY_REPO_DIR:-$(dirname "${DEPLOY_MODELS_DIR}")}"
DEPLOY_SERVICE="${DEPLOY_SERVICE:-category-classifier}"

GCLOUD_SSH=(gcloud compute ssh --zone "${DEPLOY_GCLOUD_ZONE}" "${DEPLOY_GCLOUD_INSTANCE}" --project "${DEPLOY_GCLOUD_PROJECT}")

remote_cmd() {
  "${GCLOUD_SSH[@]}" --command "$1"
}

CURRENT_BRANCH="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)"

echo "Update target:"
echo "  GCE instance:  ${DEPLOY_GCLOUD_INSTANCE}"
echo "  Zone:          ${DEPLOY_GCLOUD_ZONE}"
echo "  Remote repo:   ${DEPLOY_REPO_DIR}"
echo "  Service:       ${DEPLOY_SERVICE}"
echo "  Branch:        ${CURRENT_BRANCH}"
echo

if [[ "${AUTO_APPROVE}" == "1" ]]; then
  echo "Auto-approved (--approve): proceeding without prompt."
else
  read -r -p "Press Enter to continue or Ctrl+C to cancel: "
fi

echo "Pulling latest code on remote..."
remote_cmd "cd '${DEPLOY_REPO_DIR}' && git fetch origin && git checkout '${CURRENT_BRANCH}' && git pull origin '${CURRENT_BRANCH}'"

echo "Restarting service '${DEPLOY_SERVICE}'..."
remote_cmd "sudo systemctl restart '${DEPLOY_SERVICE}'"

echo "Waiting for service to settle..."
sleep 3

echo "Service status:"
remote_cmd "sudo systemctl status '${DEPLOY_SERVICE}' --no-pager -l" || true

echo
echo "Done. Server updated and restarted."
