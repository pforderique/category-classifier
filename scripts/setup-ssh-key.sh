#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/setup-ssh-key.sh [--env-file <path>] [--key-path <path>]

Creates an SSH key if missing and installs the public key on DEPLOY_SSH_HOST.
Reads DEPLOY_SSH_HOST/DEPLOY_SSH_USER/DEPLOY_SSH_PORT from .env by default.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_FILE="${REPO_ROOT}/.env"
KEY_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="${2:-}"
      shift 2
      ;;
    --key-path)
      KEY_PATH="${2:-}"
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

if ! command -v ssh >/dev/null 2>&1; then
  echo "'ssh' is required but not found on PATH." >&2
  exit 1
fi
if ! command -v ssh-keygen >/dev/null 2>&1; then
  echo "'ssh-keygen' is required but not found on PATH." >&2
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

require_env DEPLOY_SSH_HOST
require_env DEPLOY_SSH_USER
DEPLOY_SSH_PORT="${DEPLOY_SSH_PORT:-22}"

if [[ -z "${KEY_PATH}" ]]; then
  KEY_PATH="${DEPLOY_SSH_KEY_PATH:-${HOME}/.ssh/id_ed25519}"
fi
KEY_PATH="${KEY_PATH/#\~/$HOME}"
PUB_KEY_PATH="${KEY_PATH}.pub"

mkdir -p "${HOME}/.ssh"
chmod 700 "${HOME}/.ssh"

if [[ ! -f "${KEY_PATH}" || ! -f "${PUB_KEY_PATH}" ]]; then
  echo "Creating SSH key: ${KEY_PATH}"
  ssh-keygen -t ed25519 -f "${KEY_PATH}" -C "category-classifier-deploy" -N ""
else
  echo "Using existing SSH key: ${KEY_PATH}"
fi

REMOTE="${DEPLOY_SSH_USER}@${DEPLOY_SSH_HOST}"

if command -v ssh-copy-id >/dev/null 2>&1; then
  ssh-copy-id -i "${PUB_KEY_PATH}" -p "${DEPLOY_SSH_PORT}" "${REMOTE}"
else
  echo "ssh-copy-id not found; installing key via ssh fallback."
  ssh -p "${DEPLOY_SSH_PORT}" "${REMOTE}" "umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys"
  cat "${PUB_KEY_PATH}" | ssh -p "${DEPLOY_SSH_PORT}" "${REMOTE}" "cat >> ~/.ssh/authorized_keys"
fi

echo "Verifying key-based login..."
ssh -i "${KEY_PATH}" -p "${DEPLOY_SSH_PORT}" "${REMOTE}" "echo SSH key setup complete on \$(hostname)"
echo "Done. Set DEPLOY_SSH_KEY_PATH=${KEY_PATH} in your .env."
