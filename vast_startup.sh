#!/usr/bin/env bash
set -Eeuo pipefail

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_URL="${OMNIVOICE_REPO_URL:-https://github.com/k2-fsa/OmniVoice.git}"
REPO_DIR="${OMNIVOICE_REPO_DIR:-$SCRIPT_DIR}"
SERVER_MODE="${OMNIVOICE_SERVER_MODE:-api}"
HOST="${OMNIVOICE_HOST:-0.0.0.0}"
PORT="${OMNIVOICE_PORT:-}"
MODEL="${OMNIVOICE_MODEL:-k2-fsa/OmniVoice}"
DEVICE="${OMNIVOICE_DEVICE:-}"
ROOT_PATH="${OMNIVOICE_ROOT_PATH:-}"
NO_ASR="${OMNIVOICE_NO_ASR:-0}"
SHARE="${OMNIVOICE_SHARE:-0}"
SKIP_PULL="${OMNIVOICE_SKIP_PULL:-0}"
SKIP_SYNC="${OMNIVOICE_SKIP_SYNC:-0}"
UV_SYNC_ARGS="${OMNIVOICE_UV_SYNC_ARGS:-}"
UV_DEFAULT_INDEX="${OMNIVOICE_UV_DEFAULT_INDEX:-}"

if [[ -z "$PORT" ]]; then
  if [[ "$SERVER_MODE" == "demo" ]]; then
    PORT="8001"
  else
    PORT="8002"
  fi
fi

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi

  log "uv not found. Installing it with python3 -m pip ..."
  python3 -m pip install --upgrade uv
}

clone_or_update_repo() {
  if [[ ! -d "$REPO_DIR/.git" ]]; then
    log "Cloning OmniVoice into $REPO_DIR ..."
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone "$REPO_URL" "$REPO_DIR"
    return
  fi

  cd "$REPO_DIR"

  if [[ "$SKIP_PULL" == "1" ]]; then
    log "Skipping git pull because OMNIVOICE_SKIP_PULL=1."
    return
  fi

  if ! git diff --quiet || ! git diff --cached --quiet; then
    log "Repository has local changes. Skipping git pull to avoid overwriting them."
    return
  fi

  log "Pulling latest changes in $REPO_DIR ..."
  git pull --ff-only
}

run_uv_sync() {
  cd "$REPO_DIR"

  if [[ "$SKIP_SYNC" == "1" ]]; then
    log "Skipping uv sync because OMNIVOICE_SKIP_SYNC=1."
    return
  fi

  log "Running uv sync ..."
  if [[ -n "$UV_DEFAULT_INDEX" && -n "$UV_SYNC_ARGS" ]]; then
    # shellcheck disable=SC2086
    uv sync --default-index "$UV_DEFAULT_INDEX" $UV_SYNC_ARGS
  elif [[ -n "$UV_DEFAULT_INDEX" ]]; then
    uv sync --default-index "$UV_DEFAULT_INDEX"
  elif [[ -n "$UV_SYNC_ARGS" ]]; then
    # shellcheck disable=SC2086
    uv sync $UV_SYNC_ARGS
  else
    uv sync
  fi
}

launch_server() {
  cd "$REPO_DIR"

  local -a common_args
  common_args=(--model "$MODEL" --ip "$HOST" --port "$PORT")

  if [[ -n "$DEVICE" ]]; then
    common_args+=(--device "$DEVICE")
  fi

  if [[ -n "$ROOT_PATH" ]]; then
    common_args+=(--root-path "$ROOT_PATH")
  fi

  if [[ "$NO_ASR" == "1" ]]; then
    common_args+=(--no-asr)
  fi

  case "$SERVER_MODE" in
    api)
      log "Starting OmniVoice API server on ${HOST}:${PORT} ..."
      exec uv run omnivoice-api "${common_args[@]}"
      ;;
    demo)
      if [[ "$SHARE" == "1" ]]; then
        common_args+=(--share)
      fi
      log "Starting OmniVoice demo server on ${HOST}:${PORT} ..."
      exec uv run omnivoice-demo "${common_args[@]}"
      ;;
    none)
      log "Setup complete. OMNIVOICE_SERVER_MODE=none, so no server was started."
      ;;
    *)
      log "Unsupported OMNIVOICE_SERVER_MODE: $SERVER_MODE"
      log "Use one of: api, demo, none"
      return 1
      ;;
  esac
}

main() {
  log "Vast.ai OmniVoice startup beginning ..."
  ensure_uv
  clone_or_update_repo
  run_uv_sync
  launch_server
}

main "$@"
