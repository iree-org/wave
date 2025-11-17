#!/usr/bin/env bash
set -euxo pipefail

source /entrypoint.sh >/dev/null 2>&1

if [ -z "${VENV_DIR:-}" ]; then
  echo "VENV_DIR is not set"
  exit 1
fi
source "${VENV_DIR}/bin/activate"

export CARGO_HOME="$HOME/.cargo"
export PATH="$CARGO_HOME/bin:$PATH"
