#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/fedora-env.sh"
cd -- "$TFLITE_PROJECT_ROOT"
if [[ $# -eq 0 ]]; then
    exec "$VIRTUAL_ENV/bin/python" GuiModelMaker__MAIN__.py
fi
exec "$VIRTUAL_ENV/bin/python" "$@"
