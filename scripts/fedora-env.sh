#!/usr/bin/env bash
# Source this file to activate Python 3.9 and expose the project-local CUDA libs.
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    echo "Use: source scripts/fedora-env.sh"
    exit 1
fi

TFLITE_PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd) || return
if [[ ! -x "$TFLITE_PROJECT_ROOT/.venv/bin/python" ]]; then
    echo "Project environment missing: $TFLITE_PROJECT_ROOT/.venv"
    return 1
fi
source "$TFLITE_PROJECT_ROOT/.venv/bin/activate" || return

TFLITE_CUDA_LIBRARY_PATH=$("$VIRTUAL_ENV/bin/python" - <<'PY'
import pathlib
import site

directories = sorted({str(path) for root in site.getsitepackages()
                      for path in pathlib.Path(root, "nvidia").glob("*/lib")
                      if path.is_dir()})
print(":".join(directories))
PY
) || return
if [[ -n $TFLITE_CUDA_LIBRARY_PATH ]]; then
    export LD_LIBRARY_PATH="$TFLITE_CUDA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export TFLITE_PROJECT_ROOT TFLITE_CUDA_LIBRARY_PATH
