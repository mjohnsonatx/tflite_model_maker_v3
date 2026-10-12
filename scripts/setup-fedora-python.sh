#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$project_root"
python="$project_root/.venv/bin/python"
if [[ ! -x $python ]]; then
    echo "Create this project's Python 3.9 virtual environment first."
    exit 1
fi
"$python" -c 'import sys; assert sys.version_info[:2] == (3, 9), "Python 3.9 required"'
export PIP_CACHE_DIR="$project_root/.private_artifacts/pip-cache"
"$python" -m pip install --timeout 180 --retries 8 --progress-bar off 'pip==25.3'
"$python" -m pip install --timeout 180 --retries 8 --progress-bar off \
    -r requirements-fedora-py39.txt -r requirements-fedora-cuda11.txt
"$python" -m pip check
echo "Project packages installed. Use scripts/run-fedora.sh to launch the app."
