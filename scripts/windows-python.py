"""Register Windows CUDA DLL directories before running an application script."""
import os
from pathlib import Path
import runpy
import sys


def main():
    if os.name != "nt":
        raise RuntimeError("Use scripts/run-fedora.sh on Linux.")
    if len(sys.argv) < 2:
        raise RuntimeError("An application script path is required.")
    directories = os.environ.get("TFLITE_CUDA_LIBRARY_PATH", "").split(os.pathsep)
    dll_handles = [os.add_dll_directory(directory) for directory in directories
                   if directory and Path(directory).is_dir()]
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script)] + sys.argv[2:]
    sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        for handle in dll_handles:
            handle.close()


if __name__ == "__main__":
    main()
