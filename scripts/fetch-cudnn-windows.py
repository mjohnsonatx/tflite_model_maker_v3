"""Install verified cuDNN 8.9.6 CUDA 11 DLLs inside the Windows virtual environment."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import urllib.error
import urllib.request
import zipfile

BASE_URL = "https://developer.download.nvidia.com/compute/cudnn/redist/"
MANIFEST_URL = BASE_URL + "redistrib_8.9.6.json"
EXPECTED_SHA256 = "f7a013f9181c863d68e67083813ecc87c0f781ef7481467fa20095373798de50"
REQUIRED_DLLS = {
    "cudnn64_8.dll", "cudnn_adv_infer64_8.dll", "cudnn_adv_train64_8.dll",
    "cudnn_cnn_infer64_8.dll", "cudnn_cnn_train64_8.dll",
    "cudnn_ops_infer64_8.dll", "cudnn_ops_train64_8.dll",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, archive):
    if archive.exists() and sha256(archive) == EXPECTED_SHA256:
        return
    partial = archive.with_suffix(".zip.part")
    if partial.exists() and sha256(partial) == EXPECTED_SHA256:
        partial.replace(archive)
        return
    for attempt in range(1, 5):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": "bytes=" + str(offset) + "-"} if offset else {}
        request = urllib.request.Request(url, headers=headers)
        print("Downloading cuDNN archive (attempt " + str(attempt) + ")", flush=True)
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                resume = offset and response.status == 206
                with partial.open("ab" if resume else "wb") as stream:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
        except (OSError, urllib.error.URLError, http.client.IncompleteRead) as error:
            if attempt == 4:
                raise
            print("Download interrupted: " + str(error), flush=True)
            continue
        if sha256(partial) != EXPECTED_SHA256:
            partial.unlink()
            raise RuntimeError("cuDNN archive SHA-256 did not match NVIDIA's manifest.")
        partial.replace(archive)
        return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="Check the manifest without installing.")
    arguments = parser.parse_args()
    with urllib.request.urlopen(MANIFEST_URL, timeout=60) as response:
        manifest = json.load(response)
    artifact = manifest["cudnn"]["windows-x86_64"]["cuda11"]
    relative_path = artifact["relative_path"]
    if not relative_path.startswith("cudnn/windows-x86_64/") or not relative_path.endswith("_cuda11-archive.zip"):
        raise RuntimeError("Unexpected cuDNN archive path.")
    if artifact["sha256"] != EXPECTED_SHA256:
        raise RuntimeError("NVIDIA's archive checksum changed; review before installing.")
    if arguments.describe:
        print(json.dumps({"url": BASE_URL + relative_path, "sha256": EXPECTED_SHA256}, indent=2))
        return
    if os.name != "nt" or sys.prefix == sys.base_prefix:
        raise RuntimeError("Run this installer inside the project's Windows virtual environment.")
    project_root = Path(__file__).resolve().parent.parent
    cache = project_root / ".private_artifacts" / "nvidia-downloads"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / PurePosixPath(relative_path).name
    download(BASE_URL + relative_path, archive)
    destination = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cudnn"
    with zipfile.ZipFile(archive) as bundle:
        dlls = {PurePosixPath(name).name: name for name in bundle.namelist()
                if PurePosixPath(name).parent.name == "bin" and name.endswith(".dll")}
        if not REQUIRED_DLLS.issubset(dlls):
            raise RuntimeError("The cuDNN archive is missing required runtime DLLs.")
        (destination / "bin").mkdir(parents=True, exist_ok=True)
        for name, member in dlls.items():
            with bundle.open(member) as source, (destination / "bin" / name).open("wb") as target:
                shutil.copyfileobj(source, target)
        for member in bundle.namelist():
            if PurePosixPath(member).name in {"LICENSE", "LICENSE.txt"}:
                with bundle.open(member) as source, (destination / PurePosixPath(member).name).open("wb") as target:
                    shutil.copyfileobj(source, target)
    print("Installed verified cuDNN 8.9.6.50 DLLs into " + str(destination))


if __name__ == "__main__":
    main()
