param([string]$PythonExecutable)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvRoot = Join-Path $projectRoot '.venv'
$projectPython = Join-Path $venvRoot 'Scripts\python.exe'

function Invoke-ProjectPython {
    param([string[]]$Arguments)
    & $projectPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $projectPython)) {
        if (Test-Path -LiteralPath $venvRoot) {
            throw 'An incompatible .venv already exists. Preserve it or move it before creating the Windows environment.'
        }
        if ($PythonExecutable) {
            $basePython = $PythonExecutable
        } elseif (Get-Command py -ErrorAction SilentlyContinue) {
            $basePython = (& py -3.9 -c 'import sys; print(sys.executable)')
            if ($LASTEXITCODE -ne 0) {
                throw 'Install 64-bit Python 3.9, or pass -PythonExecutable with its full path.'
            }
        } else {
            throw 'Install 64-bit Python 3.9, or pass -PythonExecutable with its full path.'
        }
        & $basePython -c 'import struct,sys; assert sys.version_info[:2] == (3,9) and struct.calcsize(chr(80)) == 8'
        if ($LASTEXITCODE -ne 0) { throw '64-bit Python 3.9 is required.' }
        & $basePython -m venv $venvRoot
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the virtual environment.' }
    }
    Invoke-ProjectPython -Arguments @('-c', 'import struct,sys; assert sys.version_info[:2] == (3,9) and struct.calcsize(chr(80)) == 8')
    $env:PIP_CACHE_DIR = Join-Path $projectRoot '.private_artifacts\pip-cache'
    Invoke-ProjectPython -Arguments @('-m', 'pip', 'install', '--timeout', '180', '--retries', '8', '--progress-bar', 'off', 'pip==25.3', 'setuptools==58.0.0', 'wheel==0.37.1')
    Invoke-ProjectPython -Arguments @('-m', 'pip', 'install', '--timeout', '180', '--retries', '8', '--progress-bar', 'off', '-r', 'requirements-py39.txt', '-r', 'requirements-windows-cuda11.txt')
    Invoke-ProjectPython -Arguments @((Join-Path $PSScriptRoot 'fetch-cudnn-windows.py'))
    Invoke-ProjectPython -Arguments @('-m', 'pip', 'check')
    . (Join-Path $PSScriptRoot 'windows-env.ps1')
    Invoke-ProjectPython -Arguments @((Join-Path $PSScriptRoot 'windows-python.py'), (Join-Path $PSScriptRoot 'check-gpu.py'), '--require-gpu')
    Write-Host 'Windows project setup and GPU checks completed. Use scripts\run-windows.ps1 to launch.'
} finally {
    Pop-Location
}
