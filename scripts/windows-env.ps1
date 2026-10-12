# Dot-source this file to activate the project environment in PowerShell.
$TfliteProjectRoot = Split-Path -Parent $PSScriptRoot
$TflitePython = Join-Path $TfliteProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $TflitePython)) {
    throw 'Run scripts\setup-windows.ps1 to create the Windows environment first.'
}

. (Join-Path $TfliteProjectRoot '.venv\Scripts\Activate.ps1')
$nvidiaRoot = Join-Path $TfliteProjectRoot '.venv\Lib\site-packages\nvidia'
$dllDirectories = @()
if (Test-Path -LiteralPath $nvidiaRoot) {
    $dllDirectories = @(Get-ChildItem -LiteralPath $nvidiaRoot -Recurse -File -Filter '*.dll' |
        Select-Object -ExpandProperty DirectoryName -Unique | Sort-Object)
}
$env:TFLITE_CUDA_LIBRARY_PATH = $dllDirectories -join ';'
if ($dllDirectories.Count -gt 0) {
    $env:PATH = "$($env:TFLITE_CUDA_LIBRARY_PATH);$($env:PATH)"
}
$env:TFLITE_PROJECT_ROOT = $TfliteProjectRoot
