param([switch]$CheckGpu)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'windows-env.ps1')

Push-Location $TfliteProjectRoot
try {
    $bootstrap = Join-Path $PSScriptRoot 'windows-python.py'
    if ($CheckGpu) {
        & $TflitePython $bootstrap (Join-Path $PSScriptRoot 'check-gpu.py') '--require-gpu'
    } else {
        & $TflitePython $bootstrap (Join-Path $TfliteProjectRoot 'GuiModelMaker__MAIN__.py')
    }
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $exitCode
