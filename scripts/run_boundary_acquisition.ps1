param(
    [string]$RunId='boundary_formal_s42_v2',
    [ValidateSet('all','finish','fit')][string]$Stage='all',
    [int]$Workers=4
)
$ErrorActionPreference='Stop'
$bdProject=Split-Path -Parent $PSScriptRoot
$bdWorkspace=Split-Path -Parent $bdProject
$bdSource=Join-Path $bdWorkspace 'tarp_37_risk_audit'
$bdRoot=Join-Path $bdProject "artifacts\boundary_v2\$RunId"
$bdPhysical=Join-Path $bdWorkspace 'tarp_risk_discovery\.venv\Scripts\python.exe'
$bdLearning=Join-Path $bdWorkspace 'tarp_x_expert_evaluation\.venv\Scripts\python.exe'
Push-Location $bdProject
try {
    $env:PYTHONPATH=Join-Path $bdProject 'src'
    & $bdPhysical -B .\scripts\verify_boundary.py --root $bdRoot
    if ($LASTEXITCODE -ne 0) {throw 'Published output verification failed'}
    if ($Stage -eq 'all') {
        & $bdPhysical -B -u -m tarp_boundary.corpus --source $bdSource --root $bdRoot --workers $Workers
        if ($LASTEXITCODE -ne 0) {throw 'Native dose corpus failed'}
        & $bdPhysical -B -u -m tarp_boundary.measurements --root $bdRoot --workers $Workers
        if ($LASTEXITCODE -ne 0) {throw 'MPC domain measurement failed'}
    }
    if ($Stage -eq 'finish') {
        # Join already active native/domain jobs; never starts duplicate collection.
        while (-not ((Test-Path -LiteralPath (Join-Path $bdRoot 'collection_complete.json')) -and (Test-Path -LiteralPath (Join-Path $bdRoot 'measurement_complete.json')))) {
            Start-Sleep -Seconds 5
        }
    }
    & $bdLearning -B -u .\scripts\boundary_teacher.py --root $bdRoot --source $bdSource
    if ($LASTEXITCODE -ne 0) {throw 'Frozen teacher adapter failed'}
    & $bdLearning -B -u -m tarp_boundary.cli bundle --run-id $RunId --source $bdSource
    if ($LASTEXITCODE -ne 0) {throw 'Boundary Oracle packing failed'}
    & $bdLearning -B -u -m tarp_boundary.cli fit --run-id $RunId --source $bdSource
    if ($LASTEXITCODE -ne 0) {throw 'Boundary acquisition failed'}
    & $bdLearning -B .\scripts\diagnose_boundary_value.py --run-id $RunId
    if ($LASTEXITCODE -ne 0) {throw 'Acquisition-value diagnostics failed'}
    & 'D:\anaconda\python.exe' -B .\scripts\analyze_boundary.py --run-id $RunId
    if ($LASTEXITCODE -ne 0) {throw 'Boundary analysis failed'}
} finally {Pop-Location}
