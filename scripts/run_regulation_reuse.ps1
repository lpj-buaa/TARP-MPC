param(
 [string]$RunId='routine_q_reuse_v5_s42_20261101',
 [int]$Workers=4,
 [int]$MaxSources=0,
 [switch]$Resume
)
$ErrorActionPreference='Stop'
$project=Split-Path -Parent $PSScriptRoot
$workspace=Split-Path -Parent $project
$python=Join-Path $workspace 'tarp_risk_discovery\.venv\Scripts\python.exe'
$source=Join-Path $workspace 'tarp_37_risk_audit\artifacts\runs\risk37_formal_v4_s42_20261101'
$env:PYTHONPATH=Join-Path $project 'src'
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONIOENCODING='utf-8'
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'
$cliArgs=@('-B','-u','-m','tarp37.regulation_reuse','--source-run',$source,'--run-id',$RunId,'--workers',"$Workers",'--max-sources',"$MaxSources")
if ($Resume) { $cliArgs+='--resume' }
& $python @cliArgs
if ($LASTEXITCODE -ne 0) { throw "Normal-Q reuse failed with exit code $LASTEXITCODE" }
