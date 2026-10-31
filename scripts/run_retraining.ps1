param(
 [ValidateSet('pilot','formal')][string]$Mode='pilot',
 [Parameter(Mandatory=$true)][string]$DataRunId,
 [Parameter(Mandatory=$true)][string]$RunId,
 [ValidateSet('pack_initial','fit_initial','explore','pack_augmented','fit_experts','predict','fit_fusion','calibrate','evaluate','all')][string]$Task='all',
 [string]$Device='cuda',
 [ValidateSet('td3','sobol','random')][string]$Policy='td3',
 [int]$Queries=0,
 [int]$Seed=42,
 [switch]$SkipRL,
 [switch]$Resume
)
$ErrorActionPreference='Stop'
$project=Split-Path -Parent $PSScriptRoot
$workspace=Split-Path -Parent $project
$python=Join-Path $workspace 'tarp_x_expert_evaluation\.venv\Scripts\python.exe'
$env:PYTHONPATH=Join-Path $project 'src'
Push-Location -LiteralPath $project
try {
 $trainArgs=@('-B','-u','-m','tarp37.retrain_cli','--mode',$Mode,'--data-run',$DataRunId,'--run-id',$RunId,'--task',$Task,'--device',$Device,'--policy',$Policy,'--seed',"$Seed")
 if ($Queries -gt 0) { $trainArgs+=@('--queries',"$Queries") }
 if ($SkipRL) { $trainArgs+='--skip-rl' }
 if ($Resume) { $trainArgs+='--resume' }
 & $python @trainArgs
 if ($LASTEXITCODE -ne 0) { throw "Unified retraining failed: $LASTEXITCODE" }
} finally { Pop-Location }
