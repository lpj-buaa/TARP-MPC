param(
 [ValidateSet('smoke','pilot','formal')][string]$Mode='smoke',
 [Parameter(Mandatory=$true)][string]$RunId,
 [int]$Workers=1,
 [ValidateSet('audit','collect','analyze')][string]$Until='analyze',
 [switch]$Resume
)
$ErrorActionPreference='Stop'
$project=Split-Path -Parent $PSScriptRoot
$workspace=Split-Path -Parent $project
$python=Join-Path $workspace 'tarp_risk_discovery\.venv\Scripts\python.exe'
$env:PYTHONPATH=Join-Path $project 'src'
Push-Location -LiteralPath $project
try {
 & 'D:\anaconda\python.exe' -B (Join-Path $PSScriptRoot 'export_existing_data.py')
 if ($LASTEXITCODE -ne 0) { throw 'Existing data export failed' }
 $args37=@('-B','-u','-m','tarp37.cli','--mode',$Mode,'--run-id',$RunId,'--workers',"$Workers",'--until',$Until)
 if ($Resume) { $args37+='--resume' }
 & $python @args37
 if ($LASTEXITCODE -ne 0) { throw "Unified 37 failed: $LASTEXITCODE" }
 if ($Until -eq 'analyze') {
  & $python -B -m tarp37.report --data-run $RunId
  if ($LASTEXITCODE -ne 0) { throw 'Physical report generation failed' }
 }
} finally { Pop-Location }
