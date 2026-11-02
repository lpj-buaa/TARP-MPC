param(
    [ValidateSet('smoke','formal')][string]$Mode='formal',
    [string]$RunId='evidence_formal_s42_v1',
    [switch]$Resume,
    [ValidateSet('prepare','complete')][string]$Until='complete'
)
$ErrorActionPreference='Stop'
$evidenceProject=Split-Path $PSScriptRoot -Parent
$evidencePython=Join-Path (Split-Path $evidenceProject -Parent) 'tarp_x_expert_evaluation\.venv\Scripts\python.exe'
Set-Location -LiteralPath $evidenceProject
$env:PYTHONPATH=Join-Path $evidenceProject 'src'
$env:PYTHONUTF8='1'
$evidenceArgs=@('-u','-m','tarp_acq_evidence.pipeline','--mode',$Mode,'--run-id',$RunId,'--until',$Until)
if($Resume){$evidenceArgs+='--resume'}
& $evidencePython @evidenceArgs
if($LASTEXITCODE -ne 0){throw "Acquisition evidence failed: $LASTEXITCODE"}
