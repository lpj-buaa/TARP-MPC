param(
    [ValidateSet('smoke','formal')][string]$Mode='formal',
    [string]$RunId='evidence_formal_s42_v1',
    [switch]$Resume,
    [ValidateSet('prepare','complete')][string]$Until='complete',
    [string]$ReportPython='D:\anaconda\python.exe'
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
if($Until -eq 'complete'){
    $evidenceRoot=Join-Path $evidenceProject "artifacts\acquisition_evidence_v3\$RunId"
    & $evidencePython -c 'import sys; from tarp_acq_evidence.analysis import analyze; analyze(sys.argv[1])' $evidenceRoot
    if($LASTEXITCODE -ne 0){throw "Evidence analysis failed: $LASTEXITCODE"}
    & $evidencePython (Join-Path $PSScriptRoot 'diagnose_acquisition_evidence_v3.py') --root $evidenceRoot
    if($LASTEXITCODE -ne 0){throw "Evidence diagnostics failed: $LASTEXITCODE"}
    & $ReportPython (Join-Path $PSScriptRoot 'report_acquisition_evidence_v3.py') --root $evidenceRoot --out (Join-Path $evidenceRoot 'report')
    if($LASTEXITCODE -ne 0){throw "Evidence report failed: $LASTEXITCODE"}
}
