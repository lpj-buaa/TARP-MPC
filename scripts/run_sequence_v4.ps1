param(
 [ValidateSet('smoke','formal')][string]$Mode='formal',
 [string]$RunId='sequence_formal_s42_v1',
 [ValidateSet('all','finish','learn')][string]$Stage='all',
 [ValidateSet('cpu','cuda')][string]$Device='cuda',
 [int]$Workers=4,
 [switch]$Resume
)
$ErrorActionPreference='Stop'
$sequenceProject=Split-Path $PSScriptRoot -Parent
$sequenceWorkspace=Split-Path $sequenceProject -Parent
$sequencePhysical=Join-Path $sequenceWorkspace 'tarp_risk_discovery\.venv\Scripts\python.exe'
$sequenceLearning=Join-Path $sequenceWorkspace 'tarp_x_expert_evaluation\.venv\Scripts\python.exe'
$sequenceRoot=Join-Path $sequenceProject "artifacts\sequence_v4\$RunId"
Set-Location -LiteralPath $sequenceProject
$env:PYTHONPATH=Join-Path $sequenceProject 'src';$env:PYTHONUTF8='1'
$sequenceLockFolder=Join-Path $sequenceProject 'artifacts\sequence_v4\.locks'
New-Item -ItemType Directory -Path $sequenceLockFolder -Force | Out-Null
$sequenceLock=Join-Path $sequenceLockFolder "$RunId.lock"
if(Test-Path -LiteralPath $sequenceLock){
 $sequenceOwner=[int](Get-Content -LiteralPath $sequenceLock -Raw)
 if(Get-Process -Id $sequenceOwner -ErrorAction SilentlyContinue){throw 'This workflow is already running; do not launch a duplicate GPU/collector process.'}
 Remove-Item -LiteralPath $sequenceLock
}
$sequenceHandle=[System.IO.File]::Open($sequenceLock,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
$sequenceBytes=[System.Text.Encoding]::UTF8.GetBytes([string]$PID)
$sequenceHandle.Write($sequenceBytes,0,$sequenceBytes.Length);$sequenceHandle.Flush()
try {
if($Stage -eq 'all'){
 $sequenceRuntime=Join-Path $sequenceRoot 'collector_runtime.json'
 if(Test-Path -LiteralPath $sequenceRuntime){
  $sequenceExisting=Get-Content -LiteralPath $sequenceRuntime -Raw | ConvertFrom-Json
  if(Get-Process -Id $sequenceExisting.pid -ErrorAction SilentlyContinue){throw 'This collector is already active. Use -Stage finish to follow it.'}
 }
 if($Resume -and (Test-Path -LiteralPath (Join-Path $sequenceRoot 'inventory.json'))){
  # NativeTraffic.copytree requires a fresh runtime directory. Preserve every
  # interrupted runtime and log; only sealed record/measurement files are reused.
  $sequenceRootAbsolute=[System.IO.Path]::GetFullPath($sequenceRoot)
  $sequenceInventory=Get-Content -LiteralPath (Join-Path $sequenceRoot 'inventory.json') -Raw | ConvertFrom-Json
  foreach($sequenceContext in $sequenceInventory){
   $sequencePending=@($sequenceContext.items | Where-Object {-not (Test-Path -LiteralPath (Join-Path (Split-Path $_.path -Parent) 'complete.json'))})
   if($sequencePending.Count -eq 0){continue}
   $sequenceSimulation=[System.IO.Path]::GetFullPath((Join-Path $sequenceRoot "contexts\$($sequenceContext.key)\simulation"))
   if(-not $sequenceSimulation.StartsWith($sequenceRootAbsolute+[System.IO.Path]::DirectorySeparatorChar,[System.StringComparison]::OrdinalIgnoreCase)){throw 'Runtime archive path is outside the run directory'}
   if(Test-Path -LiteralPath $sequenceSimulation){
    $sequenceArchive=$sequenceSimulation+'_previous_'+(Get-Date).ToString('yyyyMMdd_HHmmss_fff')
    Move-Item -LiteralPath $sequenceSimulation -Destination $sequenceArchive
   }
   foreach($sequenceItem in $sequencePending){
    $sequenceLog=Join-Path (Split-Path $sequenceItem.path -Parent) 'native.log'
    if(Test-Path -LiteralPath $sequenceLog){Move-Item -LiteralPath $sequenceLog -Destination ($sequenceLog+'.previous_'+(Get-Date).ToString('yyyyMMdd_HHmmss_fff'))}
   }
  }
 }
 $sequenceArgs=@('-B','-u','-m','tarp_sequence.corpus','--run-id',$RunId,'--mode',$Mode,'--workers',$Workers)
 if($Resume){$sequenceArgs+='--resume'}
 & $sequencePhysical @sequenceArgs
 if($LASTEXITCODE -ne 0){throw 'Sequence physical collection failed'}
}
if($Stage -eq 'finish'){
 while(-not (Test-Path -LiteralPath (Join-Path $sequenceRoot 'collection_complete.json'))){
  Start-Sleep -Seconds 5
  $sequenceRuntime=Join-Path $sequenceRoot 'collector_runtime.json'
  if(Test-Path -LiteralPath $sequenceRuntime){
   $sequenceExisting=Get-Content -LiteralPath $sequenceRuntime -Raw | ConvertFrom-Json
   if(-not (Get-Process -Id $sequenceExisting.pid -ErrorAction SilentlyContinue)){
    if(-not (Test-Path -LiteralPath (Join-Path $sequenceRoot 'collection_complete.json'))){throw 'Collector stopped before completion; preserve outputs and inspect its log.'}
   }
  }
 }
}
& $sequenceLearning -B (Join-Path $PSScriptRoot 'verify_sequence_v4.py') --root $sequenceRoot
if($LASTEXITCODE -ne 0){throw 'Sequence frozen-input verification failed'}
& $sequencePhysical -B -u -m tarp_sequence.cli --run-id $RunId --mode $Mode --stage pack
if($LASTEXITCODE -ne 0){throw 'Sequence dataset packing failed'}
& $sequenceLearning -B -u -m tarp_sequence.cli --run-id $RunId --mode $Mode --stage all --device $Device
if($LASTEXITCODE -ne 0){throw 'Sequence learning/evaluation failed'}
& $sequenceLearning -B (Join-Path $PSScriptRoot 'diagnose_sequence_v4.py') --root $sequenceRoot
if($LASTEXITCODE -ne 0){throw 'Sequence component diagnostics failed'}
& 'D:\anaconda\python.exe' (Join-Path $PSScriptRoot 'report_sequence_v4.py') --root $sequenceRoot
if($LASTEXITCODE -ne 0){throw 'Sequence report failed'}
& $sequenceLearning -B (Join-Path $PSScriptRoot 'verify_sequence_v4.py') --root $sequenceRoot
if($LASTEXITCODE -ne 0){throw 'Sequence final verification failed'}
@{status='complete';run_id=$RunId;finished_at=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $sequenceRoot 'workflow_status.json') -Encoding utf8
} catch {
 if(Test-Path -LiteralPath $sequenceRoot){@{status='failed';run_id=$RunId;error=$_.Exception.Message;time=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $sequenceRoot 'workflow_status.json') -Encoding utf8}
 throw
} finally {
 $sequenceHandle.Close()
 Remove-Item -LiteralPath $sequenceLock
}
