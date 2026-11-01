param(
 [string]$DataRunId='risk37_formal_v4_s42_20261101',
 [string]$WorkflowId='full_v4_s42_20261101',
 [switch]$ValidateOnly
)
$ErrorActionPreference='Stop'
$project=Split-Path -Parent $PSScriptRoot
$workspace=Split-Path -Parent $project
$physical=Join-Path $workspace 'tarp_risk_discovery\.venv\Scripts\python.exe'
$training=Join-Path $workspace 'tarp_x_expert_evaluation\.venv\Scripts\python.exe'
$dataRoot=Join-Path $project "artifacts\runs\$DataRunId"
$workflow=Join-Path $project "artifacts\workflows\$WorkflowId"
foreach ($id in @($DataRunId,$WorkflowId)) {
 if ($id -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]{0,100}$') { throw 'Invalid run identity' }
}
foreach ($path in @($physical,$training,(Join-Path $dataRoot 'audit\complete.json'))) {
 if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required input missing: $path" }
}
$env:PYTHONPATH=Join-Path $project 'src'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
if ($ValidateOnly) {
 [pscustomobject]@{project=$project;physical=$physical;training=$training;data_run=$DataRunId;workflow=$workflow;training_started=$false} | ConvertTo-Json
 exit 0
}
New-Item -ItemType Directory -Path $workflow -Force | Out-Null
$statusPath=Join-Path $workflow 'status.json'
if (Test-Path -LiteralPath $statusPath) {
 $previous=Get-Content -LiteralPath $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json
 if ($previous.pid -and $previous.pid -ne $PID -and (Get-Process -Id $previous.pid -ErrorAction SilentlyContinue)) {
  throw "Workflow runner is already active: $($previous.pid)"
 }
}
$state=[ordered]@{workflow=$WorkflowId;data_run=$DataRunId;pid=$PID;status='running';stage='wait_collection';completed=@();training_runs=@();updated_utc=''}
function Save-State {
 $state.updated_utc=[DateTimeOffset]::UtcNow.ToString('o')
 $temporary=Join-Path $workflow 'status.pending.json'
 $state | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $temporary -Encoding UTF8
 Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}
function Run-Python([string]$Name,[string]$Python,[string[]]$PythonArgs) {
 $state.stage=$Name; $state.status='running'; Save-State
 $stamp=[DateTimeOffset]::UtcNow.ToString('yyyyMMdd_HHmmss')
 $stdout=Join-Path $workflow ($Name+'_'+$stamp+'.out.log')
 $stderr=Join-Path $workflow ($Name+'_'+$stamp+'.err.log')
 $state.stdout=$stdout; $state.stderr=$stderr; Save-State
 $child=Start-Process -FilePath $Python -ArgumentList $PythonArgs -WorkingDirectory $project -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
 # Retain the native handle before waiting: Windows PowerShell 5.1 can
 # otherwise report a null ExitCode for a completed Start-Process child.
 $childHandle=$child.Handle
 $state.child_pid=$child.Id; Save-State
 $child.WaitForExit()
 if ($child.ExitCode -ne 0) { throw "$Name failed, exit $($child.ExitCode); see $stderr and $stdout" }
 $state.completed+=@($Name); $state.child_pid=$null; Save-State
}
try {
 Save-State
 $summary=Join-Path $dataRoot 'analysis\summary.json'
 while (-not (Test-Path -LiteralPath $summary -PathType Leaf)) {
  $collectors=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*tarp37.cli*$DataRunId*" })
  if ($collectors.Count -eq 0) { throw 'Physical collector stopped before publishing its summary; retain source logs and resume diagnosis.' }
  $markers=@(Get-ChildItem -LiteralPath (Join-Path $dataRoot 'sources') -Filter complete.json -Recurse -ErrorAction SilentlyContinue)
  $state.completed_sources=$markers.Count; Save-State
  Start-Sleep -Seconds 30
 }
 $physicalSummary=Get-Content -LiteralPath $summary -Raw -Encoding UTF8 | ConvertFrom-Json
 if ($physicalSummary.sources -ne 28 -or $physicalSummary.branches -ne 896) { throw 'Formal collection does not match the frozen 28-source/896-branch protocol.' }
 Run-Python 'audit_physical' $physical @('-B',(Join-Path $PSScriptRoot 'check_risk_chain.py'),'--run-id',$DataRunId)
 Run-Python 'report_physical' $physical @('-B','-m','tarp37.report','--data-run',$DataRunId)
 foreach ($policy in @('td3','noacq','sobol','random')) {
  $runId='risk37_'+$policy+'_v4_s42_20261101'
  $arguments=@('-B','-u','-m','tarp37.retrain_cli','--mode','formal','--data-run',$DataRunId,'--run-id',$runId,'--task','all','--device','cuda','--seed','42','--resume')
  if ($policy -eq 'noacq') { $arguments+='--skip-rl' }
  else { $arguments+=@('--policy',$policy,'--queries','128') }
  Run-Python ('train_'+$policy) $training $arguments
  $runRoot=Join-Path $project "artifacts\retraining\$runId"
  $metrics=Get-Content -LiteralPath (Join-Path $runRoot 'evaluate\metrics.json') -Raw -Encoding UTF8 | ConvertFrom-Json
  $cal=Get-Content -LiteralPath (Join-Path $runRoot 'calibrate\thresholds.json') -Raw -Encoding UTF8 | ConvertFrom-Json
  $state.training_runs+=@([ordered]@{policy=$policy;run_id=$runId;root=$runRoot;full=$metrics.full;fusion=$metrics.evidence_fusion;best_single=$metrics.best_single;union=$metrics.union;calibration=$cal})
  Save-State
 }
 $state.status='complete'; $state.stage='all_complete'; Save-State
 $state | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $workflow 'complete.json') -Encoding UTF8
} catch {
 $state.status='failed'; $state.error=$_.Exception.Message; Save-State
 throw
}
