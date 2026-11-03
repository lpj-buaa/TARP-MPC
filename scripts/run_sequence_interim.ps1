param([string]$RunId='sequence_interim_s42_v1',[ValidateSet('cpu','cuda')][string]$Device='cuda',[switch]$Resume)
$ErrorActionPreference='Stop'
$interimProject=Split-Path $PSScriptRoot -Parent
$interimPython=Join-Path (Split-Path $interimProject -Parent) 'tarp_x_expert_evaluation\.venv\Scripts\python.exe'
$interimRoot=Join-Path $interimProject "artifacts\sequence_interim\$RunId"
Set-Location -LiteralPath $interimProject
$env:PYTHONPATH=Join-Path $interimProject 'src';$env:PYTHONUTF8='1'
$interimLock=Join-Path $interimProject 'artifacts\sequence_interim\active.lock'
New-Item -ItemType Directory -Path (Split-Path $interimLock -Parent) -Force | Out-Null
if(Test-Path -LiteralPath $interimLock){
 $interimOwner=[int](Get-Content -LiteralPath $interimLock -Raw)
 if(Get-Process -Id $interimOwner -ErrorAction SilentlyContinue){throw 'An interim GPU workflow is already running'}
 Remove-Item -LiteralPath $interimLock
}
$interimHandle=[System.IO.File]::Open($interimLock,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
$interimBytes=[System.Text.Encoding]::UTF8.GetBytes([string]$PID);$interimHandle.Write($interimBytes,0,$interimBytes.Length);$interimHandle.Flush()
try {
 $interimArgs=@('-B','-u','-m','tarp_interim.pipeline','--run-id',$RunId,'--device',$Device,'--stage','all')
 if($Resume){$interimArgs+='--resume'}
 & $interimPython @interimArgs
 if($LASTEXITCODE -ne 0){throw 'Interim learning/evaluation failed'}
 & 'D:\anaconda\python.exe' -B -u -m tarp_interim.report --root $interimRoot
 if($LASTEXITCODE -ne 0){throw 'Interim report failed'}
 & $interimPython -B -m tarp_interim.pipeline --run-id $RunId --device $Device --stage verify --resume
 if($LASTEXITCODE -ne 0){throw 'Interim final hash verification failed'}
 @{status='complete';scope='interim development only';original_confirm_used=0;time=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $interimRoot 'workflow_status.json') -Encoding utf8
} catch {
 if(Test-Path -LiteralPath $interimRoot){@{status='failed';error=$_.Exception.Message;time=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $interimRoot 'workflow_status.json') -Encoding utf8}
 throw
} finally {$interimHandle.Close();Remove-Item -LiteralPath $interimLock}
