param([string]$RunId='lv_reuse_s42_v1',[ValidateSet('smoke','reuse')][string]$Mode='reuse',
 [ValidateSet('cpu','cuda')][string]$Device='cuda',[switch]$Resume,
 [ValidateSet('all','prepare','verify')][string]$Stage='all')
$ErrorActionPreference='Stop'
$lvProject=Split-Path $PSScriptRoot -Parent
$lvPython=Join-Path (Split-Path $lvProject -Parent) 'tarp_x_expert_evaluation\.venv\Scripts\python.exe'
Set-Location -LiteralPath $lvProject
$env:PYTHONPATH=Join-Path $lvProject 'src';$env:PYTHONUTF8='1'
$env:OPENBLAS_NUM_THREADS='1';$env:MKL_NUM_THREADS='1';$env:OMP_NUM_THREADS='1'
$lvLock=Join-Path $lvProject 'artifacts\learning_value_v6\active.lock'
New-Item -ItemType Directory -Path (Split-Path $lvLock -Parent) -Force | Out-Null
foreach($lvOther in @($lvLock,(Join-Path $lvProject 'artifacts\sequence_alignment\active.lock'),(Join-Path $lvProject 'artifacts\sequence_interim\active.lock'))){
 if(Test-Path -LiteralPath $lvOther){
  $lvOwner=[int](Get-Content -LiteralPath $lvOther -Raw)
  if(Get-Process -Id $lvOwner -ErrorAction SilentlyContinue){throw 'Another learning workflow is already running'}
 }
}
if(Test-Path -LiteralPath $lvLock){Remove-Item -LiteralPath $lvLock}
$lvHandle=[System.IO.File]::Open($lvLock,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
$lvBytes=[System.Text.Encoding]::UTF8.GetBytes([string]$PID);$lvHandle.Write($lvBytes,0,$lvBytes.Length);$lvHandle.Flush()
try {
 $lvArgs=@('-B','-u','-m','tarp_learning_value_v6.pipeline','--run-id',$RunId,'--mode',$Mode,'--device',$Device,'--stage',$Stage)
 if($Resume){$lvArgs+='--resume'}
 & $lvPython @lvArgs
 if($LASTEXITCODE -ne 0){throw 'Learning-value pipeline failed'}
 if($Stage -eq 'all'){
  & 'D:\anaconda\python.exe' -B -u -m tarp_learning_value_v6.report --root (Join-Path $lvProject "artifacts\learning_value_v6\$RunId")
  if($LASTEXITCODE -ne 0){throw 'Learning-value report failed'}
 }
} finally {$lvHandle.Close();Remove-Item -LiteralPath $lvLock}
