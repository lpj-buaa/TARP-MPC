param([string]$RunId='align_reuse_s42_v1',[ValidateSet('smoke','reuse')][string]$Mode='reuse',
 [ValidateSet('cpu','cuda')][string]$Device='cuda',[switch]$Resume,
 [ValidateSet('all','prepare','fit','evaluate','analyze','verify')][string]$Stage='all')
$ErrorActionPreference='Stop'
$alignProject=Split-Path $PSScriptRoot -Parent
$alignPython=Join-Path (Split-Path $alignProject -Parent) 'tarp_x_expert_evaluation\.venv\Scripts\python.exe'
Set-Location -LiteralPath $alignProject
$env:PYTHONPATH=Join-Path $alignProject 'src';$env:PYTHONUTF8='1'
$alignLock=Join-Path $alignProject 'artifacts\sequence_alignment\active.lock'
New-Item -ItemType Directory -Path (Split-Path $alignLock -Parent) -Force | Out-Null
if(Test-Path -LiteralPath $alignLock){
 $alignOwner=[int](Get-Content -LiteralPath $alignLock -Raw)
 if(Get-Process -Id $alignOwner -ErrorAction SilentlyContinue){throw 'Alignment workflow already running'}
 Remove-Item -LiteralPath $alignLock
}
$alignHandle=[System.IO.File]::Open($alignLock,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
$alignBytes=[System.Text.Encoding]::UTF8.GetBytes([string]$PID);$alignHandle.Write($alignBytes,0,$alignBytes.Length);$alignHandle.Flush()
try {
 $alignArgs=@('-B','-u','-m','tarp_alignment_v5.pipeline','--run-id',$RunId,'--mode',$Mode,'--device',$Device,'--stage',$Stage)
 if($Resume){$alignArgs+='--resume'}
 & $alignPython @alignArgs
 if($LASTEXITCODE -ne 0){throw 'Alignment workflow failed'}
 if($Stage -eq 'all'){
  & 'D:\anaconda\python.exe' -B -u -m tarp_alignment_v5.report --root (Join-Path $alignProject "artifacts\sequence_alignment\$RunId")
  if($LASTEXITCODE -ne 0){throw 'Alignment report failed'}
 }
} finally {$alignHandle.Close();Remove-Item -LiteralPath $alignLock}
