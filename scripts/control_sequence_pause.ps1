param([ValidateSet('Inspect','Resume')][string]$Action='Inspect',[string]$RunId='sequence_formal_s42_v1')
$ErrorActionPreference='Stop'
$controlProject=Split-Path $PSScriptRoot -Parent
$controlFile=Join-Path $controlProject "artifacts\sequence_v4\$RunId\pause_state.json"
$controlState=Get-Content -LiteralPath $controlFile -Raw -Encoding UTF8 | ConvertFrom-Json
if($Action -eq 'Inspect'){
 foreach($controlEntry in $controlState.processes){
  $controlObserved=Get-CimInstance Win32_Process -Filter "ProcessId=$($controlEntry.pid)"
  $controlMatches=$controlObserved -and ($controlObserved.CreationDate.ToUniversalTime().ToString('o') -eq $controlEntry.created_at)
  if(-not $controlMatches){[pscustomobject]@{Id=$controlEntry.pid;IdentityMatches=$false;ManifestSuspended=$controlEntry.suspended};continue}
  $controlProcess=Get-Process -Id $controlEntry.pid
  [pscustomobject]@{Id=$controlEntry.pid;IdentityMatches=$true;ManifestSuspended=$controlEntry.suspended;SuspendedThreads=@($controlProcess.Threads | Where-Object {$_.ThreadState -eq 'Wait' -and $_.WaitReason -eq 'Suspended'}).Count;Threads=$controlProcess.Threads.Count}
 }
 exit
}
if($controlState.status -eq 'resumed'){Write-Output 'Already resumed; no second resume call made.';exit}
$controlLock=$controlFile+'.resume.lock'
try {$controlLockHandle=[System.IO.File]::Open($controlLock,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::None)}
catch {throw 'Cannot acquire exclusive resume lock; inspect the owner and pause state before retrying.'}
function Save-ControlState([object]$Value){
 $controlTemp=$controlFile+'.'+$PID+'.tmp'
 [System.IO.File]::WriteAllText($controlTemp,($Value | ConvertTo-Json -Depth 6),(New-Object System.Text.UTF8Encoding($false)))
 [System.IO.File]::Replace($controlTemp,$controlFile,($controlFile+'.previous'))
}
try {
 $controlState=Get-Content -LiteralPath $controlFile -Raw -Encoding UTF8 | ConvertFrom-Json
 if($controlState.status -eq 'resumed'){Write-Output 'Already resumed; no second resume call made.';return}
 if($controlState.status -notin @('paused','resuming')){throw 'Unexpected pause state; inspect before resuming'}
 foreach($controlEntry in $controlState.processes){
  if($controlEntry.resume_phase -eq 'in_flight'){throw 'Ambiguous native resume state; inspect actual suspension before any retry. No native resume was called.'}
  if(-not $controlEntry.suspended){continue}
  $controlObserved=Get-CimInstance Win32_Process -Filter "ProcessId=$($controlEntry.pid)"
  if(-not $controlObserved -or $controlObserved.CreationDate.ToUniversalTime().ToString('o') -ne $controlEntry.created_at -or $controlObserved.CommandLine -ne $controlEntry.command){throw 'Paused process identity changed. Preserve artifacts and use cold recovery; do not resume a reused PID.'}
 }
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class SequenceProcessResume {
 [DllImport("kernel32.dll", SetLastError=true)] public static extern IntPtr OpenProcess(uint access, bool inherit, uint pid);
 [DllImport("kernel32.dll", SetLastError=true)] public static extern bool CloseHandle(IntPtr handle);
 [DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr handle);
}
'@
$controlState.status='resuming';Save-ControlState $controlState
for($controlIndex=$controlState.processes.Count-1;$controlIndex -ge 0;$controlIndex--){
 $controlEntry=$controlState.processes[$controlIndex];if(-not $controlEntry.suspended){continue}
 $controlEntry | Add-Member -NotePropertyName resume_phase -NotePropertyValue 'in_flight' -Force
 Save-ControlState $controlState
 $controlHandle=[SequenceProcessResume]::OpenProcess(0x0800,$false,[uint32]$controlEntry.pid)
 if($controlHandle -eq [IntPtr]::Zero){throw 'Cannot open paused process'}
 try {$controlCode=[SequenceProcessResume]::NtResumeProcess($controlHandle)} finally {[SequenceProcessResume]::CloseHandle($controlHandle) | Out-Null}
 if($controlCode -ne 0){throw "Resume failed with native status $controlCode"}
 $controlEntry.suspended=$false;$controlEntry.resume_phase='complete';Save-ControlState $controlState
}
$controlState.status='resumed';$controlState | Add-Member -NotePropertyName resumed_at -NotePropertyValue (Get-Date).ToString('o') -Force
Save-ControlState $controlState
Write-Output 'Exact paused processes resumed once. The app heartbeat remains paused until explicitly re-enabled.'
} finally {$controlLockHandle.Close();Remove-Item -LiteralPath $controlLock}
