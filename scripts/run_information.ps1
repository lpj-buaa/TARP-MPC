param([ValidateSet('smoke','pilot')][string]$Mode='pilot',[string]$RunId='information_pilot_s42_v1',[switch]$Resume)
$ErrorActionPreference='Stop'
$infoProject=Split-Path -Parent $PSScriptRoot
$infoWorkspace=Split-Path -Parent $infoProject
$infoSource=Join-Path $infoWorkspace 'tarp_37_risk_audit'
$infoPython=Join-Path $infoWorkspace 'tarp_x_expert_evaluation\.venv\Scripts\python.exe'
$infoRoot=Join-Path $infoProject "artifacts\information\$RunId"
New-Item -ItemType Directory -Force -Path $infoRoot | Out-Null
Push-Location $infoProject
try {
    if (-not (Test-Path -LiteralPath (Join-Path $infoRoot 'cache.json'))) {
        $env:PYTHONPATH=Join-Path $infoSource 'src'
        & $infoPython -B -u .\scripts\prepare_info_utility.py --source $infoSource --output (Join-Path $infoRoot 'utility.json')
        if ($LASTEXITCODE -ne 0) { throw 'Frozen V4 forward adapter failed' }
        $env:PYTHONPATH=Join-Path $infoProject 'src'
        & $infoPython -B -u -m tarp_info.cli prepare --source $infoSource --run-id $RunId
        if ($LASTEXITCODE -ne 0) { throw 'Shared query-cache preparation failed' }
    }
    $env:PYTHONPATH=Join-Path $infoProject 'src'
    $infoBudget=if ($Mode -eq 'smoke') {8} else {128}
    $infoSeeds=if ($Mode -eq 'smoke') {'42'} else {'42,43,44'}
    $infoArgs=@('-B','-u','-m','tarp_info.cli','replay','--source',$infoSource,'--run-id',$RunId,'--budget',"$infoBudget",'--seeds',$infoSeeds)
    if ($Resume) {$infoArgs+='--resume'}
    & $infoPython @infoArgs
    if ($LASTEXITCODE -ne 0) { throw 'Acquisition replay failed' }
} finally {Pop-Location}
