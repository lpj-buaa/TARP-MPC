"""Exercise the real Windows workflow launcher without collecting or training."""
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.skipif(os.name != 'nt', reason='Windows PowerShell process lifetime')
@pytest.mark.parametrize('code', [0, 7])
def test_workflow_preserves_child_exit_code(tmp_path, code):
    project = Path(__file__).resolve().parents[1]
    runner = project / 'scripts/run_complete_v4.ps1'
    child = tmp_path / 'exit_probe.ps1'
    child.write_text(f'Start-Sleep -Milliseconds 100\nexit {code}\n', encoding='utf-8')
    harness = tmp_path / 'harness.ps1'
    # Extract only Run-Python via PowerShell's parser; never invoke the workflow.
    harness.write_text('''
$ErrorActionPreference='Stop'
$tokens=$null; $errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile($args[0],[ref]$tokens,[ref]$errors)
$fn=$ast.Find({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Run-Python'},$true)
Invoke-Expression $fn.Extent.Text
$project=$args[1]; $workflow=$args[1]
$state=@{completed=@()}
function Save-State {}
try {
 Run-Python 'probe' (Get-Command powershell.exe).Source @('-NoProfile','-ExecutionPolicy','Bypass','-File',$args[2])
 if ($state.completed -notcontains 'probe') { throw 'Success was not recorded' }
 Write-Output 'observed_success'
} catch {
 Write-Output ('observed_failure: '+$_.Exception.Message)
}
''', encoding='utf-8')
    result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                             '-File', str(harness), str(runner), str(tmp_path), str(child)],
                            text=True, capture_output=True, timeout=40)
    assert result.returncode == 0, result.stderr
    if code == 0:
        assert 'observed_success' in result.stdout, result.stdout
    else:
        assert 'failed, exit 7;' in result.stdout, result.stdout
