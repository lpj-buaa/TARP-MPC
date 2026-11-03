"""Export frozen, explicit research inputs; never capture a live simulation directory."""
from pathlib import Path
import argparse, hashlib, json, subprocess, zipfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def export(project, output):
    project, output = Path(project).resolve(), Path(output).resolve()
    workspace = project.parent
    native = Path('D:/tarp_ms_single_flow_test/external_repos/v2sim').resolve()
    protocol = json.loads((project/'artifacts/sequence_v4/sequence_formal_s42_v1/physical_protocol.json').read_text('utf-8'))
    files = {Path(p).resolve() for p in protocol['inputs'] | protocol['code']}
    for folder in [native, workspace/'tarp_v2sim_native_grid/src', workspace/'tarp_v2sim_engineering_grid/artifacts/normal_ev_5000_s42_v1/case']:
        files.update(p.resolve() for p in folder.rglob('*') if p.is_file() and not set(p.parts) & {'.git', '__pycache__'})
    files.update((workspace/'tarp_v2sim_traffic_diagnosis').glob('*.py'))
    profile = workspace/'tarp_v2sim_engineering_grid/artifacts/normal_ev_5000_s42_v1'
    files.update(p for p in profile.glob('*.json'))
    files.add(workspace/'tarp_37_risk_audit/artifacts/retraining/risk37_td3_v4_s42_20261101/pack_initial/schema.json')
    for name in ['boundary_v2/boundary_formal_s42_v2', 'acquisition_evidence_v3/evidence_formal_s42_v1', 'sequence_v4/sequence_cache_pilot_s42_v1']:
        folder = project/'artifacts'/name
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix in {'.json','.pt','.pth','.npz','.pkl','.py','.html','.md','.png','.svg'} and not set(p.relative_to(folder).parts) & {'simulation','traffic','__pycache__'}:
                files.add(p.resolve())
    # Keep stable archived outputs in their original locations. Do not read the live formal bank.
    output.mkdir(parents=True, exist_ok=True)
    archive = output/'frozen-inputs-and-completed-evidence-2026-11-03.zip'
    if archive.exists():
        raise FileExistsError(archive)
    entries = []
    for p in sorted(files):
        p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        if p.is_relative_to(workspace):
            name = 'workspace/'+p.relative_to(workspace).as_posix()
        elif p.is_relative_to(native):
            name = 'v2sim/'+p.relative_to(native).as_posix()
        else:
            raise ValueError('Unapproved export root: '+str(p))
        if p.name.startswith('.env') or p.suffix in {'.pem','.key'}:
            raise ValueError('Credential-like file rejected: '+str(p))
        entries.append(dict(path=name, original_path=str(p), bytes=p.stat().st_size, sha256=digest(p)))
    expected = {str(Path(p).resolve()): h for p,h in (protocol['inputs'] | protocol['code']).items()}
    for e in entries:
        if e['original_path'] in expected and e['sha256'] != expected[e['original_path']]:
            raise ValueError('Frozen source changed: '+e['original_path'])
    manifest = dict(version=1, source_commit=subprocess.check_output(['git','-C',str(project),'rev-parse','HEAD'],text=True).strip(),
                    original_workspace=str(workspace),original_native=str(native),files=entries,
                    scope='Frozen physical inputs, native dependency source/licenses and completed historical evidence. Live sequence_formal_s42_v1 outputs excluded.')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for e in entries:
            z.write(e['original_path'], e['path'])
        z.writestr('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    # Independently decompress every member to check archive transport integrity.
    with zipfile.ZipFile(archive) as z:
        for e in entries:
            h = hashlib.sha256()
            with z.open(e['path']) as f:
                for block in iter(lambda:f.read(2**20),b''): h.update(block)
            if h.hexdigest()!=e['sha256']: raise ValueError('Archive hash mismatch: '+e['path'])
    (output/'input-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),'utf-8')
    result=dict(archive=archive.name,bytes=archive.stat().st_size,sha256=digest(archive),files=len(entries),
                raw_bytes=sum(e['bytes'] for e in entries),verified=True,source_commit=manifest['source_commit'])
    (output/'release-summary.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();export(a.project,a.output)
