"""Restore explicit archive entries with SHA256 checks; existing mismatches are never overwritten."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, shutil, zipfile


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''): h.update(block)
    return h.hexdigest()


def restore(archive,workspace,native,verify_only=False):
    roots={'workspace':Path(workspace).resolve(),'v2sim':Path(native).resolve()}
    checked=created=0
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('MANIFEST.json'))
        for e in manifest['files']:
            parts=PurePosixPath(e['path']).parts
            if not parts or parts[0] not in roots or '..' in parts or PurePosixPath(e['path']).is_absolute():
                raise ValueError('Unsafe archive path: '+e['path'])
            dest=roots[parts[0]].joinpath(*parts[1:]).resolve()
            if not dest.is_relative_to(roots[parts[0]]):raise ValueError('Escaping destination')
            if dest.exists():
                if digest(dest)!=e['sha256']:raise ValueError('Existing file differs; preserve it: '+str(dest))
            elif verify_only:raise FileNotFoundError(dest)
            else:
                dest.parent.mkdir(parents=True,exist_ok=True)
                with z.open(e['path']) as source,dest.open('xb') as target:shutil.copyfileobj(source,target)
                if digest(dest)!=e['sha256']:raise ValueError('Restored hash mismatch: '+str(dest))
                created+=1
            checked+=1
    print(json.dumps(dict(checked=checked,created=created,verified=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True)
    p.add_argument('--workspace',default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--native',default='D:/tarp_ms_single_flow_test/external_repos/v2sim')
    p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    restore(a.archive,a.workspace,a.native,a.verify_only)
