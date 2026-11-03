"""Download public split release assets, verify each part and reconstruct the frozen ZIP."""
from pathlib import Path
import argparse, hashlib, json, shutil, urllib.request


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def download(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    base='https://github.com/lpj-buaa/TARP-MPC/releases/download/research-snapshot-2026-11-03/'
    manifest=json.loads(urllib.request.urlopen(base+'parts-manifest.json',timeout=60).read())
    for part in manifest['parts']:
        name=part['name']
        if Path(name).name!=name:raise ValueError('Invalid asset name')
        path=output/name
        if path.exists():
            if digest(path)!=part['sha256']:raise ValueError('Existing part differs: '+name)
            continue
        tmp=path.with_suffix(path.suffix+'.partial')
        with urllib.request.urlopen(base+name,timeout=120) as src,tmp.open('wb') as target:shutil.copyfileobj(src,target)
        if digest(tmp)!=part['sha256']:raise ValueError('Downloaded part hash mismatch')
        tmp.replace(path);print('Verified '+name,flush=True)
    zipname=manifest['archive']
    if Path(zipname).name!=zipname:raise ValueError('Invalid archive name')
    archive=output/zipname
    if not archive.exists():
        tmp=archive.with_suffix('.zip.partial')
        with tmp.open('wb') as target:
            for part in manifest['parts']:
                with (output/part['name']).open('rb') as src:shutil.copyfileobj(src,target)
        if digest(tmp)!=manifest['sha256']:raise ValueError('Combined archive hash mismatch')
        tmp.replace(archive)
    if digest(archive)!=manifest['sha256']:raise ValueError('Archive differs')
    print(json.dumps({'archive':str(archive),'verified':True}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(Path.home()/'Downloads/TARP-MPC'));a=p.parse_args();download(a.output)
