import hashlib,json,os
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
    os.replace(tmp,path)


def bind(path,protocol):
    path=Path(path)
    if path.exists() and read(path)!=protocol:raise ValueError('Frozen stage inputs changed: '+str(path))
    write(path,protocol)


def source_hashes(*folders):
    return {str(f.resolve()):sha(f) for folder in folders for f in Path(folder).glob('*.py')}
