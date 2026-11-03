import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile
import pytest


def restore_module():
    spec=importlib.util.spec_from_file_location('restore_release_bundle',Path('scripts/restore_release_bundle.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def archive(tmp_path,name='workspace/project/data.txt'):
    path=tmp_path/'inputs.zip';data=b'frozen research input'
    manifest=dict(files=[dict(path=name,sha256=hashlib.sha256(data).hexdigest())])
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('MANIFEST.json',json.dumps(manifest));z.writestr(name,data)
    return path,data


def test_restore_checks_hashes_and_reuses_identical_files(tmp_path):
    z,data=archive(tmp_path);m=restore_module();workspace=tmp_path/'workspace'
    m.restore(z,workspace,tmp_path/'native');m.restore(z,workspace,tmp_path/'native',True)
    assert (workspace/'project/data.txt').read_bytes()==data


def test_restore_preserves_existing_different_file(tmp_path):
    z,_=archive(tmp_path);m=restore_module();f=tmp_path/'workspace/project/data.txt'
    f.parent.mkdir(parents=True);f.write_bytes(b'keep me')
    with pytest.raises(ValueError,match='Existing file differs'):m.restore(z,tmp_path/'workspace',tmp_path/'native')
    assert f.read_bytes()==b'keep me'


def test_restore_rejects_path_escape(tmp_path):
    z,_=archive(tmp_path,'workspace/../escape.txt');m=restore_module()
    with pytest.raises(ValueError,match='Unsafe archive path'):m.restore(z,tmp_path/'workspace',tmp_path/'native')


def test_download_reassembles_and_checks_split_archive(tmp_path,monkeypatch):
    import io
    spec=importlib.util.spec_from_file_location('download_bundle',Path('scripts/download_release_bundle.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    contents=[b'first half',b'second half'];joined=b''.join(contents)
    parts=[dict(name='input.zip.part'+str(i),sha256=hashlib.sha256(b).hexdigest()) for i,b in enumerate(contents)]
    manifest=dict(parts=parts,archive='input.zip',sha256=hashlib.sha256(joined).hexdigest())
    responses={'parts-manifest.json':json.dumps(manifest).encode(),**{p['name']:b for p,b in zip(parts,contents)}}
    monkeypatch.setattr(m.urllib.request,'urlopen',lambda url,timeout:io.BytesIO(responses[url.rsplit('/',1)[-1]]))
    m.download(tmp_path/'download');assert (tmp_path/'download/input.zip').read_bytes()==joined
    m.download(tmp_path/'download')
