"""Run-wide identity before any scratch model or query can be resumed."""
from .collection import read,write


def ensure_protocol(root,identity):
    path=root/'protocol.json'
    if path.exists() and read(path)!=identity:
        raise ValueError('Training protocol changed; use a new learning RunId')
    if not path.exists(): write(path,identity)
