import hashlib
import inspect
from pathlib import Path

def original(router):
    return hashlib.sha256(inspect.getsource(type(router)).encode()).hexdigest()
