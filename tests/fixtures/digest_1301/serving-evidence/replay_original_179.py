import hashlib
import inspect
from pathlib import Path

def original(router):
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
