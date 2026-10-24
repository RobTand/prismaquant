# Original source fragments from 6d8288adc94b3c4b624c143ed372fb6513f32784:tools/build_glm_derivative_image.py
# Domain dependencies are injected by the behavioral tests.
from __future__ import annotations
import json
import hashlib
import base64
from pathlib import Path

def layer_digest(added_layer):
    return hashlib.sha256(added_layer).hexdigest()

def config_bytes(config):
    return json.dumps(config, sort_keys=True, separators=(',', ':')).encode()

def config_digest(encoded):
    return hashlib.sha256(encoded).hexdigest()

def verified_config_digest(outer, config_name):
    return hashlib.sha256(outer.extractfile(config_name).read()).hexdigest()

def verified_layer_digest(actual_layer):
    return hashlib.sha256(actual_layer).hexdigest()

def hub_digest(read):
    return hashlib.sha256(base64.b64decode(read['hub'])).hexdigest()

def result_file(result):
    return json.dumps(result, indent=2)

def result_stdout(result):
    return json.dumps(result)

def embedded_query():
    query = '''import base64,importlib.util,json
from pathlib import Path
root=Path(importlib.util.find_spec("transformers").origin).parent
p=root/"models/glm5_next/modeling_glm5_next.py"; h=root/"integrations/hub_kernels.py"
print(json.dumps(dict(path=str(p),source=base64.b64encode(p.read_bytes()).decode(),hub=base64.b64encode(h.read_bytes()).decode())))
'''
    return query
