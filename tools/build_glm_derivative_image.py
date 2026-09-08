"""Build a separate image containing only the reviewed GLM exponent correction.

Run inside an admitted CPU PB action on an eligible GB10 worker. No GPU is used.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import shutil
import tarfile
import runpy

from tools.container_runtime_identity import image_content_sha256
# This build driver is stdlib-only; the host need not install Torch merely to
# read image metadata and apply the closed byte transform.
_contract = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'prismaquant/glm_source_derivative.py'))
ORIGINAL_IMAGE_CONTENT_SHA256 = _contract['ORIGINAL_IMAGE_CONTENT_SHA256']
ORIGINAL_MODELING_SHA256 = _contract['ORIGINAL_MODELING_SHA256']
CORRECTED_MODELING_SHA256 = _contract['CORRECTED_MODELING_SHA256']
corrected_source, sha256 = _contract['corrected_source'], _contract['sha256']


def inspect_image(name):
    rows = json.loads(subprocess.check_output(['docker', 'image', 'inspect', name], text=True))
    assert len(rows) == 1
    return rows[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    before = inspect_image('prismaquant-glm-producer:content-qualified-20260908')
    assert image_content_sha256(before) == ORIGINAL_IMAGE_CONTENT_SHA256
    disk_bound = 2 * int(before['Size']) + 256 * 1024**2
    assert shutil.disk_usage(args.out).free >= disk_bound, 'insufficient disk for two bounded image archives'
    query = '''import base64,importlib.util,json
from pathlib import Path
root=Path(importlib.util.find_spec("transformers").origin).parent
p=root/"models/glm5_next/modeling_glm5_next.py"; h=root/"integrations/hub_kernels.py"
print(json.dumps(dict(path=str(p),source=base64.b64encode(p.read_bytes()).decode(),hub=base64.b64encode(h.read_bytes()).decode())))
'''
    read = json.loads(subprocess.check_output(['docker', 'run', '--rm', '--network=none',
        '--env', 'CUDA_VISIBLE_DEVICES=', '--entrypoint', 'python3', before['Id'], '-c', query], text=True))
    original = base64.b64decode(read['source'])
    patched = corrected_source(original)
    (args.out/'original-modeling_glm5_next.py').write_bytes(original)
    (args.out/'hub_kernels.py').write_bytes(base64.b64decode(read['hub']))
    path = read['path']
    assert path.startswith('/') and '..' not in Path(path).parts
    image = 'prismaquant-glm-derivative:causal-exp-v1-20260908'
    original_archive = args.out/'original-image.tar'
    subprocess.run(['docker', 'save', '--output', str(original_archive), before['Id']], check=True)
    assert original_archive.stat().st_size <= disk_bound // 2
    # A Docker daemon builder is not permitted to escape the admitted scope.
    # Compose one bounded layer and image metadata in this CPU process instead.
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:') as layer:
        member = tarfile.TarInfo(path.lstrip('/'))
        member.mode, member.size = 0o644, len(patched)
        layer.addfile(member, io.BytesIO(patched))
    added_layer = buffer.getvalue()
    layer_sha256 = hashlib.sha256(added_layer).hexdigest()
    layer_name = layer_sha256 + '/layer.tar'
    archive = args.out/'corrected-image.tar'
    with tarfile.open(original_archive, 'r:') as old:
        manifests = json.load(old.extractfile('manifest.json'))
        assert len(manifests) == 1
        manifest = manifests[0]
        config_bytes = old.extractfile(manifest['Config']).read()
        config = json.loads(config_bytes)
        assert config['rootfs']['diff_ids'] == before['RootFS']['Layers']
        assert config['config'] == before['Config'], 'saved runtime config differs'
        assert config['architecture'] == before['Architecture'] and config['os'] == before['Os']
        config['rootfs']['diff_ids'].append('sha256:' + layer_sha256)
        config.setdefault('history', []).append(dict(created_by='prismaquant reviewed GLM causal exponent v1'))
        encoded = json.dumps(config, sort_keys=True, separators=(',', ':')).encode()
        config_digest = hashlib.sha256(encoded).hexdigest()
        config_name = config_digest + '.json'
        updated = json.dumps([dict(Config=config_name, RepoTags=[image],
            Layers=[*manifest['Layers'], layer_name])], sort_keys=True).encode()
        with tarfile.open(archive, 'w:') as new:
            for member in old:
                if member.name in ('manifest.json', 'repositories', 'index.json', 'oci-layout'):
                    continue
                new.addfile(member, old.extractfile(member) if member.isfile() else None)
            for name, data in ((layer_name, added_layer), (config_name, encoded), ('manifest.json', updated)):
                member = tarfile.TarInfo(name)
                member.mode, member.size = 0o644, len(data)
                new.addfile(member, io.BytesIO(data))
    subprocess.run(['docker', 'load', '--input', str(archive)], check=True)
    after = inspect_image(image)
    assert after['Config'] == before['Config'], 'runtime image config changed'
    assert after['Architecture'] == before['Architecture'] and after['Os'] == before['Os']
    assert after['RootFS']['Layers'] == [*before['RootFS']['Layers'], 'sha256:' + layer_sha256], 'image layer digests differ'
    # Inspect the added layer, not a wrapper's claim about its contents.
    with tarfile.open(archive, 'r:') as outer:
        manifest = json.load(outer.extractfile('manifest.json'))
        assert len(manifest) == 1 and manifest[0]['Config'] == config_name
        assert hashlib.sha256(outer.extractfile(config_name).read()).hexdigest() == config_digest
        actual_layer = outer.extractfile(manifest[0]['Layers'][-1]).read()
        assert hashlib.sha256(actual_layer).hexdigest() == layer_sha256
        with tarfile.open(fileobj=io.BytesIO(actual_layer), mode='r:') as layer:
            changed = []
            for member in layer:
                if member.isdir():
                    continue
                assert member.isfile(), 'unexpected link/device in corrected layer'
                assert member.name.lstrip('./') == path.lstrip('/'), 'unreviewed changed image file'
                actual = layer.extractfile(member).read()
                assert actual == patched
                changed.append(path)
    assert changed == [path]
    # The original tag and immutable source remain independently checked.
    assert image_content_sha256(inspect_image('prismaquant-glm-producer:content-qualified-20260908')) == ORIGINAL_IMAGE_CONTENT_SHA256
    result = dict(schema='prismaquant.glm_derivative_image_build.v1', status='complete',
        original_image_content_sha256=ORIGINAL_IMAGE_CONTENT_SHA256,
        corrected_image_content_sha256=image_content_sha256(after),
        original_modeling_sha256=ORIGINAL_MODELING_SHA256,
        corrected_modeling_sha256=CORRECTED_MODELING_SHA256,
        hub_kernels_sha256=hashlib.sha256(base64.b64decode(read['hub'])).hexdigest(),
        added_layer_sha256=layer_sha256, corrected_config_sha256=config_digest,
        original_archive_bytes=original_archive.stat().st_size,
        declared_archive_disk_bound_bytes=disk_bound, persistent_python_layer_bytes=len(added_layer),
        modeling_path=path, changed_payload_files=changed, original_image=before,
        corrected_image=after, archive=dict(path=str(archive),sha256=sha256(archive)), image=image)
    (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    original_archive.unlink()  # This attempt's temporary copy is superseded by its checked derivative archive.
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
