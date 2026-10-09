"""Interpret reviewed publisher authority using the existing bound readset.

This has no cache, delivery, network fetch, or persistent manifest of its own.
The caller's sealed control-input binding establishes publication authority;
this module checks that authority's native coordinates and their PB mapping.
"""
from __future__ import annotations

import importlib.metadata
import inspect
import json
import math
import platform
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import MappingProxyType
import re
from typing import TypedDict

from .schemas import Contract, strict_json_loads
from .dev_mode import NOT_COMPUTED, dev_mode_enabled
from .digests import (
    DIRECT_ASCII_SPACED_STRICT, DIRECT_ASCII_STRICT, canonical_json_sha256, is_sha256hex,
)
from .stage_inputs import read_bound, recorded_same, require_source_identity, same
from .memory_management import reserve_allocation


_contract = Contract(RuntimeError, 'original generation: ')
_require = _contract.require
#: A correctness join: two things that must agree to be comparable. It refuses in every mode.
_same = partial(same, contract=_contract)
#: A D32 seal: a producer run's record against the identity selected for it.
#: Certified mode (PRISMAQUANT_DEV_MODE=0) refuses as ``_same`` does; dev mode
#: prints one [DEV-MODE] line and continues with the stored record.
_recorded_same = partial(recorded_same, contract=_contract, where='original source qualification')


_GIT_OBJECT_ID = re.compile(r'[0-9a-f]{40}\Z')

ORIGINAL_AUTHORITY_SCHEMA = 'prismaquant.original_source_authority.v1'
ORIGINAL_AUTHORITY_SCOPE = 'original_text_source_diagnostic_and_first_sequence_routed_capture'
ORIGINAL_AUTHORITY_KEYS = frozenset({
    'schema', 'scope', 'publisher', 'producer', 'source_paths', 'readset', 'runtime',
    'qualification', 'root_admission', 'calibration', 'source_model_identity',
    'source_execution', 'session', 'resources',
})
ORIGINAL_STATIC_AUTHORITY_KEYS = ORIGINAL_AUTHORITY_KEYS - {'session', 'root_admission'}
ORIGINAL_EXECUTION_KEYS = frozenset({
    'authority_input', 'plan_input', 'prepared_input', 'read_manifest_input',
    'execution_input', 'implementation_sha256', 'runtime', 'resources',
    'session', 'session_identity', 'resource_check',
})
ORIGINAL_PLAN_BINDING_KEYS = frozenset({
    'authority', 'base_plan', 'prepared', 'read_manifest', 'execution',
})
ORIGINAL_RUNTIME_KEYS = frozenset({
    'schema', 'prismaquant_source_sha256', 'tessera_source_sha256', 'modeling_source',
    'model_class', 'profile', 'config', 'versions', 'container_content_sha256',
    'arithmetic', 'material_pipeline', 'prismabuild',
})
ORIGINAL_RESOURCE_KEYS = frozenset({
    'schema', 'cpu_bytes', 'material_bytes', 'source_cache_bytes', 'source_prefetch',
    'copy_bytes', 'gpu_bytes', 'native_bytes', 'serialization_bytes', 'artifact_bytes',
    'deadline_seconds', 'stall_seconds', 'host_floor_bytes', 'margin_bytes', 'claim_demand',
})
ORIGINAL_SESSION_IDENTITY_KEYS = frozenset({
    'schema', 'static_authority_sha256', 'base_plan_sha256', 'prepared_sha256',
    'read_manifest_sha256', 'implementation_sha256', 'execution_sha256',
    'source_model', 'calibration_sha256', 'calibration_shape', 'calibration_dtype',
    'selected_row_diagnostic',
})
ORIGINAL_BASE_PLAN_KEYS = frozenset({
    'schema', 'model', 'output_root', 'static_authority_sha256', 'prepared',
    'read_manifest', 'execution', 'calibration_input', 'selected_row_diagnostic',
})
ORIGINAL_PREPARATION_KEYS = frozenset({
    'schema', 'scope', 'static_authority_sha256', 'implementation_sha256',
    'source_model_identity', 'source_execution', 'calibration', 'resources', 'head_source',
})
ORIGINAL_RESULT_SELECTION_KEYS = frozenset({
    'queue_root', 'action_key', 'published_unix', 'attempt', 'max_result_bytes',
    'max_evidence_bytes', 'request', 'payload_sha256', 'source_inputs',
})


class BoundOriginalInput(TypedDict):
    path: str
    sha256: str


class OriginalPublisher(TypedDict):
    id: str
    revision: str
    input: BoundOriginalInput


class OriginalSession(TypedDict):
    generation: str
    run_identity_sha256: str


class OriginalSourceAuthority(TypedDict):
    schema: str
    scope: str
    publisher: OriginalPublisher
    producer: BoundOriginalInput
    source_paths: BoundOriginalInput
    readset: BoundOriginalInput
    runtime: BoundOriginalInput
    qualification: BoundOriginalInput | None
    root_admission: BoundOriginalInput | None
    calibration: dict
    source_model_identity: dict
    source_execution: dict
    session: OriginalSession
    resources: BoundOriginalInput


def _snapshot(value):
    """An independent JSON mapping, never a mutable adopted control object."""
    return json.loads(DIRECT_ASCII_SPACED_STRICT.text(value))


def _exact(value, keys, label):
    return _contract.exact_mapping(value, keys=frozenset(keys), where=label)


def _binding(value, label):
    _exact(value, {'path', 'sha256'}, label)
    _contract.absolute_posix_path(value['path'], where=f'{label} path')
    _contract.sha256(value['sha256'], where=f'{label} SHA256')
    return value


def original_authority_static_sha256(authority):
    _exact(authority, ORIGINAL_AUTHORITY_KEYS, 'original source authority')
    return canonical_json_sha256(
        {key: authority[key] for key in ORIGINAL_STATIC_AUTHORITY_KEYS},
        where='original static source authority')


def _session(value):
    _exact(value, {'generation', 'run_identity_sha256'}, 'original source session')
    _contract.string(value['generation'], where='original session generation',
                     pattern=re.compile('[0-9a-f]{32}\\Z'))
    _contract.sha256(value['run_identity_sha256'], where='original session run identity')
    return value


def _source_execution(value):
    from .joint_aura import (
        SOURCE_EXECUTION_KEYS, SOURCE_EXECUTION_SCHEMA, _source_execution_selectors,
    )

    _exact(value, SOURCE_EXECUTION_KEYS, 'original source execution')
    _same(value['schema'], SOURCE_EXECUTION_SCHEMA, 'original execution schema')
    modules = _contract.mapping(value['modules'], where='original execution modules')
    _require(bool(modules), 'original execution has no resolved selectors')
    for name, selectors in modules.items():
        _require(type(name) is str and _source_execution_selectors(selectors),
                 'invalid original execution selector')
        for key, selector in selectors.items():
            _require(selector is None or (type(selector) is str and selector) or
                     (isinstance(selector, dict) and selector and all(
                         type(k) is str and (v is None or type(v) is str and v)
                         for k, v in selector.items())),
                     f'original execution {name}.{key} needs resolved selector values')
    return value


def _full_calibration(value, *, shape=(512, 512)):
    """Keep the original draw pinned; only Fisher explicitly passes shape=None."""
    _exact(value, {'schema', 'artifact_sha256', 'calibration_sha256', 'shape', 'dtype',
                   'provenance'}, 'original full calibration')
    _same(value['schema'], 'prismaquant.calibration_input.v1', 'full calibration schema')
    _require(isinstance(value["shape"], list) and len(value["shape"]) == 2
             and all(type(dim) is int and dim > 0 for dim in value["shape"]),
             "full calibration shape must retain two positive integer dimensions")
    if shape is not None:
        _same(value["shape"], list(shape), "full calibration shape")
    rows, seqlen = value["shape"]
    _same(value["dtype"], "torch.int64", "full calibration dtype")
    for key in ('artifact_sha256', 'calibration_sha256'):
        _contract.sha256(value[key], where=f'full calibration {key}')
    required = {"fit_ids_sha256", "fit_tokens", "model", "nsamples", "seed",
                "seqlen", "source", "split_role", "text_sha256"}
    provenance = value["provenance"]
    expected_keys = (required, required | {"fit_tokens_min"}) if shape is None else (required | {"fit_tokens_min"},)
    _require(isinstance(provenance, dict) and set(provenance) in expected_keys,
             "full calibration provenance fields differ")
    for key, expected in (("nsamples", rows), ("seqlen", seqlen), ("fit_tokens", rows * seqlen)):
        _require(type(provenance.get(key)) is int and provenance[key] == expected,
                 f"full calibration provenance {key} differs from the actual tensor shape")
    for key in ('fit_ids_sha256', 'text_sha256'):
        _contract.sha256(provenance.get(key), where=f'full calibration provenance {key}')
    if "fit_tokens_min" in provenance:
        _contract.integer(provenance["fit_tokens_min"], where="full calibration minimum fit tokens", minimum=1)
    _contract.integer(provenance['seed'], where='full calibration draw seed', minimum=0)
    for key in ('model', 'source', 'split_role'):
        _contract.string(provenance[key], where=f'full calibration provenance {key}')
    # Tokenizer bytes are in the independently bound complete producer's
    # auxiliary roster. The actual retained draw has no tokenizer field;
    # inventing one would restamp its immutable provenance/artifact identity.
    return value


def _validate_original_source_runtime(value, expected, *, sdk_version):
    """Validate one runtime against its independently selected exact SDK policy."""
    if isinstance(expected, dict) and set(expected) == {'path', 'sha256'}:
        _, expected = _control(expected, 'original expected runtime')
    _exact(value, ORIGINAL_RUNTIME_KEYS, 'original runtime')
    _same(value['schema'], 'prismaquant.original_source_runtime.v1', 'original runtime schema')
    for key in ('prismaquant_source_sha256', 'tessera_source_sha256'):
        _contract.sha256(value[key], where=f'original runtime {key}')
    _binding(value['modeling_source'], 'original modeling source')
    for key in ('model_class', 'profile'):
        _contract.string(value[key], where=f'original runtime {key}')
    _contract.mapping(value['config'], where='original runtime config')
    versions = _exact(value['versions'], {'python', 'torch', 'torch_git', 'cuda', 'transformers'},
                      'original runtime versions')
    for key in ('python', 'torch', 'transformers'):
        _contract.string(versions[key], where=f'original runtime version {key}')
    for key in ('torch_git', 'cuda'):
        _require(versions[key] is None or type(versions[key]) is str, f'invalid runtime version {key}')
    _require(value['container_content_sha256'] is None or
             is_sha256hex(value['container_content_sha256']), 'invalid original container content identity')
    _exact(value['arithmetic'], {'matmul_precision', 'allow_tf32', 'allow_bf16_reduced_precision_reduction'},
           'original runtime arithmetic')
    _require(type(value['arithmetic']['allow_tf32']) is bool and
             type(value['arithmetic']['allow_bf16_reduced_precision_reduction']) is bool and
             value['arithmetic']['matmul_precision'] in ('highest', 'high', 'medium'),
             'invalid original matmul arithmetic')
    pipeline = _exact(value['material_pipeline'], {'decoder', 'framework', 'decoder_device',
        'cast_owner', 'direct_gpu_decode', 'target_dtype', 'tensor_dtypes', 'scale_inv_map'},
        'original whole-file decoder/cast pipeline')
    _same({key: pipeline[key] for key in ('decoder', 'framework', 'decoder_device',
        'cast_owner', 'direct_gpu_decode')}, {
        'decoder': 'safetensors.safe_open', 'framework': 'pt', 'decoder_device': 'cpu',
        'cast_owner': 'prismaquant.layer_streaming', 'direct_gpu_decode': False,
    }, 'original whole-file decoder/cast pipeline')
    _contract.string(pipeline['target_dtype'], where='actual loader target dtype')
    _contract.mapping(pipeline['tensor_dtypes'], where='actual model-declared tensor cast map')
    for name, dtype in pipeline['tensor_dtypes'].items():
        _contract.string(name, where='actual tensor cast name')
        _contract.string(dtype, where='actual tensor cast dtype')
    _contract.mapping(pipeline['scale_inv_map'], where='actual inline scale/cast map')
    for name, coordinate in pipeline['scale_inv_map'].items():
        _contract.string(name, where='actual scale/cast name')
        _require(isinstance(coordinate, list) and len(coordinate) == 2 and
                 all(type(item) is str and item for item in coordinate), 'actual scale/cast coordinate malformed')
    pb = _exact(value['prismabuild'], {'sdk_version', 'helper_root', 'source_tree', 'runtime_generation'},
                'original runtime PrismaBuild')
    _same(pb['sdk_version'], sdk_version, 'original runtime SDK version')
    _contract.absolute_posix_path(pb['helper_root'], where='original runtime helper root')
    tree = _exact(pb['source_tree'], {'package_sha256', 'helper_tree_sha256'},
                  'original runtime complete shared tree')
    _contract.sha256(tree['package_sha256'], where='original shared package tree')
    _require(tree['helper_tree_sha256'] is None or is_sha256hex(tree['helper_tree_sha256']),
             'invalid original shared generation tree digest')
    _contract.string(pb['runtime_generation'], where='original published runtime generation')
    _same(value, expected, 'observed original source runtime')
    return _snapshot(value)


def validate_original_source_runtime(value, expected):
    """The current consumer's installed runtime retains its exact SDK boundary."""
    from .staged_lease import PB_CLIENT_SDK_VERSION

    return _validate_original_source_runtime(value, expected, sdk_version=PB_CLIENT_SDK_VERSION)


def _resources(value):
    _exact(value, ORIGINAL_RESOURCE_KEYS, 'original source resources')
    _same(value['schema'], 'prismaquant.original_source_resources.v1', 'original resource schema')
    for key in ORIGINAL_RESOURCE_KEYS - {'schema', 'source_prefetch', 'claim_demand',
                                         'deadline_seconds', 'stall_seconds'}:
        _contract.integer(value[key], where=f'original resource {key}',
                          minimum=0 if key in {'gpu_bytes', 'native_bytes', 'margin_bytes'} else 1)
    for key in ('deadline_seconds', 'stall_seconds'):
        _require(type(value[key]) in (int, float) and math.isfinite(value[key]) and value[key] > 0,
                 f'original resources require finite positive {key}')
    from .stage_inputs import source_prefetch

    prefetch = source_prefetch(value)
    _require((prefetch['max_cache_slots'], prefetch['prefetch_lookahead'],
              prefetch['prefetch_workers']) == (2, 1, 1), 'original source prefetch differs from 2/1/1')
    _contract.mapping(value['claim_demand'], where='original resource claim demand')
    _require(value['material_bytes'] <= value['cpu_bytes'] and
             value['serialization_bytes'] <= value['cpu_bytes'], 'original sub-envelope exceeds CPU bound')
    return value


def _control(record, label):
    # These are small authority/control inputs, not model payloads. A caller
    # must bind their digests independently in the enclosing reviewed action.
    _contract.exact_mapping(record, keys=frozenset({'path', 'sha256'}), where=label)
    _contract.absolute_posix_path(record['path'], where=f'{label} path')
    _contract.sha256(record['sha256'], where=f'{label} SHA256')
    _require(Path(record['path']).stat().st_size <= 16 * 1024**2, f'{label} control input too large')
    raw = read_bound(record, label)
    _require(0 < len(raw) <= 16 * 1024**2, f'{label} control input too large or empty')
    try:
        value = strict_json_loads(raw,
            duplicate=lambda key: RuntimeError(f'{label}: duplicate key {key}'),
            constant=lambda name: RuntimeError(f'{label}: invalid constant {name}'))
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError(f'{label}: invalid strict JSON') from exc
    return raw, value


@dataclass(frozen=True)
class OriginalCoordinate:
    path: str
    size: int
    sha256: str
    git_blob: str | None


def original_generation_coordinates(*, publisher_input, publisher_id, publisher_revision,
                                    readset_input, source_paths, producer_source, resource_check):
    """Closed native HF sibling roster projected into existing PB whole-file entries.

    A Git auxiliary's SHA256 comes from the separately bound readset; delivered
    bytes must ALSO authenticate to the publisher's native Git blob ID before
    bootstrap. LFS SHA256 and lengths are independently publisher-derived.
    """
    _publisher_raw, publisher = _control(publisher_input, 'publisher authority')
    _contract.string(publisher_id, where='explicit publisher ID')
    _contract.string(publisher_revision, where='full publisher revision', pattern=_GIT_OBJECT_ID)
    _require(isinstance(publisher, dict) and publisher.get('id') == publisher_id and
             publisher.get('sha') == publisher_revision, 'publisher/revision authority mismatch')
    siblings = publisher.get('siblings')
    _require(isinstance(siblings, list) and siblings, 'closed publisher roster missing')
    native = {}
    for row in siblings:
        _require(isinstance(row, dict), 'invalid publisher coordinate')
        name, size = row.get('rfilename'), row.get('size')
        _contract.string(name, where='publisher coordinate')
        _require(isinstance(name, str) and name not in ('', '.', '..') and
                 Path(name).name == name and '/' not in name and '\\' not in name and
                 name not in native and type(size) is int and size > 0,
                 'unsupported or duplicate publisher coordinate')
        lfs = row.get('lfs')
        if lfs is not None:
            _require(isinstance(lfs, dict) and lfs.get('size') == size and
                     is_sha256hex(lfs.get('sha256')), f'{name}: invalid publisher LFS authority')
        _contract.string(row.get('blobId'), where=f'{name}: native Git blob authority',
                         pattern=_GIT_OBJECT_ID)
        native[name] = row
    _require(isinstance(source_paths, dict) and set(source_paths) == set(native),
             'source mapping must cover exactly the closed publisher roster')
    for name, path in source_paths.items():
        _contract.absolute_posix_path(path, where=f'{name}: physical mapping')
    _require(len(set(source_paths.values())) == len(source_paths),
             'source mapping repeats a physical file')
    from .io_engine import SealedBuffer
    from .staged_lease import client_sdk

    raw, _readset_document = _control(readset_input, 'original material readset')
    client = client_sdk()  # The same qualified generation owns parsing and leases.
    reserve_allocation(resource_check, 'before_original_generation_control', cpu_bytes=len(raw))
    control = SealedBuffer(len(raw))
    try:
        control.fill_bytes(raw)
        _require(control.seal() == readset_input['sha256'], 'readset control binding changed')
        control.require_sealed()
        readset, _encoding = client.read_data_manifest(control.path)
    finally:
        control.close()
    entries = {(row['path'], row['offset']): row for row in readset['entries']}
    coordinates = {}
    for name, row in native.items():
        path = source_paths[name]
        _contract.absolute_posix_path(path, where=f'{name}: physical mapping')
        entry = entries.get((path, 0))
        _require(entry is not None and entry['bytes'] == row['size'] and
                 is_sha256hex(entry['sha256']), f'{name}: complete whole-file readset binding required')
        lfs = row.get('lfs')
        _require(lfs is None or entry['sha256'] == lfs['sha256'],
                 f'{name}: readset differs from publisher LFS digest')
        coordinates[name] = OriginalCoordinate(path, row['size'], entry['sha256'],
                                               None if lfs is not None else row['blobId'])
    producer = require_source_identity(producer_source)
    weights = {name for name in native if name.endswith('.safetensors')}
    _require(weights and set(producer['files']) == weights,
             'complete producer weight roster differs from publisher')
    _require(not (set(producer['files']) & set(producer['auxiliary_sha256'])),
             'producer weight/auxiliary identity overlaps')
    declared = {**producer['files'], **producer['auxiliary_sha256'],
                'config.json': producer['config_sha256']}
    _require(set(declared) == set(native), 'complete producer auxiliary roster differs from publisher')
    for name, digest in declared.items():
        _contract.sha256(digest, where=f'{name}: producer SHA256')
    _require('config.json' in native and 'model.safetensors.index.json' in native,
             'whole-file lane requires config and complete checkpoint index')
    for name, digest in declared.items():
        _require(name in coordinates and coordinates[name].sha256 == digest,
                 f'{name}: publisher/readset differs from census producer')
    return MappingProxyType(coordinates), dict(producer['tensors'])


def _bound_source_documents(authority, resource_check):
    publisher = _exact(authority['publisher'], {'id', 'revision', 'input'}, 'original publisher')
    _contract.string(publisher['id'], where='original publisher ID')
    _contract.string(publisher['revision'], where='original publisher revision', pattern=_GIT_OBJECT_ID)
    _, producer = _control(authority['producer'], 'original producer')
    producer = require_source_identity(producer)
    _, paths = _control(authority['source_paths'], 'original source paths')
    coordinates, tensors = original_generation_coordinates(
        publisher_input=publisher['input'], publisher_id=publisher['id'],
        publisher_revision=publisher['revision'], readset_input=authority['readset'],
        source_paths=paths, producer_source=producer, resource_check=resource_check)
    return producer, paths, coordinates, tensors


def _shared_runtime_tree(sdk, claim):
    from .production_weight_cache import _production_cache_source_sha256

    package = Path(sdk.__file__).resolve().parent
    root = Path(claim['helper_root']).resolve()
    # Existing test-only installed-SDK controls truthfully have no complete
    # shared helper/worker/proxy generation. Null is missing proof, never a
    # substitute tree or a source eligibility flag.
    sealed = package == root / 'src' / 'prismabuild'
    return {'package_sha256': _production_cache_source_sha256(package),
            'helper_tree_sha256': _production_cache_source_sha256(root) if sealed else None}


def _environment_matches(runtime):
    """Re-read the executing package/SDK/image axes without a source/profile read."""
    import torch
    import tessera
    from .joint_projection_backend import executing_image
    from .production_weight_cache import _production_cache_source_sha256
    from .staged_lease import resolve_context

    sdk, claim = resolve_context()
    _same(runtime['versions'], {
        'python': platform.python_version(), 'torch': str(torch.__version__),
        'torch_git': torch.version.git_version, 'cuda': torch.version.cuda,
        'transformers': importlib.metadata.version('transformers'),
    }, 'installed original runtime versions')
    _same(runtime['arithmetic'], {
        'matmul_precision': torch.get_float32_matmul_precision(),
        'allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
        'allow_bf16_reduced_precision_reduction':
            bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
    }, 'installed original matmul arithmetic')
    _same(runtime['container_content_sha256'], executing_image(), 'executing original image')
    _same(runtime['prismaquant_source_sha256'], _production_cache_source_sha256(),
          'executing PrismaQuant module tree')
    _same(runtime['tessera_source_sha256'],
          _production_cache_source_sha256(Path(tessera.__file__).resolve().parent),
          'executing Tessera module tree')
    _same(runtime['prismabuild']['sdk_version'], sdk.SDK_VERSION, 'executing SDK version')
    _same(runtime['prismabuild']['helper_root'], claim['helper_root'], 'executing shared SDK root')
    _same(runtime['prismabuild']['source_tree'], _shared_runtime_tree(sdk, claim),
          'executing complete shared SDK/helper/worker/proxy generation tree')
    _same(runtime['prismabuild']['runtime_generation'], Path(claim['helper_root']).name,
          'executing shared runtime generation')
    modeling = runtime['modeling_source']
    from .digests import file_sha256hex

    module = runtime['model_class'].rpartition('.')[0]
    _require(module.startswith('transformers.models.') and
             re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+', module),
             'original model class must resolve from installed stock Transformers')
    installed = Path(importlib.metadata.distribution('transformers').locate_file(
        module.replace('.', '/') + '.py')).resolve(strict=True)
    _same(modeling['path'], str(installed), 'original installed modeling source path')
    _same(file_sha256hex(Path(modeling['path'])), modeling['sha256'], 'installed modeling source')
    return sdk, claim


def original_checkpoint_description(source_model, owner):
    """Read checkpoint metadata only from the existing qualified original owner."""
    from .tessera_calibration_cache import CaptureSourceAuthentication

    if not isinstance(owner, CaptureSourceAuthentication) or not owner.is_qualified_original_material:
        raise RuntimeError("original identity requires the qualified existing original owner")
    if os.path.abspath(str(source_model)) != str(owner.root):
        raise RuntimeError("original identity source root differs from its owner")
    return owner.original_checkpoint_descriptor()


def original_source_runtime(runner, owner):
    """Observe the live model/context and installed source axes; no expected echo."""
    import torch
    import tessera
    from .joint_projection_backend import executing_image
    from .production_weight_cache import _production_cache_source_sha256
    from .staged_lease import resolve_context
    from .tessera_calibration_cache import CaptureSourceAuthentication
    from .digests import file_sha256hex

    _require(isinstance(owner, CaptureSourceAuthentication) and owner.is_qualified_original_material
             and runner.context.source_authentication is owner,
             'original runtime requires the same source context/owner')
    sdk, claim = resolve_context()
    cls = type(runner.model)
    modeling = Path(inspect.getfile(cls)).resolve(strict=True)
    value = {
        'schema': 'prismaquant.original_source_runtime.v1',
        'prismaquant_source_sha256': _production_cache_source_sha256(),
        'tessera_source_sha256': _production_cache_source_sha256(Path(tessera.__file__).resolve().parent),
        'modeling_source': {'path': str(modeling), 'sha256': file_sha256hex(modeling)},
        'model_class': f'{cls.__module__}.{cls.__qualname__}', 'profile': runner.profile.name,
        'config': runner.model.config.to_dict(),
        'versions': {'python': platform.python_version(), 'torch': str(torch.__version__),
            'torch_git': torch.version.git_version, 'cuda': torch.version.cuda,
            'transformers': importlib.metadata.version('transformers')},
        'container_content_sha256': executing_image(),
        'arithmetic': {'matmul_precision': torch.get_float32_matmul_precision(),
            'allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
            'allow_bf16_reduced_precision_reduction':
                bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)},
        'material_pipeline': {'decoder': 'safetensors.safe_open', 'framework': 'pt',
            'decoder_device': 'cpu', 'cast_owner': 'prismaquant.layer_streaming',
            'direct_gpu_decode': False, 'target_dtype': str(runner.context.dtype),
            'tensor_dtypes': {name: str(dtype) for name, dtype in sorted(runner.context.buffer_dtypes.items())},
            'scale_inv_map': {name: list(coordinate) for name, coordinate
                              in sorted(runner.context.fp8_scale_inv_map.items())}},
        'prismabuild': {'sdk_version': sdk.SDK_VERSION, 'helper_root': claim['helper_root'],
            'source_tree': _shared_runtime_tree(sdk, claim),
            'runtime_generation': Path(claim['helper_root']).name},
    }
    return validate_original_source_runtime(value, value)


def normalize_original_source_static_authority(document):
    """Pure canonical projection for issuance, not a source admission object."""
    from .cost_streaming import validate_streamed_model_identity

    _exact(document, ORIGINAL_STATIC_AUTHORITY_KEYS, 'original static authority')
    _same(document['schema'], ORIGINAL_AUTHORITY_SCHEMA, 'original static authority schema')
    _same(document['scope'], ORIGINAL_AUTHORITY_SCOPE, 'original static authority scope')
    publisher = _exact(document['publisher'], {'id', 'revision', 'input'}, 'original static publisher')
    _contract.string(publisher['id'], where='original static publisher ID')
    _contract.string(publisher['revision'], where='original static publisher revision', pattern=_GIT_OBJECT_ID)
    _binding(publisher['input'], 'original static publisher input')
    for key in ('producer', 'source_paths', 'readset', 'runtime', 'resources'):
        _binding(document[key], f'original static {key}')
    if document['qualification'] is not None:
        _binding(document['qualification'], 'original static qualification')
    _full_calibration(document['calibration'])
    validate_streamed_model_identity(document['source_model_identity'], where='original static authority')
    _source_execution(document['source_execution'])
    return _snapshot(document)


def normalize_original_diagnostic_base_plan(document):
    """The independently sealed acyclic base; it contains no final authority."""
    from .stage_a_selected_row_diagnostic import SPEC_SCHEMA, _FIELDS

    _exact(document, ORIGINAL_BASE_PLAN_KEYS, 'original diagnostic base plan')
    _same(document['schema'], 'prismaquant.original_diagnostic_base_plan.v1', 'original base plan schema')
    for key in ('model', 'output_root'):
        _contract.absolute_posix_path(document[key], where=f'original base plan {key}')
    _contract.sha256(document['static_authority_sha256'], where='base static authority')
    for key in ('prepared', 'read_manifest', 'execution', 'calibration_input'):
        _binding(document[key], f'original base plan {key}')
    diagnostic = _exact(document['selected_row_diagnostic'], _FIELDS, 'original diagnostic geometry')
    _same(diagnostic['schema'], SPEC_SCHEMA, 'original diagnostic schema')
    _contract.sha256(diagnostic['calibration_tensor_sha256'], where='diagnostic full tensor')
    for key, expected in (('calibration_shape', [512, 512]), ('calibration_dtype', 'torch.int64'),
                          ('selected_global_row', 0), ('probe_seed', 7000),
                          ('global_token_count', 262144), ('vocab_size', 154880), ('through', 6)):
        _same(diagnostic[key], expected, f'original diagnostic {key}')
        if type(expected) is int:
            _require(type(diagnostic[key]) is int, f'original diagnostic {key} must be an integer')
    return _snapshot(document)


def normalize_original_diagnostic_preparation(document):
    from .cost_streaming import validate_streamed_model_identity

    _exact(document, ORIGINAL_PREPARATION_KEYS, 'original diagnostic preparation')
    _same(document['schema'], 'prismaquant.original_diagnostic_preparation.v1', 'original preparation schema')
    _same(document['scope'], 'render_free_original_source_context', 'original preparation scope')
    for key in ('static_authority_sha256', 'implementation_sha256'):
        _contract.sha256(document[key], where=f'original preparation {key}')
    validate_streamed_model_identity(document['source_model_identity'], where='original preparation')
    _source_execution(document['source_execution'])
    _full_calibration(document['calibration'])
    _binding(document['resources'], 'original preparation resource envelope')
    head = _exact(document['head_source'], {'tensors'}, 'original prepared head source')['tensors']
    _require(isinstance(head, list) and head and all(type(name) is str and name for name in head)
             and len(set(head)) == len(head), 'original prepared head tensor roster malformed')
    return _snapshot(document)


def normalize_original_diagnostic_execution(document):
    """The existing Stage-A execution fields with an explicit render-free scope."""
    from .cost_streaming import check_boundary_storage, LAYER_MAJOR_BOUNDARY_STORAGE_SCHEMA

    _exact(document, {'schema', 'n_calib_samples', 'calib_seqlen', 'n_probes', 'seed_base',
        'probe_microbatch', 'token_scope', 'temperature', 'boundary_storage', 'stride',
        'chain_batch_size', 'chain_probe_fusion'}, 'original diagnostic execution')
    _same(document['schema'], 'prismaquant.original_diagnostic_execution.v1', 'original execution spec schema')
    for key, expected in (('n_calib_samples', 512), ('calib_seqlen', 512), ('n_probes', 1),
                          ('seed_base', 7000), ('probe_microbatch', 1), ('chain_batch_size', 1)):
        _require(type(document[key]) is int and document[key] == expected,
                 f'original execution {key} differs')
    _same(document['token_scope'], 'all', 'original execution token scope')
    _require(type(document['temperature']) in (int, float) and document['temperature'] == 1,
             'original execution temperature differs')
    _require(document['chain_probe_fusion'] is False, 'original execution cannot inherit fused chain scope')
    _contract.integer(document['stride'], where='original execution stride', minimum=1)
    policy = check_boundary_storage(document['boundary_storage'])
    _require(isinstance(policy, dict) and policy['schema'] == LAYER_MAJOR_BOUNDARY_STORAGE_SCHEMA,
             'original execution requires the existing layer-major exact boundary owner')
    _contract.absolute_posix_path(policy['directory'], where='original exact boundary directory')
    return _snapshot(document)


def normalize_original_fisher_execution(document, calibration):
    """Generate the full Fisher execution from its real, independently loaded draw.

    This is not the selected-row diagnostic: its one-probe scope remains
    unchanged. Native Stage A/B execute these returned row/context/probe
    fields on the actual tensor, never an old draw relabelled as a new one.
    """
    draw = _full_calibration(calibration, shape=None)
    rows, seqlen = draw["shape"]
    for key, expected in (("n_calib_samples", rows), ("calib_seqlen", seqlen)):
        _require(type(document.get(key)) is int and document[key] == expected,
                 f"Fisher execution {key} differs from the actual calibration tensor")
    _contract.integer(document.get("n_probes"), where="Fisher execution n_probes", minimum=2)
    _contract.integer(document.get("seed_base"), where="Fisher execution seed_base", minimum=0)
    _contract.integer(document.get("probe_microbatch"), where="Fisher execution probe_microbatch", minimum=1)
    _same(document.get("token_scope"), "all", "Fisher execution token scope")
    _require(type(document.get("temperature")) in (int, float) and document["temperature"] == 1,
             "Fisher execution temperature differs from the defined objective")
    _require(draw["provenance"]["split_role"] == "calibration",
             "Fisher execution cannot tune on held-out or final benchmark tokens")
    return _snapshot(document)


def original_diagnostic_session_identity(*, base_plan, base_plan_sha256, prepared, execution_sha256):
    """Identity the existing artifact owner hashes before it mints a generation."""
    base = normalize_original_diagnostic_base_plan(base_plan)
    preparation = normalize_original_diagnostic_preparation(prepared)
    _contract.sha256(base_plan_sha256, where='original base plan bytes')
    _contract.sha256(execution_sha256, where='original execution spec bytes')
    _same(base['execution']['sha256'], execution_sha256, 'original execution spec binding')
    _same(base['static_authority_sha256'], preparation['static_authority_sha256'], 'prepared/base static authority')
    _same(base['calibration_input']['sha256'], preparation['calibration']['artifact_sha256'],
          'prepared/base full calibration artifact')
    diagnostic = base['selected_row_diagnostic']
    _same(diagnostic['calibration_tensor_sha256'], preparation['calibration']['calibration_sha256'],
          'prepared/base full calibration tensor')
    # The tensor digest alone cannot distinguish [128, 2048] from [512, 512]
    # (both 262144 tokens): compare the shapes the records declare.
    _same(preparation['calibration']['shape'], diagnostic['calibration_shape'],
          'prepared/base full calibration shape')
    return {
        'schema': 'prismaquant.original_diagnostic_session_identity.v1',
        'static_authority_sha256': base['static_authority_sha256'],
        'base_plan_sha256': base_plan_sha256, 'prepared_sha256': base['prepared']['sha256'],
        'read_manifest_sha256': base['read_manifest']['sha256'],
        'implementation_sha256': preparation['implementation_sha256'],
        'execution_sha256': execution_sha256,
        'source_model': preparation['source_model_identity'],
        'calibration_sha256': preparation['calibration']['calibration_sha256'],
        'calibration_shape': preparation['calibration']['shape'],
        'calibration_dtype': preparation['calibration']['dtype'],
        'selected_row_diagnostic': diagnostic,
    }


def observe_original_source_execution(authority_input, plan_input, *, resource_check):
    """Build the existing-owner intake packet from independently bound controls.

    This observes no source/model/device and grants no eligibility. The public
    requirement separately revalidates active native claim/CAS readset and live
    installed environment, then requires actual qualification/root admission.
    """
    _, authority = _control(authority_input, 'original authority')
    _exact(authority, ORIGINAL_AUTHORITY_KEYS, 'original source authority')
    _, plan = _control(plan_input, 'original final plan')
    bindings = _exact(plan.get('original_source'), ORIGINAL_PLAN_BINDING_KEYS, 'original final plan bindings')
    _same(bindings['authority'], authority_input, 'independent final-plan authority binding')
    _, base = _control(bindings['base_plan'], 'original base plan')
    _, prepared = _control(bindings['prepared'], 'original preparation')
    _, execution = _control(bindings['execution'], 'original execution spec')
    execution = normalize_original_diagnostic_execution(execution)
    from .stage_a_selected_row_diagnostic import load_original_diagnostic_issued_context

    context = load_original_diagnostic_issued_context(plan.get('original_session_preparation'),
        base_plan_input=bindings['base_plan'], authority=authority)
    _, runtime = _control(authority['runtime'], 'original runtime')
    _, resources = _control(authority['resources'], 'original resources')
    return dict(authority_input=_snapshot(authority_input), plan_input=_snapshot(plan_input),
        prepared_input=_snapshot(bindings['prepared']), read_manifest_input=_snapshot(bindings['read_manifest']),
        execution_input=_snapshot(bindings['execution']), implementation_sha256=prepared['implementation_sha256'],
        runtime=runtime, resources=resources,
        session=_snapshot(context['session_preparation']['session']),
        session_identity=_snapshot(context['session_identity']),
        resource_check=resource_check)


def _normalize_original_source_authority(owner, authority_input, plan_input, admitted_execution):
    """Strict CPU control/owned-metadata joins, deliberately NOT admission.

    Missing future proof bindings remain visible as null here. Only
    require_original_source_authority may require and accept those proofs;
    consumers cannot turn this independent snapshot into an admitted owner.
    """
    from .cost_streaming import validate_streamed_model_identity
    from .staged_lease import _load_sealed_payload

    _exact(admitted_execution, ORIGINAL_EXECUTION_KEYS, 'original admitted execution')
    for key, supplied in (('authority_input', authority_input), ('plan_input', plan_input)):
        _same(admitted_execution[key], _binding(supplied, key), f'independent execution {key}')
    check = admitted_execution['resource_check']
    _require(callable(check), 'original execution requires its owning resource check')
    if owner is not None:
        from .tessera_calibration_cache import CaptureSourceAuthentication

        _require(isinstance(owner, CaptureSourceAuthentication) and owner.is_qualified_original_material,
                 'original authority requires the existing original source owner')
        owner._require_open()
        _require(owner.resource_check is check, 'original authority resource check belongs to another owner')
    _, authority = _control(authority_input, 'original authority')
    _exact(authority, ORIGINAL_AUTHORITY_KEYS, 'original source authority')
    _same(authority['schema'], ORIGINAL_AUTHORITY_SCHEMA, 'original authority schema')
    _same(authority['scope'], ORIGINAL_AUTHORITY_SCOPE, 'original authority scope')
    for key in ('qualification', 'root_admission'):
        if authority[key] is not None:
            _binding(authority[key], f'original {key}')
    _, plan = _control(plan_input, 'original final plan')
    bindings = _exact(plan.get('original_source'), ORIGINAL_PLAN_BINDING_KEYS, 'original final plan bindings')
    _same(bindings['authority'], authority_input, 'independent final-plan authority')
    for key, packet in (('prepared', 'prepared_input'), ('read_manifest', 'read_manifest_input'),
                        ('execution', 'execution_input')):
        _same(_binding(bindings[key], f'original final plan {key}'), admitted_execution[packet],
              f'independent execution {key}')
    _, base = _control(bindings['base_plan'], 'original base plan')
    base = normalize_original_diagnostic_base_plan(base)
    for key in ('prepared', 'execution'):
        _same(base[key], bindings[key], f'base/final {key} binding')
    _same(base['read_manifest'], authority['readset'], 'static source read manifest')
    _same(base['static_authority_sha256'], original_authority_static_sha256(authority), 'base static source authority')
    _same(plan.get('model'), base['model'], 'base/final source root')
    _same(plan.get('calibration_input'), base['calibration_input'], 'base/final full calibration input')
    _, prepared = _control(bindings['prepared'], 'original preparation')
    prepared = normalize_original_diagnostic_preparation(prepared)
    _same(prepared['static_authority_sha256'], base['static_authority_sha256'], 'prepared static authority')
    _same(prepared['implementation_sha256'], admitted_execution['implementation_sha256'], 'prepared implementation')
    _contract.sha256(prepared['implementation_sha256'], where='original implementation')
    for key in ('source_model_identity', 'source_execution', 'calibration', 'resources'):
        _same(prepared[key], authority[key], f'prepared original {key}')
    head = prepared['head_source']['tensors']
    _, execution = _control(bindings['execution'], 'original execution spec')
    execution = normalize_original_diagnostic_execution(execution)
    producer, paths, coordinates, tensors = _bound_source_documents(authority, check)
    _require(set(head) <= set(tensors), 'original prepared head is absent from complete checkpoint index')
    source = validate_streamed_model_identity(authority['source_model_identity'], where='original authority')
    _exact(source, {'schema', 'source', 'resolved_commit', 'content_sha256', 'config', 'weight_map',
                    'shards', 'checkpoint_weight_map'}, 'original complete streamed identity')
    _same(source['source'], base['model'], 'original identity root')
    _same(source['checkpoint_weight_map'], tensors, 'original complete source index')
    shards = {Path(row['path']).name: row for row in source['shards']}
    _require(len(shards) == len(source['shards']) and set(shards) == set(producer['files']),
             'original identity shard roster differs')
    for name, row in shards.items():
        _exact(row, {'path', 'size', 'sha256'}, f'original identity shard {name}')
        _contract.integer(row['size'], where=f'original shard {name} size', minimum=1)
        _same(row, {'path': str(Path(base['model']) / name), 'size': coordinates[name].size,
                    'sha256': coordinates[name].sha256}, f'original shard {name}')
    _source_execution(authority['source_execution'])
    calibration = _full_calibration(authority['calibration'])
    _same(base['calibration_input']['sha256'], calibration['artifact_sha256'], 'bound full calibration artifact')
    _binding(base['calibration_input'], 'base full calibration input')
    _, runtime = _control(authority['runtime'], 'original runtime')
    validate_original_source_runtime(admitted_execution['runtime'], runtime)
    _same(runtime['config'], source['config'], 'original runtime/source resolved config')
    _, resources = _control(authority['resources'], 'original resources')
    _resources(resources)
    _same(resources, admitted_execution['resources'], 'independently observed source resources')
    _same(plan.get('source_prefetch'), resources['source_prefetch'], 'final source prefetch')
    sdk, claim = _environment_matches(runtime)
    claimed = sdk.read_claimed_record(sdk.PoolQueue(claim['queue_root']), claim['action_key'])
    _require(isinstance(claimed, dict), 'original source has no active native claim')
    _same(claimed.get('resources'), resources['claim_demand'], 'native reservation/resource demand')
    actual = _load_sealed_payload(bindings['read_manifest']['sha256'])
    active = {(row['path'], row['offset']): row for row in actual['entries']}
    for row in coordinates.values():
        entry = active.get((row.path, 0))
        _require(entry is not None and entry['bytes'] == row.size and entry['sha256'] == row.sha256,
                 'original source whole-file readset differs from active enclosing manifest')
    declared_authority = active.get((authority_input['path'], 0))
    _require(declared_authority is not None and declared_authority['sha256'] == authority_input['sha256']
             and declared_authority['bytes'] == Path(authority_input['path']).stat().st_size,
             'full original authority input is not independently declared by the active manifest')
    _session(authority['session'])
    _same(authority['session'], admitted_execution['session'], 'owning original source session')
    identity = original_diagnostic_session_identity(base_plan=base,
        base_plan_sha256=bindings['base_plan']['sha256'], prepared=prepared,
        execution_sha256=bindings['execution']['sha256'])
    _exact(identity, ORIGINAL_SESSION_IDENTITY_KEYS, 'original session identity')
    _same(identity, admitted_execution['session_identity'], 'independently owning original session identity')
    _same(canonical_json_sha256(identity, where='original source session identity'),
          authority['session']['run_identity_sha256'], 'owning original session digest')
    if owner is not None:
        from .tessera_calibration_cache import CaptureSourceAuthentication

        _require(isinstance(owner, CaptureSourceAuthentication) and owner.is_qualified_original_material,
                 'original authority requires the existing original source owner')
        with owner._lock:
            owner._require_open()
            owned = json.loads(owner._original['inputs_json'])
            _same(owner.root, Path(base['model']), 'original authority same source owner root')
            _require(owner.resource_check is check, 'original authority resource check belongs to another owner')
            _same(owner._original['limit'], resources['material_bytes'], 'original owner material envelope')
            _same(owned, {'publisher': authority['publisher'], 'producer_source': producer,
                         'readset': authority['readset'], 'source_paths': paths}, 'original constructor controls')
            _same(dict(owner._original['coordinates']), dict(coordinates), 'original owner coordinates')
            descriptor = owner.original_checkpoint_descriptor()
            _same(descriptor['index']['weight_map'], tensors, 'owned complete checkpoint index')
            _same(descriptor['shards'], [dict(name=name, **shards[name]) for name in sorted(shards)],
                  'owned complete shard metadata')
    from .stage_a_selected_row_diagnostic import load_original_diagnostic_issued_context

    context = load_original_diagnostic_issued_context(plan.get('original_session_preparation'),
        base_plan_input=bindings['base_plan'], authority=authority)
    _same(context['session_preparation']['session'], admitted_execution['session'],
          'actual native issued original source session')
    _same(context['session_identity'], identity, 'actual native issued run identity')
    return _snapshot(authority)


def _verified_original_result(selection, resource_check):
    """Require SDK5's selected native producer owner, never queue/consumer guesses."""
    from .staged_lease import client_sdk
    from .digests import bytes_sha256hex

    _exact(selection, ORIGINAL_RESULT_SELECTION_KEYS, 'original qualification action selection')
    _contract.absolute_posix_path(selection['queue_root'], where='qualification queue')
    _contract.sha256(selection['action_key'], where='qualification action key')
    _contract.sha256(selection['payload_sha256'], where='qualification CAS payload')
    for key in ('attempt', 'max_result_bytes', 'max_evidence_bytes'):
        _contract.integer(selection[key], where=f'qualification {key}', minimum=1)
    _require(type(selection['published_unix']) in (int, float) and
             math.isfinite(selection['published_unix']), 'qualification needs exact finite publication')
    inputs = selection['source_inputs']
    _require(isinstance(inputs, list) and inputs and len(inputs) <= 64,
             'qualification requires the complete declared source-input bindings')
    ids = set()
    for row in inputs:
        _exact(row, {'id', 'sha256', 'bytes'}, 'qualification source input')
        _contract.string(row['id'], where='qualification source input ID')
        _contract.sha256(row['sha256'], where='qualification source input SHA256')
        _contract.integer(row['bytes'], where='qualification source input bytes', minimum=1)
        _require(row['id'] not in ids, 'qualification repeats a declared source input')
        ids.add(row['id'])
    source_bytes = sum(row['bytes'] for row in inputs)
    _require(source_bytes <= selection['max_evidence_bytes'],
             'qualification source inputs exceed the independent evidence envelope')
    reserve_allocation(resource_check, 'before_original_qualification_result',
        cpu_bytes=selection['max_evidence_bytes'] + selection['max_result_bytes'] + source_bytes)
    _, expected_request = _control(selection['request'], 'qualification sealed request')
    sdk = client_sdk()
    result = sdk.read_verified_action_result(sdk.PoolQueue(selection['queue_root']), selection['action_key'],
        published_unix=selection['published_unix'], attempt=selection['attempt'],
        max_result_bytes=selection['max_result_bytes'], max_evidence_bytes=selection['max_evidence_bytes'],
        input_limits={row['id']: row['bytes'] for row in inputs},
        require_native_producer_context=True)
    _same(result['request'], expected_request, 'selected original qualification request')
    _same(bytes_sha256hex(result['payload']), selection['payload_sha256'], 'selected original qualification payload')
    _same(result['inputs'], inputs, 'selected original qualification source-input roster')
    _require(set(result['input_payloads']) == ids, 'qualified CAS source input payloads are incomplete')
    # Bind the ACTUAL executable wrapper, not params.command asserted alone.
    sdk.bind_standard_capture_command(result['request'])
    return result


ORIGINAL_ARTIFACT_PUBLICATION_PREFIX = b'ORIGINAL_SOURCE_ARTIFACTS '
ORIGINAL_CUDA_ARTIFACT_ROLES = frozenset({
    'control', 'execution', 'action_result', 'netdata_sparky', 'netdata_sparklina', 'torch_trace',
})


def _original_artifact_publication(payload, *, node_id, roles):
    """Interpret the one actual producer publication in authenticated CAS bytes."""
    _require(type(payload) is bytes, 'qualified result payload must be owned CAS bytes')
    lines = [line[len(ORIGINAL_ARTIFACT_PUBLICATION_PREFIX):] for line in payload.splitlines()
             if line.startswith(ORIGINAL_ARTIFACT_PUBLICATION_PREFIX)]
    _require(len(lines) == 1, 'selected result lacks exactly one actual artifact publication; old receipts remain unqualified')
    try:
        publication = strict_json_loads(lines[0],
            duplicate=lambda key: RuntimeError(f'artifact publication duplicate key {key}'),
            constant=lambda key: RuntimeError(f'artifact publication invalid constant {key}'))
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError('selected artifact publication is not strict JSON') from exc
    _exact(publication, {'schema', 'node_id', 'artifacts'}, 'selected artifact publication')
    _same(publication['schema'], 'prismaquant.original_source_artifact_publication.v1', 'actual artifact publication schema')
    _same(publication['node_id'], node_id, 'actual artifact publication selected node')
    _same(lines[0], DIRECT_ASCII_STRICT.encoded(publication), 'canonical actual artifact publication')
    artifacts = _exact(publication['artifacts'], roles, 'actual published artifact roles')
    paths = set()
    for role, artifact in artifacts.items():
        _exact(artifact, {'path', 'sha256', 'bytes'}, f'actual published {role}')
        _contract.absolute_posix_path(artifact['path'], where=f'actual published {role} path')
        _contract.sha256(artifact['sha256'], where=f'actual published {role} SHA256')
        _contract.integer(artifact['bytes'], where=f'actual published {role} bytes', minimum=1)
        _require(artifact['path'] not in paths, 'artifact publication reuses a path for different roles')
        paths.add(artifact['path'])
    return artifacts


def _published_original_artifact(artifacts, role, binding, *, resource_check, max_bytes, decode_json=True):
    """Join independently selected bytes to the digest/length the result published."""
    artifact = artifacts[role]
    _same(_binding(binding, f'independent qualified {role}'),
          {key: artifact[key] for key in ('path', 'sha256')}, f'selected published {role} binding')
    _require(artifact['bytes'] <= max_bytes, f'published {role} exceeds the qualified evidence envelope')
    _same(Path(binding['path']).stat().st_size, artifact['bytes'], f'actual published {role} length')
    reserve_allocation(resource_check, f'before_original_published_artifact:{role}', cpu_bytes=2 * artifact['bytes'])
    raw = read_bound(binding, f'actual published {role}')
    _same(len(raw), artifact['bytes'], f'owned actual {role} length')
    if not decode_json:
        return raw
    try:
        return strict_json_loads(raw,
            duplicate=lambda key: RuntimeError(f'published {role}: duplicate key {key}'),
            constant=lambda key: RuntimeError(f'published {role}: invalid constant {key}'))
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError(f'published {role} is not strict JSON') from exc


def _published_original_json(artifacts, role, binding, selection, resource_check):
    return _published_original_artifact(artifacts, role, binding, resource_check=resource_check,
                                        max_bytes=selection['max_evidence_bytes'])


def _require_original_qualified_source(row, request, accepted, target_runtime):
    """Every selected member needs independently selected source-transfer proof.

    A null compatibility binding is not proof that the executed implementation
    equals the target. The existing acceptance can describe identical sources,
    but still binds the actual executed snapshot and target package/runtime.
    """
    snapshot = request['params']['checkout_snapshot']
    _same(snapshot['parent'], row['source_snapshot'], 'original actual qualified source snapshot')
    _require(row['compatibility'] is not None,
             'every qualified member requires independently bound executed-to-target source acceptance')
    family = accepted.get(row['node_id'])
    _require(family is not None, 'qualified member lacks independently selected source-family acceptance')
    _same(row['compatibility'], family['compatibility'], 'original source-family proof binding')
    _recorded_same(row['source_snapshot'], family['old_source'],
                   'original executed member source is not restamped')
    _same(family['target_prismaquant_source_sha256'], target_runtime['prismaquant_source_sha256'],
          'qualified member actual target source implementation')
    _same(family['target_runtime_sha256'], canonical_json_sha256(target_runtime, where='actual original target runtime'),
          'qualified member actual target runtime')


def _require_original_reader_target(reader_authority, authority):
    """A reader's own authority cannot substitute another target's source axes.

    Runtime, resource and artifact-session identities belong to the producer
    and may differ from the later consumer; they are joined separately below.
    """
    _exact(reader_authority, ORIGINAL_AUTHORITY_KEYS, 'reader proof source authority')
    for key in ('schema', 'scope', 'publisher', 'producer', 'source_paths', 'readset',
                'calibration', 'source_model_identity', 'source_execution'):
        _same(reader_authority[key], authority[key], f'qualified reader target {key}')


def _helper_tree(helper_root):
    """The package and complete tree digests of the helper generation on disk."""
    from .production_weight_cache import _production_cache_source_sha256

    return {'package_sha256': _production_cache_source_sha256(helper_root / 'src' / 'prismabuild'),
            'helper_tree_sha256': _production_cache_source_sha256(helper_root)}


def _require_original_reader_producer(receipt, reader_authority, result, expected_runtime_input):
    """Join material observations to the SDK-owned selected producer context.

    The SDK owns native provenance verification. This only joins its result
    to Original domain controls; it never looks up a queue or a live claim.
    """
    context = result.get('producer_context')
    _require(isinstance(context, dict), 'qualified reader requires selected native producer context')
    _same(context['attempt_source'], 'selected-immutable-attempt', 'selected reader context provenance')
    _same(context['resources_semantics'], 'selected-claim-sealed-demand', 'selected reader reservation semantics')
    for key in ('action_key', 'published_unix', 'attempt', 'generation', 'host'):
        _same(context[key], result[key], f'selected reader producer {key}')
    _same(context['worker'], result['worker_id'], 'selected reader full worker identity')
    _same(context['incarnation'], context['worker'], 'selected reader full incarnation')
    _same(context['receipt_sha256'], result['receipt']['receipt_sha256'], 'selected reader execution receipt')
    _same(context['runtime_sha256'], result['receipt']['producer']['runtime']['runtime_sha256'],
          'selected reader attested runtime')
    _, runtime = _control(reader_authority['runtime'], 'selected reader observed runtime')
    _, expected_runtime = _control(expected_runtime_input, 'independent selected reader runtime')
    _exact(expected_runtime, ORIGINAL_RUNTIME_KEYS, 'independent selected reader runtime')
    expected_pb = _exact(expected_runtime['prismabuild'],
        {'sdk_version', 'helper_root', 'source_tree', 'runtime_generation'}, 'independent selected reader SDK policy')
    _contract.integer(expected_pb['sdk_version'], where='independent selected reader SDK version', minimum=1)
    runtime = _validate_original_source_runtime(runtime, expected_runtime, sdk_version=expected_pb['sdk_version'])
    _same(runtime['config'], reader_authority['source_model_identity']['config'],
          'selected reader runtime source config')
    # From here on the reader's record of its producer run meets the identity
    # the SDK selected for it: D32 seals, stamped in dev mode.
    _recorded_same(runtime['prismabuild']['helper_root'], context['helper_root'],
                   'selected reader actual helper root')
    helper_root = Path(context['helper_root'])
    _recorded_same(runtime['prismabuild']['runtime_generation'], helper_root.name,
                   'selected reader actual helper generation')
    _, resources = _control(reader_authority['resources'], 'selected reader observed resources')
    resources = _resources(resources)
    _recorded_same(resources['claim_demand'], context['resources'],
                   'selected reader actual producer reservation')
    identity_keys = ('queue_root', 'action_key', 'nonce', 'scope_id', 'worker', 'host',
                     'incarnation', 'helper_root')
    for row in receipt['deliveries']:
        claim = row['native_delivery']['claim']
        _recorded_same({key: claim[key] for key in identity_keys},
                       {key: context[key] for key in identity_keys},
                       'actual reader delivery selected producer')
        _recorded_same(claim['attempt_source'], 'launch-env', 'actual reader delivery launch provenance')
    # Dev mode hashes no existing tree only to seal a run: NOT_COMPUTED stands for
    # the tree on disk, beside the tree the reader recorded.
    _recorded_same(NOT_COMPUTED if dev_mode_enabled() else _helper_tree(helper_root),
                   runtime['prismabuild']['source_tree'], 'selected reader actual complete helper tree')



_CUDA_CASES = frozenset({
    'layer', 'read-failure', 'copy-failure', 'cancel', 'event-record-failure',
    'event-sync-failure', 'head', 'dequant', 'parallel-layer', 'parallel-read-failure',
    'parallel-copy-failure', 'parallel-cancel', 'parallel-event-record-failure',
    'parallel-event-sync-failure', 'abandoned-owner-record', 'abandoned-owner-sync',
})
_CUDA_CONTROL_KEYS = frozenset({
    'schema', 'case', 'pages', 'source_dtype', 'reader_threads', 'copy_streams', 'parallel',
    'original_owner_type', 'device', 'source_control_override', 'automatic_capture_qualified',
    'actual_glm', 'observations', 'source_receipt', 'source_receipt_phase', 'torch', 'cuda',
})
_CUDA_FATAL_KEYS = frozenset({
    'schema', 'case', 'pages', 'source_dtype', 'source_receipt_before_recovery', 'held_fd_bytes',
    'pending_after_copy', 'pending_after_owner_disposal', 'failed_drains', 'completed_drains',
    'exact_output_parity', 'final_owner_collected', 'final_fd_closed', 'source_control_override',
    'automatic_capture_qualified', 'actual_glm', 'torch', 'cuda',
})
_CUDA_OBSERVATION_KEYS = frozenset({
    'copies', 'pending_events', 'completed_events', 'decoder_reads', 'credit_before_sync',
    'credit_after_sync', 'pending_stream_on_failure', 'source_dtype',
    'delayed_copies_pending_on_return', 'reader_attempts', 'readers_entered', 'readers_exited', 'stream_drains',
})


def _actual_cuda_control(control, execution, node):
    case = control.get('case') if isinstance(control, dict) else None
    _require(case in _CUDA_CASES, 'qualification has unknown actual CUDA case')
    fatal = case.startswith('abandoned-owner-')
    _exact(control, _CUDA_FATAL_KEYS if fatal else _CUDA_CONTROL_KEYS, 'actual CUDA control')
    _same(control['schema'], 'prismaquant.original_copy_cuda_control.v2', 'actual CUDA control schema')
    _require(control['pages'] in ('0', '1') and control['source_dtype'] in ('torch.float32', 'torch.bfloat16'),
             'actual CUDA source dtype/page identity missing')
    _require(control['automatic_capture_qualified'] is False and control['actual_glm'] is False,
             'actual primitive control cannot claim automatic/original GLM qualification')
    material_index = 0 if control['source_dtype'] == 'torch.float32' else 1
    function = ('test_actual_double_fence_failure_survives_abandoned_owner' if fatal
                else 'test_actual_original_copy_stream_ownership')
    parameter = case.removeprefix('abandoned-owner-') if fatal else case
    _same(node, f'tests/test_original_source_copy_completion_cuda.py::{function}'
               f'[{parameter}-material{material_index}-{control["pages"]}]',
          'actual control/pytest node coordinate join')
    _exact(execution, {'schema', 'passed', 'cpu_preflight', 'node_id', 'returncode', 'collected',
        'expected', 'reports', 'automatic_capture_qualified', 'actual_glm', 'torch', 'cuda',
        'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes'}, 'actual CUDA execution')
    _same(execution['schema'], 'prismaquant.original_cuda_fixture_execution.v1', 'actual CUDA execution schema')
    _require(execution['passed'] is True and execution['cpu_preflight'] is False
             and type(execution['returncode']) is int and execution['returncode'] == 0
             and type(execution['collected']) is int and execution['collected'] == 1
             and type(execution['expected']) is int and execution['expected'] == 1,
             'actual CUDA execution did not enter and pass exactly one selected control')
    _same(execution['node_id'], node, 'actual CUDA selected node')
    _same(execution['reports'], [{'node_id': node, 'phase': 'call', 'outcome': 'passed'}],
          'actual CUDA call report; skips/setup-only do not qualify')
    for key in ('automatic_capture_qualified', 'actual_glm', 'torch', 'cuda'):
        _same(execution[key], control[key], f'actual CUDA execution/control {key}')
    if fatal:
        for key in ('pending_after_copy', 'pending_after_owner_disposal', 'exact_output_parity',
                    'final_owner_collected', 'final_fd_closed'):
            _require(control[key] is True, f'actual fatal completion lacks observed {key}')
        _same(control['failed_drains'], 3, 'actual failed fatal fences')
        _same(control['completed_drains'], 1, 'actual completed fatal fence')
        _contract.integer(control['held_fd_bytes'], where='actual fatal held descriptor bytes', minimum=1)
    else:
        extra = {'completed_streams_after_traceback_disposal'} if 'event-' in case else set()
        observation = _exact(control['observations'], _CUDA_OBSERVATION_KEYS | extra, 'actual CUDA observations')
        _same(observation['source_dtype'], control['source_dtype'], 'actual decoded source dtype')
        for key in ('copies', 'decoder_reads', 'delayed_copies_pending_on_return'):
            _contract.integer(observation[key], where=f'actual CUDA {key}', minimum=1)
        _same(observation['readers_entered'], observation['readers_exited'], 'actual reader cleanup')
        _same(control['parallel'], case.startswith('parallel-'), 'actual reader concurrency identity')
        _same(control['device'], 'cuda:0', 'actual indexed CUDA device')
        if extra:
            _require(observation['pending_stream_on_failure'] is True, 'actual event failure had no pending work')
            _same(observation['completed_events'], 0, 'failed event is not a successful completion')
            _same(observation['completed_streams_after_traceback_disposal'], control['copy_streams'],
                  'actual traceback-independent stream completion')
            _same(observation['stream_drains'], control['copy_streams'], 'actual exact stream drains')
        else:
            for key in ('pending_events', 'completed_events'):
                _contract.integer(observation[key], where=f'actual CUDA {key}', minimum=1)
        _same(control['source_receipt_phase'], 'after_successful_copy_or_failure_drain', 'actual completion phase')
        _same(control['source_receipt']['material_live_bytes'], 0, 'actual source credit after completion')
    return (case, control['pages'], control['source_dtype'])


def _require_original_source_proofs(authority, resource_check):
    """No pending/partial/fixture64 record can satisfy public source authority."""
    _require(authority['qualification'] is not None,
             'original source requires independently bound actual full64 CUDA and reader qualification')
    _, qualification = _control(authority['qualification'], 'original source qualification')
    _exact(qualification, {'schema', 'reader', 'cuda', 'unchanged_family'}, 'original source qualification')
    _same(qualification['schema'], 'prismaquant.original_source_qualification.v1', 'original qualification schema')
    controls = qualification['cuda']
    _require(isinstance(controls, list) and len(controls) == 64, 'original CUDA qualification is missing/partial full64')
    _require(authority['root_admission'] is not None,
             'original source requires independent root matched-source admission')
    _, target_runtime = _control(authority['runtime'], 'qualified target runtime')
    _require(target_runtime['prismabuild']['source_tree']['helper_tree_sha256'] is not None,
             'original source requires actual shared helper/worker/proxy generation proof')
    _, admission = _control(authority['root_admission'], 'original root admission')
    _exact(admission, {'schema', 'scope', 'static_authority_sha256', 'session', 'runtime', 'resources',
                       'qualification', 'matched_source'}, 'original root admission')
    _same(admission['schema'], 'prismaquant.original_source_admission.v1', 'original root admission schema')
    for key in ('scope', 'session', 'runtime', 'resources', 'qualification'):
        _same(admission[key], authority[key], f'root matched original {key}')
    _same(admission['static_authority_sha256'], original_authority_static_sha256(authority), 'root static authority')
    _, matched = _control(admission['matched_source'], 'root matched-source protocol')
    matched_keys = ORIGINAL_STATIC_AUTHORITY_KEYS - {'scope'}
    _exact(matched, matched_keys, 'root matched-source protocol')
    _same(matched['schema'], 'prismaquant.original_matched_source_protocol.v1', 'root matched-source schema')
    for key in matched_keys - {'schema'}:
        _same(matched[key], authority[key], f'root matched-source {key}')
    families = qualification['unchanged_family']
    _require(isinstance(families, list), 'original unchanged-family acceptance must be explicit')
    accepted = {}
    for binding in families:
        _, family = _control(binding, 'original unchanged-family root acceptance')
        _exact(family, {'schema', 'old_source', 'new_source', 'compatibility', 'controls',
                        'target_prismaquant_source_sha256', 'target_runtime_sha256'},
               'original unchanged-family acceptance')
        _same(family['schema'], 'prismaquant.original_source_unchanged_family.v1', 'unchanged-family schema')
        for key in ('old_source', 'new_source'):
            _contract.string(family[key], where=f'unchanged-family {key}', pattern=_GIT_OBJECT_ID)
        _same(family['target_prismaquant_source_sha256'], target_runtime['prismaquant_source_sha256'],
              'unchanged-family actual target source implementation')
        _same(family['target_runtime_sha256'], canonical_json_sha256(target_runtime, where='actual original target runtime'),
              'unchanged-family actual target runtime')
        _control(family['compatibility'], 'independently accepted unchanged source compatibility')
        _require(isinstance(family['controls'], list) and family['controls'], 'unchanged-family accepted controls missing')
        for node in family['controls']:
            _contract.string(node, where='unchanged-family accepted original node')
            _require(node not in accepted, 'unchanged-family acceptance repeats an original node')
            accepted[node] = family
    seen = set()
    for row in controls:
        _exact(row, {'node_id', 'source_snapshot', 'control', 'execution', 'action_result', 'result',
                     'compatibility'}, 'original qualified CUDA member')
        _contract.string(row['node_id'], where='original actual CUDA node')
        _contract.string(row['source_snapshot'], where='original qualified snapshot', pattern=_GIT_OBJECT_ID)
        result = _verified_original_result(row['result'], resource_check)
        command = result['request']['params']['command']
        _require('--node-id' in command and command.index('--node-id') + 1 < len(command)
                 and command[command.index('--node-id') + 1] == row['node_id'],
                 'actual CUDA request executed a different control')
        artifacts = _original_artifact_publication(result['payload'], node_id=row['node_id'],
                                                   roles=ORIGINAL_CUDA_ARTIFACT_ROLES)
        control = _published_original_json(artifacts, 'control', row['control'], row['result'], resource_check)
        execution = _published_original_json(artifacts, 'execution', row['execution'], row['result'], resource_check)
        member = _actual_cuda_control(control, execution, row['node_id'])
        for key in ('torch', 'cuda'):
            _same(control[key], target_runtime['versions'][key], f'qualified actual source runtime {key}')
        _require(member not in seen, 'original full64 repeats an actual CUDA member')
        seen.add(member)
        ending = _published_original_json(artifacts, 'action_result', row['action_result'], row['result'], resource_check)
        _exact(ending, {'returncode', 'start_unix', 'finish_unix', 'cpu_preflight', 'netdata',
                       'netdata_errors', 'torch_trace', 'torch_trace_errors', 'automatic_capture_qualified',
                       'actual_glm'}, 'actual CUDA controller ending')
        _require(type(ending['returncode']) is int and ending['returncode'] == 0
                 and ending['cpu_preflight'] is False and not ending['netdata_errors']
                 and not ending['torch_trace_errors'] and isinstance(ending['torch_trace'], dict),
                 'actual CUDA controller/trace/native telemetry did not complete')
        trace = artifacts['torch_trace']
        _same(ending['torch_trace'], trace, 'selected controller/native trace artifact')
        _published_original_artifact(artifacts, 'torch_trace',
            {key: trace[key] for key in ('path', 'sha256')}, resource_check=resource_check,
            max_bytes=row['result']['max_evidence_bytes'], decode_json=False)
        _same({record['host'] for record in ending['netdata']}, {'sparky', 'sparklina'},
              'selected actual raw telemetry hosts')
        for host in ('sparky', 'sparklina'):
            role = 'netdata_' + host
            artifact = artifacts[role]
            raw_host = _published_original_json(artifacts, role,
                {key: artifact[key] for key in ('path', 'sha256')}, row['result'], resource_check)
            _same(raw_host['host'], host, 'selected native telemetry producer host')
            actual_host = next(record for record in ending['netdata'] if record['host'] == host)
            _same(actual_host['charts'], len(raw_host['charts']), 'selected actual raw host chart census')
            _require(raw_host['after'] <= ending['start_unix'] and raw_host['before'] <= ending['finish_unix']
                     and raw_host['after'] < raw_host['before'], 'selected native telemetry window differs from controller')
        _require_original_qualified_source(row, result['request'], accepted, target_runtime)
    _same(seen, {(case, pages, dtype) for case in _CUDA_CASES for pages in ('0', '1')
                 for dtype in ('torch.float32', 'torch.bfloat16')}, 'actual full64 CUDA coverage')
    reader = _exact(qualification['reader'], {'node_id', 'result', 'receipt', 'authority',
                    'source_snapshot', 'compatibility', 'runtime'}, 'original qualified reader')
    _contract.string(reader['node_id'], where='actual qualified reader node')
    _contract.string(reader['source_snapshot'], where='actual reader source snapshot', pattern=_GIT_OBJECT_ID)
    reader_result = _verified_original_result(reader['result'], resource_check)
    _require(reader['node_id'] in reader_result['request']['params']['command'],
             'selected native reader request executed another proof')
    artifacts = _original_artifact_publication(reader_result['payload'], node_id=reader['node_id'],
                                               roles={'receipt', 'authority'})
    receipt = _published_original_json(artifacts, 'receipt', reader['receipt'], reader['result'], resource_check)
    reader_authority = _published_original_json(artifacts, 'authority', reader['authority'], reader['result'], resource_check)
    _require_original_reader_target(reader_authority, authority)
    _require_original_qualified_source(reader, reader_result['request'], accepted, target_runtime)
    from .tessera_calibration_cache import validate_original_source_material_receipt

    validate_original_source_material_receipt(receipt, reader_authority)
    _require_original_reader_producer(receipt, reader_authority, reader_result, reader['runtime'])
    _require(receipt['deliveries'] and receipt['material_live_bytes'] == 0
             and not receipt['pending_copy_completions'], 'qualified reader retains unproved material/copy debt')
    return _snapshot(authority)
