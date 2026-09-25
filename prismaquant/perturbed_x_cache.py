"""Build activation caches under a perturbed allocation.

The regular probe cache captures BF16 model inputs. Perturbed-X iterations need
the same cache shape after upstream layers have already run with the current
allocation's weight and activation quantization. This module installs one
forward_pre_hook per quantized module: it snapshots the original input first,
then returns the activation-quantized input for the actual forward. Weights are
RTN-quantized just for that module call and restored in the forward hook.
"""
from __future__ import annotations

import hashlib
import io
import math
import json
import os
import pickle
import pickletools
import re
import stat
import struct
import sys
import threading
import zipfile
from collections import OrderedDict, defaultdict
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait as wait_futures
from contextlib import contextmanager
from functools import partial
from dataclasses import dataclass, field, replace as _dataclass_replace
from pathlib import Path
from typing import Iterator, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant import format_registry as fr
from prismaquant.build_rtn_cache import iter_quantizable_tensors
from prismaquant.memory_management import (
    enforce_gpu_memory_budget,
    env_int,
    env_truthy as _env_truthy,
    model_device as _model_device,
    register_budget_evictor,
)
from prismaquant.nvfp4_activation_contract import (
    require_matching_input_global_scale,
)

_FNAME_SUB = re.compile(r"[^A-Za-z0-9_-]")
_SHARED_FROZEN_WEIGHT_FORMAT_CACHE: OrderedDict[
    tuple[str, str, int, str, str],
    torch.Tensor,
] = OrderedDict()


# Clamping inputs to the calibrated max(|activations|) before per-group
# RTN matches the export's act-clip behavior.  Without this, dynamic
# per-group RTN sets scales from the raw input's local max, so outliers
# dominate and any pre-scaling is mathematically a no-op
# (Q(x/s)*s == Q(x) under purely dynamic Q — codex round-3 caught this).


def _activation_max_abs_lookup(
    activation_max_abs: dict,
    param_name: str | None,
) -> float | None:
    """Resolve ``param_name`` against ``activation_max_abs`` with the same
    alias-fallbacks as ``ProductionWeightCache.get`` so cache hits and
    activation-clip lookups stay consistent."""
    if param_name is None or not activation_max_abs:
        return None
    candidates = [param_name]
    if param_name.endswith(".weight"):
        candidates.append(param_name[:-len(".weight")])
    if param_name.startswith("model.language_model."):
        candidates.append("model." + param_name[len("model.language_model."):])
    for cand in candidates:
        v = activation_max_abs.get(cand)
        if v is not None:
            return v
    return None


def _maybe_clip_activations(
    x: "torch.Tensor",
    activation_max_abs: dict,
    param_name: str | None,
) -> "torch.Tensor":
    """Clamp activations to ±max_abs when a calibrated value is known.

    ``activation_max_abs`` is the dict from
    ``ProductionWeightCache.activation_max_abs`` (calibrated max(|x|)
    per fused-sibling group).  Returns ``x`` unchanged when:

      * no entry is registered for ``param_name`` (or its aliases),
      * the registered value is non-positive, or
      * ``PRISMAQUANT_PROD_ACT_SCALES`` is explicitly disabled.
    """
    max_abs = _activation_max_abs_lookup(activation_max_abs, param_name)
    if max_abs is None or max_abs <= 0:
        return x
    if not _env_truthy("PRISMAQUANT_PROD_ACT_SCALES", default=True):
        return x
    return x.clamp(-float(max_abs), float(max_abs))



def _served_nvfp4_act_qdq_enabled() -> bool:
    """Opt-in serve-faithful NVFP4 activation emulation (default OFF).

    When on, activation quantization in the emulation hooks for a spec whose
    served contract is static-scale (``FormatSpec.static_activation_contract``,
    i.e. stock NVFP4) models the SERVED two-level semantics (static
    input_global_scale + FP8 snap of the per-16-group block scale, via the
    contract's own oracle) instead of the dynamic exact-fp32-scale RTN.
    Closes the M18-residual/C1 measurement gap the 2026-07-02 audit flagged;
    default-off pending a served correlation study (the dynamic path is the
    long-standing screen baseline).  A spec whose contract says
    ``measured_as_served`` (a Tessera W4A4 rung) does not consult this lever:
    the served oracle is its only measurement."""
    return os.environ.get(
        "PRISMAQUANT_NVFP4_ACT_EMULATE_SERVED_SCALES", "0") == "1"


def _activation_qdq(
    x: torch.Tensor,
    act_spec,
    activation_max_abs: dict,
    param_name: str | None,
    priced_input_global_scales: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Shared activation quantize-dequantize for the emulation hooks.

    Which quantizer a spec serves is the SPEC's answer
    (``FormatSpec.static_activation_contract``), never a compare of its name
    against ``"NVFP4"`` -- a Tessera rung routed through the same kernel has
    the same contract and a different name (#205).

    * No static contract (FP8/MX dynamic W8A8, or an A16 row that reached
      the hook): act-clip to the calibrated max_abs, then the row's own
      dynamic quantizer.
    * Static contract, ``measured_as_served`` (Tessera W4A4): the served
      oracle at the unit's G -- NO clamp (serving does not clamp; the static
      scale itself clips blocks above the calibration amax) -- and a refusal
      by name when the unit has no calibrated maximum.
    * Static contract, screen default (stock NVFP4): the historical clip +
      dynamic RTN, or the served oracle when
      ``PRISMAQUANT_NVFP4_ACT_EMULATE_SERVED_SCALES=1`` and the maximum is
      known.

    ``priced_input_global_scales`` is the G each unit's cached render score was
    priced at (``ProductionWeightCache`` ``render_scores`` provenance, via
    ``production_cache_priced_input_global_scales``).  When one is known it is
    compared against the G this hook is about to apply, and a disagreement
    refuses by name: measuring an assignment whose costs were priced under one
    activation-scale policy through a hook quantizing under another compares
    two different quantizers (#227).  Absent -- no production cache, or a cache
    with no served rows -- there is nothing to disagree with and the hook is
    unchanged."""
    from .joint_served_activation import format_activation_maxima
    activation_max_abs = format_activation_maxima(activation_max_abs, act_spec)
    contract = getattr(act_spec, "static_activation_contract", None)
    if contract is not None:
        max_abs = _activation_max_abs_lookup(activation_max_abs, param_name)
        if contract.measured_as_served:
            g = contract.require_input_global_scale(
                max_abs, qname=param_name, consumer="assignment-KL hook")
            g = require_matching_input_global_scale(
                _activation_max_abs_lookup(
                    priced_input_global_scales or {}, param_name),
                g,
                qname=param_name,
                consumer="assignment-KL hook",
            )
            return contract.quantize_dequantize(x, g)
        if (
            _served_nvfp4_act_qdq_enabled()
            and x.shape[-1] % int(contract.group_size) == 0
            and max_abs is not None and max_abs > 0
        ):
            g = contract.input_global_scale_from_max_abs(float(max_abs))
            return contract.quantize_dequantize(x, g)
    x = _maybe_clip_activations(x, activation_max_abs, param_name)
    return act_spec.activation_quantize_dequantize(x)


def activation_cache_filename(name: str) -> str:
    return _FNAME_SUB.sub("__", name) + ".pt"


class SerializedEntryDigest:
    """Hash a serializer's output while it is written, never on a second pass.

    Torch's buffer writer only calls ``write`` and ``flush`` and never seeks,
    so the bytes hashed here are exactly the bytes the temporary file receives
    and the atomic rename publishes. The durable fence still fsyncs the real
    descriptor, which this sink does not own.
    """

    def __init__(self, *, max_bytes=None):
        self.max_bytes = max_bytes
        self.file_identity = None
        self._hash = hashlib.sha256()
        self._handle = None
        self.bytes = 0

    def sink(self, handle):
        self._handle = handle
        return self

    def write(self, data):
        view = memoryview(data).cast("B")
        try:
            if self.max_bytes is not None and self.bytes + view.nbytes > self.max_bytes:
                raise RuntimeError("serialized entry exceeds its preallocated ceiling")
            self._hash.update(view)
            self.bytes += view.nbytes
            return self._handle.write(view)
        finally:
            view.release()

    def flush(self):
        self._handle.flush()

    def hexdigest(self):
        return self._hash.hexdigest()


def write_activation_cache_entry(cache_dir, name, inputs, *, source="perturbed_x",
                                 durable=False, serialized_digest=None,
                                 preallocate_bytes=None, **metadata):
    """Atomically store already-selected rows without changing their precision."""
    import os
    path = Path(cache_dir) / activation_cache_filename(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    with temporary.open("xb" if preallocate_bytes is not None else "wb") as handle:
        if preallocate_bytes is not None:
            if type(preallocate_bytes) is not int or preallocate_bytes <= 0:
                raise ValueError("preallocated entry ceiling must be positive")
            if serialized_digest is None or serialized_digest.max_bytes != preallocate_bytes:
                raise ValueError("preallocation requires the matching bounded digest sink")
            created = os.fstat(handle.fileno())
            serialized_digest.file_identity = (created.st_dev, created.st_ino)
            os.posix_fallocate(handle.fileno(), 0, preallocate_bytes)
        torch.save({**metadata, "inputs": inputs.contiguous(), "name": name,
                    "source": source},
                   handle if serialized_digest is None else serialized_digest.sink(handle))
        if preallocate_bytes is not None:
            handle.flush()
            handle.truncate(serialized_digest.bytes)
        if durable:
            handle.flush()
            os.fsync(handle.fileno())
    if preallocate_bytes is None:
        os.replace(temporary, path)
    else:
        # Publish this precommit inode without overwriting an unexpected
        # destination created after the initial existence check.
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
    if durable:
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return path


def cache_file_stat_signature(value):
    """Stable file identity shared by the existing activation and PWC owners."""
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _preflight_torch_zip_directory(source, *, metadata_cap, label):
    """Bound directory objects before ZipFile creates any ZipInfo instances.

    Reuse zipfile's fixed-size EOCD/ZIP64 reader, then walk fixed-size central
    headers without decoding names or allocating a roster. ZipFile ignores the
    EOCD entry count when building its roster, so both count and extent matter.
    """
    source.seek(0, os.SEEK_END)
    file_bytes = source.tell()
    end = zipfile._EndRecData(source)
    if end is None:
        raise RuntimeError(f'{label} has no accountable ZIP directory')
    count = end[zipfile._ECD_ENTRIES_TOTAL]
    size = end[zipfile._ECD_SIZE]
    offset = end[zipfile._ECD_OFFSET]
    if (end[zipfile._ECD_DISK_NUMBER] != 0 or end[zipfile._ECD_DISK_START] != 0 or
            count != end[zipfile._ECD_ENTRIES_THIS_DISK] or count <= 0 or
            count*4096 + size*16 > metadata_cap//2):
        raise RuntimeError(f'{label} ZIP directory exceeds metadata scratch budget')
    # Canonical Torch files are not concatenated archives. Validate the actual
    # footer chain instead of relying on private _ECD_LOCATION semantics:
    # patched Python 3.12 reports ZIP64 EOCD there, older versions report EOCD32.
    position = offset
    directory_end = offset+size
    if position < 0 or size < 0 or directory_end > file_bytes:
        raise RuntimeError(f'{label} has an invalid ZIP directory extent')
    footer_position = directory_end
    source.seek(footer_position)
    if end[zipfile._ECD_SIGNATURE] == zipfile.stringEndArchive64:
        data = source.read(zipfile.sizeEndCentDir64)
        if len(data) != zipfile.sizeEndCentDir64:
            raise RuntimeError(f'{label} has a truncated ZIP64 footer')
        record = struct.unpack(zipfile.structEndArchive64, data)
        if (record[0] != zipfile.stringEndArchive64 or record[1]+12 != zipfile.sizeEndCentDir64 or
                tuple(record[6:]) != (count, count, size, offset)):
            raise RuntimeError(f'{label} requires a canonical fixed-size ZIP64 footer')
        locator = source.read(zipfile.sizeEndCentDir64Locator)
        if len(locator) != zipfile.sizeEndCentDir64Locator or struct.unpack(
                zipfile.structEndArchive64Locator, locator) != (
                    zipfile.stringEndArchive64Locator, 0, directory_end, 1):
            raise RuntimeError(f'{label} has an invalid ZIP64 locator')
        footer_position += zipfile.sizeEndCentDir64 + zipfile.sizeEndCentDir64Locator
    source.seek(footer_position)
    footer = source.read(zipfile.sizeEndCentDir)
    if (len(footer) != zipfile.sizeEndCentDir or not footer.startswith(zipfile.stringEndArchive) or
            footer_position+zipfile.sizeEndCentDir+len(end[zipfile._ECD_COMMENT]) != file_bytes):
        raise RuntimeError(f'{label} requires a canonical non-concatenated ZIP directory')
    stop, observed = position+size, 0
    while position < stop:
        source.seek(position)
        header = source.read(zipfile.sizeCentralDir)
        if len(header) != zipfile.sizeCentralDir:
            raise RuntimeError(f'{label} has a truncated ZIP directory')
        values = struct.unpack(zipfile.structCentralDir, header)
        if values[zipfile._CD_SIGNATURE] != zipfile.stringCentralDir:
            raise RuntimeError(f'{label} has an invalid ZIP directory header')
        observed += 1
        if observed > count or observed*4096 + size*16 > metadata_cap//2:
            raise RuntimeError(f'{label} ZIP directory exceeds metadata scratch budget')
        position += zipfile.sizeCentralDir + sum(values[index] for index in (
            zipfile._CD_FILENAME_LENGTH, zipfile._CD_EXTRA_FIELD_LENGTH, zipfile._CD_COMMENT_LENGTH))
    if position != stop or observed != count:
        raise RuntimeError(f'{label} ZIP directory count or extent disagrees')
    source.seek(0)


def _preflight_pickle_opcodes(raw, *, metadata_cap, label):
    """Bound the C unpickler's memo/frame allocations and disable extensions."""
    max_memo = metadata_cap//512
    memo_operations = 0
    last = None
    try:
        for opcode, argument, position in pickletools.genops(raw):
            last = (opcode.name, position)
            if opcode.name in ('EXT1', 'EXT2', 'EXT4', 'INST', 'OBJ', 'NEWOBJ',
                               'NEWOBJ_EX', 'BUILD'):
                raise RuntimeError(f'{label} has an opaque or extension pickle opcode')
            if opcode.name in ('PUT', 'BINPUT', 'LONG_BINPUT', 'GET', 'BINGET', 'LONG_BINGET'):
                if type(argument) is not int or not 0 <= argument < max_memo:
                    raise RuntimeError(f'{label} pickle memo exceeds scratch budget')
            if opcode.name in ('PUT', 'BINPUT', 'LONG_BINPUT', 'MEMOIZE'):
                memo_operations += 1
                if memo_operations > max_memo:
                    raise RuntimeError(f'{label} pickle memo exceeds scratch budget')
            if opcode.name == 'FRAME' and not 0 <= argument <= len(raw):
                raise RuntimeError(f'{label} pickle frame exceeds bounded metadata')
        if last != ('STOP', len(raw)-1):
            raise RuntimeError(f'{label} has trailing or incomplete pickle metadata')
    except (ValueError, OverflowError) as exc:
        raise RuntimeError(f'{label} has unaccountable pickle opcodes') from exc


def _preflight_torch_pickle_storage(archive, *, records, pickle_name, label, metadata_cap):
    """Check every declared storage size without constructing a Torch object.

    Older Torch stages CPU storage even for map_location='meta'. Only a closed
    set of tensor reconstruction markers is accepted here; those callbacks are
    inert. ZIP sizes and pickle sizes must agree before either Torch pass.
    """
    # Archive metadata was admitted before this bounded read. Inspect opcodes
    # before constructing the C Unpickler: find_class alone does not guard its
    # sparse memo allocation or cached extension registry.
    raw = archive.read(pickle_name)
    _preflight_pickle_opcodes(raw, metadata_cap=metadata_cap, label=label)
    element_bytes = {'ByteStorage': 1, 'CharStorage': 1, 'BoolStorage': 1,
        'ShortStorage': 2, 'HalfStorage': 2, 'BFloat16Storage': 2,
        'IntStorage': 4, 'FloatStorage': 4, 'LongStorage': 8,
        'DoubleStorage': 8, 'ComplexFloatStorage': 8, 'ComplexDoubleStorage': 16,
        'UntypedStorage': 1}
    storage_types = {}
    def tensor_marker(*args):
        if (len(args) < 4 or type(args[0]) is not tuple or len(args[0]) != 3 or
                args[0][0] != 'storage' or type(args[0][1]) is not str or args[0][1] not in records or
                type(args[0][2]) is not int or args[0][2] not in (1, 2, 4, 8, 16) or
                type(args[1]) is not int or args[1] < 0 or
                type(args[2]) is not tuple or type(args[3]) is not tuple or
                len(args[2]) != len(args[3]) or len(args[2]) > 64 or
                any(type(value) is not int or value < 0 for value in (*args[2], *args[3]))):
            raise RuntimeError(f'{label} has unaccountable pickle tensor geometry')
        size = args[0][2]
        if len(args) > 6 and type(args[6]) is tuple and args[6][0] == 'dtype':
            sizes = dict(float16=2, float32=4, float64=8, bfloat16=2, int8=1,
                         uint8=1, int16=2, int32=4, int64=8, bool=1, complex64=8, complex128=16)
            size = sizes[args[6][1]]
        extent = (0 if any(value == 0 for value in args[2]) else
                  args[1]+1+sum((dim-1)*stride for dim, stride in zip(args[2], args[3])))
        if extent*size > records[args[0][1]]:
            raise RuntimeError(f'{label} pickle tensor geometry exceeds its declared backing storage')
        return None
    class StoragePreflight(pickle.Unpickler):
        def find_class(self, module, name):
            if module in ('torch', 'torch.storage') and name in element_bytes:
                return ('storage_type', element_bytes[name])
            if module == 'torch._utils' and name in (
                    '_rebuild_tensor', '_rebuild_tensor_v2', '_rebuild_tensor_v3'):
                return tensor_marker
            if module == 'collections' and name == 'OrderedDict':
                return OrderedDict
            # v3 carries dtype separately; no callable Torch global escapes.
            if module == 'torch' and name in ('float16', 'float32', 'float64',
                    'bfloat16', 'int8', 'uint8', 'int16', 'int32', 'int64', 'bool',
                    'complex64', 'complex128'):
                return ('dtype', name)
            raise RuntimeError(f'{label} has an opaque pickle global: {module}.{name}')

        def persistent_load(self, value):
            if (type(value) is not tuple or len(value) != 5 or value[0] != 'storage' or
                    type(value[1]) is not tuple or len(value[1]) != 2 or value[1][0] != 'storage_type' or
                    type(value[1][1]) is not int or value[1][1] not in (1, 2, 4, 8, 16) or
                    type(value[2]) is not str or value[2] not in records or value[3] != 'cpu' or
                    type(value[4]) is not int or value[4] < 0 or
                    value[4]*value[1][1] != records[value[2]]):
                raise RuntimeError(f'{label} declared pickle storage disagrees with bounded ZIP storage')
            if storage_types.setdefault(value[2], value[1][1]) != value[1][1]:
                raise RuntimeError(f'{label} pickle storage aliases disagree on element size')
            return ('storage', value[2], value[1][1])
    try:
        StoragePreflight(io.BytesIO(raw)).load()
    except (pickle.UnpicklingError, TypeError, ValueError, AttributeError, EOFError) as exc:
        raise RuntimeError(f'{label} has unaccountable pickle storage metadata') from exc


def torch_archive_storage_bytes(source, *, label='PWC window', metadata_cap=None,
                                max_storage_bytes=None):
    """Inspect ordinary uncompressed Torch storage records without loading them."""
    try:
        if metadata_cap is not None:
            _preflight_torch_zip_directory(source, metadata_cap=metadata_cap, label=label)
        with zipfile.ZipFile(source) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            roots = {name.split('/')[0] for name in names}
            if (len(roots) != 1 or len(set(names)) != len(names)
                    or not any(name.endswith('/data.pkl') for name in names)
                    or any(entry.compress_type != zipfile.ZIP_STORED or entry.flag_bits & 1
                           or entry.file_size != entry.compress_size for entry in entries)):
                raise RuntimeError(f'{label} requires an uncompressed Torch archive')
            storage = [entry for entry in entries
                       if re.fullmatch(r'[^/]+/data/[0-9]+', entry.filename)]
            if metadata_cap is not None:
                # Price Python/ZIP/pickle metadata separately from tensor storage.
                # The bounded source adapter also refuses oversized directory reads
                # before ZipFile can construct an unbounded member list.
                metadata_bytes = sum(entry.file_size for entry in entries if entry not in storage)
                bound = sum(4096 + 8*len(name.encode()) for name in names) + 64*metadata_bytes
                if bound > metadata_cap // 2:
                    raise RuntimeError(f'{label} archive metadata exceeds scratch budget')
            total = sum(entry.file_size for entry in storage)
            if max_storage_bytes is not None:
                if total > max_storage_bytes:
                    raise RuntimeError(f'{label} archive backing storage exceeds its budget')
                _preflight_torch_pickle_storage(archive,
                    records={entry.filename.rsplit('/', 1)[1]: entry.file_size for entry in storage},
                    pickle_name=next(name for name in names if name.endswith('/data.pkl')),
                    label=label, metadata_cap=metadata_cap)
            return total
    except (OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f'{label} has an unaccountable Torch archive') from exc


def _advise_activation_descriptor(descriptor, path, expected_stat, *, offset=0,
                                  length=0, durable=False):
    expected = cache_file_stat_signature(expected_stat)
    actual = os.fstat(descriptor)
    if not stat.S_ISREG(actual.st_mode) or cache_file_stat_signature(actual) != expected:
        raise RuntimeError('capture entry changed before page advice')
    if durable:
        os.fsync(descriptor)
    if (cache_file_stat_signature(os.fstat(descriptor)) != expected or
            cache_file_stat_signature(os.stat(path, follow_symlinks=False)) != expected):
        raise RuntimeError('capture entry changed while completing durability')
    os.posix_fadvise(descriptor, offset, length, os.POSIX_FADV_DONTNEED)


def release_activation_cache_file_pages(path, *, expected_stat, durable=True):
    """Advise a verified unchanged file extent at a reader/writer boundary.

    Readers check completed-entry hashes first. A serializer may also pause
    after a synchronous tensor record and advise its stable visible prefix;
    the same inode, size and timestamp checks and durability fence apply.
    The artifact and tensor owners remain intact, and final publication still
    requires the complete seal. Advice is not proof of physical release; the
    caller's guard remains final.

    ``durable=False`` skips the fsync fence for a file that is never the
    only copy of committed work (a produced-output spool entry, PQ #1225).
    The advice then starts the file's writeback without waiting for it, and
    pages still dirty or under writeback stay cached until they are clean.
    """
    flags = os.O_RDONLY | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        _advise_activation_descriptor(descriptor, path, expected_stat,
                                      durable=durable)
    finally:
        os.close(descriptor)


VERIFIED_ACTIVATION_LOAD_SCHEMA = 'prismaquant.verified_activation_load.v1'


def normalize_verified_activation_load(config):
    if config is None:
        return None
    if (not isinstance(config, dict) or set(config) !=
            {'schema', 'max_buffer_bytes', 'max_scratch_bytes'} or
            config.get('schema') != VERIFIED_ACTIVATION_LOAD_SCHEMA):
        raise ValueError('verified activation load requires a complete closed v1 policy')
    for key in ('max_buffer_bytes', 'max_scratch_bytes'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'verified activation load requires positive {key}')
    if config['max_scratch_bytes'] < 1024**2:
        raise ValueError('verified activation load requires at least 1 MiB metadata scratch')
    return dict(config)


class _VerifiedBufferReader(io.RawIOBase):
    """Read-only access to one private buffer, with no full-copy fallback."""
    def __init__(self, buffer, *, max_copy_bytes):
        self._view = memoryview(buffer).toreadonly()
        self._position = 0
        self._max_copy_bytes = max_copy_bytes

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        self._checkClosed()
        return self._position

    def seek(self, offset, whence=0):
        self._checkClosed()
        if whence not in (0, 1, 2):
            raise ValueError('invalid verified-buffer seek origin')
        position = offset + (0 if whence == 0 else self._position if whence == 1 else len(self._view))
        if position < 0:
            raise ValueError('negative verified-buffer seek')
        self._position = position
        return position

    def read(self, size=-1):
        self._checkClosed()
        available = max(0, len(self._view) - self._position)
        size = available if size is None or size < 0 else min(size, available)
        if size > self._max_copy_bytes:
            raise RuntimeError('verified-buffer copying read exceeds scratch budget; readinto required')
        result = bytes(self._view[self._position:self._position+size])
        self._position += size
        return result

    def readinto(self, target):
        self._checkClosed()
        output = memoryview(target).cast('B')
        try:
            size = min(len(output), max(0, len(self._view)-self._position))
            output[:size] = self._view[self._position:self._position+size]
            self._position += size
            return size
        finally:
            output.release()

    def close(self):
        if not self.closed:
            self._view.release()
            self._view = None
        super().close()


def bounded_cpu_float32_isfinite(tensor, *, max_scratch_bytes):
    """Validate a resident canonical tensor with two scalar reduction outputs.

    The qualified CPU aminmax kernel propagates NaNs, preserves infinities and
    uses vector accumulators plus one scalar pair per native thread. Contiguity
    is required before the kernel's contiguous() call can create a hidden copy.
    Torch tensor allocation is eight bytes; conservatively charge reduction
    pairs and Python/Tensor metadata within M. Native thread-pool bookkeeping
    remains runtime overhead, independently covered by the physical guard.
    """
    if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu' or
            tensor.dtype != torch.float32 or tensor.layout != torch.strided or
            not tensor.is_contiguous() or tensor.requires_grad):
        raise RuntimeError('bounded finite reduction requires contiguous CPU float32 storage')
    # The pinned CPU parallel_reduce uses SmallVector<pair<float,float>,64>.
    # This overprices its pair payload and reserves independent scalar metadata.
    reduction_bytes = 1024 + 16*max(64, torch.get_num_threads())
    if type(max_scratch_bytes) is not int or reduction_bytes > max_scratch_bytes//2:
        raise RuntimeError('bounded finite reduction exceeds metadata scratch budget')
    if tensor.numel() == 0:
        return True  # Match isfinite(empty).all() without invoking empty aminmax.
    minimum, maximum = torch.aminmax(tensor)
    return math.isfinite(minimum.item()) and math.isfinite(maximum.item())


def _verified_payload_storage(payload, *, max_storage_bytes, device, max_nodes):
    """Charge complete backing storage and refuse opaque or oversized metadata."""
    pending, visited, storages = [payload], set(), {}
    while pending:
        value = pending.pop()
        identity = id(value)
        if identity in visited:
            continue
        visited.add(identity)
        if len(visited) > max_nodes:
            raise RuntimeError('verified activation payload exceeds metadata scratch budget')
        if isinstance(value, torch.Tensor):
            if (value.device.type != device or value.layout != torch.strided or value.is_quantized
                    or value.requires_grad or value.numel()*value.element_size() > max_storage_bytes):
                raise RuntimeError('verified activation payload has unaccountable tensor geometry/type')
            storage = value.untyped_storage()
            storages[storage._cdata] = storage.nbytes()
            if (storage.nbytes() > max_storage_bytes or
                    (device != 'meta' and sum(storages.values()) > max_storage_bytes)):
                raise RuntimeError('verified activation backing storage exceeds its budget')
        elif type(value) in (dict, OrderedDict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif type(value) in (tuple, list):
            pending.extend(value)
        elif value is not None and type(value) not in (str, int, float, bool, bytes):
            raise RuntimeError('verified activation payload has an opaque metadata owner')
    # Torch's meta restore does not preserve storage aliases (data_ptr is 0).
    # ZIP records already bound aggregate bytes; meta checks each geometry, and
    # the CPU pass checks the exact unique backing-storage aggregate.
    return max(storages.values(), default=0) if device == 'meta' else sum(storages.values())


def _await_entry_landing(resolver, entries, *, deadline=None) -> bool:
    """Wait for whole-file entries the map does not hold yet (PQ #1204).

    ``entries`` is ``[(declared path, file bytes), ...]``: entries a strict
    reader looked up and did not find. This is the landing-record wait the
    shard reader and the checkpoint loader use,
    ``residency_shard_reader.await_staged_spans``, over one whole-file span
    per entry, with the same proof check (``stage_cover_is_published``, and
    the batched form for a window). One call covers every entry, so a
    window waits once, however many of its entries are missing. Returns
    ``True`` when every entry landed and the caller may look it up again,
    ``False`` when the caller refuses as before.

    It waits only where waiting can end in a landing. The resolver must
    carry the sealed readset, and that readset must be bound: an unbound
    readset cannot tell a declared entry from an undeclared one, so it
    does not wait, as ``layer_streaming._await_layer_readset`` does not. A
    span the bound readset does not declare, a covering entry that failed
    a hard check, and every refusal of PrismaBuild's landing record return
    at once. ``deadline`` (``time.monotonic()``) bounds the wait only
    where no landing record covers the entries; ``None`` starts the
    configured bound here (``PRISMAQUANT_STAGED_RANGE_WAIT_S``).

    Called only after a miss, so an entry the map already holds costs
    nothing here.
    """
    if not entries:
        return True
    readset = getattr(resolver, "declared_readset", None)
    if not callable(readset) or not callable(getattr(resolver, "staged_range_outcome", None)):
        return False
    if readset().get("state") != "bound":
        return False
    import time
    from .residency_map import RANGE_HIT
    from .residency_shard_reader import await_staged_spans, staged_range_wait_s
    from .staged_lease import stage_cover_is_published, stage_covers_are_published
    if deadline is None:
        deadline = time.monotonic() + staged_range_wait_s()
    verdict = await_staged_spans(
        resolver, [(str(path), 0, size, size) for path, size in entries],
        deadline=deadline, published=stage_cover_is_published,
        published_batch=stage_covers_are_published)
    return verdict == RANGE_HIT


def _acquire_bulk_window(path, expected_sha256, *, resolver=None, declared_size=None,
                         deadline=None):
    """Strict-policy pinned window for a whole-file bulk input.

    Returns the entered ``(window, key, staged)``: the RAM leg refuses
    fast (RAM-mover covers unresolved), the SSD copy acquires honestly
    with the map's leads. Raises ``LeaseRefused`` before any pool payload
    byte when unmapped, unfenced, or unpermitted. The caller opens keys,
    reads through held descriptors, verifies content against
    ``expected_sha256``, re-checks the declared file's binding, and exits
    the window (close-then-release) on every path.

    An entry the map does not hold yet is waited on as
    :func:`_await_entry_landing` describes, then looked up again (PQ
    #1204). ``declared_size`` is the declared file's size when the caller
    already holds it; otherwise a miss stats it.

    ``resolver`` defaults to the process's input-map resolver, which is
    what every sealed input, every read-only attached generation and every
    foreign generation uses -- unchanged. A caller reading its OWN
    produced batch passes that batch's explicitly namespaced resolver
    instead; the input map is never swapped out from under anyone.
    """
    from .residency_map import residency_resolver
    from .staged_lease import LeaseRefused, acquire_entry_window
    if resolver is None:
        resolver = residency_resolver()
    if resolver is None:
        raise LeaseRefused("readset-not-staged", kind="availability")
    staged = resolver.staged_read(path, expected_sha256=expected_sha256)
    if staged is None:
        if declared_size is None:
            try:
                declared_size = os.lstat(path).st_size
            except OSError:
                declared_size = None
        if declared_size is not None and _await_entry_landing(
                resolver, [(path, declared_size)], deadline=deadline):
            staged = resolver.staged_read(path, expected_sha256=expected_sha256)
    if staged is None:
        raise LeaseRefused("staged-not-serving", kind="availability")
    # acquire_entry_window records its own acquire refusal; the LeaseRefused
    # propagates with kind intact (never converted into a pool read).
    window, key = acquire_entry_window(resolver, path, staged)
    return window, key, staged, resolver


def _enter_and_open_window(resolver, window, key, path):
    """Enter exactly once, then open one pinned key with the SDK serving
    record; exit the window on any failure so no pin leaks. The caller
    owns the entered window from here (reads, then close-then-release)."""
    from .staged_lease import LeaseRefused
    try:
        window.__enter__()
    except LeaseRefused as refusal:
        resolver.record_fallback(path, str(refusal))
        raise
    try:
        fd, serving = window.open(key)
    except LeaseRefused as refusal:
        resolver.record_fallback(path, str(refusal))
        try:
            window.__exit__(None, None, None)
        except (LeaseRefused, RuntimeError):
            pass
        raise
    tier = window.serving_tier or "stage"
    resolver.record_serving_tier(
        path, tier, pin_id=str(serving.get("pin_id") or ""),
        range_ref=str(serving.get("range_ref") or ""))
    return fd, serving, tier


def _record_served_bytes(resolver, path, tier, nbytes):
    """Count one verified read's bytes against the tier that served it.

    ``tier`` is the serving tier the window reported (``ram`` or ``stage``),
    or ``None`` for a read of the declared file, which is counted as pool
    bytes. Called only after the bytes are verified, so a refused read
    counts nothing. With no resolver there is nothing to count (PQ #1026).
    """
    if resolver is None:
        return
    if tier is None:
        resolver.record_pool_read(path, nbytes)
    elif tier == "ram":
        resolver.record_ram_read(path, nbytes)
    else:
        resolver.record_stage_read(path, nbytes)


def load_verified_activation_cache_entry(path, *, expected_sha256, policy,
                                         max_storage_bytes, validate=None,
                                         expected_stat=None, resource_check=None,
                                         release_file_pages=False):
    """Hash and deserialize one admitted byte buffer, released before return.

    The metadata pass uses meta tensors to reject malformed storage/geometry
    before a real CPU reconstruction. Torch may stage one archive storage on
    CPU during that pass; the same S cap covers it. Neither pass rereads the
    source file, and no CUDA transfer is performed by this owner.

    Under the active allowed-tier policy the payload bytes come from the
    staged source (RAM first, declared SSD only) while every declared-file
    identity fence still runs on the declared file; the post-read digest
    against the exact receipt is what admits the bytes.
    """
    policy = normalize_verified_activation_load(policy)
    if policy is None or type(max_storage_bytes) is not int or max_storage_bytes <= 0:
        raise ValueError('verified activation load requires explicit buffer and storage budgets')
    if not isinstance(expected_sha256, str) or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None:
        raise ValueError('verified activation load requires an exact SHA256 receipt')
    path = Path(path)
    before = path.lstat()
    signature = cache_file_stat_signature(before)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError('verified activation load requires a regular nonsymlink file')
    if expected_stat is not None and cache_file_stat_signature(expected_stat) != signature:
        raise RuntimeError('verified activation file changed before loading')
    if before.st_size <= 0 or before.st_size > policy['max_buffer_bytes']:
        raise RuntimeError('verified activation file exceeds serialized buffer budget')
    scratch = policy['max_scratch_bytes']
    from .staged_tier_policy import policy_is_active
    strict = policy_is_active()
    source, source_before = path, before
    source_signature = signature
    window = None
    tier = None
    if strict:
        window, key, _staged, lease_resolver = _acquire_bulk_window(
            path, expected_sha256, declared_size=before.st_size)
        descriptor, serving, tier = _enter_and_open_window(
            lease_resolver, window, key, path)
        source_signature = cache_file_stat_signature(os.fstat(descriptor))
        source, source_before = Path(window.stage_path(key) or path), os.fstat(descriptor)
    else:
        # The declared file: pool bytes when a map is bound (PQ #1026).
        from .residency_map import residency_resolver
        lease_resolver = residency_resolver()
    def check(label, reserve_bytes=0):
        if resource_check is not None:
            resource_check(label + ':' + path.name, reserve_bytes=reserve_bytes)
    def unchanged(descriptor):
        if (cache_file_stat_signature(os.fstat(descriptor)) != source_signature or
                cache_file_stat_signature(path.lstat()) != signature):
            raise RuntimeError('verified activation file changed during loading')
    if window is None:
        descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    raw = reader = value = payload = None
    try:
        unchanged(descriptor)
        # Buffered file I/O can retain all source contents in the kernel even
        # when advice is requested. Price that full F separately from private F
        # and metadata M; page rounding/bookkeeping remain in the guard margin.
        source_page_cache_bytes = before.st_size
        check('before_verified_capture_buffer',
              before.st_size + source_page_cache_bytes + max_storage_bytes + scratch)
        if release_file_pages:
            # Complete durability once, then advise only verified consumed ranges.
            os.fsync(descriptor)
            unchanged(descriptor)
        raw = bytearray(before.st_size)
        digest = hashlib.sha256()
        consumed = advised = 0
        # These are views of the already admitted F allocation, not M-sized
        # owned scratch copies. Every bounded read retains guard/stat/advice.
        block_bytes = 16*1024**2
        with os.fdopen(descriptor, 'rb', buffering=0, closefd=False) as handle:
            while consumed < len(raw):
                check('before_verified_capture_read',
                      source_page_cache_bytes + max_storage_bytes + scratch)
                view = memoryview(raw)[consumed:min(len(raw), consumed+block_bytes)]
                try:
                    size = handle.readinto(view)
                    if not size:
                        raise RuntimeError('verified activation file was truncated')
                    digest.update(view[:size])
                finally:
                    view.release()
                consumed += size
                unchanged(descriptor)
                if release_file_pages:
                    page = os.sysconf('SC_PAGE_SIZE')
                    end = consumed // page * page
                    if end > advised:
                        _advise_activation_descriptor(descriptor, source, source_before,
                                                      offset=advised, length=end-advised)
                        advised = end
            if handle.read(1):
                raise RuntimeError('verified activation file grew during loading')
        unchanged(descriptor)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError('verified activation file checksum mismatch')
        reader = _VerifiedBufferReader(raw, max_copy_bytes=min(scratch//8, 128*1024))
        archive_storage = torch_archive_storage_bytes(reader, label='verified activation load',
                                                      metadata_cap=scratch,
                                                      max_storage_bytes=max_storage_bytes)
        if archive_storage > max_storage_bytes:
            raise RuntimeError('verified activation archive backing storage exceeds its budget')
        for device in ('meta', 'cpu'):
            check('before_verified_capture_decode',
                  source_page_cache_bytes + max_storage_bytes + scratch)
            reader.seek(0)
            value = torch.load(reader, map_location=device, weights_only=True)
            observed = _verified_payload_storage(value, max_storage_bytes=max_storage_bytes,
                device=device, max_nodes=scratch//512)
            if observed > archive_storage:
                raise RuntimeError('verified activation tensor storage exceeds its archive records')
            if validate is not None:
                validate(value, check_finite=device == 'cpu')
            if device == 'cpu':
                payload = value
            value = None
        unchanged(descriptor)
        if release_file_pages:
            _advise_activation_descriptor(descriptor, source, source_before)
    except BaseException:
        value = payload = None
        raise
    finally:
        if reader is not None:
            reader.close()
        raw = reader = None
        if window is not None:
            # The held descriptor closes and the exact ref releases here,
            # after the owned buffer is fully read and verified above.
            window.__exit__(None, None, None)
        else:
            os.close(descriptor)
    _record_served_bytes(lease_resolver, path, tier, consumed)
    try:
        check('after_verified_capture_buffer_release')
    except BaseException:
        payload = None
        raise
    execution = dict(schema=VERIFIED_ACTIVATION_LOAD_SCHEMA, policy=policy,
        artifact_sha256=expected_sha256, file_bytes=before.st_size,
        storage_cap_bytes=max_storage_bytes, archive_storage_bytes=archive_storage,
        source_page_cache_reserve_bytes=source_page_cache_bytes,
        file_signature=signature, source_read_bytes=consumed, live_buffer_bytes=0)
    execution['identity_sha256'] = hashlib.sha256(json.dumps(
        {key: execution[key] for key in ('schema', 'policy', 'artifact_sha256', 'file_bytes',
                                         'storage_cap_bytes')},
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return payload, execution


EXACT_ACTIVATION_SCHEMA = "prismaquant.exact_activation_entry.v1"

# The block hashlib.file_digest read these entries in before the digest was
# folded into the load (its _bufsize default, pinned by the read-amplification
# tests). Keeping that block keeps the kernel's readahead overlapping the next
# block with the hash of the current one; one whole-file read does not.
_ENTRY_READ_BLOCK_BYTES = 2**18


@dataclass(frozen=True)
class ExactActivationReference:
    """An immutable exact tensor receipt, never an activation sample/cache."""

    path: str
    name: str
    metadata_json: str
    shape: tuple[int, ...]
    dtype: str
    tensor_bytes: int
    file_bytes: int
    sha256: str


def _exact_activation_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _activation_file_signature(path):
    import stat
    value = Path(path).lstat()
    if not stat.S_ISREG(value.st_mode):
        raise RuntimeError("exact activation entry is not a regular file")
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _is_compact_host_tensor(tensor, nbytes):
    """Whether ``tensor`` is exactly one standard-strided host storage.

    Such a tensor serializes to the bytes its compact copy would: the same
    storage bytes, offset 0, and the strides ``contiguous_format`` gives,
    including on size-1 dimensions, where ``is_contiguous`` ignores them.
    """
    if (tensor.device.type != "cpu" or tensor.storage_offset() != 0
            or tensor.is_conj() or tensor.is_neg()
            or tensor.untyped_storage().nbytes() != nbytes):
        return False
    expected = 1
    for size, stride in zip(reversed(tensor.shape), reversed(tensor.stride())):
        if stride != expected:
            return False
        expected *= size
    return True


def write_exact_activation_cache_entry(cache_dir, name, inputs, *, identity,
                                       max_tensor_bytes, max_file_bytes,
                                       release_file_pages=True, preallocate=False,
                                       durable=True):
    """Extend the ordinary atomic writer with an exact tensor/identity receipt.

    The caller reserves the one compact CPU copy before entering. No dtype,
    row selection or shape change is allowed. Compact copying also prevents a
    narrow view from serializing its entire source backing storage. A tensor
    that already is one compact host storage (``_is_compact_host_tensor``)
    is serialized as it is: the copy would be the same bytes
    (RobTand/prismaquant#1162).

    ``durable`` fsyncs the file and its directory before the entry is
    returned. Only a writer whose file is never the only copy of committed
    work may pass False: the produced-output spool (PQ #1225), whose every
    reader checks the size and the sha256 of every byte against the
    reference this returns, and whose work counts as committed only once
    PrismaBuild acknowledged the export's own fsynced copy.
    """
    if not isinstance(inputs, torch.Tensor) or inputs.layout != torch.strided or inputs.is_meta:
        raise TypeError("exact activation entry requires a materialized strided Tensor")
    nbytes = inputs.numel() * inputs.element_size()
    if not 0 < nbytes <= max_tensor_bytes:
        raise RuntimeError("exact activation entry exceeds tensor residency budget")
    metadata = {"schema": EXACT_ACTIVATION_SCHEMA, "identity": identity,
                "shape": list(inputs.shape), "dtype": str(inputs.dtype), "tensor_bytes": nbytes}
    encoded = _exact_activation_json(metadata)
    path = Path(cache_dir) / activation_cache_filename(name)
    if path.exists() or path.with_suffix(".pt.tmp").exists():
        raise RuntimeError("exact activation entry already exists")
    compact = None
    digest = None
    try:
        compact = (inputs.detach() if _is_compact_host_tensor(inputs, nbytes)
                   else inputs.detach().to(device="cpu", copy=True,
                                           memory_format=torch.contiguous_format))
        digest = SerializedEntryDigest(max_bytes=max_file_bytes if preallocate else None)
        path = write_activation_cache_entry(cache_dir, name, compact,
            source="exact_activation", durable=durable, exact=metadata,
            serialized_digest=digest,
            preallocate_bytes=max_file_bytes if preallocate else None)
        del compact
        compact = None
        published_stat = path.lstat()
        signature = _activation_file_signature(path)
        if signature[2] > max_file_bytes:
            raise RuntimeError("exact activation entry exceeds file budget")
        if signature[2] != digest.bytes:
            raise RuntimeError("exact activation entry differs from its serialized bytes")
        if _activation_file_signature(path) != signature:
            raise RuntimeError("exact activation entry changed during publication")
        if release_file_pages:
            release_activation_cache_file_pages(path, expected_stat=published_stat,
                                                durable=durable)
        return ExactActivationReference(str(path), name, encoded, tuple(inputs.shape),
            str(inputs.dtype), nbytes, signature[2], digest.hexdigest())
    except BaseException:
        for candidate in (path, path.with_suffix(".pt.tmp")):
            if preallocate:
                owned = getattr(digest, "file_identity", None)
                try:
                    current = candidate.lstat()
                except FileNotFoundError:
                    continue
                if owned != (current.st_dev, current.st_ino):
                    continue
            candidate.unlink(missing_ok=True)
        raise
    finally:
        compact = None


class ExactCotangentScratch:
    """Job-local fixed tensor slots; reads own their bytes, never mmap views.

    This disposable arithmetic workspace is not a checkpoint or an input
    cache. Its owner replays immutable checkpoint entries after interruption.
    Allocation is real disk space.

    The file holds no page cache: the cgroup charges cached file bytes to
    the job, and a plane of 2,048 16 MiB slots is far larger than its
    resident budget. When every slot is a whole number of the file's
    direct-I/O blocks (``statx`` ``STATX_DIOALIGN``), which a Stage B plane
    is, slots are written and read with ``O_DIRECT`` (PQ #1152): the device
    reads from and writes into the tensor's own memory, or one reused
    aligned buffer when its address is off the grid, and a write that
    returns has reached the device. Otherwise every write is synced and
    every completed I/O drops the file's pages, one slot at a time. Nothing
    is durable either way: a slot is published once its write returns and
    the file dies with the process.
    """

    @staticmethod
    def _require_local_disk(root):
        root = Path(root)
        if not root.is_absolute() or not root.is_dir() or root.is_symlink():
            raise ValueError("cotangent scratch requires an existing absolute local directory")
        resolved = root.resolve(strict=True)
        matches = []
        for line in Path('/proc/self/mountinfo').read_text().splitlines():
            left, right = line.split(' - ', 1)
            fields = left.split()
            mount = Path(re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), fields[4]))
            if resolved == mount or mount in resolved.parents:
                matches.append((len(mount.parts), right.split()[0]))
        if not matches or max(matches)[1] not in {'ext4', 'xfs', 'btrfs', 'zfs'}:
            raise ValueError("cotangent scratch requires local disk (NFS/tmpfs/overlay refused)")
        return resolved

    def __init__(self, records, *, directory, max_bytes, max_tensor_bytes=None):
        import tempfile
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("cotangent scratch requires a positive byte ceiling")
        if max_tensor_bytes is None:
            max_tensor_bytes = max_bytes
        if type(max_tensor_bytes) is not int or max_tensor_bytes <= 0:
            raise ValueError("cotangent scratch requires a positive resident tensor ceiling")
        self._file = None
        self._slots = {}
        self._written = set()
        self.tensor_bytes = 0
        self.max_slot_bytes = 0
        for entry in records:
            name = entry['name']
            if not re.fullmatch(r'cotangent-[0-9]+-[0-9]+', name):
                raise ValueError("cotangent scratch entry has invalid coordinates")
            key = tuple(int(part) for part in name.split('-')[1:])
            if key in self._slots:
                raise ValueError("cotangent scratch repeats coordinates")
            shape = tuple(entry['shape'])
            if not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
                raise ValueError("cotangent scratch entry has invalid shape")
            dtype = getattr(torch, str(entry['dtype']).removeprefix('torch.'), None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError("cotangent scratch entry has invalid dtype")
            size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            if type(entry['tensor_bytes']) is not int or size != entry['tensor_bytes']:
                raise ValueError("cotangent scratch entry byte size differs")
            if size > max_tensor_bytes:
                raise RuntimeError("cotangent scratch slot exceeds resident tensor ceiling")
            self._slots[key] = (self.tensor_bytes, size, shape, dtype)
            self.tensor_bytes += size
            self.max_slot_bytes = max(self.max_slot_bytes, size)
        if not self._slots or self.tensor_bytes > max_bytes:
            raise RuntimeError("cotangent scratch exceeds its sealed disk byte ceiling")
        root = self._require_local_disk(directory)
        self._file = tempfile.TemporaryFile(prefix='pq-cotangent-', dir=root)
        self._direct = None
        self._bounce = None
        try:
            os.posix_fallocate(self._file.fileno(), 0, self.tensor_bytes)
            os.fdatasync(self._file.fileno())
            self._drop_pages()
            self._direct = self._direct_grid()
        except BaseException:
            self.close()
            raise

    def _direct_grid(self):
        """Switch the file to ``O_DIRECT`` when every slot is on its grid.

        Returns ``(memory, offset)`` alignment, or ``None`` to keep the
        buffered path: a file system without direct I/O, or a slot size
        that is not a whole number of its offset blocks.
        """
        import fcntl
        fd = self._file.fileno()
        try:
            memory, offset = _direct_io_alignment(fd)
        except (OSError, RuntimeError):
            return None
        if any(size % offset for _start, size, _shape, _dtype in self._slots.values()):
            return None
        fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_DIRECT)
        return memory, offset

    def _drop_pages(self):
        os.posix_fadvise(self._file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)

    def _device_buffer(self, tensor, size):
        """A view the device may fill or drain: the tensor's own bytes when
        its address is on the memory grid, else the reused bounce buffer."""
        import mmap
        if tensor.data_ptr() % self._direct[0] == 0:
            return memoryview(tensor.view(torch.uint8).reshape(-1).numpy()), False
        if self._bounce is None:
            # Page-aligned, which is on any memory grid a device reports.
            self._bounce = mmap.mmap(-1, max(self.max_slot_bytes, mmap.PAGESIZE))
        return memoryview(self._bounce)[:size], True

    def _direct_io(self, call, view, offset, size, what):
        done = 0
        while done < size:
            moved = call(self._file.fileno(), [view[done:]], offset + done)
            if moved <= 0 or moved % self._direct[1]:
                # A resumed direct call would start off the grid.
                raise RuntimeError(f"cotangent scratch short direct {what}")
            done += moved

    def __len__(self):
        return len(self._slots)

    def __iter__(self):
        return iter(self._slots)

    def __getitem__(self, key):
        if self._file is None or key not in self._written:
            raise RuntimeError("cotangent scratch slot is not ready")
        offset, size, shape, dtype = self._slots[key]
        tensor = torch.empty(shape, dtype=dtype, device='cpu')
        if self._direct is not None:
            view, bounced = self._device_buffer(tensor, size)
            try:
                self._direct_io(os.preadv, view, offset, size, "read")
                if bounced:
                    out = memoryview(tensor.view(torch.uint8).reshape(-1).numpy())
                    try:
                        out[:] = view
                    finally:
                        out.release()
            finally:
                view.release()
            return tensor
        view = memoryview(tensor.view(torch.uint8).reshape(-1).numpy())
        try:
            done = 0
            while done < size:
                got = os.preadv(self._file.fileno(), [view[done:]], offset + done)
                if got <= 0:
                    raise RuntimeError("cotangent scratch slot is truncated")
                done += got
            self._drop_pages()
        finally:
            view.release()
        return tensor

    def __setitem__(self, key, tensor):
        if self._file is None:
            raise RuntimeError("cotangent scratch is closed")
        offset, size, shape, dtype = self._slots[key]
        if (tensor.device.type != 'cpu' or tensor.dtype != dtype
                or tuple(tensor.shape) != shape):
            raise ValueError("cotangent scratch rollover changed shape/dtype")
        self._written.discard(key)
        # At most one slot-sized compaction, as in the exact-entry writer.
        compact = tensor.detach().contiguous()
        if self._direct is not None:
            view, bounced = self._device_buffer(compact, size)
            try:
                if bounced:
                    source = memoryview(compact.view(torch.uint8).reshape(-1).numpy())
                    try:
                        view[:] = source
                    finally:
                        source.release()
                self._direct_io(os.pwritev, view, offset, size, "write")
                self._written.add(key)
            finally:
                view.release()
            return
        view = memoryview(compact.view(torch.uint8).reshape(-1).numpy())
        try:
            done = 0
            while done < size:
                put = os.pwritev(self._file.fileno(), [view[done:]], offset + done)
                if put <= 0:
                    raise RuntimeError("cotangent scratch short write")
                done += put
            os.fdatasync(self._file.fileno())
            self._drop_pages()
            self._written.add(key)
        finally:
            view.release()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._bounce is not None:
            self._bounce.close()
            self._bounce = None
        self._written.clear()


class SpillGridRefused(RuntimeError):
    """The live direct-I/O grid is coarser than the grid a ceiling was sealed on."""


class StageBSpillScratch:
    """One job-local spill file for Stage B's one-pass replay (PQ #994).

    The replay writes each probe's bf16 Linear inputs and output gradients
    here once, then accumulates every retained window from it. Like
    ``ExactCotangentScratch`` it is a disposable arithmetic workspace, not a
    checkpoint: a resumed quantum recaptures it.

    The file is created unlinked (``O_TMPFILE`` where the filesystem has
    it), so it has no name that could outlive the process: closing the
    descriptor, or the process dying for any reason, returns the space.
    There is therefore never an orphan to sweep. The whole geometry bound is
    allocated up front with ``posix_fallocate``, so a disk too small for
    the layer refuses before any GPU work instead of partway through it.

    All I/O is direct (``O_DIRECT``, PQ #1060): the device reads from and
    writes into the caller's pinned buffers, with no page cache in between.
    The buffered path cost a kernel copy out of CUDA-pinned memory (3.7 GB/s
    on a GB10, a fifth of the pageable rate), a journal commit and a cache
    flush per ``fdatasync``, and a writeback burst of a whole arena at once.
    Direct I/O needs every file offset and length to be a multiple of
    ``block`` and every buffer address to be aligned to it, which the
    kernel reports per file (``statx`` ``STATX_DIOALIGN``); the caller lays
    its buffers and the file out on that grid. Each system call carries at
    most ``call_bytes``, which bounds the requests one call puts in flight.
    """

    #: Local block filesystems only. ZFS is refused as well as NFS, tmpfs
    #: and overlay: the spill must never land on the ZFS pool, whose ARC
    #: read tier synchronous scratch writes would evict.
    FILESYSTEMS = frozenset({'ext4', 'xfs', 'btrfs'})
    SHARED_ROOT = Path('/mnt/shared')

    @classmethod
    def require_local_root(cls, root):
        """Resolve ``root`` to a local block-filesystem directory, or refuse."""
        root = Path(root)
        if not root.is_absolute():
            raise ValueError("Stage B spill root must be an absolute path")
        for candidate in (root, root.resolve()):
            if candidate == cls.SHARED_ROOT or cls.SHARED_ROOT in candidate.parents:
                raise ValueError("Stage B spill root must not be under /mnt/shared")
        if not root.is_dir() or root.is_symlink():
            raise ValueError("Stage B spill root must be an existing local directory")
        resolved = root.resolve(strict=True)
        best = None
        for line in Path('/proc/self/mountinfo').read_text().splitlines():
            left, right = line.split(' - ', 1)
            mount = Path(re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)),
                                left.split()[4]))
            if resolved == mount or mount in resolved.parents:
                # Later mountinfo rows stack over earlier ones at one target.
                if best is None or len(mount.parts) >= len(best[0].parts):
                    best = (mount, right.split()[0])
        if best is None or best[1] not in cls.FILESYSTEMS:
            raise ValueError(
                "Stage B spill root must be on a local ext4/xfs/btrfs disk "
                f"(found {best[1] if best else 'no mount'}; NFS, ZFS, tmpfs "
                "and overlay are refused)")
        return resolved

    @staticmethod
    def reservation_bytes(nbytes, *, parts, part_padding, block):
        """The file bytes ``__init__`` reserves on a ``block`` grid.

        ``nbytes`` of payload, then ``parts * (part_padding + block)`` of slot
        padding, rounded up to the grid. This is the one sizing rule for the
        spill: the scratch reserves it, and the record builder seals it as the
        row's ceiling and PrismaBuild demand
        (``joint_replay_spill.spill_reservation_bytes``).
        """
        for label, value in (("payload", nbytes), ("parts", parts),
                             ("part padding", part_padding)):
            if type(value) is not int or value < 0:
                raise ValueError(f"Stage B spill {label} must be a nonnegative integer")
        if type(block) is not int or block <= 0 or block & (block - 1):
            raise ValueError("Stage B spill grid must be a power of two")
        capacity = nbytes + parts * (part_padding + block)
        return capacity + (-capacity % block)

    def __init__(self, *, directory, max_bytes, nbytes, parts=0, part_padding=0,
                 alignment=1, max_block=None):
        """Reserve ``nbytes`` of payload plus the padding ``parts`` slots may need.

        ``block`` is the grid every offset, length and buffer address must
        sit on: the kernel's direct-I/O alignment, raised to the filesystem
        block and to ``alignment`` (a power of two) when the caller needs a
        coarser one. ext4 serializes a direct write whose ends are not on a
        filesystem block: it takes the inode lock exclusively, waits for all
        direct I/O in flight and zeroes the partial blocks. Each of at most
        ``parts`` slots starts on a ``block`` boundary, is placed fewer than
        ``part_padding`` bytes into it, and ends on the next boundary, so the
        file reserves ``parts * (part_padding + block)`` bytes beyond the
        payload (:meth:`reservation_bytes`). The ceiling covers the whole
        reservation.

        ``max_block`` is the grid a sealed ceiling was sized on. A coarser
        live grid refuses with :class:`SpillGridRefused` before the file is
        allocated: its reservation would exceed the sealed one.
        """
        import tempfile
        self._file = None
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("Stage B spill requires a positive byte ceiling")
        if type(nbytes) is not int or nbytes <= 0:
            raise ValueError("Stage B spill geometry bound must be a positive byte count")
        if type(parts) is not int or parts < 0 or type(part_padding) is not int \
                or part_padding < 0:
            raise ValueError("Stage B spill slot bound must be nonnegative integers")
        if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("Stage B spill alignment must be a power of two")
        if max_block is not None and (type(max_block) is not int or max_block <= 0
                                      or max_block & (max_block - 1)):
            raise ValueError("Stage B spill sealed grid must be a power of two")
        if nbytes > max_bytes:
            # The payload alone is over: refuse before a file exists.
            raise RuntimeError(
                f"Stage B spill needs at least {nbytes} bytes for this layer but "
                f"its ceiling is {max_bytes}")
        root = self.require_local_root(directory)
        self.root = root
        self.allocated = 0
        self._file = tempfile.TemporaryFile(prefix='pq-stage-b-spill-', dir=root)
        try:
            fd = self._file.fileno()
            filesystem = os.fstatvfs(fd).f_bsize
            if filesystem <= 0 or filesystem & (filesystem - 1):
                raise RuntimeError(
                    f"Stage B spill filesystem block {filesystem} is not a power of two")
            self.block = max(_direct_io_block(fd), filesystem, alignment)
            if max_block is not None and self.block > max_block:
                raise SpillGridRefused(
                    f"Stage B spill direct-I/O grid is {self.block} bytes on {root}, "
                    f"coarser than the {max_block}-byte grid its ceiling was "
                    "sealed on")
            capacity = self.reservation_bytes(
                nbytes, parts=parts, part_padding=part_padding, block=self.block)
            if capacity > max_bytes:
                raise RuntimeError(
                    f"Stage B spill needs {capacity} bytes for this layer "
                    f"({nbytes} of payload and slot padding for {parts} parts at "
                    f"{self.block}-byte direct-I/O alignment) but its ceiling is "
                    f"{max_bytes}")
            self.capacity = capacity
            import fcntl
            fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_DIRECT)
            os.posix_fallocate(fd, 0, capacity)
        except BaseException:
            self.close()
            raise

    def allocate(self, nbytes):
        """Reserve the next ``nbytes`` of the file; refuse past the bound."""
        if self._file is None:
            raise RuntimeError("Stage B spill is closed")
        if type(nbytes) is not int or nbytes < 0 or nbytes % self.block:
            raise ValueError("Stage B spill allocation must be a nonnegative multiple "
                             f"of its {self.block}-byte direct-I/O block")
        offset = self.allocated
        if offset + nbytes > self.capacity:
            raise RuntimeError(
                "Stage B spill exceeded its geometry bound: a target saw more "
                "rows than one invocation per sample over the declared tokens")
        self.allocated = offset + nbytes
        return offset

    def _aligned(self, offset, views, what):
        views = [view for view in views if len(view)]
        total = sum(len(view) for view in views)
        if offset < 0 or offset + total > self.allocated:
            raise RuntimeError(f"Stage B spill {what} is outside its allocation")
        if offset % self.block or any(len(view) % self.block for view in views):
            raise RuntimeError(
                f"Stage B spill {what} is not on its {self.block}-byte direct-I/O grid")
        return views, total

    def write(self, offset, views, *, call_bytes):
        """Write ``views`` back to back at ``offset``, ``call_bytes`` per call.

        Returns the number of system calls made.
        """
        fd = self._file.fileno()
        views, _total = self._aligned(offset, views, "write")
        calls = 0
        for batch in _direct_batches(views, call_bytes, self.block):
            done, size = 0, sum(len(view) for view in batch)
            while done < size:
                try:
                    put = os.pwritev(fd, _iov_tail(batch, done), offset + done)
                except OSError as exc:
                    raise _direct_io_error(exc, "write", self.block) from exc
                if put <= 0 or put % self.block:
                    # A resumed direct write would start off the grid.
                    raise RuntimeError("Stage B spill short direct write")
                done += put
            offset += size
            calls += 1
        return calls

    def read_into(self, offset, views):
        """Fill ``views`` from the file at ``offset`` in one call; return the bytes."""
        fd = self._file.fileno()
        views, total = self._aligned(offset, views, "read")
        done = 0
        while done < total:
            try:
                got = os.preadv(fd, _iov_tail(views, done), offset + done)
            except OSError as exc:
                raise _direct_io_error(exc, "read", self.block) from exc
            if got <= 0 or got % self.block:
                raise RuntimeError("Stage B spill short direct read")
            done += got
        return total

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def _direct_io_block(fd):
    """The file's direct-I/O alignment, from ``statx(STATX_DIOALIGN)``.

    The larger of the offset and the memory alignment: one grid serves
    file offsets, lengths and buffer addresses. A file that reports no
    direct-I/O support, or a kernel without ``STATX_DIOALIGN``, is refused.
    """
    block = max(_direct_io_alignment(fd))
    if block & (block - 1):
        raise RuntimeError(f"Stage B spill direct-I/O alignment {block} is not a power of two")
    return block


def _direct_io_alignment(fd):
    """``(memory, offset)``: the file's direct-I/O alignments from ``statx``.

    ``memory`` is what a buffer address must be a multiple of; ``offset`` is
    what file offsets and lengths must be multiples of. Refuses as
    :func:`_direct_io_block` does.
    """
    import ctypes
    statx_dioalign, at_empty_path = 0x2000, 0x1000
    libc = ctypes.CDLL(None, use_errno=True)
    statx = getattr(libc, 'statx', None)
    if statx is None:
        raise RuntimeError("Stage B spill needs statx(STATX_DIOALIGN) for direct I/O")
    statx.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint,
                      ctypes.c_void_p]
    buf = ctypes.create_string_buffer(256)
    if statx(fd, b'', at_empty_path, statx_dioalign, buf) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"statx on the Stage B spill: {os.strerror(err)}")
    raw = buf.raw
    mask = int.from_bytes(raw[0:4], 'little')
    memory = int.from_bytes(raw[152:156], 'little')
    offset = int.from_bytes(raw[156:160], 'little')
    if not mask & statx_dioalign or not memory or not offset:
        raise RuntimeError("Stage B spill root does not support direct I/O "
                           "(statx reports no STATX_DIOALIGN)")
    return memory, offset


def _direct_io_error(exc, what, block):
    import errno
    if exc.errno == errno.EINVAL:
        return RuntimeError(
            f"Stage B spill direct-I/O {what} refused (EINVAL): a buffer address, "
            f"offset or length is off the {block}-byte grid, or the filesystem "
            "declined O_DIRECT")
    return exc


def _direct_batches(views, call_bytes, block):
    """Group block-aligned ``views`` into calls of at most ``call_bytes``.

    A view longer than a call is cut on block boundaries. A call also holds
    at most ``IOV_MAX`` vectors.
    """
    limit = os.sysconf('SC_IOV_MAX') if 'SC_IOV_MAX' in os.sysconf_names else 1024
    step = max(block, call_bytes - call_bytes % block)
    batch, size = [], 0
    for view in views:
        for start in range(0, len(view), step):
            piece = view[start:start + step]
            if batch and (size + len(piece) > step or len(batch) == limit):
                yield batch
                batch, size = [], 0
            batch.append(piece)
            size += len(piece)
    if batch:
        yield batch


def _iov_tail(views, skip):
    """The unwritten remainder of ``views`` after ``skip`` bytes."""
    out = []
    for view in views:
        if skip >= len(view):
            skip -= len(view)
            continue
        out.append(view[skip:] if skip else view)
        skip = 0
    return out


class EntryReadScratch:
    """One reusable read buffer for exact activation entries.

    The window owner holds one, so the buffer is allocated once for a
    generation rather than once per window: measured over one window, that is
    a Python-heap peak of 14 kB instead of 270 kB and no 16 MiB allocate/free
    per window (RobTand/prismaquant#735). Its wall-clock effect was not
    separable from pass-order effects in that bench, and none is claimed.
    It hands out the bytearray, never a view, so a live export can never make
    the next grow fail.
    """

    def __init__(self):
        self._buffer = bytearray()
        # Concurrent readers (PQ #1142) borrow one buffer each. The first is
        # this scratch's own, so a serial read uses exactly the buffer
        # :meth:`buffer` returns; each further reader that ever ran at once
        # adds one more, kept for the next window like the first.
        self._lock = threading.Lock()
        self._idle = None
        self._extra = []

    def buffer(self, size):
        if len(self._buffer) < size:
            self._buffer = bytearray(size)
        return self._buffer

    @contextmanager
    def lend(self):
        """One buffer for one of several concurrent readers.

        Yields an object with :meth:`buffer`, held by one reader until it
        returns it. The scratch never holds more buffers than the most
        readers that ran at once, which the exact-entry read pool bounds.
        Reusing a buffer is safe because ``torch.load`` copies every tensor
        out of it before the reader returns it.
        """
        with self._lock:
            if self._idle is None:
                self._idle = [self]
            if self._idle:
                slot = self._idle.pop()
            else:
                slot = EntryReadScratch()
                self._extra.append(slot)
        try:
            yield slot
        finally:
            with self._lock:
                if self._idle is not None:
                    self._idle.append(slot)

    def release(self):
        self._buffer = bytearray()
        with self._lock:
            for slot in self._extra:
                slot.release()
            self._extra = []
            self._idle = None


class _ExactActivationPrefetch:
    """One borrowed, closed resident window; lookups never perform I/O."""

    def __init__(self):
        self._tensors = {}
        self.active = False

    def get(self, reference):
        if not self.active or reference not in self._tensors:
            raise RuntimeError("exact activation window is not ready for this entry")
        return self._tensors[reference]


#: How the strict exact-entry reader leased its windows, this process
#: (PQ #997). ``windows_batched`` counts multi-entry lease windows and
#: ``entries_batched`` the entries they served; ``entries_single`` counts
#: entries leased one at a time, and ``batch_fallbacks`` the groups whose
#: batched lease was refused and re-read one entry at a time.
EXACT_LEASE_COUNTERS = {"windows_batched": 0, "entries_batched": 0,
                        "entries_single": 0, "batch_fallbacks": 0}


def exact_lease_counters() -> dict:
    """A copy of :data:`EXACT_LEASE_COUNTERS` for a report."""
    return dict(EXACT_LEASE_COUNTERS)


_EXACT_READ_POOL = None
_EXACT_READ_POOL_THREADS = 0
_EXACT_READ_POOL_LOCK = threading.Lock()


def exact_read_threads() -> int:
    """How many entries of one exact read window are read at once.

    The streamed-layer read knob, ``PRISMAQUANT_LAYER_READ_THREADS``
    (:func:`~prismaquant.layer_streaming.layer_read_threads`), so a run
    declares its read concurrency once. 1 restores the serial read.
    """
    from .layer_streaming import layer_read_threads
    return layer_read_threads()


def _exact_read_pool(threads):
    """The shared pool that reads exact entries, or ``None`` for one thread.

    Its own pool, not the layer-read pool: a caller that already reads
    several windows at once (a read-ahead thread, a plane comparison) waits
    on these workers, and a worker of one pool that waits on work queued to
    the same pool can deadlock it once every worker is waiting. Nothing
    running on this pool submits to it.
    """
    global _EXACT_READ_POOL, _EXACT_READ_POOL_THREADS
    if threads <= 1:
        return None
    with _EXACT_READ_POOL_LOCK:
        if _EXACT_READ_POOL is None or _EXACT_READ_POOL_THREADS != threads:
            if _EXACT_READ_POOL is not None:
                _EXACT_READ_POOL.shutdown(wait=False)
            _EXACT_READ_POOL = ThreadPoolExecutor(
                max_workers=threads, thread_name_prefix="exactread")
            _EXACT_READ_POOL_THREADS = threads
        return _EXACT_READ_POOL


_NOT_STARTED = object()


def _run_in_order(pool, calls):
    """Run ``calls`` on ``pool``; their results, in call order.

    Every call has returned before this does, on success and on failure,
    so no worker still reads a pinned descriptor once the caller releases
    its lease. A failure raises the exception of the lowest-indexed call
    that failed, and calls after a failure that have not started yet never
    start: the refusal a serial loop raises, whichever worker finished
    first. With no pool, or one call, the calls run here in order.
    """
    if pool is None or len(calls) < 2:
        return [call() for call in calls]
    # The lowest index that has failed so far. A worker checks it before it
    # starts a call, so a call queued after a failure never starts, even
    # when the pool picks it up before this thread sees the failure. Calls
    # below that index still run: a serial loop would have run them first.
    first_failure = [len(calls)]
    lock = threading.Lock()

    def guarded(index, call):
        with lock:
            if index > first_failure[0]:
                return _NOT_STARTED
        try:
            return call()
        except BaseException:
            with lock:
                first_failure[0] = min(first_failure[0], index)
            raise

    futures = [pool.submit(guarded, index, call) for index, call in enumerate(calls)]
    try:
        remaining = set(futures)
        while remaining:
            done, remaining = wait_futures(remaining, return_when=FIRST_EXCEPTION)
            failed = [index for index, future in enumerate(futures)
                      if future in done and not future.cancelled()
                      and future.exception() is not None]
            if failed:
                for future in futures[min(failed) + 1:]:
                    future.cancel()
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    finally:
        wait_futures(futures)
    for future in futures:
        # A cancelled call follows a failed one, which raises first.
        if not future.cancelled() and future.exception() is not None:
            raise future.exception()
    return [future.result() for future in futures]


def _declared_entry_resolver(resolver, ref):
    """The resolver ``ref`` resolves through, or ``None`` when none is bound.

    The same choice :func:`_strict_lease_groups` makes, for the non-strict
    read of the declared file: its bytes are counted as pool bytes on this
    resolver, so a run's tier counters sum to every byte it read.
    """
    from .residency_map import residency_resolver
    entry_resolver = resolver(ref) if callable(resolver) else resolver
    return entry_resolver if entry_resolver is not None else residency_resolver()


def _strict_lease_groups(references, resolver, *, deadline=None):
    """``[(entry resolver, [(ref, staged entry), ...]), ...]`` in read order.

    One group per resolver object: every entry of a group is vouched in one
    material namespace, which is what one lease window may pin. A window
    that spans a boundary plane and incoming cotangent planes of different
    produced batches is therefore one group per batch. The staged entry is
    looked up here exactly as the single-entry path looks it up, and a miss
    refuses the same way, before anything is pinned.

    A miss first waits for the entry's landing (PQ #1204): the entries of
    one resolver that the map does not hold yet are awaited together in one
    :func:`_await_entry_landing` call, under ``deadline``, and looked up
    again. The window has one deadline, however many resolvers it spans. An
    entry the map already holds costs one lookup, as before.
    """
    from .residency_map import residency_resolver
    from .staged_lease import LeaseRefused
    looked = []
    missed: dict[int, tuple[object, list[int]]] = {}
    for ref in references:
        entry_resolver = resolver(ref) if callable(resolver) else resolver
        if entry_resolver is None:
            entry_resolver = residency_resolver()
        if entry_resolver is None:
            raise LeaseRefused("readset-not-staged", kind="availability")
        staged = entry_resolver.staged_read(Path(ref.path), expected_sha256=ref.sha256)
        if staged is None:
            missed.setdefault(id(entry_resolver), (entry_resolver, []))[1].append(len(looked))
        looked.append((ref, entry_resolver, staged))
    if missed and deadline is None:
        # One bound for the whole window, however many resolvers it spans,
        # as ``await_staged_spans`` has one for the whole call.
        import time
        from .residency_shard_reader import staged_range_wait_s
        deadline = time.monotonic() + staged_range_wait_s()
    for entry_resolver, indices in missed.values():
        if not _await_entry_landing(
                entry_resolver,
                [(looked[index][0].path, looked[index][0].file_bytes) for index in indices],
                deadline=deadline):
            raise LeaseRefused("staged-not-serving", kind="availability")
        for index in indices:
            ref = looked[index][0]
            staged = entry_resolver.staged_read(Path(ref.path), expected_sha256=ref.sha256)
            if staged is None:
                raise LeaseRefused("staged-not-serving", kind="availability")
            looked[index] = (ref, entry_resolver, staged)
    groups: dict[int, tuple[object, list]] = {}
    for ref, entry_resolver, staged in looked:
        groups.setdefault(id(entry_resolver), (entry_resolver, []))[1].append((ref, staged))
    return list(groups.values())


def _enter_group_lease(entry_resolver, members, live_windows, counters=None):
    """Pin one group in at most two windows (RAM, then SSD), or ``None``.

    The group-granular form of ``acquire_entry_window``'s RAM-first rule.
    Entries offering a RAM copy share one RAM window when RAM is allowed,
    and the rest share one SSD window. When the RAM cover lookup refuses as
    availability for the whole set (the epoch moved, nothing is published
    at it, the tier is not announced), each of those entries records the
    RAM fallback and joins the SSD window, which is what the single-entry
    leg does for each of them.

    Every other refusal returns ``None`` after releasing whatever this call
    pinned, and the caller re-reads the group one entry at a time, keeping
    the single-entry refusal kinds exactly. This covers a partial RAM cover,
    any refusal of the SSD lookup, and any refusal to enter. A batched
    lookup that misses one key refuses the whole set as
    ``source-coverage-gap`` (integrity), where that entry alone would
    refuse ``unpublished`` (availability). A read is never served on a
    batched proof alone: the SDK re-verifies every key under the ownership
    lock at acquire, and every key is still opened through the SDK under
    the pin.

    ``members`` is ``[(ref, staged entry), ...]`` where ``ref.path`` is the
    declared file. ``counters`` is the report this call adds to
    (:data:`EXACT_LEASE_COUNTERS` unless the caller keeps its own, as the
    PWC retained window does).
    """
    from .staged_lease import LeaseRefused, acquire_entries_window
    if counters is None:
        counters = EXACT_LEASE_COUNTERS
    from .staged_tier_policy import tier_is_allowed
    ram = ([member for member in members if member[1].get("ram_path") is not None]
           if tier_is_allowed("ram") else [])
    ssd = [member for member in members if member not in ram]
    assignments, entered = {}, []

    def items(subset):
        return [(Path(ref.path), staged) for ref, staged in subset]

    try:
        plans = []
        if ram:
            try:
                plans.append(acquire_entries_window(entry_resolver, items(ram), tier="ram")
                             + (ram,))
            except LeaseRefused as refusal:
                # ``source-coverage-gap`` (a partial RAM cover) is classed
                # integrity, so it takes the per-entry path below.
                if refusal.kind != "availability":
                    raise
                for ref, _staged in ram:
                    entry_resolver.record_ram_fallback(Path(ref.path), str(refusal))
                ssd = list(members)
        if ssd:
            plans.append(acquire_entries_window(entry_resolver, items(ssd), tier="ssd")
                         + (ssd,))
        for lease_window, keys, subset in plans:
            lease_window.__enter__()
            # Owned by the caller's cleanup from the moment it is entered,
            # so a failed release below can never strand the pin.
            live_windows.append(lease_window)
            entered.append(lease_window)
            for (ref, _staged), key in zip(subset, keys):
                assignments[ref] = (lease_window, key)
    except LeaseRefused:
        for lease_window in entered:
            lease_window.__exit__(None, None, None)
            live_windows.remove(lease_window)
        counters["batch_fallbacks"] += 1
        return None
    counters["windows_batched"] += len(entered)
    counters["entries_batched"] += len(members)
    return assignments


def _exact_entry_prechecks(ref, *, expected_session, session_for_reference, path=None):
    """The declared-file fences every read runs first: ``(path, stat, signature)``.

    ``path`` is the file the read opens when it is not ``ref.path``: the
    owner's local copy of its own entry (PQ #1110). The session is always
    bound through ``ref``, the reference the owner recorded. A copy of it
    re-pointed at the local file is a reference the owner never recorded,
    so an owner that checks sessions itself (a resumed or forward-recovered
    run) refuses it as stale.
    """
    metadata = json.loads(ref.metadata_json)
    bound_session = (expected_session if session_for_reference is None
                     else session_for_reference(ref))
    if (metadata.get("schema") != EXACT_ACTIVATION_SCHEMA
            or metadata.get("identity", {}).get("session") != bound_session):
        raise RuntimeError("exact activation reference has a different session identity")
    path = Path(ref.path if path is None else path)
    prefetched_stat = path.lstat()
    signature = _activation_file_signature(path)
    if signature[2] != ref.file_bytes:
        raise RuntimeError("exact activation entry size changed")
    return path, prefetched_stat, signature


def _read_exact_entry(ref, *, path, signature, source, source_before, lease_fd,
                      owned, release_file_pages):
    """Read, hash and deserialize one entry; returns its verified tensor.

    ``lease_fd`` is the pinned descriptor under the strict policy (read, not
    closed, here) or ``None`` to open ``source``. The declared file's
    signature is re-checked after the hash and after the load.
    """
    raw = owned.buffer(ref.file_bytes)
    running = hashlib.sha256()
    consumed = 0
    opener = (source.open("rb", buffering=0) if lease_fd is None
              else os.fdopen(lease_fd, "rb", buffering=0, closefd=False))
    with opener as handle:
        while consumed < ref.file_bytes:
            view = memoryview(raw)[consumed:min(
                ref.file_bytes, consumed + _ENTRY_READ_BLOCK_BYTES)]
            try:
                size = handle.readinto(view)
                if not size:
                    raise RuntimeError("exact activation entry size changed")
                running.update(view[:size])
            finally:
                view.release()
            consumed += size
        if handle.read(1):
            raise RuntimeError("exact activation entry size changed")
    if running.hexdigest() != ref.sha256 or _activation_file_signature(path) != signature:
        raise RuntimeError("exact activation entry checksum changed")
    body = memoryview(raw)[:ref.file_bytes]
    reader = _VerifiedBufferReader(body, max_copy_bytes=ref.file_bytes)
    try:
        payload = torch.load(reader, map_location="cpu", weights_only=True)
    finally:
        reader.close()
        body.release()
    tensor = payload.get("inputs") if isinstance(payload, dict) else None
    if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
            or set(payload) != {"inputs", "name", "source", "exact"}
            or payload["name"] != ref.name or payload["source"] != "exact_activation"
            or _exact_activation_json(payload["exact"]) != ref.metadata_json
            or tuple(tensor.shape) != ref.shape or str(tensor.dtype) != ref.dtype
            or tensor.numel() * tensor.element_size() != ref.tensor_bytes
            or tensor.untyped_storage().nbytes() != ref.tensor_bytes
            or not tensor.is_contiguous() or tensor.requires_grad):
        raise RuntimeError("exact activation entry tensor/metadata differs from its receipt")
    if _activation_file_signature(path) != signature:
        raise RuntimeError("exact activation entry changed during prefetch")
    if release_file_pages:
        release_activation_cache_file_pages(source, expected_stat=source_before)
    return tensor


@contextmanager
def prefetch_exact_activation_cache_entries(references, *, max_tensor_bytes,
                                            expected_session, residency_check=None,
                                            release_file_pages=True, scratch=None,
                                            resolver=None, session_for_reference=None,
                                            local_paths=None, deadline=None):
    """Read/verify the entire bounded window before exposing any tensor.

    This is the existing activation artifact owner's exact-input read seam.
    The consumer owns no additional cache and receives no lazy-loading path.
    ``residency_check`` reserves/releases these tensors in its aggregate owner.

    ``resolver`` names which staged reader context this window resolves
    through. ``None`` -- every existing caller -- is the process input map,
    resolved exactly as before. An owner reading back its own produced
    batch passes that batch's namespaced context, or a callable answering
    one per reference when the window spans two batches; nothing else about
    the read changes, and every identity fence below still runs on the
    declared file.

    Under the strict tier policy the window is leased per material
    namespace, not per entry (PQ #997): the entries each resolver vouches
    share one pinned lease window per tier, opened key by key and released
    once after the group is verified. A refused batched lease re-reads that
    group one entry at a time, exactly as before. Every entry's staged row
    and session fence are checked before the window pins anything, so an
    unstaged entry refuses (``staged-not-serving``) before any entry of the
    window is read, where the per-entry reader had already read the ones
    before it. The refusal and its kind are unchanged.

    ``local_paths`` maps a reference the calling owner wrote on THIS box to
    the local file its produced-output spool still holds (PQ #1110). That
    entry is read from the local file, never through a staged tier and
    never from the pool, so the strict policy's pool refusal does not apply
    to it; every other fence is the same, run on the local file: the
    session identity, the size, the sha256 of every byte read, and the
    name, metadata, shape, dtype and storage size of the loaded tensor
    against the reference the owner recorded. The window keys the tensor
    by that reference, so the caller cannot tell which copy served it.

    Under the strict policy an entry the map does not hold yet is waited on
    before the window pins anything, on PrismaBuild's landing record when
    the sealed readset declares it (PQ #1204, :func:`_await_entry_landing`).
    ``deadline`` (``time.monotonic()``) is the caller's bound for that wait
    where no landing record covers the entries; ``None`` starts the
    configured bound at the first miss. An undeclared entry, or any
    terminal verdict of the wait, still refuses ``staged-not-serving``.

    The entries are read :func:`exact_read_threads` at a time (PQ #1142),
    each on its own buffer from ``scratch``, and each through every fence
    above before the window is exposed. Leases are entered and released on
    the calling thread, and a lease is released only after every read under
    it has returned. A refusal is the one the serial read raises: the
    first failing entry in window order, with its kind unchanged.
    """
    references = tuple(references)
    if any(not isinstance(ref, ExactActivationReference) for ref in references):
        raise TypeError("exact activation prefetch requires immutable references")
    if len(set(references)) != len(references):
        raise ValueError("exact activation window repeats an entry")
    nbytes = sum(ref.tensor_bytes for ref in references)
    if nbytes > max_tensor_bytes:
        raise RuntimeError("exact activation prefetch exceeds tensor residency budget")
    window = _ExactActivationPrefetch()
    reserved = False
    owned = None
    live_windows: list = []
    from .staged_tier_policy import policy_is_active
    strict = policy_is_active()
    prechecks = dict(expected_session=expected_session,
                     session_for_reference=session_for_reference)
    local_paths = dict(local_paths or {})
    if any(ref not in references for ref in local_paths):
        raise ValueError("a local exact entry is not in the window it is read for")
    pool = _exact_read_pool(exact_read_threads())

    def read_local(ref):
        """The owner's own copy on this box: the same fences, on that file."""
        path, prefetched_stat, signature = _exact_entry_prechecks(
            ref, path=local_paths[ref], **prechecks)
        local = _dataclass_replace(ref, path=str(path))
        with owned.lend() as buffer:
            window._tensors[ref] = _read_exact_entry(
                local, path=path, signature=signature, source=path,
                source_before=prefetched_stat, lease_fd=None, owned=buffer,
                release_file_pages=release_file_pages)

    def read_single(ref, lease_resolver=None):
        """The single-entry read: its own window, released after verifying."""
        path, prefetched_stat, signature = _exact_entry_prechecks(ref, **prechecks)
        source, source_before = path, prefetched_stat
        lease_window = lease_fd = None
        if strict:
            lease_window, lease_key, _staged, lease_resolver = (
                _acquire_bulk_window(path, ref.sha256, resolver=lease_resolver,
                                     declared_size=ref.file_bytes, deadline=deadline))
            live_windows.append(lease_window)
            try:
                lease_fd, _serving, tier = _enter_and_open_window(
                    lease_resolver, lease_window, lease_key, path)
            except BaseException:
                # ``_enter_and_open_window`` already exited the window on
                # every failure it raises, so leaving it in the live list
                # makes the cleanup below exit it a SECOND time -- and a
                # released manager refuses re-exit, which replaces the
                # real refusal with "LeaseWindow re-exit is refused" and
                # hides why the read failed.
                live_windows.remove(lease_window)
                raise
            source = Path(lease_window.stage_path(lease_key) or path)
            source_before = os.fstat(lease_fd)
            EXACT_LEASE_COUNTERS["entries_single"] += 1
        else:
            # The declared file itself: pool bytes, counted on the resolver
            # this entry would be staged through, when one is bound.
            tier = None
            lease_resolver = _declared_entry_resolver(resolver, ref)
        with owned.lend() as buffer:
            window._tensors[ref] = _read_exact_entry(
                ref, path=path, signature=signature, source=source,
                source_before=source_before, lease_fd=lease_fd, owned=buffer,
                release_file_pages=release_file_pages)
        _record_served_bytes(lease_resolver, path, tier, ref.file_bytes)
        if lease_window is not None:
            # Entry verified: descriptor closed, exact ref released
            # before the next entry acquires.
            lease_window.__exit__(None, None, None)
            live_windows.remove(lease_window)

    def read_leased(entry_resolver, assignments, ref, path, signature):
        """One entry of a group, opened and read under the group's pin."""
        from .staged_lease import LeaseRefused
        lease_window, key = assignments[ref]
        try:
            lease_fd, serving = lease_window.open(key)
        except LeaseRefused as refusal:
            entry_resolver.record_fallback(path, str(refusal))
            raise
        tier = lease_window.serving_tier or "stage"
        entry_resolver.record_serving_tier(
            path, tier,
            pin_id=str(serving.get("pin_id") or ""),
            range_ref=str(serving.get("range_ref") or ""))
        try:
            with owned.lend() as buffer:
                window._tensors[ref] = _read_exact_entry(
                    ref, path=path, signature=signature,
                    source=Path(lease_window.stage_path(key) or path),
                    source_before=os.fstat(lease_fd), lease_fd=lease_fd,
                    owned=buffer, release_file_pages=release_file_pages)
        finally:
            lease_window.close_fd(lease_fd)
        _record_served_bytes(entry_resolver, path, tier, ref.file_bytes)

    def read_group(entry_resolver, members):
        """One material namespace's entries under shared lease windows.

        The fences before the pin, and the entries under it, run on the
        exact-read pool; the pin is entered and released here, on the
        calling thread, and only after every entry's read has returned.
        """
        checked = _run_in_order(pool, [partial(_exact_entry_prechecks, ref, **prechecks)
                                       for ref, _ in members])
        assignments = (_enter_group_lease(entry_resolver, members, live_windows)
                       if len(members) > 1 else None)
        if assignments is None:
            for ref, _staged in members:
                read_single(ref, entry_resolver)
            return
        _run_in_order(pool, [
            partial(read_leased, entry_resolver, assignments, ref, path, signature)
            for (ref, _staged), (path, _prefetched_stat, signature) in zip(members, checked)])
        # Every entry verified: each window closes its descriptors (already
        # closed) and releases its one ref.
        for lease_window in {id(w): w for w, _key in assignments.values()}.values():
            lease_window.__exit__(None, None, None)
            live_windows.remove(lease_window)

    try:
        if residency_check is not None:
            residency_check(nbytes)
            reserved = True
        # Each entry is hashed as it fills this buffer and deserialized from
        # it, so it is read once instead of once to hash and once to load. The
        # buffer holds one entry, not the window; a caller that owns an
        # EntryReadScratch keeps it across windows.
        owned = EntryReadScratch() if scratch is None else scratch
        _run_in_order(pool, [partial(read_local, ref)
                             for ref in references if ref in local_paths])
        staged = tuple(ref for ref in references if ref not in local_paths)
        if strict:
            for entry_resolver, members in _strict_lease_groups(
                    staged, resolver, deadline=deadline):
                read_group(entry_resolver, members)
        else:
            _run_in_order(pool, [partial(read_single, ref) for ref in staged])
        if scratch is None:
            owned.release()
        window.active = True
        yield window
    finally:
        window.active = False
        window._tensors.clear()
        owned = None
        while live_windows:
            # Error paths must not strand pins: close descriptors and
            # release exact refs, first failure raised last.
            live_windows.pop().__exit__(None, None, None)
        if reserved:
            residency_check(-nbytes)


def _tensor_hash_update(h: "hashlib._Hash", tensor: torch.Tensor) -> None:
    t = tensor.detach().to("cpu").contiguous()
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    h.update(t.view(torch.uint8).numpy().tobytes())


def calibration_data_hash(calibration_data) -> str:
    """Stable content hash used to seed shared row subsampling."""
    h = hashlib.blake2b(digest_size=16)
    if isinstance(calibration_data, torch.Tensor):
        _tensor_hash_update(h, calibration_data)
        return h.hexdigest()
    if isinstance(calibration_data, Mapping):
        for key in sorted(calibration_data):
            h.update(str(key).encode())
            value = calibration_data[key]
            if isinstance(value, torch.Tensor):
                _tensor_hash_update(h, value)
            else:
                h.update(repr(value).encode())
        return h.hexdigest()
    for sample in calibration_data:
        if isinstance(sample, torch.Tensor):
            _tensor_hash_update(h, sample)
        elif isinstance(sample, Mapping):
            for key in sorted(sample):
                h.update(str(key).encode())
                value = sample[key]
                if isinstance(value, torch.Tensor):
                    _tensor_hash_update(h, value)
                else:
                    h.update(repr(value).encode())
        else:
            h.update(repr(sample).encode())
    return h.hexdigest()


def _seed_from(cal_hash: str, group_key: str) -> int:
    digest = hashlib.blake2b(
        f"{cal_hash}:{group_key}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def fused_subsample_group(name: str, profile=None) -> str:
    """Return the deterministic row-subsample group for a recipe name."""
    if profile is not None:
        try:
            group = profile.fused_sibling_group(name)
            if group is not None:
                return str(group)
        except Exception:
            pass
    bare = name[:-7] if name.endswith(".weight") else name
    parent, _, leaf = bare.rpartition(".")
    if leaf in {"q_proj", "k_proj", "v_proj"}:
        return f"{parent.rsplit('.', 1)[0]}.qkv"
    if leaf in {"gate_proj", "up_proj"}:
        return f"{parent.rsplit('.', 1)[0]}.gate_up"
    if leaf in {"in_proj_qkv", "in_proj_z"}:
        return f"{parent.rsplit('.', 1)[0]}.in_proj_qkvz"
    if leaf in {"in_proj_a", "in_proj_b"}:
        return f"{parent.rsplit('.', 1)[0]}.in_proj_ab"
    return bare


class SharedRowSubsampler:
    """Deterministic, sibling-coherent row sampling for activation capture.

    Fused siblings (q/k/v, gate/up) must snapshot the SAME rows so their
    caches stay row-aligned for joint solvers.  ``batch_priorities``
    returns the per-row random reservoir priorities for one capture
    batch, keyed by the fused-sibling *group* and the batch index —
    every sibling observing the same batch therefore draws identical
    priorities, and the reservoir's keep/replace decisions match
    row-for-row across the whole calibration stream (the cross-batch
    analogue of the historical shared per-call ``randperm``).

    Seeding follows the existing convention: ``_seed_from(cal_hash, key)``
    so a fixed calibration set reproduces the same sample run-to-run."""

    def __init__(self, input_rows: int, cal_hash: str, profile=None):
        self.input_rows = int(input_rows)
        self.cal_hash = cal_hash
        self.profile = profile

    def batch_priorities(
        self,
        name: str,
        batch_index: int,
        n_rows: int,
    ) -> torch.Tensor:
        """Random priorities for the ``batch_index``-th capture of ``name``.

        Regenerated deterministically per (group, batch) instead of cached:
        zero retained state, and siblings that consume the same batch at
        different times still agree exactly."""
        group = fused_subsample_group(name, self.profile)
        g = torch.Generator(device="cpu")
        g.manual_seed(
            _seed_from(self.cal_hash, f"{group}#batch{int(batch_index)}")
        )
        return torch.rand(int(n_rows), generator=g, dtype=torch.float32)


@dataclass
class _ParamPlan:
    name: str
    attr: str
    spec: fr.FormatSpec


@dataclass
class _ModulePlan:
    module: nn.Module
    params: list[_ParamPlan] = field(default_factory=list)
    active_originals: list[tuple[torch.nn.Parameter, torch.Tensor]] = field(
        default_factory=list
    )
    act_spec: fr.FormatSpec | None = None
    act_conflict: bool = False

    @property
    def cache_names(self) -> list[str]:
        return [p.name for p in self.params]


def _module_input_member_name(plan: _ModulePlan, x: torch.Tensor) -> str | None:
    """Pick the plan member whose calibrated scale describes ``x``.

    Dense Linears have one ``weight`` param — trivially it. A packed-MoE
    experts module carries one plan with SEVERAL per-projection params
    (``gate_up_proj`` + ``down_proj``); their calibrated ``max_abs`` were
    measured on DIFFERENT tensors (module input vs routed post-SwiGLU
    intermediates), and this hook only ever sees the module input, so the
    clip scale must come from the projection that consumes it. Select
    structurally: the param whose in-dim (last weight axis, [E, out, in])
    equals ``x``'s feature dim. Falling back to ``params[0]`` (the old
    behavior) made the clip depend on assignment-dict ordering.
    """
    member_name = next(
        (p.name for p in plan.params if p.attr == "weight"), None)
    if member_name is not None or not plan.params:
        return member_name
    in_dim = int(x.size(-1))
    matches = []
    for p in plan.params:
        tensor = getattr(plan.module, p.attr, None)
        shape = getattr(tensor, "shape", None)
        if shape is not None and len(shape) >= 1 and int(shape[-1]) == in_dim:
            matches.append(p)
    if not matches:
        return plan.params[0].name
    if len(matches) > 1:
        # Degenerate square case (intermediate == hidden): the shape test
        # cannot separate the projections, so break the tie by role — the
        # down projection (down_proj / w2) consumes the internal
        # intermediate, never the module input.
        non_down = [
            p for p in matches
            if p.attr.rsplit(".", 1)[-1] not in ("down_proj", "w2")
        ]
        if non_down:
            return non_down[0].name
    return matches[0].name


def build_quantizable_map(
    model: nn.Module,
    profile=None,
) -> dict[str, tuple[nn.Module, str]]:
    """Map recipe/probe names to live module parameters."""
    out: dict[str, tuple[nn.Module, str]] = {}
    for full_name, mod, attr in iter_quantizable_tensors(model, profile):
        names = {full_name}
        if full_name.endswith(".weight"):
            names.add(full_name[:-7])
        if profile is not None:
            qname = (
                full_name[:-7]
                if attr == "weight" and full_name.endswith(".weight")
                else full_name
            )
            try:
                recipe_name = profile.live_to_recipe_name(qname)
            except Exception:
                recipe_name = qname
            names.add(recipe_name)
            if attr == "weight":
                names.add(f"{recipe_name}.{attr}")
        for name in list(names):
            if name.startswith("model."):
                suffix = name[len("model."):]
                names.add(f"model.language_model.{suffix}")
        for name in names:
            out[name] = (mod, attr)
    return out


def _build_module_plans(
    model: nn.Module,
    assignment: Mapping[str, str],
    profile=None,
) -> tuple[list[_ModulePlan], list[str], list[dict]]:
    quant_map = build_quantizable_map(model, profile=profile)
    by_module: dict[int, _ModulePlan] = {}
    missing: list[str] = []
    for name, fmt in assignment.items():
        target = quant_map.get(name)
        if target is None:
            missing.append(name)
            continue
        mod, attr = target
        spec = fr.get_format(fmt)
        plan = by_module.setdefault(id(mod), _ModulePlan(module=mod))
        plan.params.append(_ParamPlan(name=name, attr=attr, spec=spec))

    skipped: list[dict] = []
    for plan in by_module.values():
        low_act = {
            p.spec.name: p.spec
            for p in plan.params
            if p.spec.act_quant_changes_input
        }
        if len(low_act) == 1:
            plan.act_spec = next(iter(low_act.values()))
        elif len(low_act) > 1:
            plan.act_conflict = True
            skipped.append(
                {
                    "module": type(plan.module).__name__,
                    "weights": sorted(plan.cache_names),
                    "formats": sorted(low_act),
                }
            )
    return list(by_module.values()), missing, skipped


def _first_tensor_location(args, kwargs):
    if args:
        for idx, value in enumerate(args):
            if isinstance(value, torch.Tensor):
                return "args", idx, value
    if kwargs:
        for key in ("hidden_states", "inputs_embeds", "input"):
            value = kwargs.get(key)
            if isinstance(value, torch.Tensor):
                return "kwargs", key, value
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                return "kwargs", key, value
    return None, None, None


def _replace_tensor_input(args, kwargs, where, key, value):
    if where == "args":
        args_list = list(args)
        args_list[int(key)] = value
        return tuple(args_list), kwargs
    if where == "kwargs":
        kwargs = dict(kwargs or {})
        kwargs[key] = value
        return args, kwargs
    return args, kwargs


class PerturbedActivationCache:
    def __init__(
        self,
        model: nn.Module,
        assignment: Mapping[str, str],
        cache_dir: str | Path,
        *,
        input_rows: int = 256,
        cal_hash: str,
        profile=None,
        production_weight_cache=None,
        include_activation_quant: bool = True,
        capture_inputs: bool = True,
    ):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.input_rows = int(input_rows)
        self.include_activation_quant = bool(include_activation_quant)
        self.capture_inputs = bool(capture_inputs)
        self.subsampler = SharedRowSubsampler(input_rows, cal_hash, profile)
        self.plans, self.missing, self.skipped = _build_module_plans(
            model, assignment, profile=profile
        )
        self._production_weight_cache = production_weight_cache
        # MED-3: per-Linear calibrated max(|activations|), unified across
        # fused-sibling groups.  Used by the activation-quant hook to
        # clamp activations to ±max_abs before per-group RTN, matching
        # the export's act-clip behavior.  See production_weight_cache.py
        # for the convention note (we store max_abs directly; the export's
        # vLLM-facing metadata is derived via
        # export_native_compressed._nvfp4_input_global_scale_from_max_abs).
        if production_weight_cache is not None and (
            production_weight_cache.activation_max_abs
            or production_weight_cache.activation_scales
        ):
            src = (
                production_weight_cache.activation_max_abs
                or production_weight_cache.activation_scales
            )
            self._activation_scales: dict[str, float] = dict(src)
        else:
            self._activation_scales = {}
        # #227: the maximum above says what the unit's activations reach; it
        # does NOT say which input-global-scale policy the cache's costs were
        # priced under, and the same maximum prices G=6/amax or G=448*6/amax.
        # The cache's own render-score provenance does say, so carry the G each
        # unit was priced at and let the hook refuse by name when the G it
        # would apply is a different one.
        self._priced_input_global_scales: dict[str, float] = {}
        if production_weight_cache is not None:
            from prismaquant.production_weight_cache import (
                production_cache_priced_input_global_scales,
            )

            self._priced_input_global_scales = (
                production_cache_priced_input_global_scales(
                    production_weight_cache,
                    where="assignment-KL hooks",
                )
            )
        # Bounded uniform row reservoirs (M8): per name, at most
        # `input_rows` CPU rows + their float32 priorities, plus a batch
        # counter that keys the shared per-batch priorities.
        self._snap_rows: dict[str, torch.Tensor] = {}
        self._snap_priorities: dict[str, torch.Tensor] = {}
        self._snap_batches: dict[str, int] = defaultdict(int)
        self.max_abs: dict[str, float] = {}
        self._handles = []
        self._frozen_weight_cache: OrderedDict[
            tuple[int, str], torch.Tensor
        ] | None = None
        self._frozen_weight_format_cache: OrderedDict[
            tuple[str, str, int, str, str], torch.Tensor
        ] = (
            _SHARED_FROZEN_WEIGHT_FORMAT_CACHE
            if _env_truthy("PRISMAQUANT_SHARED_WEIGHT_FORMAT_CACHE")
            else OrderedDict()
        )
        self._fused_forward_originals: list[tuple[nn.Module, object]] = []
        self._fused_nvfp4_weight_cache: OrderedDict[
            tuple[str, str, str, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = OrderedDict()
        self._materialized_frozen_weight_depth = 0
        self._frozen_weight_cache_evictions = 0
        self._frozen_weight_cache_eviction_reported = False
        register_budget_evictor(self)

    @property
    def installed(self) -> bool:
        return bool(self._handles)

    def install(self) -> None:
        for plan in self.plans:
            if self._try_install_nvfp4_fused_forward(plan):
                continue
            self._install_packed_expert_activation_quant(plan)
            self._handles.append(
                plan.module.register_forward_pre_hook(
                    self._make_pre_hook(plan),
                    with_kwargs=True,
                )
            )
            self._handles.append(
                plan.module.register_forward_hook(
                    self._make_post_hook(plan),
                    with_kwargs=True,
                )
            )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for module, original_forward in reversed(self._fused_forward_originals):
            module.forward = original_forward
        self._fused_forward_originals.clear()
        for plan in self.plans:
            self._restore_plan(plan)

    def _find_param_plan(self, name: str) -> tuple[_ModulePlan, _ParamPlan]:
        for plan in self.plans:
            for param_plan in plan.params:
                if param_plan.name == name:
                    return plan, param_plan
        raise KeyError(f"no quantized parameter named {name!r}")

    def _quantized_weight_for(
        self,
        plan: _ModulePlan,
        param_plan: _ParamPlan,
        spec: fr.FormatSpec,
    ) -> torch.Tensor | None:
        param = getattr(plan.module, param_plan.attr)
        if not isinstance(param, torch.nn.Parameter) or param.is_meta:
            return None
        fmt = fr.canonical_format_name(spec.name)
        # Include production-cache identity in the key so a SHARED
        # frozen_weight_format_cache that's seen multiple instances
        # (with/without production cache, or different production
        # caches) doesn't return stale entries.  Same-instance reuse
        # across polish trials still hits because id() is stable.
        prod_id = (
            id(self._production_weight_cache)
            if self._production_weight_cache is not None else 0
        )
        cache_key = (
            param_plan.name,
            fmt,
            int(param.data_ptr()),
            str(param.device),
            str(param.dtype),
            prod_id,
        )
        q = self._frozen_weight_format_cache.get(cache_key)
        if q is None:
            enforce_gpu_memory_budget(
                [self],
                device=param.device if param.device.type == "cuda" else None,
                reason="frozen weight cache fill",
            )
            production = (
                self._production_weight_cache.get(param_plan.name, fmt)
                if self._production_weight_cache is not None
                else None
            )
            if production is not None:
                q = production.to(
                    device=param.device,
                    dtype=param.dtype,
                ).contiguous()
            else:
                from .nvfp4_cb_footprint import is_cb_format

                if is_cb_format(fmt):
                    raise RuntimeError(
                        f"production_weight_cache is required for CB fallback "
                        f"({param_plan.name!r}, {fmt!r}); the registry path is "
                        "unweighted legacy rendering and cannot represent the "
                        "stamped production serialization contract"
                    )
                if (
                    self._production_weight_cache is not None
                    and fmt != "BF16"
                    and _env_truthy(
                        "PRISMAQUANT_STRICT_PRODUCTION_CACHE",
                        default=True,
                    )
                ):
                    raise RuntimeError(
                        f"production_weight_cache miss for "
                        f"({param_plan.name!r}, {fmt!r}); set "
                        f"PRISMAQUANT_STRICT_PRODUCTION_CACHE=0 to fall back "
                        f"to RTN, or rebuild the cache to cover this Linear."
                    )
                original = param.data.detach().clone()
                q = spec.quantize_dequantize(original).to(
                    device=param.device,
                    dtype=param.dtype,
                ).contiguous()
            # cache_key now includes production-cache identity, so we
            # can safely populate the shared cache regardless of
            # production-active state.  Different production caches
            # (or no cache) get distinct keys; no cross-contamination.
            if self._frozen_weight_cache_max_entries() > 0:
                self._frozen_weight_format_cache[cache_key] = q
                self._evict_frozen_weight_format_cache_to_limit()
            enforce_gpu_memory_budget(
                [self],
                device=param.device if param.device.type == "cuda" else None,
                reason="frozen weight cache fill",
            )
        else:
            self._frozen_weight_format_cache.move_to_end(cache_key)
        return q

    def build_frozen_weight_cache(self) -> dict[tuple[int, str], torch.Tensor]:
        cache: OrderedDict[tuple[int, str], torch.Tensor] = OrderedDict()
        for plan in self.plans:
            seen_attrs: set[str] = set()
            for param_plan in plan.params:
                if param_plan.attr in seen_attrs:
                    continue
                seen_attrs.add(param_plan.attr)
                q = self._quantized_weight_for(plan, param_plan, param_plan.spec)
                if q is None:
                    continue
                cache[(id(plan.module), param_plan.attr)] = q
        self._frozen_weight_cache = cache
        return cache

    @contextmanager
    def frozen_weight_cache(self) -> Iterator["PerturbedActivationCache"]:
        previous = self._frozen_weight_cache
        self.build_frozen_weight_cache()
        try:
            yield self
        finally:
            self._frozen_weight_cache = previous
            self._emit_frozen_weight_cache_evictions()

    @contextmanager
    def materialized_frozen_weights(self) -> Iterator["PerturbedActivationCache"]:
        """Apply the active frozen weights to modules for whole-forward reuse."""
        if self._frozen_weight_cache is None:
            raise RuntimeError("frozen weight cache is not active")
        if self._materialized_frozen_weight_depth > 0:
            self._materialized_frozen_weight_depth += 1
            try:
                yield self
            finally:
                self._materialized_frozen_weight_depth -= 1
            return

        originals: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        seen_keys: set[tuple[int, str]] = set()
        self._materialized_frozen_weight_depth = 1
        try:
            for plan in self.plans:
                for param_plan in plan.params:
                    cache_key = (id(plan.module), param_plan.attr)
                    if cache_key in seen_keys:
                        continue
                    seen_keys.add(cache_key)
                    param = getattr(plan.module, param_plan.attr)
                    if not isinstance(param, torch.nn.Parameter) or param.is_meta:
                        continue
                    q = self._frozen_weight_cache.get(cache_key)
                    if q is None:
                        continue
                    self._frozen_weight_cache.move_to_end(cache_key)
                    originals.append((param, param.data.detach().clone()))
                    param.data.copy_(q.to(device=param.device, dtype=param.dtype))
            yield self
        finally:
            for param, original in reversed(originals):
                param.data.copy_(original.to(device=param.device, dtype=param.dtype))
            self._materialized_frozen_weight_depth = 0

    def set_frozen_weight_format(self, name: str, fmt: str) -> None:
        if self._frozen_weight_cache is None:
            raise RuntimeError("frozen weight cache is not active")
        plan, param_plan = self._find_param_plan(name)
        spec = fr.get_format(fmt)
        q = self._quantized_weight_for(plan, param_plan, spec)
        if q is None:
            return
        self._frozen_weight_cache[(id(plan.module), param_plan.attr)] = q
        self._frozen_weight_cache.move_to_end((id(plan.module), param_plan.attr))
        param_plan.spec = spec

    @contextmanager
    def temporary_frozen_weight_format(
        self,
        name: str,
        fmt: str,
    ) -> Iterator["PerturbedActivationCache"]:
        with self.override({name: fmt}):
            yield self

    @contextmanager
    def override(
        self,
        assignment_delta: Mapping[str, str],
    ) -> Iterator["PerturbedActivationCache"]:
        if self._frozen_weight_cache is None:
            raise RuntimeError("frozen weight cache is not active")
        previous: list[
            tuple[tuple[int, str], torch.Tensor | None, _ParamPlan, fr.FormatSpec]
        ] = []
        for name, fmt in assignment_delta.items():
            plan, param_plan = self._find_param_plan(name)
            cache_key = (id(plan.module), param_plan.attr)
            previous.append(
                (
                    cache_key,
                    self._frozen_weight_cache.get(cache_key),
                    param_plan,
                    param_plan.spec,
                )
            )
            self.set_frozen_weight_format(name, fmt)
        try:
            yield self
        finally:
            for cache_key, previous_q, param_plan, previous_spec in reversed(previous):
                if previous_q is None:
                    self._frozen_weight_cache.pop(cache_key, None)
                else:
                    self._frozen_weight_cache[cache_key] = previous_q
                    self._frozen_weight_cache.move_to_end(cache_key)
                param_plan.spec = previous_spec

    def _capture(self, plan: _ModulePlan, x: torch.Tensor) -> None:
        if not self.capture_inputs:
            return
        flat = x.detach().reshape(-1, x.size(-1))
        for name in plan.cache_names:
            mx = float(flat.abs().max().item())
            if mx > self.max_abs.get(name, 0.0):
                self.max_abs[name] = mx
            if self.input_rows <= 0:
                continue
            self._reservoir_update(name, flat)

    def _reservoir_update(self, name: str, flat: torch.Tensor) -> None:
        """Fold one capture batch into ``name``'s bounded uniform reservoir.

        M8 fix: the old path kept the FIRST ``input_rows`` rows of the
        calibration stream (``need = input_rows - rows_got``; all later
        batches skipped), so with default sizes the entire perturbed-X
        second moment came from calibration document #1.  This is the
        same priority-reservoir scheme as
        ``activation_sampling.update_priority_reservoir`` — uniform
        without replacement over ALL rows seen, storage bounded at
        ``input_rows`` — with two deltas that matter here:

          * priorities come from ``SharedRowSubsampler.batch_priorities``
            (keyed by fused-sibling group + batch index), so gate/up and
            q/k/v siblings keep IDENTICAL row sets across the whole
            stream, not just within one call;
          * only surviving rows are copied device→CPU
            (``update_priority_reservoir`` concatenates the full incoming
            batch onto the CPU reservoir first, which would move every
            calibration activation over the bus per module per batch).
        """
        limit = self.input_rows
        batch_index = self._snap_batches[name]
        self._snap_batches[name] = batch_index + 1
        new_pri = self.subsampler.batch_priorities(
            name, batch_index, int(flat.size(0))
        )
        cur_rows = self._snap_rows.get(name)
        cur_pri = self._snap_priorities.get(name)
        n_cur = 0 if cur_rows is None else int(cur_rows.size(0))
        merged_pri = (
            new_pri if n_cur == 0 else torch.cat([cur_pri, new_pri], dim=0)
        )
        if int(merged_pri.numel()) <= limit:
            incoming = flat.to("cpu")
            if incoming is flat:
                # `.to("cpu")` is a no-op for CPU inputs; clone so the
                # reservoir never aliases live activation storage.
                incoming = incoming.clone()
            self._snap_rows[name] = (
                incoming
                if cur_rows is None
                else torch.cat([cur_rows, incoming], dim=0)
            )
            self._snap_priorities[name] = merged_pri
            return
        keep = torch.topk(merged_pri, k=limit, largest=True, sorted=False).indices
        # Ascending order keeps retained rows in stream order and makes
        # the old-rows/new-rows concatenation below line up with the
        # reordered priorities (old indices < n_cur <= new indices).
        keep = torch.sort(keep).values
        keep_old = keep[keep < n_cur]
        keep_new = keep[keep >= n_cur] - n_cur
        parts: list[torch.Tensor] = []
        if keep_old.numel():
            parts.append(cur_rows.index_select(0, keep_old))
        if keep_new.numel():
            parts.append(
                flat.index_select(0, keep_new.to(flat.device)).to("cpu")
            )
        self._snap_rows[name] = (
            parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        )
        self._snap_priorities[name] = merged_pri.index_select(0, keep)

    def _apply_weight_quant(self, plan: _ModulePlan) -> None:
        plan.active_originals.clear()
        if self._materialized_frozen_weight_depth > 0:
            return
        if _env_truthy("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT", default=False):
            # Caller (e.g. WeightSession) has installed the desired weights
            # directly on model.params; we just observe + activation-
            # quantize, no clone/restore.  Saves ~50 MB clone per module
            # on the hot path and lets polish on big models avoid the
            # cumulative-clone OOM.
            return
        seen_attrs: set[str] = set()
        for param_plan in plan.params:
            if param_plan.attr in seen_attrs:
                continue
            seen_attrs.add(param_plan.attr)
            param = getattr(plan.module, param_plan.attr)
            if not isinstance(param, torch.nn.Parameter) or param.is_meta:
                continue
            original = param.data.detach().clone()
            q = None
            if self._frozen_weight_cache is not None:
                cache_key = (id(plan.module), param_plan.attr)
                q = self._frozen_weight_cache.get(cache_key)
                if q is not None:
                    self._frozen_weight_cache.move_to_end(cache_key)
            if q is None and _env_truthy("PRISMAQUANT_SHARED_WEIGHT_FORMAT_CACHE"):
                q = self._quantized_weight_for(plan, param_plan, param_plan.spec)
            if q is None and self._production_weight_cache is not None:
                fmt_canon = fr.canonical_format_name(param_plan.spec.name)
                production = self._production_weight_cache.get(
                    param_plan.name, fmt_canon,
                )
                if production is not None:
                    q = production.to(
                        device=param.device, dtype=param.dtype,
                    ).contiguous()
                elif (
                    fmt_canon != "BF16"
                    and _env_truthy(
                        "PRISMAQUANT_STRICT_PRODUCTION_CACHE",
                        default=True,
                    )
                ):
                    raise RuntimeError(
                        f"production_weight_cache miss for "
                        f"({param_plan.name!r}, {fmt_canon!r}); set "
                        f"PRISMAQUANT_STRICT_PRODUCTION_CACHE=0 to allow "
                        f"RTN fallback."
                    )
            if q is None:
                from .nvfp4_cb_footprint import is_cb_format

                fmt_canon = fr.canonical_format_name(param_plan.spec.name)
                if is_cb_format(fmt_canon):
                    raise RuntimeError(
                        f"production_weight_cache is required for CB fallback "
                        f"({param_plan.name!r}, {fmt_canon!r}); refusing an "
                        "unweighted legacy registry render"
                    )
                if fr.is_tessera_format_name(fmt_canon):
                    raise RuntimeError(
                        f"production_weight_cache is required for Tessera "
                        f"({param_plan.name!r}, {fmt_canon!r}); the registry "
                        "render is a weights-only reconstruction, not the "
                        "decoded wire and not the H-aware encode that ships"
                    )
                q = param_plan.spec.quantize_dequantize(original)
            if q is None:
                continue
            param.data.copy_(q.to(device=param.device, dtype=param.dtype))
            plan.active_originals.append((param, original))

    def _active_activation_spec(self, plan: _ModulePlan) -> fr.FormatSpec | None:
        if not self.include_activation_quant:
            return None
        low_act = {
            p.spec.name: p.spec
            for p in plan.params
            if p.spec.act_quant_changes_input
        }
        if len(low_act) == 1:
            return next(iter(low_act.values()))
        return None

    def _served_measurement_units(self) -> list[tuple[str, object]]:
        """``(name, contract)`` for every member the hook prices as served.

        One enumeration for both preflights below, so "which units does this
        cache measure as served, and under whose contract" cannot be answered
        two ways.  The contract travels with the name because the G rule is
        the SPEC's, never a name comparison (#205).
        """
        if not self.include_activation_quant:
            return []
        units: list[tuple[str, object]] = []
        for plan in self.plans:
            members = self._packed_act_plan(plan)
            if members is None:
                act_spec = self._active_activation_spec(plan)
                contract = getattr(act_spec, "static_activation_contract", None)
                if contract is None or not contract.measured_as_served:
                    continue
                # The name the hook looks the scale up under: the dense
                # ``weight`` member (``_module_input_member_name``), else
                # every member -- without an input tensor the structural
                # tie-break cannot run, and refusing one name too many is
                # the safe side.
                weight = next(
                    (p.name for p in plan.params if p.attr == "weight"), None)
                names = [weight] if weight is not None else plan.cache_names
                units.extend((name, contract) for name in names)
            else:
                units.extend(
                    (m.name, m.spec.static_activation_contract)
                    for m in members
                    if getattr(m.spec.static_activation_contract,
                               "measured_as_served", False)
                )
        return units

    def served_activation_scale_gaps(self) -> list[str]:
        """Names this cache would have to REFUSE in the hook, listed up front.

        A member whose spec is measured under the served static-scale
        contract (``FormatSpec.static_activation_contract.measured_as_served``,
        a Tessera W4A4 rung) needs its calibrated maximum in this cache's
        scale identity (``activation_max_abs`` from the production cache);
        ``_activation_qdq`` refuses it by name otherwise.  Consumers that
        measure (``kl_measurement.measure_assignment_kl``) ask this before the
        first forward so the refusal names every unit at once instead of the
        first hook the model happens to reach.  Capture-only builders, which
        run before any maximum exists, are not asked.
        """
        gaps: set[str] = set()
        for name, _contract in self._served_measurement_units():
            value = _activation_max_abs_lookup(self._activation_scales, name)
            if value is None or float(value) <= 0.0:
                gaps.add(name)
        return sorted(gaps)

    def served_activation_policy_conflicts(self) -> list[str]:
        """Names whose cached cost was priced at a different static G (#227).

        The sibling of :meth:`served_activation_scale_gaps`: that one asks
        whether a unit HAS a calibrated maximum, which stays true across a
        change of input-global-scale policy; this one asks whether the G that
        maximum now derives is the G the unit's retained render score was
        priced at.  Asked before the first forward for the same reason -- the
        refusal names every affected unit rather than the first one the model
        reaches -- and answered from the cache's own score provenance, never
        from the environment.
        """
        conflicts: set[str] = set()
        if not self._priced_input_global_scales:
            return []
        for name, contract in self._served_measurement_units():
            priced = _activation_max_abs_lookup(
                self._priced_input_global_scales, name)
            max_abs = _activation_max_abs_lookup(self._activation_scales, name)
            if priced is None or max_abs is None or float(max_abs) <= 0.0:
                continue
            applied = contract.input_global_scale_from_max_abs(float(max_abs))
            if float(priced) != float(applied):
                conflicts.add(name)
        return sorted(conflicts)

    def _nvfp4_fused_param_plan(self, plan: _ModulePlan) -> _ParamPlan | None:
        if not _env_truthy("PRISMAQUANT_FUSED_KERNEL_NVFP4"):
            return None
        # When a production cache is active, the fused fast path's
        # `nvfp4_pack_weight` re-computes per-group scales locally and
        # ignores the cache's joint NVFP4 sibling globals — so the
        # packed FP4 codes diverge from what the export would produce.
        # Refuse to use the fast path in that mode unless the user
        # explicitly opts in via PRISMAQUANT_FUSED_KERNEL_OVER_PROD_CACHE.
        if (
            self._production_weight_cache is not None
            and not _env_truthy("PRISMAQUANT_FUSED_KERNEL_OVER_PROD_CACHE")
        ):
            return None
        if not isinstance(plan.module, nn.Linear) or len(plan.params) != 1:
            return None
        param_plan = plan.params[0]
        if param_plan.attr != "weight":
            return None
        if fr.canonical_format_name(param_plan.spec.name) != "NVFP4":
            return None
        act_spec = self._active_activation_spec(plan)
        if act_spec is None or fr.canonical_format_name(act_spec.name) != "NVFP4":
            return None
        return param_plan

    def _try_install_nvfp4_fused_forward(self, plan: _ModulePlan) -> bool:
        param_plan = self._nvfp4_fused_param_plan(plan)
        if param_plan is None:
            return False
        try:
            from prismaquant.kernels.nvfp4_fused import nvfp4_fused_aw_matmul  # noqa: F401
        except Exception:
            return False

        module = plan.module
        original_forward = module.forward

        def _forward(x, *args, **kwargs):
            if args or kwargs or not isinstance(x, torch.Tensor):
                return original_forward(x, *args, **kwargs)
            return self._nvfp4_fused_linear_forward(plan, param_plan, x)

        module.forward = _forward
        self._fused_forward_originals.append((module, original_forward))
        return True

    def _weight_for_reference_forward(
        self,
        plan: _ModulePlan,
        param_plan: _ParamPlan,
    ) -> torch.Tensor:
        param = getattr(plan.module, param_plan.attr)
        if not isinstance(param, torch.nn.Parameter) or param.is_meta:
            return param
        q = None
        if self._frozen_weight_cache is not None:
            cache_key = (id(plan.module), param_plan.attr)
            q = self._frozen_weight_cache.get(cache_key)
            if q is not None:
                self._frozen_weight_cache.move_to_end(cache_key)
        if q is None:
            q = self._quantized_weight_for(plan, param_plan, param_plan.spec)
        if q is None:
            return param
        return q.to(device=param.device, dtype=param.dtype)

    def _reference_linear_forward(
        self,
        plan: _ModulePlan,
        param_plan: _ParamPlan,
        x: torch.Tensor,
    ) -> torch.Tensor:
        act_spec = self._active_activation_spec(plan)
        if act_spec is not None:
            # MED-3: act-clip the input to the calibrated max_abs before
            # per-group RTN.  The dynamic per-group quantizer in
            # `act_spec.activation_quantize_dequantize` would otherwise
            # set its scales from the input's per-group max — outliers
            # then dominate.  Production export does the same clipping
            # as `_resolve_act_clip_quantile`, so this matches what the
            # shipped artifact sees at runtime.  `Q(x/s)*s == Q(x)` for
            # purely dynamic Q, so the previous "pre-scale + post-multiply"
            # formulation was a no-op (codex round-3).
            x = _activation_qdq(
                x, act_spec, self._activation_scales, param_plan.name,
                self._priced_input_global_scales,
            )
        weight = self._weight_for_reference_forward(plan, param_plan)
        return F.linear(x, weight, plan.module.bias)

    def _packed_nvfp4_weight_for(
        self,
        plan: _ModulePlan,
        param_plan: _ParamPlan,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        param = getattr(plan.module, param_plan.attr)
        if not isinstance(param, torch.nn.Parameter) or param.is_meta:
            raise RuntimeError("cannot pack a missing or meta Linear weight")
        cache_key = (
            param_plan.name,
            str(param.device),
            str(param.dtype),
            int(param.data_ptr()),
        )
        packed = self._fused_nvfp4_weight_cache.get(cache_key)
        if packed is None:
            from prismaquant.kernels.nvfp4_fused import nvfp4_pack_weight

            enforce_gpu_memory_budget(
                [self],
                device=param.device if param.device.type == "cuda" else None,
                reason="NVFP4 packed weight cache fill",
            )
            # HIGH: prefer the production cache's GPTQ + scale_sweep
            # weight when present.  Without this, the fused NVFP4 fast
            # path packs the raw BF16 param and bypasses the entire
            # production cache (silently runs RTN-equivalent weights
            # through the kernel).  Strict mode raises on miss so the
            # fast path matches the slow-path miss semantics.
            source = param.detach()
            if self._production_weight_cache is not None:
                w_dq = self._production_weight_cache.get(
                    param_plan.name, "NVFP4",
                )
                if w_dq is not None:
                    source = w_dq.to(
                        device=param.device, dtype=param.dtype,
                    ).contiguous()
                elif _env_truthy(
                    "PRISMAQUANT_STRICT_PRODUCTION_CACHE",
                    default=True,
                ):
                    raise RuntimeError(
                        f"production_weight_cache miss for "
                        f"({param_plan.name!r}, 'NVFP4') on the fused "
                        f"NVFP4 fast path; set "
                        f"PRISMAQUANT_STRICT_PRODUCTION_CACHE=0 to allow "
                        f"raw-weight fallback or rebuild the cache."
                    )
            packed = nvfp4_pack_weight(source)
            self._fused_nvfp4_weight_cache[cache_key] = packed
            self._fused_nvfp4_weight_cache.move_to_end(cache_key)
            enforce_gpu_memory_budget(
                [self],
                device=param.device if param.device.type == "cuda" else None,
                reason="NVFP4 packed weight cache fill",
            )
        else:
            self._fused_nvfp4_weight_cache.move_to_end(cache_key)
        return packed

    def _frozen_weight_cache_max_entries(self) -> int:
        return env_int("PRISMAQUANT_FROZEN_WEIGHT_CACHE_MAX_ENTRIES", 400)

    def _evict_frozen_weight_format_cache_to_limit(self) -> None:
        max_entries = self._frozen_weight_cache_max_entries()
        if max_entries <= 0:
            evicted = len(self._frozen_weight_format_cache)
            self._frozen_weight_format_cache.clear()
            self._frozen_weight_cache_evictions += evicted
            return
        while len(self._frozen_weight_format_cache) > max_entries:
            self._frozen_weight_format_cache.popitem(last=False)
            self._frozen_weight_cache_evictions += 1

    def evict_oldest_for_memory_budget(self) -> bool:
        if self._frozen_weight_format_cache:
            self._frozen_weight_format_cache.popitem(last=False)
            self._frozen_weight_cache_evictions += 1
            return True
        if self._fused_nvfp4_weight_cache:
            self._fused_nvfp4_weight_cache.popitem(last=False)
            return True
        if self._frozen_weight_cache:
            self._frozen_weight_cache.popitem(last=False)
            self._frozen_weight_cache_evictions += 1
            return True
        return False

    def _emit_frozen_weight_cache_evictions(self) -> None:
        if (
            self._frozen_weight_cache_evictions <= 0
            or self._frozen_weight_cache_eviction_reported
        ):
            return
        self._frozen_weight_cache_eviction_reported = True
        print(
            "[frozen-weight-cache] evicted "
            f"{self._frozen_weight_cache_evictions} entries "
            f"(max_entries={self._frozen_weight_cache_max_entries()})",
            file=sys.stderr,
            flush=True,
        )

    def _nvfp4_fused_linear_forward(
        self,
        plan: _ModulePlan,
        param_plan: _ParamPlan,
        x: torch.Tensor,
    ) -> torch.Tensor:
        self._capture(plan, x)
        x_runtime = x
        act_spec = self._active_activation_spec(plan)
        fused_active = (
            fr.canonical_format_name(param_plan.spec.name) == "NVFP4"
            and act_spec is not None
            and fr.canonical_format_name(act_spec.name) == "NVFP4"
            and x_runtime.is_cuda
            and x_runtime.shape[-1] % 16 == 0
        )
        if not fused_active:
            return self._reference_linear_forward(plan, param_plan, x)

        from prismaquant.kernels.nvfp4_fused import nvfp4_fused_aw_matmul

        w_packed, w_scales, w_global_scale = self._packed_nvfp4_weight_for(
            plan, param_plan
        )
        flat_x = x_runtime.reshape(-1, x_runtime.shape[-1])
        # MED-3: act-clip the activation to the calibrated max_abs before
        # the fused kernel's internal per-group RTN.  Same rationale as
        # ``_reference_linear_forward``: pre-scale + post-multiply
        # cancels under dynamic per-group RTN, but clipping forces
        # outliers to the calibrated range so per-group scales are
        # bounded — matching production's act-clip semantics.
        flat_x = _maybe_clip_activations(
            flat_x, self._activation_scales, param_plan.name,
        )
        out = nvfp4_fused_aw_matmul(flat_x, w_packed, w_scales, w_global_scale)
        out = out.reshape(*x_runtime.shape[:-1], plan.module.out_features)
        if plan.module.bias is not None:
            out = out + plan.module.bias.to(device=out.device, dtype=out.dtype)
        return out

    def _restore_plan(self, plan: _ModulePlan) -> None:
        if _env_truthy("PRISMAQUANT_EXTERNAL_WEIGHT_MANAGEMENT", default=False):
            # Mirror of _apply_weight_quant's bypass — WeightSession
            # owns weight transitions, so there's nothing to restore.
            return
        for param, original in reversed(plan.active_originals):
            param.data.copy_(original.to(device=param.device, dtype=param.dtype))
        plan.active_originals.clear()

    def _packed_act_plan(self, plan: _ModulePlan) -> list[_ParamPlan] | None:
        """The per-projection params of a packed-experts plan, or None.

        A packed-experts module owns several 3-D projection parameters and is
        not an ``nn.Linear``, so the module-level pre-hook can only ever see
        ONE of their inputs -- the module input, which is gate_up's. down_proj
        consumes the post-SwiGLU intermediate produced INSIDE the forward, and
        no hook on the module boundary can reach it.

        That is not a cosmetic gap. vLLM's ``CompressedTensorsW4A4Nvfp4MoEMethod``
        registers BOTH ``w13_input_global_scale`` and ``w2_input_global_scale``:
        the served runtime quantizes both activations. A gate that emulates only
        one of them measures a cheaper model than the one that ships, on the
        half of the MoE FLOPs it left alone -- and it is the selecting gate, so
        the error goes straight into which assignment is chosen.
        """
        if not self.include_activation_quant or len(plan.params) < 2:
            return None
        if isinstance(plan.module, nn.Linear):
            return None
        members = [
            p for p in plan.params
            if getattr(p.spec, "act_quant_changes_input", False)
            and getattr(getattr(plan.module, p.attr, None), "ndim", 0) == 3
        ]
        return members or None

    def _install_packed_expert_activation_quant(self, plan: _ModulePlan) -> None:
        """Quantize each expert-slice ``F.linear`` input with ITS OWN spec.

        Same interception the probe uses to capture packed-expert Fisher
        (``sensitivity_probe.install_packed_expert_hooks``): swap ``F.linear``
        for the duration of the experts-module forward and dispatch on whether
        the weight is a dim-0 slice of one of this plan's packed parameters.
        Eval-time only -- no autograd Function, no gradient path.

        Each projection uses its own calibrated activation scale, keyed by its
        own param name. The module-level pre-hook's act-qdq is suppressed for
        these plans (see ``_make_pre_hook``) so gate_up's input is quantized
        exactly once, here, rather than once there and once again inside.
        """
        members = self._packed_act_plan(plan)
        if members is None:
            return
        from prismaquant.sensitivity_probe import _packed_expert_slice_index

        module = plan.module
        original_forward = module.forward
        owner = self

        def _forward(*args, **kwargs):
            targets: dict[int, _ParamPlan] = {}
            params: dict[int, torch.Tensor] = {}
            for member in members:
                param = getattr(module, member.attr, None)
                if not isinstance(param, torch.Tensor) or param.ndim != 3:
                    continue
                targets[id(param)] = member
                params[id(param)] = param
            if not targets:
                return original_forward(*args, **kwargs)
            orig_linear = F.linear

            def _intercepting_linear(input, weight, bias=None):
                base = weight._base if weight._is_view() else weight
                member = targets.get(id(base))
                if member is not None and isinstance(input, torch.Tensor):
                    if _packed_expert_slice_index(
                            weight, params[id(base)]) is not None:
                        input = _activation_qdq(
                            input, member.spec, owner._activation_scales,
                            member.name,
                            owner._priced_input_global_scales)
                return orig_linear(input, weight, bias)

            F.linear = _intercepting_linear
            try:
                return original_forward(*args, **kwargs)
            finally:
                F.linear = orig_linear

        module.forward = _forward
        self._fused_forward_originals.append((module, original_forward))

    def _make_pre_hook(self, plan: _ModulePlan):
        def _pre_hook(_module, args, kwargs):
            where, key, x = _first_tensor_location(args, kwargs)
            if isinstance(x, torch.Tensor):
                self._capture(plan, x)
                member_name = _module_input_member_name(plan, x)
                x_runtime = x
                act_spec = self._active_activation_spec(plan)
                if self._packed_act_plan(plan) is not None:
                    # Handled per projection inside the forward, where
                    # down_proj's input is reachable and each projection gets
                    # its own calibrated scale. Quantizing here too would put
                    # gate_up's input through the quantizer twice.
                    act_spec = None
                if act_spec is not None:
                    # MED-3: act-clip to the calibrated max_abs before the
                    # quantizer, so outliers don't dominate per-group
                    # scales.  See ``_maybe_clip_activations`` for the
                    # math; pre-scale + post-multiply was a no-op (codex
                    # round-3 caught Q(x/s)*s == Q(x)).
                    x_runtime = _activation_qdq(
                        x_runtime, act_spec, self._activation_scales,
                        member_name, self._priced_input_global_scales,
                    )
                if x_runtime is not x:
                    args, kwargs = _replace_tensor_input(
                        args, kwargs, where, key, x_runtime,
                    )
            self._apply_weight_quant(plan)
            return args, kwargs

        return _pre_hook

    def _make_post_hook(self, plan: _ModulePlan):
        def _post_hook(_module, _args, _kwargs, output):
            self._restore_plan(plan)
            return output

        return _post_hook

    def finalize(self) -> dict:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        for name, rows in self._snap_rows.items():
            if rows is None or rows.size(0) == 0:
                continue
            x = rows[:self.input_rows].to(torch.bfloat16).contiguous()
            write_activation_cache_entry(self.cache_dir, name, x)
            written.append(name)
        return {
            "cache_dir": str(self.cache_dir),
            "written": sorted(written),
            "missing": sorted(self.missing),
            "skipped_activation_quant": self.skipped,
        }


def _to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def iter_calibration_forwards(
    calibration_data,
    device: torch.device,
    *,
    microbatch_size: int = 1,
):
    """Yield (args, kwargs) for one forward pass per calibration microbatch.

    ``microbatch_size`` controls how many calibration rows are stacked into
    each forward — default 1 preserves the historical one-sample-at-a-time
    behaviour every existing caller relies on.  Callers that want to amortize
    Python and kernel launch overhead can request a larger microbatch; the
    yielded batch dim becomes ``min(microbatch_size, remaining_rows)``.
    """
    if isinstance(calibration_data, torch.Tensor):
        n = int(calibration_data.size(0))
        m = max(1, int(microbatch_size))
        for i in range(0, n, m):
            yield (calibration_data[i:i + m].to(device),), {}
        return
    if isinstance(calibration_data, Mapping):
        yield (), {k: _to_device(v, device) for k, v in calibration_data.items()}
        return
    for sample in calibration_data:
        if isinstance(sample, torch.Tensor):
            yield (sample.to(device),), {}
        elif isinstance(sample, Mapping):
            yield (), {k: _to_device(v, device) for k, v in sample.items()}
        elif isinstance(sample, tuple):
            yield tuple(_to_device(v, device) for v in sample), {}
        else:
            yield (sample,), {}


@torch.no_grad()
def capture_perturbed_activation_cache(
    model: nn.Module,
    assignment: Mapping[str, str],
    calibration_data,
    cache_dir: str | Path,
    *,
    input_rows: int = 256,
    profile=None,
    cal_hash: str | None = None,
) -> dict:
    """Run calibration forwards and write an ActivationIndex-compatible cache."""
    cal_hash = cal_hash or calibration_data_hash(calibration_data)
    builder = PerturbedActivationCache(
        model,
        assignment,
        cache_dir,
        input_rows=input_rows,
        cal_hash=cal_hash,
        profile=profile,
    )
    device = _model_device(model)
    builder.install()
    try:
        # PRISMAQUANT_L2_CUDA_GRAPHS is intentionally not applied here.
        # These forwards must execute Python hooks on every batch to snapshot
        # perturbed-X activations; CUDA graph replay would skip those hooks and
        # silently under-fill the activation cache.
        for args, kwargs in iter_calibration_forwards(calibration_data, device):
            model(*args, **kwargs)
    finally:
        builder.remove()
    manifest = builder.finalize()
    manifest["calibration_hash"] = cal_hash
    manifest["input_rows"] = int(input_rows)
    with open(Path(cache_dir) / "perturbed_x_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def stage_text_only_under_work_root(model_path: str, work_root: str | Path) -> str:
    """Text-only staging equivalent to sensitivity_probe, but never under /tmp.

    Thin wrapper around `sensitivity_probe._stage_text_only_impl` (issue
    #210: one home for the default strip-key list and the staging steps,
    shared with `sensitivity_probe.stage_text_only`). This name and
    signature stay so no caller moves; only the staging root differs
    (an explicit, caller-owned `work_root`, never /tmp, with no `atexit`
    registration).
    """
    from .sensitivity_probe import _stage_text_only_impl
    return _stage_text_only_impl(model_path, staging_root=work_root)


def load_text_model_under_work_root(
    model_path: str,
    *,
    device: str,
    dtype: torch.dtype,
    work_root: str | Path,
    device_map: str | None = None,
) -> nn.Module:
    from transformers import AutoModelForCausalLM

    staged = stage_text_only_under_work_root(model_path, work_root)
    load_device_map = device_map if device_map is not None else device
    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": load_device_map,
        "low_cpu_mem_usage": False,
        "trust_remote_code": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
    except ValueError as exc:
        if "requires `accelerate`" not in str(exc) and "requires accelerate" not in str(exc):
            raise
        load_kwargs.pop("device_map", None)
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
        model.to(torch.device(device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
