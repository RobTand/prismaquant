"""Checkpoint and streamed-model identity, verification, and reuse."""
from __future__ import annotations

from functools import partial
import json
import os
from pathlib import Path
import re


from prismaquant.dev_mode import dev_mode_enabled, dev_stamp, dev_warning, seal_check
from .digests import DIRECT_ASCII_LAX, DIRECT_UTF8_INDENT2_STRICT, file_sha256hex


STREAMED_MODEL_IDENTITY_SCHEMA = "prismaquant.streamed_model.identity.v1"

STREAMED_MODEL_IDENTITY_CACHE_SCHEMA = (
    "prismaquant.streamed_model.identity_cache.v1"
)

STREAMED_MODEL_PORTABLE_CONTENT_SCHEMA = (
    "prismaquant.streamed_model.portable_content.v1"
)

_STREAMED_MODEL_CONFIG_PROVENANCE_FIELDS = frozenset({
    "_name_or_path",
    "transformers_version",
})

_SOURCE_IDENTITY_BLOCK_BYTES = 16 * 1024 * 1024
_file_sha256 = partial(file_sha256hex, block_size=_SOURCE_IDENTITY_BLOCK_BYTES)


def _source_checkpoint_metadata_sha256(path: Path) -> str:
    """Hash auxiliary bytes without bypassing an active metadata readset."""
    from .staged_whole_file import read_source_metadata_sha256

    return read_source_metadata_sha256(
        path, label="source checkpoint metadata",
        block_size=_SOURCE_IDENTITY_BLOCK_BYTES,
    )


def stat_fingerprint(path: str | Path, observed: os.stat_result) -> dict[str, object]:
    """The six-field stat record of one source shard, from a stat taken by the caller.

    One constructor for every cache that records or compares shard stats (the
    streamed identity cache, the capture owner's adoption, Tessera's digest
    adoption), so the fields and their types cannot drift between them.
    """
    return {
        "path": str(path),
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "size": int(observed.st_size),
        "mtime_ns": int(observed.st_mtime_ns),
        # Unlike mtime, ctime cannot be restored with utime after a same-size
        # rewrite.  Matching all six fields makes a previously computed
        # content SHA safe to reuse without rereading a multi-hundred-GB
        # checkpoint.
        "ctime_ns": int(observed.st_ctime_ns),
    }


def _streamed_identity_stat_fingerprint(path: Path) -> dict[str, object]:
    """Return the mutation-sensitive local cache key for one source shard."""
    return stat_fingerprint(path.resolve(), path.stat())


def _local_checkpoint_shards(
    source_model: str | Path,
) -> tuple[dict[str, str] | None, list[Path] | None]:
    """Resolve the exact safetensors file set consumed by a local checkpoint.

    The streaming model omits auxiliary decoder namespaces it does not execute
    (DSv4's MTP towers are one example), while the exporter copies those
    tensors byte-verbatim.  The Hugging Face index, when present, is the
    authority for complete source-byte coverage.  A single-file checkpoint
    instead uses every tensor name in its validated safetensors header, not
    only ``context.weight_shard``.  Neither path reads tensor payloads here.
    """
    root = Path(source_model)
    if not root.is_dir():
        return None, None
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            from .staged_whole_file import read_source_metadata_text

            payload = json.loads(read_source_metadata_text(
                index_path, label="checkpoint index", encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"streamed model identity cannot read {index_path}"
            ) from exc
        weight_map = payload.get("weight_map") if isinstance(
            payload, dict
        ) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(
                f"streamed model identity requires a non-empty weight_map in "
                f"{index_path}"
            )
        canonical_map: dict[str, str] = {}
        shard_names: set[str] = set()
        for tensor_name, shard_name in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name:
                raise RuntimeError(
                    f"streamed model identity found an invalid tensor name in "
                    f"{index_path}"
                )
            if (
                not isinstance(shard_name, str)
                or not shard_name
                or Path(shard_name).name != shard_name
            ):
                raise RuntimeError(
                    f"streamed model identity found an unsafe shard name "
                    f"{shard_name!r} in {index_path}"
                )
            canonical_map[tensor_name] = shard_name
            shard_names.add(shard_name)
        shard_paths = sorted(
            ((root / name).resolve() for name in shard_names), key=str
        )
        missing = [str(path) for path in shard_paths if not path.is_file()]
        if missing:
            raise RuntimeError(
                "streamed model identity checkpoint index references missing "
                f"shards: {missing[:8]}"
            )
        return dict(sorted(canonical_map.items())), shard_paths
    single = root / "model.safetensors"
    if single.is_file():
        from safetensors import SafetensorError, safe_open

        try:
            with safe_open(str(single), framework="pt", device="cpu") as handle:
                tensor_names = list(handle.keys())
        except (OSError, SafetensorError) as exc:
            raise RuntimeError(
                f"streamed model identity cannot read safetensors header {single}"
            ) from exc
        if not tensor_names or any(
            not isinstance(name, str) or not name for name in tensor_names
        ):
            raise RuntimeError(
                f"streamed model identity requires non-empty tensor names in {single}"
            )
        return {name: single.name for name in sorted(tensor_names)}, [single.resolve()]
    return None, None


SOURCE_CHECKPOINT_IDENTITY_SCHEMA = (
    "prismaquant.source_checkpoint.identity.v1"
)
# The non-shard files the standalone export reads from the checkpoint root:
# `config.json` through `stage_text_only`/`AutoConfig` (streaming_model.py:102
# and the exporter's skeleton build) and the index through
# `_local_checkpoint_shards` and export_native_compressed.py:6092.  A file that
# is absent contributes no row, so one appearing later is a different source.
SOURCE_CHECKPOINT_METADATA_FILES = (
    "config.json",
    "model.safetensors.index.json",
)
# ... plus every `*.py` at the checkpoint root, because a `trust_remote_code`
# checkpoint executes modules from there to build the skeleton (MiniMax-M2
# ships `configuration_minimax_m2.py` and `modeling_minimax_m2.py`;
# DeepSeek-V4 uses the same pattern), discovered per call rather than named
# here.  Every one of
# them is bound, whether or not `auto_map` names it: binding a module nobody
# imports can cost a false refusal, naming only some can cost a false
# admission.
SOURCE_CHECKPOINT_DIGEST_CACHE_SCHEMA = (
    "prismaquant.source_checkpoint.digest_cache.v1"
)


def _read_source_checkpoint_digest_cache(
    cache_path: Path,
) -> dict[str, dict[str, object]]:
    """Digests keyed by the six-field stat fingerprint of the file they cover.

    A corrupt or foreign cache is not an error: it simply reuses nothing.
    Both modes then hash every shard; dev mode also prints a ``[DEV-MODE]``
    line with the byte count first (PQ #1147). The cache can only ever make the identity CHEAPER,
    never different -- the fingerprint it keys on includes ``ctime_ns``, which
    ``utime`` cannot restore after an in-place same-size rewrite.
    """
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != SOURCE_CHECKPOINT_DIGEST_CACHE_SCHEMA
    ):
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return {}
    reusable: dict[str, dict[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        fingerprint = entry.get("fingerprint")
        digest = str(entry.get("sha256", "")).lower()
        if (
            not isinstance(fingerprint, dict)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            continue
        reusable[canonical_fingerprint_key(fingerprint)] = {
            "fingerprint": fingerprint,
            "sha256": digest,
        }
    return reusable


canonical_fingerprint_key = DIRECT_ASCII_LAX.text


#: The stat fields that prove the bytes did not change. ``device`` is
#: deliberately absent: it names the client's mount of a shared export, not
#: the file, and differs across hosts for identical shards (sparky/sparklina
#: NFS mounts of one export). Certified comparisons use the full six-field
#: fingerprint unless the file is on a filesystem whose device number is
#: client-local; dev-portable reuse compares on these five.
_REUSE_FINGERPRINT_FIELDS = ("path", "inode", "size", "mtime_ns", "ctime_ns")
_FINGERPRINT_FIELDS = _REUSE_FINGERPRINT_FIELDS + ("device",)

#: Filesystems whose ``st_dev`` is an anonymous number that the client
#: assigns to each mount, while ``st_ino`` is the server's file id. Measured
#: on the pool's NFS export: one shard read 64 on one mount and 66 on
#: another, with identical inode, size, mtime and ctime on all 120 shards
#: (PQ #1363). Other network filesystems are not listed because nothing has
#: been measured on them.
_CLIENT_LOCAL_DEVICE_FILESYSTEMS = frozenset({"nfs", "nfs4"})
_MOUNTINFO = Path("/proc/self/mountinfo")


def _mount_filesystem_type(path: str | Path) -> str | None:
    """The filesystem type of the mount that holds ``path``, from mountinfo.

    Returns the type of the longest mount point that is a prefix of the
    resolved path, or ``None`` when mountinfo cannot be read. A bind mount
    reports its backing filesystem's type, so a container's view of the pool
    reads the same as the host's.
    """
    try:
        lines = _MOUNTINFO.read_text().splitlines()
    except OSError:
        return None
    target = str(Path(path).resolve())
    best, best_type = "", None
    for line in lines:
        fields = line.split(" ")
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        # mountinfo escapes space, tab, newline and backslash as octal.
        point = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), fields[4])
        inside = target == point or target.startswith(point.rstrip("/") + "/")
        if inside and len(point) >= len(best) and separator + 1 < len(fields):
            best, best_type = point, fields[separator + 1]
    return best_type


def device_number_is_client_local(path: str | Path) -> bool:
    """Whether ``path``'s ``st_dev`` names the client mount, not the file."""
    return _mount_filesystem_type(path) in _CLIENT_LOCAL_DEVICE_FILESYSTEMS


def _well_formed_fingerprint(value: object) -> bool:
    """The exact six-field stat shape with typed values, nothing else.

    Both-missing and extra fields refuse: a missing field is not a match,
    and an extra field could carry mutation signal no comparison reads.
    """
    return (
        isinstance(value, dict)
        and set(value) == set(_FINGERPRINT_FIELDS)
        and isinstance(value.get("path"), str)
        and all(type(value.get(key)) is int
                for key in ("device", "inode", "size", "mtime_ns", "ctime_ns"))
    )


def stat_fingerprint_reuse(live: object, cached: object) -> str | None:
    """Strict own-byte/cache publication reuse; provenance callers may stamp drift.

    Inode, path, size and both times must match. Only the existing client-local
    NFS device rule and explicit dev device-only portability remain exceptions.
    This predicate never rebinds an old digest to a drifted object.
    """
    if not (_well_formed_fingerprint(live) and _well_formed_fingerprint(cached)):
        return None
    if live == cached:
        return 'exact'
    assert isinstance(live, dict) and isinstance(cached, dict)
    if not all(live[key] == cached[key] for key in _REUSE_FINGERPRINT_FIELDS):
        return None
    if device_number_is_client_local(live['path']):
        return 'mount'
    return 'dev' if dev_mode_enabled() else None




def stat_fingerprint_reusable(live: object, cached: object) -> bool:
    """Whether :func:`stat_fingerprint_reuse` admits the recorded digest."""
    return stat_fingerprint_reuse(live, cached) is not None


def portable_fingerprint_key(fingerprint: dict[str, object]) -> str:
    """The digest-cache lookup key ignoring the client device number."""
    return canonical_fingerprint_key(
        {key: value for key, value in fingerprint.items()
         if key != "device"})


def source_identity_hash_threads() -> int:
    """Worker count for hashing uncached source shards.

    ``PRISMAQUANT_SOURCE_HASH_THREADS`` overrides; 1 restores the serial
    loop. The default is the CPU affinity this process was given (PrismaBuild
    assigns it per action), because each worker is one sequential read plus
    one sha256 stream and ``hashlib`` releases the GIL on the 16 MiB updates:
    the pool can use exactly the cores it was admitted to and no more. There is
    no fixed I/O cap -- a slow backing store (a spinning-disk NFS pool, say)
    is the caller's measured fact, so it sets the override rather than every
    caller paying a guessed limit.
    """
    raw = str(os.environ.get("PRISMAQUANT_SOURCE_HASH_THREADS", "")).strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(1, len(os.sched_getaffinity(0)))


def _hash_one_source_shard(path: Path, fingerprint: dict[str, object]) -> str:
    digest = _file_sha256(path)
    if _streamed_identity_stat_fingerprint(path) != fingerprint:
        raise RuntimeError(
            f"source checkpoint shard changed while hashing: {path}"
        )
    return digest


def _hash_source_shards(
    work: list[tuple[Path, dict[str, object]]],
) -> list[str]:
    """Hash each ``(path, fingerprint)`` pair, returning digests in input order.

    The fingerprint was taken before hashing and is re-checked after, per
    shard, exactly as the serial loop does. When several shards fail, the
    error raised is the first one in input order, whatever order the workers
    finished in, so a refusal names the same shard on every run.
    """
    threads = min(source_identity_hash_threads(), len(work))
    if threads <= 1:
        return [_hash_one_source_shard(path, fp) for path, fp in work]
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(
        max_workers=threads, thread_name_prefix="source-hash"
    )
    try:
        futures = [
            pool.submit(_hash_one_source_shard, path, fp) for path, fp in work
        ]
        # `.result()` in submission order: the first failure in `ordered`
        # order is the one raised.
        return [future.result() for future in futures]
    finally:
        # On a refusal, do not start reading shards nobody will use; workers
        # already inside a read finish it (a thread cannot be interrupted).
        pool.shutdown(wait=True, cancel_futures=True)


def _digest_cache_entries(
    reusable: dict[str, dict[str, object]],
    fingerprints: list[dict[str, object]],
) -> list[dict[str, object] | None]:
    """Stored source metadata, including the original fence as well as its digest.

    A dev provenance mismatch reuses the record, never republishes its digest
    under the running object's fingerprint. Both publishing callers retain the
    returned original fence; byte-proof consumers use strict stat reuse.
    """
    dev = dev_mode_enabled()
    key_for = (lambda fp: fp['path']) if dev else portable_fingerprint_key
    index: dict[str, dict[str, object]] = {}
    tainted: set[str] = set()
    for entry in reusable.values():
        stored = entry.get('fingerprint') if isinstance(entry, dict) else None
        if not _well_formed_fingerprint(stored):
            continue
        key = key_for(stored)
        if key in tainted:
            continue
        prior = index.get(key)
        if prior is None:
            index[key] = entry
        elif prior.get('sha256') != entry.get('sha256'):
            tainted.add(key)
            index.pop(key, None)
    found = []
    drifted = []
    for live in fingerprints:
        entry = reusable.get(canonical_fingerprint_key(live))
        if entry is None and _well_formed_fingerprint(live):
            candidate = index.get(key_for(live))
            if candidate is not None:
                stored = candidate['fingerprint']
                if stat_fingerprint_reusable(live, stored):
                    entry = candidate
                elif dev:
                    entry = candidate
                    drifted.append({'recorded': stored, 'running': live})
        found.append(entry)
    if drifted:
        seal_check('source cache stat metadata', [row['recorded'] for row in drifted],
            [row['running'] for row in drifted], where='source digest cache')
    return found



def build_source_checkpoint_identity(
    source_model: str | Path,
    *,
    extra_shard_paths: object = (),
    digest_cache_path: str | Path | None = None,
    source_authentication=None,
) -> dict[str, object]:
    """Content identity of the exact safetensors byte set a run consumes.

    This is the runner-free half of :func:`build_streamed_model_identity`, for
    consumers that hold a checkpoint path rather than a live streaming runner
    -- the standalone compressed-tensors export resume cache is the first
    (PrismaQuant #340). It binds file CONTENT, so a same-size, same-header
    value edit is a different source, and the same bytes under a different
    directory are the same source.

    The weight bytes are not the whole source.
    :data:`SOURCE_CHECKPOINT_METADATA_FILES` names the non-shard files the
    export reads from the checkpoint root, and their sha256 is folded into
    the same ``content_sha256``: ``config.json`` decides the skeleton the
    payloads were quantized against -- the dtype map, ``tie_word_embeddings``,
    any ``quantization_config``, the layer counts -- and
    ``model.safetensors.index.json`` decides which shard each tensor is read
    from.  Every ``*.py`` at the checkpoint root joins them, because a
    ``trust_remote_code`` checkpoint executes modules from there to build the
    skeleton; all of them are bound rather than only those ``auto_map`` names,
    which can refuse falsely but never admit falsely.
    Editing any of them changes what a replayed ``layer_NNN.pt`` means while
    every shard byte stays identical.  They are kilobytes, so they are hashed
    on every call rather than cached.

    ``digest_cache_path`` makes the read happen once per (file, machine): a
    shard whose complete stat fingerprint still matches reuses its recorded
    digest instead of rereading that shard. Discovery, metadata hashing,
    digest-cache JSON handling and identity construction still run. Without
    this cache, every call hashes every shard.

    An explicit qualified original ``source_authentication`` owner supplies
    independently bound expected whole-file descriptors instead. Config/index
    interpretation is authenticated by that owner; each actual tensor delivery
    still verifies its held bytes and reports them separately. This keeps the
    same identity schema/content meaning and accepts no legacy stat cache or
    extra shard roster. It does not qualify original device/capture admission.
    """
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256

    if source_authentication is not None:
        if digest_cache_path is not None or extra_shard_paths:
            raise RuntimeError('original checkpoint identity accepts no stat cache or extra roster')
        from .source_generation import original_checkpoint_description

        descriptor = original_checkpoint_description(source_model, source_authentication)
        shards = [{key: row[key] for key in ('name', 'size', 'sha256')}
                  for row in descriptor['shards']]
        metadata = [row for row in descriptor['metadata']
                    if row['name'] in SOURCE_CHECKPOINT_METADATA_FILES or row['name'].endswith('.py')]
        value = {'schema': SOURCE_CHECKPOINT_IDENTITY_SCHEMA,
                 'shards': shards, 'metadata': sorted(metadata, key=lambda row: row['name'])}
        return {**value, 'content_sha256': canonical_json_sha256(
            value, where='source checkpoint content identity')}
    root = Path(source_model)
    _, indexed_shards = _local_checkpoint_shards(source_model)
    shard_paths = {Path(path).resolve() for path in (indexed_shards or ())}
    for path in extra_shard_paths or ():
        shard_paths.add(Path(path).resolve())
    missing = sorted(str(path) for path in shard_paths if not path.is_file())
    if missing:
        raise RuntimeError(
            f"source checkpoint identity references missing shards: "
            f"{missing[:8]}"
        )
    if not shard_paths:
        raise RuntimeError(
            f"source checkpoint identity found no safetensors shards under "
            f"{root}; refusing to stamp an unidentified source"
        )
    ordered = sorted(shard_paths, key=str)
    # Name each shard by its position INSIDE the checkpoint. A Hugging Face
    # snapshot dir holds symlinks into `blobs/`, so the resolved path is named
    # by an LFS hash; naming shards by that would make the SAME checkpoint
    # reached through a snapshot and through a plain directory two different
    # sources. Recover the in-checkpoint spelling from the directory listing.
    name_by_resolved: dict[str, str] = {}
    if root.is_dir():
        for entry in root.rglob("*.safetensors"):
            try:
                name_by_resolved.setdefault(
                    str(entry.resolve()), str(entry.relative_to(root))
                )
            except (OSError, ValueError):
                continue

    reusable = (
        _read_source_checkpoint_digest_cache(Path(digest_cache_path))
        if digest_cache_path is not None
        and Path(digest_cache_path).is_file()
        else {}
    )

    fingerprints = [_streamed_identity_stat_fingerprint(path) for path in ordered]
    cached_entries = _digest_cache_entries(reusable, fingerprints)
    digests = [str(entry['sha256']) if entry is not None else None for entry in cached_entries]
    recorded_fingerprints = [entry['fingerprint'] if entry is not None and
        not stat_fingerprint_reusable(live, entry['fingerprint']) else live
        for live, entry in zip(fingerprints, cached_entries, strict=True)]
    misses = [index for index, digest in enumerate(digests) if digest is None]
    if misses and dev_mode_enabled():
        # The digests key every cache, so a miss is hashed in both modes
        # (PQ #1147): dev mode only says so, loudly. Campaign rows declare a
        # covering cache and never reach this.
        total = sum(int(fingerprints[index]["size"]) for index in misses)
        where = (f"the declared digest cache {digest_cache_path} does not "
                 "cover them" if digest_cache_path is not None
                 else "no digest cache is declared")
        dev_warning(
            f"source rehash of {total} bytes across {len(misses)} shard(s): "
            f"{where}; hashing them now")
    for index, digest in zip(
        misses,
        _hash_source_shards(
            [(ordered[index], fingerprints[index]) for index in misses]
        ),
    ):
        digests[index] = digest

    entries: list[dict[str, object]] = []
    shards: list[dict[str, object]] = []
    for path, fingerprint, digest in zip(ordered, recorded_fingerprints, digests):
        assert digest is not None
        entries.append({"fingerprint": fingerprint, "sha256": digest})
        # Relocating a checkpoint does not change its bytes, so the identity
        # is never keyed on the absolute path.
        name = name_by_resolved.get(str(path))
        if name is None:
            try:
                name = str(path.relative_to(root.resolve()))
            except ValueError:
                name = path.name
        shards.append({
            "name": name,
            "size": int(fingerprint["size"]),
            "sha256": digest,
        })
    shards.sort(key=lambda row: (str(row["name"]), str(row["sha256"])))

    # The non-shard files the export reads.  Kilobytes each, so no digest
    # cache: hashing them costs less than deciding not to.
    metadata_names = list(SOURCE_CHECKPOINT_METADATA_FILES)
    if root.is_dir():
        # `trust_remote_code` checkpoints build their skeleton from modules at
        # the checkpoint root, so those are read bytes too.  All of them, not
        # only the ones `auto_map` names: over-binding refuses falsely, and
        # under-binding admits falsely.
        metadata_names += sorted(
            path.name for path in root.glob("*.py") if path.is_file()
        )
    metadata: list[dict[str, object]] = []
    for name in metadata_names:
        path = root / name
        if not path.is_file():
            continue
        fingerprint = _streamed_identity_stat_fingerprint(path)
        digest = _source_checkpoint_metadata_sha256(path)
        if _streamed_identity_stat_fingerprint(path) != fingerprint:
            raise RuntimeError(
                f"source checkpoint metadata changed while hashing: {path}"
            )
        metadata.append({
            "name": name,
            "size": int(fingerprint["size"]),
            "sha256": digest,
        })
    metadata.sort(key=lambda row: str(row["name"]))

    identity = {
        "schema": SOURCE_CHECKPOINT_IDENTITY_SCHEMA,
        "shards": shards,
        "metadata": metadata,
        "content_sha256": canonical_json_sha256(
            {
                "schema": SOURCE_CHECKPOINT_IDENTITY_SCHEMA,
                "shards": shards,
                "metadata": metadata,
            },
            where="source checkpoint content identity",
        ),
    }

    if digest_cache_path is not None:
        from prismaquant.cost_stage_checkpoint import atomic_write_bytes

        try:
            atomic_write_bytes(
                Path(digest_cache_path),
                DIRECT_UTF8_INDENT2_STRICT.encoded(
                    {
                        "schema": SOURCE_CHECKPOINT_DIGEST_CACHE_SCHEMA,
                        "entries": entries,
                    }
                ),
            )
        except OSError:
            # The digest cache is an optimization. A read-only or full cache
            # directory costs a re-read next time; it never changes identity.
            pass
    return identity


def _read_streamed_model_identity_cache(
    cache_path: Path,
    *,
    source_model: str,
    raw: bytes | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8")
                            if raw is None else raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"streamed model identity cache {cache_path} is corrupt; "
            "refusing identity reuse"
        ) from exc
    refusal = lambda: RuntimeError(
        f"streamed model identity cache {cache_path} does not bind source {source_model!r}")
    if not isinstance(cached, dict) or cached.get('schema') != STREAMED_MODEL_IDENTITY_CACHE_SCHEMA:
        raise refusal()
    seal_check('source cache path', cached.get('source'), str(source_model),
               where=str(cache_path), refusal=refusal)
    identity = validate_streamed_model_identity(
        cached.get("identity"), where="streamed model identity cache"
    )
    return cached, identity


def _streamed_identity_record(*, config_dict, mapping, shards, checkpoint_weight_map,
                              source_model, resolved_commit):
    """The existing v1 identity serialization, shared by admitted source intake."""
    from .cost_stage_checkpoint import canonical_json, canonical_json_sha256

    value_bearing = {
        "config": canonical_json(config_dict, where="streamed model config"),
        "weight_map": mapping,
        "shards": shards,
    }
    if checkpoint_weight_map is not None:
        value_bearing["checkpoint_weight_map"] = checkpoint_weight_map
    return {
        "schema": STREAMED_MODEL_IDENTITY_SCHEMA,
        "source": str(source_model),
        "resolved_commit": resolved_commit,
        "content_sha256": canonical_json_sha256(
            value_bearing, where="streamed model content identity"),
        **value_bearing,
    }




def _original_streamed_identity(runner, source_model, config_dict, mapping, owner):
    """Existing v1 identity from admitted expected descriptors and owned resolution."""
    from .digests import indent2_json_file_bytes
    from .layer_streaming import construction_multimodal
    from .model_profiles import detect_profile
    from .sensitivity_probe import text_only_stage_config
    from .source_read_plan import live_weight_map
    from .streaming_model import _streaming_auto_config_from_bytes, _skeleton_config_and_class

    context = runner.context
    if context.source_snapshot_only or context.source_scope is not None:
        raise RuntimeError('original identity requires a complete body resolution')
    from .source_generation import original_checkpoint_description

    descriptor = original_checkpoint_description(source_model, owner)
    original = descriptor['config']
    profile = detect_profile(source_model, config=original)
    profile._declare_checkpoint_index(descriptor['index'])
    construction = construction_multimodal(profile, False)
    derived = original if construction else text_only_stage_config(original, profile=profile)
    derived = original if derived is None else derived
    expected_config = _streaming_auto_config_from_bytes(
        source_model, indent2_json_file_bytes(derived), resource_check=owner.resource_check)
    expected_config, _ = _skeleton_config_and_class(
        expected_config, multimodal=construction, log_prefix='[original-identity]')
    if canonical_streamed_model_semantic_config(config_dict) != canonical_streamed_model_semantic_config(
            expected_config.to_dict()):
        raise RuntimeError('original identity resolved config differs from authenticated bootstrap')
    expected_shards, expected_mapping = live_weight_map(
        descriptor['index']['weight_map'], source_model,
        lambda key: profile.checkpoint_to_live_name(key, multimodal=construction))
    if (mapping != dict(sorted(expected_mapping.items()))
            or dict(context.weight_shard) != expected_shards):
        raise RuntimeError('original identity live/config/index/shard roster differs')
    shards = [{key: row[key] for key in ('path', 'size', 'sha256')}
              for row in descriptor['shards']]
    return _streamed_identity_record(
        config_dict=config_dict, mapping=mapping, shards=shards,
        checkpoint_weight_map=descriptor['index']['weight_map'], source_model=source_model,
        resolved_commit=getattr(runner.model.config, '_commit_hash', None))


def build_streamed_model_identity(
    runner: StreamedCausalLM,
    source_model: str,
    *,
    identity_cache_path: str | Path | None = None,
    identity_cache_bytes: bytes | None = None,
    digest_cache_path: str | Path | None = None,
    refuse_uncovered: str | None = None,
) -> dict[str, object]:
    """Hash the complete checkpoint backing a streamed cost run.

    End-to-end KL/adjoint values depend on every body/head/norm weight, so a
    path, index, or target-unit hash is insufficient.  This hashes each unique
    source shard exactly once and folds those digests together with the live
    checkpoint-key map and resolved config.  It is an initialization integrity
    pass, not a residency mechanism; decoder execution still uses the existing
    streaming cache.

    ``identity_cache_bytes`` is a read-only cache the caller already read
    and bound by digest (the Stage B head slice, PQ #1010): it is parsed and
    reused exactly like a cache file, and nothing is ever written back.

    ``digest_cache_path`` is a runner-free source digest cache
    (:func:`build_source_checkpoint_identity`'s format), made by a CPU-only
    quantum upstream of this pass (PQ #1374). A shard the identity cache
    does not cover reuses its digest when the same fingerprint predicate
    admits it. ``refuse_uncovered`` turns any shard still uncovered after
    both caches into a refusal before a byte is hashed; its text names the
    quantum that should have produced the proof.

    With an explicitly selected qualified original owner on the context,
    owned config/index resolution and the complete live/checkpoint shard roster
    must agree with its independently authenticated publisher descriptors.
    The existing v1 serializer binds those expected whole-file facts without
    pool discovery, stat/cache reuse or redundant bulk hashing. Actual decoder
    deliveries remain independently verified and separately receipted; legacy
    identity/cache inputs refuse and original CUDA admission remains closed.
    """
    from prismaquant.cost_stage_checkpoint import (
        canonical_json,
        canonical_json_sha256,
    )

    if identity_cache_bytes is not None and identity_cache_path is not None:
        raise ValueError("a streamed identity cache is either a path or "
                         "declared bytes, never both")

    config = getattr(runner.model, "config", None)
    config_dict = config.to_dict() if hasattr(config, "to_dict") else {}
    if "_name_or_path" in config_dict:
        # The multi-shard staging workaround loads the model from a fresh
        # mkdtemp copy each launch, so the loaded config's _name_or_path is
        # a random per-launch path; keying the identity on it makes every
        # relaunch refuse its own journal ("identity mismatch ... refusing
        # reuse or recompute"). The shard digests below are the real
        # identity — pin the path field to the caller's canonical source.
        config_dict["_name_or_path"] = str(source_model)
    mapping = {
        str(live): str(checkpoint)
        for live, checkpoint in sorted(runner.context.weight_ckpt.items())
    }
    owner = getattr(runner.context, 'source_authentication', None)
    if getattr(owner, 'is_qualified_original_material', False):
        if any(value is not None for value in (identity_cache_path, identity_cache_bytes, digest_cache_path)):
            raise RuntimeError('original model identity accepts no legacy stat identity/cache inputs')
        return _original_streamed_identity(runner, str(source_model), config_dict, mapping, owner)
    runner_shard_paths = {
        Path(path).resolve()
        for path in runner.context.weight_shard.values()
    }
    checkpoint_weight_map, checkpoint_shard_paths = (
        _local_checkpoint_shards(source_model)
    )
    shard_paths = sorted(
        runner_shard_paths | set(checkpoint_shard_paths or ()), key=str
    )
    if not shard_paths:
        raise RuntimeError(
            "streamed model identity found no source checkpoint shards"
        )
    fingerprints = [
        _streamed_identity_stat_fingerprint(path) for path in shard_paths
    ]
    cache_path = Path(identity_cache_path) if identity_cache_path else None
    cached: dict[str, object] | None = None
    cached_identity: dict[str, object] | None = None
    have_cache = identity_cache_bytes is not None or (
        cache_path is not None and cache_path.is_file())
    if have_cache:
        cached, cached_identity = _read_streamed_model_identity_cache(
            cache_path if cache_path is not None
            else Path("<declared identity cache>"),
            source_model=str(source_model), raw=identity_cache_bytes,
        )
        stored = cached.get("fingerprints")
        reuse = (
            [stat_fingerprint_reuse(live, old)
             for live, old in zip(fingerprints, stored, strict=True)]
            if isinstance(stored, list) and len(stored) == len(fingerprints)
            else [None]
        )
        reusable = None not in reuse
        dev = dev_mode_enabled()
        metadata_reusable = (dev and isinstance(stored, list)
            and len(stored) == len(fingerprints)
            and all(_well_formed_fingerprint(row) for row in stored)
            and {row['path'] for row in stored} == {row['path'] for row in cached_identity['shards']}
            and {Path(row['path']).name for row in cached_identity['shards']}
                == {path.name for path in shard_paths})
        if reusable or metadata_reusable:
            running_raw = canonical_json(config_dict, where='streamed model config')
            raw_matches = cached_identity.get('config') == running_raw
            semantic_matches = (canonical_streamed_model_semantic_config(cached_identity.get('config'))
                                == canonical_streamed_model_semantic_config(config_dict))
            if ((raw_matches or dev and semantic_matches)
                    and cached_identity.get('weight_map') == mapping
                    and cached_identity.get('checkpoint_weight_map') == checkpoint_weight_map):
                seal_check('source stat metadata', stored, fingerprints,
                    where='streamed model identity', same=(stored == fingerprints) if dev else reusable)
                seal_check('source config provenance', cached_identity.get('config'), running_raw,
                           where='streamed model identity')
                return {**cached_identity, **dev_stamp(timestamped=False)} if dev else cached_identity

    # A schema-valid old cache may cover only the executable decoder shards.
    # Certified reuse requires the original stat match. Dev reuse retains
    # recorded digests across metadata drift and hashes only genuinely new
    # source shards not covered by either existing cache.
    reusable_sha: dict[str, str] = {}
    mutated_paths: list[str] = []
    recorded_by_path = {str(fp['path']): fp for fp in fingerprints}
    stat_drift = []
    if cached is not None and cached_identity is not None:
        cached_fingerprints = cached.get("fingerprints")
        cached_shards = cached_identity.get("shards")
        if isinstance(cached_fingerprints, list) and isinstance(
            cached_shards, list
        ):
            cached_fp_by_path = {
                str(row.get("path")): row
                for row in cached_fingerprints
                if isinstance(row, dict) and isinstance(row.get("path"), str)
            }
            cached_shard_by_path = {
                str(row.get("path")): row
                for row in cached_shards
                if isinstance(row, dict) and isinstance(row.get("path"), str)
            }
            for fingerprint in fingerprints:
                path_key = str(fingerprint["path"])
                prior_fp = cached_fp_by_path.get(path_key)
                prior_shard = cached_shard_by_path.get(path_key)
                if prior_fp is None or prior_shard is None:
                    continue
                reuse = stat_fingerprint_reuse(fingerprint, prior_fp)
                if reuse is None and not (dev_mode_enabled() and _well_formed_fingerprint(prior_fp)):
                    mutated_paths.append(path_key)
                    continue
                if (isinstance(prior_shard, dict)
                        and prior_shard.get('size') == prior_fp.get('size')
                        and re.fullmatch(r'[0-9a-f]{64}', str(prior_shard.get('sha256', '')).lower())):
                    reusable_sha[path_key] = str(prior_shard['sha256']).lower()
                    if reuse is None:
                        recorded_by_path[path_key] = prior_fp
                        stat_drift.append({'recorded': prior_fp, 'running': fingerprint})
        if stat_drift:
            seal_check('source stat metadata', [row['recorded'] for row in stat_drift],
                [row['running'] for row in stat_drift], where='partial streamed model identity')
    seeded = 0
    if digest_cache_path is not None:
        pending = [fingerprint for fingerprint in fingerprints
                   if str(fingerprint["path"]) not in reusable_sha]
        seed = _read_source_checkpoint_digest_cache(Path(digest_cache_path))
        for fingerprint, entry in zip(
                pending, _digest_cache_entries(seed, pending), strict=True):
            if entry is not None:
                path_key = str(fingerprint['path'])
                reusable_sha[path_key] = str(entry['sha256'])
                if not stat_fingerprint_reusable(fingerprint, entry['fingerprint']):
                    recorded_by_path[path_key] = entry['fingerprint']
                seeded += 1
    uncovered = [
        (str(fingerprint["path"]), int(fingerprint["size"]))
        for fingerprint in fingerprints
        if str(fingerprint["path"]) not in reusable_sha
    ]
    if uncovered and refuse_uncovered is not None:
        raise RuntimeError(
            f"source identity would hash {sum(size for _, size in uncovered)} "
            f"bytes across {len(uncovered)} shard(s) (first: "
            f"{uncovered[0][0]}) inside this pass: neither the identity cache "
            f"{cache_path} nor the digest cache {digest_cache_path} covers "
            f"them; {refuse_uncovered}")
    if uncovered and dev_mode_enabled():
        # The digests key every cache, so uncovered shards are hashed below in
        # both modes (PQ #1147): dev mode only says so, loudly.
        if not have_cache and not seeded:
            total_live = sum(
                int(fingerprint["size"]) for fingerprint in fingerprints)
            where = (f"at the declared {cache_path}" if cache_path is not None
                     else "with none declared")
            dev_warning(
                f"source rehash of {total_live} bytes: no usable identity "
                f"cache {where}; hashing every shard now")
        else:
            total = sum(size for _, size in uncovered)
            new_paths = [path for path, _ in uncovered
                         if path not in mutated_paths]
            dev_warning(
                f"source rehash of {total} bytes across {len(uncovered)} "
                f"shard(s) ({len(mutated_paths)} mutated, {len(new_paths)} "
                f"new; first: {uncovered[0][0]}): the declared cache "
                f"{cache_path} does not cover them; hashing them now")

    # Uncovered shards go through the one source hasher, in parallel over the
    # admitted CPUs. This loop used to hash them one at a time in the main
    # thread: 253 MB/s over NFS, about 42 min for GLM-5.3 (PQ #1363).
    misses = [(path, fingerprint)
              for path, fingerprint in zip(shard_paths, fingerprints, strict=True)
              if str(path.resolve()) not in reusable_sha]
    for (path, _), digest in zip(misses, _hash_source_shards(misses), strict=True):
        reusable_sha[str(path.resolve())] = digest
    shards: list[dict[str, object]] = []
    for path in shard_paths:
        path_key = str(path.resolve())
        fingerprint = recorded_by_path[path_key]
        digest = reusable_sha[path_key]
        shards.append({
            "path": path_key,
            "size": int(fingerprint["size"]),
            "sha256": digest,
        })
    identity = _streamed_identity_record(
        config_dict=config_dict, mapping=mapping, shards=shards,
        checkpoint_weight_map=checkpoint_weight_map, source_model=source_model,
        resolved_commit=getattr(config, "_commit_hash", None))
    if cache_path is not None:
        from prismaquant.cost_stage_checkpoint import atomic_write_bytes

        atomic_write_bytes(
            cache_path,
            DIRECT_UTF8_INDENT2_STRICT.encoded(
                {
                    "schema": STREAMED_MODEL_IDENTITY_CACHE_SCHEMA,
                    "source": str(source_model),
                    "fingerprints": [recorded_by_path[str(path.resolve())] for path in shard_paths],
                    "identity": identity,
                }
            ),
        )
    return {**identity, **dev_stamp(timestamped=False)} if dev_mode_enabled() else identity


def validate_streamed_model_identity(
    identity: object, *, where: str
) -> dict[str, object]:
    """Require a value-bearing full-checkpoint identity, never a name stamp."""
    from collections.abc import Mapping
    from prismaquant.cost_stage_checkpoint import (
        canonical_json,
        canonical_json_sha256,
    )

    if not isinstance(identity, Mapping):
        raise RuntimeError(
            f"{where} requires a full streamed model identity object"
        )
    if identity.get("schema") != STREAMED_MODEL_IDENTITY_SCHEMA:
        raise RuntimeError(
            f"{where} requires model identity schema "
            f"{STREAMED_MODEL_IDENTITY_SCHEMA!r}"
        )
    digest = str(identity.get("content_sha256", "")).lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RuntimeError(
            f"{where} requires exact model content_sha256"
        )
    shards = identity.get("shards")
    if not isinstance(shards, list) or not shards:
        raise RuntimeError(f"{where} requires source shard content identities")
    for index, shard in enumerate(shards):
        if not isinstance(shard, Mapping):
            raise RuntimeError(f"{where} model shard {index} is malformed")
        shard_digest = str(shard.get("sha256", "")).lower()
        if re.fullmatch(r"[0-9a-f]{64}", shard_digest) is None:
            raise RuntimeError(
                f"{where} model shard {index} lacks content sha256"
            )
    canonical = canonical_json(identity, where=f"{where} model identity")
    value_bearing = {
        "config": canonical.get("config"),
        "weight_map": canonical.get("weight_map"),
        "shards": canonical.get("shards"),
    }
    if "checkpoint_weight_map" in canonical:
        checkpoint_weight_map = canonical.get("checkpoint_weight_map")
        if not isinstance(checkpoint_weight_map, dict) or not all(
            isinstance(name, str)
            and name
            and isinstance(shard, str)
            and shard
            for name, shard in checkpoint_weight_map.items()
        ):
            raise RuntimeError(
                f"{where} model checkpoint_weight_map is malformed"
            )
        value_bearing["checkpoint_weight_map"] = checkpoint_weight_map
    expected = canonical_json_sha256(
        value_bearing, where=f"{where} model content identity"
    )
    if digest != expected:
        raise RuntimeError(
            f"{where} model content_sha256 does not match its source shard "
            "identity"
        )
    return canonical


def canonical_streamed_model_semantic_config(
    config: object,
    *,
    where: str = "streamed model config",
) -> dict[str, object]:
    """Return config semantics without host/runtime provenance fields.

    ``PretrainedConfig.to_dict()`` records the path it was loaded from and the
    installed Transformers version.  A text-only staging directory therefore
    makes two executions of the same checkpoint byte-distinct.  Those values
    remain in the v1 host-local identity for backward compatibility, but they
    cannot participate in a cross-host content join.  Strip them recursively
    so composed configs cannot reintroduce the same provenance below the top
    level.
    """
    from prismaquant.cost_stage_checkpoint import canonical_json

    canonical = canonical_json(config, where=where)
    if not isinstance(canonical, dict):
        raise RuntimeError(f"{where} must be a JSON mapping")

    def _strip(value: object) -> object:
        if isinstance(value, dict):
            return {
                str(key): _strip(item)
                for key, item in value.items()
                if key not in _STREAMED_MODEL_CONFIG_PROVENANCE_FIELDS
            }
        if isinstance(value, list):
            return [_strip(item) for item in value]
        return value

    stripped = _strip(canonical)
    assert isinstance(stripped, dict)
    return stripped


def portable_streamed_model_content_identity(
    identity: object,
    *,
    where: str = "streamed model portable content identity",
) -> dict[str, object]:
    """Derive one path-neutral digest from a complete v1 local identity.

    The serialized v1 ``content_sha256`` intentionally remains unchanged: it
    binds absolute shard paths and the exact runtime config stored in an
    identity cache.  This additive projection is derivable from old caches and
    is the value suitable for comparing independently built caches on separate
    hosts.  Local cache validation must still happen before callers use it.
    """
    from collections.abc import Mapping
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256

    canonical = validate_streamed_model_identity(identity, where=where)
    config = canonical_streamed_model_semantic_config(
        canonical.get("config"), where=f"{where} config",
    )

    def _string_map(value: object, *, field: str) -> dict[str, str]:
        if not isinstance(value, Mapping) or not value:
            raise RuntimeError(f"{where} requires nonempty {field}")
        result: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            if (
                not isinstance(raw_key, str)
                or not raw_key
                or not isinstance(raw_value, str)
                or not raw_value
            ):
                raise RuntimeError(f"{where} {field} is malformed")
            result[raw_key] = raw_value
        return dict(sorted(result.items()))

    weight_map = _string_map(canonical.get("weight_map"), field="weight_map")
    checkpoint_weight_map = _string_map(
        canonical.get("checkpoint_weight_map"),
        field="checkpoint_weight_map",
    )
    raw_shards = canonical.get("shards")
    assert isinstance(raw_shards, list)  # validated above
    shards: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for index, raw in enumerate(raw_shards):
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"{where} shard {index} is malformed")
        path = raw.get("path")
        size = raw.get("size")
        sha256 = str(raw.get("sha256", "")).lower()
        if not isinstance(path, str) or not path:
            raise RuntimeError(f"{where} shard {index} has no path")
        name = Path(path).name
        if (
            not name
            or name in seen_names
            or type(size) is not int
            or int(size) < 1
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        ):
            raise RuntimeError(
                f"{where} shard {index} has no unique basename/size/content"
            )
        seen_names.add(name)
        shards.append({"name": name, "size": int(size), "sha256": sha256})
    shards.sort(key=lambda row: str(row["name"]))

    body = {
        "schema": STREAMED_MODEL_PORTABLE_CONTENT_SCHEMA,
        "source_identity_schema": STREAMED_MODEL_IDENTITY_SCHEMA,
        "config": config,
        "weight_map": weight_map,
        "checkpoint_weight_map": checkpoint_weight_map,
        "shards": shards,
    }
    return {
        "schema": STREAMED_MODEL_PORTABLE_CONTENT_SCHEMA,
        "portable_content_sha256": canonical_json_sha256(
            body, where=where,
        ),
        "checkpoint_shards": len(shards),
        "checkpoint_tensors": len(checkpoint_weight_map),
    }


def compact_streamed_model_identity(
    identity: object,
    *,
    where: str = "streamed model identity",
) -> dict[str, object]:
    """Return the compact, value-bearing identity stored in an artifact.

    The full identity can be several MiB because it carries the complete
    tensor-to-shard map and one SHA-256 per source shard.  ``content_sha256``
    already binds those fields; the compact form retains coverage counts so a
    partial checkpoint identity cannot be mistaken for the full source.
    """

    canonical = validate_streamed_model_identity(identity, where=where)
    shards = canonical.get("shards")
    checkpoint_weight_map = canonical.get("checkpoint_weight_map")
    if not isinstance(shards, list) or not isinstance(
        checkpoint_weight_map, dict
    ):
        raise RuntimeError(
            f"{where} does not attest a complete indexed checkpoint"
        )
    return {
        "schema": canonical.get("schema"),
        "content_sha256": canonical.get("content_sha256"),
        "resolved_commit": canonical.get("resolved_commit"),
        "checkpoint_shards": len(shards),
        "checkpoint_tensors": len(checkpoint_weight_map),
    }


def live_streaming_runner_config(source_model: str | Path) -> dict[str, object]:
    """Derive the live checkpoint config exactly as the streaming runner does.

    Profile-gated staging, offline ``AutoConfig`` load, then the real meta
    skeleton: the same three calls ``_build_streaming_context`` makes before
    materializing weights (``prismaquant/streaming_model.py``).  Sharing that
    derivation -- rather than normalizing after the fact -- is what keeps
    this validator bound to the config the runner actually runs:

    - a multimodal-skeleton family (``glm5_next``) runs the umbrella config,
      so a hardcoded text-only derivation can never agree with its cache;
    - the model constructor applies config defaults (measured: nested
      ``text_config.dtype`` resolves ``"bfloat16"`` at load and ``null``
      after construction), so stopping at ``AutoConfig`` also disagrees.

    Meta tensors only: no weight payload is read, no device is touched, and
    ``attn_implementation`` is left at the context default (measured to not
    affect the derived config).  Raises on any failure; callers report it
    fail-closed.
    """
    from prismaquant.model_profiles import detect_profile
    from prismaquant.sensitivity_probe import stage_multimodal, stage_text_only
    from prismaquant.streaming_model import (
        build_streaming_skeleton, load_streaming_auto_config,
    )

    source = str(source_model)
    profile = detect_profile(source)
    if profile.requires_multimodal_skeleton():
        staged = stage_multimodal(source)
        multimodal = True
    else:
        staged = stage_text_only(source)
        multimodal = False
    config = load_streaming_auto_config(source, staged, local_files_only=True)
    skeleton = build_streaming_skeleton(config, multimodal=multimodal)
    config_dict = skeleton.config.to_dict()
    if not isinstance(config_dict, dict):
        raise TypeError("streaming runner config is not a mapping")
    return config_dict


def validate_cached_streamed_model_identity(
    source_model: str | Path,
    identity_cache_path: str | Path,
    *,
    require_complete_checkpoint: bool = True,
) -> dict[str, object]:
    """Validate a cached full-checkpoint identity without rereading weights.

    Exact per-shard SHA-256 values remain valid only while every mutation-
    sensitive stat fingerprint matches.  For a local indexed checkpoint the
    validator also requires coverage of the complete index shard set (not just
    the decoder shards loaded by a calibration runner) and binds the complete
    tensor-to-shard map.  This is the cheap, fail-closed handoff used before a
    large streaming export.
    """
    source = str(source_model)
    cache_path = Path(identity_cache_path)
    cached, identity = _read_streamed_model_identity_cache(
        cache_path, source_model=source
    )
    fingerprints = cached.get("fingerprints")
    if not isinstance(fingerprints, list) or not fingerprints:
        raise RuntimeError(
            f"streamed model identity cache {cache_path} has no fingerprints"
        )
    fingerprint_by_path: dict[str, dict[str, object]] = {}
    for index, row in enumerate(fingerprints):
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise RuntimeError(
                f"streamed model identity cache fingerprint {index} is malformed"
            )
        path = str(Path(row["path"]).resolve())
        if path in fingerprint_by_path:
            raise RuntimeError(
                f"streamed model identity cache repeats shard path {path}"
            )
        fingerprint_by_path[path] = row

    identity_shards = identity.get("shards")
    assert isinstance(identity_shards, list)  # validated above
    shard_by_path: dict[str, dict[str, object]] = {}
    for index, row in enumerate(identity_shards):
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise RuntimeError(
                f"streamed model identity shard {index} has no exact path"
            )
        path = str(Path(row["path"]).resolve())
        if path in shard_by_path:
            raise RuntimeError(
                f"streamed model identity repeats shard path {path}"
            )
        shard_by_path[path] = row
    if set(shard_by_path) != set(fingerprint_by_path):
        raise RuntimeError(
            "streamed model identity cache fingerprint coverage differs from "
            "its value-bearing shard identity"
        )

    checkpoint_weight_map, checkpoint_paths = _local_checkpoint_shards(source)
    if require_complete_checkpoint:
        if checkpoint_paths is None:
            raise RuntimeError(
                "complete streamed model identity validation requires a local "
                "safetensors checkpoint"
            )
        expected_paths = {str(path.resolve()) for path in checkpoint_paths}
        if set(shard_by_path) != expected_paths:
            missing = sorted(expected_paths - set(shard_by_path))
            extra = sorted(set(shard_by_path) - expected_paths)
            raise RuntimeError(
                "streamed model identity does not cover the complete source "
                f"checkpoint: missing={missing[:8]}, extra={extra[:8]}"
            )
        if checkpoint_weight_map is not None and identity.get(
            "checkpoint_weight_map"
        ) != checkpoint_weight_map:
            raise RuntimeError(
                "streamed model identity tensor-to-shard map differs from the "
                "current source checkpoint index"
            )

    # The shard/index fingerprints above do not cover config.json.  Recreate
    # the live Transformers config through the same derivation the streaming
    # runner uses (profile-gated staging plus the meta skeleton, which applies
    # constructor defaults) and compare its semantic JSON to the config carried
    # by the cached content identity.  `_name_or_path` is host-local provenance
    # and `transformers_version` belongs to the separately pinned runtime image;
    # neither is model semantics.  All other fields must agree exactly, so a
    # same-shape change such as rope scaling cannot reuse old source hashes.
    try:
        config_path = Path(source) / "config.json"
        config_before = _streamed_identity_stat_fingerprint(config_path)
        live_config = canonical_streamed_model_semantic_config(
            live_streaming_runner_config(source),
            where="live streamed model config",
        )
        cached_config = canonical_streamed_model_semantic_config(
            identity.get("config"), where="cached streamed model config",
        )
        if not isinstance(live_config, dict) or not isinstance(
            cached_config, dict
        ):
            raise TypeError("streamed model config is not a mapping")
        config_after = _streamed_identity_stat_fingerprint(config_path)
    except Exception as exc:
        raise RuntimeError(
            "streamed model identity cannot validate the live source config"
        ) from exc
    # The config bytes were just read to derive the live semantic identity; a
    # change during that read is a torn snapshot (partial data), not a seal.
    if config_before != config_after:
        raise RuntimeError(
            "streamed model identity source config changed while validating"
        )
    if live_config != cached_config:
        changed = sorted(
            key for key in set(live_config) | set(cached_config)
            if live_config.get(key) != cached_config.get(key)
        )
        raise RuntimeError(
            "streamed model identity live config differs from its cached "
            f"content identity: changed={changed[:12]}"
        )

    for path_key, expected in fingerprint_by_path.items():
        path = Path(path_key)
        if not path.is_file():
            raise RuntimeError(
                f"streamed model identity source shard is missing: {path}"
            )
        observed = _streamed_identity_stat_fingerprint(path)
        shard = shard_by_path[path_key]
        # Internal recorded byte length is data consistency, not run provenance.
        if shard.get('size') != expected.get('size'):
            raise RuntimeError(f'streamed model identity shard size disagrees for {path}')
        reuse = stat_fingerprint_reuse(observed, expected)
        seal_check('source shard stat metadata', expected, observed, where=str(path),
            same=reuse is not None,
            refusal=lambda: RuntimeError('streamed model identity source shard stat drifted; refusing '
                                         f'cached content SHA for {path}'))
    return identity
