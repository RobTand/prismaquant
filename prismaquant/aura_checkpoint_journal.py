"""Durable AURA checkpoint schema, identity, unit state and journal custody.

The numerical cost passes supply measurement inputs and unit states. This
owner authenticates persisted bytes and keeps resume and publication rules
in one place. Existing v1 checkpoint encodings remain unchanged.
"""
from __future__ import annotations

import io
import json
import os
import pickle
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from . import aura_provenance
from .cost_stage_checkpoint import atomic_write_bytes, unit_path
from .digests import (DIRECT_UTF8_INDENT2_STRICT, DIRECT_UTF8_STRICT,
                      bytes_sha256hex, canonical_json)
from .perturbed_x_cache import calibration_data_hash

AURA_CHECKPOINT_IDENTITY_SCHEMA = "prismaquant.aura_checkpoint.identity.v1"
AURA_CHECKPOINT_MANIFEST_SCHEMA = "prismaquant.aura_checkpoint.manifest.v1"
AURA_CHECKPOINT_UNIT_SCHEMA = "prismaquant.aura_checkpoint.unit.v1"

_canonical_json = canonical_json


def _canonical_json_sha256(value: object, *, where: str) -> str:
    # Not ``digests.canonical_json_sha256``: that one wraps an encode failure
    # (a lone surrogate) in ``ValueError``; this site has always let the
    # ``UnicodeEncodeError`` through. Same bytes on every input it accepts.
    return DIRECT_UTF8_STRICT.sha256(_canonical_json(value, where=where))




def _checkpoint_identity_mismatch(
    *,
    field: str,
    stored: object,
    expected: object,
) -> RuntimeError:
    from prismaquant.production_weight_cache import (
        _production_cache_recent_writes,
        identity_value_for_error,
    )

    detail = ""
    if field == "producer_source_sha256":
        # The two digests are opaque; name the package inputs themselves so
        # the writer is identifiable while its mtime is still fresh (#2218).
        detail = f"; {_production_cache_recent_writes()}"
    return RuntimeError(
        f"AURA checkpoint identity mismatch at {field}: "
        f"stored={identity_value_for_error(stored)} "
        f"current={identity_value_for_error(expected)}{detail}; refusing "
        "reuse or recompute"
    )


def _raise_checkpoint_identity_mismatch(
    *,
    field: str,
    stored: object,
    expected: object,
) -> None:
    raise _checkpoint_identity_mismatch(field=field, stored=stored, expected=expected)


def _write_aura_checkpoint_manifest(
    checkpoint_dir: Path,
    identity: Mapping[str, object],
    names: Sequence[str],
) -> str:
    identity_sha256 = _canonical_json_sha256(
        identity,
        where="AURA checkpoint identity",
    )
    from prismaquant.production_weight_cache import (
        _production_cache_source_profile,
    )

    manifest_source_sha256, manifest_source_files = (
        _production_cache_source_profile())
    manifest = {
        "schema": AURA_CHECKPOINT_MANIFEST_SCHEMA,
        "identity_sha256": identity_sha256,
        "identity": dict(identity),
        "units": [
            {
                "qname": str(name),
                "file": str(
                    unit_path(checkpoint_dir, name).relative_to(
                        checkpoint_dir
                    )
                ),
            }
            for name in names
        ],
        # Diagnostic only (#2218): one pass over the producer tree at manifest
        # write -- its aggregate digest and per-file digests -- so a later
        # producer_source_sha256 mismatch can name the first file that moved,
        # or say when the tree changed between the identity's digest and this
        # write. No gate reads these; the seal stays the identity's aggregate.
        "producer_source_sha256": manifest_source_sha256,
        "producer_source_files_sha256": manifest_source_files,
    }
    encoded = DIRECT_UTF8_INDENT2_STRICT.encoded(manifest)
    atomic_write_bytes(checkpoint_dir / "manifest.json", encoded)
    return identity_sha256


def _dev_archive_checkpoint_lineage(root: Path, reason: str) -> Path:
    """Rename a checkpoint lineage aside and start a fresh one.

    The house pattern used manually on 2026-09-19 (Rob's dev-mode decision):
    the lineage is archived whole -- renamed with a ``.dev-archived-<iso>``
    suffix, never deleted, never silently reused -- and a fresh lineage
    starts in its place. Since PQ #1147 only unit checkpoints that exist
    without a manifest come here: they have no recorded identity to be
    reused under. Certified mode never calls this; it refuses reuse AND
    recompute.
    """
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = root.with_name(f"{root.name}.dev-archived-{stamp}")
    suffix = 0
    while archived.exists():  # two archives inside one second
        suffix += 1
        archived = root.with_name(f"{root.name}.dev-archived-{stamp}-{suffix}")
    root.rename(archived)
    root.mkdir(parents=True, exist_ok=True)
    from prismaquant.dev_mode import dev_warning

    dev_warning(
        f"archived AURA checkpoint lineage to {archived.name} and started a "
        f"fresh one (dev mode): {reason}")
    return archived


def _name_first_differing_source_file(
    refusal: RuntimeError,
    manifest: Mapping[str, object],
    stored_digest: object,
    expected_digest: object,
) -> RuntimeError:
    """Name the first moved producer file in a source-digest refusal (#2218).

    The manifest's write pass -- one aggregate plus per-file digests, recorded
    with it -- against one pass over the executing tree: a recurrence of the
    xdist producer_source_sha256 flake is then diagnosable from the refusal
    alone. When a file moved, the refusal names the first differing relative
    path with both of its digests; when the per-file maps agree but the
    aggregates differ, the tree changed between the identity's digest and the
    manifest write. Best effort: a manifest written before the map existed, or
    an unreadable executing tree, leaves ``refusal`` unchanged.
    """
    stored_files = manifest.get("producer_source_files_sha256")
    if not isinstance(stored_files, Mapping):
        return refusal
    try:
        from prismaquant.production_weight_cache import (
            _production_cache_source_profile,
        )

        _current_sha256, current_files = _production_cache_source_profile()
    except (OSError, RuntimeError):
        return refusal
    for relative in sorted(set(stored_files) | set(current_files)):
        stored_file_digest = stored_files.get(relative)
        current_file_digest = current_files.get(relative)
        if stored_file_digest != current_file_digest:
            return RuntimeError(
                f"{refusal}; first differing source file {relative!r}: "
                f"stored={stored_file_digest if isinstance(stored_file_digest, str) else '<missing>'} "
                f"current={current_file_digest if isinstance(current_file_digest, str) else '<missing>'}"
            )
    if stored_digest != expected_digest:
        manifest_sha256 = manifest.get("producer_source_sha256")
        return RuntimeError(
            f"{refusal}; the producer tree changed between identity and "
            "manifest write (per-file digests all match; manifest pass "
            f"sha256={manifest_sha256 if isinstance(manifest_sha256, str) else '<unrecorded>'})"
        )
    return refusal


def _load_aura_checkpoint_manifest(
    checkpoint_dir: Path,
    expected_identity: Mapping[str, object],
) -> str:
    path = checkpoint_dir / "manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except Exception as exc:
        _raise_checkpoint_identity_mismatch(
            field="manifest_json",
            stored="<invalid>",
            expected="<valid canonical JSON>",
        )
        raise AssertionError("unreachable") from exc
    if not isinstance(manifest, Mapping):
        _raise_checkpoint_identity_mismatch(
            field="manifest",
            stored=manifest,
            expected="<object>",
        )
    if manifest.get("schema") != AURA_CHECKPOINT_MANIFEST_SCHEMA:
        _raise_checkpoint_identity_mismatch(
            field="manifest.schema",
            stored=manifest.get("schema"),
            expected=AURA_CHECKPOINT_MANIFEST_SCHEMA,
        )
    stored_identity = manifest.get("identity")
    from prismaquant.dev_mode import seal_check
    from prismaquant.production_weight_cache import first_identity_difference

    def attribution_surface(identity):
        if not isinstance(identity, Mapping):
            return None
        extra = identity.get("extra", {})
        joint = extra.get("joint_aura", {}) if isinstance(extra, Mapping) else {}
        return joint.get("sequence_attribution") if isinstance(joint, Mapping) else None

    # This is the requested measurement surface, not a producer-source seal:
    # reusing a different selector would silently ignore the instrument request.
    stored_attribution = attribution_surface(stored_identity)
    expected_attribution = attribution_surface(expected_identity)
    if stored_attribution != expected_attribution:
        _raise_checkpoint_identity_mismatch(
            field="extra.joint_aura.sequence_attribution",
            stored=stored_attribution, expected=expected_attribution)

    difference = first_identity_difference(stored_identity, expected_identity)
    if difference is not None:
        field, stored, expected = difference

        def refusal() -> RuntimeError:
            # Lazy: the producer-source listing and the per-file comparison
            # stat and hash the whole package, and dev mode never raises this
            # refusal (#2218).
            error = _checkpoint_identity_mismatch(
                field=field, stored=stored, expected=expected)
            if field == "producer_source_sha256":
                error = _name_first_differing_source_file(
                    error, manifest, stored, expected)
            return error

        # The checkpoint identity binds the producer source and the run's
        # inputs: a run seal (PQ #1147). Dev mode prints the difference and
        # reuses the lineage under its own recorded identity -- no archive,
        # no recompute -- so its unit envelopes still match their manifest.
        if not seal_check(
                "AURA checkpoint identity", expected_identity, stored_identity,
                where=str(checkpoint_dir),
                refusal=refusal):
            expected_identity = stored_identity
    expected_digest = _canonical_json_sha256(
        expected_identity,
        where="AURA checkpoint identity",
    )
    if manifest.get("identity_sha256") != expected_digest:
        _raise_checkpoint_identity_mismatch(
            field="manifest.identity_sha256",
            stored=manifest.get("identity_sha256"),
            expected=expected_digest,
        )
    return expected_digest


class _BoundedCheckpointBytes(io.BytesIO):
    """Refuse serialization beyond already-reserved host staging."""

    def __init__(self, limit: int):
        super().__init__()
        self._limit = limit

    def write(self, buffer, /) -> int:
        if self.tell() + len(buffer) > self._limit:
            raise ValueError("checkpoint encoding exceeds staging bytes limit")
        return super().write(buffer)


def _checkpoint_pickle_bytes(value, *, max_bytes: int | None) -> bytes:
    if max_bytes is None:
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("checkpoint encoding bytes limit must be a positive integer")
    with _BoundedCheckpointBytes(max_bytes) as sink:
        pickle.Pickler(sink, protocol=pickle.HIGHEST_PROTOCOL).dump(value)
        return sink.getvalue()


def encode_unit(
    *,
    qname: str,
    identity_sha256: str,
    state: Mapping[str, object],
    max_bytes: int | None = None,
) -> bytes:
    """Freeze the existing envelope; caller reserves peak staging first."""
    state_bytes = _checkpoint_pickle_bytes(dict(state), max_bytes=max_bytes)
    envelope = {
        "schema": AURA_CHECKPOINT_UNIT_SCHEMA,
        "qname": str(qname),
        "identity_sha256": str(identity_sha256),
        "payload_sha256": bytes_sha256hex(state_bytes),
        "payload": state_bytes,
    }
    return _checkpoint_pickle_bytes(envelope, max_bytes=max_bytes)


def write_unit(
    checkpoint_dir: Path,
    *,
    qname: str,
    identity_sha256: str,
    state: Mapping[str, object],
) -> None:
    encoded = encode_unit(
        qname=qname, identity_sha256=identity_sha256, state=state)
    atomic_write_bytes(unit_path(checkpoint_dir, qname), encoded)


def load_unit(
    path: Path,
    *,
    qname: str,
    identity_sha256: str,
) -> dict[str, object]:
    try:
        with path.open("rb") as handle:
            envelope = pickle.load(handle)
    except Exception as exc:
        raise RuntimeError(
            f"AURA unit checkpoint {path} is corrupt for {qname}; refusing "
            "reuse or recompute"
        ) from exc
    return decode_unit(
        envelope, path=path, qname=qname, identity_sha256=identity_sha256)


def decode_unit(
    envelope: object,
    *,
    path: Path,
    qname: str,
    identity_sha256: str,
) -> dict[str, object]:
    """Validate and decode one envelope using the existing checkpoint contract."""
    if not isinstance(envelope, Mapping):
        raise RuntimeError(
            f"AURA unit checkpoint {path} is not an envelope for {qname}; "
            "refusing reuse or recompute"
        )
    for field, expected in (
        ("schema", AURA_CHECKPOINT_UNIT_SCHEMA),
        ("qname", str(qname)),
    ):
        if envelope.get(field) != expected:
            _raise_checkpoint_identity_mismatch(
                field=f"unit[{qname}].{field}",
                stored=envelope.get(field),
                expected=expected,
            )
    # The envelope's identity digest is the lineage seal (PQ #1147): dev mode
    # prints a difference and reuses the unit. Its payload digest below is
    # integrity and refuses in both modes.
    from prismaquant.dev_mode import seal_check

    seal_check(
        "AURA unit checkpoint identity", str(identity_sha256), envelope.get("identity_sha256"),
        where=str(path),
        refusal=_checkpoint_identity_mismatch(
            field=f"unit[{qname}].identity_sha256",
            stored=envelope.get("identity_sha256"),
            expected=str(identity_sha256)))
    payload = envelope.get("payload")
    if not isinstance(payload, bytes):
        raise RuntimeError(
            f"AURA unit checkpoint {path} has no byte payload for {qname}; "
            "refusing reuse or recompute"
        )
    payload_sha256 = bytes_sha256hex(payload)
    if envelope.get("payload_sha256") != payload_sha256:
        raise RuntimeError(
            f"AURA unit checkpoint {path} payload_sha256 differs for {qname}; "
            "refusing reuse or recompute"
        )
    try:
        state = pickle.loads(payload)
    except Exception as exc:
        raise RuntimeError(
            f"AURA unit checkpoint {path} state is corrupt for {qname}; "
            "refusing reuse or recompute"
        ) from exc
    if not isinstance(state, Mapping):
        raise RuntimeError(
            f"AURA unit checkpoint {path} state is not an object for {qname}; "
            "refusing reuse or recompute"
        )
    return dict(state)


def prepare_journal(
    checkpoint_dir: str | Path,
    *,
    resume: bool,
    identity: Mapping[str, object],
    names: Sequence[str],
) -> tuple[Path, str, dict[str, dict[str, object]]]:
    from prismaquant.dev_mode import dev_mode_enabled, seal_check

    root = Path(checkpoint_dir)
    if root.exists() and not root.is_dir():
        raise RuntimeError(f"AURA checkpoint path is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        if not resume:
            raise RuntimeError(
                f"AURA checkpoint manifest already exists at {manifest_path}; "
                "pass --resume to validate and reuse it"
            )
        # An identity mismatch is a run seal: dev mode reuses the lineage
        # under its recorded identity (PQ #1147). A corrupt manifest or a
        # foreign schema refuses in both modes: bit-rot is not a seal.
        identity_sha256 = _load_aura_checkpoint_manifest(root, identity)
    else:
        existing_units = sorted((root / "units").glob("*.pkl"))
        if existing_units:
            if dev_mode_enabled():
                # Units with no manifest have no recorded identity to reuse
                # them under (an interrupted transition): dev mode archives
                # them whole and recomputes, as before PQ #1147.
                _dev_archive_checkpoint_lineage(
                    root,
                    f"{len(existing_units)} unit checkpoints exist without a manifest")
            else:
                raise RuntimeError(
                    "AURA checkpoint units exist without a manifest; refusing "
                    f"name-gated reuse or recompute. sample={existing_units[:8]}"
                )
        identity_sha256 = _write_aura_checkpoint_manifest(
            root,
            identity,
            names,
        )

    expected_paths = {
        unit_path(root, name): str(name)
        for name in names
    }
    unexpected = sorted(
        path for path in (root / "units").glob("*.pkl")
        if path not in expected_paths
    )
    if unexpected:
        # A unit outside this run's roster is left where it is and not read
        # in dev mode (PQ #1147); certified mode refuses the lineage.
        seal_check(
            "AURA checkpoint unit roster", [], [str(path.name) for path in unexpected[:8]],
            where=str(root),
            refusal=_checkpoint_identity_mismatch(
                field="units.unexpected",
                stored=[str(path.name) for path in unexpected[:8]],
                expected=[]))
    completed: dict[str, dict[str, object]] = {}
    for path, name in expected_paths.items():
        if not path.is_file():
            continue
        completed[name] = load_unit(
            path,
            qname=name,
            identity_sha256=identity_sha256,
        )
    return root, identity_sha256, completed


def build_checkpoint_identity(
    *,
    model: nn.Module,
    calib_ids: torch.Tensor,
    names: Sequence[str],
    linears: Mapping[str, nn.Linear],
    formats: Sequence[str],
    chunks: Sequence[Sequence[str]],
    n_probes: int,
    token_scope: str,
    temperature: float,
    seed_base: int,
    dw_dtype: str,
    include_lm_head: bool,
    hook_harvest: bool,
    allow_packed_expert_omission: bool,
    probe_microbatch: int,
    collect_col_energy: bool,
    require_production_cache: bool,
    production_cache: object,
    git_commit: str,
    extra_identity: Mapping[str, object] | None,
) -> dict[str, object]:
    raw_calib_sha256 = bytes_sha256hex(
        calib_ids.detach().cpu().contiguous().numpy().tobytes()
    )
    cache_metadata = getattr(production_cache, "metadata", None)
    cache_metadata = cache_metadata if isinstance(cache_metadata, Mapping) else {}
    identity = {
        "schema": AURA_CHECKPOINT_IDENTITY_SCHEMA,
        "git_commit": str(git_commit),
        "producer_source_sha256": aura_provenance.source_sha256(),
        "calibration": {
            "shape": [int(dim) for dim in calib_ids.shape],
            "dtype": str(calib_ids.dtype),
            "sha256": raw_calib_sha256,
            "calib_hash": calibration_data_hash(calib_ids),
        },
        "formats": [str(fmt) for fmt in formats],
        "units": [
            {
                "qname": str(name),
                "shape": [int(dim) for dim in linears[name].weight.shape],
                "dtype": str(linears[name].weight.dtype),
                "n_params": int(linears[name].weight.numel()),
            }
            for name in names
        ],
        "chunks": [[str(name) for name in chunk] for chunk in chunks],
        "n_probes": int(n_probes),
        "token_scope": str(token_scope),
        "temperature": float(temperature),
        "seed_base": int(seed_base),
        "measurement_dtype": str(next(model.parameters()).dtype),
        "dw_dtype": str(dw_dtype),
        "include_lm_head": bool(include_lm_head),
        "hook_harvest": bool(hook_harvest),
        "allow_packed_expert_omission": bool(allow_packed_expert_omission),
        "probe_microbatch": int(probe_microbatch),
        "collect_col_energy": bool(collect_col_energy),
        "require_production_cache": bool(require_production_cache),
        "production_cache_calib_hash": cache_metadata.get("calib_hash"),
        # These two keys carried the retired codebook lane's cache-pair and
        # render identities (archived 2026-09-25, #1304). They are always null
        # now, and they stay in the identity so a checkpoint written before
        # the removal still matches under a source transition, which refuses
        # any non-source identity change.
        "production_cache_pair_identity": None,
        "cb_render_identity": None,
        "extra": dict(extra_identity or {}),
    }
    return _canonical_json(identity, where="AURA checkpoint identity")


def capture_unit_state(
    name: str,
    nonzero_formats: Sequence[str],
    *,
    s2: Mapping[tuple[str, str], float],
    s4: Mapping[tuple[str, str], float],
    x2_probe: Mapping[tuple[str, str], list[float]],
    dw_src: Mapping[tuple[str, str], str],
    g_trace: Mapping[str, float],
    col_energy: Mapping[str, torch.Tensor],
    weight_mse_diagnostic: Mapping[tuple[str, str], float] | None = None,
    source_weight_identity: Mapping[str, Mapping[str, object]] | None = None,
    observation_counts: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, object]:
    rows: dict[str, dict[str, object]] = {}
    for fmt in nonzero_formats:
        key = (name, fmt)
        if key not in s2:
            continue
        rows[fmt] = {
            "s2": float(s2[key]),
            "s4": float(s4[key]),
            "x2_probe": list(x2_probe[key]),
            "dw_src": str(dw_src[key]),
        }
        if (
            weight_mse_diagnostic is not None
            and key in weight_mse_diagnostic
        ):
            rows[fmt]["weight_mse_diagnostic"] = float(
                weight_mse_diagnostic[key]
            )
    state = {
        "g_trace": float(g_trace[name]),
        "rows": rows,
        "col_energy": (
            col_energy[name].detach().to(device="cpu", dtype=torch.float32)
            if name in col_energy
            else None
        ),
    }
    if source_weight_identity is not None and name in source_weight_identity:
        state["source_weight_identity"] = dict(
            source_weight_identity[name]
        )
    if observation_counts is not None:
        state['joint_eval_observations'] = dict(observation_counts[name])
    return state


def restore_unit_state(
    name: str,
    state: Mapping[str, object],
    *,
    nonzero_formats: Sequence[str],
    n_probes: int,
    collect_col_energy: bool,
    s2: dict[tuple[str, str], float],
    s4: dict[tuple[str, str], float],
    x2_probe: dict[tuple[str, str], list[float]],
    dw_src: dict[tuple[str, str], str],
    g_trace: dict[str, float],
    col_energy: dict[str, torch.Tensor],
    diagnostic_weight_mse_pairs: set[tuple[str, str]] | None = None,
    weight_mse_diagnostic: dict[tuple[str, str], float] | None = None,
    require_source_weight_identity: bool = False,
    source_weight_identity: dict[str, dict[str, object]] | None = None,
    observation_counts: dict[str, dict[str, int]] | None = None,
) -> None:
    if observation_counts is not None:
        observation = state.get('joint_eval_observations')
        if (not isinstance(observation, Mapping)
                or set(observation) != {'tokens', 'calls', 'per_probe', 'count_scope', 'n_probes'}
                or observation.get('count_scope') != 'summed_over_probes'
                or observation.get('n_probes') != n_probes
                or not isinstance(observation.get('per_probe'), list)
                or len(observation['per_probe']) != n_probes
                or any(not isinstance(item, Mapping) or set(item) != {'tokens', 'calls'}
                       or any(type(value) is not int or value < 0 for value in item.values())
                       for item in observation['per_probe'])
                or any(type(observation.get(key)) is not int
                       or observation[key] != sum(item[key] for item in observation['per_probe'])
                       for key in ('tokens', 'calls'))):
            raise RuntimeError(f'joint pilot checkpoint observation counts missing for {name}')
        observation_counts[name] = {**observation, 'per_probe': [dict(item) for item in observation['per_probe']]}
    try:
        g_trace[name] = float(state["g_trace"])
    except Exception as exc:
        raise RuntimeError(
            f"AURA unit checkpoint state for {name} has invalid g_trace; "
            "refusing reuse or recompute"
        ) from exc
    rows = state.get("rows")
    if not isinstance(rows, Mapping):
        raise RuntimeError(
            f"AURA unit checkpoint state for {name} has invalid rows; "
            "refusing reuse or recompute"
        )
    unexpected = sorted(set(str(fmt) for fmt in rows) - set(nonzero_formats))
    if unexpected:
        raise RuntimeError(
            f"AURA unit checkpoint state for {name} has unexpected formats "
            f"{unexpected}; refusing reuse or recompute"
        )
    for raw_fmt, raw_row in rows.items():
        fmt = str(raw_fmt)
        if not isinstance(raw_row, Mapping):
            raise RuntimeError(
                f"AURA unit checkpoint state for {name}@{fmt} is invalid; "
                "refusing reuse or recompute"
            )
        samples = raw_row.get("x2_probe")
        if (
            not isinstance(samples, Sequence)
            or isinstance(samples, (str, bytes))
            or len(samples) != int(n_probes)
        ):
            raise RuntimeError(
                f"AURA unit checkpoint state for {name}@{fmt} has "
                f"x2_probe length {len(samples) if isinstance(samples, Sequence) else '<invalid>'}; "
                f"expected {n_probes}; refusing reuse or recompute"
            )
        key = (name, fmt)
        try:
            s2[key] = float(raw_row["s2"])
            s4[key] = float(raw_row["s4"])
            x2_probe[key] = [float(value) for value in samples]
            dw_src[key] = str(raw_row["dw_src"])
        except Exception as exc:
            raise RuntimeError(
                f"AURA unit checkpoint state for {name}@{fmt} is malformed; "
                "refusing reuse or recompute"
            ) from exc
        expected_diagnostic = (
            diagnostic_weight_mse_pairs is not None
            and key in diagnostic_weight_mse_pairs
        )
        observed_diagnostic = "weight_mse_diagnostic" in raw_row
        if observed_diagnostic != expected_diagnostic:
            raise RuntimeError(
                f"AURA unit checkpoint state for {name}@{fmt} diagnostic "
                "weight-MSE scope differs; refusing reuse or recompute"
            )
        if expected_diagnostic:
            if weight_mse_diagnostic is None:
                raise RuntimeError(
                    "AURA diagnostic weight-MSE restore has no destination"
                )
            try:
                weight_mse_diagnostic[key] = float(
                    raw_row["weight_mse_diagnostic"]
                )
            except Exception as exc:
                raise RuntimeError(
                    f"AURA unit checkpoint state for {name}@{fmt} has an "
                    "invalid diagnostic weight-MSE"
                ) from exc
    stored_col_energy = state.get("col_energy")
    if collect_col_energy:
        if stored_col_energy is not None and not isinstance(
            stored_col_energy, torch.Tensor
        ):
            raise RuntimeError(
                f"AURA unit checkpoint state for {name} has invalid "
                "col_energy; refusing reuse or recompute"
            )
        if isinstance(stored_col_energy, torch.Tensor):
            col_energy[name] = stored_col_energy.detach().to(
                device="cpu", dtype=torch.float32
            )
    elif stored_col_energy is not None:
        raise RuntimeError(
            f"AURA unit checkpoint state for {name} unexpectedly carries "
            "col_energy; refusing reuse or recompute"
        )
    raw_source = state.get("source_weight_identity")
    if require_source_weight_identity:
        if not isinstance(raw_source, Mapping):
            raise RuntimeError(
                f"AURA unit checkpoint state for {name} has no production "
                "anchor source-weight identity; refusing reuse or recompute"
            )
        try:
            shape = [int(dim) for dim in raw_source["shape"]]
            digest = str(raw_source["sha256"]).lower()
        except Exception as exc:
            raise RuntimeError(
                f"AURA unit checkpoint state for {name} has malformed "
                "source-weight identity"
            ) from exc
        if (
            len(shape) != 2
            or any(dim <= 0 for dim in shape)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise RuntimeError(
                f"AURA unit checkpoint state for {name} has invalid "
                "source-weight shape/hash"
            )
        if source_weight_identity is None:
            raise RuntimeError(
                "AURA source-weight identity restore has no destination"
            )
        source_weight_identity[name] = {
            "shape": shape,
            "sha256": digest,
        }
    elif raw_source is not None:
        raise RuntimeError(
            f"AURA unit checkpoint state for {name} unexpectedly carries a "
            "production-anchor source identity"
        )


