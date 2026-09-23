"""Shared calibration-data helpers for production measurement paths."""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path
import re

import torch


def _read_calibration_payload(path: Path, expected_sha256: str) -> bytes:
    """One calibration artifact, staged-pinned under the active tier policy.

    The generic reader (``staged_whole_file.read_staged_whole_file``) under
    the ``calibration`` label, so every refusal keeps its historical name.
    """
    from .staged_whole_file import read_staged_whole_file

    return read_staged_whole_file(path, expected_sha256, label="calibration")


def _decode_calibration_buffer(raw: bytes) -> tuple[torch.Tensor, dict]:
    """The pinned ``(calibration_ids, provenance)`` out of staged bytes.

    Tensor decoding is the installed public bytes decoder
    (``safetensors.torch.load``) — the canonical parser, which rejects
    noncontiguous and trailing framing a span reader would accept — never
    a private reimplementation. Only the ``calibration_provenance``
    metadata still comes from a bounded header extraction. The decoded
    tensors retain decoder-owned storage and stay valid after the lease
    releases. The reader freezes its bytearray once; the public decoder
    manages its own allocations, so this is not a zero-copy guarantee.
    Structural problems are the offline contract's
    ValueErrors, exactly as the ``safe_open`` path reports them (the
    decoders' own exception types are wrapped, never compared).
    """
    from .residency_shard_reader import MAX_HEADER_BYTES

    from safetensors.torch import load

    if len(raw) < 8:
        raise ValueError("exact calibration input has no safetensors header length")
    header_bytes = int.from_bytes(raw[:8], "little")
    if not 0 < header_bytes <= min(MAX_HEADER_BYTES, len(raw) - 8):
        raise ValueError("exact calibration input header length is out of range")
    try:
        header = json.loads(raw[8:8 + header_bytes])
    except ValueError as exc:
        raise ValueError("exact calibration input header is not an object") from exc
    if type(header) is not dict:
        raise ValueError("exact calibration input header is not an object")
    try:
        provenance = json.loads(header["__metadata__"]["calibration_provenance"])
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError("exact calibration input requires calibration_provenance JSON") from exc
    try:
        tensors = load(bytes(raw))
    except Exception as exc:
        raise ValueError(
            "exact calibration input is not a decodable safetensors artifact") from exc
    if set(tensors) != {"calibration_ids"}:
        raise ValueError("exact calibration input requires only calibration_ids")
    return tensors["calibration_ids"], provenance


def _validate_calibration_draw(ids, provenance, *, artifact_sha256, n_samples, seqlen):
    """The offline draw contract, shared by both read paths.

    dtype/shape/domain/provenance/draw-hash checks and both identity hash
    conventions, exactly as the legacy reader reports them.
    """
    if ids.dtype != torch.int64 or list(ids.shape) != [n_samples, seqlen] or ids.numel() == 0:
        raise ValueError("exact calibration input dtype/shape differs from requested draw")
    if bool((ids < 0).any()) or bool((ids > torch.iinfo(torch.int32).max).any()):
        raise ValueError("exact calibration token IDs exceed the nonnegative int32 identity domain")
    if not isinstance(provenance, dict):
        raise ValueError("exact calibration provenance must be an object")
    draw_sha256 = hashlib.sha256(ids.to(torch.int32).numpy().tobytes()).hexdigest()
    if (provenance.get("fit_ids_sha256") != draw_sha256
            or provenance.get("fit_tokens") != ids.numel()
            or provenance.get("nsamples") != n_samples
            or provenance.get("seqlen") != seqlen):
        raise ValueError("exact calibration input differs from declared draw provenance")
    return ids, {
        "schema": "prismaquant.calibration_input.v1", "artifact_sha256": artifact_sha256,
        "calibration_sha256": hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest(),
        "shape": list(ids.shape), "dtype": str(ids.dtype), "provenance": provenance,
    }


def load_calibration_input(path, *, expected_sha256, n_samples, seqlen):
    """Load an independently pinned token draw without invoking a sampler.

    The campaign's ``fit_ids_sha256`` hashes int32 token bytes; joint AURA
    hashes the actual int64 tensor. Preserve and check both conventions.
    This is explicit artifact preparation, outside the measurement hot path.

    Under the active staged-tier policy the artifact comes off the stage
    through a lifetime-pinned window (RAM first, sealed-allowed SSD
    fallback, never the pool) and decodes from the same owned bytes the
    digest verifies; inactive policy keeps the legacy pool read exactly.
    """
    from safetensors import safe_open

    from .staged_tier_policy import policy_is_active

    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("exact calibration input requires its independent SHA256")
    path = Path(path)
    if policy_is_active():
        from .staged_lease import LeaseRefused

        raw = _read_calibration_payload(path, expected_sha256)
        artifact_sha256 = hashlib.sha256(raw).hexdigest()
        if artifact_sha256 != expected_sha256:
            raise LeaseRefused("calibration-staged-digest-divergent",
                               kind="integrity")
        ids, provenance = _decode_calibration_buffer(raw)
        return _validate_calibration_draw(
            ids, provenance, artifact_sha256=artifact_sha256,
            n_samples=n_samples, seqlen=seqlen)
    artifact_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if artifact_sha256 != expected_sha256:
        raise ValueError("exact calibration input artifact SHA256 mismatch")
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        if set(stream.keys()) != {"calibration_ids"}:
            raise ValueError("exact calibration input requires only calibration_ids")
        try:
            provenance = json.loads(stream.metadata()["calibration_provenance"])
        except (TypeError, KeyError, ValueError) as exc:
            raise ValueError("exact calibration input requires calibration_provenance JSON") from exc
        ids = stream.get_tensor("calibration_ids")
    ids, receipt = _validate_calibration_draw(
        ids, provenance, artifact_sha256=artifact_sha256,
        n_samples=n_samples, seqlen=seqlen)
    # Refuse a file replacement between its digest check and tensor loading.
    if hashlib.sha256(path.read_bytes()).hexdigest() != artifact_sha256:
        raise ValueError("exact calibration input changed while loading")
    return ids, receipt


def _sample_token_windows_from_texts(
    texts: Sequence[str],
    tokenizer,
    n_samples: int,
    seqlen: int,
    *,
    seed: int,
) -> torch.Tensor:
    import random

    rng = random.Random(int(seed))
    order = list(range(len(texts)))
    rng.shuffle(order)
    windows: list[torch.Tensor] = []
    buffer: list[int] = []
    eos = tokenizer.eos_token_id
    for idx in order:
        text = str(texts[idx]).strip()
        if not text:
            continue
        ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        ).input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if not ids:
            continue
        buffer.extend(int(v) for v in ids)
        if eos is not None:
            buffer.append(int(eos))
        while len(buffer) >= int(seqlen) and len(windows) < int(n_samples):
            max_start = len(buffer) - int(seqlen)
            start = rng.randint(0, max_start) if max_start > 0 else 0
            window = buffer[start:start + int(seqlen)]
            windows.append(torch.tensor(window, dtype=torch.long))
            del buffer[:start + int(seqlen)]
        if len(windows) >= int(n_samples):
            break
    if len(windows) < int(n_samples):
        raise RuntimeError(
            f"only built {len(windows)} calibration windows; "
            f"needed {int(n_samples)}"
        )
    return torch.stack(windows, dim=0)


def load_wikitext2_raw(split: str = "train"):
    """Load wikitext-2-raw, tolerant of dataset-id deprecation.

    Newer ``huggingface_hub`` rejects the bare ``"wikitext"`` repo id
    (requires ``namespace/name``), so try the canonical mirror first and
    fall back to the legacy id for older stacks."""
    from datasets import load_dataset

    last = None
    for ds_id in ("Salesforce/wikitext", "wikitext", "mindchain/wikitext-2"):
        try:
            return load_dataset(ds_id, "wikitext-2-raw-v1", split=split)
        except Exception as e:  # try the next id
            last = e
    raise RuntimeError(f"could not load wikitext-2-raw-v1: {last}")


def load_wikitext_calibration_windowed(
    tokenizer,
    n_samples: int,
    seqlen: int,
    *,
    split: str = "train",
    seed: int = 42,
) -> torch.Tensor:
    """Load small WikiText calibration windows without tokenizing the full corpus."""
    ds = load_wikitext2_raw(split=split)
    texts = [row["text"] for row in ds if str(row.get("text", "")).strip()]
    del ds
    return _sample_token_windows_from_texts(
        texts,
        tokenizer,
        n_samples,
        seqlen,
        seed=seed,
    )


def _dtype_from_name(name: str) -> torch.dtype:
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype {name!r}")
