"""Shared calibration-data helpers for production measurement paths."""
from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
import re
import random

import torch

from .digests import bytes_sha256hex
from .source_read_plan import safetensors_prefix_length


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

    header_bytes = safetensors_prefix_length(
        raw[:8], len(raw), max_bytes=MAX_HEADER_BYTES,
        short_error="exact calibration input has no safetensors header length",
        range_error="exact calibration input header length is out of range")
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
    draw_sha256 = bytes_sha256hex(ids.to(torch.int32).numpy().tobytes())
    if (any(type(provenance.get(field)) is not int
            for field in ("fit_tokens", "nsamples", "seqlen"))
            or provenance.get("fit_ids_sha256") != draw_sha256
            or provenance.get("fit_tokens") != ids.numel()
            or provenance.get("nsamples") != n_samples
            or provenance.get("seqlen") != seqlen):
        raise ValueError("exact calibration input differs from declared draw provenance")
    return ids, {
        "schema": "prismaquant.calibration_input.v1", "artifact_sha256": artifact_sha256,
        "calibration_sha256": bytes_sha256hex(ids.contiguous().numpy().tobytes()),
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
        artifact_sha256 = bytes_sha256hex(raw)
        if artifact_sha256 != expected_sha256:
            raise LeaseRefused("calibration-staged-digest-divergent",
                               kind="integrity")
        ids, provenance = _decode_calibration_buffer(raw)
        return _validate_calibration_draw(
            ids, provenance, artifact_sha256=artifact_sha256,
            n_samples=n_samples, seqlen=seqlen)
    artifact_sha256 = bytes_sha256hex(path.read_bytes())
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
    if bytes_sha256hex(path.read_bytes()) != artifact_sha256:
        raise ValueError("exact calibration input changed while loading")
    return ids, receipt


def load_calibration_corpus(path, *, expected_sha256: str) -> str:
    """Decode actual UTF-8 corpus bytes bound by the pinned draw's provenance."""
    from .staged_tier_policy import policy_is_active
    from .staged_whole_file import read_staged_whole_file

    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("exact calibration corpus requires its provenance SHA256")
    path = Path(path)
    raw = (read_staged_whole_file(path, expected_sha256, label="calibration-corpus")
           if policy_is_active() else path.read_bytes())
    if bytes_sha256hex(raw) != expected_sha256:
        raise ValueError("exact calibration corpus SHA256 mismatch")
    text = raw.decode("utf-8")
    if not text.strip():
        raise ValueError("exact calibration corpus must contain real text")
    return text


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


def load_calibration(tokenizer, source: str, n_samples: int,
                     seqlen: int, *, calib_seed: int = 42) -> torch.Tensor:
    """Load calibration from a HuggingFace dataset id, a local .jsonl, or
    a local .txt file. Extensionless regular files are content-sniffed so a
    bind-mounted JSONL at a canonical container path such as ``/dataset``
    remains local. JSONL rows can have either {"text": ...} or
    {"messages": [...]} for chat-style data.
    """
    import os

    # A local-file source that does not exist must NOT fall through to the
    # HuggingFace loader. It used to: the `.jsonl`/`.txt` branches were guarded
    # by os.path.exists, so a missing file skipped to the generic `else` and
    # HF reported `Dataset '/home/.../diverse-v1.jsonl' doesn't exist on the
    # Hub` -- an error that names a filesystem path as a dataset id and sends
    # the reader looking for something to download. Nothing is downloadable;
    # the file is simply absent.
    if (source.endswith((".jsonl", ".txt")) or os.sep in source) \
            and not os.path.exists(source):
        raise FileNotFoundError(
            f"calibration file not found: {source}\n"
            "This is a local path, not a HuggingFace dataset id, so there is "
            "nothing to download. Either:\n"
            "  * build the default corpus:  python tools/"
            "build_diverse_calibration.py --output <path> --tokenizer <model>\n"
            "  * point DATASET at your own .jsonl ({\"text\": ...} or "
            "{\"messages\": [...]} rows) or .txt (one sample per line)\n"
            "  * point DATASET at a HuggingFace dataset id "
            "(e.g. ultrachat_200k)"
        )

    local_kind: str | None = None
    if source.endswith(".jsonl") and os.path.isfile(source):
        local_kind = "jsonl"
    elif source.endswith(".txt") and os.path.isfile(source):
        local_kind = "txt"
    elif (
        os.path.isfile(source)
        and not os.path.splitext(os.path.basename(source))[1]
    ):
        # Docker commonly mounts one host file at a path-neutral canonical
        # name.  Do not reinterpret that existing regular file as a Hub
        # dataset merely because the container target has no suffix.  Sniff
        # only extensionless files; known alternate local formats retain the
        # generic datasets-loader behavior they had before this contract.
        first_nonempty = ""
        with open(source) as handle:
            for line in handle:
                if line.strip():
                    first_nonempty = line
                    break
        try:
            first_value = json.loads(first_nonempty)
        except (json.JSONDecodeError, TypeError):
            first_value = None
        local_kind = (
            "jsonl"
            if isinstance(first_value, dict)
            and ("text" in first_value or "messages" in first_value)
            else "txt"
        )

    texts: list[str] = []
    if local_kind == "jsonl":
        with open(source) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "messages" in obj:
                    try:
                        texts.append(tokenizer.apply_chat_template(
                            obj["messages"], tokenize=False))
                    except Exception:
                        # Tokenizer has no chat template (base / pt models
                        # like DSv4-Flash-Base). Fall back to concatenating
                        # the message contents — good enough as calibration
                        # text, mirrors the ultrachat fallback below.
                        parts = []
                        for m in obj["messages"]:
                            if isinstance(m, dict):
                                c = m.get("content")
                                if isinstance(c, str) and c:
                                    parts.append(c)
                        if parts:
                            texts.append("\n\n".join(parts))
                elif "text" in obj:
                    texts.append(obj["text"])
    elif local_kind == "txt":
        with open(source) as f:
            texts = [ln.strip() for ln in f if ln.strip()]
    elif source == "ultrachat_200k":
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceH4/ultrachat_200k",
                          split="train_sft", streaming=True)
        for row in ds:
            msgs = row.get("messages", [])
            if not msgs:
                continue
            try:
                texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
            except Exception:
                # Tokenizer has no chat template (base / pt models).
                # Join plain message contents — good enough as calibration.
                parts = []
                for m in msgs:
                    if isinstance(m, dict):
                        c = m.get("content")
                        if isinstance(c, str) and c:
                            parts.append(c)
                if parts:
                    texts.append("\n\n".join(parts))
            if len(texts) >= n_samples * 8:
                break
    else:
        try:
            from datasets import load_dataset
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Loading HuggingFace calibration datasets requires the "
                "'datasets' package. Local .jsonl and .txt calibration files "
                "do not require it."
            ) from exc
        # Generic HF dataset loader. Handles three common schemas:
        #   1. {"text": "..."} — raw text corpora (pile, wikitext, etc.)
        #   2. {"messages": [...]} — chat-format SFT (ultrachat, tulu-3, etc.)
        #   3. anything else — falls back to first string column
        # Streaming when possible so we don't download the full dataset for
        # just 32 samples.
        try:
            ds = load_dataset(source, split="train", streaming=True)
            stream = True
        except Exception:
            ds = load_dataset(source, split="train")
            stream = False

        # Probe one row to detect schema
        iterator = iter(ds) if stream else ds
        first = next(iterator) if stream else (ds[0] if len(ds) else {})
        schema = None
        if "messages" in first:
            schema = "messages"
        elif "text" in first:
            schema = "text"
        else:
            # pick first string-valued column
            for k, v in first.items():
                if isinstance(v, str):
                    schema = k
                    break
        if schema is None:
            raise ValueError(f"Could not find text or messages field in {source}")
        print(f"[probe] {source} schema: {schema}", flush=True)

        # Re-iterate (we consumed the first row)
        if stream:
            ds = load_dataset(source, split="train", streaming=True)
            iterator = iter(ds)
        else:
            iterator = iter(ds)

        for row in iterator:
            if schema == "messages":
                msgs = row.get("messages") or row.get("conversations") or []
                if not msgs:
                    continue
                try:
                    texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
                except Exception:
                    # Tokenizer has no chat template (base / pt models)
                    # or the template rejects this message shape.
                    # Fall back to concatenating the message contents —
                    # good enough as calibration text, decouples the
                    # probe from the tokenizer's chat-template state.
                    parts = []
                    for m in msgs:
                        if isinstance(m, dict):
                            c = m.get("content")
                            if isinstance(c, str) and c:
                                parts.append(c)
                    if parts:
                        texts.append("\n\n".join(parts))
            else:
                v = row.get(schema)
                if isinstance(v, str) and v.strip():
                    texts.append(v)
            if len(texts) >= n_samples * 8:
                break

    # Two-pass sampling:
    #   1) first pass picks any sample already >= seqlen tokens
    #   2) fallback packs multiple short samples together (separated by
    #      EOS) to reach seqlen. This makes SFT/chat datasets with short
    #      turns (tulu-3, glaive) usable without lowering seqlen.
    # calib_seed controls (a) which subset of dataset texts is tried (shuffle)
    # and (b) where within each text the seqlen window starts. Default 42 is
    # the historical value and is byte-stable.
    if calib_seed != 42:
        rng_shuffle = random.Random(calib_seed)
        rng_shuffle.shuffle(texts)
    random.seed(calib_seed)
    samples = []
    used_documents = set()
    for document, t in enumerate(texts):
        ids = tokenizer(t, return_tensors="pt", truncation=False).input_ids
        if ids.size(1) < seqlen:
            continue
        start = random.randint(0, ids.size(1) - seqlen)
        samples.append(ids[0, start:start + seqlen])
        used_documents.add(document)
        if len(samples) >= n_samples:
            break

    if len(samples) < n_samples:
        eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        # Pack short samples by concatenating with EOS separator
        buf: list[int] = []
        for document, t in enumerate(texts):
            if document in used_documents:
                continue
            ids = tokenizer(t, return_tensors="pt", truncation=False).input_ids[0].tolist()
            buf.extend(ids)
            buf.append(eos)
            while len(buf) >= seqlen and len(samples) < n_samples:
                samples.append(torch.tensor(buf[:seqlen], dtype=torch.long))
                buf = buf[seqlen:]
            if len(samples) >= n_samples:
                break

    if len(samples) < n_samples:
        print(f"[probe] warning: only got {len(samples)}/{n_samples} samples "
              f"(even with packing). Consider wider corpus.",
              flush=True)
    return torch.stack(samples[:n_samples], dim=0)
