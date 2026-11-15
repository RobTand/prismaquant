#!/usr/bin/env python3
"""tied_embeddings.py — weight tying (`tie_word_embeddings`) support.

A checkpoint that declares ``tie_word_embeddings: True`` and ships **no**
``lm_head.*`` tensor stores exactly one matrix: the input embedding. The
output head is an *alias* of that same storage, materialized at load time
by `transformers`' `tie_weights()`.

PrismaQuant's streaming loader never calls `tie_weights()` — it builds a
meta skeleton and installs tensors it finds in the safetensors index. On a
tied checkpoint there is no `lm_head.weight` to install, so the head stays
on `meta` and the first consumer to touch it dies:

    NotImplementedError: Cannot copy out of meta tensor; no data!

(seen on `google/gemma-4-31b-it`, whose embedding is additionally behind a
VL wrapper prefix: `model.language_model.embed_tokens.weight`).

Two separate things follow from a tie, and this module owns both:

1. **Materialization** (:func:`resolve_tied_output_embedding`). The head
   must be a real tensor — the probe's Phase-2 CE backward runs through
   `model.lm_head(...)`. The alias is resolved through transformers'
   own `get_input_embeddings()` / `get_output_embeddings()` accessors, so
   no embedding name is hardcoded here and the VL-wrapper prefix resolves
   exactly like the plain one.

2. **Non-quantizability** (:func:`lm_head_is_tied_alias`). A tied head is
   the *same storage* as the embedding, and the embedding is part of the
   non-quantizable floor (`footprint.py`). Re-encoding the head would
   therefore silently re-encode every token embedding — an error no
   PrismaQuant measurement observes (probe/cost see only the head's own
   output MSE, never the embed-side error injected into layer 0's input),
   which by design principle #1 makes the candidate *illegal* rather than
   merely expensive. It is also unrepresentable downstream: the source
   manifest has no `lm_head.weight` bytes to move out of the floor
   (`footprint.resolve_reencoded_source_bytes` raises on the unresolved
   name), and the exporter's shared-storage handling
   (`_clone_shared_storage_for_safetensors`) would *add* a head tensor the
   source never had. So a tied head is structurally passthrough-only, in
   the same sense as `BF16`/`FP8_SOURCE` (design principle #11): the only
   thing that exists to ship is the source tensor itself.

Detection is from the *config declaration* (`tie_word_embeddings`, at the
top level or in `text_config`), and the alias is only asserted when the
declaration is combined with the absence of a head tensor in the index —
a declared-tied checkpoint that nonetheless ships `lm_head.weight` has an
independent head and is treated as untied here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


__all__ = [
    "config_declares_tied_embeddings",
    "source_has_lm_head_tensor",
    "lm_head_is_tied_alias",
    "resolve_tied_output_embedding",
]


def _cfg_sections(cfg: Any) -> list[Any]:
    """The config and its text sub-config (VL wrappers declare the tie on
    both, but some declare it only on the text config)."""
    out = [cfg]
    if isinstance(cfg, dict):
        sub = cfg.get("text_config")
    else:
        sub = getattr(cfg, "text_config", None)
    if sub is not None:
        out.append(sub)
    return out


def config_declares_tied_embeddings(cfg: Any) -> bool:
    """True when `tie_word_embeddings` is declared on the config or on its
    `text_config`. Accepts an HF config object or the raw config dict."""
    for section in _cfg_sections(cfg):
        if isinstance(section, dict):
            value = section.get("tie_word_embeddings")
        else:
            value = getattr(section, "tie_word_embeddings", None)
        if bool(value):
            return True
    return False


def _load_config_dict(model_path: str) -> dict:
    with open(Path(model_path) / "config.json") as f:
        return json.load(f)


def _source_tensor_keys(model_path: str) -> set[str]:
    """Every tensor key in the source checkpoint, from the safetensors
    index when present, else from the shard headers."""
    src = Path(model_path)
    idx_path = src / "model.safetensors.index.json"
    if idx_path.exists():
        with open(idx_path) as f:
            return set(json.load(f).get("weight_map", {}))
    keys: set[str] = set()
    from safetensors import safe_open
    for name in sorted(os.listdir(src)):
        if not name.endswith(".safetensors"):
            continue
        with safe_open(str(src / name), framework="pt") as sf:
            keys.update(sf.keys())
    return keys


def source_has_lm_head_tensor(model_path: str, lm_head_name: str) -> bool:
    """True when the checkpoint ships an independent head weight.

    Matches the profile-declared head name either exactly or as a suffix
    component, so a wrapper prefix (`model.language_model.lm_head.weight`)
    counts the same as the bare `lm_head.weight`.
    """
    target = f"{lm_head_name}.weight"
    for key in _source_tensor_keys(model_path):
        if key == target or key.endswith("." + target):
            return True
    return False


def lm_head_is_tied_alias(model_path: str, *, profile) -> bool:
    """True when this checkpoint's LM head is an alias of the embedding.

    The precise condition: the config declares `tie_word_embeddings` AND
    the source ships no head tensor of its own. Such a head has no source
    bytes to re-encode and shares storage with the (non-quantizable)
    embedding, so it must be excluded from cost measurement and from the
    allocator's budget.
    """
    try:
        cfg = _load_config_dict(model_path)
    except (OSError, ValueError):
        return False
    if not config_declares_tied_embeddings(cfg):
        return False
    return not source_has_lm_head_tensor(model_path, profile.lm_head_name())


def _module_qname(root, target) -> str:
    for name, mod in root.named_modules():
        if mod is target:
            return name or "<root>"
    return "<unknown>"


def resolve_tied_output_embedding(model, *, log_prefix: str = "[tied]") -> bool:
    """Alias a meta output head onto the materialized input embedding.

    Returns True when an alias was installed. Raises when the head is on
    meta but the config does not declare a tie (that is a genuinely
    missing tensor, not a tie — failing here beats a
    `Cannot copy out of meta tensor` thousands of lines downstream).
    """
    get_out = getattr(model, "get_output_embeddings", None)
    out_emb = get_out() if callable(get_out) else None
    if out_emb is None:
        return False
    out_w = getattr(out_emb, "weight", None)
    if out_w is None or not out_w.is_meta:
        return False

    out_name = _module_qname(model, out_emb)
    if not config_declares_tied_embeddings(getattr(model, "config", None)):
        raise RuntimeError(
            f"{log_prefix} output embedding `{out_name}.weight` is still on "
            "meta after head materialization and the config does NOT declare "
            "`tie_word_embeddings` — the checkpoint is missing its LM head "
            "weight. Refusing to continue; every downstream forward would "
            "fail with 'Cannot copy out of meta tensor'.")

    get_in = getattr(model, "get_input_embeddings", None)
    in_emb = get_in() if callable(get_in) else None
    in_w = getattr(in_emb, "weight", None) if in_emb is not None else None
    if in_w is None or in_w.is_meta:
        raise RuntimeError(
            f"{log_prefix} config declares `tie_word_embeddings` but the "
            f"input embedding is unavailable or still on meta, so "
            f"`{out_name}.weight` cannot be resolved. Input embedding: "
            f"{in_emb!r}.")

    out_emb.weight = in_w
    print(f"{log_prefix} tied `{out_name}.weight` -> "
          f"`{_module_qname(model, in_emb)}.weight` "
          f"(shared storage, {tuple(in_w.shape)} {in_w.dtype} "
          f"on {in_w.device})", flush=True)
    return True
