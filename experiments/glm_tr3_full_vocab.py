"""Sealed TR3 final-panel experiment; raw logits, full-vocabulary FP64 KL.

This module is deliberately outside the pricing package and gold v1/v2 tools.
It never samples inputs or feeds final-panel measurements to allocation.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

PANEL_SHA256 = "35f0c5c973be614f29db757e9bd4bce407ea218b974a8407ec7e64c571aad72b"
DATASET_REVISION = "95f4fdd94bf29989db2e0d1054e4931f55edb6aa"
REFERENCE_REVISION = "a6c167b62691b2bac901344b65cb651a70f53e43"
TOKENIZER_SHA256 = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
VOCAB_SIZE, CONTEXT_LENGTH, WINDOW_COUNT = 154880, 2048, 25


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bound_json(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"JSON digest mismatch: {path}")
    return json.loads(raw)


def cached_checkpoint_identity(model, cache):
    """Use the existing cache, refusing any shard that would require rehashing."""
    from prismaquant.cost_streaming import (
        _local_checkpoint_shards, _read_source_checkpoint_digest_cache,
        _streamed_identity_stat_fingerprint, canonical_fingerprint_key,
        build_source_checkpoint_identity,
    )
    _, shards = _local_checkpoint_shards(model)
    extras = list(Path(model).rglob("*.safetensors"))
    required = {p.resolve() for p in list(shards or ()) + extras}
    entries = _read_source_checkpoint_digest_cache(Path(cache))
    if not required or any(canonical_fingerprint_key(_streamed_identity_stat_fingerprint(p)) not in entries
                           for p in required):
        raise ValueError("checkpoint digest cache is incomplete/stale; prepare authenticated identity before launch")
    return build_source_checkpoint_identity(model, extra_shard_paths=extras, digest_cache_path=cache)


def load_panel(path, *, arrays_root=None):
    """Authenticate handoff bytes and load each NPY from its hashed buffer."""
    panel = bound_json(path, PANEL_SHA256)
    required = {"schema": "prismaquant.exl3-final-panel-handoff.v1",
                "dataset_revision": DATASET_REVISION,
                "reference_model": "zai-org/GLM-5.3-Flash-BF16",
                "reference_revision": REFERENCE_REVISION,
                "tokenizer_sha256": TOKENIZER_SHA256, "vocab_size": VOCAB_SIZE,
                "context_length": CONTEXT_LENGTH, "window_count": WINDOW_COUNT,
                "prediction_positions_per_window": CONTEXT_LENGTH - 1,
                "total_prediction_positions": WINDOW_COUNT * (CONTEXT_LENGTH - 1)}
    if any(panel.get(k) != v for k, v in required.items()):
        raise ValueError("sealed final-panel contract mismatch")
    def array(original, digest, size=None):
        p = Path(original) if arrays_root is None else Path(arrays_root) / Path(original).name
        raw = p.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest or (size is not None and len(raw) != size):
            raise ValueError(f"panel array digest/size mismatch: {p}")
        return np.load(io.BytesIO(raw), allow_pickle=False)
    mask = array(panel["causal_mask_array"], panel["causal_mask_sha256"])
    if mask.dtype != np.uint8 or mask.shape != (CONTEXT_LENGTH,) or not np.all(mask == 1):
        raise ValueError("panel must use the original unpadded causal input")
    windows, inputs = panel["windows"], []
    if len(windows) != WINDOW_COUNT:
        raise ValueError("final-panel window count mismatch")
    for index, window in enumerate(windows):
        if (window["window_id"] != f"final-{index:04d}" or window["role"] != "final"
                or window["prediction_positions"] != CONTEXT_LENGTH - 1
                or window["attention_mask_sha256"] != panel["causal_mask_sha256"]
                or window["tokens_sha256"] != window["panel_token_ids_sha256"]):
            raise ValueError("final-panel window order/role/causal contract mismatch")
        tokens = array(window["tokens_path"], window["tokens_sha256"], window["tokens_bytes"])
        if (tokens.dtype != np.dtype("<i4") or tokens.shape != (CONTEXT_LENGTH,)
                or np.any(tokens < 0) or np.any(tokens >= panel["maximum_token_id_exclusive"])):
            raise ValueError("invalid final-panel token array")
        inputs.append(torch.from_numpy(tokens.astype(np.int64)).unsqueeze(0))
    if len({w["document_id"] for w in windows}) != 4:
        raise ValueError("final-panel document grouping changed")
    return panel, inputs


def token_kl(teacher, candidate, *, tile_rows=32, require_cuda=True):
    """Match upstream token_kld_chunk on raw logits without CPU vocabulary work."""
    if (teacher.ndim != 2 or teacher.shape != candidate.shape or teacher.shape[1] <= 1
            or teacher.shape[0] <= 0 or type(tile_rows) is not int or tile_rows <= 0):
        raise ValueError("teacher/candidate logit geometry or tile size mismatch")
    if teacher.device != candidate.device or (require_cuda and teacher.device.type != "cuda"):
        raise ValueError("KL requires co-resident CUDA tensors")
    result = torch.empty(teacher.shape[0], dtype=torch.float64, device=teacher.device)
    for start in range(0, teacher.shape[0], tile_rows):
        end = min(start + tile_rows, teacher.shape[0])
        t, c = teacher[start:end].to(torch.float64), candidate[start:end].to(torch.float64)
        if not bool(torch.isfinite(t).all() & torch.isfinite(c).all()):
            raise ValueError("teacher/candidate logits must be finite")
        t = torch.log_softmax(t, dim=-1)
        c = torch.log_softmax(c, dim=-1)
        result[start:end] = (t.exp() * (t - c)).sum(dim=-1, dtype=torch.float64)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("nonfinite full-vocabulary KL")
    return result


LOGITS_LAYOUTS = ("legacy_single", "vllm_v2_chunk1024")


def prompt_logit_shapes(rows, vocab_size, logits_layout):
    if logits_layout == "legacy_single":
        return [(1, vocab_size), (rows, vocab_size)]
    if logits_layout == "vllm_v2_chunk1024" and rows == CONTEXT_LENGTH - 1:
        # The native V2 runner samples first, computes all 2048 prompt rows
        # in fixed 1024-row chunks, then drops the last prompt score.
        return [(1, vocab_size), (1024, vocab_size), (1024, vocab_size)]
    raise ValueError("unsupported prompt logits layout/geometry")


class PromptLogitsCapture:
    """One armed request with an explicit native prompt-logits layout.

    Stock vLLM may expose gathered logits on every TP rank. Only rank zero
    scores; every rank reports its call geometry so ownership is checkable.
    A non-owner may see None from a gather-to-owner implementation. Any partial
    vocabulary, undeclared chunking, missing or extra call refuses. The V2
    logits chunks are independent of scheduler chunked prefill, which stays off.
    """
    def __init__(self, *, rank, world_size, rows, vocab_size, tile_rows=32,
                 require_cuda=True, logits_layout="legacy_single"):
        if (type(rank) is not int or type(world_size) is not int or world_size < 1
                or not 0 <= rank < world_size or rows <= 1 or vocab_size <= 1):
            raise ValueError("invalid prompt hook topology/geometry")
        self.rank, self.world_size = rank, world_size
        self.rows, self.vocab_size = rows, vocab_size
        self.tile_rows, self.require_cuda = tile_rows, require_cuda
        self.logits_layout = logits_layout
        self.expected_calls = prompt_logit_shapes(rows, vocab_size, logits_layout)
        self.window_id = None
        self.teacher = None
        self.calls = []
        self.values = None
        self.target_ids = self.target_logprobs = None
        self.next_index = 0

    def arm(self, index, window_id, teacher, target_ids=None):
        if self.window_id is not None or index != self.next_index:
            raise ValueError("hook request order or outstanding request mismatch")
        if self.rank == 0:
            if teacher is None or tuple(teacher.shape) != (self.rows, self.vocab_size):
                raise ValueError("owner requires the complete resident teacher window")
            if self.require_cuda and teacher.device.type != "cuda":
                raise ValueError("teacher must be resident on CUDA before request")
        elif teacher is not None:
            raise ValueError("only TP rank zero owns the teacher")
        if target_ids is not None and (self.rank != 0 or tuple(target_ids.shape) != (self.rows,)
                                      or target_ids.device != teacher.device):
            raise ValueError("prompt target IDs must be co-resident on the owner")
        self.window_id, self.teacher = window_id, teacher
        self.calls, self.values = [], None
        self.target_ids, self.target_logprobs = target_ids, None

    def __call__(self, _module, _args, output):
        if self.window_id is None:
            raise ValueError("unarmed prompt-logits hook call")
        if output is None and self.rank != 0:
            self.calls.append(None)
        else:
            if not isinstance(output, torch.Tensor) or output.ndim != 2:
                raise ValueError("prompt logits must be a tensor on the owner")
            shape = tuple(output.shape)
            call_index = len(self.calls)
            if self.logits_layout == "legacy_single":
                if shape not in self.expected_calls:
                    raise ValueError("chunked prompt or partial vocabulary refused")
                if shape in self.calls:
                    raise ValueError("duplicate prompt/sample logit call")
                start, stop = 0, self.rows
            else:
                if call_index >= len(self.expected_calls) or shape != self.expected_calls[call_index]:
                    raise ValueError(f"native logits call {call_index} has unexpected shape {shape}")
                start = max(0, call_index - 1) * 1024
                stop = min(start + shape[0], self.rows)
            self.calls.append(shape)
            if shape[0] != 1 and self.rank == 0:
                # The final V2 chunk includes the last prompt position, whose
                # next token is outside the sealed teacher. Never score it.
                logits = output[:stop - start]
                values = token_kl(self.teacher[start:stop], logits, tile_rows=self.tile_rows,
                                  require_cuda=self.require_cuda).cpu().tolist()
                self.values = (self.values or []) + values
                if self.target_ids is not None:
                    count = stop - start
                    target_lps = torch.empty(count, device=output.device, dtype=torch.float32)
                    for first in range(0, count, self.tile_rows):
                        last = min(first + self.tile_rows, count)
                        lp = torch.log_softmax(logits[first:last].float(), dim=-1)
                        target_lps[first:last] = lp.gather(
                            1, self.target_ids[start + first:start + last, None]).squeeze(1)
                    self.target_logprobs = (self.target_logprobs or []) + target_lps.cpu().tolist()
        if len(self.calls) > len(self.expected_calls):
            raise ValueError("unexpected extra logit call")
        # Returning None leaves stock model output unchanged.

    def finish(self, window_id):
        if window_id != self.window_id or len(self.calls) != len(self.expected_calls):
            raise ValueError("missing prompt/sample call or wrong window completion")
        if self.rank == 0 and (self.values is None or None in self.calls):
            raise ValueError("TP owner did not observe full prompt logits")
        if self.rank != 0 and self.values is not None:
            raise ValueError("duplicate TP score owner")
        if None in self.calls and self.calls != [None] * len(self.expected_calls):
            raise ValueError("inconsistent TP gather ownership")
        result = {"rank": self.rank, "world_size": self.world_size,
                  "window_id": window_id, "calls": self.calls, "values": self.values,
                  "target_logprobs": self.target_logprobs, "logits_layout": self.logits_layout}
        self.window_id = self.teacher = self.values = None
        self.target_ids = self.target_logprobs = None
        self.next_index += 1
        return result


def collect_tp_result(results, *, window_id, world_size, rows, vocab_size,
                      logits_layout="legacy_single"):
    if len(results) != world_size or sorted(r["rank"] for r in results) != list(range(world_size)):
        raise ValueError("missing or duplicate TP rank evidence")
    expected_calls = prompt_logit_shapes(rows, vocab_size, logits_layout)
    for r in results:
        calls = [None if x is None else tuple(x) for x in r["calls"]]
        complete = ((sorted(calls) == expected_calls if logits_layout == "legacy_single"
                     else calls == expected_calls) if None not in calls else False)
        if (r["window_id"] != window_id or r["world_size"] != world_size
                or r.get("logits_layout", "legacy_single") != logits_layout
                or not (complete or (r["rank"] != 0 and calls == [None] * len(expected_calls)))
                or (r["rank"] != 0 and r["values"] is not None)):
            raise ValueError("TP ownership or prompt geometry evidence mismatch")
    values = next(r["values"] for r in results if r["rank"] == 0)
    if values is None or len(values) != rows or not np.isfinite(values).all():
        raise ValueError("missing/nonfinite owner KL vector")
    return values


def summarize_panel(panel, vectors):
    if len(vectors) != len(panel["windows"]):
        raise ValueError("panel summary requires every ordered window")
    windows, domains, documents = [], {}, {}
    for window, values in zip(panel["windows"], vectors):
        a = np.asarray(values, dtype=np.float64)
        if a.shape != (window["prediction_positions"],) or not np.isfinite(a).all():
            raise ValueError("invalid per-position result vector")
        row = {k: window[k] for k in ("window_id", "domain", "document_id")}
        row.update(positions=int(a.size), mean=float(a.mean()), sum=float(a.sum()),
                   min=float(a.min()), max=float(a.max()))
        windows.append(row)
        domains.setdefault(row["domain"], []).extend(a.tolist())
        documents.setdefault(row["document_id"], []).extend(a.tolist())
    def grouped(groups):
        return {key: {"positions": len(values), "mean": float(np.mean(values))}
                for key, values in groups.items()}
    return {"windows": windows, "domains": grouped(domains), "documents": grouped(documents),
            "mean": float(np.mean([v for vec in vectors for v in vec])),
            "interpretation": "Sealed final-panel benchmark; correlated positions within the reported documents; no independent-token significance or broad generalization claim."}
