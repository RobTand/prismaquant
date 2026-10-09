#!/usr/bin/env python3
"""G3: offline decoded-forward KL of a shipped Tessera artifact, on the sealed TR3 panel.

The tr3-teacher-04 streamed transformers runner (experiments/build_glm_tr3_teacher.py, PQ
7882eda3) is reused, not modified: the same build_streamed_causal_lm call, the same visit_panel,
the same consume geometry, the same torch.profiler first-layer trace and the same
CaptureObserver/Netdata telemetry.  The only addition is a proxy around the runner's
visit_layer_batches: once a layer's source weights are resident and before its first window
runs, every priced Tessera Linear of that layer is swapped for the decode of the shipped wire.

Arms:
  null    source weights through the glue: every installed slice is hash-checked against the
          payload's source identity; nothing is substituted.  Fidelity (i).
  a8_w    A8 export (TESSERA_E4M3_K1_R1024 everywhere), weights only.
  a8_wa   A8 weights plus the served A side: fp8_per_token_dynamic QDQ on every Tessera GEMM input.
  t8r_w   T8R export (the release pick), weights only.
  t8r_wa  T8R weights plus the served A side.
  a8_shared_src_{w,wa}   A8 pick with the shared experts kept as source BF16 (passthrough).
  a8_routed_only_{w,wa}  A8 pick on the routed experts only; shared and dense kept as source BF16.
          A passthrough unit is the installed source slice (verified by the layer's source gate,
          never written by an earlier arm; plan_arms refuses an order that would need it back)
          and gets no activation QDQ in the _wa arm: serving runs it as a BF16 GEMM on BF16
          activations.  These four run only in a multi-arm pass.

Byte integrity (every used unit, every arm): a source slice actually used must hash to the payload's
source_weight.content_sha256; the decoded tensor,
cast to bf16, must hash to the payload's rendered_weight.content_sha256 for the arm's format.
Every hash of a layer is verified before that layer's first window runs; any mismatch aborts
the run with no result.

Scoring: raw fp32 logits of positions 0..2046, KL(teacher || candidate) in fp64 over all
154,880 columns, unmasked (glm_tr3_full_vocab.token_kl), against tr3-teacher-04 and against
TEACHER2 (tr3-teacher-exl3ref-01), each window's teacher array sha256-verified before use.

--pilot (G0): no forward.  Decode T8R layer 5's routed units and all of layer 10 on CUDA,
hash-gate each, time the read/decode/copy/hash legs, check the framing restatement against
tessera.fused.parse_fused, and check the CUDA QDQ against the pinned test's
_kernel_arithmetic on its own input classes.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PQ = Path(os.environ.get("G3_PQ_ROOT", "/pq"))
TESSERA_SRC = os.environ.get("TESSERA_SRC", "/tessera/src")
sys.path.insert(0, TESSERA_SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from g3_pq_policy import g3_exl3
import g3_lib as L  # noqa: E402
from g3_residency import read_g3_input, receipt as residency_receipt
from g3_readset import SourceReads, source_indices
# G3 uses this checkout's shared policy files at its harness seams.
# The pinned numerical package needs no module overlay.
from g3_pq_policy.digests import file_sha256hex as sha256_file
from g3_pq_policy.dev_mode import dev_mode_enabled, seal_check, NOT_COMPUTED, dev_stamp
from g3_pq_policy.digests import bytes_sha256hex, JsonProfile
from g3_pq_policy.io_engine import ENGINE

G3_JSON = JsonProfile("g3-result", ensure_ascii=True, allow_nan=True,
                      default=str, separators=(",", ": "), indent=1)

T0 = time.time()
ARMS = {"null": (None, False), "a8_w": ("a8", False), "a8_wa": ("a8", True),
        "t8r_w": ("t8r", False), "t8r_wa": ("t8r", True),
        # the pre-registered law-fit pick (out9/lawfit_total_pick.csv, sha 2a1c7dda); its wires come
        # from the T8R export, the A8 export or the census wire cache, per row (manifest v2 "root")
        "lawfit_w": ("lawfit", False), "lawfit_wa": ("lawfit", True),
        # placement arms (codec-decomp 2026-09-30): the A8 pick with some unit kinds kept as the
        # source BF16 tensor (passthrough), which serving runs as a BF16 GEMM on BF16 activations,
        # so the W+A arm puts no activation QDQ on them.  Multi-pass only.
        "a8_shared_src_w": ("a8", False), "a8_shared_src_wa": ("a8", True),
        "a8_routed_only_w": ("a8", False), "a8_routed_only_wa": ("a8", True)}
#: unit kinds an arm keeps as source BF16 passthrough (not substituted, no W+A hooks)
SRC_KINDS = {"a8_shared_src_w": frozenset({"shared"}), "a8_shared_src_wa": frozenset({"shared"}),
             "a8_routed_only_w": frozenset({"shared", "dense"}),
             "a8_routed_only_wa": frozenset({"shared", "dense"})}


def arm_wants(arm, rows):
    """Per row, the identity the arm runs on: the arm's rendered sha256, or None for the source
    slice (the null arm, a unit kind the arm keeps as passthrough, or a pick row whose format is
    SOURCE)."""
    key, keep = ARMS[arm][0], SRC_KINDS.get(arm, frozenset())
    return [None if key is None or r["kind"] in keep or r.get(key + "_format") == "SOURCE"
            else r[key + "_rendered_sha256"] for r in rows]


# ------------------------------------------------------------------ manifest v3 picks
#: activation contracts (pinned runtime_contract.json lane_eligibility sm_121 "executes", copied
#: per format into manifest v3) that serving runs on unquantized 16-bit activations: no W+A hook
UNQUANTIZED_CONTRACTS = frozenset({"bf16_unquantized", "source_bf16"})
QUANTIZED_CONTRACT = "fp8_per_token_dynamic"
EXL3_FORMAT = "EXL3"


def register_picks(m):
    """Manifest v3: every named pick becomes two arms, NAME_w and NAME_wa, keyed "pick_NAME".
    Per row the pick's format selects rendered identity, wire and activation contract from the
    row's formats table ("SOURCE" = the source slice, no wire, 16-bit activations)."""
    names = m.get("picks") or []
    for name in names:
        key = "pick_" + name
        for arm, wa in ((name + "_w", False), (name + "_wa", True)):
            if arm in ARMS:
                raise SystemExit(f"pick arm {arm} collides with a built-in arm")
            ARMS[arm] = (key, wa)
        for r in m["rows"]:
            fmt = r["pick"][name]
            r[key + "_format"] = fmt
            if fmt == "SOURCE":
                r[key + "_contract"] = "source_bf16"
                continue
            ent = r["formats"][fmt]
            r[key + "_rendered_sha256"] = ent["rendered_sha256"]
            r[key + "_rendered_shape"] = ent["rendered_shape"]
            r[key + "_wire"] = ent["wire"]
            r[key + "_contract"] = ent["contract"]
    return names


def hook_skip(arm, rows):
    """What a W+A arm leaves unquantized on one layer: the placement arms' SRC_KINDS, or, for a
    pick arm, every (kind, role) whose contract runs 16-bit activations ("routed" for the whole
    routed stack, which must carry one contract per layer).  Refuses a contract G3 has no
    activation model for (e2m1 static, EXL3)."""
    if arm in SRC_KINDS:
        return SRC_KINDS[arm]
    key = ARMS[arm][0]
    if key is None or not key.startswith("pick_"):
        return frozenset()
    by = {}
    for r in rows:
        by.setdefault("routed" if r["kind"] == "routed" else (r["kind"], r["role"]), set()).add(r[key + "_contract"])
    skip = set()
    for unit, cs in sorted(by.items(), key=str):
        if len(cs) != 1:
            raise SystemExit(f"{arm}: {unit} mixes activation contracts {sorted(cs)} within one layer")
        (c,) = cs
        if c in UNQUANTIZED_CONTRACTS:
            skip.add(unit)
        elif c != QUANTIZED_CONTRACT:
            raise SystemExit(f"{arm}: {unit} executes {c!r}, which G3 has no activation model for")
    return frozenset(skip)


def emit_g3_progress(*a):
    print(f"[g3 {time.time() - T0:8.1f}s]", *a, flush=True)




def sha_bytes(buf):
    return bytes_sha256hex(buf)


def write_g3_record(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(G3_JSON.text(obj))
    os.replace(tmp, path)


def source_identity_check(teacher, identity_sha, binding_path, binding_sha, identity, model):
    seal_check('source checkpoint identity', teacher['source_model_identity_sha256'], identity_sha,
               where='G3', refusal=SystemExit("source checkpoint identity differs from teacher-04's"))
    from g3_prepared_source import read_prepared_source_json
    binding = read_prepared_source_json(binding_path, binding_sha)  # Own bytes remain mandatory.
    if dev_mode_enabled():
        # Preserve the pinned owner's actual two-data comparison (capture
        # versus upstream); D32 suspends identity provenance, not this fact.
        for row in binding.get("source_files", []):
            name = row["name"]
            if Path(name).name != name or row["capture_sha256"] != row["upstream_sha256"]:
                raise ValueError("upstream source filename or digest mismatch")
        roster = binding.get("source_files", [])
        if len(roster) != 126 or len({row["name"] for row in roster}) != 126:
            raise ValueError("upstream source roster must contain every original file exactly once")
        seal_check('source reference binding', binding.get('schema'), NOT_COMPUTED, where='G3')
    else:
        from experiments.build_glm_tr3_teacher import require_reference_binding
        require_reference_binding(binding, identity, model)


def run_metadata_stamp():
    return dev_stamp() if dev_mode_enabled() else {}


# ------------------------------------------------------------------ manifest
def read_g3_manifest(path, expected_sha, exports):
    raw = read_g3_input(path)
    got = sha_bytes(raw)
    if expected_sha and got != expected_sha:
        raise SystemExit(f"manifest sha256 {got} != declared {expected_sha}")
    m = json.loads(raw)
    if m["problems"]:
        raise SystemExit(f"manifest carries {len(m['problems'])} problems; refusing")
    seal_check('manifest payload identity', '6f158c986b2aa21f5e41e9960b0403b25efc9ae6f44d7b93b6d294869fd27341',
               m['payload_sha256'], where='G3 manifest', refusal=SystemExit('manifest payload identity differs'))
    by_layer = {}
    for r in m["rows"]:
        by_layer.setdefault(int(r["layer"]), []).append(r)
    want = {**{i: 3 for i in range(3)}, **{i: 867 for i in range(3, 45)}}
    if {k: len(v) for k, v in by_layer.items()} != want:
        raise SystemExit("manifest rows per layer differ from 3 x L0-2, 867 x L3-44")
    return m, by_layer, got


# ------------------------------------------------------------------ hashing pipeline
class HashPool:
    """Bounded byte batches: reserve CPU/GPU staging before copying; drain verifies all slices.

    Callers submit stable installed views, or flush a temporary decode before
    releasing it. No forward/next arm starts until drain verifies every batch.
    """
    def __init__(self, max_inflight_bytes, strict=True, batch_bytes=64 << 20):
        if max_inflight_bytes < 2 or batch_bytes < 1:
            raise ValueError('hash staging bounds must be positive')
        self.pool = ENGINE
        self.cv = threading.Condition()
        self.inflight, self.cap = 0, max_inflight_bytes
        self.batch_cap = min(batch_bytes, max_inflight_bytes // 2)
        self.pending, self.pending_bytes, self.futures = [], 0, []
        self.checked, self.bytes, self.wait_s = 0, 0, 0.0
        self.submitted = 0
        self.strict, self.mismatches = strict, []
        self.profile = {'copies': 0, 'peak_batch_bytes': 0, 'peak_inflight_bytes': 0, 'copy_s': 0.0}

    def _job(self, block, jobs, charged):
        cpu = block.pop("cpu")
        buf = memoryview(cpu.numpy()).cast("B")
        try:
            for start, end, expected, what in jobs:
                got = bytes_sha256hex(buf[start:end])
                if got != expected:
                    if self.strict:
                        raise L.HashGateError(f"{what}: sha256 {got[:16]} != priced {expected[:16]}")
                    self.mismatches.append(what)
            return len(jobs)
        finally:
            del buf, cpu  # drop storage before releasing the reservation
            with self.cv:
                self.inflight -= charged
                self.cv.notify_all()

    def flush(self):
        if not self.pending:
            return
        pending, size = self.pending, self.pending_bytes
        self.pending, self.pending_bytes = [], 0
        charged = 2 * size  # GPU byte pack and CPU byte pack, including contiguous materialization
        t0 = time.perf_counter()
        with self.cv:
            while self.inflight + charged > self.cap:
                self.cv.wait()
            self.inflight += charged
            self.profile['peak_inflight_bytes'] = max(self.profile['peak_inflight_bytes'], self.inflight)
        self.wait_s += time.perf_counter() - t0
        jobs, cur = [], 0
        packed = cpu = block = None
        try:
            device = pending[0][0].device
            packed = torch.empty(size, dtype=torch.uint8, device=device)
            for tensor, expected, what in pending:
                n = tensor.numel() * tensor.element_size()
                # copy_ handles noncontiguous unit views without another device allocation.
                packed[cur:cur + n].view(tensor.dtype).reshape(tensor.shape).copy_(tensor)
                jobs.append((cur, cur + n, expected, what))
                cur += n
            copied_at = time.perf_counter()
            cpu = packed.to('cpu')
            self.profile["max_pack_storage_bytes"] = max(self.profile.get("max_pack_storage_bytes", 0), packed.untyped_storage().nbytes())
            self.profile["max_host_storage_bytes"] = max(self.profile.get("max_host_storage_bytes", 0), cpu.untyped_storage().nbytes())
            self.profile['copy_s'] += time.perf_counter() - copied_at
            self.profile['copies'] += 1
            self.profile['peak_batch_bytes'] = max(self.profile['peak_batch_bytes'], size)
            block = {"cpu": cpu}
            del packed, cpu
            self.futures.append(self.pool.submit(self._job, block, jobs, charged))
        except BaseException:
            packed = cpu = block = None
            with self.cv:
                self.inflight -= charged
                self.cv.notify_all()
            raise

    def submit(self, t, expected, what):
        n = t.numel() * t.element_size()
        if n > self.cap // 2:
            raise ValueError(f'{what}: {n} hash bytes exceed staging bound {self.cap // 2}')
        if self.pending and (self.pending_bytes + n > self.batch_cap or t.device != self.pending[0][0].device
                             or t.dtype != self.pending[0][0].dtype):
            self.flush()
        self.pending.append((t.detach(), expected, what))
        self.submitted += 1
        self.pending_bytes += n
        self.bytes += n
        if n >= self.batch_cap:
            self.flush()

    def drain(self):
        self.flush()
        futures, self.futures = self.futures, []
        count, failure = 0, None
        for f in futures:
            try:
                count += f.result()
            except BaseException as exc:
                failure = failure or exc
        if failure is not None:
            raise failure
        self.checked += count
        return count

    @property
    def all_matched(self):
        return (self.checked == self.submitted and not self.mismatches
                and not self.pending and not self.futures)

    def close(self):
        self.drain()

# ------------------------------------------------------------------ wire read-ahead
class WireReader:
    """Reads one layer's unit blobs on worker threads, at most ``ahead`` layers in advance."""

    def __init__(self, arm_key, by_layer, roots, ahead=1):
        self.arm = arm_key
        self.by_layer = by_layer
        self.roots = roots
        self.pool = ENGINE
        self.pending = {}
        self.ahead = ahead
        self.bytes = 0

    def _read(self, row):
        loc = row[self.arm + "_wire"]
        root = self.roots.get(loc.get("root", self.arm))
        if not root:
            raise SystemExit(f"{row['qname']}: wire root {loc.get('root', self.arm)!r} not given")
        if "ranges" in loc:
            # EXL3: the unit's stored tensors are not contiguous; the wire is their framing
            # suh||svh||trellis||mcg, gated on the pre-pass sha256 before anything decodes it
            blob = b"".join(L.read_range(os.path.join(root, sh), off, n) for sh, off, n in loc["ranges"])
            if len(blob) != loc["member_bytes"] or sha_bytes(blob) != loc["wire_sha256"]:
                raise L.HashGateError(f"{row['qname']}: EXL3 wire ({len(blob)} B) differs from the pre-pass framing")
            return blob
        off, n = L.member_location(loc)
        return L.read_range(os.path.join(root, loc["shard"]), off, n)

    def schedule(self, layer):
        if layer in self.by_layer and layer not in self.pending:
            self.pending[layer] = [self.pool.submit(self._read, r) for r in self.by_layer[layer]]

    def take(self, layer):
        self.schedule(layer)
        for nxt in range(layer + 1, layer + 1 + self.ahead):
            self.schedule(nxt)
        return self.pending.pop(layer)

    def next_layer(self, layer):
        """The first layer after ``layer`` this reader has rows for, or None."""
        return min((x for x in self.by_layer if x > layer), default=None)

    def close(self):
        try:
            for futures in self.pending.values():
                for future in futures:
                    future.result()
        finally:
            self.pending.clear()


# ------------------------------------------------------------------ the substitution proxy
class Substitution:
    def __init__(self, runner, arm, by_layer, roots, decoder, out, profile_layer):
        self.runner = runner
        self.arm = arm
        self.arm_key, self.wa = ARMS[arm]
        self.by_layer = by_layer
        self.hash = HashPool(max_inflight_bytes=2 << 30)
        self.decode_rows = {layer: [r for r, want in zip(rows, arm_wants(arm, rows)) if want is not None]
                            for layer, rows in by_layer.items()}
        self.reader = WireReader(self.arm_key, self.decode_rows, roots) if self.arm_key else None
        self.decoder = decoder
        self.out = out
        self.profile_layer = profile_layer
        self.records = []
        self.remove_hooks = None
        self.source_checked = 0
        self.rendered_checked = 0
        self.copied = 0
        self.layer_started = None
        self.profile_error = None

    def _substitute(self, layer):
        mod = self.runner.layers[layer]
        rows = self.by_layer[layer]
        rec = {"layer": layer, "units": len(rows)}
        t = time.perf_counter()
        futures = self.reader.take(layer) if self.reader else None
        views = [L.unit_view(mod, r) for r in rows]
        # Only source bytes a requested forward consumes are read/hash-checked.
        needed = source_indices([self.arm], rows)
        for i in needed:
            self.hash.submit(views[i], rows[i]['source_sha256'], rows[i]['qname'] + ' [used source]')
        rec['source_hashes'] = self.hash.drain()
        torch.cuda.synchronize()
        rec['source_copy_s'] = time.perf_counter() - t
        self.source_checked += len(needed)
        if self.reader:
            t2 = time.perf_counter()
            wait_s = dec_s = 0.0
            nbytes = 0
            selected = [(r, v) for r, v, want in zip(rows, views, arm_wants(self.arm, rows)) if want is not None]
            for (r, v), f in zip(selected, futures):
                tw = time.perf_counter()
                blob = f.result()
                wait_s += time.perf_counter() - tw
                nbytes += len(blob)
                td = time.perf_counter()
                dec = self.decoder(blob, device=v.device).to(torch.bfloat16)
                if tuple(dec.shape) != tuple(v.shape) or dec.dtype != v.dtype:
                    raise L.HashGateError(f"{r['qname']}: decoded {tuple(dec.shape)}/{dec.dtype} "
                                          f"!= executed {tuple(v.shape)}/{v.dtype}")
                dec_s += time.perf_counter() - td
                v.copy_(dec)
                self.hash.submit(v, r[self.arm_key + '_rendered_sha256'], r['qname'] + ' [decoded]')
                del dec, blob
            torch.cuda.synchronize()
            self.reader.bytes += nbytes
            rec.update(wire_bytes=nbytes, wire_wait_s=wait_s, decode_s=dec_s,
                       decode_loop_s=time.perf_counter() - t2)
            self.rendered_checked += len(selected)
            self.copied += len(selected)
        th = time.perf_counter()
        rec["hashes_verified"] = rec["source_hashes"] + self.hash.drain()
        rec["hash_drain_s"] = time.perf_counter() - th
        rec["hash_backpressure_s"] = self.hash.wait_s
        self.hash.wait_s = 0.0
        if self.wa:
            self.remove_hooks = L.install_wa_hooks(mod, tp=L.TP_SERVED)
        rec["substitution_s"] = time.perf_counter() - t
        return rec

    def before(self, layer):
        prof = None
        if layer == self.profile_layer:
            try:
                prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                          torch.profiler.ProfilerActivity.CUDA],
                                              record_shapes=False, profile_memory=False)
                prof.__enter__()
            except Exception as exc:  # a profiler that cannot start must not cost the arm
                emit_g3_progress("substitution profiler did not start:", repr(exc))
                self.profile_error = repr(exc)
                prof = None
        rec = self._substitute(layer)
        if prof is not None:
            prof.__exit__(None, None, None)
            trace = self.out / f"substitution-L{layer}.trace.json"
            prof.export_chrome_trace(str(trace))
            rec["profile_trace"] = trace.name
            rec["profile_top"] = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)
            emit_g3_progress("substitution profile L%d\n%s" % (layer, rec["profile_top"]))
        rec["cuda_allocated"] = torch.cuda.memory_allocated()
        rec["cuda_max_allocated"] = torch.cuda.max_memory_allocated()
        self.layer_started = time.perf_counter()
        return rec

    def after(self, layer, rec):
        rec["forward_s"] = time.perf_counter() - self.layer_started
        if self.remove_hooks is not None:
            self.remove_hooks()
            self.remove_hooks = None
        self.records.append(rec)
        emit_g3_progress(f"layer {layer}: units={rec['units']} sub={rec['substitution_s']:.1f}s "
            f"(src {rec['source_copy_s']:.1f}, dec {rec.get('decode_s', 0):.1f}, wirewait {rec.get('wire_wait_s', 0):.1f}, "
            f"hashdrain {rec['hash_drain_s']:.1f}) fwd={rec['forward_s']:.1f}s verified={rec['hashes_verified']}")


class RunnerProxy:
    """What visit_panel sees: the real runner, with the substitution around each layer's visit."""

    def __init__(self, runner, sub, on_last_layer=None):
        self._runner = runner
        self._sub = sub
        self._on_last = on_last_layer

    def visit_layer_batches(self, inputs, visitor, *, output_consumer=None):
        last = self._runner.num_layers - 1

        def wrapped(layer, forward_batch):
            rec = self._sub.before(layer)
            if layer == last and self._on_last is not None:
                self._on_last()
            try:
                visitor(layer, forward_batch)
            finally:
                self._sub.after(layer, rec)
        return self._runner.visit_layer_batches(inputs, wrapped, output_consumer=output_consumer)


# ------------------------------------------------------------------ teachers
class Teachers:
    """Bound teacher manifests; window arrays read ahead (depth 2) and sha256-verified."""

    def __init__(self, specs, window_ids, repeats=1):
        self.specs = []
        for name, root, expected in specs:
            raw = read_g3_input(Path(root) / 'teacher.json')
            if sha_bytes(raw) != expected:
                raise SystemExit(f"teacher {name}: teacher.json sha256 differs from {expected}")
            j = json.loads(raw)
            wins = {w["window_id"]: w for w in j["windows"]}
            if [w for w in window_ids if w not in wins]:
                raise SystemExit(f"teacher {name} lacks panel windows")
            self.specs.append((name, Path(root), expected, wins, j))
        self.window_ids = window_ids
        self.total = len(window_ids) * repeats   # a multi-arm pass scores every window once per arm
        self.pool = ENGINE
        self.futures = {}

    def _load(self, name, root, win):
        raw = read_g3_input(root / win['path'], win['bytes'])
        if sha_bytes(raw) != win["sha256"] or len(raw) != win["bytes"]:
            raise SystemExit(f"teacher {name} {win['window_id']}: array bytes differ from its manifest")
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        if arr.dtype != np.float32 or list(arr.shape) != win["shape"]:
            raise SystemExit(f"teacher {name} {win['window_id']}: array geometry")
        data_sha = sha_bytes(memoryview(np.ascontiguousarray(arr)).cast("B"))
        return arr, data_sha

    def start(self, index):
        for i in range(index, min(index + 2, self.total)):
            if i not in self.futures:
                wid = self.window_ids[i % len(self.window_ids)]
                self.futures[i] = [self.pool.submit(self._load, n, r, w[wid]) for n, r, _e, w, _j in self.specs]

    def get(self, index):
        self.start(index)
        out = [f.result() for f in self.futures.pop(index)]
        self.start(index + 1)
        return out

    def close(self):
        try:
            for futures in self.futures.values():
                for future in futures:
                    future.result()
        finally:
            self.futures.clear()


# ------------------------------------------------------------------ pilot (G0)
def pinned_kernel_namespace():
    pin = Path(TESSERA_SRC).parent
    route = ast.parse((pin / "tests" / "test_serving_fp8_route.py").read_text())
    fp8_max = [ast.literal_eval(n.value) for n in route.body if isinstance(n, ast.Assign)
               and any(getattr(t, "id", None) == "FP8_MAX" for t in n.targets)][0]
    tree = ast.parse((pin / "tests" / "test_native_fp8_quant.py").read_text())
    want = {"MIN_SCALE", "_e4m3_positive_values", "_bf16_exact", "_random", "_tiny_amax", "_outliers",
            "_powers_of_two", "MIDPOINT_SCALES", "_midpoints", "CLASSES", "_kernel_arithmetic"}
    body = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in want) or
            (isinstance(n, ast.Assign) and {getattr(t, "id", None) for t in n.targets} & want)]
    ns = {"torch": torch, "FP8_MAX": fp8_max}
    exec(compile(ast.Module(body=body, type_ignores=[]), "pinned", "exec"), ns)
    return ns


def run_pilot(args, m, by_layer, manifest_sha, decoder):
    from tessera.fused import parse_fused
    import tessera
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    res = {"schema": "surrogate-diag.g3.pilot.v1", "manifest_sha256": manifest_sha,
           "tessera": tessera.__file__, "torch": torch.__version__, "cuda": torch.version.cuda,
           "device": torch.cuda.get_device_name(0), "hostname": os.uname().nodename, "argv": sys.argv}
    # CUDA QDQ vs the pinned kernel restatement
    ns = pinned_kernel_namespace()
    qdq = {}
    for name, make in ns["CLASSES"].items():
        x = make().cuda()
        q_k, s_k = ns["_kernel_arithmetic"](x)
        q, s = L.fp8_per_token_dynamic(x)
        qdq[name] = {"codes_differ": int((q.view(torch.uint8) != q_k.view(torch.uint8)).sum()),
                     "scales_differ": int((s != s_k).sum())}
    res["cuda_qdq_vs_pinned_kernel_arithmetic"] = qdq
    emit_g3_progress("qdq", qdq)
    roots = {"t8r": args.t8r_export, "a8": args.a8_export}
    reader = WireReader("t8r", by_layer, roots, ahead=0)
    hashp = HashPool(max_inflight_bytes=2 << 30, strict=False)
    legs = []
    framing = {"containers": 0, "agree": 0}
    for layer, kinds in ((5, {"routed"}), (10, {"routed", "shared"})):
        rows = [r for r in by_layer[layer] if r["kind"] in kinds]
        t = time.perf_counter()
        futs = [reader.pool.submit(reader._read, r) for r in rows]
        blobs = [f.result() for f in futs]
        read_s = time.perf_counter() - t
        nbytes = sum(len(b) for b in blobs)
        # framing restatement vs tessera on each distinct container of the layer
        seen = set()
        for r in rows:
            loc = r["t8r_wire"]
            if loc["tensor"] in seen:
                continue
            seen.add(loc["tensor"])
            if len(seen) > 40 and r["kind"] == "routed":
                continue
            data = L.read_range(os.path.join(roots["t8r"], loc["shard"]), loc["offset"], loc["length"])
            framing["containers"] += 1
            framing["agree"] += int(L.unwrap_members(data) == {x.name: x.blob for x in parse_fused(data)})
        torch.cuda.synchronize()
        dec_times = []
        t = time.perf_counter()
        copy_s = 0.0
        for r, blob in zip(rows, blobs):
            td = time.perf_counter()
            dec = decoder(blob, device="cuda").to(torch.bfloat16)
            torch.cuda.synchronize()
            dec_times.append(time.perf_counter() - td)
            tc = time.perf_counter()
            hashp.submit(dec, r["t8r_rendered_sha256"], r["qname"])
            copy_s += time.perf_counter() - tc
            del dec
        loop_s = time.perf_counter() - t
        th = time.perf_counter()
        verified = hashp.drain()
        drain_s = time.perf_counter() - th
        total = time.perf_counter() - t
        leg = {"layer": layer, "kinds": sorted(kinds), "units": len(rows), "wire_bytes": nbytes,
               "read_s": read_s, "decode_loop_s": loop_s, "decode_s_sum": sum(dec_times),
               "decode_s_p50": float(np.median(dec_times)), "decode_s_max": max(dec_times),
               "to_host_s": copy_s, "hash_drain_s": drain_s, "hashes_verified": verified,
               "decode_plus_hash_s": total, "decoded_bytes": hashp.bytes,
               "hash_mismatches": len(hashp.mismatches), "hash_mismatch_examples": hashp.mismatches[:10],
               "formats": sorted({r["t8r_format"] for r in rows}),
               "cuda_max_allocated": torch.cuda.max_memory_allocated()}
        hashp.bytes = 0
        hashp.mismatches = []
        legs.append(leg)
        emit_g3_progress("pilot leg", {k: v for k, v in leg.items() if k != "formats"})
    res["legs"] = legs
    res["framing_vs_parse_fused"] = framing
    units = sum(x["units"] for x in legs)
    per_unit = sum(x["decode_plus_hash_s"] for x in legs) / units
    per_byte_read = sum(x["read_s"] for x in legs) / sum(x["wire_bytes"] for x in legs)
    res["extrapolation"] = {
        "units_all": 36423, "decode_plus_hash_s_per_unit": per_unit,
        "decode_plus_hash_s_all_units": per_unit * 36423,
        "wire_read_s_per_gb": per_byte_read * 1e9,
        "note": "only source slices actually used by a requested arm are source-hashed; "
                "this pilot extrapolation is not an end-to-end I/O, residency, power or speed measurement"}
    write_g3_record(res, out / "pilot.json")
    emit_g3_progress("pilot extrapolation", res["extrapolation"])
    ok = (all(v["codes_differ"] == 0 and v["scales_differ"] == 0 for v in qdq.values())
          and framing["agree"] == framing["containers"]
          and all(x["hash_mismatches"] == 0 for x in legs))
    res["ok"] = ok
    write_g3_record(res, out / "pilot.json")
    return 0 if ok else 1


# ------------------------------------------------------------------ arm (G3)
def run_arm(args, m, by_layer, manifest_sha, decoder):
    sys.path.insert(0, str(PQ))
    sys.path.insert(0, str(PQ / "tools"))
    sys.path.insert(0, str(PQ / "experiments"))
    import importlib
    teacher_mod = importlib.import_module("experiments.build_glm_tr3_teacher")
    from experiments.glm_tr3_full_vocab import CONTEXT_LENGTH, VOCAB_SIZE, load_panel, token_kl
    from g3_prepared_source import cached_checkpoint_identity
    from tools.build_streamed_full_kl_teacher import (_require_source_execution_policy,
                                                      _source_derivative_policy)
    from tools.full_kl_teacher_payload import canonical_sha256
    from prismaquant.cost_streaming import build_streamed_causal_lm
    from prismaquant.gpu_guard import require_cuda_hot_path
    from prismaquant.joint_aura import source_execution_identity
    from prismaquant.model_profiles import detect_profile
    import transformers

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    panel, inputs = load_panel(args.panel, arrays_root=args.arrays_root)
    window_ids = [w["window_id"] for w in panel["windows"]]
    teachers = Teachers([("teacher04", args.teacher, args.teacher_sha256),
                         ("teacher2", args.teacher2, args.teacher2_sha256)], window_ids)
    t04 = teachers.specs[0][4]
    model = Path(args.model).resolve(strict=True)
    identity = cached_checkpoint_identity(model, args.source_preparation, args.source_preparation_sha256,
                                          stored_identity=t04['source_model_identity'])
    identity_sha = t04['source_model_identity_sha256'] if dev_mode_enabled() else canonical_sha256(identity)
    source_identity_check(t04, identity_sha, args.reference_binding, args.reference_binding_sha256, identity, model)
    policy = _source_derivative_policy(args)
    device = require_cuda_hot_path("g3_offline_decoded_kl", "cuda")
    producer = teacher_mod.producer_identity()
    source_reads = SourceReads(model, [args.arm], by_layer)
    source_reads.install()
    runner = build_streamed_causal_lm(
        str(model), device=device, dtype=torch.bfloat16,
        offload_folder=args.offload_folder, profile=detect_profile(str(model)),
        cache_headroom_gb=args.cache_headroom_gb, max_cache_slots=2,
        prefetch_workers=1, prefetch_lookahead=1, require_prefetched_residency=True,
        attn_implementation="eager", source_derivative=policy,
    )
    result = {"schema": "surrogate-diag.g3.arm.v1", "arm": args.arm, "run_tag": args.run_tag,
              "manifest_sha256": manifest_sha, "hostname": os.uname().nodename,
              "container_content_sha256": os.environ.get("PRISMAQUANT_CONTAINER_CONTENT_SHA256"),
              "torch": torch.__version__, "transformers": transformers.__version__,
              "tessera": sys.modules["tessera"].__file__ if "tessera" in sys.modules else None,
              "source_model_identity_sha256": identity_sha,
              "producer_identity_equals_teacher04": producer == t04["producer_identity"],
              "producer_identity": producer, "argv": sys.argv}
    result.update(run_metadata_stamp())
    kl_rows = {"teacher04": [], "teacher2": []}
    per_window = []
    started = time.monotonic()
    sub = Substitution(runner, args.arm, by_layer,
                       {"t8r": args.t8r_export, "a8": args.a8_export, "wirecache": args.wire_cache},
                       decoder, out, args.profile_layer)
    try:
        if runner.context.max_cache_slots != 2 or runner.prefetch_lookahead != 1 or not runner.require_prefetched_residency:
            raise SystemExit("runner is not teacher-04's two-slot resident prefetch policy")
        runner.model.eval()
        runner.context.begin_source_initialization_audit()
        source_execution = source_execution_identity(runner.model)
        _require_source_execution_policy(source_execution, policy)
        result["source_execution_equals_teacher04"] = source_execution == t04["source_execution"]
        experts_impl = sorted({v.get("experts") for v in source_execution.get("modules", {}).values()})
        result["experts_implementation"] = experts_impl
        result["resident_plan"] = {"estimated_layer_bytes": runner.context.estimated_layer_bytes,
                                   "cache_max_bytes": runner.context.layer_cache.max_bytes,
                                   "prefetch_min_available_bytes": runner.context.prefetch_min_available_bytes,
                                   "num_layers": runner.num_layers}
        if runner.num_layers != 45:
            raise SystemExit(f"runner has {runner.num_layers} decoder layers, manifest 45")
        emit_g3_progress("runner ready", result["resident_plan"], "experts", experts_impl)

        def consume(index, logits):
            if tuple(logits.shape) != (1, CONTEXT_LENGTH, VOCAB_SIZE) or logits.device.type != "cuda":
                raise SystemExit("candidate logits geometry/device")
            raw = logits[0, :-1].float()
            if not bool(torch.isfinite(raw).all()):
                raise SystemExit("candidate emitted nonfinite logits")
            cand_sha = sha_bytes(memoryview(raw.cpu().numpy()).cast("B"))
            loaded = teachers.get(index)
            row = {"window_id": window_ids[index], "candidate_logits_data_sha256": cand_sha}
            for (name, *_rest), (arr, data_sha) in zip(teachers.specs, loaded):
                t = torch.from_numpy(arr).to(device)
                kl = token_kl(t, raw)
                kl_rows[name].append(kl.cpu().numpy())
                top1 = float((t.argmax(dim=-1) == raw.argmax(dim=-1)).double().mean())
                row[name] = {"mean_kl": float(kl.mean()), "top1_agreement": top1,
                             "teacher_logits_data_sha256": data_sha,
                             "logits_bitwise_equal": data_sha == cand_sha,
                             "max_abs_logit_diff": float((t - raw).abs().max())}
                del t, kl
            per_window.append(row)
            emit_g3_progress(f"window {row['window_id']}: KL t04 {row['teacher04']['mean_kl']:.6f} "
                f"t2 {row['teacher2']['mean_kl']:.6f} maxdiff04 {row['teacher04']['max_abs_logit_diff']:.4g}")

        proxy = RunnerProxy(runner, sub, on_last_layer=lambda: teachers.start(0))
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                on_trace_ready=lambda p: p.export_chrome_trace(str(out / "first-layer.trace.json")),
                record_shapes=False, profile_memory=True) as profiler:
            with torch.inference_mode():
                teacher_mod.visit_panel(proxy, inputs, consume, profiler)
        result["source_initialization_contract_sha256"] = canonical_sha256(
            runner.context.source_initialization_contract())
        seal_check('source execution', source_execution, source_execution_identity(runner.model), where='G3',
                   refusal=SystemExit('source execution changed during the run'))
    finally:
        runner.shutdown()
    elapsed = time.monotonic() - started
    for name, rows in kl_rows.items():
        allk = np.concatenate(rows).astype(np.float64)
        np.save(out / f"per_position_kl.{name}.npy", np.stack(rows), allow_pickle=False)
        result[name] = {"mean_kl": float(allk.mean()), "positions": int(allk.size),
                        "per_window_mean": [float(r.mean()) for r in rows],
                        "p99_kl": float(np.quantile(allk, 0.99)), "max_kl": float(allk.max()),
                        "top1_agreement": float(np.mean([w[name]["top1_agreement"] for w in per_window])),
                        "windows_bitwise_equal": sum(w[name]["logits_bitwise_equal"] for w in per_window),
                        "max_abs_logit_diff": max(w[name]["max_abs_logit_diff"] for w in per_window)}
    result["windows"] = per_window
    result["hash_gate"] = {"source_verified": sub.source_checked, "rendered_verified": sub.rendered_checked,
                           "copied": sub.copied, "rows": sum(len(v) for v in by_layer.values()),
                           "all_hashes_matched": sub.hash.all_matched}
    result["layers"] = sub.records
    result['source_readset'] = dict(source_reads.stats, omitted_names=sorted(source_reads.omitted))
    result['residency'] = residency_receipt()
    result['hash_copy_profile'] = dict(sub.hash.profile)
    result["substitution_profile_error"] = sub.profile_error
    result["wire_bytes_read"] = sub.reader.bytes if sub.reader else 0
    result["elapsed_seconds"] = elapsed
    result["cuda_max_allocated"] = torch.cuda.max_memory_allocated()
    return result


# ------------------------------------------------------------------ multi-arm single pass
# One streamed source read serves every arm.  The runner's visit_layer_batches is given the 25
# panel windows once per arm (arm-major), so each arm keeps its own 25 hidden states; per layer
# the visitor source-gates the installed slices once, then for each arm installs that arm's
# weights, runs the arm's 25 windows and removes its A-side hooks.  An arm decodes only the rows
# whose installed rendered identity differs from its own (W and W+A arms share weights; the
# law-fit pick shares 26,850 rendered identities with T8R and 3,460 with A8), so a row it does
# not decode was hash-gated, earlier in the same layer, against the identical identity, and no
# forward writes a weight.

def plan_arms(arms, by_layer):
    """Per layer, per arm: the row indices the arm decodes, and how many rows it carries.

    The null arm runs on the installed source, so it may only come first.  After each arm's
    install every row carries exactly the identity that arm names (arm_wants): its rendered
    identity, or the source slice for the null arm and for the unit kinds a placement arm keeps as
    passthrough.  Nothing restores a source slice once overwritten, so an arm order that would
    need one is refused (run the arms that keep more source first)."""
    if not arms or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise SystemExit(f"arms must be distinct names from {sorted(ARMS)}: {arms}")
    if "null" in arms and arms[0] != "null":
        raise SystemExit("the null arm runs on the installed source and must come first")
    plan, totals = {}, {a: {"decoded": 0, "carried": 0} for a in arms}
    for a in arms:
        if SRC_KINDS.get(a) or (ARMS[a][0] or "").startswith("pick_"):
            totals[a]["source_kept"] = 0
    for layer, rows in sorted(by_layer.items()):
        installed = [None] * len(rows)          # None = the source slice
        steps = []
        for arm in arms:
            key = ARMS[arm][0]
            if key is None:
                steps.append({"arm": arm, "key": None, "decode": []})
                continue
            want = arm_wants(arm, rows)
            restore = [i for i, (have, need) in enumerate(zip(installed, want)) if need is None and have is not None]
            if restore:
                raise SystemExit(f"layer {layer}: arm {arm} keeps {len(restore)} source rows an earlier arm "
                                 "overwrote; order the arms that keep more source first")
            idx = [i for i, (have, need) in enumerate(zip(installed, want)) if have != need]
            for i in idx:
                installed[i] = want[i]
            if installed != want:
                raise SystemExit(f"layer {layer}: arm {arm} would not run on its own identities")
            steps.append({"arm": arm, "key": key, "decode": idx})
            kept = sum(1 for w in want if w is None)
            totals[arm]["decoded"] += len(idx)
            totals[arm]["carried"] += len(rows) - len(idx) - kept
            if kept:
                totals[arm]["source_kept"] = totals[arm].get("source_kept", 0) + kept
        plan[layer] = steps
    return plan, totals


class MultiInstaller:
    """Source gate once per layer; per arm, decode-and-gate the rows the plan names."""

    def __init__(self, runner, arms, by_layer, plan, roots, decoder, decode_threads):
        self.runner, self.arms, self.by_layer, self.plan = runner, arms, by_layer, plan
        self.hash = HashPool(max_inflight_bytes=2 << 30)
        self.decoder = decoder
        self.decode_pool = ENGINE
        self.window = 2 * decode_threads
        self.readers = {}
        for k, arm in enumerate(arms):
            key = ARMS[arm][0]
            rows_by_layer = {layer: [by_layer[layer][i] for i in steps[k]["decode"]]
                             for layer, steps in plan.items() if steps[k]["decode"]}
            if rows_by_layer:
                # ahead=0: an arm's next wires are scheduled by schedule_after, once its current
                # layer's wires are consumed, so each arm holds at most one layer of wire bytes.
                self.readers[arm] = WireReader(key, rows_by_layer, roots, ahead=0)
        self.counts = {a: {"source_verified": 0, "rendered_verified": 0, "rendered_carried": 0, "copied": 0}
                       | ({"source_kept": 0} if SRC_KINDS.get(a) or (ARMS[a][0] or "").startswith("pick_") else {})
                       for a in arms}
        self.records = []
        self.wire_bytes = 0

    def decode_unit(self, key, row, blob, device):
        """Use the shared EXL3 decoder or the pinned Tessera reader for the row's format."""
        if row.get(key + "_format") == EXL3_FORMAT:
            out_f, in_f = row[key + "_rendered_shape"]
            return g3_exl3.decode_wire(blob, in_f, out_f, device=device)
        return self.decoder(blob, device=device)

    def prefetch(self, layer):
        for r in self.readers.values():
            r.schedule(layer)

    def prefetch_first(self):
        """Schedule every decoding arm's first layer."""
        for r in self.readers.values():
            r.schedule(min(r.by_layer))

    def schedule_after(self, layer, k):
        """After arm k installed ``layer``: start reading its next decode layer's wires."""
        r = self.readers.get(self.arms[k])
        if r is not None:
            nxt = r.next_layer(layer)
            if nxt is not None:
                r.schedule(nxt)
        return None

    def pending_layers(self):
        """{arm: sorted layers whose wires are scheduled and not yet taken} (read-ahead bound check)."""
        return {a: sorted(r.pending) for a, r in self.readers.items()}

    def source_gate(self, layer, views):
        rows = self.by_layer[layer]
        t = time.perf_counter()
        needed = source_indices(self.arms, rows)
        for i in needed:
            r, v = rows[i], views[i]
            self.hash.submit(v, r['source_sha256'], r['qname'] + ' [used source]')
        n = self.hash.drain()
        for a in self.arms:
            self.counts[a]['source_verified'] += len(needed)
        return {'source_gate_s': time.perf_counter() - t, 'source_hashes': n,
                'source_unused': len(rows) - len(needed)}

    def install(self, layer, k, views):
        arm = self.arms[k]
        key = ARMS[arm][0]
        idx = self.plan[layer][k]["decode"]
        rows = self.by_layer[layer]
        kept = 0 if key is None else sum(1 for w in arm_wants(arm, rows) if w is None)
        rec = {"arm": arm, "decoded": len(idx), "carried": 0 if key is None else len(rows) - len(idx) - kept}
        if kept:
            rec["source_kept"] = kept
        t = time.perf_counter()
        if idx:
            blobs = self.readers[arm].take(layer)
            if len(blobs) != len(idx):
                raise SystemExit(f"layer {layer} {arm}: {len(blobs)} wires for {len(idx)} planned rows")
            device = views[0].device
            sizes = [0] * len(idx)

            def decode(j):
                blob = blobs[j].result()
                blobs[j] = None                  # release the wire bytes as soon as this unit decodes
                sizes[j] = len(blob)
                return self.decode_unit(key, rows[idx[j]], blob, device).to(torch.bfloat16)

            pending = {}
            wait_s = 0.0
            for j in range(min(self.window, len(idx))):
                pending[j] = self.decode_pool.submit(decode, j)
            for j, i in enumerate(idx):
                tw = time.perf_counter()
                dec = pending.pop(j).result()
                wait_s += time.perf_counter() - tw
                nxt = j + self.window
                if nxt < len(idx):
                    pending[nxt] = self.decode_pool.submit(decode, nxt)
                r, v = rows[i], views[i]
                if tuple(dec.shape) != tuple(v.shape) or dec.dtype != v.dtype:
                    raise L.HashGateError(f"{r['qname']}: decoded {tuple(dec.shape)}/{dec.dtype} "
                                          f"!= executed {tuple(v.shape)}/{v.dtype}")
                v.copy_(dec)
                self.hash.submit(v, r[key + '_rendered_sha256'], f"{r['qname']} [decoded {arm}]")
                del dec
            if device.type == "cuda":
                torch.cuda.synchronize()
            rec["wire_bytes"] = sum(sizes)
            self.wire_bytes += rec["wire_bytes"]
            rec["decode_wait_s"] = wait_s
        th = time.perf_counter()
        rec["hashes_verified"] = self.hash.drain()
        rec["hash_drain_s"] = time.perf_counter() - th
        rec["install_s"] = time.perf_counter() - t
        rec['hash_copy_profile'] = dict(self.hash.profile)
        c = self.counts[arm]
        c["rendered_verified"] += len(idx)
        c["rendered_carried"] += rec["carried"]
        c["copied"] += len(idx)
        if kept:
            c["source_kept"] = c.get("source_kept", 0) + kept  # checked used source, never overwritten
        self.schedule_after(layer, k)
        return rec


class SourcePrefetchKeeper:
    """Keeps the runner's read of the next source layer in flight during the multi-arm pass.

    The runner schedules layer L+1 once, when it installs L, and its scheduler refuses while
    MemAvailable is under its floor (2 x layer bytes); with require_prefetched_residency the
    install of L+1 then fails closed (mp2 9dd35ced, layer 36).  In the one-pass walk layer L-1
    is finished when L's visit starts, so this releases it through the runner's own one-pass
    boundary (release_completed_layer: unload, drop cache ownership, empty the CUDA cache) and
    re-attempts the runner's own schedule_prefetch at the visit start, after every arm, and
    at the visit end with a bounded wait. Scheduling never changes bytes: used
    source slices and candidate decodes are checked before a requested forward.
    The runner's own fail-closed install still refuses an unscheduled read.
    """

    def __init__(self, ctx, num_layers, *, empty_cache, mem_available=None, sleep=time.sleep,
                 wait_s=300.0, poll_s=2.0):
        self.ctx, self.num_layers = ctx, num_layers
        self.empty_cache, self.sleep = empty_cache, sleep
        self.mem_available = mem_available or (lambda: None)
        self.wait_s, self.poll_s = wait_s, poll_s

    def _present(self, nxt):
        with self.ctx._inflight_lock:            # read-only look at the runner's in-flight map
            fut = self.ctx._inflight.get(nxt)
        return bool(self.ctx.layer_cache.peek(nxt)) or fut is not None

    def release(self, layer):
        """Release the finished layer ``layer`` (the walk never returns to it).

        Returns whether the cache still held it (the runner's own unload may already have trimmed it
        under memory pressure, in which case the release frees only the CUDA cache)."""
        if 0 <= layer < self.num_layers:
            held = bool(self.ctx.layer_cache.peek(layer))
            self.ctx.release_completed_layer(layer)
            return held
        return None

    def ensure(self, layer, where, rec, wait=False, schedule=True, record_mem=False):
        """Make layer+1's source read resident or in flight; record who scheduled it.

        schedule=False only records whether it is already there (the runner's own schedule)."""
        nxt = layer + 1
        st = rec.setdefault("src_prefetch", {"next": nxt, "by": None, "attempts": 0, "waited_s": 0.0,
                                             "skips_before": getattr(self.ctx, "prefetch_memory_skips", None),
                                             "mem_available_at": {}})
        if record_mem:
            st["mem_available_at"][where] = self.mem_available()
        if nxt >= self.num_layers:
            st["by"] = "none-needed"
            return True
        t0 = time.monotonic()
        while True:
            if self._present(nxt):
                if st["by"] is None:
                    st["by"] = where
                break
            st["by"] = None                      # an earlier schedule the worker declined
            if not schedule:
                break
            self.empty_cache()
            st["attempts"] += 1
            st["mem_available_at"][where] = self.mem_available()
            self.ctx.schedule_prefetch(nxt)
            # the runner's worker re-checks memory before its read and drops its own in-flight
            # entry if it declines; the pool is idle (lookahead 1), so that happens at once
            self.sleep(0.2)
            if self._present(nxt):
                st["by"] = where
                break
            if not wait or time.monotonic() - t0 >= self.wait_s:
                break
            self.sleep(self.poll_s)
        st["waited_s"] += time.monotonic() - t0
        st["skips_after"] = getattr(self.ctx, "prefetch_memory_skips", None)
        return st["by"] is not None


def visit_layer_arms(layer, mod, arms, inst, views, inputs, forward_batch, *, profile_layer, out, wa_hooks, sync,
                     after_arm=lambda k: None, rows=None):
    """One layer of the multi-arm pass: per arm, install its weights, run every window's forward, record.

    The first quantized arm's install on ``profile_layer`` runs under torch.profiler (trace + table).
    W+A arms run with the served activation hooks, removed before the next arm installs.
    """
    recs = []
    profile_pending = layer == profile_layer
    for k, arm in enumerate(arms):
        prof = None
        if profile_pending and ARMS[arm][0] is not None:
            profile_pending = False
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU] + (
                                              [torch.profiler.ProfilerActivity.CUDA] if torch.cuda.is_available() else []),
                                          record_shapes=False, profile_memory=False)
            prof.__enter__()
        a = inst.install(layer, k, views)
        if prof is not None:
            prof.__exit__(None, None, None)
            trace = out / f"install-L{layer}-{arm}.trace.json"
            prof.export_chrome_trace(str(trace))
            a["profile_trace"] = trace.name
            a["profile_top"] = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
            emit_g3_progress(f"install profile L{layer} {arm}\n{a['profile_top']}")
        keep = hook_skip(arm, rows if rows is not None else []) if ARMS[arm][1] else None
        remove = (wa_hooks(mod, keep) if keep else wa_hooks(mod)) if ARMS[arm][1] else None
        if keep:
            a["hook_skip"] = sorted(map(str, keep))
        tf = time.perf_counter()
        try:
            for tokens in inputs:
                forward_batch(tokens)
            sync()
        finally:
            if remove is not None:
                remove()
        a["forward_s"] = time.perf_counter() - tf
        recs.append(a)
        after_arm(k)
    return recs


def multi_visit_layer(layer, forward_batch, profiler, *, runner, inst, keeper, by_layer, arms, inputs, teachers,
                      last, profile_layer, out, layer_records, unit_view, wa_hooks, sync, max_allocated,
                      reserved=lambda: None):
    """The multi-arm pass's layer visitor: keep the next source read in flight, gate the source once,
    run every arm (visit_layer_arms), and record the layer."""
    mod = runner.layers[layer]
    rec = {"layer": layer}
    keeper.ensure(layer, "runner", rec, schedule=False, record_mem=True)
    rec["released_prev_was_cached"] = keeper.release(layer - 1)
    keeper.ensure(layer, "visit_start", rec, record_mem=True)
    rows = by_layer[layer]
    views = [unit_view(mod, r) for r in rows]
    rec.update({"units": len(rows), **inst.source_gate(layer, views), "arms": []})
    if layer == last:
        teachers.start(0)

    def after_arm(k):
        if profiler is not None and layer == 0 and k == 0:
            profiler.step()          # first-layer trace: setup, layer-0 gate and the first arm
        keeper.ensure(layer, f"after_{arms[k]}", rec)

    rec["arms"] = visit_layer_arms(layer, mod, arms, inst, views, inputs, forward_batch,
                                   profile_layer=profile_layer, out=out, wa_hooks=wa_hooks, sync=sync,
                                   after_arm=after_arm, rows=rows)
    keeper.ensure(layer, "visit_end", rec, wait=True)
    rec["cuda_max_allocated"] = max_allocated()
    rec["cuda_memory_reserved"] = reserved()
    rec["wire_pending_layers"] = inst.pending_layers()
    layer_records.append(rec)
    sp = rec["src_prefetch"]
    emit_g3_progress(f"layer {layer}: src {rec['source_gate_s']:.1f}s | " + " | ".join(
        f"{a['arm']} dec {a['decoded']} {a['install_s']:.1f}s fwd {a['forward_s']:.1f}s" for a in rec["arms"])
        + f" | next-src by {sp['by']} tries {sp['attempts']} wait {sp['waited_s']:.1f}s "
          f"skips {sp.get('skips_before')}->{sp.get('skips_after')} mem {sp['mem_available_at']}")
    return rec


def run_multi(args, arms, m, by_layer, manifest_sha, decoder):
    """All arms in one streamed source pass; one result.json per arm in the single-arm schema."""
    sys.path.insert(0, str(PQ))
    sys.path.insert(0, str(PQ / "tools"))
    sys.path.insert(0, str(PQ / "experiments"))
    import importlib
    teacher_mod = importlib.import_module("experiments.build_glm_tr3_teacher")
    from experiments.glm_tr3_full_vocab import CONTEXT_LENGTH, VOCAB_SIZE, load_panel, token_kl
    from g3_prepared_source import cached_checkpoint_identity
    from tools.build_streamed_full_kl_teacher import (_require_source_execution_policy,
                                                      _source_derivative_policy)
    from tools.full_kl_teacher_payload import canonical_sha256
    from prismaquant.cost_streaming import build_streamed_causal_lm
    from prismaquant.gpu_guard import require_cuda_hot_path
    from prismaquant.joint_aura import source_execution_identity
    from prismaquant.model_profiles import detect_profile
    import transformers

    out = Path(args.out)
    arm_dirs = {a: out.parent / f"{a}-{args.run_tag}" for a in arms}
    for d in arm_dirs.values():
        if d.exists():
            raise SystemExit(f"{d} exists; refusing to overwrite a run")
    plan, totals = plan_arms(arms, by_layer)
    out.mkdir(parents=True, exist_ok=False)
    write_g3_record({"arms": arms, "totals": totals}, out / "plan.json")
    emit_g3_progress("plan", json.dumps(totals))
    panel, inputs = load_panel(args.panel, arrays_root=args.arrays_root)
    window_ids = [w["window_id"] for w in panel["windows"]]
    nwin = len(inputs)
    teachers = Teachers([("teacher04", args.teacher, args.teacher_sha256),
                         ("teacher2", args.teacher2, args.teacher2_sha256)], window_ids, repeats=len(arms))
    t04 = teachers.specs[0][4]
    model = Path(args.model).resolve(strict=True)
    identity = cached_checkpoint_identity(model, args.source_preparation, args.source_preparation_sha256,
                                          stored_identity=t04['source_model_identity'])
    identity_sha = t04['source_model_identity_sha256'] if dev_mode_enabled() else canonical_sha256(identity)
    source_identity_check(t04, identity_sha, args.reference_binding, args.reference_binding_sha256, identity, model)
    policy = _source_derivative_policy(args)
    device = require_cuda_hot_path("g3_offline_decoded_kl", "cuda")
    producer = teacher_mod.producer_identity()
    source_reads = SourceReads(model, arms, by_layer)
    source_reads.install()
    runner = build_streamed_causal_lm(
        str(model), device=device, dtype=torch.bfloat16,
        offload_folder=args.offload_folder, profile=detect_profile(str(model)),
        cache_headroom_gb=args.cache_headroom_gb, max_cache_slots=2,
        prefetch_workers=1, prefetch_lookahead=1, require_prefetched_residency=True,
        attn_implementation="eager", source_derivative=policy,
    )
    base = {"schema": "surrogate-diag.g3.arm.v1", "run_tag": args.run_tag,
            "manifest_sha256": manifest_sha, "hostname": os.uname().nodename,
            "container_content_sha256": os.environ.get("PRISMAQUANT_CONTAINER_CONTENT_SHA256"),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "tessera": sys.modules["tessera"].__file__ if "tessera" in sys.modules else None,
            "decoder": getattr(decoder, "__qualname__", type(decoder).__name__),
            "decode_threads": args.decode_threads,
            "source_model_identity_sha256": identity_sha,
            "producer_identity_equals_teacher04": producer == t04["producer_identity"],
            "producer_identity": producer, "argv": sys.argv,
            "multi_pass": {"arms": arms, "dir": out.name, "plan_totals": totals}}
    base.update(run_metadata_stamp())
    kl_rows = {a: {"teacher04": [], "teacher2": []} for a in arms}
    per_window = {a: [] for a in arms}
    started = time.monotonic()
    inst = MultiInstaller(runner, arms, by_layer, plan,
                          {"t8r": args.t8r_export, "a8": args.a8_export, "wirecache": args.wire_cache,
                           "exl3": args.exl3_dir},
                          decoder, args.decode_threads)
    layer_records = []
    try:
        if runner.context.max_cache_slots != 2 or runner.prefetch_lookahead != 1 or not runner.require_prefetched_residency:
            raise SystemExit("runner is not teacher-04's two-slot resident prefetch policy")
        runner.model.eval()
        runner.context.begin_source_initialization_audit()
        source_execution = source_execution_identity(runner.model)
        _require_source_execution_policy(source_execution, policy)
        base["source_execution_equals_teacher04"] = source_execution == t04["source_execution"]
        base["experts_implementation"] = sorted({v.get("experts") for v in source_execution.get("modules", {}).values()})
        base["resident_plan"] = {"estimated_layer_bytes": runner.context.estimated_layer_bytes,
                                 "cache_max_bytes": runner.context.layer_cache.max_bytes,
                                 "prefetch_min_available_bytes": runner.context.prefetch_min_available_bytes,
                                 "num_layers": runner.num_layers}
        if runner.num_layers != 45:
            raise SystemExit(f"runner has {runner.num_layers} decoder layers, manifest 45")
        emit_g3_progress("runner ready", base["resident_plan"], "arms", arms)
        inst.prefetch_first()
        last = runner.num_layers - 1
        try:
            import psutil
            mem_available = lambda: round(psutil.virtual_memory().available / 2 ** 30, 2)
        except ImportError:
            mem_available = None
        keeper = SourcePrefetchKeeper(runner.context, runner.num_layers, empty_cache=torch.cuda.empty_cache,
                                      mem_available=mem_available)

        def visitor(layer, forward_batch, profiler):
            multi_visit_layer(layer, forward_batch, profiler, runner=runner, inst=inst, keeper=keeper,
                              by_layer=by_layer, arms=arms, inputs=inputs, teachers=teachers, last=last,
                              profile_layer=args.profile_layer, out=out, layer_records=layer_records,
                              unit_view=L.unit_view,
                              wa_hooks=lambda m_, skip=frozenset(): L.install_wa_hooks(m_, tp=L.TP_SERVED, skip=skip),
                              sync=torch.cuda.synchronize, max_allocated=torch.cuda.max_memory_allocated,
                              reserved=torch.cuda.memory_reserved)

        def consume(index, logits):
            arm, w = arms[index // nwin], index % nwin
            if tuple(logits.shape) != (1, CONTEXT_LENGTH, VOCAB_SIZE) or logits.device.type != "cuda":
                raise SystemExit("candidate logits geometry/device")
            raw = logits[0, :-1].float()
            if not bool(torch.isfinite(raw).all()):
                raise SystemExit(f"{arm}: candidate emitted nonfinite logits")
            cand_sha = sha_bytes(memoryview(raw.cpu().numpy()).cast("B"))
            loaded = teachers.get(index)
            row = {"window_id": window_ids[w], "candidate_logits_data_sha256": cand_sha}
            for (name, *_rest), (arr, data_sha) in zip(teachers.specs, loaded):
                t = torch.from_numpy(arr).to(device)
                kl = token_kl(t, raw)
                kl_rows[arm][name].append(kl.cpu().numpy())
                top1 = float((t.argmax(dim=-1) == raw.argmax(dim=-1)).double().mean())
                row[name] = {"mean_kl": float(kl.mean()), "top1_agreement": top1,
                             "teacher_logits_data_sha256": data_sha,
                             "logits_bitwise_equal": data_sha == cand_sha,
                             "max_abs_logit_diff": float((t - raw).abs().max())}
                del t, kl
            per_window[arm].append(row)
            emit_g3_progress(f"{arm} window {row['window_id']}: KL t04 {row['teacher04']['mean_kl']:.6f} "
                f"t2 {row['teacher2']['mean_kl']:.6f} maxdiff04 {row['teacher04']['max_abs_logit_diff']:.4g}")

        next_output = 0

        def output(index, logits):
            nonlocal next_output
            if index != next_output:
                raise ValueError("output window order changed")
            consume(index, logits)
            next_output += 1

        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                on_trace_ready=lambda p: p.export_chrome_trace(str(out / "first-layer.trace.json")),
                record_shapes=False, profile_memory=True) as profiler:
            with torch.inference_mode():
                runner.visit_layer_batches(list(inputs) * len(arms),
                                           lambda layer, fb: visitor(layer, fb, profiler),
                                           output_consumer=output)
        if next_output != nwin * len(arms):
            raise ValueError("traversal omitted final windows")
        base["source_initialization_contract_sha256"] = canonical_sha256(
            runner.context.source_initialization_contract())
        seal_check('source execution', source_execution, source_execution_identity(runner.model), where='G3',
                   refusal=SystemExit('source execution changed during the run'))
    finally:
        runner.shutdown()
    elapsed = time.monotonic() - started
    write_g3_record({"layers": layer_records, "elapsed_seconds": elapsed, "wire_bytes_read": inst.wire_bytes,
                 "profile_error": None}, out / "layers.json")
    results = {}
    for arm in arms:
        d = arm_dirs[arm]
        d.mkdir(parents=True, exist_ok=False)
        result = dict(base, arm=arm)
        result['source_readset'] = dict(source_reads.stats, omitted_names=sorted(source_reads.omitted))
        result['residency'] = residency_receipt()
        result['hash_copy_profile'] = dict(inst.hash.profile)
        for name, rows in kl_rows[arm].items():
            allk = np.concatenate(rows).astype(np.float64)
            np.save(d / f"per_position_kl.{name}.npy", np.stack(rows), allow_pickle=False)
            result[name] = {"mean_kl": float(allk.mean()), "positions": int(allk.size),
                            "per_window_mean": [float(r.mean()) for r in rows],
                            "p99_kl": float(np.quantile(allk, 0.99)), "max_kl": float(allk.max()),
                            "top1_agreement": float(np.mean([w[name]["top1_agreement"] for w in per_window[arm]])),
                            "windows_bitwise_equal": sum(w[name]["logits_bitwise_equal"] for w in per_window[arm]),
                            "max_abs_logit_diff": max(w[name]["max_abs_logit_diff"] for w in per_window[arm])}
        result["windows"] = per_window[arm]
        c = inst.counts[arm]
        result["hash_gate"] = {**c, "rows": sum(len(v) for v in by_layer.values()),
                               "all_hashes_matched": inst.hash.all_matched}
        result["passthrough_kinds"] = sorted(SRC_KINDS.get(arm, ()))   # source BF16, no W+A hooks
        key = ARMS[arm][0] or ""
        if key.startswith("pick_"):
            fc = {}
            for rows_ in by_layer.values():
                for r in rows_:
                    k = f"{r['kind']}|{r[key + '_format']}"
                    fc[k] = fc.get(k, 0) + 1
            result["pick"] = {"name": key[5:], "formats": dict(sorted(fc.items())),
                              "manifest_schema": m.get("schema")}
        result["elapsed_seconds"] = elapsed
        result["cuda_max_allocated"] = torch.cuda.max_memory_allocated()
        results[arm] = result
    return results, arm_dirs


def exl3_cuda_preflight(args, arms, m):
    """Before the runner loads anything: decode one EXL3 expert (gate, up, down) on the GPU exactly
    as the pass will and refuse unless each hashes to the CPU pre-pass's rendered identity."""
    for a in arms:
        key = ARMS[a][0] or ""
        rows = [r for r in m["rows"] if r.get(key + "_format") == EXL3_FORMAT][:3]
        if not rows:
            continue
        reader = WireReader(key, {}, {"exl3": args.exl3_dir})
        for r in rows:
            out_f, in_f = r[key + "_rendered_shape"]
            dec = g3_exl3.decode_wire(reader._read(r), in_f, out_f, device="cuda").to(torch.bfloat16)
            L.check_identity(dec, r[key + "_rendered_sha256"], f"{r['qname']} [EXL3 CUDA preflight]")
        reader.close()
        emit_g3_progress("EXL3 CUDA preflight: 3 units decode on the GPU to the pre-pass identities", [r["qname"] for r in rows])
        return


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--arm", choices=sorted(ARMS))
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--arms", help="comma-separated arms for one multi-arm source pass (null first)")
    p.add_argument("--decode-threads", type=int, default=3)
    p.add_argument("--run-tag", default="cold1")
    p.add_argument("--out", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--t8r-export", required=True)
    p.add_argument("--a8-export", required=True)
    p.add_argument("--wire-cache", help="census wire cache (manifest v2 rows with root 'wirecache')")
    p.add_argument("--exl3-dir", help="the EXL3 reference artifact (manifest v3 rows with root 'exl3')")
    p.add_argument("--profile-layer", type=int, default=5)
    for name in ("model", "source-preparation", "source-preparation-sha256", "panel", "reference-binding", "reference-binding-sha256",
                 "source-derivative-json", "source-derivative-sha256", "offload-folder",
                 "teacher", "teacher-sha256", "teacher2", "teacher2-sha256"):
        p.add_argument("--" + name)
    p.add_argument("--arrays-root")
    p.add_argument("--cache-headroom-gb", type=float, default=12.)
    args = p.parse_args()
    import tessera
    from tessera.unit_artifact import read_unit_artifact
    seal_check('Tessera import pin', TESSERA_SRC, tessera.__file__, where='G3',
               same=tessera.__file__.startswith(TESSERA_SRC), refusal=SystemExit('Tessera import pin differs'))
    m, by_layer, msha = read_g3_manifest(args.manifest, args.manifest_sha256, None)
    picks = register_picks(m) if m.get("schema") == "surrogate-diag.g3.unit_manifest.v3" else []
    emit_g3_progress("manifest", msha, "rows", m["n_rows"], "tessera", tessera.__file__, "picks", picks)
    if args.pilot:
        raise SystemExit(run_pilot(args, m, by_layer, msha, read_unit_artifact))
    arms = args.arms.split(",") if args.arms else [args.arm]
    missing = [n for n in ('model', 'panel', 'reference_binding', 'reference_binding_sha256',
                           "source_derivative_json", "source_derivative_sha256", "offload_folder",
                           "teacher", "teacher_sha256", "teacher2", "teacher2_sha256") if getattr(args, n) is None]
    if missing:
        p.error(f"an arm needs {missing}")
    if any(a not in ARMS for a in arms):
        p.error(f"unknown arm in {arms}")
    if not args.arms and SRC_KINDS.get(arms[0]):
        p.error(f"{arms[0]} keeps source passthrough units; it runs only in a multi-arm pass (--arms)")
    if not args.exl3_dir and any(r.get(ARMS[a][0] + "_format") == EXL3_FORMAT for a in arms
                                 if (ARMS[a][0] or "").startswith("pick_") for r in m["rows"]):
        p.error("an arm reads EXL3 reference units; --exl3-dir is required")
    if "lawfit" in {ARMS[a][0] for a in arms}:
        if m.get("schema") != "surrogate-diag.g3.unit_manifest.v2" or any(
                "lawfit_wire" not in r or "lawfit_rendered_sha256" not in r for r in m["rows"]):
            p.error("a lawfit arm needs manifest v2 with every row's lawfit wire and rendered identity")
        if m.get("lawfit_csv_sha256") != "2a1c7ddabcb2b14175fdad1e80359758bace0a43fa4ca7f491c651e63e6e2b4a":
            p.error("manifest v2 is not bound to the pre-registered law-fit pick 2a1c7dda")
        if any(r["lawfit_wire"]["root"] == "wirecache" for r in m["rows"]) and not args.wire_cache:
            p.error("the law-fit pick reads wire-cache units; --wire-cache is required")
    sys.path.insert(0, str(PQ))
    sys.path.insert(0, str(PQ / "tools"))
    from experiments.glm_full_capture_profile import CaptureObserver
    import experiments.workspace_netdata as wn
    from experiments.workspace_netdata import sample_netdata
    # PB's mount-latency collector publishes no prismabuild.mount_* charts on either GB10 host
    # (absent on sparky and sparklina, 2026-09-29 ~23:25Z; PB action 76fb0791 died on it before any
    # work).  Sample every other required chart, keep the prismabuild.mount_* pattern so the charts
    # are recorded if they return, and stamp the relaxation into the result.
    relaxed = sorted(c for c in wn.REQUIRED_CHARTS if c.startswith("prismabuild.mount_"))
    wn.REQUIRED_CHARTS = frozenset(c for c in wn.REQUIRED_CHARTS if c not in relaxed)
    out = Path(args.out)
    telemetry = out.parent / (out.name + ".telemetry")
    with CaptureObserver(telemetry, profile_layers=()) as observer:
        write_g3_record([sample_netdata(h) for h in ("sparky", "sparklina")], telemetry / "before.json")
        if args.arms:
            exl3_cuda_preflight(args, arms, m)
            results, dirs = run_multi(args, arms, m, by_layer, msha, read_unit_artifact)
        else:
            results, dirs = {args.arm: run_arm(args, m, by_layer, msha, read_unit_artifact)}, {args.arm: out}
        write_g3_record([sample_netdata(h) for h in ("sparky", "sparklina")], telemetry / "after.json")
    telem = {"dir": telemetry.name, "status": observer.result["status"],
             "errors": observer.result.get("errors"),
             "required_charts_relaxed": relaxed,
             "files": {n: sha256_file(telemetry / n) for n in
                       ("netdata.jsonl", "python_sampler.jsonl", "result.json",
                        "before.json", "after.json") if (telemetry / n).exists()}}
    for arm, result in results.items():
        result["host_telemetry"] = telem
        write_g3_record(result, dirs[arm] / "result.json")
        emit_g3_progress("DONE", arm, {k: result[k]["mean_kl"] for k in ("teacher04", "teacher2")},
            "hashes", result["hash_gate"])


if __name__ == "__main__":
    main()
