"""The produced-render writer/reader connection (PQ #870).

Exercises the ACTUAL paths end to end on tiny real tensors: the renderer's
atomic tensor writer inside ``ProductionWeightCache.store_rendered_weight_
published`` (PB prewrite admission BEFORE any byte, the existing
``_store_rendered_weight_entry``, prepaid batch publication funded from the
producer's reserved window, the mover's real staged output read back),
valid cache reuse without a fresh charge, prewrite refusal before a file
appears, idempotent retry, and retire/release capacity.  The executable
readset seam accepts exactly what the real queue validator accepts.

PrismaBuild's produced-output API is pinned to the DEPLOYED runtime
generation ``d1c640e74d97-1790008584-459339399864`` (see
``produced_render_pb_pin.json``): the tests resolve that generation's
directory, verify every listed file by digest, and skip loudly -- never
silently pass -- when the bytes do not resolve.  Rebased from PR #871
(which pinned the unqualified candidate) onto the qualified deployed
bytes; PQ-side attribution to that PR is kept on the implementation.
"""
from __future__ import annotations

import glob as _glob
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys

import pytest
import torch

try:
    torch.set_num_threads(1)
except RuntimeError:
    pass
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from prismaquant.joint_layer_quanta import (  # noqa: E402
    build_quantum_executable_manifest, seal_manifest_bytes)

TIER = "prismabuild-stage:dl380g10"
KIND = "stage_gib"
PIN_PATH = HERE / "produced_render_pb_pin.json"
GENERATIONS_ROOT = Path("/mnt/shared/prismabuild-fleet/runtime-generations")


def _pb_source() -> tuple[Path, Path]:
    """Resolve the pinned deployed PrismaBuild produced-output source.

    Verifies every file named by ``produced_render_pb_pin.json`` by digest
    against a generation directory (the pinned generation first, then any
    digest-identical directory), and imports ``prismabuild`` from those
    exact bytes -- no import shadowing, no mixed generations.  Returns
    ``(src_dir, generation_root)`` where the root carries the generation's
    ``tools/fleet``.  Skips loudly when nothing holds the pinned bytes
    (for example a rotated generation, or a shard where another test
    already imported a different prismabuild): never silently passes.
    """

    try:
        pin = json.loads(PIN_PATH.read_text())
    except (OSError, ValueError) as exc:
        pytest.skip(f"produced-output pin unreadable: {exc}")
    try:
        files = dict(pin["files"])
        generation = str(pin["generation"])
    except (KeyError, TypeError, ValueError) as exc:
        pytest.skip(f"produced-output pin has no generation/files: {exc}")

    def _digest_ok(root: Path) -> bool:
        try:
            for name, digest in files.items():
                data = (root / name).read_bytes()
                if hashlib.sha256(data).hexdigest() != digest:
                    return False
            return True
        except OSError:
            return False

    prior = sys.modules.get("prismabuild")
    if prior is not None:
        prior_file = getattr(prior, "__file__", "") or ""
        if prior_file:
            src = Path(prior_file).resolve().parent
            if _digest_ok(src.parents[1]):
                return src, src.parents[1]
        pytest.skip(
            "a different prismabuild is already imported in this shard; "
            "the produced-render tests need the pinned generation bytes")
    candidates: list[Path] = []
    env = os.environ.get("PRISMAQUANT_PRODUCED_PB_SRC", "")
    if env:
        candidates.append(Path(env))
    candidates.append(GENERATIONS_ROOT / generation)
    try:
        candidates += [Path(p) for p in
                       sorted(_glob.glob(str(GENERATIONS_ROOT / "*")),
                              reverse=True)]
    except OSError:
        pass
    seen: set[Path] = set()
    for root in candidates:
        if root in seen:
            continue
        seen.add(root)
        if not _digest_ok(root):
            continue
        src = root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        try:
            from prismabuild import produced_output as _po
            from prismabuild import pool as _pool
        except ImportError as exc:
            pytest.skip(f"pinned prismabuild does not import: {exc}")
        for module in (_po, _pool):
            if not Path(module.__file__).resolve().is_relative_to(
                    src.resolve()):
                pytest.skip("a different prismabuild is imported")
        return src, root
    pytest.skip(
        "no PrismaBuild source resolves against "
        "tests/produced_render_pb_pin.json "
        f"(pinned generation {generation}; refusing to test other bytes)")


def _hexkey(seed: str) -> str:
    return (seed.encode().hex() * 64)[:64]


# -- PB-side fixtures (the shapes pbrun/tier_loop produce) -------------------

def _queue(tmp_path: Path, gib: int = 4):
    from prismabuild import pool
    q = pool.PoolQueue(tmp_path / "pb-queue")
    q.ensure_layout()
    q.mint_tier_capacity(TIER, {KIND: gib})
    return q


def _broker_control(q, owner: str) -> dict:
    from prismabuild import pool, resource_scope
    nonce = secrets.token_hex(16)
    scope = resource_scope.ResourceScope(
        owner, nonce, 1 * 1024 ** 3, q.root / "telemetry" / f"{owner}.json")
    token = secrets.token_hex(32)
    unit = ("prismabuild-job"
            + hashlib.sha256((owner + nonce).encode()).hexdigest()[:32]
            + ".slice")
    scope._adopt_created_scope({
        "scope_id": unit, "token": token,
        "cgroup_path": str(Path("/sys/fs/cgroup/prismabuild.slice") / unit),
    })
    control = scope.control_record()
    path = q.item_path(pool.CLAIMED, owner)
    live = pool._read_json(path)
    assert live is not None
    live["resource_scope"] = control
    pool._write_json_atomic(path, live)
    return control


def _template(prefix: str, *, payload_max_bytes: int = 1 << 20,
              temp_max_bytes: int | None = None) -> dict:
    from prismabuild import produced_output as po
    # The atomic writer's temp lifetime holds the whole serialization
    # during the write, so the temp ceiling must cover the same bound as
    # the payload ceiling; a 1 KiB temp ceiling would refuse every real
    # write at prewrite (deployed contract, honestly kept).
    if temp_max_bytes is None:
        temp_max_bytes = payload_max_bytes
    return po.validate_template({
        "schema": po.TEMPLATE_SCHEMA_V1,
        "version": 1,
        "template_id": "pq-produced-render-v1",
        "output_prefix": prefix,
        "slots": {"rendered-weights": {"class": "payload"}},
        "durable_maxima": {"payload_max_bytes": payload_max_bytes,
                           "checkpoint_max_bytes": 1 << 10,
                           "temp_max_bytes": temp_max_bytes},
        "working_demands": {TIER: {"minimum_gib": 1, "window_gib": 2}},
        "permitted_tiers": [TIER],
    })


def _sealed_producer_request(tmp_path: Path, cas_root: Path,
                             pb_repo: Path) -> str:
    """The owner's own sealed request, filed once (the parent identity).

    The movement template publish_prepaid_batch derives at runtime comes
    from exactly this request; the code closure covers a checkout carrying
    the fleet tools, sealed ONCE per producer -- never re-sealed per batch.
    """

    from prismabuild import core as pb
    checkout = tmp_path / "mover-checkout"
    tools = checkout / "tools" / "fleet"
    tools.mkdir(parents=True, exist_ok=True)
    for name in ("stage_move.py", "prewarm_loop.py", "stage_release.py"):
        (tools / name).write_bytes(
            (pb_repo / "tools" / "fleet" / name).read_bytes())
    body = {
        "schema": pb.ACTION_SCHEMA_V2,
        "task": {"definition_id": "tests/pq-produced-render-producer",
                  "definition_version": "v1", "task_class": "generation",
                  "determinism": "deterministic",
                  "artifact_family": "generic", "artifact_kind": "generic",
                  "argv": ["/bin/true"], "working_directory": ".",
                  "result_path": "result"},
        "inputs": [],
        "code_closure": pb.build_code_closure(
            checkout, ["tools/fleet/stage_move.py",
                       "tools/fleet/prewarm_loop.py",
                       "tools/fleet/stage_release.py"]),
        "params": {"cwd": "."},
        "environment": {"variables": {"PATH": "/usr/bin:/bin"},
                        "toolchain": {}},
        "execution_scope": {"portability": "portable", "platform_key": None,
                            "host_class": None},
    }
    action = pb.seal_action(body)
    pb.PrismaBuildCAS(cas_root).publish_action_request(action)
    return str(action["action_key"])


def _announce_tier(q, stage_root: Path, pb_repo: Path) -> None:
    """The tier record the storage role announces (tier_loop shape)."""

    from prismabuild import pool
    stage_root.mkdir(parents=True, exist_ok=True)
    registered = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, sys.argv[4]);"
         "sys.path.insert(0, sys.argv[5]);"
         "from prismabuild import pool;"
         "import stage_release;"
         "print(stage_release.register_stage_root("
         "pool.PoolQueue(sys.argv[1]), tier_id=sys.argv[2],"
         " stage_root=sys.argv[3]))",
         str(q.root), TIER, str(stage_root),
         str(pb_repo / "tools" / "fleet"), str(pb_repo / "src")],
        capture_output=True, text=True, timeout=60)
    assert registered.returncode == 0, registered.stdout + registered.stderr
    assert "registered" in registered.stdout, registered.stdout
    record = {
        "tier": "stage", "tier_id": TIER, "host": socket.gethostname(),
        "mountpoint": str(stage_root),
        "mover_python": sys.executable,
        "mover_tools_root": str(pb_repo / "tools" / "fleet"),
    }
    path = Path(q.root) / "tiers" / f"{TIER}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    pool._write_json_atomic(path, record)


def _execute_mover(q, cas_root: Path, mover: str, checkout: Path,
                   pb_repo: Path) -> dict:
    """Run the SEALED mover argv through the real local executor."""

    from prismabuild import core as pb
    runner = Path(q.root) / "run-local-action.py"
    runner.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "from prismabuild import core as pb\n"
        "req, cas, root = sys.argv[1], sys.argv[2], sys.argv[3]\n"
        "out = pb.run_local_action(json.loads(Path(req).read_text()),\n"
        "                          cas_root=cas, checkout_root=root,\n"
        "                          timeout_seconds=180)\n"
        "print(out['status'])\n")
    request_path = Path(cas_root) / "requests" / mover[:2] / f"{mover}.json"
    scrub = {k: v for k, v in os.environ.items()
             if k not in (pb.ACTION_NONCE_ENV, pb.ACTION_SCOPE_ENV,
                          pb.READER_HELPER_ROOT_ENV, pb.ACTION_KEY_ENV)}
    scrub["PYTHONPATH"] = str(pb_repo / "src")
    done = subprocess.run(
        [sys.executable, str(runner), str(request_path),
         str(cas_root), str(checkout)],
        env=scrub, capture_output=True, text=True, timeout=240)
    assert done.returncode == 0, done.stdout + done.stderr
    # The sealed mover prints its own JSON report; the executor's status
    # is the last line.
    assert done.stdout.strip().splitlines()[-1] in (
        "published", "cache_hit"), done.stdout
    receipt = q.move_record(mover)
    assert isinstance(receipt, dict), "mover recorded no receipt"
    return receipt


def _stage_root(q) -> Path:
    return Path(json.loads(
        (Path(q.root) / "tiers" / f"{TIER}.json").read_text())["mountpoint"])


# -- the PQ-side fixture: one bound publication + PWC ------------------------

def _bound_publication(tmp_path: Path, *, gib: int = 4,
                       payload_max_bytes: int = 1 << 20):
    """A real queue, admitted owner, declared template, bound instance."""

    # The pinned source FIRST: importing prismabuild before the
    # digest-verified generation is on sys.path would bind the wrong
    # bytes (a red run on main caught exactly this ordering).
    _src, pb_repo = _pb_source()
    from prismabuild import produced_output as po
    from prismaquant.produced_render_publication import (
        ProducedRenderPublication)
    cas_root = tmp_path / "cas"
    owner = _sealed_producer_request(tmp_path, cas_root, pb_repo)
    q = _queue(tmp_path, gib=gib)
    template = _template(str(tmp_path / "outputs"),
                         payload_max_bytes=payload_max_bytes)
    terms = po.owner_demand_terms(template)
    # The owner's row files the REAL cas_root; the publication binds to
    # that filed value (the live CAS is a sibling of the queue, not a
    # child, so topology is never guessed).
    q.publish(action_key=owner, cas_root=str(cas_root),
              worker_script="/w.py",
              checkout_root="/co", resources={"cpu": 1, "mem_gb": 1, **terms},
              produced_output_template=template)
    claimed = q.claim(owner="w-owner")
    assert claimed is not None and claimed["action_key"] == owner
    control = _broker_control(q, owner)
    env = {"PRISMABUILD_ACTION_KEY": owner,
           "PRISMABUILD_ACTION_NONCE": control["nonce"],
           "PRISMABUILD_ACTION_SCOPE": control["scope_id"]}
    po.declare_template(q.root, template)
    publication = ProducedRenderPublication.bind_from_admitted_owner(
        queue_root=q.root, tier=TIER, env=env)
    publication.command_extra = ("--unpaced",)
    window = publication.admit_window()
    assert window.get("ok") is True, window
    _announce_tier(q, tmp_path / "stage", pb_repo)
    return publication, q, cas_root, pb_repo


def _cache(publication) -> "object":
    from prismaquant.production_weight_cache import ProductionWeightCache
    cache_dir = Path(publication.template["output_prefix"]) / "renders"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = ProductionWeightCache(
        weights={}, levers={"tessera_campaign": True},
        cache_dir=str(cache_dir), metadata={"schema": "test"})
    cache.produced_render_publication = publication
    return cache


# -- pin-independent seam tests (stub validator; no PB bytes needed) ---------

def _stub_accept(binding):
    """A stand-in for the real queue validator: echo the checked binding."""

    assert isinstance(binding, dict)
    return dict(binding)


def test_builder_seals_stub_validated_binding(tmp_path: Path) -> None:
    """The builder seals a binding exactly when a validator accepts it."""

    from test_quantum_executable_readset import (  # noqa: E402
        CALIB, N_PROBES, RENDER_PREREQ, STRIDED, _layer2)
    record, receipt, parent = _layer2(tmp_path)
    binding = {"scope": "pb732", "material": "f" * 64}
    manifest = build_quantum_executable_manifest(
        record, receipt, parent, strided_boundaries=STRIDED,
        n_probes=N_PROBES, calib=dict(CALIB),
        render_prerequisite={**RENDER_PREREQ, "binding": dict(binding)},
        binding_validator=_stub_accept)
    assert manifest["annotations"]["render_prerequisite"][
        "binding"] == binding


def test_builder_refuses_binding_without_validator(tmp_path: Path) -> None:
    """A binding without a validator still refuses: no sealing fiction."""

    from test_quantum_executable_readset import (  # noqa: E402
        CALIB, N_PROBES, RENDER_PREREQ, STRIDED, _layer2)
    record, receipt, parent = _layer2(tmp_path)
    with pytest.raises(ValueError, match="binding validator"):
        build_quantum_executable_manifest(
            record, receipt, parent, strided_boundaries=STRIDED,
            n_probes=N_PROBES, calib=dict(CALIB),
            render_prerequisite={**RENDER_PREREQ,
                                 "binding": {"scope": "pb732"}})


def test_dispatcher_admits_stub_validated_binding(tmp_path: Path,
                                                  monkeypatch) -> None:
    """quantum_argv admits an executable row on a validated binding.

    The launch refusal this task removes: on main every executable row
    raises ExecutableBindingUnsupported, even with a binding a validator
    accepts.
    """

    import dispatch_joint_quanta as dispatch
    from test_quantum_executable_readset import (  # noqa: E402
        CALIB, N_PROBES, RENDER_PREREQ, STRIDED, _layer2)
    record, receipt, parent = _layer2(tmp_path)
    binding = {"scope": "pb732", "material": "e" * 64}
    manifest = build_quantum_executable_manifest(
        record, receipt, parent, strided_boundaries=STRIDED,
        n_probes=N_PROBES, calib=dict(CALIB),
        render_prerequisite={**RENDER_PREREQ, "binding": dict(binding)},
        binding_validator=_stub_accept)
    manifest_path = tmp_path / "exec.json.gz"
    wire = seal_manifest_bytes(manifest)
    manifest_path.write_bytes(wire)
    row = {
        "quantum_id": "layer-002", "layer": 2,
        "campaign": {"plan_path": "plan.json", "plan_sha256": "0" * 64,
                     "prepared_path": "prep.json",
                     "prepared_sha256": "1" * 64},
        "read_set": {"manifest_path": str(tmp_path / "slice.gz"),
                     "manifest_sha256": hashlib.sha256(
                         b"slice").hexdigest()},
        "chunks": [{"name": "layer-002-c000"}],
        "adjoint": {},
        "executable_readset": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": hashlib.sha256(wire).hexdigest(),
            "entry_count": manifest["entry_count"],
            "total_bytes": manifest["total_bytes"],
            "read_bytes": manifest["read_plan"]["read_bytes"],
            "phases": [p["name"]
                       for p in manifest["read_plan"]["phases"]],
            "receipt_sha256": "0" * 64,
        },
    }
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(row))
    adjoint_path = tmp_path / "adjoint.json"
    adjoint_path.write_text("{}")
    (tmp_path / "slice.gz").write_bytes(b"slice")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"container": {"image": "sha256:" + "0" * 64},
                                "env": {}}))
    monkeypatch.setattr(dispatch, "SPEC_PATH", spec)
    argv = dispatch.quantum_argv(
        row, record_path=record_path, output_root=tmp_path,
        adjoint_path=adjoint_path, binding_validator=_stub_accept)
    assert argv[argv.index("--data-manifest") + 1].endswith("exec.json.gz")


# -- PB-backed tests (deployed generation; skip loudly without it) -----------

def test_prewrite_refusal_leaves_no_file(tmp_path: Path) -> None:
    """A budget refusal happens BEFORE the first byte: no file appears."""
    from prismaquant.produced_render_publication import (
        ProducedRenderPrewriteRefused)
    publication, q, _cas, _repo = _bound_publication(
        tmp_path, payload_max_bytes=16)
    cache = _cache(publication)
    tensor = torch.arange(64, dtype=torch.float32).reshape(4, 16)
    qname = "model.layers.0.self_attn.q_proj"
    with pytest.raises(ProducedRenderPrewriteRefused) as caught:
        cache.store_rendered_weight_published(
            qname=qname, fmt="NVFP4", tensor=tensor,
            weight_dtype=torch.bfloat16)
    refusal = caught.value.refusal
    assert refusal.get("ok") is False
    assert "payload" in str(refusal.get("refusal", "")), refusal
    renders = Path(publication.template["output_prefix"]) / "renders"
    assert not any(renders.glob("*.pt")), "a refused prewrite left a file"
    assert not cache.weights, "a refused prewrite filed a weights entry"


def test_published_render_roundtrip_with_real_mover(tmp_path: Path) -> None:
    """Write -> bound-admitted prewrite -> atomic store -> publish -> mover."""
    publication, q, cas_root, pb_repo = _bound_publication(tmp_path)
    cache = _cache(publication)
    ledger = q.tier_ledger(TIER)
    owner = publication.instance["owner_action_key"]
    window_held = int(ledger.holder_tokens(owner).get(KIND, 0))
    tensor = torch.arange(64, dtype=torch.float32).reshape(4, 16)
    qname = "model.layers.0.self_attn.q_proj"
    out = cache.store_rendered_weight_published(
        qname=qname, fmt="NVFP4", tensor=tensor,
        weight_dtype=torch.bfloat16)
    assert out["ok"] is True and out["reused"] is False, out
    render_path = Path(out["path"])
    assert render_path.is_file()
    # The budget was admitted from a storage-metadata BOUND (never a full
    # serialization pass to count bytes) and covers the actual write.
    assert out["admitted_bytes"] >= out["actual_bytes"] > 0
    assert render_path.stat().st_size == out["actual_bytes"]
    # The written bytes are the PWC's canonical tensor, readable back.
    stored = torch.load(render_path, weights_only=True)
    assert torch.equal(stored, tensor.to(torch.bfloat16))
    assert cache.weights[(qname, "NVFP4")] == render_path.name
    # Prepaid funding: the owner's window moved to the mover, free
    # untouched, no second acquisition.
    mover = str(out["mover_key"])
    assert int(ledger.holder_tokens(owner).get(KIND, 0)) == window_held - 1
    assert int(ledger.holder_tokens(mover).get(KIND, 0)) == 1
    # The reader path: execute the sealed mover argv for real, then read
    # the staged bytes it landed on the tier.
    receipt = _execute_mover(q, cas_root, mover,
                             tmp_path / "mover-checkout", pb_repo)
    assert receipt.get("complete") is True, receipt
    staged = list(_stage_root(q).rglob(render_path.name))
    assert staged, "the mover landed no render file on the tier"
    assert torch.load(staged[0], weights_only=True).equal(stored)


def test_size_bound_covers_real_serialization(tmp_path: Path) -> None:
    """The admitted bound is honest: >= torch.save's actual output."""
    from prismaquant.production_weight_cache import _torch_save_size_bound
    for shape, dtype in (((4, 16), torch.float32),
                         ((7, 3, 64), torch.bfloat16),
                         ((1, 1), torch.float16),
                         ((257, 91), torch.bfloat16)):
        tensor = torch.arange(
            int(torch.tensor(shape).prod().item()),
            dtype=torch.float32).reshape(shape).to(dtype)
        path = tmp_path / f"{shape}-{dtype}.pt"
        torch.save(tensor, path)
        assert _torch_save_size_bound(tensor) >= path.stat().st_size, (
            shape, dtype, path.stat().st_size)


def test_cache_hit_reuses_without_charge_or_rewrite(tmp_path: Path) -> None:
    """A PROVEN publication is a clean hit: no prewrite, publish, rewrite."""
    publication, q, _cas, _repo = _bound_publication(tmp_path)
    cache = _cache(publication)
    tensor = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    qname = "model.layers.1.mlp.gate_proj"
    first = cache.store_rendered_weight_published(
        qname=qname, fmt="NVFP4", tensor=tensor,
        weight_dtype=torch.bfloat16)
    assert first["reused"] is False
    render_path = Path(first["path"])
    stat_before = render_path.stat()
    prewrites_before = sorted(Path(q.root).rglob("*.prewrite.json"))
    # Clean hit: this process proved the publication, and the file's stat
    # signature still matches -- no publish, no charge, no rewrite.
    second = cache.store_rendered_weight_published(
        qname=qname, fmt="NVFP4", tensor=tensor,
        weight_dtype=torch.bfloat16)
    assert second["ok"] is True and second["reused"] is True, second
    assert second["reconciled"] is False and second["publish"] is None
    stat_after = render_path.stat()
    assert (stat_before.st_ino, stat_before.st_mtime_ns) == (
        stat_after.st_ino, stat_after.st_mtime_ns), \
        "a cache hit rewrote the render file"
    assert sorted(Path(q.root).rglob("*.prewrite.json")) == prewrites_before, \
        "a cache hit filed a fresh prewrite charge"


def test_unproven_publication_reconciles_never_guesses(tmp_path: Path) -> None:
    """A filed render with unproven publication reconciles or refuses.

    Existence is not proof: with the recorded success dropped (a crash
    between write and record, or an earlier failed publication), the next
    store re-publishes the SAME batch id with the descriptor the file's
    current size derives -- a typed duplicate on success -- and a file
    whose bytes changed under the same batch id refuses loudly instead of
    answering a bogus cache-hit success.
    """
    publication, q, _cas, _repo = _bound_publication(tmp_path)
    cache = _cache(publication)
    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    qname = "model.layers.2.mlp.up_proj"
    first = cache.store_rendered_weight_published(
        qname=qname, fmt="NVFP4", tensor=tensor,
        weight_dtype=torch.bfloat16)
    render_path = Path(first["path"])
    # Drop the proven-success record: the next store must reconcile.
    cache.metadata["produced_render_batches"].clear()
    reconciled = cache.store_rendered_weight_published(
        qname=qname, fmt="NVFP4", tensor=tensor,
        weight_dtype=torch.bfloat16)
    assert reconciled["reused"] is True and reconciled["reconciled"] is True
    assert reconciled["publish"].get("ok") is True
    assert reconciled["publish"].get("duplicate") is True, reconciled
    assert str(reconciled["publish"]["mover_key"]) == str(first["mover_key"])
    # Bytes changed under the same batch id: loud refusal, never a hit.
    cache.metadata["produced_render_batches"].clear()
    render_path.write_bytes(b"tampered-bytes")
    from prismaquant.produced_render_publication import (
        ProducedRenderPublicationFailed)
    with pytest.raises(ProducedRenderPublicationFailed):
        cache.store_rendered_weight_published(
            qname=qname, fmt="NVFP4", tensor=tensor,
            weight_dtype=torch.bfloat16)


def test_failed_publication_retry_reconciles(tmp_path: Path) -> None:
    """A durable write whose publish failed retries to reconciliation.

    The write is on disk; the publication refused (here: the tier record
    the movement resolution needs is gone).  The retry re-derives the
    same batch, descriptor and mover from existing state and completes.
    """
    from prismaquant.produced_render_publication import (
        ProducedRenderPublicationFailed)
    publication, q, _cas, _repo = _bound_publication(tmp_path)
    cache = _cache(publication)
    tensor = torch.arange(20, dtype=torch.float32).reshape(2, 10)
    qname = "model.layers.3.mlp.down_proj"
    tiers_path = Path(q.root) / "tiers" / f"{TIER}.json"
    filed_tier = tiers_path.read_bytes()
    tiers_path.unlink()
    with pytest.raises(ProducedRenderPublicationFailed) as caught:
        cache.store_rendered_weight_published(
            qname=qname, fmt="NVFP4", tensor=tensor,
            weight_dtype=torch.bfloat16)
    assert caught.value.refusal.get("ok") is False
    render_path = Path(cache.cache_dir) / cache.weights[(qname, "NVFP4")]
    assert render_path.is_file(), "the durable write must survive"
    # The fault clears: the retry re-derives the same batch and descriptor
    # from existing state and completes.  The tier-not-announced refusal
    # filed nothing, so this reconciliation is a fresh publish; the
    # follow-up with identical inputs answers a typed duplicate carrying
    # the same content-addressed mover key.
    tiers_path.write_bytes(filed_tier)
    retry = cache.store_rendered_weight_published(
        qname=qname, fmt="NVFP4", tensor=tensor,
        weight_dtype=torch.bfloat16)
    assert retry["reused"] is True and retry["reconciled"] is True, retry
    assert retry["publish"].get("ok") is True, retry
    descriptor = publication.descriptor_for(
        retry["path"], slot="rendered-weights", artifact_class="payload",
        producer_generation=retry["batch_id"])
    again = publication.publish(batch_id=retry["batch_id"],
                                descriptors=[descriptor])
    assert again.get("duplicate") is True, again
    assert str(again["mover_key"]) == str(retry["publish"]["mover_key"])


def test_retry_publish_is_idempotent(tmp_path: Path) -> None:
    """Identical retry inputs re-derive the mover; typed duplicate."""
    publication, _q, _cas, _repo = _bound_publication(tmp_path)
    cache = _cache(publication)
    tensor = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    out = cache.store_rendered_weight_published(
        qname="model.layers.2.self_attn.k_proj", fmt="NVFP4",
        tensor=tensor, weight_dtype=torch.bfloat16)
    batch_id = out["batch_id"]
    descriptor = publication.descriptor_for(
        out["path"], slot="rendered-weights", artifact_class="payload",
        producer_generation=out["batch_id"])
    again = publication.publish(batch_id=batch_id,
                                descriptors=[descriptor])
    assert again.get("ok") is True, again
    assert again.get("duplicate") is True, again
    assert str(again["mover_key"]) == str(out["mover_key"])


def test_retire_returns_capacity_and_releases(tmp_path: Path) -> None:
    """retire_batch cleans the staged batch.

    Release while the owner claim is live honestly retains
    (owner-active-retain): the deployed contract releases leftovers only
    with no live claim plus a broker containment certificate, a proof
    chain only a real broker run produces (PB's own suite owns that
    positive path). The adapter proves it passes the coherent SDK by
    getting past lease-sdk-missing to the containment verdict.
    """
    publication, q, _cas, pb_repo = _bound_publication(tmp_path)
    # The fixture process plays tier host here: the in-process egress
    # needs the fleet tool importable, exactly as the tier host's own
    # action environment provides it (deployed retire contract).
    fleet_tools = str(pb_repo / "tools" / "fleet")
    if fleet_tools not in sys.path:
        sys.path.insert(0, fleet_tools)
    cache = _cache(publication)
    tensor = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    out = cache.store_rendered_weight_published(
        qname="model.layers.3.self_attn.v_proj", fmt="NVFP4",
        tensor=tensor, weight_dtype=torch.bfloat16)
    mover = str(out["mover_key"])
    ledger = q.tier_ledger(TIER)
    retired = publication.retire(out["batch_id"],
                                 stage_root=_stage_root(q))
    assert retired.get("ok") is True, retired
    assert int(ledger.holder_tokens(mover).get(KIND, 0)) == 0
    released = publication.release()
    assert released.get("ok") is False, released
    assert released.get("refusal") == "owner-active-retain", released


def _validated_ref(publication, q, qname: str) -> dict:
    cache = _cache(publication)
    tensor = torch.arange(16, dtype=torch.float32).reshape(1, 16)
    out = cache.store_rendered_weight_published(
        qname=qname, fmt="NVFP4", tensor=tensor,
        weight_dtype=torch.bfloat16)
    descriptor = publication.descriptor_for(
        out["path"], slot="rendered-weights", artifact_class="payload",
        producer_generation=out["batch_id"])
    return q.build_produced_output_batch_ref(
        instance=publication.instance, template=publication.template,
        batch_id=out["batch_id"], descriptors=[descriptor], tier_id=TIER)


def test_executable_readset_seals_validated_binding(tmp_path: Path) -> None:
    """The manifest seals a binding the real queue validator accepted."""
    publication, q, _cas, _repo = _bound_publication(tmp_path)
    ref = _validated_ref(publication, q, "model.layers.4.mlp.up_proj")
    from prismaquant.produced_render_publication import (
        produced_batch_binding_validator)
    validator = produced_batch_binding_validator(q.root)
    assert validator(ref) == ref
    from test_quantum_executable_readset import (  # noqa: E402
        CALIB, N_PROBES, RENDER_PREREQ, STRIDED, _layer2)
    record, receipt, parent = _layer2(tmp_path)
    manifest = build_quantum_executable_manifest(
        record, receipt, parent, strided_boundaries=STRIDED,
        n_probes=N_PROBES, calib=dict(CALIB),
        render_prerequisite={**RENDER_PREREQ, "binding": dict(ref)},
        binding_validator=validator)
    sealed = manifest["annotations"]["render_prerequisite"]["binding"]
    assert sealed == ref
    # Without a validator the same binding refuses; None still seals.
    with pytest.raises(ValueError, match="binding validator"):
        build_quantum_executable_manifest(
            record, receipt, parent, strided_boundaries=STRIDED,
            n_probes=N_PROBES, calib=dict(CALIB),
            render_prerequisite={**RENDER_PREREQ, "binding": dict(ref)})
    sequencing = build_quantum_executable_manifest(
        record, receipt, parent, strided_boundaries=STRIDED,
        n_probes=N_PROBES, calib=dict(CALIB),
        render_prerequisite=dict(RENDER_PREREQ))
    assert sequencing["annotations"]["render_prerequisite"][
        "binding"] is None
    # A tampered binding must be refused by the real validator.
    tampered = dict(ref, batch_id="other-batch")
    with pytest.raises(ValueError, match="not an accepted"):
        build_quantum_executable_manifest(
            record, receipt, parent, strided_boundaries=STRIDED,
            n_probes=N_PROBES, calib=dict(CALIB),
            render_prerequisite={**RENDER_PREREQ, "binding": tampered},
            binding_validator=validator)


def test_dispatcher_admits_validated_binding(tmp_path: Path,
                                             monkeypatch) -> None:
    """quantum_argv admits an executable row exactly on a validated
    binding; sequencing-only and unvalidated rows keep the typed gate."""

    import dispatch_joint_quanta as dispatch
    publication, q, _cas, _repo = _bound_publication(tmp_path)
    ref = _validated_ref(publication, q, "model.layers.5.mlp.down_proj")
    from prismaquant.produced_render_publication import (
        produced_batch_binding_validator)
    validator = produced_batch_binding_validator(q.root)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"container": {"image": "sha256:" + "0" * 64},
                                "env": {}}))
    monkeypatch.setattr(dispatch, "SPEC_PATH", spec)
    from test_quantum_executable_readset import (  # noqa: E402
        CALIB, N_PROBES, RENDER_PREREQ, STRIDED, _layer2)
    record, receipt, parent = _layer2(tmp_path)

    def _row(binding, *, tamper=None):
        prerequisite = dict(RENDER_PREREQ)
        if binding is not None:
            prerequisite["binding"] = dict(binding)
        manifest = build_quantum_executable_manifest(
            record, receipt, parent, strided_boundaries=STRIDED,
            n_probes=N_PROBES, calib=dict(CALIB),
            render_prerequisite=prerequisite,
            binding_validator=(validator if binding is not None else None))
        if tamper is not None:
            # A sealed-but-tampered binding: dispatch must validate
            # independently of the builder (defense in depth), so mutate
            # the sealed body by hand and re-seal its wire.
            tamper(manifest["annotations"]["render_prerequisite"]["binding"])
        manifest_path = tmp_path / "exec.json.gz"
        wire = seal_manifest_bytes(manifest)
        manifest_path.write_bytes(wire)
        row = {
            "quantum_id": "layer-002", "layer": 2,
            "campaign": {"plan_path": "plan.json", "plan_sha256": "0" * 64,
                         "prepared_path": "prep.json",
                         "prepared_sha256": "1" * 64},
            "read_set": {"manifest_path": str(tmp_path / "slice.gz"),
                         "manifest_sha256": hashlib.sha256(
                             b"slice").hexdigest()},
            "chunks": [{"name": "layer-002-c000"}],
            "adjoint": {},
            "executable_readset": {
                "manifest_path": str(manifest_path),
                "manifest_sha256": hashlib.sha256(wire).hexdigest(),
                "entry_count": manifest["entry_count"],
                "total_bytes": manifest["total_bytes"],
                "read_bytes": manifest["read_plan"]["read_bytes"],
                "phases": [p["name"]
                           for p in manifest["read_plan"]["phases"]],
                "receipt_sha256": "0" * 64,
            },
        }
        record_path = tmp_path / "record.json"
        record_path.write_text(json.dumps(row))
        adjoint_path = tmp_path / "adjoint.json"
        adjoint_path.write_text("{}")
        (tmp_path / "slice.gz").write_bytes(b"slice")
        return row, record_path, adjoint_path

    # A validated binding admits the row: real argv, executable phases.
    row, record_path, adjoint_path = _row(ref)
    argv = dispatch.quantum_argv(
        row, record_path=record_path, output_root=tmp_path,
        adjoint_path=adjoint_path, binding_validator=validator)
    assert argv[argv.index("--data-manifest") + 1].endswith("exec.json.gz")
    # The same row without a validator keeps the typed gate.
    with pytest.raises(dispatch.ExecutableBindingUnsupported,
                       match="no binding validator"):
        dispatch.quantum_argv(
            row, record_path=record_path, output_root=tmp_path,
            adjoint_path=adjoint_path)
    # A sequencing-only manifest (binding None) keeps the typed gate.
    seq_row, seq_record, seq_adjoint = _row(None)
    with pytest.raises(dispatch.ExecutableBindingUnsupported,
                       match="sequencing-only"):
        dispatch.quantum_argv(
            seq_row, record_path=seq_record, output_root=tmp_path,
            adjoint_path=seq_adjoint, binding_validator=validator)
    # A tampered binding is refused by the real validator at dispatch.
    tampered_row, tampered_record, tampered_adjoint = _row(
        ref, tamper=lambda binding: binding.__setitem__(
            "batch_id", "other-batch"))
    with pytest.raises(dispatch.ExecutableBindingUnsupported,
                       match="not an accepted"):
        dispatch.quantum_argv(
            tampered_row, record_path=tampered_record,
            output_root=tmp_path, adjoint_path=tampered_adjoint,
            binding_validator=validator)


def test_emit_seals_stub_validated_binding_for_every_record(
        tmp_path: Path) -> None:
    """emit_ threads the validator through build_ and bind_ (PQ #870)."""
    from prismaquant.joint_layer_quanta import (
        bind_adjoint_receipt, canonical_sha256,
        emit_quantum_executable_readsets)
    from test_quantum_executable_readset import (  # noqa: E402
        CALIB, N_PROBES, RENDER_PREREQ, STRIDED, _layer2)
    record, receipt, parent = _layer2(tmp_path)
    campaign = record["campaign"]
    digest = bind_adjoint_receipt(
        receipt, plan_sha256=campaign["plan_sha256"],
        prepared_sha256=campaign["prepared_sha256"],
        scope=campaign["campaign_scope"], checkpoints=STRIDED)
    record = dict(
        record, adjoint=dict(record["adjoint"], receipt_sha256=digest))
    body = {k: v for k, v in record.items() if k != "identity_sha256"}
    record["identity_sha256"] = canonical_sha256(body, where="fixture")
    binding = {"scope": "pb732", "material": "d" * 64}
    kwargs = dict(
        strided_boundaries=STRIDED, n_probes=N_PROBES, calib=dict(CALIB),
        render_prerequisite={**RENDER_PREREQ, "binding": dict(binding)},
        output_root=str(tmp_path))
    emitted = emit_quantum_executable_readsets(
        receipt, [record], parent, **kwargs,
        binding_validator=_stub_accept)
    assert len(emitted) == 1
    row = emitted[0]
    assert row["manifest"]["annotations"]["render_prerequisite"][
        "binding"] == binding
    assert row["record"]["executable_readset"][
        "manifest_sha256"] == row["manifest_sha256"]
    # Without a validator the same emission refuses whole: nothing
    # seals, no partial campaign.
    with pytest.raises(ValueError, match="binding validator"):
        emit_quantum_executable_readsets(receipt, [record], parent, **kwargs)
