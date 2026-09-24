"""The dispatcher's dry run reads a re-declared plan in dev mode (PQ #1147).

``dispatch_joint_quanta.main`` checks, before it publishes, that every
executable row's readset declares the streaming loader's source reads
(``readset_coverage.quantum_rows_gaps``). That check read the campaign plan
against the digest the record sealed. A plan re-declared after the records
were cut (v7's measured resource plan) failed the comparison, became an
``unreadable`` gap, and the dry run refused in dev mode too. The earlier
tests called ``quantum_argv`` directly or passed an empty coverage function,
so none of them reached the check.

This test drives ``main`` itself with ``--dry-run`` and no injected coverage:
one layer-2 record with an executable readset over a real tiny safetensors
model, real plan and prepared digests, and its stage-A receipt. The plan is
then re-declared at the record's path: the same content in new bytes, so
only the digest differs. Dev mode stamps and publishes; certified mode
refuses with its message unchanged.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from prismaquant import joint_layer_quanta as jl  # noqa: E402
from prismaquant.layer_streaming import streaming_source_plan  # noqa: E402
from test_quantum_executable_readset import (  # noqa: E402
    CALIB, N_PROBES, RENDER_PREREQ, STRIDED, _bind_slice, _layers,
    _tiny_parent, _tiny_receipt,
)
from test_readset_coverage_1095 import _head_source  # noqa: E402
from test_stageb_prepared_render_inputs import (  # noqa: E402
    _prepared_inputs, _rebound_record_receipt, _render_files,
)
from test_stageb_readset_source_coverage import (  # noqa: E402
    PREFIX, _source_model, _with_source_paths,
)

#: The resource policy the plan binds, as ``verify_policy`` returns it. The
#: policy file itself is not under test; the spec below matches its limits.
POLICY = {"limits": {"physical_bytes": 100 << 30, "host_bytes": 28 << 30,
                     "gpu_bytes": 72 << 30}}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign(tmp_path, *, resource_bound):
    """A dispatch layout whose record binds real plan and prepared files."""
    model, spans = _source_model(tmp_path)
    root = str(tmp_path / "run")
    plan = {"model": str(model), "output_root": root, "distributed_campaign": {},
            "source_prefetch": {"prefetch_lookahead": 1, "prefetch_workers": 1},
            "execution": {"operator_windows": {"prefetch_workers": 4}}}
    if resource_bound:
        plan["stage_b_resource_policy"] = {"path": "/resource", "sha256": "0" * 64}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    prepared_path = tmp_path / "prep.json"
    prepared_path.write_text(json.dumps({"formats_by_qname": {
        f"model.layers.{layer}.mlp.gate_proj": {} for layer in _layers()}},
        sort_keys=True))
    parent = _tiny_parent()
    records = jl.layer_quanta(
        plan, json.loads(prepared_path.read_text()), parent,
        chunk_target_bytes=200, stride=2, output_root=root,
        plan_path=str(plan_path), plan_sha256=_sha(plan_path),
        prepared_path=str(prepared_path), prepared_sha256=_sha(prepared_path),
        parent_manifest_sha256="2" * 64,
        window_partition={"windows_by_layer": {str(n): 2 for n in _layers()}},
        ram_window_gib=160, max_resident_consumers=2)["records"]
    record = next(r for r in records if r["layer"] == 2)
    receipt = _tiny_receipt(tmp_path, record["campaign"])
    record, _ = _bind_slice(record, receipt, tmp_path / "adjoint-slices")
    record, receipt = _rebound_record_receipt(record, receipt)
    parent = _with_source_paths(parent, model)
    head = _head_source(streaming_source_plan(
        str(model), layers_prefix=PREFIX, layers=range(4)))
    prepared = _prepared_inputs(record, _render_files(tmp_path))
    inputs = dict(strided_boundaries=STRIDED, n_probes=N_PROBES,
                  calib=dict(CALIB), render_prerequisite=dict(RENDER_PREREQ),
                  layer_source_spans=spans, head_source=head,
                  prepared_inputs=prepared)
    manifest = jl.build_quantum_executable_manifest(record, receipt, parent, **inputs)
    wire = jl.seal_manifest_bytes(manifest)
    manifest_path = Path(root) / "layer-quanta/adjoint/bound-readsets/layer-002.executable.json.gz"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(wire)
    bound = jl.bind_quantum_executable(
        record, receipt, parent, manifest=manifest, manifest_path=str(manifest_path),
        manifest_sha256=hashlib.sha256(wire).hexdigest(), output_root=root, **inputs)
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    (records_dir / "layer-002.json").write_text(json.dumps(bound))
    receipt_path = tmp_path / "adjoint-capture.json"
    receipt_path.write_text(json.dumps(receipt))
    out = tmp_path / "out"
    out.mkdir()
    return plan_path, ["--records", str(records_dir), "--output-root", str(out),
                       "--adjoint-receipt", str(receipt_path), "--dry-run"]


@pytest.fixture
def dispatch(tmp_path, monkeypatch):
    import dispatch_joint_quanta as dispatch
    from prismaquant import joint_stageb_resources as resources
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"container": {"image": "sha256:" + "0" * 64},
                                "cpu_memory_gb": 28,
                                "env": {"PRISMAQUANT_MAX_GPU_MEM_GB": "72"}}))
    monkeypatch.setattr(dispatch, "SPEC_PATH", spec)
    monkeypatch.setattr(resources, "verify_policy", lambda _bound: POLICY)
    return dispatch


def _main(dispatch, argv, capsys):
    code = dispatch.main(argv, _gateway=dispatch.FakeGateway())
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.mark.parametrize("resource_bound", [True, False],
                         ids=["resource-plan", "plain-plan"])
def test_dev_mode_dry_run_reads_a_re_declared_plan(
        tmp_path, dispatch, monkeypatch, capsys, resource_bound):
    plan_path, argv = _campaign(tmp_path, resource_bound=resource_bound)
    # The fixture itself is covered and sealed: certified mode publishes it.
    code, _out, err = _main(dispatch, argv, capsys)
    assert code == 0, err
    sealed = _sha(plan_path)
    plan_path.write_text(json.dumps(json.loads(plan_path.read_text()), indent=2))
    assert _sha(plan_path) != sealed
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    code, out, err = _main(dispatch, argv, capsys)
    assert code == 0, err
    assert f"[DEV-MODE] seal campaign plan differs (readset coverage at {plan_path})" in out
    assert ("[DEV-MODE] seal resource-bound plan differs" in out) is resource_bound
    rows = json.loads(out[out.index("{\n"):])["rows"]
    assert [row["quantum_id"] for row in rows] == ["layer-002"]


@pytest.mark.parametrize("resource_bound", [True, False],
                         ids=["resource-plan", "plain-plan"])
def test_certified_mode_refuses_a_re_declared_plan(
        tmp_path, dispatch, capsys, resource_bound):
    plan_path, argv = _campaign(tmp_path, resource_bound=resource_bound)
    plan_path.write_text(json.dumps(json.loads(plan_path.read_text()), indent=2))
    code, out, err = _main(dispatch, argv, capsys)
    assert code == dispatch.EXIT_PRECONDITION_REFUSED
    assert "[DEV-MODE]" not in out
    if resource_bound:
        # The row's own seal refuses first, as before #1147.
        assert "resource-bound plan differs from its quantum seal" in err
    else:
        # The coverage gap is the one main's check reported before #1147.
        assert (f"campaign inputs: plan at {plan_path} does not hash to its "
                "sealed digest") in err
        assert "1 source read(s) of 1 row(s) are not declared" in err
