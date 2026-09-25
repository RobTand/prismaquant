"""The dispatcher compares a row's projection shapes with its kernel's qualification (PQ #1175).

The R13 Stage B plan selects the fused projection kernel ``fused_fp32_v1``.
Its packaged qualification covers six first-model shapes and none of
GLM-5.3-Flash's. Nothing compared the two before a row ran: v7's first
attempt (PB ``80dbab43ee2b``) found out at its first ``finish_observations``,
after 480 s, 60.5 GB read and 43.3 GB written.

Every joint reduction is ``product_sum(operator, weight)`` with both operands
shaped as the target's weight, ``(out, in)`` (``joint_aura.py``). The shapes a
row runs are therefore the weight shapes of its quantum's targets, which the
source checkpoint's safetensors headers state. These tests drive
``dispatch_joint_quanta.main`` with ``--dry-run`` over one layer-2 record whose
targets carry GLM-5.3's routed and shared expert shapes, and a plan that
selects the fused kernel:

* certified mode refuses before anything is submitted, and names the
  unqualified shapes and the qualified set;
* dev mode prints one ``[DEV-MODE]`` line with the number of shapes that will
  run on the reference arithmetic and the shapes themselves, and publishes.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from prismaquant import joint_layer_quanta as jl  # noqa: E402
from prismaquant import joint_projection_backend as backend  # noqa: E402
from prismaquant.layer_streaming import streaming_source_plan  # noqa: E402
from test_quantum_executable_readset import (  # noqa: E402
    CALIB, N_PROBES, RENDER_PREREQ, STRIDED, _bind_slice, _layers,
    _tiny_parent, _tiny_receipt,
)
from test_readset_coverage_1095 import _head_source  # noqa: E402
from test_stageb_prepared_render_inputs import _rebound_record_receipt  # noqa: E402
from test_stageb_readset_source_coverage import (  # noqa: E402
    PREFIX, _source_model, _with_source_paths,
)

#: GLM-5.3-Flash (hidden 4096, moe_intermediate 2048): the routed and shared
#: expert projections of one MoE layer, as its checkpoint headers state them.
LAYER = "model.language_model.layers.2.mlp."
GLM_TARGETS = {
    LAYER + "experts.0.gate_proj": [2048, 4096],
    LAYER + "experts.0.up_proj": [2048, 4096],
    LAYER + "experts.0.down_proj": [4096, 2048],
    LAYER + "shared_experts.down_proj": [4096, 2048],
}
GLM_SHAPES = ((2048, 4096), (4096, 2048))

#: The retained windows the record seals, over the GLM targets.
WINDOW_PAIRS = (
    (0, ((LAYER + "experts.0.gate_proj", "FMT-A"), (LAYER + "experts.0.up_proj", "FMT-A"))),
    (1, ((LAYER + "experts.0.down_proj", "FMT-A"), (LAYER + "shared_experts.down_proj", "FMT-A"))),
)

#: The R13 plan's selector: the packaged binary, bound by its digest. The
#: dispatcher never loads it; ``normalize_projection_backend`` checks the
#: digest against the packaged qualification only.
FUSED = {"name": backend.FUSED_NAME,
         "binary": {"path": "/qualified/pq_joint_projection_reduce.so",
                    "sha256": backend._qualification()[0]["build"]["binary_sha256"]}}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _header_only_checkpoint(root: Path, tensors: dict) -> Path:
    """A checkpoint directory whose one shard holds its header and no payload.

    The dispatcher reads headers only, so the payload the offsets name is
    never read and is not written.
    """
    root.mkdir()
    header, offset = {}, 0
    for name, shape in tensors.items():
        nbytes = 2 * shape[0] * shape[1]
        header[name + ".weight"] = {"dtype": "BF16", "shape": shape,
                                    "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    raw = json.dumps(header).encode()
    shard = "model-00001-of-00001.safetensors"
    (root / shard).write_bytes(struct.pack("<Q", len(raw)) + raw)
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {name: shard for name in header}}))
    return root


def _render_files(tmp_path):
    files = {}
    for window, pairs in WINDOW_PAIRS:
        for name, fmt in pairs:
            path = tmp_path / "renders" / f"{name}.{fmt}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.arange(8, dtype=torch.float32) + window, path)
            raw = path.read_bytes()
            files[(name, fmt)] = {
                "qname": name, "fmt": fmt, "path": str(path), "offset": 0,
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return files


def _prepared_inputs(record, files):
    return {
        "schema": jl.PREPARED_INPUT_SCHEMA,
        "production_pkl_sha256": RENDER_PREREQ["production_pkl_sha256"],
        "unit_roster_sha256": RENDER_PREREQ["unit_roster_sha256"],
        "prepared_sha256": record["campaign"]["prepared_sha256"],
        "windows": [
            {"window_index": window,
             "members": [[name, fmt] for name, fmt in pairs],
             "entries": [dict(files[pair]) for pair in pairs]}
            for window, pairs in WINDOW_PAIRS],
    }


def _campaign(tmp_path, *, projection_backend):
    """A dispatch layout: one executable layer-2 row over GLM-shaped targets."""
    source, spans = _source_model(tmp_path)
    checkpoint = _header_only_checkpoint(tmp_path / "glm", GLM_TARGETS)
    root = str(tmp_path / "run")
    execution = {"operator_windows": {"prefetch_workers": 4}}
    if projection_backend is not None:
        execution["projection_backend"] = projection_backend
    plan = {"model": str(checkpoint), "output_root": root, "distributed_campaign": {},
            "source_prefetch": {"prefetch_lookahead": 1, "prefetch_workers": 1},
            "execution": execution}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True))
    prepared_path = tmp_path / "prep.json"
    prepared_path.write_text(json.dumps({"formats_by_qname": {
        f"model.layers.{layer}.mlp.gate_proj": {} for layer in _layers()}},
        sort_keys=True))
    parent = _tiny_parent()
    records = jl.layer_quanta(
        {**plan, "model": "/fixture/model"}, json.loads(prepared_path.read_text()), parent,
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
    parent = _with_source_paths(parent, source)
    head = _head_source(streaming_source_plan(
        str(source), layers_prefix=PREFIX, layers=range(4)))
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
    return ["--records", str(records_dir), "--output-root", str(out),
            "--adjoint-receipt", str(receipt_path), "--dry-run"]


@pytest.fixture
def dispatch(tmp_path, monkeypatch):
    import dispatch_joint_quanta as dispatch
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"container": {"image": "sha256:" + "0" * 64},
                                "cpu_memory_gb": 28,
                                "env": {"PRISMAQUANT_MAX_GPU_MEM_GB": "72"}}))
    monkeypatch.setattr(dispatch, "SPEC_PATH", spec)
    return dispatch


def _main(dispatch, argv, capsys):
    # The source-coverage check reads the plan's model for the loader's
    # layer spans; this checkpoint holds only the targets' headers, and the
    # coverage check is not under test here.
    code = dispatch.main(argv, _gateway=dispatch.FakeGateway(), _coverage=lambda rows: [])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _shape_lines(out):
    return [line for line in out.splitlines()
            if line.startswith("[DEV-MODE]") and "joint projection qualified shape" in line]


def test_certified_mode_refuses_glm_shapes_before_the_row_runs(tmp_path, dispatch, capsys):
    argv = _campaign(tmp_path, projection_backend=FUSED)
    gateway = dispatch.FakeGateway()
    code = dispatch.main([arg for arg in argv if arg != "--dry-run"], _gateway=gateway,
                         _coverage=lambda rows: [])
    captured = capsys.readouterr()
    assert code == dispatch.EXIT_PRECONDITION_REFUSED, captured.err
    # Nothing reached PrismaBuild: the refusal comes before the submission.
    assert gateway.submitted == []
    assert "[DEV-MODE]" not in captured.out + captured.err
    assert "outside the packaged qualification" in captured.err
    # The message names the unqualified shapes and the qualified set.
    for shape in GLM_SHAPES:
        assert str(shape) in captured.err
    for shape in backend._qualification()[0]["qualified_shapes"]:
        assert str(tuple(shape)) in captured.err


def test_dev_mode_prints_the_glm_shapes_and_publishes(tmp_path, dispatch, monkeypatch, capsys):
    argv = _campaign(tmp_path, projection_backend=FUSED)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    code, out, err = _main(dispatch, argv, capsys)
    assert code == 0, err
    # A dry run's [DEV-MODE] line is a log on stderr; stdout is the plan
    # alone, one JSON document (PQ #1087).
    assert _shape_lines(out) == []
    lines = _shape_lines(err)
    assert len(lines) == 1, err
    # The count of shapes that run the reference arithmetic, and the shapes.
    assert "2 of 2 reduction shapes run the reference arithmetic" in lines[0]
    for shape in GLM_SHAPES:
        assert str(shape) in lines[0]
    rows = json.loads(out)["rows"]
    assert [row["quantum_id"] for row in rows] == ["layer-002"]


@pytest.mark.parametrize("dev", ["0", "1"])
def test_the_reference_backend_accepts_every_shape(tmp_path, dispatch, monkeypatch, capsys, dev):
    argv = _campaign(tmp_path, projection_backend=None)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev)
    code, out, err = _main(dispatch, argv, capsys)
    assert code == 0, err
    assert _shape_lines(out) == [] and _shape_lines(err) == []
    json.loads(out)


# ---- the check itself -------------------------------------------------------


def _plan(model, projection_backend=FUSED):
    execution = {} if projection_backend is None else {"projection_backend": projection_backend}
    return {"model": str(model), "execution": execution}


def _windows(names):
    return {"windows": [{"window_index": 0, "members": [[name, "FMT-A"] for name in names]}]}


@pytest.mark.parametrize("dev", ["0", "1"])
def test_a_qualified_roster_passes_silently(tmp_path, monkeypatch, capsys, dev):
    import dispatch_joint_quanta as dispatch
    # Two of the first model's qualified shapes.
    targets = {"model.layers.0.mlp.gate_proj": [7168, 2048],
               "model.layers.0.mlp.down_proj": [2048, 7168]}
    model = _header_only_checkpoint(tmp_path / "first", targets)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev)
    assert dispatch.check_projection_shapes(
        {"quantum_id": "layer-000", "layer": 0}, plan=_plan(model),
        prepared_input=_windows(targets)) == []
    assert "[DEV-MODE]" not in capsys.readouterr().out


@pytest.mark.parametrize("dev", ["0", "1"])
def test_the_reference_backend_reads_nothing(tmp_path, monkeypatch, dev):
    import dispatch_joint_quanta as dispatch
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev)
    # No checkpoint exists at the plan's model: torch accepts every shape.
    assert dispatch.check_projection_shapes(
        {"quantum_id": "layer-002", "layer": 2},
        plan=_plan(tmp_path / "absent", projection_backend=None),
        prepared_input=_windows(GLM_TARGETS)) == []
    assert dispatch.check_projection_shapes(
        {"quantum_id": "layer-002", "layer": 2},
        plan=_plan(tmp_path / "absent", projection_backend={"name": "torch"}),
        prepared_input=_windows(GLM_TARGETS)) == []


@pytest.mark.parametrize("dev", ["0", "1"])
def test_a_target_the_headers_do_not_hold_refuses_in_both_modes(tmp_path, monkeypatch, dev):
    import dispatch_joint_quanta as dispatch
    model = _header_only_checkpoint(tmp_path / "glm", GLM_TARGETS)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev)
    absent = LAYER + "experts.1.gate_proj"
    with pytest.raises(dispatch.DispatchRefused, match="hold no weight for 1 target"):
        dispatch.check_projection_shapes(
            {"quantum_id": "layer-002", "layer": 2}, plan=_plan(model),
            prepared_input=_windows([*GLM_TARGETS, absent]))


def test_a_record_without_prepared_windows_reads_the_prepared_roster(
        tmp_path, monkeypatch, capsys):
    import dispatch_joint_quanta as dispatch
    model = _header_only_checkpoint(tmp_path / "glm", GLM_TARGETS)
    prepared = tmp_path / "prepared.json"
    # A unit of another layer is not this quantum's target, and the
    # checkpoint above does not hold it.
    prepared.write_text(json.dumps({"formats_by_qname": {
        **{name: ["FMT-A"] for name in GLM_TARGETS},
        "model.language_model.layers.3.mlp.experts.0.gate_proj": ["FMT-A"]}}))
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    assert dispatch.check_projection_shapes(
        {"quantum_id": "layer-002", "layer": 2,
         "campaign": {"prepared_path": str(prepared)}},
        plan=_plan(model), prepared_input=None) == list(GLM_SHAPES)
    assert len(_shape_lines(capsys.readouterr().out)) == 1


def test_a_selector_outside_the_qualification_refuses(tmp_path):
    import dispatch_joint_quanta as dispatch
    foreign = {"name": backend.FUSED_NAME,
               "binary": {"path": "/other.so", "sha256": "0" * 64}}
    with pytest.raises(dispatch.DispatchRefused, match="outside the packaged qualification"):
        dispatch.check_projection_shapes(
            {"quantum_id": "layer-002", "layer": 2}, plan=_plan(tmp_path, foreign),
            prepared_input=_windows(GLM_TARGETS))
