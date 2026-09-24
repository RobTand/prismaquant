"""The Stage B replay regime's identity (PQ #994, batched spill and mode 1b).

The default regime must leave the statistics identity exactly as it was, and
any other regime must reach every cost row, so the campaign join refuses a
quantum whose regime differs from the rest.
"""
from __future__ import annotations

import copy
import pickle

import pytest
import torch

from prismaquant.joint_aura import identity_sha256
from prismaquant.joint_projection_backend import REFERENCE_IDENTITY
from prismaquant.joint_quanta_join import JoinRefused, join_joint_quanta
from prismaquant.joint_replay_regime import (
    DEFAULT_REPLAY_REGIME,
    OPERATOR_GEMM_ACCUMULATION,
    REPLAY_REGIME_ENV,
    REPLAY_REGIME_FIELD,
    REPLAY_REGIME_SCHEMA,
    ReplayRegimeRefused,
    normalize_replay_regime,
    replay_regime_from_environment,
    replay_regime_identity,
    replay_regime_of,
    stamp_replay_regime,
)
from prismaquant.joint_statistics_replay import statistics_arithmetic_identity
from tests.test_joint_quanta_allocator_bridge import _generated_outputs
from tests.test_joint_quanta_join import campaign, probe  # noqa: F401 (fixtures)

#: The statistics identity before replay regimes existed, field for field.
PINNED_DEFAULT = {
    "projection_backend": dict(REFERENCE_IDENTITY),
    "projection_dtype": "torch.float32", "delta_dtype": "torch.float32",
    "measurement_dtype": "torch.bfloat16",
    "matmul_precision": "highest", "allow_tf32": False,
    "weight_projection": "summed_output_operator_fp32_gemm",
    "residual": "X_dW_T+dX_W_T+dX_dW_T",
    "aggregation": "sum_signed_invocations_then_square",
    "operator_accumulation": "sum_fp32_matrices_in_backward_invocation_order",
    "contraction_order": "sum_operators_then_project_each_signed_component",
}


@pytest.fixture
def pinned_torch():
    precision = torch.get_float32_matmul_precision()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.set_float32_matmul_precision(precision)
    torch.backends.cuda.matmul.allow_tf32 = tf32


def test_default_regime_leaves_the_statistics_identity_unchanged(pinned_torch):
    unset = statistics_arithmetic_identity(torch.bfloat16, None)
    assert unset == PINNED_DEFAULT
    assert statistics_arithmetic_identity(torch.bfloat16, None, replay_regime=None) == unset
    assert statistics_arithmetic_identity(
        torch.bfloat16, None, replay_regime=dict(DEFAULT_REPLAY_REGIME)) == unset
    assert replay_regime_identity(None) is None
    assert replay_regime_of(unset) == DEFAULT_REPLAY_REGIME


@pytest.mark.parametrize("text,regime", [
    ("capture_batch=4", {"capture_batch": 4, "accumulation": "per_invocation",
                         "chunk_rows": None}),
    ("accumulation=operator_gemm,chunk_rows=65536",
     {"capture_batch": 1, "accumulation": "operator_gemm", "chunk_rows": 65536}),
    ("chunk_rows=4096,accumulation=operator_gemm,capture_batch=16",
     {"capture_batch": 16, "accumulation": "operator_gemm", "chunk_rows": 4096}),
])
def test_non_default_regime_is_stamped(pinned_torch, text, regime):
    assert normalize_replay_regime(text) == regime
    stamped = statistics_arithmetic_identity(torch.bfloat16, None, replay_regime=text)
    assert stamped[REPLAY_REGIME_FIELD] == {"schema": REPLAY_REGIME_SCHEMA, **regime}
    gemm = regime["accumulation"] == "operator_gemm"
    assert stamped["operator_accumulation"] == (
        OPERATOR_GEMM_ACCUMULATION if gemm else PINNED_DEFAULT["operator_accumulation"])
    rest = {key: value for key, value in stamped.items()
            if key not in (REPLAY_REGIME_FIELD, "operator_accumulation")}
    assert rest == {key: value for key, value in PINNED_DEFAULT.items()
                    if key != "operator_accumulation"}
    assert replay_regime_of(stamped) == regime
    assert identity_sha256(stamped) != identity_sha256(PINNED_DEFAULT)


def test_each_regime_field_moves_the_identity(pinned_torch):
    texts = ["capture_batch=2", "capture_batch=4",
             "accumulation=operator_gemm,chunk_rows=1024",
             "accumulation=operator_gemm,chunk_rows=2048",
             "capture_batch=2,accumulation=operator_gemm,chunk_rows=1024"]
    digests = {identity_sha256(statistics_arithmetic_identity(
        torch.bfloat16, None, replay_regime=text)) for text in texts}
    assert len(digests) == len(texts)


@pytest.mark.parametrize("text,message", [
    ("", "empty"),
    (" capture_batch=2", "padded"),
    ("capture_batch=2,", "at most once"),
    ("capture_batch=2,capture_batch=2", "at most once"),
    ("batch=2", "at most once"),
    ("capture_batch", "at most once"),
    ("capture_batch=0", "positive integer"),
    ("capture_batch=02", "canonical integer"),
    ("capture_batch=-1", "canonical integer"),
    ("capture_batch=two", "canonical integer"),
    ("accumulation=sum", "not one of"),
    ("accumulation=operator_gemm", "explicit positive chunk_rows"),
    ("accumulation=operator_gemm,chunk_rows=0", "explicit positive chunk_rows"),
    ("chunk_rows=4096", "only to operator_gemm"),
])
def test_malformed_regime_is_refused(text, message):
    with pytest.raises(ReplayRegimeRefused, match=message):
        normalize_replay_regime(text)


def test_mapping_regime_refuses_unknown_fields_and_bool_batch():
    with pytest.raises(ReplayRegimeRefused, match="unknown"):
        normalize_replay_regime({"capture_batch": 2, "mode": "1b"})
    with pytest.raises(ReplayRegimeRefused, match="positive integer"):
        normalize_replay_regime({"capture_batch": True})


@pytest.mark.parametrize("text", [
    "capture_batch=1", "accumulation=per_invocation",
    "capture_batch=1,accumulation=per_invocation"])
def test_the_default_spelled_out_is_refused_at_launch(text):
    assert replay_regime_from_environment({}) is None
    with pytest.raises(ReplayRegimeRefused, match="spells the default"):
        replay_regime_from_environment({REPLAY_REGIME_ENV: text})


def test_a_stamped_default_or_inconsistent_stamp_is_refused(pinned_torch):
    default_block = {"schema": REPLAY_REGIME_SCHEMA, **DEFAULT_REPLAY_REGIME}
    with pytest.raises(ReplayRegimeRefused, match="stamped default"):
        replay_regime_of({**PINNED_DEFAULT, REPLAY_REGIME_FIELD: default_block})
    with pytest.raises(ReplayRegimeRefused, match="malformed"):
        replay_regime_of({**PINNED_DEFAULT, REPLAY_REGIME_FIELD: {"capture_batch": 2}})
    gemm = stamp_replay_regime(PINNED_DEFAULT, "accumulation=operator_gemm,chunk_rows=8")
    with pytest.raises(ReplayRegimeRefused, match="disagrees"):
        replay_regime_of({**gemm, "operator_accumulation":
                          PINNED_DEFAULT["operator_accumulation"]})
    with pytest.raises(ReplayRegimeRefused, match="disagrees"):
        replay_regime_of({**PINNED_DEFAULT, "operator_accumulation": OPERATOR_GEMM_ACCUMULATION})
    with pytest.raises(ReplayRegimeRefused, match="already carries"):
        stamp_replay_regime(stamp_replay_regime(PINNED_DEFAULT, "capture_batch=2"),
                            "capture_batch=4")


# ---- the campaign join -----------------------------------------------------

def _restamp(root, quantum, restamp):
    """Rewrite one quantum's cost rows as if its arithmetic were ``restamp``'s.

    Every digest that covers the arithmetic is recomputed, so each row still
    validates on its own; only the join sees the difference.
    """
    path = root / "layer-quanta" / quantum / "cost.pkl"
    payload = pickle.loads(path.read_bytes())
    for rows in payload["costs"].values():
        for fmt, entry in list(rows.items()):
            if not isinstance(entry, dict) or "error" in entry:
                continue
            entry = copy.deepcopy(entry)
            probe_identity = entry["probe_identity"]
            operator = entry["joint_operator_identity"]
            probe_identity["arithmetic"] = restamp(probe_identity["arithmetic"])
            operator["arithmetic"] = copy.deepcopy(probe_identity["arithmetic"])
            digest = identity_sha256(probe_identity)
            entry["probe_identity_sha256"] = operator["probe_identity_sha256"] = digest
            entry["joint_operator_identity_sha256"] = identity_sha256(operator)
            rows[fmt] = entry
    path.write_bytes(pickle.dumps(payload))
    return payload


def _quanta(root):
    return sorted(path.name for path in (root / "layer-quanta").iterdir()
                  if path.name != "records")


def test_join_refuses_a_quantum_whose_replay_regime_differs(tmp_path, campaign):
    root, campaign, _, _, _ = _generated_outputs(tmp_path, campaign)
    quanta = _quanta(root)
    assert len(quanta) == 3
    _restamp(root, quanta[1], lambda arithmetic: stamp_replay_regime(
        arithmetic, "capture_batch=2"))
    with pytest.raises(JoinRefused, match="probe or measurement identity differs"):
        join_joint_quanta(receipts=None, campaign=campaign, input_root=root,
                          output_dir=tmp_path / "refused")


def test_join_accepts_one_uniform_replay_regime(tmp_path, campaign):
    root, campaign, _, _, _ = _generated_outputs(tmp_path, campaign)
    regime = "capture_batch=4,accumulation=operator_gemm,chunk_rows=65536"
    for quantum in _quanta(root):
        _restamp(root, quantum, lambda arithmetic: stamp_replay_regime(arithmetic, regime))
    result = join_joint_quanta(receipts=None, campaign=campaign, input_root=root,
                               output_dir=tmp_path / "joined")
    assert result["status"] == "complete"
    joined = pickle.loads(open(result["joint_cost_path"], "rb").read())
    for rows in joined["costs"].values():
        for entry in rows.values():
            assert replay_regime_of(entry["probe_identity"]["arithmetic"]) == (
                normalize_replay_regime(regime))


def test_a_stamped_default_row_fails_its_currency_check(tmp_path, campaign):
    root, campaign, _, _, _ = _generated_outputs(tmp_path, campaign)
    block = {"schema": REPLAY_REGIME_SCHEMA, **DEFAULT_REPLAY_REGIME}
    _restamp(root, _quanta(root)[0],
             lambda arithmetic: {**arithmetic, REPLAY_REGIME_FIELD: block})
    with pytest.raises(JoinRefused, match="stamped default"):
        join_joint_quanta(receipts=None, campaign=campaign, input_root=root,
                          output_dir=tmp_path / "refused")


# ---- the launcher ----------------------------------------------------------

def test_launcher_reads_the_regime_from_the_environment_only(monkeypatch):
    """The plan cannot carry a regime, and the default cannot be spelled out.

    Both refusals come before any source, cache or GPU work.
    """
    import prismaquant.gpu_guard as gpu_guard
    from prismaquant.joint_cost_quantum import QuantumIdentityRefused, run_layer_quantum

    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *args, **kwargs: None)
    arguments = dict(record={}, adjoint_slice={}, plan_sha256="", prepared="",
                     output_root="")
    monkeypatch.delenv(REPLAY_REGIME_ENV, raising=False)
    with pytest.raises(QuantumIdentityRefused, match="launch setting"):
        run_layer_quantum({"execution": {"replay_regime": "capture_batch=2"}}, **arguments)
    monkeypatch.setenv(REPLAY_REGIME_ENV, "capture_batch=1")
    with pytest.raises(QuantumIdentityRefused, match="spells the default"):
        run_layer_quantum({"execution": {}}, **arguments)
    monkeypatch.setenv(REPLAY_REGIME_ENV, "capture_batch=x")
    with pytest.raises(QuantumIdentityRefused, match="canonical integer"):
        run_layer_quantum({"execution": {}}, **arguments)
    # A band-serial producer (#996) captures at its slice's chain batch size
    # (PQ #997), and refuses before it binds its handoff publication: a slice
    # without a run identity has no chain regime to compare with, and an
    # unstamped identity rolls at batch 1.
    monkeypatch.setenv(REPLAY_REGIME_ENV, "capture_batch=2")
    with pytest.raises(QuantumIdentityRefused, match="chain regime"):
        run_layer_quantum({"execution": {}}, emit_handoff=True, **arguments)
    batch_one = dict(arguments, adjoint_slice={"run_identity": {}})
    with pytest.raises(QuantumIdentityRefused,
                       match="capture_batch=2 .* batch size 1; .* capture at batch 1"):
        run_layer_quantum({"execution": {}}, emit_handoff=True, **batch_one)


def test_the_container_forwards_the_sealed_regime_verbatim():
    """The regime rides the spec's ``env``, which the container forwards as is."""
    from tools.tessera_campaign_container import docker_command

    regime = "capture_batch=2,accumulation=operator_gemm,chunk_rows=4096"
    spec = {"model": "/mnt/shared/model", "cwd": "/original/checkout",
            "python": "python3", "campaign_argv": [],
            "env": {"PYTHONPATH": ".", REPLAY_REGIME_ENV: regime},
            "container": {"image": "qualified:fixed", "mounts": [
                {"source": "/mnt/shared", "target": "/mnt/shared"}]}}
    argv = docker_command(spec, ["python3"], cwd="/snapshot", uid=1, gid=1,
                          image_id="sha256:x", environ={})
    assert f"{REPLAY_REGIME_ENV}={regime}" in argv


@pytest.mark.parametrize("regime,chain_batch_size,refused", [
    (None, 1, False),
    ("accumulation=operator_gemm,chunk_rows=65536", 1, False),
    ("capture_batch=2", 1, True),
    ({"capture_batch": 16, "accumulation": "operator_gemm", "chunk_rows": 7}, 1, True),
    # The campaign: capture batch 4 under R13's batch-4 chain regime.
    ("capture_batch=4,accumulation=operator_gemm,chunk_rows=65536", 4, False),
    ("capture_batch=4,accumulation=operator_gemm,chunk_rows=65536", 2, True),
    (None, 4, True),
    ({"capture_batch": 16, "accumulation": "operator_gemm", "chunk_rows": 7}, 16, False),
])
def test_a_band_serial_producer_admits_only_the_chains_batch_size(
        regime, chain_batch_size, refused):
    """The handed-off plane is the chain's plane exactly when the capture
    batch equals the slice's chain batch size (PQ #994, #996, #997)."""
    from prismaquant.joint_replay_regime import (
        ReplayRegimeRefused, handoff_regime_refusal, normalize_replay_regime)

    reason = handoff_regime_refusal(regime, chain_batch_size=chain_batch_size)
    assert (reason is not None) is refused
    if refused:
        batch = normalize_replay_regime(regime)["capture_batch"]
        assert f"capture_batch={batch}" in reason
        assert f"batch size {chain_batch_size}" in reason
        assert f"capture at batch {chain_batch_size}" in reason
    with pytest.raises(ReplayRegimeRefused, match="chain batch size"):
        handoff_regime_refusal(regime, chain_batch_size=0)
    with pytest.raises(TypeError):
        handoff_regime_refusal(regime)  # the chain batch size is never implied
