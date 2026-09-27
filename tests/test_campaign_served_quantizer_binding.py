"""The Stage A campaign binds the served A4 quantiser, and a failed rung fails the row.

RobTand/prismaquant#1481: G2 w02 row-0002 priced every TESSERA_E2M1_K2 rung
into ``ServedQuantizerUnboundError`` because nothing on the campaign path bound
the served activation quantiser, then exited 0 with an empty cost table.
"""
import pickle
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from prismaquant import nvfp4_activation_contract as contract  # noqa: E402
from prismaquant import tessera_campaign  # noqa: E402
from tests.test_tessera_campaign_resume import UNIT, _main_fixture  # noqa: E402

A4 = "TESSERA_E2M1_K2_R640"
A16 = "TESSERA_BF16_K1_R1024"


@pytest.fixture(autouse=True)
def unbound_process(monkeypatch):
    """Every test starts in a process that has bound nothing, as production does."""
    contract._reset_served_quantizer_identity_for_tests()
    monkeypatch.setattr(tessera_campaign, "_SERVED_QUANTIZER_RECORD", None)
    yield
    contract._reset_served_quantizer_identity_for_tests()


def _owner_binds_the_model(monkeypatch):
    """Stand in for the registered operator a CPU box cannot load.

    The fake is the owner's own dependency, so the call still goes through
    ``joint_cost_quantum.bind_joint_served_quantizer``: the test proves the
    campaign reaches the Stage B owner, not that it binds by some other route.
    """
    calls = []
    real = contract.bind_served_quantizer_identity

    def bind(**kwargs):
        calls.append(kwargs)
        return real(identity=contract.ServedQuantizerIdentity(
            backend=contract.SERVED_QUANTIZER_BACKEND_MODEL),
            require=False, context=kwargs["context"])

    monkeypatch.setattr(contract, "bind_served_quantizer_identity", bind)
    return calls


def _prepare(format_name):
    return tessera_campaign._prepare_anchor(
        qname=UNIT, format_name=format_name, activation_kwargs_for=None,
        hessian_required=False, static_input_scale=1.0)


def test_an_a4_rung_the_campaign_prepared_can_be_priced(monkeypatch):
    """The #1481 reproduction: before the fix this raised ServedQuantizerUnboundError."""
    calls = _owner_binds_the_model(monkeypatch)
    prepared = _prepare(A4)
    rows = torch.randn(4, 64, generator=torch.Generator().manual_seed(1481))
    assert prepared["activation_qdq"](rows).shape == rows.shape
    assert len(calls) == 1
    assert calls[0]["require"] is True
    assert calls[0]["context"] == tessera_campaign.SERVED_QUANTIZER_CONTEXT
    assert tessera_campaign._SERVED_QUANTIZER_RECORD["backend"] == \
        contract.SERVED_QUANTIZER_BACKEND_MODEL


def test_the_campaign_binds_once_per_process(monkeypatch):
    calls = _owner_binds_the_model(monkeypatch)
    _prepare(A4)
    _prepare("TESSERA_E2M1_K2_R768")
    assert len(calls) == 1


def test_a_missing_operator_refuses_before_the_encode(monkeypatch):
    def refuse(**kwargs):
        raise contract.ServedQuantizerUnboundError("operator unavailable")

    monkeypatch.setattr(contract, "bind_served_quantizer_identity", refuse)
    with pytest.raises(contract.ServedQuantizerUnboundError, match="operator unavailable"):
        _prepare(A4)


def test_an_a16_rung_never_asks_for_the_serving_extension(monkeypatch):
    def unwanted(**kwargs):
        pytest.fail("an A16 rung has no served static activation quantiser")

    monkeypatch.setattr(contract, "bind_served_quantizer_identity", unwanted)
    _prepare(A16)
    assert tessera_campaign._SERVED_QUANTIZER_RECORD is None


def test_a_declared_screen_binding_is_kept(monkeypatch):
    """A CPU screen that declared the model is not re-bound into a refusal."""
    contract.bind_served_quantizer_identity(
        identity=contract.ServedQuantizerIdentity(
            backend=contract.SERVED_QUANTIZER_BACKEND_MODEL),
        require=False, context="pytest (CPU screen)")

    def unwanted(**kwargs):
        pytest.fail("the process already declared its arithmetic")

    monkeypatch.setattr(contract, "bind_served_quantizer_identity", unwanted)
    _prepare(A4)


def _two_rung_row(monkeypatch, tmp_path, fail_rates):
    campaign, _checkpoint, argv, _model, inputs = _main_fixture(
        monkeypatch, tmp_path, priced=True)
    family = "TESSERA_E4M3_K1"
    inputs["menu"] = [SimpleNamespace(
        format_name=f"{family}_R{rate}", family=family, body_rate_q256=rate,
        bpp=rate / 256, admission=SimpleNamespace(activation_contract="a8"),
    ) for rate in (1024, 1536)]
    real = campaign._measure_anchor

    def measure(**kwargs):
        if int(kwargs["format_name"].rsplit("_R", 1)[1]) in fail_rates:
            raise RuntimeError(f"injected failure at {kwargs['format_name']}")
        return real(**kwargs)

    monkeypatch.setattr(campaign, "_measure_anchor", measure)
    return campaign, [*argv, "--anchors", "2", "--anchor-budget", "2",
                      "--max-artifact-bpp", "0"]


@pytest.mark.parametrize("fail_rates", [{1024, 1536}, {1536}], ids=["all", "part"])
def test_a_row_whose_rungs_fail_exits_nonzero_and_writes_no_table(
        monkeypatch, tmp_path, capsys, fail_rates):
    campaign, argv = _two_rung_row(monkeypatch, tmp_path, fail_rates)
    with pytest.raises(RuntimeError, match="injected failure"):
        campaign.main(argv)
    assert not (tmp_path / "cost.pkl").exists()
    assert ": FAILED RuntimeError: injected failure" in capsys.readouterr().out


def test_a_row_whose_rungs_all_price_still_writes_its_table(monkeypatch, tmp_path):
    campaign, argv = _two_rung_row(monkeypatch, tmp_path, set())
    assert campaign.main(argv) == 0
    with (tmp_path / "cost.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    assert sorted(payload["costs"][UNIT]) == ["TESSERA_E4M3_K1_R1024", "TESSERA_E4M3_K1_R1536"]
    assert "served_quantizer" not in payload["provenance"]
