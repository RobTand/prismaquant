"""Exact authenticated acquisition work on the existing campaign renderer seam."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from prismaquant import tessera_campaign as campaign
from prismaquant.model_profiles import DefaultProfile
from prismaquant.production_weight_cache import _cb_cache_tensor_identity

FAMILY = "TESSERA_E4M3_K1"
DEFERRED = "TESSERA_BF16_K1"


def arguments(**changes):
    values = dict(menu_mode="research", max_rounds=1, rate_band=None,
        exhaustive_rate_grid=False, anchors=3, max_artifact_bpp=0,
        anchor_batch_size=2, seed_checkpoint=[], finalize_checkpoint=False,
        acquisition_request="request.json", acquisition_request_sha256="a" * 64,
        census_out=None, capture_calibration_out=None, capture_chain=None,
        research_exact_member=None, units=None)
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.fixture
def domain():
    pytest.importorskip("tessera")
    from prismaquant.tessera_menu import expand_tessera_menu
    names = [f"model.layers.0.self_attn.{leaf}" for leaf in ("q_proj", "k_proj", "v_proj")]
    names += ["model.layers.1.mlp.down_proj"]
    profile = DefaultProfile()
    groups = campaign.resolve_anchor_groups(names, profile=profile, expert_members={})
    weights = {name: torch.arange(32 * 256, dtype=torch.float32).reshape(32, 256).to(torch.bfloat16)
               for name in names}
    menu = expand_tessera_menu((32, 256), mode="research")
    menus = {name: menu for name in names}
    rates = {name: {} for name in names}
    for name in names:
        for rung in menu:
            rates[name].setdefault(rung.family, set()).add(rung.body_rate_q256)
    grids, _ = campaign.anchor_group_rate_grids(groups, rates,
        encode_structure=None, projected_units={})
    legal = sorted(rates[names[0]][FAMILY])
    assert len(legal) >= 5
    qs = [legal[1], legal[len(legal) // 2], legal[-2]]
    requests = {names[0]: {FAMILY: [qs[0]], DEFERRED: []},
                names[1]: {FAMILY: [qs[1]], DEFERRED: []},
                names[2]: {FAMILY: [], DEFERRED: []},
                names[3]: {FAMILY: [qs[2]], DEFERRED: []}}
    return SimpleNamespace(names=names, profile=profile, groups=groups, weights=weights,
        menus=menus, rates=rates, grids=grids, requests=requests, qs=qs)


def schedule(domain, **kwargs):
    return campaign._campaign_round_one_schedule(domain.groups, domain.grids,
        args=kwargs.pop("args", arguments()), audit_units=kwargs.pop("audit_units", set()),
        snap=lambda *_: pytest.fail("acquisition must not snap"),
        requests=kwargs.pop("requests", domain.requests), rates_by_unit=domain.rates, **kwargs)


def packet(domain):
    return dict(requests=domain.requests,
        source_weights={name: _cb_cache_tensor_identity(weight) for name, weight in domain.weights.items()},
        identity={key: str(i) * 64 for i, key in enumerate(("request_sha256", "cost_sha256",
            "joint_aura_identity_sha256", "probe_identity_sha256", "request_control_sha256"), 1)})


def test_nonuniform_requests_union_only_inside_actual_atomic_group(domain):
    actual = schedule(domain)
    assert len(domain.groups) == 2
    for name in domain.names[:3]:
        assert actual[name][FAMILY] == domain.qs[:2]
        assert actual[name][DEFERRED] == []
    assert actual[domain.names[3]][FAMILY] == [domain.qs[2]]
    reordered = {name: dict(reversed(list(domain.requests[name].items()))) for name in reversed(domain.names)}
    assert schedule(domain, requests=reordered) == actual
    keys, scope = campaign._campaign_acquisition_scope(packet(domain), domain.groups)
    assert keys == sorted(domain.groups)
    assert scope == set(domain.names)
    origin = campaign._campaign_acquisition_origin(packet(domain), actual, domain.menus)
    assert origin["expanded_actual_work_count"] == 7
    assert all(DEFERRED in families for families in origin["deferred_domain"].values())
    assert origin["currency"] == campaign.CURRENCY
    assert "path" not in origin


def test_projected_experts_expand_by_actual_stack_group():
    names = ["expert.0.gate_proj", "expert.0.up_proj", "expert.1.gate_proj"]
    members = {name: SimpleNamespace(module_qname="model.layers.0.mlp.experts") for name in names}
    groups = campaign.resolve_anchor_groups(names, profile=DefaultProfile(), expert_members=members)
    assert groups == {"s:model.layers.0.mlp.experts": sorted(names)}
    acquisition = dict(requests={name: {FAMILY: [896] if i == 0 else []}
                                for i, name in enumerate(names)}, source_weights=dict.fromkeys(names, {}))
    keys, scope = campaign._campaign_acquisition_scope(acquisition, groups)
    assert scope == set(names)
    expanded = campaign._campaign_round_one_schedule(groups, {keys[0]: {FAMILY: [896, 1024]}},
        args=arguments(), audit_units=set(), snap=None, requests=acquisition["requests"],
        rates_by_unit={name: {FAMILY: {896, 1024}} for name in names})
    assert all(families[FAMILY] == [896] for families in expanded.values())


@pytest.mark.parametrize("change", ["unknown", "missing_request", "missing_source", "extra_selected", "partial_selected"])
def test_scope_refuses_unknown_missing_atomic_member_and_extra_work(domain, change):
    acquisition = packet(domain)
    selected = set(domain.names)
    if change == "unknown":
        acquisition["requests"] = {**acquisition["requests"], "unknown.unit": {FAMILY: [896]}}
    elif change == "missing_request":
        acquisition["requests"] = {name: r for name, r in domain.requests.items() if name != domain.names[1]}
    elif change == "missing_source":
        acquisition["source_weights"].pop(domain.names[1])
    elif change == "extra_selected":
        selected.add("other.unit")
    else:
        selected.remove(domain.names[1])
    with pytest.raises(ValueError, match="scope|member"):
        campaign._campaign_acquisition_scope(acquisition, domain.groups, selected=selected)


def remove_deferred_grid(domain, absent):
    removed = domain.names if absent == "all" else domain.names[:1]
    for name in removed:
        domain.rates[name].pop(DEFERRED, None)
        domain.menus[name] = [r for r in domain.menus[name] if r.family != DEFERRED]
    domain.grids, _ = campaign.anchor_group_rate_grids(domain.groups, domain.rates,
        encode_structure=None, projected_units={})
    return removed


@pytest.mark.parametrize("absent", ["all", "member"])
def test_absent_empty_family_is_retained_deferred_without_extra_prime_work(domain, tmp_path, absent):
    before = schedule(domain)
    removed = remove_deferred_grid(domain, absent)
    actual = schedule(domain)
    assert actual == before
    origin = campaign._campaign_acquisition_origin(packet(domain), actual, domain.menus)
    assert origin["expanded_actual_work_count"] == 7
    assert all(DEFERRED in origin["deferred_domain"][name] for name in removed)
    assert all(actual[name][DEFERRED] == [] for name in domain.names)
    primed = []
    campaign._prime_first_anchor_batch(SimpleNamespace(prime=primed.append), args=arguments(),
        checkpoint=tmp_path / "anchors.json", targets=domain.names, menus=domain.menus,
        weights=domain.weights, profile=domain.profile, expert_members={},
        encode_structure=None, projected_units={}, audit_units=set(), route_cache={},
        acquisition_requests=domain.requests)
    pending = [(name, family, rate) for key, members in sorted(domain.groups.items())
               for family in sorted({f for name in members for f in actual[name]})
               for name in members for rate in actual[name][family]]
    assert len(pending) == 7
    assert {family for _name, family, _rate in pending} == {FAMILY}
    batches = campaign._anchor_batches(pending, weights=domain.weights, batch_size=2)
    assert primed == [[item[0] for item in batches[0]]]


@pytest.mark.parametrize("absent", ["all", "member"])
def test_absent_family_with_one_positive_request_refuses_before_prime(domain, tmp_path, absent):
    remove_deferred_grid(domain, absent)
    requests = deepcopy(domain.requests)
    requests[domain.names[0]][DEFERRED] = [domain.qs[0]]
    with pytest.raises(ValueError, match="family.*atomic-member grid"):
        schedule(domain, requests=requests)
    with pytest.raises(ValueError, match="family.*atomic-member grid"):
        campaign._prime_first_anchor_batch(
            SimpleNamespace(prime=lambda _: pytest.fail("unsupported positive work primed")),
            args=arguments(), checkpoint=tmp_path / "anchors.json", targets=domain.names,
            menus=domain.menus, weights=domain.weights, profile=domain.profile, expert_members={},
            encode_structure=None, projected_units={}, audit_units=set(), route_cache={},
            acquisition_requests=requests)


@pytest.mark.parametrize("change", ["family", "q", "bool", "missing_grid_member", "route_refused"])
def test_entire_grid_refuses_without_snap_drop_or_substitution(domain, change):
    requests = deepcopy(domain.requests)
    name = domain.names[-1]  # Invalid tail must refuse before any prefix prime.
    if change == "family":
        requests[name]["TESSERA_UNKNOWN_K1"] = [domain.qs[0]]
    elif change == "q":
        requests[name][FAMILY] = [-1]
    elif change == "bool":
        requests[name][FAMILY] = [True]
    elif change == "missing_grid_member":
        domain.rates[domain.names[1]].pop(FAMILY)
    else:
        key = next(key for key, members in domain.groups.items() if name in members)
        domain.grids[key][FAMILY] = [q for q in domain.grids[key][FAMILY] if q != domain.qs[2]]
    with pytest.raises(ValueError, match="family|q256"):
        schedule(domain, requests=requests)


@pytest.mark.parametrize("args,audit", [(arguments(rate_band="896,1024"), set()),
    (arguments(exhaustive_rate_grid=True), set()), (arguments(max_rounds=2), set()),
    (arguments(), {"audited.unit"})])
def test_schedule_refuses_band_audit_exhaustive_and_refinement(domain, args, audit):
    with pytest.raises(ValueError, match="refuses"):
        schedule(domain, args=args, audit_units=audit)


def test_prime_and_round_one_use_the_same_exact_schedule(domain, tmp_path, monkeypatch):
    observed, primed = [], []
    real = campaign._campaign_round_one_schedule
    def record(*args, **kwargs):
        value = real(*args, **kwargs)
        observed.append(value)
        return value
    monkeypatch.setattr(campaign, "_campaign_round_one_schedule", record)
    campaign._prime_first_anchor_batch(SimpleNamespace(prime=primed.append), args=arguments(),
        checkpoint=tmp_path / "anchors.json", targets=domain.names, menus=domain.menus,
        weights=domain.weights, profile=domain.profile, expert_members={},
        encode_structure=None, projected_units={}, audit_units=set(), route_cache={},
        acquisition_requests=domain.requests)
    round_one = schedule(domain)
    assert observed == [round_one, round_one]
    pending = [(name, family, rate) for key, members in sorted(domain.groups.items())
               for family in sorted({f for name in members for f in round_one[name]})
               for name in members for rate in round_one[name][family]]
    batches = campaign._anchor_batches(pending, weights=domain.weights, batch_size=2)
    assert primed == [[item[0] for item in batches[0]]]
    assert len(pending) == 7


def test_prime_validates_bad_tail_before_first_read(domain, tmp_path):
    requests = deepcopy(domain.requests)
    requests[domain.names[-1]][FAMILY] = [-1]
    with pytest.raises(ValueError, match="q256"):
        campaign._prime_first_anchor_batch(SimpleNamespace(prime=lambda _: pytest.fail("partial prefix primed")),
            args=arguments(), checkpoint=tmp_path / "anchors.json", targets=domain.names,
            menus=domain.menus, weights=domain.weights, profile=domain.profile, expert_members={},
            encode_structure=None, projected_units={}, audit_units=set(), route_cache={},
            acquisition_requests=requests)


@pytest.mark.parametrize("flags", [[], ["--menu-mode", "research"],
    ["--menu-mode", "research", "--max-rounds", "2"],
    ["--menu-mode", "research", "--max-rounds", "1", "--rate-band", "896,1024"],
    ["--menu-mode", "research", "--max-rounds", "1", "--exhaustive-rate-grid"],
    ["--menu-mode", "research", "--max-rounds", "1", "--census-out", "census.json"],
    ["--menu-mode", "research", "--max-rounds", "1", "--capture-calibration-out", "capture"],
    ["--menu-mode", "research", "--max-rounds", "1", "--research-exact-member", "q"]])
def test_negative_cli_before_calibration_profile_model_pool_or_outputs(tmp_path, monkeypatch, flags):
    from prismaquant import calibration_data, model_profiles
    from prismaquant import tessera_full_domain_acquisition as intake
    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid acquisition passed early CLI gates")
    monkeypatch.setattr(calibration_data, "load_calibration_input", forbidden)
    monkeypatch.setattr(model_profiles, "detect_profile", forbidden)
    monkeypatch.setattr(campaign, "_start_encoder_source_seal_ahead", forbidden)
    monkeypatch.setattr(intake, "load_joint_campaign_acquisition", forbidden)
    with pytest.raises(SystemExit) as refusal:
        campaign.main(["--model", "unread-model", "--out", str(tmp_path / "out.pkl"),
            "--cache-dir", str(tmp_path / "cache"), "--acquisition-request", "request.json",
            "--acquisition-request-sha256", "a" * 64, *flags])
    assert refusal.value.code == 2
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "out.pkl").exists()


@pytest.mark.parametrize("kind", ["sample", "audit", "partition"])
def test_partial_and_audit_selections_refuse_before_model(tmp_path, monkeypatch, kind):
    import json
    from prismaquant import tessera_full_domain_acquisition as intake, model_profiles
    entry = {"key": "s:experts", "members": ["q", "k"]}
    schema = campaign.UNITS_SCHEMA_V2
    if kind == "partition":
        schema = campaign.UNITS_SCHEMA_V3
        entry["partition"] = dict(schema=campaign.EXPERT_PARTITION_SCHEMA,
            experts_per_row=1, index=0, count=2, rate_q256=896, members=["q"])
    else:
        entry.update(sampled=["q"], inclusion_probability={"q": 0.5})
        if kind == "audit":
            entry["audit"] = ["q"]
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"schema": schema, "groups": [entry]}))
    def forbidden(*_a, **_kw):
        pytest.fail("partial/audit selection passed early gate")
    monkeypatch.setattr(model_profiles, "detect_profile", forbidden)
    monkeypatch.setattr(intake, "load_joint_campaign_acquisition", forbidden)
    with pytest.raises(SystemExit) as refusal:
        campaign.main(["--model", "unread-model", "--out", str(tmp_path / "out.pkl"),
            "--cache-dir", str(tmp_path / "cache"), "--menu-mode", "research", "--max-rounds", "1",
            "--acquisition-request", "request.json", "--acquisition-request-sha256", "a" * 64,
            "--units", str(path)])
    assert refusal.value.code == 2
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("missing", ["acquisition_request", "acquisition_request_sha256"])
def test_request_flags_are_paired(missing):
    import argparse
    args = arguments(**{missing: None})
    with pytest.raises(SystemExit):
        campaign._load_campaign_acquisition(args, argparse.ArgumentParser())


def test_authenticated_loader_runs_before_any_calibration_preparation(tmp_path, monkeypatch):
    from prismaquant import tessera_full_domain_acquisition as intake
    from prismaquant import calibration_data
    calls = []
    def refuse(*, binding):
        calls.append(binding)
        raise ValueError("root request SHA mismatch")
    monkeypatch.setattr(intake, "load_joint_campaign_acquisition", refuse)
    monkeypatch.setattr(calibration_data, "load_calibration_input",
        lambda *_a, **_k: pytest.fail("calibration read before acquisition authentication"))
    with pytest.raises(ValueError, match="SHA mismatch"):
        campaign.main(["--model", "unread-model", "--out", str(tmp_path / "out.pkl"),
            "--cache-dir", str(tmp_path / "cache"), "--menu-mode", "research", "--max-rounds", "1",
            "--acquisition-request", "request.json", "--acquisition-request-sha256", "a" * 64,
            "--calibration-input", "unread-calibration"])
    assert calls == [{"path": "request.json", "sha256": "a" * 64}]
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("field", ["shape", "dtype", "logical_bytes", "content_sha256"])
def test_source_identity_refuses_every_joint_field(field):
    weight = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    expected = _cb_cache_tensor_identity(weight)
    expected[field] = {"shape": [4, 16], "dtype": "torch.bfloat16", "logical_bytes": 128,
                       "content_sha256": "0" * 64}[field]
    with pytest.raises(ValueError, match="source weight identity"):
        campaign._require_campaign_acquisition_source("q", weight, expected)


def test_raw_owned_receipt_avoids_redundant_hash(monkeypatch):
    from prismaquant import production_weight_cache as pwc
    weight = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    receipt = _cb_cache_tensor_identity(weight)
    monkeypatch.setattr(pwc, "_cb_cache_tensor_identity",
        lambda _: pytest.fail("raw owned receipt already supplies the joint identity"))
    campaign._require_campaign_acquisition_source("q", weight, receipt, receipt=receipt)


def test_producer_header_digest_is_not_misrepresented_as_raw_digest(monkeypatch):
    from prismaquant import production_weight_cache as pwc
    weight = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    expected = _cb_cache_tensor_identity(weight)
    real, reads = pwc._cb_cache_tensor_identity, []
    def observe(value):
        reads.append(value)
        return real(value)
    monkeypatch.setattr(pwc, "_cb_cache_tensor_identity", observe)
    campaign._require_campaign_acquisition_source("q", weight, expected,
        receipt=dict(algorithm="sha256.dtype_shape_contiguous.v1", dtype=str(weight.dtype),
                     shape=list(weight.shape), sha256="f" * 64))
    assert len(reads) == 1


def test_checkpoint_seed_extra_rungs_are_refused(domain):
    actual = schedule(domain)
    allowed = SimpleNamespace(qname=domain.names[0], family=FAMILY, body_rate_q256=domain.qs[1])
    campaign._require_campaign_acquisition_anchor(allowed, actual)
    for bad in [SimpleNamespace(qname="extra", family=FAMILY, body_rate_q256=domain.qs[0]),
                SimpleNamespace(qname=domain.names[0], family=DEFERRED, body_rate_q256=domain.qs[0]),
                SimpleNamespace(qname=domain.names[0], family=FAMILY, body_rate_q256=-1)]:
        with pytest.raises(ValueError, match="extra"):
            campaign._require_campaign_acquisition_anchor(bad, actual)


def test_unset_flags_leave_checkpoint_identity_settings_byte_identical(monkeypatch):
    from prismaquant.cost_stage_checkpoint import canonical_json
    from prismaquant import production_weight_cache as pwc
    monkeypatch.setattr(campaign, "_checkpoint_identity_api",
        lambda: SimpleNamespace(encoder_source_sha256=lambda: "a" * 64))
    monkeypatch.setattr(campaign.th, "encoder_recipe", lambda: {})
    monkeypatch.setattr(pwc, "_production_cache_source_sha256", lambda: "b" * 64)
    common = dict(weights={}, acts={}, hessians={}, menus={}, calibration_identity={},
                  serving_scope=None, static_scales={}, static_scale_policy=None)
    old = campaign._campaign_checkpoint_identity(args=SimpleNamespace(), **common)
    new = campaign._campaign_checkpoint_identity(args=SimpleNamespace(
        acquisition_request=None, acquisition_request_sha256=None), **common)
    assert canonical_json(old, where="old") == canonical_json(new, where="new")
    schedule_value = {"q": {FAMILY: [896]}}
    origin = {"request_sha256": "a" * 64, "expanded_actual_work_count": 1}
    bound = [campaign._campaign_checkpoint_identity(args=SimpleNamespace(
        acquisition_request=path, acquisition_request_sha256="a" * 64,
        acquisition_schedule=schedule_value, acquisition_origin=origin), **common)
        for path in ("first/request.json", "second/request.json")]
    assert bound[0] == bound[1]
    assert bound[0]["settings"]["acquisition_schedule"] == schedule_value


def test_real_cpu_scalar_renderer_publishes_exact_requested_wire(tmp_path):
    pytest.importorskip("tessera.export")
    from tessera.unit_artifact import read_unit_artifact
    from prismaquant import tessera_hessian as th
    name, fmt, q = "model.layers.0.proj", "TESSERA_E4M3_K1_R1024", 1024
    weight = (torch.arange(32 * 256, dtype=torch.float32).reshape(32, 256) % 23 - 11).to(torch.bfloat16) / 32
    rows = torch.linspace(-1, 1, 8 * 256).reshape(8, 256)
    source = th.activation_source({name: th.hessian_from_rows(rows)},
        th.calibration_identity("cpu acquisition fixture", [torch.arange(8).reshape(1, 8)], fit_tokens=8))
    requests = {name: {FAMILY: [q], DEFERRED: []}}
    actual = campaign._campaign_round_one_schedule({"u:" + name: [name]},
        {"u:" + name: {FAMILY: [q], DEFERRED: [q]}}, args=arguments(), audit_units=set(),
        snap=None, requests=requests, rates_by_unit={name: {FAMILY: {q}, DEFERRED: {q}}})
    campaign._require_campaign_acquisition_source(name, weight, _cb_cache_tensor_identity(weight))
    cache = SimpleNamespace(weights={}, cache_dir=None)
    anchor = campaign._measure_anchor(qname=name, weight=weight, activations=rows,
        format_name=fmt, cache=cache, wire_dir=tmp_path, hessian_required=True,
        activation_kwargs_for=campaign._activation_kwargs_memo(source, {name: weight}, "cpu"))
    campaign._require_campaign_acquisition_anchor(anchor, actual)
    wire = campaign._wire_path(tmp_path, name, fmt)
    assert wire.is_file() and anchor.wire_bytes == wire.stat().st_size
    decoded = read_unit_artifact(wire.read_bytes(), device="cpu")
    assert torch.equal(decoded.to(torch.bfloat16), cache.weights[name, fmt])
    assert anchor.hessian_applied and anchor.body_rate_q256 == q
    assert actual[name][DEFERRED] == []
    identity = campaign._checkpoint_anchor_identity(anchor, weights={name: weight},
        menus={name: [SimpleNamespace(format_name=fmt)]}, calibration_source=source,
        static_scales={}, projected_units={})
    record = campaign._checkpoint_wire_record(anchor, tmp_path, identity)
    assert campaign._checkpoint_wire_record(anchor, tmp_path, identity, existing=record) == record
