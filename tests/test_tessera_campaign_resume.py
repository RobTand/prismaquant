"""A resumed cost must belong to this run's actual encoding inputs."""
import hashlib
import json
import pickle
import shutil
import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

UNIT = "model.layers.0.proj"


def _main_fixture(monkeypatch, tmp_path, *, priced=False):
    from prismaquant import model_profiles, tessera_campaign, tessera_render
    from prismaquant.model_profiles import DefaultProfile

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].proj = torch.nn.Linear(256, 32, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        model.model.layers[0].proj.weight.copy_(
            torch.randn(32, 256, generator=torch.Generator().manual_seed(186)))
    inputs = {
        "tokens": [torch.ones(1, 256, dtype=torch.long)],
        "text": "one draw",
        "rows": torch.randn(4, 256, generator=torch.Generator().manual_seed(183)),
        "hessian": torch.eye(256),
        # The calibration maximum behind the unit's static input_global_scale;
        # a scoring input of every W4A4 row, bound like the rows themselves.
        "max_abs": 3.0,
        "menu": ([SimpleNamespace(
            format_name="TESSERA_E4M3_K1_R1024", family="TESSERA_E4M3_K1",
            body_rate_q256=1024, bpp=4.0)] if priced else []),
    }
    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: model)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    # The fresh run prices under the default static-scale policy so a test
    # can change the policy afterwards and see the identity refuse it.
    monkeypatch.delenv("PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE", raising=False)
    monkeypatch.setattr(model_profiles, "detect_profile", lambda _path: DefaultProfile())
    monkeypatch.setattr(tessera_render, "tessera_encoder_hessian_status", lambda: {
        "accepted": True, "reason": "CPU test fixture", "kwargs": [], "recipe": {},
    })
    monkeypatch.setattr(tessera_campaign, "_calibration_tokens",
                        lambda *_args: (inputs["tokens"], inputs["text"]))
    monkeypatch.setattr(tessera_campaign, "_collect_activations", lambda *_args, **kwargs: (
        {UNIT: inputs["rows"]},
        {UNIT: inputs["hessian"]} if kwargs["want_hessian"] else {},
        {UNIT: len(inputs["rows"]) if kwargs["want_hessian"] else 0},
        {UNIT: float(inputs["max_abs"])},
    ))
    monkeypatch.setattr(tessera_campaign, "expand_menus_for_targets",
                        lambda _weights, targets, **_kwargs: {
                            name: inputs["menu"] for name in targets})

    def unverified_anchors_reached_payload(anchors, *_args, **_kwargs):
        assert not anchors, "unverified checkpoint anchors reached the current cost payload"
        return {**_kwargs["provenance"], "costs": {}, "formats": []}

    if not priced:
        monkeypatch.setattr(tessera_campaign, "campaign_cost_payload",
                            unverified_anchors_reached_payload)
    checkpoint = tmp_path / "campaign.anchors.json"
    argv = ["--model", "synthetic-current-model", "--out", str(tmp_path / "cost.pkl"),
            "--cache-dir", str(tmp_path / "cache"), "--checkpoint", str(checkpoint),
            "--hessian", "off", "--menu-mode", "research", "--max-rounds", "1"]
    return tessera_campaign, checkpoint, argv, model, inputs


@pytest.mark.parametrize("checkpoint_unit", [UNIT, "model.layers.8.other_model"])
def test_main_refuses_unbound_checkpoint_before_accepting_anchors(
    monkeypatch, tmp_path, checkpoint_unit,
):
    campaign, checkpoint, argv, _model, _inputs = _main_fixture(monkeypatch, tmp_path)
    anchor = campaign.CampaignAnchor(
        qname=checkpoint_unit, family="TESSERA_E4M3_K1",
        format_name="TESSERA_E4M3_K1_R1024", body_rate_q256=1024,
        dloss=0.25, dloss_stderr=0.0, memory_bytes=8, bits_per_param=4.0,
        activation_contract="a8", activation_quantized=True, wire_bytes=8,
        seconds=1.0, hessian_applied=False,
    )
    checkpoint.write_text(json.dumps({"schema": campaign.SCHEMA, "anchors": [vars(anchor)]}))
    with pytest.raises(RuntimeError, match="checkpoint.*identity|resume.*identity"):
        campaign.main(argv)


def _fresh_priced_campaign(monkeypatch, tmp_path, *, hessian=False):
    fixture = _main_fixture(monkeypatch, tmp_path, priced=True)
    campaign, checkpoint, argv, _model, _inputs = fixture
    if hessian:
        argv[argv.index("--hessian") + 1] = "require"
    assert campaign.main(argv) == 0
    return fixture, _priced_cost_payload(tmp_path)


def _priced_cost_payload(tmp_path):
    """Check the actual measured table before a consumer receives it."""
    with (tmp_path / "cost.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    assert payload["costs"][UNIT]["TESSERA_E4M3_K1_R1024"]["output_mse_measured"]
    return payload


def _priced_file_digests(root):
    """The completed producer's exact file bytes, without path relocation."""
    return {path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture(scope="module")
def _completed_priced_campaigns(tmp_path_factory):
    """Two real baselines, invalidated with this pytest module's lifetime.

    Build lazily inside a consumer's normal CPU/provenance fixture context.
    Retain only completed files and their byte digests; model/input tensors,
    monkeypatches and open owners never survive setup. Refinement cases still
    call the real campaign independently.
    """
    completed = {}

    def baseline(hessian):
        if hessian not in completed:
            root = tmp_path_factory.mktemp(f"priced-campaign-{int(hessian)}")
            with pytest.MonkeyPatch.context() as patch:
                _fresh_priced_campaign(patch, root, hessian=hessian)
            completed[hessian] = root, _priced_file_digests(root)
        return completed[hessian]

    return baseline


@pytest.fixture
def priced_campaign(monkeypatch, tmp_path, _completed_priced_campaigns):
    """Each mutation owns fresh inputs and private copies of actual bytes.

    Run identities omit cache/output locations; unit receipts name files by
    basename. Copy the whole completed tree unchanged, including journal unit
    envelopes and export inputs, so every original resume/seed/byte gate still
    verifies the same real producer output under the consumer's own paths.
    """
    used = []

    def prepare(*, hessian=False):
        root, digests = _completed_priced_campaigns(hessian)
        assert _priced_file_digests(root) == digests, "the priced baseline was mutated"
        shutil.copytree(root, tmp_path, dirs_exist_ok=True)
        assert _priced_file_digests(tmp_path) == digests
        for relative in digests:
            original = (root / relative).stat()
            private = (tmp_path / relative).stat()
            assert (original.st_dev, original.st_ino) != (private.st_dev, private.st_ino), (
                "resume mutations must own private file inodes", relative)
        used.append((root, digests))
        # _main_fixture creates every model/input tensor anew with the same
        # explicit seeds. Mutating W, X, H or tokens cannot reach another case.
        fixture = _main_fixture(monkeypatch, tmp_path, priced=True)
        if hessian:
            fixture[2][fixture[2].index("--hessian") + 1] = "require"
        return fixture, _priced_cost_payload(tmp_path)

    yield prepare
    for root, digests in used:
        assert _priced_file_digests(root) == digests, "a resume consumer mutated the baseline"


@pytest.mark.parametrize("initial,rounds,budget,rates,expected", [
    (1, 2, 3, [1024, 1280, 1536], [1024, 1280, 1536]),
    (2, 2, 3, [1024, 1280, 1536], [1024, 1280, 1536]),
    (2, 1, 3, [1024, 1280, 1536], [1024, 1536]),
    (2, 2, 2, [1024, 1280, 1536], [1024, 1536]),
    (2, 2, 3, [1024, 1536], [1024, 1536]),
    (2, 2, 3, [1024], [1024]),
])
def test_main_bootstraps_loo_from_two_endpoints(
    monkeypatch, tmp_path, initial, rounds, budget, rates, expected,
):
    """One/two requested initial anchors still refine when there is room."""
    campaign, _checkpoint, argv, _model, inputs = _main_fixture(
        monkeypatch, tmp_path, priced=True)
    family = "TESSERA_E4M3_K1"
    inputs["menu"] = [SimpleNamespace(
        format_name=f"{family}_R{rate}", family=family,
        body_rate_q256=rate, bpp=rate / 256,
        admission=SimpleNamespace(activation_contract="a8"),
    ) for rate in rates]
    argv[argv.index("--max-rounds") + 1] = str(rounds)
    assert campaign.main([
        *argv, "--anchors", str(initial), "--anchor-budget", str(budget),
        "--max-artifact-bpp", "0",
    ]) == 0
    with (tmp_path / "cost.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    surface = payload["provenance"]["surfaces"][UNIT][family]
    assert surface["rungs"] == expected
    assert surface["anchors"] == len(expected)
    for rate in expected:
        assert payload["costs"][UNIT][f"{family}_R{rate}"]["output_mse_measured"]
    if len(expected) < 3:
        assert surface["loo_max_abs_log2_error"] is None
        assert surface["gate_closed"] is False


def _forbid_reencode(monkeypatch, campaign):
    def forbidden(**_kwargs):
        pytest.fail("resume attempted another encode instead of validating the priced bytes")
    monkeypatch.setattr(campaign, "_measure_anchor", forbidden)


def test_main_resumes_identical_cost_and_wire_without_reencoding(
        monkeypatch, tmp_path, priced_campaign):
    from prismaquant.cost_stage_checkpoint import MANIFEST_SCHEMA

    (campaign, checkpoint, argv, _model, _inputs), initial = priced_campaign()
    manifest = json.loads(checkpoint.read_text())
    assert manifest["schema"] == MANIFEST_SCHEMA
    original_manifest = checkpoint.read_bytes()
    wire = next((tmp_path / "cache" / "wire").glob("*.tessera"))
    original_wire = wire.read_bytes()
    _forbid_reencode(monkeypatch, campaign)
    # Output location and interruption limit are not encoding/scoring inputs.
    argv[argv.index("--out") + 1] = str(tmp_path / "resumed.pkl")
    assert campaign.main([*argv, "--deadline-seconds", "1"]) == 0
    with (tmp_path / "resumed.pkl").open("rb") as handle:
        resumed = pickle.load(handle)
    assert resumed["costs"] == initial["costs"]
    assert checkpoint.read_bytes() == original_manifest
    assert wire.read_bytes() == original_wire


def test_main_registers_resumed_wire_before_receipt_read(
        monkeypatch, tmp_path, priced_campaign):
    from prismaquant import production_weight_cache as pwc

    (campaign, _checkpoint, argv, _model, inputs), _payload = priced_campaign()
    caches = []
    cache_type = pwc.ProductionWeightCache
    verify = campaign._checkpoint_wire_record
    expected = {(UNIT, inputs["menu"][0].format_name)}

    def capture_cache(**kwargs):
        cache = cache_type(**kwargs)
        caches.append(cache)
        return cache

    reads = []

    def read_wire(*args, **kwargs):
        assert caches[-1].weights == {}
        assert getattr(caches[-1], "_campaign_wire_coordinates", set()) == expected, (
            "resume did not register its wire before the receipt read")
        reads.append(args)
        return verify(*args, **kwargs)

    monkeypatch.setattr(pwc, "ProductionWeightCache", capture_cache)
    monkeypatch.setattr(campaign, "_checkpoint_wire_record", read_wire)
    _forbid_reencode(monkeypatch, campaign)
    assert campaign.main(argv) == 0
    assert len(reads) == 1
    assert caches[-1]._campaign_wire_coordinates == expected


def test_main_refuses_changed_hessian_values_under_same_draw(
        monkeypatch, tmp_path, priced_campaign):
    (campaign, checkpoint, argv, _model, inputs), _payload = priced_campaign(hessian=True)
    original_manifest = checkpoint.read_bytes()
    _forbid_reencode(monkeypatch, campaign)
    inputs["hessian"][0, 0] += 1
    with pytest.raises(RuntimeError, match="checkpoint identity mismatch"):
        campaign.main(argv)
    assert checkpoint.read_bytes() == original_manifest


@pytest.mark.parametrize("changed", [
    "weight", "scoring_rows", "input_scale", "scale_policy", "corpus", "tokens",
    "hessian_mode", "menu", "recipe", "encoder_source", "prismaquant_source", "scope",
])
def test_main_refuses_changed_encoding_or_scoring_inputs(
        monkeypatch, tmp_path, changed, priced_campaign):
    (campaign, checkpoint, argv, model, inputs), _payload = priced_campaign()
    original_manifest = checkpoint.read_bytes()
    _forbid_reencode(monkeypatch, campaign)
    if changed == "weight":
        with torch.no_grad():
            model.model.layers[0].proj.weight[0, 0] += 1
    elif changed == "scoring_rows":
        inputs["rows"][0, 0] += 1
    elif changed == "input_scale":
        # The served A-side contract is a scoring input: another calibration
        # maximum is another static input_global_scale for the unit.
        inputs["max_abs"] *= 2
    elif changed == "scale_policy":
        # ...and so is the env-resolved policy that turns the maximum into it.
        monkeypatch.setenv("PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE", "1")
    elif changed == "corpus":
        inputs["text"] = "different corpus, same selected tokens"
    elif changed == "tokens":
        inputs["tokens"][0][0, 0] += 1
    elif changed == "hessian_mode":
        argv[argv.index("--hessian") + 1] = "require"
    elif changed == "menu":
        inputs["menu"] = []
    elif changed == "recipe":
        recipe = campaign.th.encoder_recipe()
        monkeypatch.setattr(campaign.th, "encoder_recipe", lambda: {**recipe, "changed": True})
    elif changed == "encoder_source":
        from tessera import cached_unit
        monkeypatch.setattr(cached_unit, "encoder_source_sha256", lambda: "0" * 64)
    elif changed == "prismaquant_source":
        from prismaquant import production_weight_cache
        monkeypatch.setattr(production_weight_cache, "_production_cache_source_sha256",
                            lambda: "0" * 64)
    elif changed == "scope":
        argv.extend(["--tp-degree", "2"])
    with pytest.raises(RuntimeError, match="checkpoint identity mismatch"):
        campaign.main(argv)
    assert checkpoint.read_bytes() == original_manifest


def _export_inputs_state(cache_dir):
    """What the export leg would be handed, read back from the cache."""
    from safetensors.torch import load_file

    capture = torch.load(cache_dir / "hessian_capture.pt", weights_only=False)
    sidecar = json.loads(
        (cache_dir / "hessian_capture.pt.provenance.json").read_text())
    scales = load_file(str(cache_dir / "input_scales.safetensors"))
    return {
        "hessian": float(capture["H"][UNIT][0, 0]),
        "capture_provenance": capture["provenance"],
        "capture_sha256": sidecar["capture_sha256"],
        "input_global_scale": float(scales[f"{UNIT}.input_global_scale"]),
    }


def test_refused_resume_leaves_the_surviving_tables_export_inputs(monkeypatch, tmp_path):
    """A refused resume must not have rewritten the export leg's inputs first.

    The checkpoint and the previous cost file already survive a refusal, but
    ``hessian_capture.pt``, its sidecar and ``input_scales.safetensors`` are
    the export leg's half of the same surviving table: destroy them and the
    table that survived cannot be exported at all (#211).  The pytest form of
    ``pq-audit-caches/pq-204/proofs/rejected_resume_postfix.py``; it needs
    Tessera's ``cached_unit`` receipt API to reach ``main()``'s resume.
    The menu is left empty (the ``priced=False`` fixture) so the refusal
    exercises run-level identity independently of route admission.  Since
    PrismaQuant #291 that empty menu is itself a refusal
    (``EXIT_EMPTY_MENU``), which sharpens rather than weakens what is being
    pinned here: the capture and the scales are facts about the CALIBRATION,
    so a run that refuses on its menu still leaves them behind for the next
    run, while a run that refuses on its identity must not overwrite them.
    """
    pytest.importorskip("tessera.cached_unit")
    campaign, checkpoint, argv, _model, inputs = _main_fixture(monkeypatch, tmp_path)
    argv[argv.index("--hessian") + 1] = "require"
    # Refused on the menu -- and no cost table written, which is #291 itself.
    assert campaign.main(argv) == campaign.EXIT_EMPTY_MENU
    assert not (tmp_path / "cost.pkl").exists()
    cache_dir = tmp_path / "cache"
    before = _export_inputs_state(cache_dir)
    original_manifest = checkpoint.read_bytes()
    # Another draw's Hessian and another static A-side scale: exactly the run
    # the run-level checkpoint identity refuses.  It must refuse THERE, before
    # it reaches the menu, and without having rewritten the export inputs on
    # the way.
    inputs["hessian"] = 2 * torch.eye(256)
    inputs["max_abs"] *= 2
    with pytest.raises(RuntimeError, match="checkpoint identity mismatch"):
        campaign.main(argv)
    assert checkpoint.read_bytes() == original_manifest
    assert not (tmp_path / "cost.pkl").exists()
    assert _export_inputs_state(cache_dir) == before


@pytest.mark.parametrize("damage", ["missing", "bytes", "symlink"])
def test_main_refuses_missing_or_changed_priced_wire(
        monkeypatch, tmp_path, damage, priced_campaign):
    (campaign, checkpoint, argv, _model, _inputs), _payload = priced_campaign()
    _forbid_reencode(monkeypatch, campaign)
    original_manifest = checkpoint.read_bytes()
    wire = next((tmp_path / "cache" / "wire").glob("*.tessera"))
    if damage == "missing":
        wire.unlink()
    elif damage == "bytes":
        blob = bytearray(wire.read_bytes())
        blob[-1] ^= 1
        wire.write_bytes(blob)
    else:
        moved = wire.with_suffix(".elsewhere")
        wire.rename(moved)
        wire.symlink_to(moved.name)
    with pytest.raises(RuntimeError, match="checkpoint cached wire"):
        campaign.main(argv)
    assert checkpoint.read_bytes() == original_manifest


@pytest.mark.parametrize('changed', [False, True])
def test_seed_refuses_changed_scoring_rows_before_linking_wire(
        monkeypatch, tmp_path, changed, priced_campaign):
    (campaign, checkpoint, argv, _model, inputs), _payload = priced_campaign(hessian=True)
    original_manifest = checkpoint.read_bytes()
    _forbid_reencode(monkeypatch, campaign)
    # The encoder still sees identical W, H, draw and static scale. Only the
    # bounded rows used to score its decoded weight have changed.
    if changed:
        inputs['rows'][0, 0] += 1
    new_cache = tmp_path/'new-cache'
    argv[argv.index('--checkpoint')+1] = str(tmp_path/'new.anchors.json')
    argv[argv.index('--cache-dir')+1] = str(new_cache)
    argv[argv.index('--out')+1] = str(tmp_path/'new-cost.pkl')
    if changed:
        with pytest.raises(RuntimeError, match='seed.*scoring_rows'):
            campaign.main([*argv, '--seed-checkpoint', str(checkpoint)])
        assert list((new_cache/'wire').glob('*.tessera')) == []
    else:
        assert campaign.main([*argv, '--seed-checkpoint', str(checkpoint)]) == 0
        with (tmp_path/'new-cost.pkl').open('rb') as handle:
            assert pickle.load(handle)['costs'] == _payload['costs']
        assert list((new_cache/'wire').glob('*.tessera'))
    assert checkpoint.read_bytes() == original_manifest



@pytest.mark.parametrize("changed", ["producer", "encoder", "mixed", "certified",
                                     "tampered_wire", "other_wire_identity"])
def test_main_producer_identity_resume_policy(monkeypatch, tmp_path, priced_campaign, capsys, changed):
    from prismaquant import production_weight_cache as pwc
    from prismaquant.cost_stage_checkpoint import prepare_journal, unit_path, write_unit
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    (campaign, checkpoint, argv, _model, inputs), initial = priced_campaign()
    original_manifest = checkpoint.read_bytes()
    root = checkpoint.with_name(checkpoint.name + ".parts")
    stored_manifest = json.loads(original_manifest)
    stored = stored_manifest["identity"]
    if changed == "other_wire_identity":
        state = prepare_journal(root, stage="Tessera campaign", resume=True,
            identity=stored, qnames=[UNIT], manifest_path=checkpoint)[2][UNIT]
        state["wire_records"]["TESSERA_E4M3_K1_R1024"]["identity"]["encoder_fixture_id"] = "other fixture"
        write_unit(root, stage="Tessera campaign", qname=UNIT,
                   identity_sha256=stored_manifest["identity_sha256"], state=state)
    original_shard = unit_path(root, UNIT).read_bytes()
    if changed == "tampered_wire":
        wire = next((tmp_path / "cache" / "wire").glob("*.tessera"))
        blob = bytearray(wire.read_bytes())
        blob[-1] ^= 1
        wire.write_bytes(blob)
    api = campaign._checkpoint_identity_api()
    verifications = []
    original_verify = api.verify_cached_unit

    def verify(blob, record, expected):
        verifications.append((record["identity"], expected))
        return original_verify(blob, record, expected)

    monkeypatch.setattr(api, "verify_cached_unit", verify)
    if changed != "encoder":
        monkeypatch.setattr(pwc, "_production_cache_source_sha256", lambda: "a" * 64)
    monkeypatch.setattr(api, "encoder_source_sha256", lambda: "b" * 64)
    _forbid_reencode(monkeypatch, campaign)
    capsys.readouterr()
    if changed == "mixed":
        inputs["rows"][0, 0] += 1
    elif changed == "certified":
        monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    if changed in ("producer", "encoder"):
        assert campaign.main(argv) == 0
        assert _priced_cost_payload(tmp_path)["costs"] == initial["costs"]
        assert verifications and all(observed["encoder_source_sha256"] == expected["encoder_source_sha256"]
                                     for observed, expected in verifications)
        lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[DEV-MODE]")]
        journal_lines = [line for line in lines if "Tessera campaign checkpoint" in line]
        assert len(journal_lines) == 1
        fields = [("encoder_source_sha256", "b" * 64)]
        if changed == "producer":
            fields.append(("prismaquant_source_sha256", "a" * 64))
        for field, current in fields:
            assert field in journal_lines[0] and stored[field] in journal_lines[0] and current in journal_lines[0]
        wire_lines = [line for line in lines if "campaign wire" in line]
        assert len(wire_lines) == 1
        assert all(text in wire_lines[0] for text in ("encoder_source_sha256", stored["encoder_source_sha256"], "b" * 64))
    else:
        message = {"tampered_wire": "blob size/sha256 mismatch",
                   "other_wire_identity": "encoder_fixture_id identity mismatch"}.get(changed, "checkpoint identity mismatch")
        with pytest.raises(RuntimeError, match=message):
            campaign.main(argv)
        if changed in ("tampered_wire", "other_wire_identity"):
            assert verifications, "wire verifier was skipped"
        else:
            assert "[DEV-MODE]" not in capsys.readouterr().out
    assert checkpoint.read_bytes() == original_manifest
    assert unit_path(root, UNIT).read_bytes() == original_shard


