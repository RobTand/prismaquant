"""pipeline.py as the settings-hash authority (re-vet R5, debt D6).

The guard's contract, in one place:
  * artifact absent            -> record this stage's key set, exit 0
  * recorded projection equal  -> exit 0
  * recorded projection differs-> exit 2, naming every diff
  * no record for this stage   -> WARN, reuse, and record that the artifact's
                                  identity is UNKNOWN (pre-guard artifacts are
                                  still reusable, but they never acquire
                                  today's settings as their provenance)
Plus the property that made R5 worth doing: WHICH keys an artifact depends on
is declared once, in `STAGE_SETTINGS_KEYS`, not re-decided at every call site.
"""

import json
import re
from pathlib import Path

from prismaquant import pipeline

ROOT = Path(__file__).resolve().parents[1]


#: The on-disk key that files an unverified reuse. Read through `getattr` so
#: the same assertions also run against a tree that predates the marker -- that
#: is the point of the red reproduction: there the guard writes a settings
#: projection instead, and each test below fails on the manifest's content
#: rather than on a missing attribute.
MARKER = getattr(pipeline, "UNVERIFIED_SETTINGS_KEY", "_unverified_settings")


def _entry(artifact: Path, stage: str = "probe"):
    """Whatever `artifact`'s manifest records for `stage`, or None."""
    manifest = Path(f"{artifact}.settings.json")
    if not manifest.exists():
        return None
    return (json.loads(manifest.read_text()).get("stages") or {}).get(stage)


def _admission(artifact: Path, stage: str = "probe"):
    """The unverified-reuse admission filed for `artifact`, or None."""
    entry = _entry(artifact, stage)
    if isinstance(entry, dict) and set(entry) == {MARKER}:
        return entry[MARKER]
    return None


def _document(**settings):
    return pipeline.stage_settings_document(settings)


def _full_settings():
    """Every source name any stage declares, with a placeholder value."""
    sources = {
        source
        for keys in pipeline.STAGE_SETTINGS_KEYS.values()
        for _manifest_key, source in keys
    }
    return {name: f"<{name}>" for name in sources}


def test_every_declared_stage_projects_cleanly_from_the_full_settings():
    doc = _document(**_full_settings())
    assert doc["unresolved"] == {}
    assert set(doc["artifacts"]) == set(pipeline.STAGE_SETTINGS_KEYS)
    for stage, keys in pipeline.STAGE_SETTINGS_KEYS.items():
        assert set(doc["artifacts"][stage]) == {mk for mk, _ in keys}


def test_run_pipeline_supplies_every_declared_setting():
    """The shell must supply a value for every source pipeline.py declares,
    either in STAGE_SETTINGS_ENV or as a late override at the call site. A
    declared-but-unsupplied key is what the guard hard-fails on, so pin it."""
    script = (ROOT / "prismaquant" / "run-pipeline.sh").read_text()
    block = script.split("STAGE_SETTINGS_ENV=(", 1)[1].split("\n)", 1)[0]
    supplied = set(re.findall(r'"([A-Z0-9_]+)=', block))
    # RENDER_ENV_SETTINGS is spliced in by reference.
    render_block = script.split("RENDER_ENV_SETTINGS=(", 1)[1].split("\n)", 1)[0]
    supplied |= set(re.findall(r'"([A-Z0-9_]+)=', render_block))
    # Late-computed values passed as overrides at their call sites.
    supplied |= set(re.findall(r'require_stage_settings [^\n]*\\?\n?\s*"([A-Z0-9_]+)=',
                               script))
    declared = {
        source
        for keys in pipeline.STAGE_SETTINGS_KEYS.values()
        for _mk, source in keys
    }
    missing = sorted(declared - supplied)
    assert not missing, (
        f"run-pipeline.sh supplies no value for {missing}; the guard would "
        "exit 2 on a partial key set")


def test_every_guarded_stage_id_in_the_shell_is_declared():
    script = (ROOT / "prismaquant" / "run-pipeline.sh").read_text()
    used = set(re.findall(r'require_stage_settings "[^"]+" ([a-z0-9-]+)', script))
    assert used, "no guard call sites found"
    unknown = sorted(used - set(pipeline.STAGE_SETTINGS_KEYS))
    assert not unknown, f"undeclared settings-hash stages: {unknown}"


def test_absent_artifact_records_then_matches(tmp_path):
    doc = _document(MODEL_PATH="m", DATASET="d", NSAMPLES="8", SEQLEN="512",
                    CALIBRATION_MODALITY="text-only")
    artifact = tmp_path / "probe.pkl"
    code, _ = pipeline.check_stage_settings(artifact, "probe", doc)
    assert code == 0
    manifest = json.loads((tmp_path / "probe.pkl.settings.json").read_text())
    assert manifest["stages"]["probe"]["NSAMPLES"] == "8"
    # Fresh production is the VERIFIED path: no unverified admission is filed
    # beside a projection this run is about to produce.
    assert _admission(artifact) is None

    artifact.write_bytes(b"x")
    code, messages = pipeline.check_stage_settings(artifact, "probe", doc)
    assert code == 0 and messages == []


def test_changed_setting_is_exit_2_naming_the_diff(tmp_path):
    base = dict(MODEL_PATH="m", DATASET="d", NSAMPLES="8", SEQLEN="512",
                CALIBRATION_MODALITY="text-only")
    artifact = tmp_path / "probe.pkl"
    pipeline.check_stage_settings(artifact, "probe", _document(**base))
    artifact.write_bytes(b"x")

    changed = dict(base, NSAMPLES="32")
    code, messages = pipeline.check_stage_settings(
        artifact, "probe", _document(**changed))
    assert code == 2
    joined = "\n".join(messages)
    assert "NSAMPLES" in joined and "'8'" in joined and "'32'" in joined
    assert "refusing silent reuse" in joined


def test_aura_cost_streaming_and_checkpoint_directory_are_reuse_identity(tmp_path):
    base = _full_settings()
    base.update({
        "AURA_COST_STREAMING": "1",
        "AURA_COST_CHECKPOINT_DIR": "/runs/aura-a",
    })

    for changed_key, changed_value in (
        ("AURA_COST_STREAMING", "0"),
        ("AURA_COST_CHECKPOINT_DIR", "/runs/aura-b"),
    ):
        artifact = tmp_path / f"cost-{changed_key}.pkl"
        pipeline.check_stage_settings(
            artifact, "aura-cost", _document(**base)
        )
        artifact.write_bytes(b"cost")

        changed = dict(base, **{changed_key: changed_value})
        code, messages = pipeline.check_stage_settings(
            artifact, "aura-cost", _document(**changed)
        )
        assert code == 2
        assert changed_key in "\n".join(messages)


def test_aura_hybrid_streaming_and_checkpoint_directory_are_reuse_identity(tmp_path):
    base = _full_settings()
    base.update({
        "AURA_COST_STREAMING": "1",
        "AURA_COST_CHECKPOINT_DIR": "/runs/aura-a",
    })

    for changed_key, changed_value in (
        ("AURA_COST_STREAMING", "0"),
        ("AURA_COST_CHECKPOINT_DIR", "/runs/aura-b"),
    ):
        artifact = tmp_path / f"aura-hybrid-{changed_key}.pkl"
        pipeline.check_stage_settings(
            artifact, "aura-hybrid-cost", _document(**base)
        )
        artifact.write_bytes(b"cost")

        changed = dict(base, **{changed_key: changed_value})
        code, messages = pipeline.check_stage_settings(
            artifact, "aura-hybrid-cost", _document(**changed)
        )
        assert code == 2
        assert changed_key in "\n".join(messages)


def test_cb_imatrix_source_is_reuse_identity_for_harvest_and_render(tmp_path):
    base = _full_settings()
    base["CB_IMATRIX_SOURCE"] = "activation-cache"

    for stage in ("cb-col-weights", "aura-dw-cache", "frontier-cache"):
        artifact = tmp_path / f"{stage}.pkl"
        pipeline.check_stage_settings(artifact, stage, _document(**base))
        artifact.write_bytes(b"artifact")

        changed = dict(base, CB_IMATRIX_SOURCE="probe")
        code, messages = pipeline.check_stage_settings(
            artifact, stage, _document(**changed)
        )
        assert code == 2
        assert "CB_IMATRIX_SOURCE" in "\n".join(messages)


def test_missing_manifest_is_reused_unverified_and_never_stamped(tmp_path):
    """Renamed 2026-09-16 from `test_missing_manifest_only_warns`.

    Raw file presence still reuses the artifact (Rob, 2026-09-16: old data may
    be reused while the pipeline is in flux). What changed is what the guard
    records about it: this run's settings are NOT written as the artifact's
    settings, because nobody compared them to the bytes.
    """
    artifact = tmp_path / "probe.pkl"
    artifact.write_bytes(b"x")
    request = _document(MODEL_PATH="m", DATASET="d", NSAMPLES="8",
                        SEQLEN="512", CALIBRATION_MODALITY="t")
    code, messages = pipeline.check_stage_settings(artifact, "probe", request)
    assert code == 0
    assert "WARNING" in messages[0] and "no settings manifest" in messages[0]

    admitted = _admission(artifact)
    assert admitted is not None, "the reuse was not recorded at all"
    assert admitted["settings_identity"] == "unknown"
    assert admitted["attests_this_artifact"] is False
    # The request is kept, but only ever as an observation of this run.
    assert admitted["observed_current_request"]["NSAMPLES"] == "8"
    stored = json.loads(
        (tmp_path / "probe.pkl.settings.json").read_text())["stages"]["probe"]
    assert set(stored) == {MARKER}, (
        "the guard stamped a settings projection for an artifact it never compared"
    )


def test_pre_guard_stage_entry_is_not_restamped_with_today_settings(tmp_path):
    """The pre-fix defect, pinned (RobTand/prismaquant#654).

    `origin/main` at 2fa95995bd appended the CURRENT projection to a manifest
    whose stage entry said nothing about this stage's artifact, so a cost table
    measured under other probes/seed/dataset/menu was recorded as if this run
    had verified it -- and every later check read that as a match.
    """
    base = dict(MODEL_PATH="m", DATASET="d", NSAMPLES="8", SEQLEN="512",
                CALIBRATION_MODALITY="text-only")
    artifact = tmp_path / "cost_aura.pkl"
    artifact.write_bytes(b"cost")
    # Written by an older or partial run that never recorded this stage.
    (tmp_path / "cost_aura.pkl.settings.json").write_text(json.dumps({
        "schema": pipeline.STAGE_MANIFEST_SCHEMA, "stages": {"other": {"X": "1"}},
    }))

    code, messages = pipeline.check_stage_settings(
        artifact, "probe", _document(**base))
    assert code == 0
    assert any("predates this stage's settings guard" in m for m in messages)

    stored = json.loads((tmp_path / "cost_aura.pkl.settings.json").read_text())
    entry = stored["stages"]["probe"]
    assert _admission(artifact) is not None
    assert set(entry) == {MARKER}, (
        "today's projection was written as the artifact's settings"
    )
    # The stage the manifest already knew about is left exactly as it was.
    assert stored["stages"]["other"] == {"X": "1"}


def test_unverified_reuse_stays_unknown_across_retries_and_changed_requests(
        tmp_path):
    request = dict(MODEL_PATH="m", DATASET="d", NSAMPLES="8", SEQLEN="512",
                   CALIBRATION_MODALITY="text-only")
    artifact = tmp_path / "cost.pkl"
    artifact.write_bytes(b"cost")

    assert pipeline.check_stage_settings(
        artifact, "probe", _document(**request))[0] == 0
    recorded = _admission(artifact)
    assert recorded is not None

    # A retry reads the same admission back and never upgrades it.
    code, messages = pipeline.check_stage_settings(
        artifact, "probe", _document(**request))
    assert code == 0 and any("still UNVERIFIED" in m for m in messages)
    assert _admission(artifact) == recorded

    # A changed request does not make the artifact's identity known either: the
    # run still reuses it (legacy reuse is allowed) and still records only that
    # its settings are unknown.
    code, messages = pipeline.check_stage_settings(
        artifact, "probe", _document(**dict(request, NSAMPLES="64")))
    assert code == 0
    assert any("identity is unknown" in m for m in messages)
    assert _admission(artifact) == recorded, (
        "the first observation was rewritten by a later request"
    )
    stages = json.loads((tmp_path / "cost.pkl.settings.json").read_text())["stages"]
    assert set(stages["probe"]) == {MARKER}


def test_a_recorded_projection_still_refuses_a_mismatch(tmp_path):
    """The legacy allowance is not a bypass: once a projection IS on file, a
    different request refuses exactly as before."""
    base = dict(MODEL_PATH="m", DATASET="d", NSAMPLES="8", SEQLEN="512",
                CALIBRATION_MODALITY="text-only")
    artifact = tmp_path / "probe.pkl"
    assert pipeline.check_stage_settings(artifact, "probe", _document(**base))[0] == 0
    artifact.write_bytes(b"x")
    code, messages = pipeline.check_stage_settings(
        artifact, "probe", _document(**dict(base, NSAMPLES="32")))
    assert code == 2
    assert "refusing silent reuse" in "\n".join(messages)
    assert _admission(artifact) is None, "a refused reuse filed an admission"


def test_tessera_plan_still_refuses_and_files_no_admission(tmp_path):
    plan = tmp_path / "tessera_plan.json"
    plan.write_text("old translated plan")
    code, messages = pipeline.check_stage_settings(
        plan, "tessera-plan", _document(**_full_settings()))
    assert code == 2
    assert "allocation content binding" in "\n".join(messages)
    assert not (tmp_path / "tessera_plan.json.settings.json").exists(), (
        "a refused plan was recorded as a reusable artifact"
    )


def test_tessera_plan_refuses_an_unverified_admission_not_only_a_missing_one(
        tmp_path):
    """An admission is not a binding (root review of c45784cb, #654).

    The first cut of the legacy allowance let a stage entry that holds only the
    unverified marker take the generic "still UNVERIFIED, reuse continues" path
    before the plan gate could see it. Tessera's gate has to refuse on the
    marker itself: the marker says the artifact's identity is *unknown*, and a
    translated plan needs a real allocation-content binding, so reusing it
    would allocate against a binding nobody ever checked. Reached through
    `check_stage_settings` so the ordering, not just the helper, is pinned.
    """
    plan = tmp_path / "tessera_plan.json"
    plan.write_text("old translated plan")
    manifest = tmp_path / "tessera_plan.json.settings.json"
    manifest.write_text(json.dumps({
        "schema": pipeline.STAGE_MANIFEST_SCHEMA,
        "stages": {
            "tessera-plan": {
                MARKER: {
                    "settings_identity": "unknown",
                    "attests_this_artifact": False,
                    "artifact": str(plan),
                    "reason": "hand-written marker",
                    "observed_current_request": {},
                    "first_observed_unix": 0,
                }
            }
        },
    }))
    before = manifest.read_text()

    code, messages = pipeline.check_stage_settings(
        plan, "tessera-plan", _document(**_full_settings()))
    assert code == 2, messages
    assert "unverified-reuse admission" in "\n".join(messages)
    assert manifest.read_text() == before, (
        "the refusal rewrote the manifest instead of leaving it alone"
    )


def test_the_marker_cannot_collide_with_a_declared_manifest_key():
    declared = {mk for keys in pipeline.STAGE_SETTINGS_KEYS.values()
                for mk, _source in keys}
    assert MARKER not in declared
    assert not any(key.startswith("_") for key in declared), sorted(declared)


def test_the_admission_makes_a_projection_only_reader_refuse_rather_than_stamp(
        tmp_path):
    """Cross-version safety, asserted against the comparison itself.

    A tree that predates this change has no idea what the marker means; it
    compares `stages[stage]` to its declared keys and exits 2 on any diff. That
    is the point of filing the admission under a key no projection ever
    carries: the artifact cannot be laundered into a verified record by an
    older reader either.
    """
    artifact = tmp_path / "probe.pkl"
    artifact.write_bytes(b"x")
    document = _document(MODEL_PATH="m", DATASET="d", NSAMPLES="8", SEQLEN="512",
                         CALIBRATION_MODALITY="text-only")
    pipeline.check_stage_settings(artifact, "probe", document)
    prev = json.loads(
        (tmp_path / "probe.pkl.settings.json").read_text())["stages"]["probe"]
    declared = document["artifacts"]["probe"]
    diffs = {key for key in set(prev) | set(declared)
             if prev.get(key) != declared.get(key)}
    assert MARKER in diffs


def test_legacy_flat_manifest_still_guards_its_stage(tmp_path):
    """Pre-R5 manifests are a flat {key: value} dict. Same key set -> still
    compared (a changed setting must still fail); different stage -> not
    invalidated."""
    artifact = tmp_path / "probe.pkl"
    artifact.write_bytes(b"x")
    (tmp_path / "probe.pkl.settings.json").write_text(json.dumps({
        "MODEL_PATH": "m", "DATASET": "d", "NSAMPLES": "8", "SEQLEN": "512",
        "CALIBRATION_MODALITY": "text-only",
    }))
    same = _document(MODEL_PATH="m", DATASET="d", NSAMPLES="8", SEQLEN="512",
                     CALIBRATION_MODALITY="text-only")
    assert pipeline.check_stage_settings(artifact, "probe", same)[0] == 0
    drift = _document(MODEL_PATH="m", DATASET="OTHER", NSAMPLES="8",
                      SEQLEN="512", CALIBRATION_MODALITY="text-only")
    assert pipeline.check_stage_settings(artifact, "probe", drift)[0] == 2


def test_two_stages_can_own_one_path(tmp_path):
    """COST_MODE=aura + validated-surrogate points the AURA dW cache and the
    frontier cache at the SAME file; both key sets must coexist."""
    settings = _full_settings()
    doc = _document(**settings)
    artifact = tmp_path / "cache.pkl"
    assert pipeline.check_stage_settings(artifact, "aura-dw-cache", doc)[0] == 0
    artifact.write_bytes(b"x")
    code, messages = pipeline.check_stage_settings(artifact, "frontier-cache", doc)
    assert code == 0 and any("predates this stage's" in m for m in messages)
    stored = json.loads((tmp_path / "cache.pkl.settings.json").read_text())
    assert set(stored["stages"]) == {"aura-dw-cache", "frontier-cache"}
    # The stage that recorded a projection still holds one; the second stage
    # holds the admission that its artifact's identity is unknown, NOT a copy
    # of the first stage's projection or of today's request.
    assert _admission(artifact, "aura-dw-cache") is None
    assert _admission(artifact, "frontier-cache") is not None
    # …and both keep guarding independently: the verified one still compares
    # (a drift refuses), the unverified one stays unverified across retries.
    assert pipeline.check_stage_settings(artifact, "aura-dw-cache", doc)[0] == 0
    code, messages = pipeline.check_stage_settings(artifact, "frontier-cache", doc)
    assert code == 0 and any("still UNVERIFIED" in m for m in messages)
    assert pipeline.check_stage_settings(
        artifact, "aura-dw-cache",
        _document(**dict(settings, NSAMPLES="999")))[0] == 2


def test_unsupplied_declared_key_is_a_hard_stop(tmp_path):
    doc = _document(MODEL_PATH="m")  # DATASET/NSAMPLES/... missing
    code, messages = pipeline.check_stage_settings(
        tmp_path / "probe.pkl", "probe", doc)
    assert code == 2
    assert "no value was supplied" in messages[0]


def test_late_override_resolves_a_missing_key(tmp_path):
    doc = _document(MODEL_PATH="m", DATASET="d", NSAMPLES="8", SEQLEN="512",
                    FORMATS="NVFP4", TARGET_BITS="4.75",
                    LM_HEAD_FORMAT="BF16", LM_HEAD_RENDER_ACTIVE="0",
                    LM_HEAD_DP_UNPINNED="0",
                    PRISMAQUANT_NVFP4_SCALE_RULE="", PRISMAQUANT_GPTQ_DAMP_SWEEP="0",
                    PRISMAQUANT_GPTQ_DAMP="", PRISMAQUANT_ACT_CLIP_QUANTILE="0.999",
                    PRODUCTION_CACHE_LEVERS="gptq", PRODUCTION_CACHE_DISABLE_LEVERS="",
                    CB_SCALE_CODING="", CB_CODEBOOK_SOURCE="",
                    CB_CODEBOOK_SOURCE_SCOPE="", CB_CODEBOOK_BUNDLE="",
                    CB_ROUTED_MOE_BOOK_SELECTION="",
                    CB_ROUTED_MOE_BOOK_SELECTION_SHA256="",
                    CB_SCALE_SWEEP="1", CB_SCALE_SWEEP_SCOPE="",
                    CB_ACTIVATION_SCOPE="nvfp4",
                    CB_IMATRIX_SOURCE="activation-cache",
                    PRISMAQUANT_CB_LDLQ="0",
                    PRISMAQUANT_CB_MINCHAIN="0",
                    PRISMAQUANT_CB_MINCHAIN_ANCHORS="",
                    PRISMAQUANT_CB_MINCHAIN_HOLDBACKS="",
                    PRISMAQUANT_CB_MINCHAIN_AUDIT_SEED="42",
                    PRISMAQUANT_CB_MINCHAIN_BACKSTOP="0.25",
                    PRISMAQUANT_CB_MINCHAIN_AUDIT_MEDIAN="0.05",
                    PRISMAQUANT_CB_MINCHAIN_AUDIT_P95="0.15",
                    PRISMAQUANT_CB_ENCODE_TIER="balanced")
    artifact = tmp_path / "recached.pkl"
    assert doc["unresolved"]["production-cache-recached"] == ["ASSIGNMENT_DIGEST"]
    code, _ = pipeline.check_stage_settings(
        artifact, "production-cache-recached", doc,
        overrides={"ASSIGNMENT_DIGEST": "deadbeef"})
    assert code == 0
    stored = json.loads((tmp_path / "recached.pkl.settings.json").read_text())
    assert stored["stages"]["production-cache-recached"]["ASSIGNMENT_DIGEST"] == "deadbeef"


def test_approved_resource_owners_name_real_implementations():
    """D10: two of the three owner names were never implemented anywhere.

    The name SET is not pinned (#150-#155's defect class): it was
    `== {"ProductionWeightCache", "PerturbedActivationCache", "LayerCache"}`,
    which goes red on a legitimate fourth owner and stays green on a corrupted
    mapping, since swapping two values preserves the union. What is load-
    bearing is per-owner and per-resource, so that is what is asserted; the
    mapping itself is checked against the stages that consume it below.
    """
    owners = {o for names in pipeline.APPROVED_RESOURCE_OWNERS.values()
              for o in names}
    # Non-vacuity: an emptied mapping must not pass the loop below on nothing.
    assert len(owners) >= 2, owners
    for resource, allowed in pipeline.APPROVED_RESOURCE_OWNERS.items():
        assert allowed, f"{resource}: approves no owner, so it governs nothing"
    sources = [f.read_text(encoding="utf-8")
               for f in (ROOT / "prismaquant").rglob("*.py")]
    for owner in owners:
        assert any(f"class {owner}" in text for text in sources), (
            f"{owner} has no implementation in the tree")


def test_the_owners_the_mapping_approves_are_the_owners_the_stages_declare():
    """The mapping is only as true as the pipeline that reads it.

    `APPROVED_RESOURCE_OWNERS` is enforced per resource (`pipeline.py`'s
    validation rejects `resource.owner not in allowed`), so the claim it
    carries is a claim about the live stage graph: rendered weights flow
    through `ProductionWeightCache`, activations through
    `PerturbedActivationCache`, streaming weights through
    `layer_streaming.LayerCache`. Reading the mapping's union tells you none of
    that -- swap two values and the union is unchanged. This asks the stages
    instead, which is the only place the pairing is observable.
    """
    spec = pipeline.default_production_pipeline_spec()
    declared: dict[str, set[str]] = {}
    for stage in spec.stages:
        for resource in stage.resources:
            declared.setdefault(resource.resource, set()).add(resource.owner)

    governed = sorted(set(declared) & set(pipeline.APPROVED_RESOURCE_OWNERS))
    # Non-vacuity twice over: the mapping must govern resources the pipeline
    # actually declares, and it must govern more than one of them, or a stage
    # graph that stopped declaring resources would satisfy this silently.
    assert len(governed) >= 2, (governed, sorted(declared))
    assert set(pipeline.APPROVED_RESOURCE_OWNERS) <= set(declared), (
        "the mapping governs a resource no stage declares: "
        f"{sorted(set(pipeline.APPROVED_RESOURCE_OWNERS) - set(declared))}")

    owner_errors = [e for e in spec.validate().errors if "must use one of" in e]
    assert not owner_errors, owner_errors


def test_a_wrong_but_implemented_owner_is_refused_by_stage_validation():
    """The reverse rule, and the reason the union is the wrong thing to pin.

    `LayerCache` is a real class and a legitimate owner -- of streaming model
    weights. Handed `rendered_weights` it is the corrupted mapping, and the
    enforcement has to name it. A test that only checked "every owner names a
    real class" cannot tell this apart from the truth.
    """
    import dataclasses

    spec = pipeline.default_production_pipeline_spec()
    resource = "rendered_weights"
    approved = pipeline.APPROVED_RESOURCE_OWNERS[resource]
    wrong = sorted(
        {o for r, names in pipeline.APPROVED_RESOURCE_OWNERS.items()
         if r != resource for o in names} - set(approved))
    assert wrong, "no other approved owner to mis-assign; the rule is untestable"

    stages = []
    swapped = 0
    for stage in spec.stages:
        contracts = []
        for contract in stage.resources:
            if contract.resource == resource:
                contract = dataclasses.replace(contract, owner=wrong[0])
                swapped += 1
            contracts.append(contract)
        stages.append(dataclasses.replace(stage, resources=tuple(contracts)))
    assert swapped, f"no stage declares {resource}"

    errors = dataclasses.replace(spec, stages=tuple(stages)).validate().errors
    naming = [e for e in errors
              if resource in e and "must use one of" in e and wrong[0] in e]
    assert len(naming) == swapped, (naming, errors)
