"""The live source has no retired executor mechanisms; prove the scanner detects them."""
from tests import prismabuild_scan as scan


def test_no_live_source_carries_a_prismabuild_mechanism():
    assert scan.live_findings() == {}


def test_the_ban_sees_every_mechanism_form_and_spares_wire_ids():
    planted = {
        "PRISMABUILD_ACTION_KEY=1": "PRISMABUILD_",
        "pb_action = key": "pb_action",
        "import prismaquant.prismabuild.client": "import prismaquant.prismabuild.client",
        "from prismaquant import prismabuild_progress": "from prismaquant import prismabuild_progress",
        "importlib.util.find_spec('prismabuild')": "find_spec('prismabuild",
        "__import__('prismabuild')": "__import__('prismabuild",
        "python pbrun.py --x": "pbrun.py",
        "pbtest --json r": "pbtest",
        "run pbcampaign now": "pbcampaign",
        "x = '/mnt/shared/prismabuild-fleet/cas'": "prismabuild-fleet",
        "from prismabuild import pool": "from prismabuild",
        "import prismabuild.core": "import prismabuild",
        "import fleet_sdk": "import fleet_sdk",
        "pip install --group pbsdk": "pbsdk",
    }
    for text, token in planted.items():
        assert token in " ".join(scan.mechanisms(text)), text
    for text in ("a pbwait row", "pbgang x"):
        assert scan.mechanisms(text)
    spared = "schema = 'prismaquant.prismabuild.cas_receipt.v3'  # prose: PrismaBuild was retired"
    assert scan.mechanisms(spared) == []


def test_a_planted_violation_in_a_live_file_is_a_finding(tmp_path, monkeypatch):
    module = scan
    (tmp_path / "prismaquant").mkdir()
    (tmp_path / "prismaquant" / "planted.py").write_text("import prismabuild.pool\n")
    (tmp_path / "docs" / "results").mkdir(parents=True)
    (tmp_path / "docs" / "results" / "old.md").write_text("pbrun --old\n")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "tracked", lambda: ["prismaquant/planted.py", "docs/results/old.md"])
    assert module.live_findings() == {"prismaquant/planted.py": ["import prismabuild"]}


def test_token_carveouts_do_not_exempt_the_rest_of_a_file():
    for path in scan.REFUSAL_TOKENS:
        assert not scan.excluded(path)
        assert scan.mechanisms(scan.current_text(path, "pb_action = new_live_job")) == ["pb_action"]


def test_only_explicit_recorded_evidence_files_are_exempt():
    assert all(scan.excluded(path) for path in scan.RECORDED_EVIDENCE)
    assert not scan.excluded("prismaquant/kernels/new_qualification.json")
    assert not scan.excluded("docs/design/operator_runbook.md")
    assert not scan.excluded("docs/design/old_run_2026-09-01.md")
    assert not scan.excluded("docs/design/operator_runbook_2026-11-10.md")


def test_token_carveouts_still_name_existing_source():
    for path, tokens in scan.REFUSAL_TOKENS.items():
        text = scan.read_text(path)
        assert text is not None
        for token in tokens:
            assert token in text, (path, token)


def test_recorded_evidence_exemptions_still_have_mechanism_hits():
    for path in scan.RECORDED_EVIDENCE:
        text = scan.read_text(path)
        assert text is not None, path
        assert scan.mechanisms(text), path
