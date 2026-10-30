from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("synthetic_graph_lane")


@pytest.fixture
def publication_fixture(tmp_path):
    fixture_path = Path.cwd() / "tests" / "test_publish_artifact.py"
    assert (Path.cwd() / "tools").is_dir(), "publisher audit requires the full checkout"
    spec = importlib.util.spec_from_file_location("_publish_fixture_audit", fixture_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        artifact = module._artifact(tmp_path)
        module._close_all_slots(artifact)
        yield module, artifact
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.parametrize("standing", ["ordinary", "research_initial", "research_after_preflight"])
def test_frozen_publication_replays_research_standing(publication_fixture, monkeypatch, capsys, standing):
    from prismaquant import shipcard
    import tools.publish_artifact as publisher

    fixture, artifact = publication_fixture
    path = artifact / shipcard.SHIPCARD_FILENAME
    assert publisher.check_shipcard(artifact, path)[1] == []
    observations = []

    def mark_research():
        card = shipcard.load_shipcard(path)
        card["build"]["research_only"] = True
        shipcard.write_shipcard(path, card)

    if standing == "research_initial":
        mark_research()
    real_observer = publisher._print_bpp_claim_with_control

    def observe(card, root):
        real_observer(card, root)
        observations.append(card["build"].get("research_only"))
        if standing == "research_after_preflight":
            assert observations == [None]
            mark_research()

    monkeypatch.setattr(publisher, "_print_bpp_claim_with_control", observe)
    result = publisher.main(fixture._argv(artifact))
    captured = capsys.readouterr()

    if standing == "ordinary":
        assert result == 0
        assert "frozen snapshot VERIFIED" in captured.out
    else:
        assert result == 1, captured.out + captured.err
        assert "research_only" in captured.err
        assert "frozen snapshot VERIFIED" not in captured.out
    assert "forced_unverified" not in shipcard.load_shipcard(path)
