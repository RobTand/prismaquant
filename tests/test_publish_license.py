"""#1268: new publications cannot waive licensing or publish before gating."""
from __future__ import annotations

import contextlib

import pytest

from test_publish_artifact import (
    _FakeHubState,
    _argv,
    _artifact,
    _close_all_slots,
    _install_fake_hub,
    publish_cli,
    publisher,
)


@pytest.mark.parametrize("change", [
    "license_missing", "license_wrong_repo", "license_edited", "license_unfilled",
    "readme_missing", "no_frontmatter", "license", "license_name", "license_link",
    "duplicate_key", "yaml_merge", "malformed_yaml", "tldr_missing",
])
def test_publication_license_driver_mutations_refuse_before_force_or_hub(
    tmp_path, monkeypatch, capsys, change,
):
    model = _artifact(tmp_path)
    card_before = (model / "shipcard.json").read_bytes()
    license_path = model / "LICENSE"
    readme = model / "README.md"
    if change == "license_missing":
        license_path.unlink()
    elif change == "license_wrong_repo":
        license_path.write_text(license_path.read_text().replace("test-artifact", "different"))
    elif change == "license_edited":
        license_path.write_text(license_path.read_text() + "\nTerms removed.\n")
    elif change == "license_unfilled":
        license_path.write_text(license_path.read_text().replace("test-artifact", "<repository>"))
    elif change == "readme_missing":
        readme.unlink()
    elif change == "no_frontmatter":
        readme.write_text(readme.read_text().removeprefix("---\n"))
    elif change in {"license", "license_name", "license_link"}:
        lines = readme.read_text().splitlines()
        readme.write_text("\n".join(
            f"{change}: wrong" if line.startswith(change + ":") else line
            for line in lines
        ) + "\n")
    elif change == "duplicate_key":
        readme.write_text(readme.read_text().replace("license: other", "license: wrong\nlicense: other"))
    elif change == "yaml_merge":
        readme.write_text(readme.read_text().replace("license: other", "<<: {license: other}"))
    elif change == "malformed_yaml":
        readme.write_text(readme.read_text().replace("license: other", "license: ["))
    elif change == "tldr_missing":
        readme.write_text(readme.read_text().split("> **TL;DR.**", 1)[0])
    else:  # pragma: no cover
        raise AssertionError(change)

    def no_hub():
        pytest.fail("policy refusal must precede all Hub bindings")

    monkeypatch.setattr(publisher, "_load_hub_bindings", no_hub)
    argv = [str(model), "--repo-id", "rdtand/test-artifact",
            "--force-unverified", "--confirm-name", model.name]
    assert publish_cli(argv) == 2
    assert "license policy" in capsys.readouterr().err
    assert (model / "shipcard.json").read_bytes() == card_before


@pytest.mark.parametrize("name", ["LICENSE", "README.md"])
def test_publication_license_files_must_be_regular_not_symlinks(tmp_path, name):
    model = _artifact(tmp_path)
    path = model / name
    external = tmp_path / name
    path.rename(external)
    path.symlink_to(external)
    assert publish_cli(_argv(model)) == 2
    path.unlink()
    path.mkdir()
    assert publish_cli(_argv(model)) == 2


def test_valid_license_dry_run_has_no_hub_calls(tmp_path, monkeypatch):
    model = _artifact(tmp_path)
    _close_all_slots(model)
    monkeypatch.setattr(publisher, "_load_hub_bindings",
                        lambda: pytest.fail("dry-run must not load Hub bindings"))
    assert publish_cli(_argv(model)) == 0


@pytest.mark.parametrize("name", ["LICENSE", "README.md"])
def test_publication_license_replayed_on_frozen_bytes(tmp_path, monkeypatch, capsys, name):
    model = _artifact(tmp_path)
    _close_all_slots(model)
    real_freeze = publisher._freeze_artifact

    @contextlib.contextmanager
    def change_before_freeze(*args, **kwargs):
        (model / name).write_text("wrong policy bytes\n")
        with real_freeze(*args, **kwargs) as snapshot:
            yield snapshot

    monkeypatch.setattr(publisher, "_freeze_artifact", change_before_freeze)
    assert publish_cli(_argv(model)) == 2
    assert "license policy" in capsys.readouterr().err


def test_real_publish_sets_and_reads_auto_before_any_upload(tmp_path, monkeypatch, capsys):
    model = _artifact(tmp_path)
    _close_all_slots(model)
    state = _FakeHubState(remote={".gitattributes": b"managed"})
    _install_fake_hub(monkeypatch, state)
    assert publish_cli([str(model), "--repo-id", "rdtand/test-artifact"]) == 0
    assert state.gate_updates == [{"repo_id": "rdtand/test-artifact", "repo_type": "model", "gated": "auto"}]
    assert state.events == ["repo_info", "gate", "repo_info", "preupload", "commit", "repo_info"]
    assert 'gated="auto"' in capsys.readouterr().out


@pytest.mark.parametrize("observed", [False, True, None, "manual"])
def test_auto_gate_readback_refuses_before_upload(tmp_path, monkeypatch, observed):
    model = _artifact(tmp_path)
    _close_all_slots(model)
    state = _FakeHubState(remote={}, gated=observed, accept_gate_setting=False)
    _install_fake_hub(monkeypatch, state)
    assert publish_cli([str(model), "--repo-id", "rdtand/test-artifact"]) == 1
    assert len(state.gate_updates) == 1
    assert not state.preupload_calls
    assert not state.create_calls


@pytest.mark.parametrize("field", ["gate_error", "gate_read_error"])
def test_auto_gate_api_failure_refuses_before_upload(tmp_path, monkeypatch, field):
    model = _artifact(tmp_path)
    _close_all_slots(model)
    state = _FakeHubState(remote={})
    setattr(state, field, OSError("Hub unavailable"))
    _install_fake_hub(monkeypatch, state)
    assert publish_cli([str(model), "--repo-id", "rdtand/test-artifact"]) == 1
    assert not state.preupload_calls
    assert not state.create_calls


def test_gate_removed_during_commit_cannot_report_success(tmp_path, monkeypatch, capsys):
    model = _artifact(tmp_path)
    _close_all_slots(model)
    state = _FakeHubState(remote={})
    state.mutate_before_commit = lambda: setattr(state, "gated", False)
    _install_fake_hub(monkeypatch, state)
    assert publish_cli([str(model), "--repo-id", "rdtand/test-artifact"]) == 1
    assert len(state.create_calls) == 1
    assert "inspect the repository" in capsys.readouterr().err


def test_destination_repository_must_match_canonical_license(tmp_path):
    model = _artifact(tmp_path)
    _close_all_slots(model)
    assert publish_cli(_argv(model, "--repo-id", "rdtand/different")) == 2
    assert publish_cli(_argv(model, "--repo-id", "different/test-artifact")) == 2
