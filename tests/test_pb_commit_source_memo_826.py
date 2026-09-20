"""A dev-mode progress commit hashes the source tree once, not per unit (#826).

Stage A v14 (action 398c81b4, 2026-09-20) spent its whole attempt at ~0.3
units/s with the main thread inside ``_production_cache_source_sha256``'s
rglob/sort/hash, called from ``_pb_commit``'s dev stamp: every durable unit
re-derived the executing package's identity. The walk's stall budget held
only because the commits trickled out at all.

Two properties:

* **Once per process** -- however many units commit, the underlying
  ``_aura_source_sha256`` runs exactly one time; every stamp after the
  first reuses it.
* **Same value every stamp** -- the memo returns the first digest
  verbatim, so all progress records in one run carry one identity.
"""

from __future__ import annotations

import pytest

from prismaquant import tessera_joint_aura


def test_dev_source_sha256_computed_once_across_many_commits(monkeypatch):
    calls = []

    def fake_sha():
        calls.append(1)
        return "a" * 64

    import prismaquant.aura_cost as aura_cost

    monkeypatch.setattr(aura_cost, "_aura_source_sha256", fake_sha, raising=True)
    monkeypatch.setattr(tessera_joint_aura, "_DEV_SOURCE_SHA256_MEMO", None, raising=True)
    try:
        first = tessera_joint_aura._progress_dev_source_sha256()
        for _ in range(1000):
            assert tessera_joint_aura._progress_dev_source_sha256() == first
        assert calls == [1]
        assert first == "a" * 64
    finally:
        tessera_joint_aura._DEV_SOURCE_SHA256_MEMO = None


def test_pb_commit_dev_stamp_reuses_the_memo(monkeypatch, tmp_path):
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_PATH", str(tmp_path / "progress.json"))
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    monkeypatch.setenv("PRISMAQUANT_DEV_PROGRESS_STAMP", "1")
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_TOKEN", "tok")

    calls = []
    monkeypatch.setattr(tessera_joint_aura, "_DEV_SOURCE_SHA256_MEMO", "b" * 64, raising=True)
    real_stamp = tessera_joint_aura.dev_stamp

    def counting(value):
        calls.append(value)
        return real_stamp(value)

    monkeypatch.setattr(tessera_joint_aura, "dev_stamp", counting, raising=True)
    try:
        assert tessera_joint_aura._pb_commit(1, "head", unit="layers.0.a")
        assert tessera_joint_aura._pb_commit(2, "head", unit="layers.0.b")
        assert calls == ["b" * 64, "b" * 64]
        import json

        # The progress record is atomically replaced per commit, so the file
        # holds the latest line: the second commit's record.
        row = json.loads((tmp_path / "progress.json").read_text())
        assert row["units_completed"] == 2
        assert row["unit"] == "layers.0.b"
    finally:
        tessera_joint_aura._DEV_SOURCE_SHA256_MEMO = None


def test_pb_commit_default_dev_path_never_hashes_the_source(monkeypatch, tmp_path):
    """The default dev path pays no sealing ceremony: no stamp, no hash.

    The per-line stamp is opt-in (PR #828): with ``PRISMAQUANT_DEV_MODE=1``
    and no ``PRISMAQUANT_DEV_PROGRESS_STAMP``, a durable-unit commit must
    neither hash the executing tree nor alter the certified six-field record.
    The tree walk on this path is the regression #826 measured.
    """
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_PATH", str(tmp_path / "progress.json"))
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_TOKEN", "tok")

    def refuse_hash():
        raise AssertionError("default dev progress commits must not hash the source tree")

    monkeypatch.setattr(tessera_joint_aura, "_progress_dev_source_sha256",
                        refuse_hash, raising=True)
    assert tessera_joint_aura._pb_commit(1, "head", unit="layers.0.a")
    assert tessera_joint_aura._pb_commit(2, "head", unit="layers.0.b")

    import json

    row = json.loads((tmp_path / "progress.json").read_text())
    assert set(row) == {"schema", "token", "phase", "units_completed", "unit",
                        "reported_unix"}
    assert row["units_completed"] == 2
    assert row["unit"] == "layers.0.b"
