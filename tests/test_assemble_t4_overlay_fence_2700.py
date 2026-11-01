"""The release-gate regressions must not depend on filesystem clock progress.

Exercise the original tests themselves: ordinary writes and hardlinks may have
identical observed timestamps. The catalog's stat fast path is intentional;
the gate fixtures must arrange drift before claiming to test a content rehash.
"""

import importlib.util
import inspect
from pathlib import Path

import pytest

import prismaquant.joint_catalog_extension as jce


@pytest.fixture
def release_gate():
    path = Path(__file__).with_name("test_assemble_t4_overlay_fence.py")
    spec = importlib.util.spec_from_file_location("overlay_release_gate_2700", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_gate(module, name, request):
    test = getattr(module, name)
    test(**{name: request.getfixturevalue(name) for name in inspect.signature(test).parameters})


@pytest.fixture
def colliding_stamps(release_gate, monkeypatch):
    # Model indistinguishable timestamp observations, not unchanged file bytes.
    # Both the fixture's recording and the production fence see the same clock.
    # Do not change inode/size, stub a digest, or replace the fence decision.
    def frozen(stamp):
        return dict(stamp, mtime_ns=1_700_000_000_000_000_000, ctime_ns=1_700_000_000_000_000_000)

    stamp = release_gate._stamp
    catalog_stamp = jce.catalog_stat_fence
    monkeypatch.setattr(release_gate, "_stamp", lambda path: frozen(stamp(path)))
    monkeypatch.setattr(jce, "catalog_stat_fence", lambda stat: frozen(catalog_stamp(stat)))
    monkeypatch.setattr(jce, "FENCE_REHASHED", {})
    hashes = []
    real_hash = jce.file_digest_sha256hex

    def observed_hash(handle):
        digest = real_hash(handle)
        hashes.append((Path(handle.name).name, digest))
        return digest

    monkeypatch.setattr(jce, "file_digest_sha256hex", observed_hash)
    return hashes


def test_hardlink_gate_exercises_real_rehash_without_clock_progress(release_gate, colliding_stamps, request):
    _run_gate(release_gate, "test_assemble_accepts_a_wire_whose_ctime_moved_under_a_new_hardlink", request)
    scratch = request.getfixturevalue("tmp_path")
    assert (scratch / "wire.bin").samefile(scratch / "linked-by-a-later-merge.bin")
    assert len(colliding_stamps) == 1, "the gate must actually hash the drifted wire"
    assert colliding_stamps[0][0] == "wire.bin"
    assert jce.FENCE_REHASHED == {"assemble catalog wire fence": 1}


def test_corruption_gate_exercises_real_rehash_without_clock_progress(release_gate, colliding_stamps, request):
    refusals = []
    real_rehash = jce._rehash_drifted

    def observed_rehash(*args, **kwargs):
        try:
            return real_rehash(*args, **kwargs)
        except ValueError as error:
            refusals.append(str(error))
            raise

    request.getfixturevalue("monkeypatch").setattr(jce, "_rehash_drifted", observed_rehash)
    _run_gate(release_gate, "test_assemble_refuses_a_same_size_wire_with_different_bytes", request)
    assert len(colliding_stamps) == 1, "corrupt bytes must be read by the real digest owner"
    assert colliding_stamps[0][0] == "wire.bin"
    assert len(refusals) == 1
    assert "content re-hash after stat drift" in refusals[0]
    assert jce.FENCE_REHASHED == {}, "a refused wire is never counted as an admitted rehash"


@pytest.mark.parametrize(
    "gate_name",
    [
        "test_assemble_accepts_a_wire_whose_ctime_moved_under_a_new_hardlink",
        "test_assemble_refuses_a_same_size_wire_with_different_bytes",
    ],
    ids=["hardlink", "corruption"],
)
def test_release_gate_restores_existing_rehash_report(release_gate, monkeypatch, request, gate_name):
    """Running either gate must not erase or add to another caller's report.

    The outer fixture protects the suite's real report. The inner scope models
    teardown of the gate's own monkeypatch fixture, including when the gate is
    invoked by the frozen-clock regression tests above.
    """
    existing_report = {"an earlier catalog fence": 7}
    monkeypatch.setattr(jce, "FENCE_REHASHED", existing_report)
    gate = getattr(release_gate, gate_name)
    with pytest.MonkeyPatch.context() as gate_patch:
        fixtures = {
            name: gate_patch if name == "monkeypatch" else request.getfixturevalue(name)
            for name in inspect.signature(gate).parameters
        }
        gate(**fixtures)

    assert jce.FENCE_REHASHED is existing_report, "gate teardown must restore the incoming report object"
    assert existing_report == {"an earlier catalog fence": 7}, "gate must not mutate the incoming report"


def test_corruption_gate_does_not_mistake_unrelated_errors_for_content_refusal(release_gate, monkeypatch, request):
    real_tool = release_gate._tool

    def broken_fence(cell):
        raise RuntimeError("2700: unrelated fence failure")

    def tool_with_unrelated_failure():
        tool = real_tool()
        monkeypatch.setattr(tool, "fence_cell_artifacts", broken_fence)
        return tool

    monkeypatch.setattr(release_gate, "_tool", tool_with_unrelated_failure)
    with pytest.raises(RuntimeError, match="2700: unrelated fence failure"):
        _run_gate(release_gate, "test_assemble_refuses_a_same_size_wire_with_different_bytes", request)
