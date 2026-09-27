"""Exercise the prerequisite guards without claiming real DIO coverage."""
import inspect

import pytest
import test_render_identity_once_1192 as render_tests
import test_stageb_cotangent_scratch as scratch_tests
import test_stageb_one_pass_spill as spill_tests

from prismaquant.perturbed_x_cache import StageBSpillScratch


@pytest.mark.parametrize("supported", [False, True])
def test_direct_io_guard_records_missing_capability(monkeypatch, tmp_path, supported):
    probed = []

    def probe(directory):
        probed.append(directory)
        return supported

    monkeypatch.setattr(scratch_tests, "_direct_io_supported", probe)
    if supported:
        scratch_tests.require_direct_io(tmp_path)
    else:
        with pytest.raises(pytest.skip.Exception, match="requires a DIO-capable local PB worker"):
            scratch_tests.require_direct_io(tmp_path)
    assert probed == [tmp_path]


def test_preflight_only_spill_root_keeps_local_check_without_dio(monkeypatch, tmp_path):
    checked = []

    def needs_no_dio(directory):
        raise AssertionError("a pre-acquisition refusal must not probe DIO")

    monkeypatch.setenv("PQ_STAGE_B_SPILL_TEST_ROOT", str(tmp_path))
    monkeypatch.setattr(StageBSpillScratch, "require_local_root",
                        classmethod(lambda cls, root: checked.append(root)))
    monkeypatch.setattr(scratch_tests, "require_direct_io", needs_no_dio)
    root = spill_tests._spill_root(tmp_path, needs_direct_io=False)
    assert root.is_dir()
    assert checked == [root]


@pytest.mark.parametrize("cpus", [{0}, {0, 1}])
def test_two_worker_guard_respects_assigned_affinity(monkeypatch, cpus):
    # Calling the fixture body directly tests only the guard, not a PWC window.
    monkeypatch.setattr(render_tests.os, "sched_getaffinity", lambda pid: cpus)
    if len(cpus) < 2:
        with pytest.raises(pytest.skip.Exception, match="requires two assigned CPUs"):
            inspect.unwrap(render_tests.two_assigned_cpus)()
    else:
        inspect.unwrap(render_tests.two_assigned_cpus)()
