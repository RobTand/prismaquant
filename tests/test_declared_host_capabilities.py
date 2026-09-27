"""Exercise the prerequisite guards without claiming real DIO coverage."""
import inspect

import pytest
import test_render_identity_once_1192 as render_tests
import test_stageb_cotangent_scratch as scratch_tests


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


@pytest.mark.parametrize("cpus", [{0}, {0, 1}])
def test_two_worker_guard_respects_assigned_affinity(monkeypatch, cpus):
    # Calling the fixture body directly tests only the guard, not a PWC window.
    monkeypatch.setattr(render_tests.os, "sched_getaffinity", lambda pid: cpus)
    if len(cpus) < 2:
        with pytest.raises(pytest.skip.Exception, match="requires two assigned CPUs"):
            inspect.unwrap(render_tests.two_assigned_cpus)()
    else:
        inspect.unwrap(render_tests.two_assigned_cpus)()
