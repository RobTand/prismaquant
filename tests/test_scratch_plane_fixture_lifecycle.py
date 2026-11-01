"""Exercise the real scratch-plane yield fixture without qualifying spill IO.

The adapters below model unavailable prerequisites and observation events only;
no result here is a spill/chain numerical-equality or disk-qualification result.
"""
import os
from types import SimpleNamespace

import pytest

import test_band_serial_scratch_plane as plane_tests
import test_stageb_cotangent_scratch as cotangent_tests
from prismaquant.joint_quantum_handoff import HandoffStream
from prismaquant.perturbed_x_cache import ExactCotangentScratch, StageBSpillScratch


SPILL_CASES = (
    plane_tests.test_a_band_serial_producer_runs_the_campaign_regime,
    plane_tests.test_the_batched_capture_plane_is_the_batched_chain_plane,
    plane_tests.test_the_control_at_batch_one_still_holds,
    plane_tests.test_band_serial_under_the_spill_equals_the_chain_rebuild,
)
PLANE_ENV = (
    "PRISMAQUANT_STAGE_B_COTANGENT_ROOT",
    "PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES",
)


def _fixture(case, monkeypatch, directory):
    directory.mkdir()
    return plane_tests.cotangent_scratch_plane.__wrapped__(
        tmp_path=directory, monkeypatch=monkeypatch,
        request=SimpleNamespace(function=case),
    )


def _assert_execution_observations_remain_strict(case, monkeypatch, tmp_path):
    """These are observer test doubles, not a replacement scratch IO engine."""
    with monkeypatch.context() as patch:
        patch.setenv("PQ_STAGE_B_SPILL_TEST_ROOT", str(tmp_path / "unit-spill"))
        patch.setattr(StageBSpillScratch, "require_local_root", lambda root: None)
        dio_probes = []
        patch.setattr(cotangent_tests, "require_direct_io",
                      lambda root: dio_probes.append(root.name))
        patch.setattr(ExactCotangentScratch, "__init__", lambda self, *a, **kw: None)
        patch.setattr(HandoffStream, "__exit__", lambda *a: None)

        observations = (
            # An executed body with no observed construction must still fail.
            ("unobserved", False, None, "no cotangent scratch was built"),
            ("no-handoff", True, None, "no handoff was written from a scratch plane"),
            ("dict-plane", True, {"scratch": False}, "no handoff was written from a scratch plane"),
            ("lost-entry", True, {"entries": 3, "tee_hits": 1, "scratch_reads": 1}, ""),
            ("read-error", True, {"error": "controlled read failure"}, ""),
            ("accounted", True, {}, None),
        )
        for name, construct, stream, failure in observations:
            # Each generator owns fresh monkeypatch state, just as pytest does.
            with patch.context() as execution_patch:
                dio_probes.clear()
                fixture = _fixture(case, execution_patch, tmp_path / name)
                next(fixture)
                assert "cotangent-scratch" in dio_probes
                if construct:
                    ExactCotangentScratch()  # records construction; no allocation
                if stream is not None:
                    values = dict(entries=3, tee_hits=1, scratch_reads=2, error=None)
                    values.update({key: value for key, value in stream.items()
                                   if key != "scratch"})
                    handoff = SimpleNamespace(
                        _scratch=object() if stream.get("scratch", True) else None,
                        telemetry=values,
                    )
                    HandoffStream.__exit__(handoff, None, None, None)
                try:
                    if failure is None:
                        with pytest.raises(StopIteration):
                            next(fixture)
                    else:
                        with pytest.raises(AssertionError, match=failure or None):
                            next(fixture)
                finally:
                    fixture.close()


@pytest.mark.parametrize("case", SPILL_CASES, ids=lambda case: case.__name__)
@pytest.mark.parametrize("gate", ("missing-root", "spill-direct-io", "invalid-root"))
def test_spill_prerequisites_precede_execution_observation(case, gate, monkeypatch,
                                                          tmp_path):
    # Preserve the strict observer contract as part of every regression check.
    _assert_execution_observations_remain_strict(case, monkeypatch, tmp_path)

    with monkeypatch.context() as patch:
        if gate == "missing-root":
            patch.delenv("PQ_STAGE_B_SPILL_TEST_ROOT", raising=False)
            expected = pytest.skip.Exception
            reason = "PQ_STAGE_B_SPILL_TEST_ROOT"
        else:
            # An explicitly labeled unit adapter: never a qualified storage path.
            patch.setenv("PQ_STAGE_B_SPILL_TEST_ROOT", str(tmp_path / "unit-spill"))
            expected = ValueError if gate == "invalid-root" else pytest.skip.Exception
            reason = "controlled local-root refusal" if gate == "invalid-root" else "controlled spill DIO unavailable"

        original_init = ExactCotangentScratch.__init__
        original_exit = HandoffStream.__exit__
        environment = {name: os.environ.get(name) for name in PLANE_ENV}
        probes = []

        def assert_uninstrumented():
            assert ExactCotangentScratch.__init__ is original_init
            assert HandoffStream.__exit__ is original_exit
            assert {name: os.environ.get(name) for name in PLANE_ENV} == environment

        def local_root(root):
            assert_uninstrumented()
            probes.append("local-root")
            if gate == "invalid-root":
                raise ValueError("controlled local-root refusal")

        def direct_io(root):
            assert_uninstrumented()
            if root.name == "cotangent-scratch":
                probes.append("cotangent-dio")
            else:
                probes.append("spill-dio")
                if gate == "spill-direct-io":
                    pytest.skip("controlled spill DIO unavailable")

        patch.setattr(StageBSpillScratch, "require_local_root", local_root)
        patch.setattr(cotangent_tests, "require_direct_io", direct_io)
        fixture = _fixture(case, patch, tmp_path / "preflight")
        try:
            # A prerequisite skip/refusal must occur on setup, not in the body
            # after this generator has yielded and acquired strict postconditions.
            with pytest.raises(expected, match=reason):
                next(fixture)
            assert_uninstrumented()
            with pytest.raises(StopIteration):
                next(fixture)  # no teardown assertion may follow the skip
            if gate == "spill-direct-io":
                assert "local-root" in probes and "spill-dio" in probes
            elif gate == "invalid-root":
                assert "local-root" in probes
        finally:
            fixture.close()
