"""The G3 release oracle must actually reach PB's current manifest validator."""
from __future__ import annotations

from pathlib import Path
import runpy
import sys

import pytest


def test_g3_release_oracle_reaches_current_manifest_validator(tmp_path):
    """Exercise the existing tiny oracle, without freezing its old API import.

    Keep this separate from the oracle: its API lookup is the implementation
    being repaired, while its omission/read-plan assertions remain coverage.
    Observe the real validator in whichever source scope the oracle owns;
    importing it here first would preselect the installed (older) PB graph.
    """
    validated = []
    previous_profile = sys.getprofile()

    def record_validation(frame, event, result):
        if previous_profile is not None:
            previous_profile(frame, event, result)
        if (event == "return"
                and frame.f_globals.get("__name__") == "prismabuild.data_manifest"
                and frame.f_code.co_name == "validate_data_manifest"):
            validated.append((frame.f_locals["value"], result))

    oracle_path = Path(__file__).parent / "g3job" / "test_g3_readset.py"
    oracle = runpy.run_path(str(oracle_path))[
        "test_manifest_omits_unused_source_and_tracks_repeated_teacher_reads"]
    try:
        sys.setprofile(record_validation)
        oracle(tmp_path)
    except ImportError as exc:
        pytest.fail(f"G3 release oracle still uses an unavailable PB API: {exc}", pytrace=False)
    except pytest.skip.Exception as exc:
        pytest.fail(f"G3 manifest contract must be validated, not skipped: {exc}", pytrace=False)
    finally:
        sys.setprofile(previous_profile)

    assert validated, "G3 release oracle did not call the current PB manifest validator"
    for manifest, normalized in validated:
        assert normalized["read_plan"] == manifest["read_plan"]
        assert manifest["annotations"]["teacher_repeats"] == 2
        assert [phase["name"] for phase in manifest["read_plan"]["phases"]] == [
            "setup", "layer-00", "teachers"]
