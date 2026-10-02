"""Metadata/admission composition; no Docker, GPU, Netdata or native profiler."""
import copy
import json
from pathlib import Path

import pytest

from tests.test_tessera_campaign_namespace_1986 import arguments, replace_fixture_request
from tools import dispatch_tessera_campaign as dispatch
from tools import pq_admitted_profile as profiler
from tools import tessera_campaign_namespace as namespace


def profiled_arguments(tmp_path):
    kwargs = arguments(tmp_path)
    row = kwargs["requests"][0]
    spec = json.loads(row["argv"][4])
    spec["namespace_profile"] = {
        "schema": "prismaquant.tessera_namespace_profile.v1",
        "profiler": {"path": "/readonly/py-spy", "sha256": "3" * 64},
        "observations": "/old/profile",
        "profile_local": str(profiler.PROFILE_LOCAL_ROOT / "unbound"),
        "row_s": 900,
    }
    local = str(profiler.PROFILE_LOCAL_ROOT)
    spec["container"]["mounts"].append({"source": local, "target": local, "readonly": False})
    row["argv"][4] = json.dumps(spec)
    row["argv"] = profiler.profiled_row_command(row["argv"],
        destination="/old/profile/child-profile.speedscope", profiler_executable="/readonly/py-spy")
    replace_fixture_request(kwargs, row)
    return kwargs


def test_composed_request_is_prepared_before_binding(tmp_path):
    kwargs = profiled_arguments(tmp_path)
    before = copy.deepcopy(kwargs["requests"])
    rows = dispatch.prepare_namespace_requests(**kwargs)
    assert rows == dispatch.prepare_namespace_requests(**kwargs)
    assert kwargs["requests"] == before
    assert "tools.pq_profile_child" in rows[0]["argv"]
    namespace.validate_namespace_request(rows[0])


def test_post_binding_wrapper_still_refuses_actual_request_change(tmp_path):
    row = dispatch.prepare_namespace_requests(**arguments(tmp_path))[0]
    row["argv"] = profiler.profiled_row_command(row["argv"],
        destination=str(tmp_path / "foreign-profile"), profiler_executable="/readonly/py-spy")
    with pytest.raises(RuntimeError):
        namespace.validate_namespace_request(row)
