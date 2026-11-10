"""Namespace v2 binds native profile/runtime provenance and refuses the retired PB commit."""
import copy

import pytest

from container_admission import ADMISSION, with_admission
from tests.test_tessera_campaign_namespace_1986 import PROVENANCE, arguments
from tools import dispatch_tessera_campaign as dispatch
from tools import tessera_campaign_container as adapter
from tools import tessera_campaign_namespace as namespace


def prepare(tmp_path, provenance):
    kwargs = arguments(tmp_path)
    kwargs["provenance"] = kwargs["expected_provenance"] = provenance
    return dispatch.prepare_namespace_requests(**kwargs)


def test_the_schema_is_v2_and_the_provenance_keys_are_exact(tmp_path):
    assert namespace.SCHEMA == "prismaquant.tessera_campaign_namespace.v2"
    row = prepare(tmp_path, PROVENANCE)[0]
    binding = namespace.validate_namespace_request(row)
    assert set(binding["provenance"]) == {"tessera_commit", "container_content_sha256",
                                         "native_profile", "native_runtime_sha256"}
    assert set(binding["provenance"]["native_profile"]) == {"profile", "config_sha256", "r_sha256"}


@pytest.mark.parametrize("change", ["retired_key", "missing_runtime", "extra_key", "profile_extra",
                                    "profile_missing", "short_runtime", "empty_profile", "not_hex"])
def test_malformed_provenance_is_refused(tmp_path, change):
    provenance = copy.deepcopy(PROVENANCE)
    if change == "retired_key":
        del provenance["native_profile"]
        provenance["prismabuild_commit"] = "d" * 40
    elif change == "missing_runtime":
        del provenance["native_runtime_sha256"]
    elif change == "extra_key":
        provenance["extra"] = "x"
    elif change == "profile_extra":
        provenance["native_profile"]["uri"] = "local:///x"
    elif change == "profile_missing":
        del provenance["native_profile"]["r_sha256"]
    elif change == "short_runtime":
        provenance["native_runtime_sha256"] = "f" * 63
    elif change == "empty_profile":
        provenance["native_profile"]["profile"] = ""
    elif change == "not_hex":
        provenance["native_profile"]["config_sha256"] = "G" * 64
    with pytest.raises(RuntimeError, match="exact dependency provenance"):
        prepare(tmp_path, provenance)


def test_a_v1_binding_is_refused(tmp_path):
    row = prepare(tmp_path, PROVENANCE)[0]
    import json
    spec = json.loads(row["argv"][4])
    spec["namespace_binding"]["schema"] = "prismaquant.tessera_campaign_namespace.v1"
    row["argv"][4] = json.dumps(spec)
    with pytest.raises(RuntimeError, match="namespace binding is missing or malformed"):
        namespace.validate_namespace_request(row)


def test_the_sealed_native_profile_must_be_the_admitting_launchs(tmp_path):
    import json
    spec = json.loads(prepare(tmp_path, PROVENANCE)[0]["argv"][4])
    adapter.require_namespace_native_profile(spec, ADMISSION)
    other = with_admission(profile={**ADMISSION["profile"], "config_sha256": "9" * 64})
    with pytest.raises(RuntimeError, match="native profile differs from the admitting launch"):
        adapter.require_namespace_native_profile(spec, other)
    adapter.require_namespace_native_profile({}, other)  # a legacy unbound row is not a namespace row
