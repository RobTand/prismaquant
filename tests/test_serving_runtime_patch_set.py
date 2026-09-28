"""The recorded GLM-5.3 MLA patch set says what it is, and refuses what it is not."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from prismaquant.serving_runtime_patch_set import (
    SERVING_RUNTIME_PATCH_SETS_DIR,
    ServingRuntimePatchSetError,
    load_serving_runtime_patch_set,
)

NAME = "glm53_nope_sm120"


def _manifest() -> dict:
    path = SERVING_RUNTIME_PATCH_SETS_DIR / NAME / "MANIFEST.json"
    return json.loads(path.read_text())


def _write(tmp_path: Path, manifest: dict) -> Path:
    """Stage a copy of the real patch set under a mutated manifest.

    The driver is mutated, never the fixture's shape: a bad input has to prove
    the check bites on the artifact this repository actually ships.
    """
    directory = tmp_path / NAME
    directory.mkdir()
    for patch in _manifest()["patches"]:
        source = SERVING_RUNTIME_PATCH_SETS_DIR / NAME / patch["script"]
        (directory / patch["script"]).write_bytes(source.read_bytes())
    (directory / "MANIFEST.json").write_text(json.dumps(manifest))
    return tmp_path


def test_the_shipped_patch_set_loads_and_carries_its_scope() -> None:
    patch_set = load_serving_runtime_patch_set(NAME)
    assert patch_set.base_image.startswith("eugr/spark-vllm@sha256:")
    assert patch_set.qualified_model_layers == 4
    assert patch_set.full_model_layers == 45
    assert not patch_set.is_fully_qualified
    sentence = patch_set.scope_sentence()
    assert "4 of 45 layers" in sentence
    assert "RECORDED, not attested" in sentence


def test_the_five_edits_are_carried_as_files_not_as_an_image_tag() -> None:
    patch_set = load_serving_runtime_patch_set(NAME)
    by_script = {p["script"]: p for p in patch_set.patches}
    mla = by_script["patch_glm53_nope_sm120.py"]
    assert len(mla["edits"]) == 5, "the issue's five edits are the artifact"
    body = (patch_set.directory / "patch_glm53_nope_sm120.py").read_bytes()
    assert hashlib.sha256(body).hexdigest() == mla["sha256"]
    # The blocking edit, by name, so a silent rewrite of the manifest shows up.
    assert any(e["tag"] == "do_kv_cache_update" for e in mla["edits"])
    assert (patch_set.directory / "Dockerfile").is_file()


def test_the_full_body_is_refused_because_four_layers_is_not_forty_five() -> None:
    patch_set = load_serving_runtime_patch_set(NAME)
    patch_set.require_qualified_for(model_layers=4)
    with pytest.raises(ServingRuntimePatchSetError, match="qualified on 4 layers"):
        patch_set.require_qualified_for(model_layers=45)


def test_a_local_image_id_is_not_offered_as_a_pullable_reference() -> None:
    patch_set = load_serving_runtime_patch_set(NAME)
    assert patch_set.derived_image is None
    assert patch_set.derived_image_local_id.startswith("sha256:")
    with pytest.raises(ServingRuntimePatchSetError, match="cannot be pulled"):
        patch_set.serve_image_reference()


def test_a_tagged_base_image_is_refused(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["base_image"] = "eugr/spark-vllm:latest"
    with pytest.raises(ServingRuntimePatchSetError, match="A tag is not a pin"):
        load_serving_runtime_patch_set(NAME, root=_write(tmp_path, manifest))


def test_a_derived_image_with_a_registry_port_is_refused(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["derived_image"] = (
        "192.168.1.107:5000/prismaquant/glm53-nope-sm121@sha256:" + "0" * 64)
    with pytest.raises(ServingRuntimePatchSetError, match="port 80"):
        load_serving_runtime_patch_set(NAME, root=_write(tmp_path, manifest))


def test_a_patch_set_claiming_attestation_is_refused(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["attested"] = True
    with pytest.raises(ServingRuntimePatchSetError, match="principle 14"):
        load_serving_runtime_patch_set(NAME, root=_write(tmp_path, manifest))


def test_a_patch_set_with_no_stated_limits_is_refused(tmp_path: Path) -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["qualification"]["not_qualified"] = []
    with pytest.raises(ServingRuntimePatchSetError, match="limits were not looked for"):
        load_serving_runtime_patch_set(NAME, root=_write(tmp_path, manifest))


def test_a_manifest_naming_a_script_it_does_not_carry_is_refused(tmp_path: Path) -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["patches"][0]["script"] = "patch_that_lives_in_a_scratch_dir.py"
    with pytest.raises(ServingRuntimePatchSetError, match="image tag problem again"):
        load_serving_runtime_patch_set(NAME, root=_write(tmp_path, manifest))


MTP_MAPPER = "glm53_mtp_mapper"


def test_the_mtp_mapper_set_names_its_image_by_registry_digest() -> None:
    patch_set = load_serving_runtime_patch_set(MTP_MAPPER)
    reference = patch_set.serve_image_reference()
    assert reference == (
        "192.168.1.107/prismaquant/spark-vllm-nccl230@sha256:"
        "f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5")
    # Built FROM the image the routed Tessera cells name, by digest, not a tag.
    assert patch_set.base_image.endswith(
        "@sha256:a5424378322071f4c33e63d1372a2bb028e46b03f0da0e5edb0cdd7418e2cebb")


def test_the_mtp_mapper_script_is_the_bytes_the_image_ran() -> None:
    patch_set = load_serving_runtime_patch_set(MTP_MAPPER)
    (patch,) = patch_set.patches
    body = (patch_set.directory / patch["script"]).read_bytes()
    assert hashlib.sha256(body).hexdigest() == patch["sha256"]
    # The script refuses any mtp.py but the base image's, by this exact hash.
    assert patch["base_sha256"].encode() in body
    assert {e["tag"] for e in patch["edits"]} == {
        "import_weights_mapper", "glm5next_mtp_hf_to_vllm_mapper"}


def test_the_mtp_mapper_set_is_qualified_by_the_r6_census_and_no_wider() -> None:
    """r6 (u4-BAL-20260928T0540Z-2c-r6-2c) served the 45-layer body plus the
    drafter on the derived image, with a VALID census receipt (PQ #1490)."""
    patch_set = load_serving_runtime_patch_set(MTP_MAPPER)
    assert patch_set.qualified_model_layers == patch_set.full_model_layers == 45
    patch_set.require_qualified_for(model_layers=45)
    with pytest.raises(ServingRuntimePatchSetError, match="qualified on 45 layers"):
        patch_set.require_qualified_for(model_layers=46)
    assert patch_set.qualification["modes_ready"] == ["eager"]
    assert patch_set.raw["receipts"]["census_run_id"] == "u4-BAL-20260928T0540Z-2c-r6-2c"


KPOOL_TAIL = "glm53_kpool_tail_slot_mapping"


def test_the_kpool_tail_set_is_built_on_the_mtp_mapper_image_by_digest() -> None:
    """The backport stacks on the image the GLM Tessera cells name, not on a tag."""
    patch_set = load_serving_runtime_patch_set(KPOOL_TAIL)
    assert patch_set.serve_image_reference() == (
        "192.168.1.107/prismaquant/spark-vllm-nccl230@sha256:"
        "c2e75e03cfc52c15489b40fe58e65acb7347f6fa3ddf2e81afda86760698147b")
    assert patch_set.base_image == (
        load_serving_runtime_patch_set(MTP_MAPPER).serve_image_reference())


def test_the_kpool_tail_script_is_the_bytes_the_image_ran() -> None:
    patch_set = load_serving_runtime_patch_set(KPOOL_TAIL)
    (patch,) = patch_set.patches
    body = (patch_set.directory / patch["script"]).read_bytes()
    assert hashlib.sha256(body).hexdigest() == patch["sha256"]
    # The script refuses any model_runner.py but the base image's, by this hash.
    assert patch["base_sha256"].encode() in body
    assert patch["target_file"] == "vllm/v1/worker/gpu/model_runner.py"
    assert {e["tag"] for e in patch["edits"]} == {
        "import_kpool_tail_spec", "kpool_tail_takes_no_generic_slot_mapping"}
    # The image's self-check asserts the edited file's hash, and leaves the
    # slot-mapping kernel itself stock: only the runner's rule moved.
    check = (patch_set.directory / patch_set.raw["self_check_script"]).read_text()
    assert patch["patched_sha256"] in check
    assert patch_set.raw["unchanged_stock_sources"]["v1/worker/gpu/block_table.py"] in check


def test_the_kpool_tail_set_is_qualified_on_the_stub_and_no_wider() -> None:
    patch_set = load_serving_runtime_patch_set(KPOOL_TAIL)
    assert patch_set.qualified_model_layers == 4
    assert patch_set.full_model_layers == 45
    assert not patch_set.is_fully_qualified
    patch_set.require_qualified_for(model_layers=4)
    with pytest.raises(ServingRuntimePatchSetError, match="qualified on 4 layers"):
        patch_set.require_qualified_for(model_layers=45)
    assert "4 of 45 layers" in patch_set.scope_sentence()


def test_the_kpool_tail_set_records_the_graph_modes_tessera_508_measured() -> None:
    """The stub serves on this image eager and in the three CUDA-graph modes
    the Tessera gate admits on its runner (tessera PR #699); the drafter and a
    padded capture list stay stated limits, not modes."""
    patch_set = load_serving_runtime_patch_set(KPOOL_TAIL)
    q = patch_set.qualification
    assert q["status"] == "measured_on_a_4_layer_stub"
    assert q["modes_ready"] == [
        "eager", "cuda_graph_full_decode_only", "cuda_graph_piecewise",
        "cuda_graph_full_and_piecewise"]
    assert q["tensor_parallel"] == 1
    limits = " ".join(q["not_qualified"])
    assert "MTP drafter" in limits and "padded" in limits and "2048 tokens" in limits
    # The runner the Tessera gate admits graphs on is the one this image carries.
    (patch,) = patch_set.patches
    assert patch["patched_sha256"] == (
        "1c30b8c0d3ffc96172cba57965cb7c3648156ce693ea6b3232edd2f36a9d1781")
    assert patch_set.raw["receipts"]["tessera_pr"].endswith("/tessera/pull/699")
