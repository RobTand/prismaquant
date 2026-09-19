"""An attested rounding rule covers the image it was generated in, and no other.

The pinned Tessera contract generates its activation quantiser table by RUNNING
the runtime's kernel, and says in which image
(``activation_quantizers.platforms.sm_121.generated.image =
vllm/vllm-openai@sha256:61fc8a89…``, vLLM ``0.28.0``). The seven
``TESSERA_E2M1_K2`` cells of 2026-09-18 executed in
``eugr/spark-vllm@sha256:0afec8d4…`` (vLLM ``0.28.1rc1.dev397+gfd4a15126``) --
a different build of the same operator -- and carried an attestation that says
nothing about it. Nothing refused them (RobTand/prismaquant#715).

These drive the real reader and the real consumer over the real receipt of one
of those cells:

* the stamp is produced by ``require_activation_quantizer_attested`` reading the
  published table, never hand-written here, so what is compared is what the
  contract bytes say;
* the receipt is the one that was measured, re-digested after each mutation, so
  every other gate in ``consume_native_receipt`` still has to pass;
* the mutation moves the EXECUTING image and the answer moves with it.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from prismaquant import tessera_runtime_contract as trc
from prismaquant.joint_aura import identity_sha256
from prismaquant.native_operator_panel import consume_native_receipt


def _scope(panel, executing_image):
    """The gate under test, imported here so the file still COLLECTS without it.

    Before this change the symbol does not exist, and a collection error would
    have made every case in this file red for the same uninformative reason.
    The cases that matter -- the measured cell, and the stamp with no scope --
    are then red because the consumer ADMITS them, which is the defect.
    """
    from prismaquant.native_operator_panel import require_panel_execution_scope

    return require_panel_execution_scope(panel, executing_image=executing_image)


CONTRACT = "e2m1_group16_ue4m3_static"
PLATFORM = "sm_121"
#: The pinned contract's digest, as the cells stamped it. Re-pinned with the
#: v31 contract (Tessera #551, PrismaQuant #699); the quantizer table fixture
#: is byte-identical there, so only the digest moves.
PIN_SHA256 = "80d58f1a528638339a2d74c6e5b97a9a8f0458687515db531a4685489aa05809"
EXECUTED = ("eugr/spark-vllm@sha256:"
            "0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c")
ATTESTED = ("vllm/vllm-openai@sha256:"
            "61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14")

CELL = Path(__file__).parent / "fixtures" / "native_fp4_o_proj_20260918"
TABLE = Path(__file__).parent / "fixtures" / "tessera_activation_quantizers_v25.json"


def _stamp():
    """The stamp the producer freezes, from the published table's own bytes."""
    table = trc._parse_activation_quantizers(
        json.loads(TABLE.read_text()), "<fixture>")
    return trc.require_activation_quantizer_attested(
        CONTRACT, platform=PLATFORM, table=table, contract_sha256=PIN_SHA256)


def _cell(tmp_path, *, stamp, executing_image):
    """The measured receipt, with one panel field moved and everything redigested.

    `consume_native_receipt` binds the receipt to the panel four ways (the
    panel, its digest, the runtime, its digest) and to the file by sha256, so a
    mutation that is not carried through all of them fails on a different gate
    than the one under test.
    """
    panel = json.loads((CELL / "panel.json").read_text())
    receipt = json.loads((CELL / "receipt.json").read_text())
    panel["activation_quantizer_attestation"] = copy.deepcopy(stamp)
    panel["runtime"] = copy.deepcopy(panel["runtime"])
    panel["runtime"]["image"] = executing_image
    # The launcher's own record of the same resolution. Nothing cross-checks
    # it against `image` and this gate does not read it -- it is the reference
    # the launcher resolved, not a second witness -- but a fixture whose
    # runtime disagrees with itself would be reading as evidence of something
    # nobody measured.
    declaration = panel["runtime"].get("image_declaration")
    if isinstance(declaration, dict):
        for field in ("requested", "required", "resolved_reference"):
            if field in declaration:
                declaration[field] = executing_image
        if "resolved_digest" in declaration:
            declaration["resolved_digest"] = executing_image.split("@", 1)[1]
        if "repo_digests" in declaration:
            declaration["repo_digests"] = [executing_image]
    receipt["panel"] = copy.deepcopy(panel)
    receipt["panel_sha256"] = identity_sha256(panel)
    receipt["runtime"] = copy.deepcopy(panel["runtime"])
    receipt["runtime_sha256"] = identity_sha256(panel["runtime"])
    raw = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)
           + "\n").encode()
    path = tmp_path / "receipt.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest(), panel


def test_the_published_table_carries_the_image_it_was_generated_in():
    """Read from the contract bytes, not asserted by a driver."""
    stamp = _stamp()
    assert stamp["generated"] == {
        "image": ATTESTED, "vllm": "0.28.0", "torch": "2.13.0+cu130",
        "device": "NVIDIA GB10", "compute_capability": "12.1",
        "driver": "595.84",
        "generator_sha256":
            "5af82866c9b7a26808a9311a1402c605a0ee04c205ba709283dc733301de3af0"}
    assert stamp["generated_absent_because"] is None


def test_a_cell_measured_in_the_attested_image_is_admitted(tmp_path):
    """The control: same image on both sides, and the consumer admits."""
    panel_path, digest, panel = _cell(tmp_path, stamp=_stamp(),
                                      executing_image=ATTESTED)
    observation = consume_native_receipt(panel_path, expected_sha256=digest,
                                         expected_panel=panel)
    assert observation["status"] == "operator_evidence"
    scope = observation["activation_scope"]
    assert scope["attested_image"] == ATTESTED
    assert scope["executing_image"] == ATTESTED
    assert scope["compared"] == "image_content_digest"


def test_the_host_facts_are_recorded_on_both_sides_and_compared_on_neither(tmp_path):
    """The digest pins what is inside the image, and nothing outside it.

    This cell ran on driver 595.91.07; the table was generated on 595.84. The
    two disagree, the image digests agree, and the panel is admitted -- by
    ruling (Rob, 2026-09-18): a driver change never invalidates an attestation,
    drivers move all the time and nothing is revalidated for one. Both sides
    are recorded for the card and never compared.
    """
    panel_path, digest, panel = _cell(tmp_path, stamp=_stamp(),
                                      executing_image=ATTESTED)
    gpu = panel["runtime"]["gpu"]
    assert gpu["driver_version"] == "595.91.07"
    scope = consume_native_receipt(panel_path, expected_sha256=digest,
                                   expected_panel=panel)["activation_scope"]
    assert scope["schema"] == "prismaquant.panel_execution_scope.v2"
    assert (scope["attested_driver"], scope["executing_driver"]) == (
        "595.84", "595.91.07")
    assert (scope["attested_device"], scope["executing_device"]) == (
        "NVIDIA GB10", "NVIDIA GB10")
    assert scope["attested_compute_capability"] == "12.1"
    assert scope["executing_compute_capability"] == gpu["capability"]
    assert scope["not_compared"] == {
        "pinned_by_the_image_digest": ["vllm", "torch"],
        "host_facts_outside_the_image": ["driver", "device",
                                         "compute_capability"]}
    assert "inside the image" in scope["why_not_compared"][
        "pinned_by_the_image_digest"]
    assert "host facts outside the image" in scope["why_not_compared"][
        "host_facts_outside_the_image"]


def test_a_panel_with_no_gpu_record_says_none_rather_than_a_placeholder():
    stamp = _stamp()
    scope = _scope({"activation_quantizer_attestation": stamp,
                    "runtime": {"image": ATTESTED}}, ATTESTED)
    assert scope["executing_driver"] is None
    assert scope["executing_device"] is None
    assert scope["executing_compute_capability"] is None


def test_the_measured_09_18_cell_is_refused_and_both_digests_are_named(tmp_path):
    """The shape that was actually measured: eugr, attested vllm/vllm-openai."""
    panel_path, digest, panel = _cell(tmp_path, stamp=_stamp(),
                                      executing_image=EXECUTED)
    with pytest.raises(ValueError) as refusal:
        consume_native_receipt(panel_path, expected_sha256=digest,
                               expected_panel=panel)
    message = str(refusal.value)
    assert "sha256:sha256:" not in message
    assert f"sha256:{ATTESTED.split('@sha256:')[1]}" in message
    assert f"sha256:{EXECUTED.split('@sha256:')[1]}" in message
    assert "0.28.0" in message


def test_the_receipt_as_frozen_carries_no_scope_and_is_not_verified(tmp_path):
    """The fixture verbatim: a real attestation, frozen before #715.

    Not a pass and not a mismatch -- there is no scope to compare, which is the
    state the stamp used to leave behind silently.
    """
    frozen = json.loads((CELL / "panel.json").read_text())
    stamp = frozen["activation_quantizer_attestation"]
    assert stamp["activation_contract"] == CONTRACT
    assert "generated" not in stamp
    panel_path, digest, panel = _cell(tmp_path, stamp=stamp,
                                      executing_image=EXECUTED)
    with pytest.raises(ValueError, match="NOT VERIFIED"):
        consume_native_receipt(panel_path, expected_sha256=digest,
                               expected_panel=panel)


def test_a_tag_on_either_side_is_not_verified_rather_than_compared():
    """A tag moves, so there is nothing to compare; neither side may carry one."""
    stamp = _stamp()
    tagged = copy.deepcopy(stamp)
    tagged["generated"] = {**stamp["generated"], "image": "vllm/vllm-openai:v0.28.0"}
    panel = {"activation_quantizer_attestation": tagged}
    with pytest.raises(ValueError, match="not a digest reference"):
        _scope(panel, EXECUTED)
    with pytest.raises(ValueError, match="not a digest reference"):
        _scope({"activation_quantizer_attestation": stamp},
               "eugr/spark-vllm:latest")
    with pytest.raises(ValueError, match="not a digest reference"):
        _scope({"activation_quantizer_attestation": stamp}, None)


def test_a_dynamic_scale_stamp_attests_no_rule_so_there_is_no_scope():
    """The 42 fp8 cells of 2026-09-17, and every bf16 cell.

    Their stamp attests no rounding rule -- both sides compute the same
    function of ``x`` -- so scoping it to an image would be inventing a claim
    to check. The absence is written down rather than inferred.
    """
    from prismaquant.native_operator_panel import require_attested_activation_oracle

    stamp = require_attested_activation_oracle(
        {"quantizes_input": True, "quantizer": "f"}, platform=PLATFORM)
    assert stamp["status"] == "unattested_dynamic_scale"
    assert stamp["generated"] is None
    assert "derived from x" in stamp["generated_absent_because"]
    assert _scope({"activation_quantizer_attestation": stamp}, EXECUTED) is None
    assert _scope({"activation_quantizer_attestation": None}, EXECUTED) is None


def test_a_half_written_scope_is_refused_by_the_reader():
    """A block that names the box but not the build is worse than none."""
    payload = json.loads(TABLE.read_text())
    payload["activation_quantizers"]["platforms"][PLATFORM]["generated"].pop("vllm")
    with pytest.raises(trc.TesseraContractError, match="must publish exactly"):
        trc._parse_activation_quantizers(payload, "<fixture>")


def test_an_unknown_field_beside_the_platform_tables_is_a_review():
    payload = json.loads(TABLE.read_text())
    payload["activation_quantizers"]["platforms"][PLATFORM]["notes"] = "fine"
    with pytest.raises(trc.TesseraContractError, match="which this reader does not know"):
        trc._parse_activation_quantizers(payload, "<fixture>")


def test_a_table_with_no_generated_block_stamps_its_absence_with_a_reason():
    payload = json.loads(TABLE.read_text())
    payload["activation_quantizers"]["platforms"][PLATFORM].pop("generated")
    table = trc._parse_activation_quantizers(payload, "<fixture>")
    stamp = trc.require_activation_quantizer_attested(
        CONTRACT, platform=PLATFORM, table=table, contract_sha256=PIN_SHA256)
    assert stamp["generated"] is None
    assert "no image or build it was taken under" in stamp["generated_absent_because"]
    with pytest.raises(ValueError, match="NOT VERIFIED"):
        _scope({"activation_quantizer_attestation": stamp}, ATTESTED)
