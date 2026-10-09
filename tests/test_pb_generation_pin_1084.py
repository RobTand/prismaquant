"""PQ #1084: one pin per PrismaBuild bundle the tests run against.

Two pins once named the same published generation (81d95cba8d91) with
different file sets, so bumping one left the other suite on the old bundle.
The write-only produced-output suites now share
``pb_runtime_generation_pin.json``; this module refuses a second pin file
naming a bundle, generation or commit another pin already names, and checks
that every suite reads the shared pin.

PQ #2152/#2455: the shared pin and the consumer's reader pin
(``staged_lease.PB_READER_LEASE_PIN_COMMIT``) are one commit. The consumer
moved to SDK5 while this pin still named the SDK4 archive, and every connected
suite then refused its own transport by version.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from prismaquant import staged_lease

TESTS = Path(__file__).resolve().parent
SCHEMA = "prismaquant.stagea_produced_boundaries.pb_candidate_pin.v1"
SHARED = TESTS / "pb_runtime_generation_pin.json"


def _pb_pins():
    pins = {}
    for path in sorted(TESTS.rglob("*.json")):
        try:
            body = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(body, dict) and body.get("schema") == SCHEMA:
            pins[path] = body
    return pins


def test_no_two_pin_files_name_one_prismabuild_bundle():
    pins = _pb_pins()
    assert SHARED in pins
    for field in ("bundle_root", "candidate_commit", "runtime_generation"):
        owners = {}
        for path, body in pins.items():
            value = body.get(field)
            if value is None:
                continue
            assert value not in owners, (
                f"{path.name} and {owners[value].name} both name {field}={value}: "
                f"keep one pin and give it the union of the files both need (PQ #1084)")
            owners[value] = path


def test_the_shared_pin_names_the_reviewed_source_bundle():
    pin = json.loads(SHARED.read_text())
    generation = pin["runtime_generation"]
    assert Path(pin["bundle_root"]).name == generation
    assert generation == pin["candidate_commit"]
    assert (f"/qualification/pq-pb-sdk{staged_lease.PB_CLIENT_SDK_VERSION}-"
            in pin["bundle_root"])
    assert generation.startswith(pin["candidate_commit"][:12])
    assert {"path": pin["bundle_root"]} in pin["search_paths"]
    for consumer in pin["consumers"]:
        assert (TESTS.parent / consumer).is_file(), consumer


def test_the_shared_pin_and_the_consumer_pin_move_together():
    """A reader repin carries its connected-fixture bundle in the same change.

    The consumer refuses any other SDK by exact version, so a pin that names
    another commit cannot serve the suites that read it.
    """
    pin = json.loads(SHARED.read_text())
    assert pin["candidate_commit"] == staged_lease.PB_READER_LEASE_PIN_COMMIT
    assert pin["runtime_generation"] == staged_lease.PINNED_SDK_COMMIT


#: Each suite on the shared generation, and the name its pin path has there.
CONSUMER_PIN_NAMES = {
    "fullstack_pb_generation": "PIN_PATH",
    "test_stage_a_produced_boundary_chain": "PIN_PATH",
    "test_band_serial_handoff_produced": "ORIGIN_PIN",
    "test_band_serial_handoff_spool_real_pb": "ORIGIN_PIN",
    "test_produced_output_spool_real_pb": "PIN_PATH",
    "test_stage_a_retirement_pb_1073": "PIN_PATH",
    "test_stage_b_prep_produced_1070": "PIN_PATH",
}


def test_every_generation_suite_reads_the_shared_pin():
    import importlib

    if str(TESTS) not in sys.path:
        sys.path.insert(0, str(TESTS))
    pin = json.loads(SHARED.read_text())
    assert sorted(pin["consumers"]) == sorted(
        f"tests/{name}.py" for name in CONSUMER_PIN_NAMES)
    for name, attribute in CONSUMER_PIN_NAMES.items():
        module = importlib.import_module(name)
        assert Path(getattr(module, attribute)).resolve() == SHARED, name
