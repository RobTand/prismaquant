"""The SDK5 smoke must prove its contract without retained fleet evidence.

These observations delegate to the real installed SDK and consumer join. They
must not turn a synthetic carrier fixture into a claim of fleet qualification.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import os

import pytest

import test_held_producer_sdk5_smoke_2152 as smoke
from prismaquant import source_generation as sg


def test_sdk5_smoke_is_self_contained_and_preserves_the_join_controls(
        installed_client_sdk, tmp_path, monkeypatch, capsys):
    """Exercise the release selector itself, not a second substitute smoke."""
    real_read = installed_client_sdk.read_verified_action_result
    real_bind = installed_client_sdk.bind_standard_capture_command
    real_join = sg._require_original_reader_producer
    reads = []
    bindings = []
    joins = []

    def read(*args, **kwargs):
        queue = args[0] if args else kwargs["queue"]
        assert Path(queue.root).resolve().is_relative_to(tmp_path.resolve()), (
            "release-gate smoke must read a fixture-owned queue under tmp_path; "
            "retained fleet attempts and their CAS/log/helper files are not inputs")
        assert kwargs.get("require_native_producer_context") is True
        result = real_read(*args, **kwargs)
        reads.append(deepcopy(result))
        return result

    def bind(request):
        answer = real_bind(request)
        bindings.append(deepcopy(request))
        return answer

    def join(receipt, authority, selected, runtime):
        row = {"mode": os.environ.get("PRISMAQUANT_DEV_MODE"),
               "claim": deepcopy(receipt["deliveries"][0]["native_delivery"]["claim"]),
               "context": deepcopy(selected["producer_context"]), "error": None}
        joins.append(row)
        try:
            return real_join(receipt, authority, selected, runtime)
        except RuntimeError as exc:
            row["error"] = str(exc)
            raise

    monkeypatch.setattr(installed_client_sdk, "read_verified_action_result", read)
    monkeypatch.setattr(installed_client_sdk, "bind_standard_capture_command", bind)
    monkeypatch.setattr(sg, "_require_original_reader_producer", join)
    try:
        smoke.test_held_producer_context_authenticates_and_joins(
            installed_client_sdk, tmp_path, monkeypatch, capsys)
    except pytest.skip.Exception as exc:
        pytest.fail(f"installed-SDK CPU smoke must run, not skip: {exc}")

    assert reads, "smoke must authenticate evidence through the real public SDK"
    result = reads[0]
    context = result["producer_context"]
    assert context["schema"] == "prismabuild.native_producer_context.v1"
    assert context["attempt_source"] == "selected-immutable-attempt"
    assert context["resources_semantics"] == "selected-claim-sealed-demand"
    for field in ("action_key", "attempt", "published_unix", "generation", "host"):
        assert context[field] == result[field]
    assert context["worker"] == context["incarnation"] == result["worker_id"]
    assert context["receipt_sha256"] == result["receipt"]["receipt_sha256"]
    assert Path(context["queue_root"]).resolve().is_relative_to(tmp_path.resolve())
    assert Path(context["helper_root"]).resolve().is_relative_to(tmp_path.resolve())
    assert result["payload"]
    assert result["request"] in bindings, "verify the sealed capture wrapper too"

    def same_identity(row):
        return all(row["claim"][field] == row["context"][field] for field in (
            "queue_root", "action_key", "nonce", "scope_id", "worker", "host",
            "incarnation", "helper_root"))

    assert any(row["mode"] == "0" and same_identity(row)
               and "selected reader actual complete helper tree" in (row["error"] or "")
               for row in joins), "certified control must reach the helper-tree seal"
    for field in ("nonce", "scope_id"):
        assert any(row["mode"] == "0"
                   and row["claim"][field] != row["context"][field]
                   and "actual reader delivery selected producer" in (row["error"] or "")
                   for row in joins), f"certified foreign {field} must still refuse"
    assert any(row["mode"] == "1" and same_identity(row) and row["error"] is None
               for row in joins), "matching dev-mode identity must join"
    assert any(row["mode"] == "1"
               and row["claim"]["nonce"] != row["context"]["nonce"]
               and row["error"] is None for row in joins), (
                   "D32 dev-mode foreign identity remains a stamp, not a refusal")
