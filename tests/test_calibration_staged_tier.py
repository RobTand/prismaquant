"""Real calibration tensors, hashes and decoder parity through immutable inputs."""
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from prismaquant.residency_map import (
    StagedReadRefused, residency_resolver,
)
from prismaquant.staged_tier_policy import (
    TierPolicyRefused, activate_staged_tier_policy,
)
from staged_native_fixtures import bind, native_staged_state, publish_inputs


@pytest.fixture(autouse=True)
def _forget_state(native_staged_state):
    yield


def _artifact(tmp_path, *, n_samples=4, seqlen=8):
    ids = torch.arange(n_samples * seqlen).reshape(n_samples, seqlen)
    provenance = {"source": "fixture", "fit_tokens": ids.numel(),
                  "nsamples": n_samples, "seqlen": seqlen,
                  "fit_ids_sha256": hashlib.sha256(
                      ids.to(torch.int32).numpy().tobytes()).hexdigest()}
    path = tmp_path / "pool" / "tokens.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"calibration_ids": ids}, str(path),
              metadata={"calibration_provenance": json.dumps(provenance)})
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), ids


def _legacy_load(path, sha):
    from prismaquant.calibration_data import load_calibration_input
    return load_calibration_input(path, expected_sha256=sha, n_samples=4, seqlen=8)


def _guard_pool(monkeypatch, pool_path):
    """Record any pool open on the Python, native, or decoder route.

    ``os.open`` alone does not intercept ``Path.read_bytes``/``io.open``,
    so all three routes are logged; the staged path must need none of them
    for the pool file.
    """
    opened_os, opened_io, safe_opens = [], [], []
    real_open = os.open
    target = os.fspath(pool_path)

    def counting_os(given, *args, **kwargs):
        try:
            name = os.fspath(given)
        except TypeError:
            name = None
        if name == target:
            opened_os.append(name)
        return real_open(given, *args, **kwargs)

    monkeypatch.setattr(os, "open", counting_os)
    import io as _io

    real_io_open = _io.open

    def counting_io(file, *args, **kwargs):
        try:
            name = os.fspath(file)
        except TypeError:
            name = None
        if name == target:
            opened_io.append(name)
        return real_io_open(file, *args, **kwargs)

    monkeypatch.setattr(_io, "open", counting_io)
    import safetensors
    real_safe_open = safetensors.safe_open
    calls = []

    def guarded(*args, **kwargs):
        calls.append(args[0] if args else None)
        return real_safe_open(*args, **kwargs)

    monkeypatch.setattr(safetensors, "safe_open", guarded)
    return opened_os, opened_io, calls


def test_pool_guard_detects_deliberate_pool_reads(tmp_path, monkeypatch):
    """The guard itself is proven: a deliberate pool read is logged."""
    path, _sha, _ids = _artifact(tmp_path)
    opened_os, opened_io, _calls = _guard_pool(monkeypatch, path)
    assert path.read_bytes()
    fd = os.open(path, os.O_RDONLY)
    try:
        assert os.pread(fd, 8, 0)
    finally:
        os.close(fd)
    assert opened_io == [os.fspath(path)]
    assert opened_os == [os.fspath(path)]


def _stage_calibration(tmp_path, monkeypatch, path):
    binding = publish_inputs(tmp_path, [path])
    resolver = bind(monkeypatch, binding)
    activate_staged_tier_policy('ssd')
    return Path(resolver.manifest.extents[0].object_path), resolver


def _close_reader(resolver):
    assert resolver._windows == 0
    witnesses = [fd for fd, _signature in resolver._fds.values()]
    resolver.close()
    assert resolver._fds == {}
    for fd in witnesses:
        with pytest.raises(OSError):
            os.fstat(fd)


# -- RED: the pool-reading hole ----------------------------------------------

def test_strict_unmapped_calibration_refuses_without_pool_bytes(tmp_path, monkeypatch):
    """Active policy, calibration absent from the readset: refuse, no pool.

    Fails on the pool-reading main (the load succeeds off the pool path);
    passes once the reader honors the staged tiers.
    """
    from prismaquant.calibration_data import load_calibration_input

    path, sha, _ids = _artifact(tmp_path)
    other = tmp_path / "pool" / "unrelated.safetensors"
    other.write_bytes(b"unrelated")
    _stage_calibration(tmp_path, monkeypatch, other)
    opened_os, opened_io, safe_opens = _guard_pool(monkeypatch, path)
    with pytest.raises(TierPolicyRefused):
        load_calibration_input(path, expected_sha256=sha, n_samples=4, seqlen=8)
    assert opened_os == [] and opened_io == [] and safe_opens == []
    assert residency_resolver().report()['bytes_from_pool'] == 0


# -- positive: published stage serves pinned -----------------------------------

def test_strict_calibration_stage_serves_pinned_with_exact_release(tmp_path, monkeypatch):
    from prismaquant.calibration_data import load_calibration_input

    path, sha, ids = _artifact(tmp_path)
    want_ids, want_receipt = _legacy_load(path, sha)
    assert torch.equal(want_ids, ids)
    _staged, resolver = _stage_calibration(tmp_path, monkeypatch, path)
    opened_os, opened_io, safe_opens = _guard_pool(monkeypatch, path)
    got_ids, receipt = load_calibration_input(
        path, expected_sha256=sha, n_samples=4, seqlen=8)
    assert opened_os == [] and opened_io == [] and safe_opens == []
    assert torch.equal(got_ids, ids)
    assert receipt == want_receipt
    assert receipt["artifact_sha256"] == sha
    assert receipt["calibration_sha256"] == hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    assert receipt["provenance"]["fit_ids_sha256"] != receipt["calibration_sha256"]
    assert got_ids.is_contiguous()
    report = resolver.report()
    assert report['bytes_from_pool'] == 0
    assert report['bytes_from_stage'] == path.stat().st_size
    assert any(row.get('pin_id') for row in report['serving_tiers'])
    _close_reader(resolver)
    assert torch.equal(got_ids, ids), 'decoded tokens must survive owner closure'


def test_strict_calibration_uses_only_the_declared_object(tmp_path, monkeypatch):
    """An undeclared copy cannot choose the calibration tensor or its identity."""
    from prismaquant.calibration_data import load_calibration_input

    path, sha, ids = _artifact(tmp_path)
    raw = path.read_bytes()
    alternative = tmp_path / 'undeclared-copy'
    alternative.write_bytes(b'X' + raw[1:])
    _staged, resolver = _stage_calibration(tmp_path, monkeypatch, path)
    opened_os, opened_io, safe_opens = _guard_pool(monkeypatch, path)
    got_ids, receipt = load_calibration_input(
        path, expected_sha256=sha, n_samples=4, seqlen=8)
    assert opened_os == [] and opened_io == [] and safe_opens == []
    assert torch.equal(got_ids, ids)
    assert receipt["artifact_sha256"] == sha
    report = resolver.report()
    assert report['bytes_from_ram'] == 0
    assert report['bytes_from_stage'] == len(raw)
    assert report['bytes_from_pool'] == 0
    row = report['serving_tiers'][-1]
    assert row['tier'] == 'stage' and row['pin_id'] and row['range_ref']
    _close_reader(resolver)
    assert alternative.read_bytes() == b'X' + raw[1:]


def test_strict_calibration_never_falls_back_from_a_changed_declared_object(tmp_path, monkeypatch):
    from prismaquant.calibration_data import load_calibration_input

    path, sha, ids = _artifact(tmp_path)
    raw = path.read_bytes()
    alternative = tmp_path / 'valid-undeclared-copy'
    alternative.write_bytes(raw)
    staged, resolver = _stage_calibration(tmp_path, monkeypatch, path)
    staged.chmod(0o600)
    staged.write_bytes(b'X' + raw[1:])
    opened_os, opened_io, safe_opens = _guard_pool(monkeypatch, path)
    with pytest.raises(TierPolicyRefused):
        load_calibration_input(path, expected_sha256=sha, n_samples=4, seqlen=8)
    assert opened_os == [] and opened_io == [] and safe_opens == []
    report = resolver.report()
    assert report['bytes_from_pool'] == 0
    _close_reader(resolver)
    assert alternative.read_bytes() == raw
    # The ordinary decoder independently proves that the ignored copy is valid.
    from prismaquant.staged_tier_policy import deactivate_staged_tier_policy_for_tests
    deactivate_staged_tier_policy_for_tests()
    got, receipt = _legacy_load(alternative, sha)
    assert torch.equal(got, ids) and receipt['artifact_sha256'] == sha


# -- refusals: forbidden tier, corruption, wrong pin -----------------------------

def test_strict_calibration_ram_only_refuses_before_payload(tmp_path, monkeypatch):
    from prismaquant.calibration_data import load_calibration_input

    path, sha, _ids = _artifact(tmp_path)
    _staged, resolver = _stage_calibration(tmp_path, monkeypatch, path)
    from prismaquant import staged_tier_policy
    staged_tier_policy.deactivate_staged_tier_policy_for_tests()
    activate_staged_tier_policy('ram')
    opened_os, opened_io, safe_opens = _guard_pool(monkeypatch, path)
    with pytest.raises(TierPolicyRefused, match="ssd-not-allowed"):
        load_calibration_input(path, expected_sha256=sha, n_samples=4, seqlen=8)
    assert opened_os == [] and opened_io == [] and safe_opens == []
    assert resolver.report()['bytes_from_pool'] == 0


def test_strict_calibration_corrupt_stage_fails_clear(tmp_path, monkeypatch):
    from prismaquant.calibration_data import load_calibration_input

    path, sha, _ids = _artifact(tmp_path)
    staged, resolver = _stage_calibration(tmp_path, monkeypatch, path)
    blob = staged.read_bytes()
    staged.chmod(0o600)
    staged.write_bytes(blob[:-1] + bytes([blob[-1] ^ 0xFF]))
    opened_os, opened_io, safe_opens = _guard_pool(monkeypatch, path)
    with pytest.raises(TierPolicyRefused):
        load_calibration_input(path, expected_sha256=sha, n_samples=4, seqlen=8)
    assert opened_os == [] and opened_io == [] and safe_opens == []
    assert resolver.report()['bytes_from_pool'] == 0
    _close_reader(resolver)


def test_strict_calibration_wrong_pin_refuses_without_pool_bytes(tmp_path, monkeypatch):
    from prismaquant.calibration_data import load_calibration_input

    path, _sha, _ids = _artifact(tmp_path)
    _staged, resolver = _stage_calibration(tmp_path, monkeypatch, path)
    opened_os, opened_io, safe_opens = _guard_pool(monkeypatch, path)
    with pytest.raises(StagedReadRefused, match='source digest differs from requested identity'):
        load_calibration_input(path, expected_sha256="0" * 64, n_samples=4, seqlen=8)
    assert opened_os == [] and opened_io == [] and safe_opens == []
    assert resolver.report()['bytes_from_pool'] == 0


# -- malformed-format parity: canonical decoder is the oracle ------------------

def _mutated_artifact(tmp_path, *, name, mutate):
    """A real artifact with byte-level framing mutations, pinned by its own hash.

    ``mutate`` maps the full valid file bytes to malformed bytes; the pin
    covers the malformed bytes so the staged decoder is actually reached.
    """
    path, _sha, _ids = _artifact(tmp_path)
    raw = path.read_bytes()
    bad = mutate(raw)
    assert bad != raw
    target = tmp_path / "pool" / name
    target.write_bytes(bad)
    return target, hashlib.sha256(bad).hexdigest()


def _replace_once(blob: bytes, old: bytes, new: bytes) -> bytes:
    assert len(old) == len(new)
    assert blob.count(old) == 1, blob.count(old)
    return blob.replace(old, new)


def _gap_artifact(tmp_path):
    """Payload bytes the header skips: a leading gap the shape never covers.

    Eight framing bytes precede the tensor span while the shape still
    counts exactly the tensor's own bytes, so a span-only parser accepts
    the tensor while the canonical decoder rejects the noncontiguous
    framing.
    """
    def mutate(raw):
        size = int.from_bytes(raw[:8], "little")
        head, payload = raw[:8 + size], raw[8 + size:]
        assert b'"data_offsets":[0,256]' in head
        head = _replace_once(head, b'"data_offsets":[0,256]', b'"data_offsets":[8,264]')
        return head + b"\x00" * 8 + payload
    return _mutated_artifact(tmp_path, name="gap.safetensors", mutate=mutate)


def _trailing_artifact(tmp_path):
    """Valid tensor framing with bytes past the payload end."""
    def mutate(raw):
        return raw + b"\x00" * 16
    return _mutated_artifact(tmp_path, name="trailing.safetensors", mutate=mutate)


def _bad_row_artifact(tmp_path):
    """A tensor row the format's dtype table cannot name."""
    def mutate(raw):
        size = int.from_bytes(raw[:8], "little")
        header = raw[8:8 + size]
        assert b'"dtype":"I64"' in header
        return raw[:8] + _replace_once(header, b'"dtype":"I64"', b'"dtype":"XXX"') + raw[8 + size:]
    return _mutated_artifact(tmp_path, name="badrow.safetensors", mutate=mutate)


def _oracle_refuses(path, sha):
    """The canonical offline reader is the oracle; the staged path must match.

    Exception types are the decoders' own business and are not compared —
    only the refusal itself.
    """
    from prismaquant.calibration_data import load_calibration_input

    with pytest.raises(Exception):
        load_calibration_input(path, expected_sha256=sha, n_samples=4, seqlen=8)


@pytest.mark.parametrize("make", [_gap_artifact, _trailing_artifact, _bad_row_artifact])
def test_strict_malformed_calibration_refuses_parity(tmp_path, monkeypatch, make):
    from prismaquant.calibration_data import load_calibration_input

    path, sha = make(tmp_path)
    _oracle_refuses(path, sha)
    _staged, resolver = _stage_calibration(tmp_path, monkeypatch, path)
    opened_os, opened_io, safe_opens = _guard_pool(monkeypatch, path)
    with pytest.raises(Exception):
        load_calibration_input(path, expected_sha256=sha, n_samples=4, seqlen=8)
    assert opened_os == [] and opened_io == [] and safe_opens == []
    assert resolver.report()['bytes_from_pool'] == 0
    _close_reader(resolver)
