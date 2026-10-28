"""Observe source payload/hash access and candidate installation on tiny CPU tensors."""
import pytest
import torch

import g3_lib as library
import g3_offline_decoded_kl as evaluator
import g3_readset as readset
import g3_residency as residency

from .test_g3_readset import fixture


@pytest.mark.parametrize("arms", [
    ["a8_w"], ["a8_w", "a8_wa"], ["a8_w", "t8r_w"], ["a8_routed_only_w"],
])
def test_overwritten_source_is_neither_read_nor_hashed_before_candidate_forwards(
    tmp_path, monkeypatch, arms,
):
    qname, tensors, row = fixture(tmp_path)
    source_name = qname + ".weight"
    replacements = {
        b"A8!!": tensors[source_name].flip(0).contiguous(),
        b"T8R!": tensors[source_name].add(10),
    }
    row.update(
        a8_rendered_sha256=library.tensor_sha256(replacements[b"A8!!"]),
        t8r_rendered_shape=[2, 4],
        t8r_rendered_sha256=library.tensor_sha256(replacements[b"T8R!"]),
        t8r_wire={"shard": "candidate", "offset": 7, "length": 4},
    )
    (tmp_path / "candidate").write_bytes(b"abcA8!!T8R!")
    rows = {0: [row]}
    reads = readset.SourceReads(tmp_path, arms, rows)
    assert reads.omitted == {source_name}
    plan, _ = evaluator.plan_arms(arms, rows)

    def decode(blob, *, device):
        return replacements[blob].to(device)

    installer = evaluator.MultiInstaller(
        None, arms, rows, plan, {"a8": str(tmp_path), "t8r": str(tmp_path)}, decode, 1,
    )

    def forbidden_access(*args, **kwargs):
        pytest.fail("overwritten source payload or source hash accessed")

    try:
        with monkeypatch.context() as guard:
            guard.setattr(readset, "safe_open", forbidden_access)
            guard.setattr(residency, "staged_range", forbidden_access)
            guard.setattr(evaluator, "bytes_sha256hex", forbidden_access)
            with reads.open(str(tmp_path / "model.safetensors"), framework="pt") as handle:
                source = handle.get_tensor(source_name)
            source.fill_(123)
            assert installer.source_gate(0, [source])["source_hashes"] == 0
            assert installer.hash.submitted == 0
        inputs = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        for position, arm in enumerate(arms):
            installer.install(0, position, [source])
            replacement = replacements[b"T8R!" if evaluator.ARMS[arm][0] == "t8r" else b"A8!!"]
            assert torch.equal(source, replacement)
            assert torch.equal(inputs @ source.float().T, inputs @ replacement.float().T)
        assert installer.hash.submitted == sum(len(step["decode"]) for step in plan[0])
        assert installer.hash.all_matched
        assert reads.stats["source_bytes"] == 0
        assert reads.stats["omitted_bytes"] == tensors[source_name].numel() * 2
    finally:
        for reader in installer.readers.values():
            reader.close()
        installer.hash.close()


@pytest.mark.parametrize("mode", ["null", "SOURCE", "shared", "dense"])
@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("backend", ["staged", "safetensors"])
def test_consumed_original_retains_payload_read_and_own_digest_check(
    tmp_path, monkeypatch, mode, corrupt, backend,
):
    qname, tensors, row = fixture(tmp_path)
    source_name = qname + ".weight"
    arms = ["null", "a8_w"] if mode == "null" else ["a8_w"]
    if mode == "SOURCE":
        row["a8_format"] = "SOURCE"
    if mode in {"shared", "dense"}:
        row["kind"] = mode
        arms = ["a8_routed_only_w", "a8_w"]
    rows = {0: [row]}
    reads = readset.SourceReads(tmp_path, arms, rows)
    assert not reads.omitted
    payload_reads, hashes = [], []
    original_open = readset.safe_open
    original_hash = evaluator.bytes_sha256hex

    def staged_payload(path, offset, size):
        if backend != "staged":
            return None
        payload_reads.append((path, offset, size))
        with open(path, "rb") as handle:
            handle.seek(offset)
            return bytearray(handle.read(size))

    def observed_open(path, **kwargs):
        assert backend == "safetensors"
        payload_reads.append(path)
        return original_open(path, **kwargs)

    def observed_hash(raw):
        hashes.append(len(raw))
        return original_hash(raw)

    monkeypatch.setattr(residency, "staged_range", staged_payload)
    monkeypatch.setattr(readset, "safe_open", observed_open)
    monkeypatch.setattr(evaluator, "bytes_sha256hex", observed_hash)
    with reads.open(str(tmp_path / "model.safetensors"), framework="pt") as handle:
        source = handle.get_tensor(source_name)
    assert torch.equal(source, tensors[source_name])
    assert len(payload_reads) == 1
    assert reads.stats["source_bytes"] == tensors[source_name].numel() * 2
    if corrupt:
        source.add_(1)
    plan, _ = evaluator.plan_arms(arms, rows)
    installer = evaluator.MultiInstaller(None, arms, rows, plan, {"a8": "/"}, None, 1)
    try:
        if corrupt:
            with pytest.raises(library.HashGateError):
                installer.source_gate(0, [source])
        else:
            assert installer.source_gate(0, [source])["source_hashes"] == 1
        assert hashes == [tensors[source_name].numel() * 2]
        assert installer.hash.submitted == 1
    finally:
        installer.hash.close()
