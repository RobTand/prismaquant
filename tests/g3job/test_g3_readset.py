"""Tiny own-byte/read-set regressions; no full model or confirming-arm replay."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "g3job"))
import g3_lib as L
import g3_offline_decoded_kl as G


def fixture(tmp_path):
    q = 'model.language_model.layers.0.mlp.experts.0.gate_proj'
    tensors = {q + '.weight': torch.arange(8).reshape(2, 4).to(torch.bfloat16),
               'model.language_model.layers.0.self_attn.q_proj.weight': torch.ones(4, 4, dtype=torch.bfloat16)}
    save_file(tensors, str(tmp_path / 'model.safetensors'))
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {k: 'model.safetensors' for k in tensors}}))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "glm4_moe"}))
    row = {'qname': q, 'kind': 'routed', 'role': 'gate_proj', 'expert': 0,
           'source_sha256': L.tensor_sha256(tensors[q + '.weight']),
           'a8_rendered_shape': [2, 4],
           'a8_rendered_sha256': 'a' * 64, 'a8_wire': {'shard': 'candidate', 'offset': 3, 'length': 4}}
    return q, tensors, row


def test_omission_uses_requested_arm_plan_and_preserves_unassigned_source(tmp_path):
    from g3_readset import SourceReads
    q, tensors, row = fixture(tmp_path)
    reads = SourceReads(tmp_path, ['a8_w'], {0: [row]})
    assert reads.omitted == {q + '.weight'}
    with reads.open(str(tmp_path / 'model.safetensors'), framework='pt') as handle:
        assert torch.equal(handle.get_tensor(next(k for k in tensors if k != q + '.weight')), tensors[next(k for k in tensors if k != q + '.weight')])
        assert handle.get_tensor(q + '.weight').shape == tensors[q + '.weight'].shape
    assert reads.stats['omitted_bytes'] == 16 and reads.stats['source_bytes'] == 32
    assert not SourceReads(tmp_path, ['null', 'a8_w'], {0: [row]}).omitted
    source = dict(row, a8_format='SOURCE')
    assert not SourceReads(tmp_path, ['a8_w'], {0: [source]}).omitted


def test_source_gate_does_not_hash_overwritten_rows(tmp_path):
    q, tensors, row = fixture(tmp_path)
    plan, _ = G.plan_arms(['a8_w'], {0: [row]})
    inst = G.MultiInstaller(None, ['a8_w'], {0: [row]}, plan, {'a8': '/'}, None, 1)
    got = inst.source_gate(0, [torch.full_like(tensors[q + '.weight'], 123)])
    assert got['source_hashes'] == 0
    assert inst.counts['a8_w']['source_verified'] == 0


def test_hash_batches_preserve_bits_and_charge_before_allocation():
    tensors = [torch.arange(16, dtype=torch.int16).reshape(4, 4) + i for i in range(9)]
    pool = G.HashPool(max_inflight_bytes=256, batch_bytes=64)
    for i, t in enumerate(tensors):
        pool.submit(t, L.tensor_sha256(t), str(i))
    assert pool.drain() == len(tensors)
    assert pool.all_matched
    assert pool.profile['copies'] == 5
    assert pool.profile['peak_inflight_bytes'] <= 256
    assert pool.profile['peak_batch_bytes'] <= 64
    assert pool.profile["max_pack_storage_bytes"] <= 64
    assert pool.profile["max_host_storage_bytes"] <= 64
    print("MEASURED hash-storage", json.dumps(pool.profile))
    with pytest.raises(L.HashGateError):
        pool.submit(tensors[0], '0' * 64, 'corrupt')
        pool.drain()
    assert not pool.all_matched
    pool.close()
    nonstrict = G.HashPool(max_inflight_bytes=256, strict=False, batch_bytes=64)
    nonstrict.submit(tensors[0], "0" * 64, "recorded mismatch")
    assert nonstrict.drain() == 1 and not nonstrict.all_matched
    nonstrict.close()


def test_manifest_omits_unused_source_and_tracks_repeated_teacher_reads(tmp_path):
    from g3_readset import SourceReads, build_manifest
    q, tensors, row = fixture(tmp_path)
    (tmp_path / 'candidate').write_bytes(b'abcDEFG')
    teacher = tmp_path / 'teacher'
    teacher.mkdir()
    (teacher / 'teacher.json').write_text('{}')
    (teacher / 'w.npy').write_bytes(b'array')
    specs = [(teacher, {'windows': [{'window_id': 'w', 'path': 'w.npy', 'bytes': 5, 'sha256': hashlib.sha256(b'array').hexdigest()}]})]
    reads = SourceReads(tmp_path, ['a8_w', 'a8_wa'], {0: [row]})
    manifest = build_manifest(reads, ['a8_w', 'a8_wa'], {0: [row]}, {'a8': str(tmp_path)}, specs, ['w'], mount_prefix=str(tmp_path))
    layers = manifest['read_plan']['phases']
    assert [p['name'] for p in layers] == ['setup', 'layer-00', 'teachers']
    source = reads.entries['model.safetensors'][q + '.weight']
    assert not any(e['path'].endswith('model.safetensors') and e['offset'] == source['offset'] for e in manifest['entries'])
    assert layers[-1]['bytes'] == 5
    assert manifest['annotations']['teacher_repeats'] == 2
    assert sum(e['bytes'] for e in manifest['entries']) == manifest['total_bytes']
    assert list(dict.fromkeys(i for p in layers for i in p['entry_indices'])) == list(range(manifest['entry_count']))
    # The staged read contract refuses repeated or overlapping ranges of one source; check the same
    # independent invariant on this plan's own entries.
    ranges = {}
    for entry in manifest['entries']:
        ranges.setdefault(entry['path'], []).append((entry['offset'], entry['offset'] + entry['bytes']))
    for spans in ranges.values():
        ordered = sorted(spans)
        assert all(start >= previous_end for (_, previous_end), (start, _) in zip(ordered, ordered[1:]))


def test_one_arm_omission_is_bitwise_equal_in_logits_and_fp64_kl(tmp_path):
    from g3_readset import SourceReads
    q, tensors, row = fixture(tmp_path)
    replacement = torch.arange(8, dtype=torch.float32).reshape(2, 4).flip(0).to(torch.bfloat16)
    row['a8_rendered_sha256'] = L.tensor_sha256(replacement)
    outputs = []
    for arms in (['null', 'a8_w'], ['a8_w']):
        reads = SourceReads(tmp_path, arms, {0: [row]})
        with reads.open(str(tmp_path / 'model.safetensors'), framework='pt') as handle:
            weight = handle.get_tensor(q + '.weight')
        if not reads.omitted:
            L.check_identity(weight, row['source_sha256'], q)
        L.check_identity(replacement, row['a8_rendered_sha256'], q)
        weight.copy_(replacement)
        outputs.append(torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).float() @ weight.float().T)
    teacher = torch.tensor([[0.3, -0.1], [0.9, 0.4], [-0.5, 0.7]], dtype=torch.float32)
    # Exercise the actual compatibility entry point, including its imports and
    # delegation. Extracting only its AST body loses module import bindings.
    from experiments.glm_tr3_full_vocab import token_kl
    import runpy
    old_token_kl = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                    "fixtures" / "token_kl_old_2778.py"))["token_kl"]
    assert torch.equal(outputs[0].view(torch.uint8), outputs[1].view(torch.uint8))
    scores = [token_kl(teacher, output, require_cuda=False) for output in outputs]
    assert torch.equal(scores[0].view(torch.uint8), scores[1].view(torch.uint8))
    for output, score in zip(outputs, scores):
        assert score.dtype == torch.float64
        assert torch.equal(score.view(torch.uint8),
                           old_token_kl(teacher, output, require_cuda=False).view(torch.uint8))
    print('BITWISE one-arm tiny BF16 forward + FP64 KL old-source-read/new-omission: equal')


def test_progress_moves_only_after_completed_layers():
    from g3_progress_launch import ReadProgress
    progress = ReadProgress(["a8_w"], num_layers=2)
    assert progress.observe("[g3 0.1s] manifest x")["phase"] == "setup"
    assert progress.observe("[g3 0.2s] runner ready {}")["phase"] == "layer-00"
    assert progress.observe("[g3 0.3s] layer 0: src 0")["phase"] == "layer-01"
    assert progress.observe("[g3 0.4s] layer 1: src 0")["phase"] == "teachers"
    assert progress.observe("[g3 0.5s] a8_w window final-0: KL 0")["phase"] == "teachers"
    with pytest.raises(ValueError, match="sequence"):
        progress.observe("[g3 0.6s] layer 1: src 0")



def test_progress_accepts_the_single_arm_window_log_and_completes_its_population(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import g3_progress_launch as progress
    lines = ["[g3 0.1s] manifest x", "[g3 0.2s] runner ready {}"]
    lines.extend(f"[g3 0.3s] layer {layer}: sub 0" for layer in range(45))
    lines.extend(f"[g3 0.4s] window final-{w}: KL t04 0.000001 t2 0.000002 maxdiff04 0.001"
                 for w in range(25))
    commits = []
    monkeypatch.setattr(progress, "log_progress", lambda units, phase, unit: commits.append((units, phase, unit)))
    monkeypatch.setattr(progress.subprocess, "Popen", lambda *a, **k:
        SimpleNamespace(stdout=[line + "\n" for line in lines], wait=lambda: 0))
    monkeypatch.setattr(progress.sys, "argv", ["g3_progress_launch.py", "--arm", "a8_w",
                                           "--output-root", str(tmp_path / "out")])
    assert progress.main() == 0
    ledger = [json.loads(line) for line in (tmp_path / "out/semantic-progress.jsonl").read_text().splitlines()]
    assert len(ledger) == len(commits) == 72
    assert sum(event["unit"] == "scored_window" for event in ledger) == 25
    assert ledger[-1]["phase"] == "teachers"


def test_progress_refuses_pilot_explicitly(monkeypatch):
    import g3_progress_launch as progress
    monkeypatch.setattr(progress.sys, "argv", ["g3_progress_launch.py", "--pilot"])
    with pytest.raises(SystemExit, match="pilot"):
        progress.main()
    assert progress.select_arms(["--arm", "a8_w"]) == ["a8_w"]
    assert progress.select_arms(["--arms", "null,a8_w"]) == ["null", "a8_w"]
    assert progress.select_arms(["--reader-smoke", "{}"]) == []

def test_d32_uses_stored_metadata_without_stat_or_proof_owner(tmp_path, monkeypatch):
    import g3_prepared_source as prepared
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    proof = tmp_path / "stored.json"
    identity = {"shards": [{"name": "absent.safetensors", "sha256": "a" * 64}]}
    proof.write_text(json.dumps({"identity": identity, "identity_sha256": "b" * 64,
                                  "cache_owner": {"path": "/absent/owner.py"},
                                  "logical_source_root": "/old", "source_bindings": []}))
    loaded, receipt = prepared.validate_prepared_source("/absent/model", proof, prepared.sha256_file(proof))
    assert loaded == identity and receipt["dev_uncertified"]
    assert receipt["source_body_bytes_read"] == 0
    with pytest.raises(ValueError, match="bytes changed"):
        prepared.validate_prepared_source("/absent/model", proof, "0" * 64)


def test_container_contract_mounts_only_readonly_manifest_dirs_and_translates_origins(tmp_path, monkeypatch):
    from g3_residency import container_contract, resolve_g3_origin
    from staged_native_fixtures import bind, publish_inputs
    source = tmp_path / "source"
    source.write_bytes(b"source")
    binding = publish_inputs(tmp_path, [source])
    bind(monkeypatch, binding)
    mounts, environment = container_contract()
    assert environment == {}
    assert mounts == [{"source": str(tmp_path), "target": str(tmp_path), "readonly": True}]
    assert not any("PRISMABUILD" in json.dumps(m) for m in mounts)
    assert resolve_g3_origin("/teacher1/w.npy", {"/teacher1": "/mnt/shared/teacher"}) == "/mnt/shared/teacher/w.npy"
    monkeypatch.delenv("PRISMAQUANT_STAGED_MANIFEST")
    assert container_contract() == ([], {})


def test_a_host_local_manifest_cannot_be_launched_into_the_container(tmp_path, monkeypatch):
    from g3_residency import container_contract
    from staged_native_fixtures import bind, fixture
    from prismaquant.staged_manifest import publish_staged_json
    _source, data, binding, _contract = fixture(tmp_path)
    local = {**data, "lifetime": {"mode": "host_local_cache", "root": str(tmp_path / "cache"),
                                  "owner_host": "dl380g10"}}
    path = tmp_path / "host-local.json"
    binding = {**binding, "path": str(path), "sha256": publish_staged_json(path, local)}
    bind(monkeypatch, binding)
    with pytest.raises(RuntimeError, match="host_local_cache.*cannot cross the container boundary"):
        container_contract()


@pytest.mark.parametrize("corrupt", [False, True])
def test_named_range_pin_outlives_descriptor_and_integrity_never_falls_back(tmp_path, monkeypatch, corrupt):
    import os
    import g3_residency as residency
    from g3_pq_policy import residency_map as maps
    from staged_native_fixtures import bind, publish_inputs
    origin = tmp_path / "file"
    origin.write_bytes(b"0123456staged789")
    binding = publish_inputs(tmp_path, [origin], ranges={origin: [(7, 13)]})
    bind(monkeypatch, binding)
    monkeypatch.delenv("G3_READ_PLAN", raising=False)
    maps.reset_residency_resolver_for_tests()
    reader = residency.StagedReader()
    try:
        if corrupt:
            target = next(e.object_path for e in reader.resolver.manifest.extents)
            os.chmod(target, 0o644)
            with open(target, "r+b") as stream:
                stream.write(b"X")
            with pytest.raises(RuntimeError, match="own digest"):
                reader.read(str(origin), 7, 6)
            assert origin.read_bytes() == b"0123456staged789"  # origin untouched and never selected
        else:
            assert reader.read(str(origin), 7, 6) == b"staged"
        window, = reader.windows["smoke"]
        assert window["lease"]._fds == {} and reader.resolver._windows == 1  # the pin outlives the descriptor
        assert reader.read("/not-in-manifest", 0, 3) is None
    finally:
        reader.close()
        assert reader.resolver._windows == 0
        maps.reset_residency_resolver_for_tests()


def test_d32_does_not_suspend_two_data_digest_comparability(tmp_path, monkeypatch):
    import g3_prepared_source as prepared
    monkeypatch.delenv("PRISMAQUANT_DEV_MODE", raising=False)
    path = tmp_path / "binding.json"
    path.write_text(json.dumps({"source_files": [{"name": "x.safetensors",
                    "capture_sha256": "a" * 64, "upstream_sha256": "b" * 64}]}))
    with pytest.raises(ValueError, match="filename or digest mismatch"):
        G.source_identity_check({"source_model_identity_sha256": "a" * 64}, "a" * 64,
                                path, prepared.sha256_file(path), {}, tmp_path)

    for payload in ({}, {"source_files": []}):
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="upstream source roster"):
            G.source_identity_check({"source_model_identity_sha256": "a" * 64}, "a" * 64,
                                    path, prepared.sha256_file(path), {}, tmp_path)
