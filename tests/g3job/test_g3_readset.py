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
    fleet_src = Path("/mnt/shared/prismabuild-fleet/repo/src")
    if not fleet_src.is_dir():
        pytest.skip("PrismaBuild runtime source is not mounted on this box")
    sys.path.insert(0, str(fleet_src))
    from prismabuild.core import validate_data_manifest
    assert validate_data_manifest(manifest)["read_plan"] == manifest["read_plan"]


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
    import ast
    source = (G.PQ / "experiments/glm_tr3_full_vocab.py").read_text()
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "token_kl")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "glm_tr3_full_vocab.py", "exec"), namespace)
    token_kl = namespace["token_kl"]  # exactly the unchanged pinned KL, no second recipe
    assert torch.equal(outputs[0].view(torch.uint8), outputs[1].view(torch.uint8))
    assert torch.equal(token_kl(teacher, outputs[0], require_cuda=False).view(torch.uint8),
                       token_kl(teacher, outputs[1], require_cuda=False).view(torch.uint8))
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
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_HELPER", "unit-progress-helper")
    monkeypatch.setattr(progress.runpy, "run_path", lambda _: {
        "commit": lambda units, **fields: commits.append((units, fields)) or True})
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
    monkeypatch.delenv("PRISMABUILD_ACTION_PROGRESS_HELPER", raising=False)
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


def test_docker_contract_translates_every_input_and_mounts_live_helper(monkeypatch):
    from g3_residency import container_contract, resolve_g3_origin
    names = {"PRISMABUILD_RESIDENCY_MAP": "/queue/residency/key.json",
             "PRISMABUILD_READER_HELPER_ROOT": "/helper/exact-generation",
             "PRISMABUILD_QUEUE_ROOT": "/queue", "PRISMABUILD_ACTION_KEY": "a" * 64,
             "PRISMABUILD_ACTION_NONCE": "nonce", "PRISMABUILD_ACTION_SCOPE": "scope"}
    for name, value in names.items():
        monkeypatch.setenv(name, value)
    mounts, environment = container_contract()
    assert environment == {**names, "PRISMABUILD_RESIDENCY_MAP": "/g3-pb-residency-map/key.json"}
    assert next(m for m in mounts if m["source"] == "/queue")["readonly"] is False
    assert next(m for m in mounts if m["source"] == "/helper/exact-generation")["readonly"] is True
    assert next(m for m in mounts if m["source"] == "/queue/residency")["target"] == "/g3-pb-residency-map"
    assert not any(m["readonly"] and Path(m["target"]).is_relative_to("/queue") for m in mounts)
    assert resolve_g3_origin("/teacher1/w.npy", {"/teacher1": "/mnt/shared/teacher"}) == "/mnt/shared/teacher/w.npy"
    assert resolve_g3_origin("/sourcex/w", {"/source": "/mnt/shared/model"}) == "/sourcex/w"


@pytest.mark.parametrize("corrupt", [False, True])
def test_named_range_pin_outlives_descriptor_and_integrity_never_falls_back(tmp_path, monkeypatch, corrupt):
    import os
    import threading
    from types import SimpleNamespace
    import g3_residency as residency
    stage = tmp_path / "range"
    stage.write_bytes(b"staged")
    events = []
    opened = []
    def acquire(*args, **kwargs):
        events.append("acquire")
        return {"ok": True, "pin": {}, "pin_id": "pin", "ref_id": "ref"}
    def pinned(*args, **kwargs):
        events.append("open")
        fd = os.open(stage, os.O_RDONLY)
        opened.append(fd)
        return fd, {"tier_id": "stage"}
    def release(*args, **kwargs):
        with pytest.raises(OSError):
            os.fstat(opened[0])
        events.append("release-after-close")
        return True
    digest = "0" * 64 if corrupt else hashlib.sha256(b"staged").hexdigest()
    reader = residency.StagedReader.__new__(residency.StagedReader)
    reader.maps = SimpleNamespace(residency_map_key=lambda p, o: f"{o}:{p}",
        read_map=lambda _: {"tier_id": "stage", "manifest_sha256": "a" * 64,
                            "entries": {"7:/host/file": {"bytes": 6, "sha256": digest}}})
    reader.lease = SimpleNamespace(covers_for_keys=lambda *a, **k: {"ok": True, "covers": []},
                                  acquire_for=acquire, open_pinned=pinned, release=release)
    reader.ctx = {"map_path": "map", "action_key": "a" * 64}
    reader.queue, reader.root, reader.lock = None, tmp_path, threading.Lock()
    reader.stats = {"staged_bytes": 0, "staged_reads": 0, "read_s": 0.0, "tiers": {}}
    if corrupt:
        with pytest.raises(RuntimeError, match="own digest"):
            reader.read("/host/file", 7, 6)
    else:
        assert reader.read("/host/file", 7, 6) == b"staged"
    assert events == ["acquire", "open", "release-after-close"]
    assert reader.read("/not-in-map", 0, 3) is None


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
