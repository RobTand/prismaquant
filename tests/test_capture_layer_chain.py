"""Streamed calibration capture as retryable layer-chain quanta (PQ #1885)."""
import json
from pathlib import Path

import torch

from prismaquant import tessera_calibration_cache as cc
from test_tessera_calibration_cache import capture  # noqa: F401  (fixture)


def test_writer_finish_publishes_journal_completed_records(capture, tmp_path):  # noqa: F811
    """A later process publishes the units an earlier one journalled.

    Each chain quantum writes only its own layers' units, so the process that
    finishes the capture holds none of them in memory. ``finish`` must
    publish the journal's completed records, not only this process's.
    """
    _root, path, census, identity, acts, hessians, monolith = capture
    root = tmp_path/'chain'
    first = cc.CaptureWriter(root, census_path=path, identity=identity)
    first.write(acts={'a': acts['a']}, hessians={'a': hessians['a']},
                counts={'a': census['counts']['a']}, maxima={'a': census['max_abs']['a']})
    del first
    second = cc.CaptureWriter(root, census_path=path, identity=identity)
    second.write(acts={'b': acts['b']}, hessians={'b': hessians['b']},
                 counts={'b': census['counts']['b']}, maxima={'b': census['max_abs']['b']})
    record = second.finish(model_load_contract=identity['model_load_contract'])
    published = json.loads(Path(record['path']).read_text())
    assert set(published['entries']) == {'a', 'b'}
    assert published == json.loads(Path(monolith['path']).read_text())
    for name in ('a', 'b'):
        entry = torch.load(root/published['entries'][name]['path'], weights_only=True)
        assert torch.equal(entry['inputs'], acts[name])
        assert torch.equal(entry['hessian'], hessians[name])


# -- shared helpers ---------------------------------------------------------------

import importlib.metadata
import os

import pytest

from prismaquant import capture_layer_chain as chain
from prismaquant.capture_layer_chain import CaptureChainRefused
from prismaquant.cost_streaming import LAYER_MAJOR_BOUNDARY_STORAGE_SCHEMA
from prismaquant.streaming_model import (
    _SELECTED_INITIALIZATION_SCHEMA, _initialization_digest,
    merge_selected_initialization_witnesses, validate_streaming_selected_initialization_witness,
)


def _witness(layers, total, *, head_shape=(4,)):
    """A valid selected witness over a two-record head and one record per layer."""
    state = {"model.embed.weight": {"shape": list(head_shape), "dtype": "torch.bfloat16",
                                    "kind": "checkpoint"},
             "model.norm.weight": {"shape": [4], "dtype": "torch.bfloat16", "kind": "checkpoint"}}
    for layer in layers:
        state[f"model.layers.{layer}.w"] = {"shape": [4, 4], "dtype": "torch.bfloat16",
                                            "kind": "checkpoint"}
    return validate_streaming_selected_initialization_witness({
        "schema": _SELECTED_INITIALIZATION_SCHEMA, "scope": "streamed_text_source_selected",
        "status": "completed", "model_class": "fixture.LM",
        # The capture contract binds the witness to the running transformers.
        "transformers_version": importlib.metadata.version("transformers"),
        "dtype": "torch.bfloat16", "layers_prefix": "model.layers.", "total_model_layers": total,
        "observed_layers": sorted(layers), "head_state_names": sorted(
            ["model.embed.weight", "model.norm.weight"]),
        "state": state, "persistent_tensors": len(state), "derived_buffers": 0,
        "state_sha256": _initialization_digest(state), "source_map_sha256": "0" * 64})


def _boundary_policy(path):
    return {"schema": LAYER_MAJOR_BOUNDARY_STORAGE_SCHEMA, "capture_order": "layer_major",
            "directory": str(path), "max_resident_bytes": 1 << 20,
            "max_auxiliary_bytes": 1 << 20, "max_artifact_bytes": 1 << 24, "prefetch_batches": 1}


class _Chain:
    """A two-layer capture over the ``capture`` fixture's units: ``a`` in layer 0, ``b`` in 1."""

    UNITS = {(0, 1): ["a"], (1, 2): ["b"]}

    def __init__(self, tmp_path, *, dtype=torch.bfloat16, n_batches=2, root=None,
                 prepare=True):
        from prismaquant import tessera_calibration_cache as cache
        self.cache = cache
        self.source = tmp_path / "source"
        self.source.mkdir()
        (self.source / "config.json").write_text("{}")
        (self.source / "model.safetensors").write_bytes(b"capture chain source fixture")
        self.witnesses = {(0, 1): _witness([0], 2), (1, 2): _witness([1], 2)}
        contract = merge_selected_initialization_witnesses(self.witnesses.values())
        runtime = dict(torch=torch.__version__, cuda=torch.version.cuda,
                       transformers=importlib.metadata.version("transformers"))
        census = dict(model=str(self.source), counts={"a": 5, "b": 7},
                      max_abs={"a": 4.0, "b": 8.0}, unit_shapes={"a": [3, 2], "b": [3, 2]},
                      layer_stride=1, anchor_groups={"u:a": ["a"], "u:b": ["b"]},
                      model_load_contract=contract, attention_implementation="eager",
                      capture_runtime=runtime)
        self.census = tmp_path / "census.json"
        self.census.write_text(json.dumps(census))
        self.counts, self.maxima = census["counts"], census["max_abs"]
        self.acts = {"a": torch.tensor([[1., 2.], [3., 4.]]), "b": torch.ones(2, 2)}
        self.hessians = {"a": torch.eye(2) * 13, "b": torch.eye(2) * 25}
        self.contract = contract
        self.root = tmp_path / "capture" if root is None else Path(root)
        self.boundaries = tmp_path / "boundaries"
        self.mono = tmp_path / "monolith"
        self.n_batches = n_batches
        torch.manual_seed(1885)
        self.hidden = [torch.randn(1, 3, 4).to(dtype) for _ in range(n_batches)]
        self.received = {}
        if prepare:
            self.prepare()

    def prepare(self):
        # The prep seals the traversal identity and reads no payload (PQ #1896).
        with self.cache.record_capture_source(self.census, model=self.source) as source:
            self.prep = chain.prepare(self.root, census_path=self.census, ranges=[(0, 1), (1, 2)],
                n_batches=self.n_batches, boundary_storage=_boundary_policy(self.boundaries),
                identity=lambda: self.identity(source))["document"]
        return self.prep

    def identity(self, source_authentication=None):
        return self.cache.capture_identity(self.census, calibration={"fit_ids_sha256": "draw"},
            max_act_rows=2, model_load_contract=self.contract, attention_implementation="eager",
            source_authentication=source_authentication)

    def quantum(self, layers, *, fail=False):
        owner = chain.authenticate_quantum_source(self.root, census_path=self.census,
                                                  model=self.source)
        with owner:
            quantum = chain.ChainQuantum(self.root, layers, num_layers=2,
                                         source_authentication=owner)
            identity = self.identity(owner)
            identity = quantum.require_identity(identity, n_batches=self.n_batches)
            writer = self.cache.CaptureWriter(self.root, census_path=self.census, identity=identity)
            units = self.UNITS[layers]
            with quantum.owner():
                frontier = quantum.frontier()
                if frontier is not None:
                    self.received[layers] = list(frontier.hidden_batches())
                consume = quantum.boundary_consumer()
                for index, hidden in enumerate(self.hidden if consume is not None else ()):
                    consume(index, hidden)
                writer.write(acts={u: self.acts[u] for u in units},
                             hessians={u: self.hessians[u] for u in units},
                             counts={u: self.counts[u] for u in units},
                             maxima={u: self.maxima[u] for u in units})
                if fail:
                    raise RuntimeError("fixture quantum failure")
                return quantum.complete(witness=self.witnesses[layers],
                                        verified=writer.verify_entries(units))

    def monolith(self):
        return self.cache.publish_capture(self.mono, census_path=self.census,
            identity=self.identity(), acts=self.acts, hessians=self.hessians,
            counts=self.counts, maxima=self.maxima)

    def reseal(self, path, field, change):
        document = json.loads(Path(path).read_text())
        change(document)
        body = {key: value for key, value in document.items() if key != field}
        document[field] = chain.canonical_json_sha256(body, where="fixture")
        Path(path).write_text(json.dumps(document))

    def boundary_files(self):
        fragment = json.loads(chain.fragment_path(self.root, 0, 1).read_text())
        return [Path(record["path"]) for record in fragment["boundary"]]


# -- boundaries and the chained traversal -----------------------------------------

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_boundary_round_trip_is_bit_exact_and_keeps_its_dtype(tmp_path, dtype):
    """Boundary b written by one owner reads back in the next owner byte for byte."""
    fixture = _Chain(tmp_path, dtype=dtype, n_batches=3)
    fixture.quantum((0, 1))
    fixture.quantum((1, 2))
    received = fixture.received[(1, 2)]
    assert len(received) == 3
    for written, read in zip(fixture.hidden, received):
        assert read.dtype == dtype
        assert torch.equal(written.view(torch.uint8), read.view(torch.uint8))
    # The quantum that read boundary 1 left it in place; only the join retires it.
    assert all(path.is_file() for path in fixture.boundary_files())


def _dense_chain_runner(layers):
    from test_layer_major_boundary_capture import fixture as dense_fixture
    _model, context, runner, _cache = dense_fixture(layers=layers)
    return context, runner


class _ListFrontier:
    def __init__(self, layer, hidden):
        self.layer, self.hidden = layer, list(hidden)

    def hidden_batches(self):
        yield from self.hidden


def _traverse(runner, tokens, *, start=None, stop_layer=None):
    """Each visited layer's per-batch output, and the hidden after the last layer."""
    outputs, final = {}, []

    def visit(layer, forward_batch):
        seen = []
        handle = runner.model.model.layers[layer].register_forward_hook(
            lambda _module, _args, output: seen.append(output.detach().clone()))
        try:
            for ids in tokens:
                forward_batch(ids)
        finally:
            handle.remove()
        outputs[layer] = seen

    runner.visit_layer_batches(tokens, visit, start=start, stop_layer=stop_layer,
                               boundary_consumer=lambda index, hidden: final.append(hidden.clone()))
    return outputs, final


@pytest.mark.parametrize("ranges", [[(0, 1), (1, 3)], [(0, 1), (1, 2), (2, 3)]])
def test_chained_layer_visits_equal_the_monolith(ranges):
    """visit_layer_batches(start=a, stop_layer=b) chained is the monolith, batch for batch."""
    tokens = [torch.tensor([[1, 2, 3, 4]]), torch.tensor([[4, 3, 2, 1]]), torch.tensor([[2, 2, 1, 3]])]
    _context, runner = _dense_chain_runner(3)
    expected, expected_final = _traverse(runner, tokens)
    hidden = None
    for start, stop in ranges:
        context, runner = _dense_chain_runner(3)
        frontier = None if start == 0 else _ListFrontier(start, hidden)
        outputs, hidden = _traverse(runner, tokens, start=frontier, stop_layer=stop)
        assert sorted(outputs) == list(range(start, stop))
        for layer in range(start, stop):
            assert all(torch.equal(a, b) for a, b in zip(outputs[layer], expected[layer]))
        # Nothing before the frontier or at/after the stop is read or installed.
        touched = {layer for kind, layer in context.events if kind in ("prefetch", "install")}
        assert touched == set(range(start, stop))
    assert len(hidden) == len(expected_final)
    assert all(torch.equal(a, b) for a, b in zip(hidden, expected_final))


def test_layer_frontier_refusals():
    tokens = [torch.tensor([[1, 2, 3, 4]]), torch.tensor([[4, 3, 2, 1]])]
    _context, runner = _dense_chain_runner(3)
    hidden = [torch.zeros(1, 4, 16), torch.zeros(1, 4, 16)]
    visit = lambda layer, forward_batch: [forward_batch(ids) for ids in tokens]
    with pytest.raises(ValueError, match="exact storage"):
        runner.visit_layer_batches(tokens, visit, boundary_storage=object(), stop_layer=1)
    with pytest.raises(ValueError, match="0 <= start < stop"):
        runner.visit_layer_batches(tokens, visit, stop_layer=4)
    with pytest.raises(ValueError, match="0 <= start < stop"):
        runner.visit_layer_batches(tokens, visit, start=_ListFrontier(0, hidden))
    with pytest.raises(ValueError, match="0 <= start < stop"):
        runner.visit_layer_batches(tokens, visit, start=_ListFrontier(2, hidden), stop_layer=2)
    with pytest.raises(ValueError, match="tail logits"):
        runner.visit_layer_batches(tokens, visit, stop_layer=2, output_consumer=lambda *_: None)
    with pytest.raises(RuntimeError, match="fewer batches"):
        runner.visit_layer_batches(tokens, visit, start=_ListFrontier(1, hidden[:1]))
    with pytest.raises(RuntimeError, match="more batches"):
        runner.visit_layer_batches(tokens, visit, start=_ListFrontier(1, [*hidden, hidden[0]]))
    with pytest.raises(RuntimeError, match="stores torch.bfloat16"):
        runner.visit_layer_batches(tokens, visit, start=_ListFrontier(
            1, [value.to(torch.bfloat16) for value in hidden]))


# -- the selected witness merge --------------------------------------------------

def test_witness_merge_is_the_union_contract_and_refuses_disagreement():
    merged = merge_selected_initialization_witnesses([_witness([1, 2], 3), _witness([0], 3)])
    union = {**_witness([0], 3)["state"], **_witness([1, 2], 3)["state"]}
    assert merged["schema"] == "prismaquant.streaming_initialization.v1"
    assert merged["num_layers"] == 3 and merged["persistent_tensors"] == len(union)
    assert merged["state_sha256"] == _initialization_digest(union)
    with pytest.raises(ValueError, match="exactly once"):
        merge_selected_initialization_witnesses([_witness([0, 1], 3), _witness([1, 2], 3)])
    with pytest.raises(ValueError, match="exactly once"):
        merge_selected_initialization_witnesses([_witness([0], 3), _witness([2], 3)])
    with pytest.raises(ValueError, match="head record"):
        merge_selected_initialization_witnesses([_witness([0], 2), _witness([1], 2, head_shape=(5,))])


# -- prep, quanta and join -------------------------------------------------------

def test_join_publishes_the_monolith_manifest_without_reading_an_entry(tmp_path, monkeypatch):
    fixture = _Chain(tmp_path)
    fixture.quantum((0, 1))
    fixture.quantum((1, 2))
    boundary = fixture.boundary_files()
    assert boundary and all(path.is_file() for path in boundary)
    expected = json.loads(Path(fixture.monolith()["path"]).read_text())
    from prismaquant import tessera_calibration_cache as cache
    for name in ("_reverify_capture_entry", "_verified_capture_entry", "_validate_tensors"):
        monkeypatch.setattr(cache, name, lambda *_a, **_k: pytest.fail("the join read an entry"))
    record = chain.join(fixture.root, census_path=fixture.census)
    published = json.loads(Path(record["manifest"]["path"]).read_text())
    assert published == expected
    # Downstream rows bind it as they bind a monolithic capture.
    cache.require_capture_contract(record["manifest"]["path"], record["manifest"]["sha256"])
    for name, entry in published["entries"].items():
        actual = torch.load(fixture.root / entry["path"], weights_only=True)
        monolith = torch.load(fixture.mono / entry["path"], weights_only=True)
        assert torch.equal(actual["inputs"], monolith["inputs"])
        assert torch.equal(actual["hessian"], monolith["hessian"])
        assert actual["count"] == monolith["count"] and actual["max_abs"] == monolith["max_abs"]
    assert not any(path.exists() for path in boundary)
    assert record["retired_boundary_entries"] == len(boundary)
    # A retried join publishes the same manifest and retires nothing more.
    assert chain.join(fixture.root, census_path=fixture.census)["retired_boundary_entries"] == 0


def test_a_quantum_reads_its_source_through_the_prep_roster(tmp_path):
    fixture = _Chain(tmp_path)
    with pytest.raises(CaptureChainRefused, match="prep's recording owner"):
        chain.ChainQuantum(fixture.root, (0, 1), num_layers=2, source_authentication=None)
    fixture.quantum((0, 1))
    receipt = json.loads(chain.fragment_path(fixture.root, 0, 1).read_text())["source_authentication"]
    # The quantum's owner records what its reads hash (PQ #1896). The identity
    # reads nothing, and this fixture reads no payload, so nothing is hashed.
    assert receipt["schema"] == cc.RECORDING_RECEIPT_SCHEMA
    assert receipt["verified_files"] == []
    assert receipt["payload_bytes_hashed"] == 0
    assert receipt["binding_sha256"] == fixture.prep["prep_sha256"]


def test_quanta_run_in_order_and_once(tmp_path):
    fixture = _Chain(tmp_path)
    with pytest.raises(CaptureChainRefused, match="owner status None"):
        fixture.quantum((1, 2))
    with pytest.raises(CaptureChainRefused, match="not a range"):
        fixture.quantum((0, 2))
    fixture.quantum((0, 1))
    with pytest.raises(CaptureChainRefused, match="already complete"):
        fixture.quantum((0, 1))
    with pytest.raises(CaptureChainRefused, match="prepped once"):
        chain.prepare(fixture.root, census_path=fixture.census, ranges=[(0, 2)], n_batches=2,
                      boundary_storage=_boundary_policy(tmp_path / "other"),
                      identity=lambda: pytest.fail("a second prep hashed the source"))


def test_a_failed_quantum_retries_over_its_own_stale_boundary(tmp_path):
    fixture = _Chain(tmp_path)
    with pytest.raises(RuntimeError, match="fixture quantum failure"):
        fixture.quantum((0, 1), fail=True)
    status = json.loads(chain.owner_status_path(fixture.prep, 0, 1).read_text())
    assert status["status"] == "failed"
    fixture.quantum((0, 1))
    fixture.quantum((1, 2))
    assert json.loads(Path(chain.join(fixture.root, census_path=fixture.census)
                           ["manifest"]["path"]).read_text())["status"] == "complete"


@pytest.mark.parametrize("ranges,match", [([[0, 1], [0, 2]], "overlaps"),
                                          ([[0, 1], [2, 3]], "uncovered")])
def test_join_refuses_ranges_that_do_not_tile(tmp_path, ranges, match):
    fixture = _Chain(tmp_path)
    fixture.quantum((0, 1))
    fixture.quantum((1, 2))
    fixture.reseal(chain.prep_path(fixture.root), "prep_sha256",
                   lambda document: document.update(ranges=ranges))
    with pytest.raises(CaptureChainRefused, match=match):
        chain.join(fixture.root, census_path=fixture.census)


def test_join_refuses_a_missing_or_incomplete_quantum(tmp_path):
    fixture = _Chain(tmp_path)
    fixture.quantum((0, 1))
    with pytest.raises(CaptureChainRefused, match="1:2 have owner status None"):
        chain.join(fixture.root, census_path=fixture.census)
    with pytest.raises(RuntimeError, match="fixture quantum failure"):
        fixture.quantum((1, 2), fail=True)
    with pytest.raises(CaptureChainRefused, match="1:2 have owner status 'failed'"):
        chain.join(fixture.root, census_path=fixture.census)
    assert not (fixture.root / "capture_manifest.json").exists()
    assert all(path.is_file() for path in fixture.boundary_files())


def test_join_refuses_a_witness_that_disagrees_about_the_head(tmp_path):
    fixture = _Chain(tmp_path)
    fixture.quantum((0, 1))
    fixture.quantum((1, 2))

    def tamper(document):
        witness = document["witness"]
        witness["state"]["model.embed.weight"]["shape"] = [5]
        witness["state_sha256"] = _initialization_digest(witness["state"])
    fixture.reseal(chain.fragment_path(fixture.root, 1, 2), "fragment_sha256", tamper)
    with pytest.raises(ValueError, match="head record"):
        chain.join(fixture.root, census_path=fixture.census)
    assert not (fixture.root / "capture_manifest.json").exists()


def test_join_refuses_a_changed_source_fingerprint(tmp_path):
    fixture = _Chain(tmp_path)
    fixture.quantum((0, 1))
    fixture.quantum((1, 2))
    config = fixture.source / "config.json"
    observed = config.stat()
    os.utime(config, ns=(observed.st_atime_ns, observed.st_mtime_ns + 1_000_000_000))
    with pytest.raises(CaptureChainRefused, match="changed since the prep"):
        chain.join(fixture.root, census_path=fixture.census)
    assert not (fixture.root / "capture_manifest.json").exists()


def test_join_refuses_an_entry_changed_after_its_quantum_verified_it(tmp_path):
    fixture = _Chain(tmp_path)
    fixture.quantum((0, 1))
    fixture.quantum((1, 2))
    entry = fixture.root / json.loads(chain.fragment_path(fixture.root, 0, 1).read_text())[
        "units"]["a"]["path"]
    observed = entry.stat()
    os.utime(entry, ns=(observed.st_atime_ns, observed.st_mtime_ns + 1_000_000_000))
    with pytest.raises(RuntimeError, match="changed since its writer or quantum verified it"):
        chain.join(fixture.root, census_path=fixture.census)


# -- the CLI's refusals ----------------------------------------------------------

@pytest.mark.parametrize("extra,match", [
    (["--capture-chain", "quantum"], "every quantum names one"),
    (["--capture-chain", "prep", "--capture-layer-range", "0:1"], "every quantum names one"),
    (["--capture-chain", "prep", "--capture-chain-ranges", "0:1,1:2"], "the prep names both"),
    (["--capture-chain", "join", "--capture-chain-ranges", "0:1,1:2",
      "--capture-chain-boundary-storage", "{}"], "the prep names both"),
    (["--capture-chain", "prep", "--capture-chain-ranges", "0:2,1:3",
      "--capture-chain-boundary-storage", "{}"], "overlaps"),
    (["--capture-chain", "quantum", "--capture-layer-range", "2:1"], "0 <= A < B"),
    (["--capture-layer-range", "0:1"], "require --capture-chain"),
])
def test_capture_chain_arguments_refuse(tmp_path, capsys, extra, match):
    from prismaquant import tessera_campaign as campaign
    argv = ["--model", str(tmp_path / "model"), "--out", str(tmp_path / "out.pkl"),
            "--cache-dir", str(tmp_path / "cache"), "--streaming",
            "--capture-calibration-out", str(tmp_path / "capture"),
            "--calibration-census", str(tmp_path / "census.json"),
            "--attention-implementation", "eager", *extra]
    with pytest.raises(SystemExit):
        campaign.main(argv)
    assert match in capsys.readouterr().err


def test_runtime_drift_keeps_the_prep_identity_and_readable_capture(tmp_path, monkeypatch):
    fixture = _Chain(tmp_path)
    original_identity = fixture.identity

    def current_identity(owner=None):
        identity = original_identity(owner)
        identity["capture_runtime"] = {
            "torch": "2.11.0+cu130", "cuda": "13.0", "transformers": "5.16.1"}
        return identity

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    monkeypatch.setattr(fixture, "identity", current_identity)
    for witness in fixture.witnesses.values():
        witness["transformers_version"] = "5.16.1"
    fixture.quantum((0, 1))
    fixture.quantum((1, 2))
    record = chain.join(fixture.root, census_path=fixture.census)
    manifest = cc.require_capture_contract(record["manifest"]["path"],
                                           record["manifest"]["sha256"])
    assert manifest["identity"] == cc.bind_capture_source(
        fixture.prep["identity"], manifest["identity"]["source_files"])
    for name, entry in manifest["entries"].items():
        actual = torch.load(fixture.root / entry["path"], weights_only=True)
        torch.testing.assert_close(actual["inputs"], fixture.acts[name], rtol=0, atol=0)
        torch.testing.assert_close(actual["hessian"], fixture.hessians[name], rtol=0, atol=0)
        assert actual["count"] == fixture.counts[name]


@pytest.mark.parametrize("field,value", [
    ("calibration", {"fit_ids_sha256": "foreign-draw"}),
    ("units", {"a": [3, 4], "b": [3, 2]}),
    ("unit_scope", "selected"),
    ("max_act_rows", 1),
])
def test_runtime_stamp_does_not_admit_incomparable_capture(tmp_path, monkeypatch, field, value):
    fixture = _Chain(tmp_path)
    original_identity = fixture.identity

    def incomparable(owner=None):
        identity = original_identity(owner)
        identity["capture_runtime"]["cuda"] = "13.0"
        identity[field] = value
        return identity

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    monkeypatch.setattr(fixture, "identity", incomparable)
    with pytest.raises(CaptureChainRefused):
        fixture.quantum((0, 1))


def test_certified_capture_still_refuses_runtime_drift(tmp_path, monkeypatch):
    fixture = _Chain(tmp_path)
    original_identity = fixture.identity

    def changed_runtime(owner=None):
        identity = original_identity(owner)
        identity["capture_runtime"]["cuda"] = "13.0"
        return identity

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    monkeypatch.setattr(fixture, "identity", changed_runtime)
    with pytest.raises(CaptureChainRefused):
        fixture.quantum((0, 1))
