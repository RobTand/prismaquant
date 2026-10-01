"""An executable head declares whole checkpoint config/index metadata."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
from typing import Any

import pytest

from prismaquant import joint_layer_quanta as jl
from prismaquant.layer_streaming import streaming_source_plan
from test_quantum_executable_readset import _bound_inputs
from test_stage_b_fp8_source_strict_reads_1219 import (
    FP8_CONFIG, PREFIX, SPILL, _fp8_layout, _unbind_staged_reads,
)
from test_stage_b_prep_io_1070 import generator_args
from test_stageb_readset_source_coverage import (
    _build, _phase_entries, _source_model, _with_source_paths,
)


def _plan(tmp_path):
    model, _ = _source_model(tmp_path)
    (model / "config.json").write_text("{}")
    return model, streaming_source_plan(
        str(model), layers_prefix=PREFIX, layers=range(4))


def _metadata_spans(model):
    return [(str(path), 0, path.stat().st_size) for path in (
        model / "config.json", model / "model.safetensors.index.json")]


def _head(plan):
    factory = getattr(jl, "head_source_from_streaming_plan", None)
    assert callable(factory), "streaming head projection is not shared"
    head = factory(plan)
    assert isinstance(head, dict)
    return head


def test_head_projection_preserves_tensor_roster_and_covers_metadata(tmp_path):
    model, plan = _plan(tmp_path)
    before = copy.deepcopy(plan)
    head = _head(plan)
    assert head["tensors"] == plan["head_tensors"]
    assert head["layers_prefix"] == plan["layers_prefix"]
    entries = [{"path": path, "offset": begin, "bytes": end - begin}
               for path, begin, end in head["spans"]]
    assert not jl.uncovered_source_spans(entries, _metadata_spans(model))
    assert not jl.uncovered_source_spans(entries, plan["head_spans"])
    assert plan == before
    assert _head(plan) == head


def test_real_regenerator_declares_config_and_index_in_each_executable_head(tmp_path):
    import regenerate_joint_quanta as regen

    pool, campaign, receipt, model, _preparation, _argv = _fp8_layout(
        tmp_path, FP8_CONFIG)
    metadata = pool / "executable-metadata"
    assert regen.main(generator_args(pool, campaign, receipt, metadata)
                      + list(SPILL)) == 0
    manifests = sorted((metadata / "adjoint" / "bound-readsets").glob(
        "*.executable.json.gz"))
    assert manifests, "the generator emitted no executable readsets"
    wanted = _metadata_spans(model)
    for path in manifests:
        manifest = json.loads(gzip.decompress(path.read_bytes()))
        missing = jl.uncovered_source_spans(_phase_entries(manifest, "head"), wanted)
        assert not missing, f"{path.name}: undeclared bootstrap metadata {missing}"


def test_binder_refuses_rehashed_manifest_missing_checkpoint_metadata(tmp_path):
    kwargs: dict[str, Any]
    record, receipt, parent, kwargs = _bound_inputs(tmp_path)
    model, plan = _plan(tmp_path)
    parent = _with_source_paths(parent, model)
    head = _head(plan)
    full = _build(record, receipt, parent, head_source=head)
    before = copy.deepcopy(record)
    kwargs.update(manifest=full, head_source=head,
                  manifest_sha256=hashlib.sha256(jl.seal_manifest_bytes(full)).hexdigest())
    bound = jl.bind_quantum_executable(record, receipt, parent, **kwargs)
    assert record == before
    assert bound["identity_sha256"] != record["identity_sha256"]
    # This is a valid old tensor-only head, not a malformed/hash-invalid file.
    tensor_only = {**head, "spans": plan["head_spans"]}
    incomplete = _build(record, receipt, parent, head_source=tensor_only)
    kwargs.update(manifest=incomplete,
                  manifest_sha256=hashlib.sha256(jl.seal_manifest_bytes(incomplete)).hexdigest())
    with pytest.raises(ValueError, match="do not originate"):
        jl.bind_quantum_executable(record, receipt, parent, **kwargs)


@pytest.mark.parametrize("bad", [
    None, [], [["too-short"]],
    [["/config.json", 1, 2]], [["/config.json", False, 2]],
    [["/config.json", 0, True]], [["/config.json", 0, 0]],
    [["relative-config.json", 0, 2]], [[[], 0, 2]],
])
def test_head_projection_refuses_invalid_whole_metadata(tmp_path, bad):
    _model, plan = _plan(tmp_path)
    plan["metadata_reads"] = bad
    with pytest.raises(ValueError, match="metadata|malformed"):
        _head(plan)


def test_repeated_metadata_keeps_one_declaration_without_mutating_plan(tmp_path):
    _model, plan = _plan(tmp_path)
    expected = _head(plan)
    plan["metadata_reads"] += list(plan["metadata_reads"])
    before = copy.deepcopy(plan)
    assert _head(plan) == expected
    assert plan == before
