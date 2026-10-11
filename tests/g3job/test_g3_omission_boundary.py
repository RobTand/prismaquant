"""Retain checkpoint tensors unless each requested use replaces the whole tensor."""
import copy

from test_g3_readset import fixture
from g3_readset import SourceReads


def test_source_use_of_shared_checkpoint_name_prevents_omission(tmp_path):
    q, tensors, row = fixture(tmp_path)
    row["a8_rendered_shape"] = list(tensors[q + ".weight"].shape)
    used = dict(row, a8_format="SOURCE")
    reads = SourceReads(tmp_path, ["a8_w"], {0: [row, used]})
    assert q + ".weight" not in reads.omitted
    with reads.open(str(tmp_path / "model.safetensors"), framework="pt") as handle:
        from g3_lib import check_identity
        check_identity(handle.get_tensor(q + ".weight"), row["source_sha256"], q)


def test_partial_or_unknown_replacement_shape_prevents_omission(tmp_path):
    q, tensors, row = fixture(tmp_path)
    for shape in ([1, 4], None):
        partial = copy.deepcopy(row)
        if shape is not None:
            partial["a8_rendered_shape"] = shape
        else:
            partial.pop("a8_rendered_shape", None)
        reads = SourceReads(tmp_path, ["a8_w"], {0: [partial]})
        assert q + ".weight" not in reads.omitted
    row["a8_rendered_shape"] = list(tensors[q + ".weight"].shape)
    assert SourceReads(tmp_path, ["a8_w"], {0: [row]}).omitted == {q + ".weight"}


def test_manifest_assigns_wrapped_glm_source_to_its_layer(tmp_path):
    import json
    from g3_readset import build_manifest
    q, tensors, row = fixture(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "glm5_next", "num_hidden_layers": 1, "num_nextn_predict_layers": 0}))
    (tmp_path / "candidate").write_bytes(b"abcDEFG")
    reads = SourceReads(tmp_path, ["a8_w"], {0: [row]})
    manifest = build_manifest(reads, ["a8_w"], {0: [row]}, {"a8": str(tmp_path)}, [], [],
                              mount_prefix=str(tmp_path))
    attention = next(name for name in tensors if name != q + ".weight")
    offset = reads.entries["model.safetensors"][attention]["offset"]
    index = next(i for i, e in enumerate(manifest["entries"])
                 if e["path"].endswith("model.safetensors") and e["offset"] == offset)
    phases = {p["name"]: p["entry_indices"] for p in manifest["read_plan"]["phases"]}
    assert index in phases["layer-00"]
    assert index not in phases["setup"]
