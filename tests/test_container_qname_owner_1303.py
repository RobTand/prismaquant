"""The safetensors container grammar and the shared qname grammars have one
home each, and the remaining callers read through it (PQ #1303).

Consumer behavior under test:

* ``prismaquant.model_profiles.validate._safetensors_header`` and
  ``tools.chain_roll_bench._safetensors_spans`` return the same header bytes
  and span arithmetic on well-formed shards, and refuse a corrupt prefix with
  the container grammar owner's named refusal
  (``prismaquant.source_read_plan.read_safetensors_header``) rather than a
  bare JSON decode error.
* The streaming-initialization prefix/layer grammar is stated once.
* The per-expert cost-name grammar is stated once in measure_quant_cost.
* The fused NVFP4 kernel module does not grow a second E2M1 maximum.
"""

from __future__ import annotations

import ast
import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))


def _safetensors_bytes(entries: dict, payload: bytes = b"\x00" * 32) -> bytes:
    """One well-formed safetensors file: u64 LE header length, JSON, payload."""
    header = json.dumps(entries, separators=(",", ":")).encode()
    return struct.pack("<Q", len(header)) + header + payload


_DTYPE_HEAD = {
    "w0": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
    "w1": {"dtype": "F32", "shape": [1], "data_offsets": [16, 20]},
}


def _write_shard(path: Path, entries: dict) -> int:
    raw = _safetensors_bytes(entries)
    path.write_bytes(raw)
    return len(raw)


class TestValidateHeaderRoutesThroughOwner:
    """model_profiles/validate.py reads headers through the grammar owner."""

    def test_valid_header_bytes_come_back(self, tmp_path):
        from prismaquant.model_profiles.validate import _safetensors_header

        shard = tmp_path / "model.safetensors"
        entries = {"__metadata__": {"fmt": "pt"}, **_DTYPE_HEAD}
        _write_shard(shard, entries)
        header = _safetensors_header(shard)
        assert header["__metadata__"] == {"fmt": "pt"}
        assert header["w0"]["data_offsets"] == [0, 8]
        assert header["w1"]["shape"] == [1]

    def test_corrupt_prefix_is_owner_refusal_not_json_error(self, tmp_path):
        from prismaquant.model_profiles.validate import _safetensors_header

        shard = tmp_path / "model.safetensors"
        shard.write_bytes(struct.pack("<Q", 1 << 40) + b"{not the header")
        with pytest.raises(
            ValueError, match=r"has an invalid safetensors header length",
        ):
            _safetensors_header(shard)

    def test_short_file_is_owner_refusal(self, tmp_path):
        from prismaquant.model_profiles.validate import _safetensors_header

        shard = tmp_path / "model.safetensors"
        shard.write_bytes(b"\x01\x02")
        with pytest.raises(
            ValueError, match=r"is too short to be a safetensors file",
        ):
            _safetensors_header(shard)


class TestChainRollSpansRoutesThroughOwner:
    """chain_roll_bench's manifest spans keep their arithmetic and gain the
    owner's refusals; well-formed output is unchanged."""

    def _fixture(self, tmp_path):
        head_a = json.dumps(
            {"__metadata__": {}, **_DTYPE_HEAD}, separators=(",", ":")
        ).encode()
        head_b = json.dumps(
            {"__metadata__": {}, "w2": {"dtype": "F32", "shape": [1],
                                        "data_offsets": [0, 4]}},
            separators=(",", ":"),
        ).encode()
        shard_a = tmp_path / "model-00001-of-00002.safetensors"
        shard_b = tmp_path / "model-00002-of-00002.safetensors"
        shard_a.write_bytes(struct.pack("<Q", len(head_a)) + head_a + b"\x00" * 32)
        shard_b.write_bytes(struct.pack("<Q", len(head_b)) + head_b + b"\x00" * 8)
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"w0": shard_a.name, "w1": shard_a.name,
                           "w2": shard_b.name},
        }))
        return shard_a, shard_b, 8 + len(head_a), 8 + len(head_b)

    def test_spans_are_header_plus_tensor_ranges_merged(self, tmp_path):
        import chain_roll_bench

        shard_a, shard_b, base_a, base_b = self._fixture(tmp_path)
        entries = chain_roll_bench._safetensors_spans(
            tmp_path, ["w0", "w1", "w2"])
        # Shard A: header [0, base_a) merges with w0 [base_a, base_a+8);
        # w1 sits [base_a+16, base_a+20) across a gap, so it stays its own
        # span.  Shard B: header and w2 are adjacent and merge.
        assert entries == [
            {"path": str(shard_a), "offset": 0, "bytes": base_a + 8,
             "sha256": None},
            {"path": str(shard_a), "offset": base_a + 16, "bytes": 4,
             "sha256": None},
            {"path": str(shard_b), "offset": 0, "bytes": base_b + 4,
             "sha256": None},
        ]

    def test_corrupt_shard_prefix_is_owner_refusal(self, tmp_path):
        import chain_roll_bench

        shard_a, shard_b, _base_a, _base_b = self._fixture(tmp_path)
        shard_b.write_bytes(struct.pack("<Q", 1 << 40) + b"junk")
        with pytest.raises(
            ValueError, match=r"has an invalid safetensors header length",
        ):
            chain_roll_bench._safetensors_spans(tmp_path, ["w0", "w2"])


class TestQnameGrammarsStatedOnce:
    """Each shared name grammar is written once in its owning module."""

    def test_per_expert_pattern_is_defined_once(self):
        source = (ROOT / "prismaquant" / "measure_quant_cost.py").read_text()
        assert source.count(r"^(.+\.experts)\.(\d+)\.([^.]+)$") == 1

    def test_canonical_linear_name_uses_the_compiled_grammar(self):
        module = pytest.importorskip("prismaquant.measure_quant_cost")
        tree = ast.parse(
            (ROOT / "prismaquant" / "measure_quant_cost.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == (
                "canonical_linear_name"
            ):
                called = [
                    ast.unparse(n.func) for n in ast.walk(node)
                    if isinstance(n, ast.Call)
                ]
                assert "_PER_EXPERT_NAME_RE.match" in called
                break
        else:
            pytest.fail("canonical_linear_name not found")
        # The compiled grammar is the one the function already used: a
        # per-expert live name decomposes into (prefix, id, projection).
        assert module._PER_EXPERT_NAME_RE.match(
            "model.layers.3.mlp.experts.7.gate_proj"
        ).groups() == ("model.layers.3.mlp.experts", "7", "gate_proj")

    def test_streaming_prefix_grammar_is_stated_once(self):
        source = (
            ROOT / "prismaquant" / "streaming_initialization.py").read_text()
        assert source.count(r'r"(\d+)\..+"') == 1

    def test_prefix_layer_index_semantics_unchanged(self):
        from prismaquant.streaming_initialization import _prefix_layer_index

        assert _prefix_layer_index("model.layers.3.foo", "model.layers.") == 3
        assert _prefix_layer_index(
            "model.language_model.layers.45.x", "model.language_model.layers.",
        ) == 45
        assert _prefix_layer_index("model.layers.x.foo", "model.layers.") is None
        assert _prefix_layer_index("model.layers.3", "model.layers.") is None
        # A deeper dotted tail is still one layer component: 3, not a refusal.
        assert _prefix_layer_index("model.layers.3.other.4.z", "model.layers.") == 3
        # The prefix is a literal, never a pattern: metacharacters are escaped.
        assert _prefix_layer_index(
            "model(1).layers.3.foo", "model(1).layers.") == 3


class TestFusedKernelHasNoSecondE2M1Max:
    """The E2M1 maximum is owned by the activation contract; the fused kernel
    module re-declaring it locally is how a second convention grows."""

    def test_no_local_e2m1_max(self):
        source = (ROOT / "prismaquant" / "kernels" / "nvfp4_fused.py").read_text()
        tree = ast.parse(source)
        local_max_names = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name) and "E2M1" in target.id.upper()
        }
        assert not local_max_names, (
            "the fused kernel module declares its own E2M1 constant: "
            f"{sorted(local_max_names)}")
        assert "_FP4_E2M1_MAX" not in source
