"""Graph release evidence must exceed a successful generation smoke."""
from prismaquant.shipcard import open_shipcard, verify
from prismaquant.shipcard_slots import make_record


def test_graph_smoke_without_equality_receipt_is_refused(tmp_path):
    model = tmp_path / "artifact"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3"}')
    (model / "model.safetensors").write_bytes(b"owned tiny weights")
    card = open_shipcard(model, build={"achieved_bpp": {"value": 4.75}})
    slot = "native_export.graph"
    card["slots"][slot] = make_record(
        slot=slot, tool="validate_native_export.py", passed=True,
        model_sha=card["model_sha"], git_commit="a" * 40,
        metrics={"arm": "graph", "enforce_eager": False,
                 "generated_chars": 128, "max_new_tokens": 16},
    )
    problems = verify(card, model_dir=model, required=[slot])
    assert any(p.startswith(slot + ":") and "receipt" in p for p in problems), problems
