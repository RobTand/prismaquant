"""Cut the route_trace_union fixtures from the real A8 (GLM-5.3, MTP) inputs.

The A8 artifact prices the MTP draft layer's routed experts
(`model.language_model.layers.45.mlp.experts`, BF16) and no non-speculative
serve ever dispatches it.  This script keeps the real shapes, headers and
module-name spellings of the A8 tr3 traces and cuts them down to a handful of
modules so the test suite stays small:

  layer 0   mlp.gate_up_proj, mlp.down_proj           (FP8 dense)
  layer 10  mlp.experts (FP8 routed_moe), mlp.shared_experts.{gate_up,down}_proj
  layer 45  mlp.experts (BF16 routed_moe, the MTP draft layer; config only)

Only the M1 and M2048 entries are kept.  The `nonspec-rank{0,1}.json` traces are
real entries.  `spec-rank{0,1}.json` is the nonspec cut plus one synthesized
entry per token count, layer 45 on the BF16 contract (see PROVENANCE.md): no real speculative
trace with the draft layer dispatched exists yet.  `empty-rank{0,1}.json` is the
real header-only 2c trace A8 produced (a serve that wrote no dispatches).

Usage: generate.py RUN_DIR CONFIG_JSON   (RUN_DIR is the A8 window's run dir)
"""
import copy
import json
import pathlib
import sys

OUT = pathlib.Path(__file__).resolve().parent
KEEP_LAYERS = {
    "layers.0.mlp.gate_up_proj", "layers.0.mlp.down_proj",
    "layers.10.mlp.experts", "layers.10.mlp.shared_experts.gate_up_proj",
    "layers.10.mlp.shared_experts.down_proj",
}
MTP_TARGET = "model.language_model.layers.45.mlp.experts"
KEEP_SHAPES = ("M1:", "M2048:")


def _kept(name):
    return any(name.endswith("." + layer) for layer in KEEP_LAYERS)


def cut_trace(doc):
    out = copy.deepcopy(doc)
    entries = []
    for entry in doc["entries"]:
        if not entry["shape"].startswith(KEEP_SHAPES):
            continue
        names = [n for n in entry["module_names"] if _kept(n)]
        if not names:
            continue
        entry = copy.deepcopy(entry)
        entry["module_names"] = names
        entry["modules"] = len(names)
        entries.append(entry)
    out["entries"] = entries
    return out


def with_mtp_layer(doc):
    """Add the synthesized layer-45 BF16 routed-MoE dispatch at every M.

    The gate requires the same modules at every token count, so the draft
    layer gets one entry per M, each a copy of that M's real FP8 routed-MoE
    entry with the family, contract and module name replaced.
    """
    out = copy.deepcopy(doc)
    for moe in [e for e in out["entries"] if e["kind"] == "moe"]:
        entry = copy.deepcopy(moe)
        entry.update(policy="TESSERA_BF16:resident", contract="bf16_unquantized",
                     module_names=["model.layers.45.mlp.experts"],
                     modules=1, launches=1)
        out["entries"].append(entry)
    return out


def cut_config(config):
    out = copy.deepcopy(config)
    q = out["quantization_config"]
    q["config_groups"] = {
        name: group for name, group in q["config_groups"].items()
        if any(t == MTP_TARGET or _kept(t) for t in group["targets"])}
    q["ignore"] = []
    return out


def main(run_dir, config_path):
    run = pathlib.Path(run_dir)
    ranks = {0: run / "head/route/tr3-rank0.json", 1: run / "route/tr3-rank1.json"}
    empties = {r: run / f"2c-evidence/route-trace-rank{r}.json" for r in (0, 1)}
    for rank, path in ranks.items():
        nonspec = cut_trace(json.loads(path.read_text()))
        (OUT / f"nonspec-rank{rank}.json").write_text(json.dumps(nonspec, indent=1) + "\n")
        (OUT / f"spec-rank{rank}.json").write_text(
            json.dumps(with_mtp_layer(nonspec), indent=1) + "\n")
        (OUT / f"empty-rank{rank}.json").write_text(empties[rank].read_text())
    (OUT / "config.json").write_text(
        json.dumps(cut_config(json.loads(pathlib.Path(config_path).read_text())), indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
