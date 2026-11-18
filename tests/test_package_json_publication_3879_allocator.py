import json
import pickle
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import allocator
from prismaquant.digests import DIRECT_ASCII_INDENT2_LAX, DIRECT_ASCII_LAX, JsonProfile
from prismaquant.serve_dispatch_table import SCHEMA

_DIGESTS = (
    "cfb38aa4faf58cc3efff862c4121ce238f4380cfd5aa42a83f2829f1915833e8",
    "a6bcc9e9dc912d661b8e0225ea93bdde484bf5b019ab009713c7987ac02d2cb8",
    "3735ae8eb6ecf8837e168542460da5eb085119614eac16568f57df72acc3edff",
    "9c50c5c199e50e8e2174bc6138c1a4aca7c1162263a30397f95a00af8d4f7c56",
    "7d2254ce93999f81d71004d038a5d505ed8b5f04934cfc54525b90fdb8438fca",
    "a36a4ef19a7589d288ee296bac98ad144b25e77f93d98d4c585c109708461620",
    "b05cd9a9ba0e839468f6192b365d58c7420c2e04b927b6dbccf3710ef625d72e",
    "ddb79ce1523b959aadfbffd7ede90bdeff56f16774d331ec6a1f18e8bb982cd6",
    "8059af3b2b5ef3aafd7072b25a43fc55b612e51a92b30fb8229b0920ecc11630",
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "d4128392f3d8af8ed571f66a8c9b3ab4b03f7f863902a1dde888bd382c621153",
    "35f273c3c2f9a569b64179acdacb8d1ab7fc2ce8b4a1b62b64fd882d6fc9e081",
    "7ded9604b88b1986fb0023a0661423f2734dc7c85a98d75fa6e34e37f6624771",
    "4a2f0b7ab5ec41ed5054152875ce4ae1c93a7e3910b23cac25cd8741abbb7b5b",
    "fe307a822814b5eb8afe959d1854e4068db8b0b9a0678051e343cc93e41a647e",
    "1680d971f3d3edf5a68da9ec2ba368580d9b0cb9b0731e729052454ef5523f9a",
    "0fd36846a0c21062dbe99bd0368834b7acb4bda0abf51561dfadbd5ca8019bdb",
    "cf0efcb316c2d0a5722504fea1d4b9cd21daf3890886729fef318b29261f4190",
    "45649035a0d679cc447d99bb715ed943f54c70aabb6b8ab3503ac7a79afb31ef",
    "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356",
    "6b0b49150ffdd527b0508fb768a055f35c083f26d69695763b11043ee791f408",
    "aaffa53e4e6a2c9cb8b066b295f7eb7491952bbb66fb5572a69731f1b7169f33",
    "61e7e36d356c28ffabca2ce37ee72079d71f8de0b144f7c69cd32cbb8042d368",
    "96a02392e0df90b95ad9ec2a5f97852ef340e608a4a089f54610a86b39f68994",
)
_APP = "format_applicability.json"
_KNEES = "pareto.knees.json"
_NV_SEED = "seeds/allocator_target_4p6000_achieved_4p5010_d4128392f3d8.json"
_FP_SEED = "seeds/allocator_target_8p2000_achieved_8p1250_35f273c3c2f9.json"
_MANIFEST = "seeds/manifest.json"
_SELECTION = "selection.json"
_ORDER = (_APP, _KNEES, 10, _NV_SEED, 11, _FP_SEED, _MANIFEST, _SELECTION)
_NORMAL = {_APP: 0, "pareto.csv": 1, _KNEES: 2, _SELECTION: 3, "layer.json": 4,
           _NV_SEED: 5, _FP_SEED: 6, _MANIFEST: 7}
_RICH = {"unicode": "snowman ☃ / café", "nested": [None, True, -1, 1.5, [], {}],
         "nan": float("nan"), "infinity": float("inf"), "negative_infinity": -float("inf")}
_KNEE_VARIANTS = {"knee-empty": {}, "knee-rich": _RICH,
                  "knee-mixed": {1: "number", "x": "text"}, "knee-object": {"bad": object()}}
_TERMINALS = {
    "below-floor": (SystemExit,
                    "[alloc] --target-disk-gb=0.000 is below the floor: the cheapest allocation is "
                    "0.000GB. Raise the budget, fix lm_head to a smaller measured format "
                    "(--lm-head-format FP8_E4M3), unpin it for a research DP (--allow-pinned "
                    "lm_head), or widen the format menu. Selection written to selection.json."),
    "serve-refusal": (SystemExit,
                      "[alloc] every allocation that fits the 1.000GB card misses a hard serving "
                      "constraint (p95_ttft_ms). Policy §1 makes these constraints, not penalties: "
                      "nothing here is 'close enough'. Raise the SLO, widen --formats, change "
                      "--serve-workload-mix, or supply a dispatch table that prices the families in "
                      "this menu. Selection written to selection.json."),
    "knee-mixed": (TypeError, "'<' not supported between instances of 'str' and 'int'"),
    "knee-object": (TypeError, "Object of type object is not JSON serializable"),
}


@pytest.fixture(params=["normal", "below-floor", "serve-refusal", *_KNEE_VARIANTS])
def case(request, tmp_path, monkeypatch):
    name = request.param
    monkeypatch.chdir(tmp_path)
    names = [f"model.layers.{i}.self_attn.o_proj" for i in range(4)]
    formats = {"NVFP4": 0.0001, "FP8_E4M3": 0.000001}
    tensors = {f"{n}.weight": torch.zeros((256, 256), dtype=torch.bfloat16) for n in names}
    tensors.update({n: torch.zeros((512, 64), dtype=torch.bfloat16)
                    for n in ("model.embed_tokens.weight", "lm_head.weight")})
    Path("model").mkdir()
    save_file(tensors, "model/model.safetensors")
    stats = {n: dict(h_trace=1.0, n_params=65536, in_features=256, out_features=256)
             for n in names}
    Path("probe.pkl").write_bytes(pickle.dumps({"stats": stats, "meta": {"model": "model"}}))
    costs = {fmt: dict(predicted_dloss=loss) for fmt, loss in formats.items()}
    Path("cost.pkl").write_bytes(pickle.dumps({"costs": dict.fromkeys(names, costs),
                                              "meta": {"formats": list(formats)}}))

    def solve(stats, candidates, target_bits, format_specs, format_rank, bit_precision, **kw):
        fmt = "NVFP4" if target_bits < 6 else "FP8_E4M3"
        bits = sum(8 * next(c.memory_bytes for c in rows if c.fmt == fmt)
                   for rows in candidates.values())
        achieved = bits / sum(stats[n]["n_params"] for n in candidates)
        if kw.get("diagnostics") is not None:
            kw["diagnostics"].update(feasible=True, achieved_bits=achieved, predicted_dloss=None, evals=1)
        return dict.fromkeys(candidates, fmt), achieved

    monkeypatch.setattr(allocator, "solve_with_promotion", solve)
    argv = ["allocator", "--probe", "probe.pkl", "--costs", "cost.pkl", "--formats", ",".join(formats),
            "--pareto-targets", "4.6,8.2", "--target-disk-gb", "0.00001" if name == "below-floor" else "1.0",
            "--artifact-overhead-reserve-bytes", "512", "--allow-default-profile",
            "--layer-config", "layer.json", "--pareto-csv", "pareto.csv", "--pareto-output-dir", "seeds"]
    if name in _KNEE_VARIANTS:
        monkeypatch.setattr(allocator, "_pareto_knee_summary", lambda curve: _KNEE_VARIANTS[name])
    if name == "serve-refusal":
        provenance = dict(source="CPU fixture", date="2026-11-18", gpu="synthetic",
                          measured_quantity="synthetic relative serving cost", units="dimensionless",
                          derivation="fixture constant")
        arenas = [dict(phase=phase, m_regime=regime, m=m, reference_route="fixture native",
                       metric=metric, absolute_value=value, statistic=statistic, provenance=provenance)
                  for phase, regime, m, metric, value, statistic in (
                      ("prefill", "dense", 1400, "ttft_ms", 1000.0, "p95"),
                      ("decode", "batch1", 1, "decode_tok_s", 10.0, "p05"))]
        rows = [dict(format_family=fmt, phase=arena["phase"], m_regime=arena["m_regime"],
                     lane="native", relative_unit_cost=cost if arena["phase"] == "prefill" else 1.0,
                     provenance=provenance)
                for fmt, cost in zip(formats, (5.0, 9.0)) for arena in arenas]
        Path("table.json").write_text(json.dumps(dict(schema=SCHEMA, table_id="fixture",
            status="proposal_data", description="synthetic fixture", arenas=arenas, rows=rows)))
        argv.extend(["--serve-dispatch-table", "table.json", "--serve-workload-mix", "prefill:dense=1.0",
                     "--slo-prefill-p95-ttft-ms", "1200"])
    monkeypatch.setattr(sys, "argv", argv)
    return name


def _run(case):
    if case not in _TERMINALS:
        assert allocator.main() is None
    else:
        kind, message = _TERMINALS[case]
        with pytest.raises(kind) as caught:
            allocator.main()
        assert (caught.type, str(caught.value), caught.value.__cause__) == (kind, message, None)


def _records(case):
    files, stdout = _NORMAL.copy(), 8
    if case == "below-floor":
        files.update({_SELECTION: 12, _NV_SEED: 13, _FP_SEED: 14, _MANIFEST: 15})
        stdout = 16
    elif case == "serve-refusal":
        files[_SELECTION], stdout = 17, 18
    elif case == "knee-empty":
        files[_KNEES], files[_MANIFEST] = 19, 20
    elif case == "knee-rich":
        files[_KNEES], files[_MANIFEST] = 21, 22
    elif case in _TERMINALS:
        return {_APP: 0, "pareto.csv": 1}, 23
    if case in _TERMINALS:
        del files["layer.json"]
    return files, stdout


def test_allocator_publication_recorded_values(case, monkeypatch, capsysbinary):
    compact = []

    def record_sha256(data):
        compact.append(data)
        return sha256(data)

    monkeypatch.setattr(allocator, "hashlib", SimpleNamespace(sha256=record_sha256))
    inputs = set(Path().rglob("*"))
    _run(case)
    captured = capsysbinary.readouterr()
    expected, stdout = _records(case)
    outputs = set(Path().rglob("*")) - inputs
    assert Path("seeds").is_dir() == (_MANIFEST in expected)
    actual = {str(p): p.read_bytes() for p in outputs if p.is_file()}
    assert {name: sha256(data).hexdigest() for name, data in actual.items()} == {
        name: _DIGESTS[index] for name, index in expected.items()
    }
    assert [sha256(data).hexdigest() for data in compact] == (
        list(_DIGESTS[10:12]) if _MANIFEST in expected else []
    )
    assert sha256(captured.out).hexdigest() == _DIGESTS[stdout]
    assert sha256(captured.err).hexdigest() == _DIGESTS[9]
    for data in compact:
        assert b'"model.layers.0.self_attn.o_proj"' in data
    literals = {_APP: b'"activation_fair_pricing"', _NV_SEED: b'"achieved_bits"',
                _FP_SEED: b'"achieved_bits"', _MANIFEST: b'"candidates"'}
    if case not in _KNEE_VARIANTS:
        literals[_KNEES] = b'"too_few_feasible_points"'
    literals[_SELECTION] = {"below-floor": b'"below_floor": true',
                            "serve-refusal": b'"below_floor": false'}.get(
                                case, b'"activation_fair_pricing"')
    for name in literals.keys() & actual.keys():
        assert literals[name] in actual[name]


def test_allocator_publication_profile_calls(case, monkeypatch):
    calls = []
    original = JsonProfile.text

    def recording_text(self, value):
        if sys._getframe(1).f_code is not allocator.main.__code__:
            return original(self, value)
        index = len(calls)
        calls.append((id(self), None))
        text = original(self, value)
        data = text.encode("utf-8") + (b"" if self is DIRECT_ASCII_LAX else b"\n")
        calls[index] = (id(self), sha256(data).hexdigest())
        return text

    monkeypatch.setattr(JsonProfile, "text", recording_text)
    _run(case)
    expected, _ = _records(case)
    indent, compact = id(DIRECT_ASCII_INDENT2_LAX), id(DIRECT_ASCII_LAX)
    if _KNEES not in expected:
        trace = [(indent, _DIGESTS[0]), (indent, None)]
    else:
        trace = [(compact, _DIGESTS[site]) if isinstance(site, int)
                 else (indent, _DIGESTS[expected[site]]) for site in _ORDER]
    assert calls == trace
