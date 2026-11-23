import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

from prismaquant import digests
from prismaquant.joint_retained_window_plan import EXECUTION_SCHEMA, RetainedWindowBudget


KEY = ("model.layers.0.q_proj", "FP8_E4M3")


class _Cache:
    def __init__(self, render):
        self.render = render

    def resolve_key(self, name, fmt):
        return KEY if (name, fmt) == KEY else None

    def estimate_nbytes(self, keys):
        assert keys == [KEY]
        return self.render.stat().st_size


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cache_and_report_use_owner_profiles(tmp_path, monkeypatch, capsysbinary):
    spec = importlib.util.spec_from_file_location(
        "derive_retained_window_budget",
        Path(__file__).resolve().parents[1] / "tools/derive_retained_window_budget.py",
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    monkeypatch.setattr(tool, "time", SimpleNamespace(time=lambda: 0.0))

    plan, prepared, census, payload, render, out, sizes = (
        tmp_path / name for name in (
            "plan.json", "prepared.json", "census.json", "cache.pkl", "render.bin",
            "out.json", "sizes.json",
        )
    )
    unit, fmt = KEY
    render.write_bytes(b"render")
    payload.write_bytes(pickle.dumps(_Cache(render)))
    prepared.write_text(json.dumps({
        "status": "complete", "formats_by_qname": {unit: [fmt]},
        "production_cache": {"path": str(payload), "sha256": _sha(payload)},
    }))
    census.write_text(json.dumps({"unit_shapes": {unit: [32, 32]}, "max_abs": {unit: 1.0}}))
    mib = 1024**2
    budget = RetainedWindowBudget(
        physical_limit_bytes=64 * mib, safety_margin_bytes=mib,
        metadata_reserve_bytes=mib, runtime_reserve_bytes=mib,
        workspace_reserve_bytes=mib, boundary_reserve_bytes=mib,
        auxiliary_reserve_bytes=mib, load_buffer_bytes=mib,
        read_page_reserve_bytes=4096, candidate_delta_bytes=mib,
        statistics_cap_bytes=mib, retained_render_cap_bytes=mib,
        max_windows_per_layer=8,
    )
    plan.write_text(json.dumps({
        "schema": "prismaquant.tessera_joint_aura.plan.v1", "max_gpu_bytes": 32 * mib,
        "execution": {
            "retained_operator_windows": {
                "schema": EXECUTION_SCHEMA, "budget": budget.as_dict(),
                "source_reserve_bytes": mib, "source_loading_reserve_bytes": mib,
            },
            "operator_windows": {
                "max_statistics_bytes": mib, "max_candidate_bytes": mib,
                "max_load_buffer_bytes": mib, "prefetch_workers": 1,
            },
            "boundary_storage": {
                "capture_order": "layer_major", "max_resident_bytes": mib,
                "max_auxiliary_bytes": mib,
            },
        },
    }))
    plan_sha = _sha(plan)
    argv = [
        "--plan", str(plan), "--plan-sha256", plan_sha,
        "--prepared", str(prepared), "--prepared-sha256", _sha(prepared),
        "--census", str(census), "--out", str(out), "--render-size-cache", str(sizes),
    ]

    calls = []
    prefixes = ("", "", "")
    spaced = id(digests.INSERTION_ASCII_SPACED_LAX)
    indent1 = id(digests.INSERTION_ASCII_INDENT1_LAX)
    real_text = digests.JsonProfile.text

    def observe(self, value):
        text = real_text(self, value)
        if id(self) in (spaced, indent1):
            prefix = prefixes[len(calls)] if len(calls) < len(prefixes) else ""
            calls.append((id(self), value))
            return prefix + text
        return text

    monkeypatch.setattr(digests.JsonProfile, "text", observe)
    assert tool.main(argv) == 0
    output = json.loads(out.read_text())
    derived = output["execution"]["retained_operator_windows"]["budget"]
    record = output["retained_window_budget_derivation"]
    key_text = json.dumps(list(KEY))
    cache = {key_text: render.stat().st_size}
    report = {
        "out": str(out.resolve()), "sha256": _sha(out), "superseded_plan_sha256": plan_sha,
        "changed": {
            "load_buffer_bytes": [mib, derived["load_buffer_bytes"]],
            "candidate_delta_bytes": [mib, derived["candidate_delta_bytes"]],
            "statistics_cap_bytes": [mib, derived["statistics_cap_bytes"]],
            "retained_render_cap_bytes": [mib, derived["retained_render_cap_bytes"]],
            "max_windows_per_layer": [8, 1],
        },
        "available_window_bytes": record["available_window_bytes"],
        "host_cap_bytes": 32 * mib,
        "host_render_bound_bytes": record["host_render_bound_bytes"],
        "windows_by_layer_max": 1,
        "retained_window_replay_multiplier": record["retained_window_replay_multiplier"],
        "candidate_files_measured": 1,
        "measure_seconds": 0.0,
    }
    stdout = (json.dumps(report, indent=1) + "\n").encode()
    assert sizes.read_bytes() == json.dumps(cache).encode()
    assert capsysbinary.readouterr().out == stdout
    assert calls == [(spaced, list(KEY)), (spaced, cache), (indent1, report)]

    calls.clear()
    prefixes = (" ", "  ", "   ")
    assert tool.main(argv) == 0
    prefixed_cache = {prefixes[0] + key_text: cache[key_text]}
    assert sizes.read_bytes() == (prefixes[1] + json.dumps(prefixed_cache)).encode()
    assert capsysbinary.readouterr().out == prefixes[2].encode() + stdout
