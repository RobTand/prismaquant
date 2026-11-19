"""G3 replay verifies owned numeric data in both execution modes."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from prismaquant.g3_v2 import main, measure_g3
from prismaquant.quality_stage import artifact, evaluate_criteria, verify_result, write_result
from test_g3_v2 import make_retained_configuration


def make_g3_replay_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    config = make_retained_configuration(root)
    facts = measure_g3(config, root / "fixture.json")
    teacher = json.loads(Path(config["teacher"]["path"]).read_text())
    candidate = json.loads(Path(config["candidate"]["receipt"]["path"]).read_text())
    agreements = np.stack([np.load(left["path"]).argmax(-1) == np.load(right["path"]).argmax(-1)
                          for left, right in zip(teacher["arrays"], candidate["arrays"], strict=True)])
    agreement_path = root / "fixture.agreement.npy"
    np.save(agreement_path, agreements, allow_pickle=False)
    facts["artifacts"] = [facts["artifacts"][0], artifact(agreement_path)]
    facts["identity"]["panel"] = config["panel"]
    config["criteria"] = [{"metric": "mean_kl", "op": "le", "threshold": 1.0}]
    result = {"schema": "prismaquant.quality_stage/1", "stage": "g3_v2",
        "configuration": {"schema": config["schema"], "sha256": "a" * 64, "path": str(root / "config.json")},
        "measurement": {"status": "succeeded", "metric_kind": "offline_decoded_kl",
                        "metrics": facts.pop("metrics"), "error": None}, **facts}
    result["gate"] = evaluate_criteria(result["measurement"]["metrics"], config["criteria"])
    return config, json.loads(json.dumps(result)), Path(result["artifacts"][0]["path"]), agreement_path


@pytest.fixture
def g3_replay_case(tmp_path):
    return make_g3_replay_fixture(tmp_path)


@pytest.mark.parametrize("dev", [False, True])
@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_g3_owned_kl_array_refuses_damage_in_both_modes(g3_replay_case, monkeypatch, dev, damage):
    config, result, kl, _ = g3_replay_case
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1" if dev else "0")
    kl.unlink() if damage == "missing" else kl.write_bytes(b"corrupt KL array")
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_result(result, config, config_sha256="a" * 64)


@pytest.mark.parametrize("dev", [False, True])
@pytest.mark.parametrize("metric", ["mean_kl", "p99_kl", "max_kl", "top1_agreement"])
def test_g3_edited_metric_and_consistent_gate_still_refuse(g3_replay_case, monkeypatch, dev, metric):
    config, result, _, _ = g3_replay_case
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1" if dev else "0")
    config["criteria"] = [{"metric": metric, "op": "le", "threshold": 2.0}]
    result["measurement"]["metrics"][metric] = 0.75
    result["gate"] = evaluate_criteria(result["measurement"]["metrics"], config["criteria"])
    with pytest.raises(ValueError, match="metric"):
        verify_result(result, config, config_sha256="a" * 64)


@pytest.mark.parametrize("dev", [False, True])
@pytest.mark.parametrize("field", ["positions", "windows", "window_ids", "per_window"])
def test_g3_population_comes_from_owned_arrays(g3_replay_case, monkeypatch, dev, field):
    config, result, _, _ = g3_replay_case
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1" if dev else "0")
    if field in ("positions", "windows"):
        result["population"][field] += 1
    elif field == "window_ids":
        result["population"][field].reverse()
    else:
        result["population"][field][0]["mean_kl"] = 0.75
    with pytest.raises(ValueError, match="population|window"):
        verify_result(result, config, config_sha256="a" * 64)


@pytest.mark.parametrize("dev", [False, True])
def test_g3_owned_agreement_array_is_not_an_unchecked_gate_metric(g3_replay_case, monkeypatch, dev):
    config, result, _, agreement = g3_replay_case
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1" if dev else "0")
    agreement.write_bytes(b"corrupt agreement array")
    with pytest.raises(ValueError):
        verify_result(result, config, config_sha256="a" * 64)


@pytest.fixture(scope="module")
def g3_comparison_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("g3-comparison")
    config = make_retained_configuration(root)
    protocol, panel, teacher, candidate = (
        json.loads(Path(link["path"]).read_text()) for link in
        (config["protocol"], config["panel"], config["teacher"], config["candidate"]["receipt"])
    )
    names = [f"w{i}" for i in range(25)]

    def bind_record(name, record):
        path = root / f"{name}.json"
        write_result(path, record)
        return artifact(path)

    def expand(rows):
        return [dict(rows[i % 2], window_id=name) for i, name in enumerate(names)]

    protocol["window_count"] = len(names)
    panel["windows"] = expand(panel["windows"])
    config["protocol"] = bind_record("paired-protocol", protocol)
    config["panel"] = bind_record("paired-panel", panel)
    teacher.update(windows=names, arrays=expand(teacher["arrays"]), panel=config["panel"])
    teacher_link = bind_record("paired-teacher", teacher)
    perturbed = expand(candidate["arrays"])
    paths, links = [], []
    for parity, label in enumerate(("pick", "baseline")):
        teacher_path = root / f"{label}-teacher.json"
        teacher_path.write_bytes(Path(teacher_link["path"]).read_bytes())
        config["teacher"] = artifact(teacher_path)
        rows = [row if i % 2 == parity else teacher["arrays"][i] for i, row in enumerate(perturbed)]
        config["candidate"]["receipt"] = bind_record(f"{label}-candidate", {"windows": names, "arrays": rows})
        config_path = bind_record(f"{label}-config", config)["path"]
        path = root / f"{label}-result.json"
        assert main(["--config", config_path, "--output", str(path)]) == 0
        paths.append(path)
        links.append(config["teacher"])
    return paths, names, links


def _compare_g3(paths):
    return subprocess.run([sys.executable, "-m", "prismaquant.g3_v2", "compare", *map(str, paths)],
                          capture_output=True, text=True, check=False)


def test_g3_comparison_reports_signed_deltas(g3_comparison_case):
    paths, names, links = g3_comparison_case
    pick, baseline = (np.load(json.loads(path.read_text())["artifacts"][0]["path"], allow_pickle=False)
                      for path in paths)
    expected = pick.mean(axis=1) - baseline.mean(axis=1)
    assert np.all(expected[::2] > 0) and np.all(expected[1::2] < 0)
    process = _compare_g3(paths)
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert [report["pick_teacher"], report["baseline_teacher"]] == links
    assert [row["window_id"] for row in report["per_window"]] == names
    assert [row["delta_kl"] for row in report["per_window"]] == pytest.approx(expected, rel=1e-12, abs=1e-15)
    assert report["mean_delta_kl"] == pytest.approx(expected.mean(), rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("damage", ["digest", "short", "reverse_pick", "reverse_both"])
def test_g3_comparison_refuses_unpaired_results(g3_comparison_case, tmp_path, damage):
    paths, _, links = g3_comparison_case
    pick, baseline = (json.loads(path.read_text()) for path in paths)
    if damage in ("digest", "short"):
        teacher_path = tmp_path / "teacher.json"
        raw = Path(links[0]["path"]).read_bytes()
        if damage == "digest":
            teacher_path.write_bytes(raw + b"\n")
            pick["identity"]["teacher"] = artifact(teacher_path)
        else:
            teacher = json.loads(raw)
            for field in ("windows", "arrays"):
                teacher[field].pop()
            write_result(teacher_path, teacher)
            for result in (pick, baseline):
                result["identity"]["teacher"] = artifact(teacher_path)
                population = result["population"]
                population["windows"] = 24
                population["window_ids"].pop()
                population["per_window"].pop()
    else:
        for result in (pick,) if damage == "reverse_pick" else (pick, baseline):
            result["population"]["window_ids"].reverse()
            result["population"]["per_window"].reverse()
    paths = [tmp_path / "pick.json", tmp_path / "baseline.json"]
    for path, result in zip(paths, (pick, baseline), strict=True):
        write_result(path, result)
    process = _compare_g3(paths)
    assert process.returncode != 0
    assert process.stdout == ""


