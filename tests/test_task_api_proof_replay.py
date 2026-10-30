"""Replay genuine retained CPU API bytes without model or endpoint access."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from prismaquant.digests import bytes_sha256hex


EVIDENCE = Path(__file__).parent / "fixtures" / "task_api_proof"


def _replay(directory):
    from tools.replay_task_api_proof import replay_proof
    return replay_proof(directory)


@pytest.fixture
def evidence(tmp_path):
    return Path(shutil.copytree(EVIDENCE, tmp_path / "proof"))


def _change_json(directory, name, mutate):
    path = directory / name
    value = json.loads(path.read_text())
    mutate(value)
    raw = (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(raw)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][name].update(bytes=len(raw), sha256=bytes_sha256hex(raw))
    manifest_path.write_text(json.dumps(manifest))


def test_replay_real_cpu_api_proof():
    result = _replay(EVIDENCE)
    assert result["execution"] == "historical_retained_replay"
    assert result["runtime_witness_qualification"] == "not_run"
    assert result["population"] == {"device": "cpu", "documents": 2,
                                    "scoring_choices": 4, "generation_requests": 2}
    assert [row["sum_logprobs"] for row in result["scoring"]] == [
        -1.9225385934114456, -2.3094848692417145,
        -8.170122146606445, -1.5276103019714355]
    assert result["metrics"]["d50_cpu_facts_scoring/acc,none"] == 1.0


def test_cli_replays_without_heavy_imports_or_network():
    code = """
import builtins, runpy, socket, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'lm_eval', 'transformers', 'tessera'}:
        raise AssertionError('unexpected heavy import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
def refuse(*args, **kwargs):
    raise AssertionError('unexpected network access')
socket.socket = refuse
sys.argv = ['replay_task_api_proof', '--evidence-dir', sys.argv[1]]
runpy.run_module('tools.replay_task_api_proof', run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", code, str(EVIDENCE)],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["population"]["scoring_choices"] == 4


@pytest.mark.parametrize("name", ["proof3/llama_cpp.http.0003.response.bin",
                                  "proof3/llama_cpp/scoring.result.json"])
def test_replay_refuses_missing_owned_bytes(evidence, name):
    (evidence / name).unlink()
    with pytest.raises(ValueError, match="missing evidence"):
        _replay(evidence)


def test_replay_refuses_corrupt_owned_bytes(evidence):
    (evidence / "proof3/llama_cpp.http.0003.response.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="evidence bytes differ"):
        _replay(evidence)


@pytest.mark.parametrize("field", ["metric", "count", "seed"])
def test_replay_refuses_changed_raw_policy_or_metric(evidence, field):
    def mutate(raw):
        if field == "metric":
            raw["results"]["d50_cpu_facts_scoring"]["acc,none"] = 0.0
        elif field == "count":
            raw["n-samples"]["d50_cpu_facts_scoring"]["effective"] = 1
        else:
            raw["config"]["torch_seed"] = 19
    _change_json(evidence, "proof3/llama_cpp/scoring.result.json", mutate)
    with pytest.raises(ValueError, match="metric|population|sampling"):
        _replay(evidence)


@pytest.mark.parametrize("field", ["definition", "version"])
def test_replay_refuses_changed_task_definition_or_version(evidence, field):
    def mutate(raw):
        if field == "definition":
            raw["configs"]["d50_cpu_facts_scoring"]["doc_to_target"] = "answer"
        else:
            raw["versions"]["d50_cpu_facts_scoring"] = 2
    _change_json(evidence, "proof3/llama_cpp/scoring.result.json", mutate)
    with pytest.raises(ValueError, match="definitions|versions"):
        _replay(evidence)


def test_replay_refuses_changed_raw_score(evidence):
    def mutate(raw):
        sample = raw["samples"]["d50_cpu_facts_scoring"][0]
        sample["resps"][0][0][0] = -1.0
        sample["filtered_resps"][0][0] = -1.0
    _change_json(evidence, "proof3/llama_cpp/scoring.result.json", mutate)
    with pytest.raises(ValueError, match="raw scoring result"):
        _replay(evidence)


def test_replay_refuses_token_alignment_even_with_updated_digest(evidence):
    response_name = "proof3/llama_cpp.http.0003.response.bin"
    _change_json(evidence, response_name,
                 lambda raw: raw["choices"][0]["logprobs"]["tokens"].__setitem__(7, " wrong"))
    manifest = json.loads((evidence / "manifest.json").read_text())
    response = json.loads((evidence / response_name).read_text())
    def mutate(transcript):
        transcript[3]["response"].update({k: manifest["files"][response_name][k]
                                         for k in ("bytes", "sha256")})
        transcript[3]["response_json"] = response
    _change_json(evidence, "proof3/transcript.json", mutate)
    manifest = json.loads((evidence / "manifest.json").read_text())
    _change_json(evidence, "proof3/capabilities.json", lambda raw:
        raw["transcript"].update({key: manifest["files"]["proof3/transcript.json"][key]
                                  for key in ("bytes", "sha256")}))
    with pytest.raises(ValueError, match="token alignment"):
        _replay(evidence)
