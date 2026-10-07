"""Real Unigram tokenizers drive G3 preflight and retained measurement."""
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import Unigram

from prismaquant.g3_v2 import measure_g3, preflight_g3
from prismaquant.quality_stage import artifact, verify_result, evaluate_criteria
from test_g3_v2 import make_retained_configuration, put


def make_unigram_configuration(root, *, prefix):
    root.mkdir(parents=True, exist_ok=True)
    config = make_retained_configuration(root)
    tokenizer = Tokenizer(Unigram([("<unk>", 0.0), ("one", -1.0), ("[prefix]", -2.0),
        ("three", -3.0), ("four", -4.0), ("five", -5.0), ("six", -6.0)], unk_id=0))
    path = Path(config["tokenizer"]["path"])
    path.write_text(tokenizer.to_str())
    config["tokenizer"] = artifact(path)
    protocol_path = Path(config["protocol"]["path"])
    protocol = json.loads(protocol_path.read_text())
    protocol.update(prefix_tokens=["[prefix]"] if prefix else [], prefix_ids=[2] if prefix else [])
    config["protocol"] = put(protocol_path, protocol)
    teacher_path = Path(config["teacher"]["path"])
    teacher = json.loads(teacher_path.read_text())
    teacher.update(tokenizer=config["tokenizer"], prefix={"ids": protocol["prefix_ids"]})
    config["teacher"] = put(teacher_path, teacher)
    return config


@pytest.mark.parametrize("prefix", [False, True])
def test_real_unigram_preflight_and_retained_measurement(tmp_path, prefix):
    config = make_unigram_configuration(tmp_path, prefix=prefix)
    preflight = preflight_g3(config)
    assert preflight["population"]["prefix_ids"] == ([2] if prefix else [])
    facts = measure_g3(config, tmp_path / "result.json")
    assert facts["population"]["window_ids"] == ["w0", "w1"]
    assert facts["population"]["windows"] == 2 and facts["population"]["positions"] == 8
    result = {"schema": "prismaquant.quality_stage/1", "stage": "g3_v2",
        "configuration": {"schema": config["schema"], "sha256": "a" * 64},
        "measurement": {"status": "succeeded", "metric_kind": "offline_decoded_kl",
                        "metrics": facts.pop("metrics"), "error": None}, **facts}
    result["gate"] = evaluate_criteria(result["measurement"]["metrics"], config["criteria"])
    assert verify_result(result, config, config_sha256="a" * 64)["status"] == "not_evaluated"
