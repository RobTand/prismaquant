import builtins
import json
import pickle

from prismaquant import aura_additivity_gate, digests


def test_listed_calls_use_insertion_ascii_indent1_lax(tmp_path, monkeypatch, capsysbinary):
    costs, assignment, report = (tmp_path / name for name in ("cost.pkl", "a.json", "report.json"))
    payload = {"costs": {}}
    formats = {"layer_é": "FP8_E4M3"}
    costs.write_bytes(pickle.dumps(payload))
    assignment.write_text(json.dumps(formats))
    expected = json.dumps(aura_additivity_gate.additivity_gate(payload, formats, 1.0), indent=1).encode()
    profile_outputs, printed = [], []
    real_text = digests.JsonProfile.text

    def observe_text(self, value):
        text = real_text(self, value)
        if self is digests.INSERTION_ASCII_INDENT1_LAX:
            profile_outputs.append(text)
        return text

    def observe_print(text):
        printed.append(text)
        builtins.print(text)

    monkeypatch.setattr(digests.JsonProfile, "text", observe_text)
    monkeypatch.setattr(aura_additivity_gate, "print", observe_print, raising=False)
    assert aura_additivity_gate.main([
        "--costs", str(costs), "--assignment", str(assignment), "--measured-kl", "1",
        "--output", str(report),
    ]) == 0
    assert report.read_bytes() == expected
    assert capsysbinary.readouterr().out == expected + b"\n"
    assert any(printed[0] is text for text in profile_outputs), "Print the text from the shared profile."
