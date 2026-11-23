import torch

from prismaquant import digests, index_entropy


def test_index_entropy_report_uses_indent2_profile(tmp_path, monkeypatch, capsysbinary):
    fields = tmp_path / "fields.pt"
    report = tmp_path / "report.json"
    torch.save({"indices": torch.empty(0, dtype=torch.long, device="cpu"), "k": 2}, fields)
    expected = b'''{
  "k": 2,
  "n": 0,
  "n_unique": 0,
  "H": 0.0,
  "redundancy": 2.0,
  "H_conditional": 0.0,
  "conditional_gain": 0.0
}'''
    calls = []
    real_text = digests.JsonProfile.text

    def observe_text(self, value):
        text = real_text(self, value)
        if self is digests.INSERTION_ASCII_INDENT2_LAX:
            calls.append(value)
            return marker + text
        return text

    monkeypatch.setattr(digests.JsonProfile, "text", observe_text)
    for marker in ("", " "):
        calls.clear()
        index_entropy.main([str(fields), "--output", str(report)])
        assert report.read_bytes() == marker.encode("ascii") + expected
        assert capsysbinary.readouterr().out == marker.encode("ascii") + expected + b"\n"
        assert len(calls) == 1, "The shared JSON profile did not get exactly one call."
