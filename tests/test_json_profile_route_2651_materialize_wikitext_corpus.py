import hashlib
import json
import sys
from types import SimpleNamespace

from prismaquant import digests
from tools import materialize_wikitext_corpus


def test_corpus_index_uses_indent1_profile(tmp_path, monkeypatch, capsys):
    out_dir = tmp_path / "corpus"
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(sys, "argv", [
        "materialize_wikitext_corpus.py", "--out-dir", str(out_dir),
        "--cache-dir", str(cache_dir), "--split", "test", "--revision", "rév",
    ])
    rows = [{"text": "café"}, {"text": " \t"}, {"text": "Wiki"}]
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **kw: rows))
    corpus = b"caf\xc3\xa9\n\nWiki"
    path = out_dir / "wikitext-2-raw-v1.test.txt"
    digest = hashlib.sha256(corpus).hexdigest()
    expected = b'''{
 "test": {
  "path": %s,
  "bytes": 11,
  "sha256": "%s",
  "dataset": "wikitext",
  "config": "wikitext-2-raw-v1",
  "revision": "r\\u00e9v",
  "split": "test",
  "datasets_fingerprint": null,
  "n_rows_kept": 2
 }
}''' % (json.dumps(str(path)).encode("ascii"), digest.encode("ascii"))
    real_text = digests.JsonProfile.text
    calls = []

    def observe_text(self, value):
        text = real_text(self, value)
        if self is digests.INSERTION_ASCII_INDENT1_LAX:
            calls.append(self)
            return prefix.decode("ascii") + text
        return text

    monkeypatch.setattr(digests.JsonProfile, "text", observe_text)
    for prefix in (b"", b" "):
        calls.clear()
        assert materialize_wikitext_corpus.main() == 0
        assert path.read_bytes() == corpus
        assert capsys.readouterr().out == f"[corpus] test: 11 bytes  sha256 {digest[:16]}  rows 2\n"
        assert (out_dir / "corpus_index.json").read_bytes() == prefix + expected
        assert len(calls) == 1, "Call the shared JSON profile once."
