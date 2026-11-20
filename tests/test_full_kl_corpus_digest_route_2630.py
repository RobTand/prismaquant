import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from tools import measure_vllm_full_kl as full_kl, serve_fingerprint
from prismaquant import digests


_BASE_OUTCOMES = {
    "text_file": (
        ([[13, 14], [11, 12]], [2, 0], 5, {
            "source": "text_file",
            "path": "corpus.txt",
            "bytes": 14,
            "corpus_sha256": "dc8e24d7545010e5c8c07ce64709b970f8371efe21416c3991caa4c7271c13d4",
            "total_tokens": 5,
        }),
        b'{\n  "corpus": {\n    "source": "text_file",\n    "path": "corpus.txt",\n    "bytes": 14'
        b',\n    "corpus_sha256": "dc8e24d7545010e5c8c07ce64709b970f8371efe21416c3991caa4c7271c13d4'
        b'",\n    "total_tokens": 5\n  }\n}',
    ),
    "datasets": (
        ([[13, 14], [11, 12]], [2, 0], 5, {
            "source": "datasets",
            "cache_dir": "cache",
            "bytes": 17,
            "corpus_sha256": "63c0b48bdd61e2d7548b56b253bdf75a048ad70a620a84ab7fda4234bf417ff6",
            "datasets_fingerprint": "révision-fixe",
            "total_tokens": 5,
        }),
        b'{\n  "corpus": {\n    "source": "datasets",\n    "cache_dir": "cache",\n    "bytes": 17,'
        b'\n    "corpus_sha256": "63c0b48bdd61e2d7548b56b253bdf75a048ad70a620a84ab7fda4234bf417ff6"'
        b',\n    "datasets_fingerprint": "r\\u00e9vision-fixe",\n    "total_tokens": 5\n  }\n}',
    ),
    "missing": {
        "raised": "builtins.FileNotFoundError",
        "text": "[Errno 2] No such file or directory: 'corpus.txt'",
        "cause": None,
    },
    "invalid_utf8": {
        "raised": "builtins.UnicodeDecodeError",
        "text": "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte",
        "cause": None,
    },
    "non_string_row": {
        "raised": "builtins.AttributeError",
        "text": "'int' object has no attribute 'strip'",
        "cause": None,
    },
}

_BASE_FILES = {
    "prismaquant/tessera_runtime": {
        "tessera_serving_runtime_pin.json": {
            "bytes": 2603,
            "sha256": "0e6c4703e20da961d0225f8632a4aee858d4ccff311638123f6a2ca2f25d67ac",
        },
    },
    "tools": {
        "dsv4_wikitext_inputs.py": {
            "bytes": 23806,
            "sha256": "c11176b862278e5e8248f4c6e417c7b27317808e7e475632d9d1c3c300d0f774",
        },
        "full_kl_teacher_payload.py": {
            "bytes": 55452,
            "sha256": "6be0b005ef3977ff49aae774b1a59a93fa2b391d1a5d65cda53c940fcc9195c4",
        },
        "gold_engine_options.py": {
            "bytes": 18196,
            "sha256": "22a7584f07cbcf2490bd0a437a11acbc4c20d6b6258d58b080a63a3d0bc3a2dd",
        },
        "gold_measurement_fidelity.py": {
            "bytes": 6239,
            "sha256": "b068d9e02b95f007491201b19131a5bc97a4cd902ea0c697dcbd50847a753930",
        },
        "measure_vllm_full_kl.py": {
            "bytes": 35436,
            "sha256": "9ea4e7cbc25c08a6e15a971ea5a29f82283799bbf0e13fb0a37216cb0ae32c65",
        },
        "prepare_dsv4_wikitext_inputs.py": {
            "bytes": 8347,
            "sha256": "cc3ff5c8772ed5855238efa6bb676b3807a0e61784e757c0019b22526a533013",
        },
        "prismaquant_source_bootstrap.py": {
            "bytes": 8458,
            "sha256": "b571e52aa7586213781b8798c9f1f330ceed0e4b226fdb98ec9d128cc54230cb",
        },
        "serve_fingerprint.py": {
            "bytes": 103007,
            "sha256": "8e36386743104664c298e350890ee59c066ee5affba7bd5e24bb43c7793e0707",
        },
        "spec_decode_guard.py": {
            "bytes": 3733,
            "sha256": "1170ffdb48660956dee102b7278bd7491adc4e59989d753f9a0229774a593977",
        },
    },
}


@pytest.mark.parametrize("case", _BASE_OUTCOMES)
def test_corpus_bytes_and_owner_calls(monkeypatch, tmp_path, case):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "corpus.txt"
    expected_calls = []
    if case in {"datasets", "non_string_row"}:
        class Dataset(list):
            _fingerprint = "révision-fixe"

        rows = Dataset({"text": row} for row in ("café 😀", "", " \t", " fin "))
        if case == "non_string_row":
            rows.insert(1, {"text": 123})
        else:
            expected_calls = [call.text_sha256hex("café 😀\n\n fin ")]
        monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **kw: rows))
        text_file = None
    else:
        text_file = path.name
        if case == "text_file":
            raw = "\ufeffcafé 😀\x00".encode("utf-8")
            path.write_bytes(raw)
            expected_calls = [call.bytes_sha256hex(raw)]
        elif case == "invalid_utf8":
            path.write_bytes(b"\xff")

    owner = Mock()
    for name in ("bytes_sha256hex", "text_sha256hex"):
        spy = Mock(wraps=getattr(digests, name))
        owner.attach_mock(spy, name)
        monkeypatch.setattr(digests, name, spy)

    def tokenize(text, **options):
        ids = full_kl.torch.tensor([[11, 12, 13, 14, 15]], device="cpu")
        return SimpleNamespace(input_ids=ids)

    kwargs = dict(cache_dir="cache", n_samples=2, seqlen=2, window_seed=42, text_file=text_file)
    expected = _BASE_OUTCOMES[case]
    if case in {"text_file", "datasets"}:
        result = full_kl._load_wikitext_calibration(tokenize, **kwargs)
        assert result == expected[0]
        assert full_kl._strict_json_text({"corpus": result[3]}).encode("utf-8") == expected[1]
    else:
        with pytest.raises((FileNotFoundError, UnicodeDecodeError, AttributeError)) as error:
            full_kl._load_wikitext_calibration(tokenize, **kwargs)
        raised = type(error.value)
        assert f"{raised.__module__}.{raised.__qualname__}" == expected["raised"]
        assert str(error.value) == expected["text"]
        assert error.value.__cause__ is expected["cause"]
    assert owner.mock_calls == expected_calls


def test_gold_producer_identity_bumps_only_this_tool(monkeypatch):
    monkeypatch.setattr(serve_fingerprint, "git_provenance", lambda: {
        "commit": "b97871ab1d425616949ebe365b3ec6489be6e1aa",
        "dirty": False,
    })
    live = serve_fingerprint.gold_producer_identity("measure_vllm_full_kl")["source_files"]
    base = {
        f"{directory}/{basename}": entry
        for directory, files in _BASE_FILES.items()
        for basename, entry in files.items()
    }
    assert live.keys() == base.keys()
    target = "tools/measure_vllm_full_kl.py"
    assert live.pop(target)["sha256"] != base.pop(target)["sha256"]
    assert live == base
