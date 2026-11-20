import sys
from types import SimpleNamespace

import pytest

from prismaquant import digests
from tools import measure_vllm_wikitext_ppl as ppl, serve_fingerprint


_BASE_OUTCOMES = {'text_file': (([11, 12, 13, 14],
                {'fingerprint': 'file:sha256:dc8e24d7545010e5c8c07ce64709b970f8371efe21416c3991caa4c7271c13d4',
                 'corpus_sha256': 'dc8e24d7545010e5c8c07ce64709b970f8371efe21416c3991caa4c7271c13d4',
                 'total_tokens': 5}),
               b'{"corpus_construction":{"join_separator":"\\n\\n","normalization":"none","row_filter":"inc'
               b'lude iff bool(text.strip()); preserve text verbatim"},"dataset":{"config":"wikitext-2-ra'
               b'w-v1","corpus_sha256":"dc8e24d7545010e5c8c07ce64709b970f8371efe21416c3991caa4c7271c13d4"'
               b',"fingerprint":"file:sha256:dc8e24d7545010e5c8c07ce64709b970f8371efe21416c3991caa4c7271c'
               b'13d4","name":"wikitext","revision":"b08601e04326c79dfdd32d625aee71d232d685c3","split":"t'
               b'est"},"schema":"prismaquant.wikitext_ppl_calibration/1","scoring":{"chunk_starts":[0,2],'
               b'"chunk_token_counts":[2,2],"chunking":"nonoverlapping_contiguous/v1","detokenize":false,'
               b'"max_tokens":1,"n_tokens_scored":2,"positions":"within_each_chunk_positions_1_through_N_'
               b'minus_1","prompt_logprobs":1,"seqlen":2,"temperature":0.0},"token_selection":{"digest_en'
               b'coding":"canonical_json_integer_array/v1","n_tokens_available":5,"n_tokens_requested":4,'
               b'"selected_token_count":4,"strategy":"contiguous_prefix_after_full_corpus_tokenization/v1'
               b'","token_ids_sha256":"15225829a74894d4d186cc6d46095a2847e22d6223b6ec9ecc74e9e1d043175b"}'
               b',"tokenizer":{"add_special_tokens":false,"identity_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaa'
               b'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","trust_remote_code":true}}',
               'ec1acc371b0bc28db93f30f901840f6255ac266499b61a61642862c7552af841'),
 'datasets': (([11, 12, 13, 14],
               {'fingerprint': 'révision-fixe',
                'corpus_sha256': '63c0b48bdd61e2d7548b56b253bdf75a048ad70a620a84ab7fda4234bf417ff6',
                'total_tokens': 5}),
              b'{"corpus_construction":{"join_separator":"\\n\\n","normalization":"none","row_filter":"inc'
              b'lude iff bool(text.strip()); preserve text verbatim"},"dataset":{"config":"wikitext-2-ra'
              b'w-v1","corpus_sha256":"63c0b48bdd61e2d7548b56b253bdf75a048ad70a620a84ab7fda4234bf417ff6"'
              b',"fingerprint":"r\xc3\xa9vision-fixe","name":"wikitext","revision":"b08601e04326c79dfdd32d'
              b'625aee71d232d685c3","split":"test"},"schema":"prismaquant.wikitext_ppl_calibration/1","s'
              b'coring":{"chunk_starts":[0,2],"chunk_token_counts":[2,2],"chunking":"nonoverlapping_cont'
              b'iguous/v1","detokenize":false,"max_tokens":1,"n_tokens_scored":2,"positions":"within_eac'
              b'h_chunk_positions_1_through_N_minus_1","prompt_logprobs":1,"seqlen":2,"temperature":0.0}'
              b',"token_selection":{"digest_encoding":"canonical_json_integer_array/v1","n_tokens_availa'
              b'ble":5,"n_tokens_requested":4,"selected_token_count":4,"strategy":"contiguous_prefix_aft'
              b'er_full_corpus_tokenization/v1","token_ids_sha256":"15225829a74894d4d186cc6d46095a2847e2'
              b'2d6223b6ec9ecc74e9e1d043175b"},"tokenizer":{"add_special_tokens":false,"identity_sha256"'
              b':"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","trust_remote_code":'
              b'true}}',
              '60aa63b7d8fbb4ca83df98cada4af1e468486202daddcd625e81ae65410f74e4'),
 'missing': {'raised': 'builtins.FileNotFoundError',
             'text': "[Errno 2] No such file or directory: 'corpus.txt'",
             'cause': None},
 'invalid_utf8': {'raised': 'builtins.UnicodeDecodeError',
                  'text': "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte",
                  'cause': None}}

_BASE_FILES = {'prismaquant/tessera_runtime': {'tessera_serving_runtime_pin.json': {'bytes': 2603,
                                                                      'sha256': '0e6c4703e20da961d0225f8632a4aee858d4ccff311638123f6a2ca2f25d67ac'}},
 'tools': {'dsv4_wikitext_inputs.py': {'bytes': 23806,
                                       'sha256': 'c11176b862278e5e8248f4c6e417c7b27317808e7e475632d9d1c3c300d0f774'},
           'full_kl_teacher_payload.py': {'bytes': 55452,
                                          'sha256': '6be0b005ef3977ff49aae774b1a59a93fa2b391d1a5d65cda53c940fcc9195c4'},
           'gold_engine_options.py': {'bytes': 18196,
                                      'sha256': '22a7584f07cbcf2490bd0a437a11acbc4c20d6b6258d58b080a63a3d0bc3a2dd'},
           'gold_measurement_fidelity.py': {'bytes': 6239,
                                            'sha256': 'b068d9e02b95f007491201b19131a5bc97a4cd902ea0c697dcbd50847a753930'},
           'measure_vllm_wikitext_ppl.py': {'bytes': 23899,
                                            'sha256': '4dc767ab5535432cc5a90ae05c3268476fed17c4caf2581de1f8b5637a606847'},
           'prepare_dsv4_wikitext_inputs.py': {'bytes': 8347,
                                               'sha256': 'cc3ff5c8772ed5855238efa6bb676b3807a0e61784e757c0019b22526a533013'},
           'prismaquant_source_bootstrap.py': {'bytes': 8458,
                                               'sha256': 'b571e52aa7586213781b8798c9f1f330ceed0e4b226fdb98ec9d128cc54230cb'},
           'serve_fingerprint.py': {'bytes': 103007,
                                    'sha256': '8e36386743104664c298e350890ee59c066ee5affba7bd5e24bb43c7793e0707'},
           'spec_decode_guard.py': {'bytes': 3733,
                                    'sha256': '1170ffdb48660956dee102b7278bd7491adc4e59989d753f9a0229774a593977'}}}


@pytest.fixture
def owner_calls(monkeypatch):
    calls = []
    bytes_sha256hex = digests.bytes_sha256hex
    text_sha256hex = digests.text_sha256hex

    def hash_bytes(raw):
        calls.append(("bytes", raw))
        return bytes_sha256hex(raw)

    def hash_text(text):
        calls.append(("text", text))
        return text_sha256hex(text)

    monkeypatch.setattr(digests, "bytes_sha256hex", hash_bytes)
    monkeypatch.setattr(digests, "text_sha256hex", hash_text)
    return calls


@pytest.mark.parametrize("source", ["text_file", "datasets"])
def test_corpus_bytes_and_owner_calls(monkeypatch, tmp_path, owner_calls, source):
    kwargs = {"cache_dir": "cache", "split": "test", "n_tokens": 4}
    if source == "text_file":
        text = "\ufeffcafé 😀\x00"
        raw = text.encode("utf-8")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "corpus.txt").write_bytes(raw)
        kwargs["text_file"] = "corpus.txt"
        expected_calls = [("bytes", raw), ("text", text)]
    else:
        class Dataset(list):
            _fingerprint = "révision-fixe"

        rows = Dataset({"text": row} for row in ("café 😀", 123, "", " \t", " fin "))
        monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **kw: rows))
        text = "café 😀\n\n fin "
        expected_calls = [("text", text)]

    def tokenize(observed, **options):
        assert observed == text
        assert options == {"return_tensors": "pt", "add_special_tokens": False}
        return SimpleNamespace(input_ids=ppl.torch.tensor([[11, 12, 13, 14, 15]], device="cpu"))

    expected, encoded, sha256 = _BASE_OUTCOMES[source]
    result = ppl._load_ids(tokenize, **kwargs)
    corpus_calls = list(owner_calls)
    assert result == expected
    ids, evidence = result
    contract = ppl._ppl_calibration_contract(
        args=SimpleNamespace(n_tokens=4, seqlen=2, split="test"),
        ids=ids, dataset_evidence=evidence, tokenizer_identity_sha256="a" * 64,
        chunks=[[11, 12], [13, 14]], n_tokens_scored=2,
    )
    assert digests.DIRECT_UTF8_STRICT.encoded(contract) == encoded
    assert ppl._canonical_sha256(contract) == sha256
    assert corpus_calls == expected_calls


@pytest.mark.parametrize("source, error_type", [
    ("missing", FileNotFoundError), ("invalid_utf8", UnicodeDecodeError),
], ids=["missing", "invalid_utf8"])
def test_corpus_refusal_before_owner_calls(monkeypatch, tmp_path, owner_calls, source, error_type):
    monkeypatch.chdir(tmp_path)
    if source == "invalid_utf8":
        (tmp_path / "corpus.txt").write_bytes(b"\xff")

    def tokenize(*args, **kwargs):
        pytest.fail("The tokenizer received a refused corpus.")

    with pytest.raises(error_type) as error:
        ppl._load_ids(tokenize, cache_dir="cache", split="test", n_tokens=2, text_file="corpus.txt")
    raised = type(error.value)
    assert {
        "raised": f"{raised.__module__}.{raised.__qualname__}",
        "text": str(error.value),
        "cause": error.value.__cause__,
    } == _BASE_OUTCOMES[source]
    assert not owner_calls


def test_gold_producer_identity_bumps_only_this_tool(monkeypatch):
    monkeypatch.setattr(serve_fingerprint, "git_provenance", lambda: {
        "commit": "b97871ab1d425616949ebe365b3ec6489be6e1aa", "dirty": False, "tree": None,
    })
    live = serve_fingerprint.gold_producer_identity("measure_vllm_wikitext_ppl")["source_files"]
    base = {f"{directory}/{name}": entry for directory, files in _BASE_FILES.items()
            for name, entry in files.items()}
    assert live.keys() == base.keys()
    tool = "tools/measure_vllm_wikitext_ppl.py"
    assert {name: entry for name, entry in live.items() if name != tool} == {
        name: entry for name, entry in base.items() if name != tool
    }
    assert live[tool]["sha256"] != base[tool]["sha256"]
