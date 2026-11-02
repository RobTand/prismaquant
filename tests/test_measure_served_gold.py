"""The served-lane gold tool: what it computes, and what it refuses.

Before this tool the only gold measurement tools were DSv4's, which construct
an in-process `LLM` under a contract pinning `tokenizer_mode="deepseek_v4"`.
Every other lane therefore measured with run-local scripts that emitted a bare
number and no identity -- no serve fingerprint, no producer commit, no
spec-decode state, no workload contract -- which is exactly the evidence a ship
record binds. These tests pin the arithmetic and, more importantly, the
refusals: a gold number that cannot say which stack produced it is the failure
mode the whole R15 fingerprint mechanism exists to prevent.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tools import measure_served_gold as gold


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------
def test_kl_is_zero_when_the_two_arms_agree():
    row = {"1": math.log(0.7), "2": math.log(0.2), "3": math.log(0.1)}
    assert gold._tail_logprob(row, vocab_size=3) == pytest.approx(
        min(row.values()))
    divergence = sum(
        math.exp(lp) * (lp - row[token]) for token, lp in row.items())
    assert divergence == pytest.approx(0.0, abs=1e-12)


def test_the_untabulated_tail_uses_a_clamped_max_entropy_estimate():
    """Spread residual mass uniformly, capped at the observed K-th value.

    This max-entropy estimate can be above or below an unknown token's true
    probability. It does not bound the true full-vocabulary KL.
    """
    row = {"1": math.log(0.5), "2": math.log(0.3)}      # 0.2 mass unaccounted
    tail = gold._tail_logprob(row, vocab_size=1002)
    assert tail == pytest.approx(math.log(0.2 / 1000))
    assert tail < min(row.values()), "the tail must sit below the K-th entry"

    # A row that already accounts for all its mass has no residual to spread.
    full = {"1": math.log(0.6), "2": math.log(0.4)}
    assert gold._tail_logprob(full, vocab_size=10) == pytest.approx(
        min(full.values()))


def test_coverage_is_recomputed_from_the_teacher_not_declared():
    """Top-K KL is only meaningful with its truncation reported alongside."""
    row = {"1": math.log(0.4), "2": math.log(0.3)}
    assert sum(math.exp(lp) for lp in row.values()) == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def _dump(tmp_path, name, *, corpus_text="one two one two", **overrides):
    corpus = tmp_path / "fixture-corpus.txt"
    corpus.write_text(corpus_text)
    tokenizer = tmp_path / "fixture-tokenizer"
    tokenizer.mkdir(exist_ok=True)
    (tokenizer / "tokenizer.json").write_text('{"synthetic_test_tokenizer":true}')
    contract = gold._calibration_contract(
        corpus, tokenizer, [1, 2, 1, 2], seqlen=2, n_tokens_requested=4)
    payload = {
        "schema": gold.SERVED_DUMP_SCHEMA,
        "model": "qwen",
        "artifact_dir": str(tmp_path / "exported"),
        "prompt_top_k": 1024,
        "seqlen": 2,
        "n_samples": 2,
        "n_positions": 2,
        "vocab_size": 1000,
        "spec_decode_detected": False,
        "serve_fingerprint": "a" * 64,
        "performance_stack_fingerprint": "b" * 64,
        "serve_manifest": {"schema": gold.SERVE_MANIFEST_SCHEMA},
        "git_commit": "c" * 40,
        "gold_producer_identity": {"git_commit": "c" * 40},
        "calibration_contract": contract,
        "calibration_contract_sha256": gold._canonical_sha256(contract),
        "positions": [
            {"1": math.log(0.9), "2": math.log(0.1)},
            {"1": math.log(0.6), "2": math.log(0.4)},
        ],
    }
    payload.update(overrides)
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def _run_kl(tmp_path, teacher, student, monkeypatch):
    monkeypatch.setattr(gold, "_producer_identity",
                        lambda: {"git_commit": "e" * 40})
    out = tmp_path / "record.json"
    gold.main(["kl", "--teacher", str(teacher), "--student", str(student),
               "--out", str(out)])
    return json.loads(out.read_text())


def test_two_arms_dumped_at_different_top_k_are_refused(tmp_path, monkeypatch):
    """Their truncation floors differ, so the delta is not the quantization."""
    teacher = _dump(tmp_path, "t.json")
    student = _dump(tmp_path, "s.json", prompt_top_k=20)
    with pytest.raises(SystemExit, match="different top-K"):
        _run_kl(tmp_path, teacher, student, monkeypatch)


def test_two_arms_that_scored_different_text_are_refused(tmp_path, monkeypatch):
    """KL is only the quantization when both models saw identical inputs."""
    teacher = _dump(tmp_path, "t.json")
    student = _dump(tmp_path, "s.json", corpus_text="different input text")
    with pytest.raises(SystemExit, match="different text/geometry"):
        _run_kl(tmp_path, teacher, student, monkeypatch)


def test_a_record_carries_every_field_the_ship_slot_binds(tmp_path, monkeypatch):
    """`shipcard_cli fill` lifts these by name; a missing one records
    `passed=false`, and `verify` refuses the card."""
    teacher = _dump(tmp_path, "t.json")
    student = _dump(tmp_path, "s.json")
    record = _run_kl(tmp_path, teacher, student, monkeypatch)

    assert record["score_positions"] == "all", (
        "'final' is the last-token hook screen: triage only, never promotion")
    assert record["spec_decode_detected"] is False
    assert len(record["serve_fingerprint"]) == 64
    assert len(record["git_commit"]) == 40
    assert record["n_positions"] > 0
    assert math.isfinite(record["kl_mean"])
    assert math.isfinite(record["kl_confident_mean"])
    # Identical arms: the divergence is zero and agreement is total.
    assert record["kl_mean"] == pytest.approx(0.0, abs=1e-12)
    assert record["top1_agreement_all"] == 1.0
    # The truncation is reported, never implied.
    assert record["prompt_top_k"] == 1024
    assert 0.0 < record["topk_coverage_min"] <= 1.0
    assert record["student_tail_model"] == "uniform_over_untabulated_vocabulary"
    # Only the position whose teacher top-1 exceeds 0.5 twice over is
    # "confident"; 0.6 clears it and 0.9 clears it, so both count here.
    assert record["n_confident"] == 2


def test_the_teacher_arm_is_bound_by_digest_not_by_assertion(
    tmp_path, monkeypatch,
):
    """A KL number names two serves. The second one has to be identified."""
    teacher = _dump(tmp_path, "t.json")
    student = _dump(tmp_path, "s.json", serve_fingerprint="9" * 64)
    record = _run_kl(tmp_path, teacher, student, monkeypatch)

    evidence = record["teacher_evidence"]
    assert evidence["schema"] == gold.TEACHER_ARM_SCHEMA
    assert evidence["serve_fingerprint"] == "a" * 64
    assert record["serve_fingerprint"] == "9" * 64, (
        "the record's own fingerprint is the STUDENT's serve")
    assert len(evidence["dump_sha256"]) == 64


def test_a_missing_serve_manifest_is_refused_not_warned(tmp_path):
    """A gold number with no fingerprint cannot be compared to anything."""
    artifact = tmp_path / "exported"
    artifact.mkdir()
    with pytest.raises(SystemExit, match="no .*serve_manifest.json"):
        gold._load_serve_manifest(artifact)


def test_a_manifest_without_a_fingerprint_is_refused(tmp_path):
    artifact = tmp_path / "exported"
    artifact.mkdir()
    (artifact / "serve_manifest.json").write_text(
        json.dumps({"schema": gold.SERVE_MANIFEST_SCHEMA}))
    with pytest.raises(SystemExit, match="carries no serve fingerprint"):
        gold._load_serve_manifest(artifact)


def test_a_stale_manifest_is_refused(tmp_path, monkeypatch):
    """A manifest describing a DIFFERENT serve is worse than none: it attests
    the wrong stack while looking like evidence."""
    manifest = {
        "schema": gold.SERVE_MANIFEST_SCHEMA,
        "serve_fingerprint": "a" * 64,
        "served_model_name": "some-other-model",
        "speculative_config": None,
    }
    with pytest.raises(SystemExit, match="the manifest is stale"):
        gold._check_manifest_describes_this_serve(
            manifest, base_url="http://localhost:8000",
            served_model_name="qwen")


def test_a_manifest_recording_speculative_decoding_is_refused(tmp_path):
    """With --speculative-config serving, /v1/completions logprobs are the
    DRAFT model's NLL (CLAUDE.md operational landmine)."""
    manifest = {
        "schema": gold.SERVE_MANIFEST_SCHEMA,
        "serve_fingerprint": "a" * 64,
        "served_model_name": "qwen",
        "speculative_config": {"model": "draft"},
    }
    with pytest.raises(SystemExit, match="speculative config"):
        gold._check_manifest_describes_this_serve(
            manifest, base_url="http://localhost:8000",
            served_model_name="qwen")


@pytest.mark.parametrize("observed", [None, True])
def test_spec_decode_that_is_not_a_clean_false_is_refused(monkeypatch, observed):
    """`None` means 'could not inspect', which is what the original trap
    looked like -- an unverified negative is not a negative."""
    monkeypatch.setattr(gold, "spec_decode_from_metrics", lambda url: observed)
    with pytest.raises(SystemExit, match="spec_decode_detected"):
        gold._refuse_spec_decode("http://localhost:8000")


def test_the_tool_is_a_registered_gold_producer():
    """`gold_producer_identity` refuses an unknown tool, and refuses a dirty
    tree -- which is what binds the number to reviewable source."""
    from tools.serve_fingerprint import _GOLD_PRODUCER_TOOL_FILES

    files = _GOLD_PRODUCER_TOOL_FILES[gold.MEASUREMENT_TOOL]
    assert "tools/measure_served_gold.py" in files
    root = Path(__file__).resolve().parents[1]
    for name in files:
        assert (root / name).is_file(), name


# ---------------------------------------------------------------------------
# Digest owner routing (#2629)
# ---------------------------------------------------------------------------
# Inline copies of the client's original recipes, before routing. They are the
# byte oracle the routed sites must reproduce.
import hashlib  # noqa: E402
import types  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_GOLD_FILE = "tools/measure_served_gold.py"
_MIGRATION = "migrated-by-pq2629-2026-11-02"


def _original_sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _original_canonical_sha256(payload: object) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _original_calib_ids_sha256(token_ids: list[int]) -> str:
    return hashlib.sha256(
        ",".join(str(i) for i in token_ids).encode("utf-8")
    ).hexdigest()


def _original_calibration_contract(corpus, tokenizer_dir, token_ids, *, seqlen,
                                   n_tokens_requested):
    starts, _ = gold._window_geometry(len(token_ids), seqlen)
    if (type(n_tokens_requested) is not int or n_tokens_requested <= 0
            or len(token_ids) > n_tokens_requested):
        raise ValueError("invalid calibration token request")
    tokenizer_files = {
        name: _original_sha256_file(tokenizer_dir / name)
        for name in ("tokenizer.json", "tokenizer_config.json")
        if (tokenizer_dir / name).is_file()
    }
    return {
        "schema": gold.CALIBRATION_SCHEMA,
        "corpus_path": str(corpus.resolve()),
        "corpus_sha256": _original_sha256_file(corpus),
        "corpus_bytes": corpus.stat().st_size,
        "tokenizer_files_sha256": tokenizer_files,
        "add_special_tokens": False,
        "n_tokens_requested": n_tokens_requested,
        "n_tokens_selected": len(token_ids),
        "seqlen": seqlen,
        "n_windows": len(starts),
        "window_overlap": 0,
        "calib_ids_sha256": _original_calib_ids_sha256(token_ids),
    }


def _raised(fn, *args, **kwargs) -> BaseException:
    try:
        fn(*args, **kwargs)
    except BaseException as exc:  # noqa: BLE001 - compared field by field
        return exc
    raise AssertionError(f"{fn!r} raised nothing")


def _assert_same_exception(routed: BaseException, original: BaseException) -> None:
    assert type(routed) is type(original)
    assert str(routed) == str(original)
    assert routed.args == original.args
    assert type(routed.__cause__) is type(original.__cause__)
    assert repr(routed.__cause__) == repr(original.__cause__)
    assert type(routed.__context__) is type(original.__context__)
    assert repr(routed.__context__) == repr(original.__context__)
    assert routed.__suppress_context__ == original.__suppress_context__
    assert getattr(routed, "__notes__", None) == getattr(original, "__notes__", None)
    if isinstance(original, OSError):
        assert routed.errno == original.errno
        assert routed.filename == original.filename


_FILE_SENTINEL = "f" * 64
_TEXT_SENTINEL = "7" * 64
_BYTES_SENTINEL = "b" * 64
_ASCII_SENTINEL = "a" * 64
_UTF8_SENTINEL = "0" * 64


def _sentinel_owner(calls: list) -> types.SimpleNamespace:
    """A fake digest owner whose every route returns a distinct sentinel."""
    def file_sha256hex(path, *, block_size=None):
        calls.append(("file", Path(path), block_size))
        return _FILE_SENTINEL

    def text_sha256hex(text):
        calls.append(("text", text))
        return _TEXT_SENTINEL

    def bytes_sha256hex(data):
        calls.append(("bytes", bytes(data)))
        return _BYTES_SENTINEL

    def ascii_sha256(value):
        calls.append(("ascii", value))
        return _ASCII_SENTINEL

    def utf8_sha256(value):
        calls.append(("utf8", value))
        return _UTF8_SENTINEL

    return types.SimpleNamespace(
        file_sha256hex=file_sha256hex,
        text_sha256hex=text_sha256hex,
        bytes_sha256hex=bytes_sha256hex,
        DIRECT_ASCII_STRICT=types.SimpleNamespace(sha256=ascii_sha256),
        DIRECT_UTF8_STRICT=types.SimpleNamespace(sha256=utf8_sha256),
    )


def _seed_calibration_inputs(root: Path, *, tokenizer=("tokenizer.json",
                                                       "tokenizer_config.json")):
    corpus = root / "corpus.txt"
    corpus.write_text(
        "= Valkyrie =\nh\u00e9llo \u2713 \u00fcber na\u00efve \u4e2d\u6587 "
        "\U0001F600 \U0001D11E\n" * 3, encoding="utf-8")
    tokenizer_dir = root / "tok"
    tokenizer_dir.mkdir()
    payloads = {
        "tokenizer.json": '{"model": {"vocab": {"h\u00e9": 1, "\u2713": 2}}}',
        "tokenizer_config.json": '{"add_bos_token": false, "x": "\U0001F600"}',
    }
    for name in tokenizer:
        (tokenizer_dir / name).write_text(payloads[name], encoding="utf-8")
    return corpus, tokenizer_dir


def test_sha256_file_routes_to_owner_byte_identically(tmp_path, monkeypatch):
    unicode_file = tmp_path / "unicode.txt"
    unicode_file.write_text(
        "h\u00e9llo \u2713 \u4e2d\u6587\n\U0001F600\U0001D11E\n", encoding="utf-8")
    nul_file = tmp_path / "nul.bin"
    nul_file.write_bytes(b"a\x00b\x00\x00\xff\xfe" * 1000 + b"\x00")
    empty_file = tmp_path / "empty.bin"
    empty_file.write_bytes(b"")
    for path in (unicode_file, nul_file, empty_file):
        assert gold._sha256_file(path) == _original_sha256_file(path), path

    for path in (tmp_path / "absent.bin", tmp_path):
        _assert_same_exception(_raised(gold._sha256_file, path),
                               _raised(_original_sha256_file, path))

    # Routing proof: the digest comes from the owner's file hasher.
    calls: list = []
    monkeypatch.setattr(gold, "_DIGESTS", _sentinel_owner(calls), raising=False)
    assert gold._sha256_file(nul_file) == _FILE_SENTINEL
    assert calls == [("file", nul_file, 8 * 1024 * 1024)]


def test_canonical_sha256_routes_to_direct_ascii_strict_byte_identically(monkeypatch):
    payloads = [
        {"text": "h\u00e9llo\u2713", "astral": "\U0001F600\U0001D11E"},
        {"b": [1, 2.5, None, True], "a": {"z": {"y": "\u4e2d"}, "c": []}},
        {3: "three", 1: "one", 20: "twenty"},
        {"nested": {"k\u00e9y": {"\U0001F600": [{"deep": "\u2713"}]}}},
        ["plain", 0, -1, 1e300],
        "\u00e9",
    ]
    for payload in payloads:
        assert gold._canonical_sha256(payload) == _original_canonical_sha256(payload)

    refused = [
        {"x": float("nan")},
        {"x": float("inf")},
        {"x": [float("-inf")]},
        {"x": {1, 2}},
        {"x": object()},
        {(1, 2): "tuple key"},
    ]
    for payload in refused:
        _assert_same_exception(_raised(gold._canonical_sha256, payload),
                               _raised(_original_canonical_sha256, payload))

    # Routing proof: the ASCII-escaped strict profile, never the UTF-8 twin.
    calls: list = []
    monkeypatch.setattr(gold, "_DIGESTS", _sentinel_owner(calls), raising=False)
    assert gold._canonical_sha256(payloads[0]) == _ASCII_SENTINEL
    assert calls == [("ascii", payloads[0])]


def test_calib_ids_digest_routes_to_owner_keeping_client_join(tmp_path, monkeypatch):
    corpus, tokenizer_dir = _seed_calibration_inputs(tmp_path)
    cases = [[], [-1, -2, 0, -151_000], [2**63, 10**30, 151_000, 0, 7]]
    for token_ids in cases:
        record = gold._calibration_contract(
            corpus, tokenizer_dir, token_ids, seqlen=2,
            n_tokens_requested=max(len(token_ids), 1))
        assert record["calib_ids_sha256"] == _original_calib_ids_sha256(token_ids)

    # Routing proof: the client joins, the owner hashes the joined text.
    calls: list = []
    monkeypatch.setattr(gold, "_DIGESTS", _sentinel_owner(calls), raising=False)
    record = gold._calibration_contract(
        corpus, tokenizer_dir, [-1, -2, 0], seqlen=2, n_tokens_requested=3)
    assert record["calib_ids_sha256"] == _TEXT_SENTINEL
    assert ("text", "-1,-2,0") in calls


def test_calibration_contract_matches_original_recipe_record(tmp_path, monkeypatch):
    token_ids = [5, 17, -3, 2**40, 9, 11, 13]
    corpus, tokenizer_dir = _seed_calibration_inputs(tmp_path)
    kwargs = dict(seqlen=3, n_tokens_requested=8)
    routed = gold._calibration_contract(corpus, tokenizer_dir, token_ids, **kwargs)
    original = _original_calibration_contract(corpus, tokenizer_dir, token_ids, **kwargs)
    assert list(routed) == list(original)
    assert json.dumps(routed) == json.dumps(original)
    assert gold._canonical_sha256(routed) == _original_canonical_sha256(original)
    assert set(routed["tokenizer_files_sha256"]) == {
        "tokenizer.json", "tokenizer_config.json"}

    # A missing tokenizer file is silently omitted, never refused.
    for keep in (("tokenizer_config.json",), ("tokenizer.json",), ()):
        partial = tmp_path / ("partial-" + "-".join(keep or ("none",)))
        partial.mkdir()
        part_corpus, part_tok = _seed_calibration_inputs(partial, tokenizer=keep)
        routed = gold._calibration_contract(part_corpus, part_tok, token_ids, **kwargs)
        original = _original_calibration_contract(
            part_corpus, part_tok, token_ids, **kwargs)
        assert json.dumps(routed) == json.dumps(original)
        assert sorted(routed["tokenizer_files_sha256"]) == sorted(keep)

    # A missing corpus is refused exactly as before.
    missing = tmp_path / "no-such-corpus.txt"
    _assert_same_exception(
        _raised(gold._calibration_contract, missing, tokenizer_dir, token_ids, **kwargs),
        _raised(_original_calibration_contract, missing, tokenizer_dir, token_ids,
                **kwargs))

    # Argument validation is unchanged.
    for ids, extra in (
        (token_ids, dict(seqlen=3, n_tokens_requested=0)),
        (token_ids, dict(seqlen=3, n_tokens_requested=6)),
        (token_ids, dict(seqlen=3, n_tokens_requested=8.0)),
        (token_ids, dict(seqlen=1, n_tokens_requested=8)),
        (token_ids, dict(seqlen=3.0, n_tokens_requested=8)),
    ):
        _assert_same_exception(
            _raised(gold._calibration_contract, corpus, tokenizer_dir, ids, **extra),
            _raised(_original_calibration_contract, corpus, tokenizer_dir, ids,
                    **extra))

    # Routing proof: every digest in the record comes from the owner.
    calls: list = []
    monkeypatch.setattr(gold, "_DIGESTS", _sentinel_owner(calls), raising=False)
    routed = gold._calibration_contract(corpus, tokenizer_dir, token_ids, **kwargs)
    assert routed["corpus_sha256"] == _FILE_SENTINEL
    assert set(routed["tokenizer_files_sha256"].values()) == {_FILE_SENTINEL}
    assert routed["calib_ids_sha256"] == _TEXT_SENTINEL


def test_gold_client_has_no_package_import_and_loads_owner_by_path(tmp_path, monkeypatch):
    import ast
    import sys

    tool = _ROOT / _GOLD_FILE
    source = tool.read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source, str(tool))):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module)
    assert not {n for n in imported
                if n == "prismaquant" or n.startswith("prismaquant.")}, imported
    assert "prismaquant_source_bootstrap" not in imported
    assert "activate_prismaquant_source" not in source
    assert "hashlib" not in imported
    # Module level only: _tokenize lazily imports transformers, as it always did.
    top_level: set[str] = set()
    for node in ast.parse(source, str(tool)).body:
        if isinstance(node, ast.Import):
            top_level |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            top_level.add(node.module)
    assert not {n for n in top_level if n.split(".")[0] in ("torch", "transformers")}

    owner = getattr(gold, "_DIGESTS", None)
    assert owner is not None, "no file-path-loaded digest owner"
    assert Path(owner.__file__).resolve() == _ROOT / "prismaquant" / "digests.py"
    assert owner.__name__ != "prismaquant.digests"
    assert not owner.__name__.startswith("prismaquant")
    assert sys.modules.get(owner.__name__) is owner

    # The teacher dump digest goes through the same loaded owner.
    calls: list = []
    monkeypatch.setattr(gold, "_DIGESTS", _sentinel_owner(calls), raising=False)
    teacher = tmp_path / "teacher.json"
    teacher.write_bytes(b'{"schema": "unsupported"}')
    student = tmp_path / "student.json"
    student.write_bytes(b'{"schema": "unsupported"}')
    args = types.SimpleNamespace(teacher=str(teacher), student=str(student),
                                 out=str(tmp_path / "out.json"))
    with pytest.raises(SystemExit, match="unsupported schema"):
        gold._cmd_kl(args)
    assert ("bytes", b'{"schema": "unsupported"}') in calls

    # A missing owner is refused explicitly; no hashlib fallback is built.
    before = dict(sys.modules)
    with pytest.raises(RuntimeError, match="digests.py"):
        gold._load_digest_owner(tmp_path / "prismaquant" / "digests.py")
    assert sys.modules.get(owner.__name__) is owner
    assert set(sys.modules) <= set(before) | {"ast"}


def test_gold_client_digest_sites_reconciled_in_baseline_and_census():
    sites = {
        "hashlib:tools/measure_served_gold.py::_calibration_contract": "text_sha256hex",
        "hashlib:tools/measure_served_gold.py::_canonical_sha256": "DIRECT_ASCII_STRICT",
        "hashlib:tools/measure_served_gold.py::_sha256_file": "file_sha256hex",
        "sorted-json:tools/measure_served_gold.py::_canonical_sha256": "DIRECT_ASCII_STRICT",
    }
    baseline = json.loads(
        (_ROOT / "tests/fixtures/duplication_baseline.json").read_text(encoding="utf-8"))
    assert not set(sites) & set(baseline["primitive_digest_sites"])
    # The thin client wrappers remain, so their same-name entries stay.
    helpers = baseline["same_name_helpers"]
    assert _GOLD_FILE in helpers["_canonical_sha256"]
    assert _GOLD_FILE in helpers["_sha256_file"]

    census = json.loads((_ROOT / "docs/audits/digest_site_census_pq1301_2026-10-04.json")
                        .read_text(encoding="utf-8"))
    rows = {row["site"]: row for row in census["rows"] if row["site"] in sites}
    assert set(rows) == set(sites)
    for site, verdict in sites.items():
        row = rows[site]
        assert row["disposition"] == _MIGRATION, site
        assert row["route_verdict"] == verdict, site
        assert row["migrated_calls"], site
        for call in row["migrated_calls"]:
            assert call["owner"].startswith("prismaquant.digests."), call
            assert "hashlib" not in call["routed_expression"], call
            assert "json.dumps" not in call["routed_expression"], call
        assert "file path" in row["note"], site
        assert "#2629" in row["note"], site

    decision = (_ROOT / "docs/audits/digest_reconciliation_1301.md").read_text(
        encoding="utf-8")
    assert "## Gold client digest owner (#2629)" in decision
    section = decision.split("## Gold client digest owner (#2629)", 1)[1]
    for phrase in ("spec_from_file_location", "DIRECT_ASCII_STRICT",
                   "DIRECT_UTF8_STRICT", "_GOLD_PRODUCER", "unchanged"):
        assert phrase in section, phrase

    changelog = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    head = changelog.split("\n## ", 2)[1]
    assert "#2629" in head
