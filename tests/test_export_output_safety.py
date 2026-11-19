from __future__ import annotations

import os
from pathlib import Path

import pytest

from prismaquant.export_output_safety import (
    directory_publication_target,
    prepare_fresh_export_directory,
    prepare_fresh_export_file,
    transactional_export_directory,
    transactional_export_file,
)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_directory_preflight_rejects_in_place_single_file_source(tmp_path):
    source = tmp_path / "model"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"single-source")
    before = _snapshot(source)

    with pytest.raises(RuntimeError, match="resolve to the same path"):
        prepare_fresh_export_directory(source, source, where="unit-export")

    assert _snapshot(source) == before


def test_directory_preflight_rejects_symlink_alias_to_sharded_source(tmp_path):
    source = tmp_path / "model"
    source.mkdir()
    (source / "model-00001-of-00002.safetensors").write_bytes(b"shard-one")
    (source / "model-00002-of-00002.safetensors").write_bytes(b"shard-two")
    (source / "model.safetensors.index.json").write_text("{}")
    alias = tmp_path / "output-alias"
    alias.symlink_to(source, target_is_directory=True)
    before = _snapshot(source)

    with pytest.raises(RuntimeError, match="resolve to the same path"):
        prepare_fresh_export_directory(source, alias, where="unit-export")

    assert alias.is_symlink()
    assert _snapshot(source) == before


def test_directory_preflight_rejects_broken_output_symlink(tmp_path):
    source = tmp_path / "model"
    source.mkdir()
    missing_target = tmp_path / "missing-target"
    output = tmp_path / "output"
    output.symlink_to(missing_target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="is a symlink"):
        prepare_fresh_export_directory(source, output, where="unit-export")

    assert os.path.lexists(output)
    assert output.is_symlink()
    assert not missing_target.exists()


def test_directory_preflight_rejects_stale_auxiliary_file(tmp_path):
    source = tmp_path / "model"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    stale = output / "tokenizer_config.json"
    stale.write_text('{"stale": true}')

    with pytest.raises(RuntimeError, match="is not empty"):
        prepare_fresh_export_directory(source, output, where="unit-export")

    assert stale.read_text() == '{"stale": true}'


def test_directory_preflight_rejects_output_inside_source(tmp_path):
    source = tmp_path / "model"
    source.mkdir()
    payload = source / "model.safetensors"
    payload.write_bytes(b"source")
    output = source / "exported"

    with pytest.raises(RuntimeError, match="ancestor/descendant"):
        prepare_fresh_export_directory(source, output, where="unit-export")

    assert not output.exists()
    assert payload.read_bytes() == b"source"


def test_directory_preflight_rejects_source_inside_output(tmp_path):
    output = tmp_path / "artifact-tree"
    source = output / "source-model"
    source.mkdir(parents=True)
    payload = source / "model.safetensors"
    payload.write_bytes(b"source")

    with pytest.raises(RuntimeError, match="ancestor/descendant"):
        prepare_fresh_export_directory(source, output, where="unit-export")

    assert payload.read_bytes() == b"source"
    assert set(output.iterdir()) == {source}


def test_directory_preflight_accepts_only_new_or_empty_real_directory(tmp_path):
    source = tmp_path / "model"
    source.mkdir()
    new_output = prepare_fresh_export_directory(
        source,
        tmp_path / "new-output",
        where="unit-export",
    )
    assert new_output.is_dir() and not any(new_output.iterdir())

    empty = tmp_path / "empty-output"
    empty.mkdir()
    assert prepare_fresh_export_directory(
        source,
        empty,
        where="unit-export",
    ) == empty


def test_file_preflight_rejects_exact_and_symlink_source_aliases(tmp_path):
    source = tmp_path / "skeleton.gguf"
    source.write_bytes(b"gguf-source")
    alias = tmp_path / "alias.gguf"
    alias.symlink_to(source)

    for output in (source, alias):
        with pytest.raises(RuntimeError, match="resolve to the same path"):
            prepare_fresh_export_file(source, output, where="unit-gguf")

    assert source.read_bytes() == b"gguf-source"
    assert alias.is_symlink()


def test_file_preflight_rejects_existing_and_broken_symlink_outputs(tmp_path):
    source = tmp_path / "skeleton.gguf"
    source.write_bytes(b"gguf-source")
    existing = tmp_path / "existing.gguf"
    existing.write_bytes(b"old-output")
    broken = tmp_path / "broken.gguf"
    missing_target = tmp_path / "missing.gguf"
    broken.symlink_to(missing_target)

    with pytest.raises(RuntimeError, match="already exists"):
        prepare_fresh_export_file(source, existing, where="unit-gguf")
    with pytest.raises(RuntimeError, match="already exists as a symlink"):
        prepare_fresh_export_file(source, broken, where="unit-gguf")

    assert existing.read_bytes() == b"old-output"
    assert broken.is_symlink() and not missing_target.exists()


def test_gguf_export_guard_runs_before_reader_or_writer(tmp_path):
    pytest.importorskip("gguf")
    from prismaquant.export_gguf import export_gguf

    skeleton = tmp_path / "skeleton.gguf"
    skeleton.write_bytes(b"not parsed because path guard runs first")
    layer_config = tmp_path / "assignment.json"
    layer_config.write_text("{}")

    with pytest.raises(RuntimeError, match="resolve to the same path"):
        export_gguf(skeleton, layer_config, skeleton, device="cpu")

    assert skeleton.read_bytes() == b"not parsed because path guard runs first"


def test_gguf_assignment_identity_matches_original_bytes(tmp_path, monkeypatch):
    import hashlib
    import json
    from types import SimpleNamespace

    fixture = Path(__file__).parent / "fixtures/gguf_assignment_identity_2541.json"
    assert fixture.is_file(), "The original GGUF assignment record is missing."

    import gguf
    import numpy as np

    from prismaquant import export_gguf as module
    from prismaquant.digests import DIRECT_ASCII_LAX, bytes_sha256hex, file_sha256hex

    original = json.loads(fixture.read_text(encoding="utf-8"))
    expected_bytes = original["assignment_digest_input"].encode("ascii")
    skeleton = tmp_path / "skeleton.gguf"
    writer = gguf.GGUFWriter(skeleton, "llama")
    writer.add_block_count(1)
    writer.add_tensor("blk.0.ffn_up.weight", np.array([[1, 2]], dtype=np.float32))
    writer.add_tensor("blk.0.ffn_down.weight", np.array([[3, 4]], dtype=np.float32))
    writer.add_tensor("output.weight", np.array([[5, 6]], dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    config = tmp_path / "assignment.json"
    config.write_text(original["config_json"], encoding="utf-8")
    output = tmp_path / "artifact.gguf"
    inputs = []

    def capture_sha256(payload):
        inputs.append(payload)
        return hashlib.sha256(payload)

    monkeypatch.setattr(module, "hashlib", SimpleNamespace(sha256=capture_sha256))
    monkeypatch.setattr(module, "_git_commit", lambda: "fixture")
    module.export_gguf(skeleton, config, output, device="cpu")
    identity = gguf.GGUFReader(output).fields["prismaquant.assignment_sha256"].contents()
    assert inputs == [expected_bytes]
    assert identity == original["assignment_sha256"]
    assert file_sha256hex(output) == original["artifact_sha256"]

    owner_bytes = DIRECT_ASCII_LAX.encoded(module.load_assignment(config))
    assert owner_bytes == expected_bytes
    assert bytes_sha256hex(owner_bytes) == original["assignment_sha256"]


def _transaction_temps(output: Path) -> list[Path]:
    return sorted(output.parent.glob(f".{output.name}.tmp-*"))


def test_directory_transaction_publishes_complete_tree_only_after_success(
    tmp_path,
):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "artifact"

    with transactional_export_directory(
        source,
        output,
        where="unit-directory-transaction",
    ) as staged:
        assert staged.parent == output.parent
        assert staged != output
        assert not output.exists()
        (staged / "model.safetensors").write_bytes(b"complete-model")
        (staged / "config.json").write_text('{"complete": true}')

    assert (output / "model.safetensors").read_bytes() == b"complete-model"
    assert (output / "config.json").read_text() == '{"complete": true}'
    # mkdtemp is private 0700, but the published root must have the same mode a
    # normal mkdir under this process's umask would have used.
    mode_probe = tmp_path / "mode-probe"
    mode_probe.mkdir()
    assert (output.stat().st_mode & 0o7777) == (mode_probe.stat().st_mode & 0o7777)
    assert not _transaction_temps(output)


def test_directory_transaction_exposes_final_publication_target_and_resets(
    tmp_path,
):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "artifact"

    with transactional_export_directory(
        source,
        output,
        where="unit-directory-transaction",
    ) as staged:
        assert directory_publication_target(staged) == output.resolve()
        unrelated = tmp_path / "unrelated"
        assert directory_publication_target(unrelated) == unrelated.resolve()
        (staged / "config.json").write_text("{}")

    # Context must not leak into subsequent work after publication.
    assert directory_publication_target(output) == output.resolve()


def test_shipcard_built_in_transaction_names_published_directory(tmp_path):
    from prismaquant.shipcard import build_shipcard, load_shipcard, write_shipcard

    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "artifact"
    with transactional_export_directory(
        source,
        output,
        where="unit-directory-transaction",
    ) as staged:
        (staged / "config.json").write_text("{}")
        (staged / "model.safetensors").write_bytes(b"weights")
        write_shipcard(staged / "shipcard.json", build_shipcard(staged))

    card = load_shipcard(output / "shipcard.json")
    assert Path(card["model_dir"]) == output.resolve()
    assert ".tmp-" not in card["model_dir"]


def test_directory_publication_target_context_resets_after_failure(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "artifact"

    with pytest.raises(RuntimeError, match="late failure"):
        with transactional_export_directory(
            source,
            output,
            where="unit-directory-transaction",
        ) as staged:
            assert directory_publication_target(staged) == output.resolve()
            raise RuntimeError("late failure")

    assert directory_publication_target(output) == output.resolve()


def test_directory_transaction_preserves_existing_empty_destination_mode(
    tmp_path,
):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "artifact"
    output.mkdir(mode=0o750)
    output.chmod(0o750)

    with transactional_export_directory(
        source,
        output,
        where="unit-directory-transaction",
    ) as staged:
        (staged / "model.safetensors").write_bytes(b"complete-model")

    assert (output.stat().st_mode & 0o7777) == 0o750
    assert (output / "model.safetensors").read_bytes() == b"complete-model"


def test_directory_transaction_rolls_back_post_model_failure_and_preserves_empty(
    tmp_path,
):
    source = tmp_path / "source"
    source.mkdir()

    new_output = tmp_path / "new-artifact"
    with pytest.raises(RuntimeError, match="post-model validation failed"):
        with transactional_export_directory(
            source,
            new_output,
            where="unit-directory-transaction",
        ) as staged:
            (staged / "model.safetensors").write_bytes(b"partial")
            raise RuntimeError("post-model validation failed")
    assert not new_output.exists()
    # Failure now PRESERVES the partial temp root so a late gate (metadata-only
    # completeness failures run after every tensor is written) does not discard
    # byte-identical work; --reuse-prior resumes from it. The load-bearing
    # invariant is unchanged and asserted above: the publish path is untouched.
    preserved = _transaction_temps(new_output)
    assert len(preserved) == 1, preserved
    assert preserved[0].name.startswith(".new-artifact.tmp-")

    empty_output = tmp_path / "empty-artifact"
    empty_output.mkdir()
    before = empty_output.stat()
    with pytest.raises(RuntimeError, match="hard budget exceeded"):
        with transactional_export_directory(
            source,
            empty_output,
            where="unit-directory-transaction",
        ) as staged:
            (staged / "model.safetensors").write_bytes(b"over-budget")
            raise RuntimeError("hard budget exceeded")
    after = empty_output.stat()
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert not any(empty_output.iterdir())
    preserved_empty = _transaction_temps(empty_output)
    assert len(preserved_empty) == 1, preserved_empty
    assert preserved_empty[0].name.startswith(".empty-artifact.tmp-")


def test_directory_transaction_does_not_clobber_concurrent_destination(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "artifact"

    with pytest.raises(RuntimeError, match="appeared during export"):
        with transactional_export_directory(
            source,
            output,
            where="unit-directory-transaction",
        ) as staged:
            (staged / "model.safetensors").write_bytes(b"ours")
            output.mkdir()
            (output / "concurrent.txt").write_text("theirs")

    assert (output / "concurrent.txt").read_text() == "theirs"
    preserved = _transaction_temps(output)
    assert len(preserved) == 1, preserved
    assert preserved[0].name.startswith(".artifact.tmp-")


def test_directory_transaction_rejects_replaced_owned_temp_root(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "artifact"

    with pytest.raises(RuntimeError, match="changed identity"):
        with transactional_export_directory(
            source,
            output,
            where="unit-directory-transaction",
        ) as staged:
            moved_original = staged.with_name(staged.name + "-moved")
            staged.rename(moved_original)
            staged.mkdir()
            (staged / "untrusted").write_text("do not publish or clean")

    assert not output.exists()
    assert (staged / "untrusted").read_text() == "do not publish or clean"
    assert moved_original.is_dir()


def test_file_transaction_hard_link_publishes_and_rolls_back(tmp_path):
    source = tmp_path / "source.gguf"
    source.write_bytes(b"source")
    output = tmp_path / "artifact.gguf"

    with transactional_export_file(
        source,
        output,
        where="unit-file-transaction",
    ) as staged:
        staged.write_bytes(b"complete-gguf")
        assert not output.exists()
    assert output.read_bytes() == b"complete-gguf"
    assert not _transaction_temps(output)

    failed = tmp_path / "failed.gguf"
    with pytest.raises(RuntimeError, match="hard budget exceeded"):
        with transactional_export_file(
            source,
            failed,
            where="unit-file-transaction",
        ) as staged:
            staged.write_bytes(b"over-budget")
            raise RuntimeError("hard budget exceeded")
    assert not failed.exists()
    assert not _transaction_temps(failed)


def test_file_transaction_does_not_clobber_concurrent_destination(tmp_path):
    source = tmp_path / "source.gguf"
    source.write_bytes(b"source")
    output = tmp_path / "artifact.gguf"

    with pytest.raises(RuntimeError, match="appeared during export"):
        with transactional_export_file(
            source,
            output,
            where="unit-file-transaction",
        ) as staged:
            staged.write_bytes(b"ours")
            output.write_bytes(b"theirs")

    assert output.read_bytes() == b"theirs"
    assert not _transaction_temps(output)


@pytest.mark.parametrize("budget_failure", [False, True])
def test_gguf_transaction_publishes_or_cleans_after_budget_check(
    tmp_path,
    monkeypatch,
    budget_failure,
):
    pytest.importorskip("gguf")
    from prismaquant import export_gguf as module

    skeleton = tmp_path / "skeleton.gguf"
    skeleton.write_bytes(b"source")
    layer_config = tmp_path / "assignment.json"
    layer_config.write_text("{}")
    output = tmp_path / "artifact.gguf"

    class _Field:
        def __init__(self, value):
            self.value = value

        def contents(self):
            return self.value

    class _Reader:
        fields = {
            "general.architecture": _Field("test"),
            "test.block_count": _Field(0),
        }
        tensors = []

    class _Writer:
        def __init__(self, path, _architecture):
            self.path = Path(path)

        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

        def write_header_to_file(self):
            self.path.write_bytes(b"complete-gguf")

    monkeypatch.setattr(module.gguf, "GGUFReader", lambda _path: _Reader())
    monkeypatch.setattr(module.gguf, "GGUFWriter", _Writer)
    monkeypatch.setattr(module, "_copy_metadata", lambda *_args: None)
    monkeypatch.setattr(
        module,
        "_map_assignment_to_gguf",
        lambda *_args: ({}, set()),
    )
    monkeypatch.setattr(
        module,
        "_resolve_token_embedding_format",
        lambda *_args: None,
    )
    if budget_failure:
        monkeypatch.setattr(
            module,
            "enforce_whole_artifact_budget",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("hard whole-artifact budget exceeded")
            ),
        )
        with pytest.raises(RuntimeError, match="hard whole-artifact budget"):
            module.export_gguf(
                skeleton,
                layer_config,
                output,
                device="cpu",
            )
        assert not output.exists()
    else:
        monkeypatch.setattr(
            module,
            "enforce_whole_artifact_budget",
            lambda *_args, **_kwargs: None,
        )
        module.export_gguf(skeleton, layer_config, output, device="cpu")
        assert output.read_bytes() == b"complete-gguf"
    assert not _transaction_temps(output)


def test_export_rollback_preserves_a_replaced_published_file(tmp_path):
    from prismaquant.export_output_safety import _remove_owned_file_link

    published = tmp_path / "published"
    published.write_bytes(b"owned")
    info = published.stat()
    identity = (info.st_dev, info.st_ino)
    held = tmp_path / "held"
    published.rename(held)
    published.write_bytes(b"replacement")
    with pytest.raises(RuntimeError, match="refusing to unlink an untrusted replacement"):
        _remove_owned_file_link(published, identity, where="test-export")
    assert published.read_bytes() == b"replacement"
    assert held.read_bytes() == b"owned"
    _remove_owned_file_link(held, identity, where="test-export")
    assert not held.exists()
