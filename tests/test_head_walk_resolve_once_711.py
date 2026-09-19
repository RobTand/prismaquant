"""The head walk resolves shared directories once, not once per cell (#711, #722).

The GLM-5.3 run spent ~1 h single-threaded resolving paths before any GPU
work: per cell the walk called ``wire.resolve()`` twice,
``wire_dir.resolve()`` (the same directory for every cell of the campaign)
and ``render.resolve()`` -- ~62 NFS LOOKUPs per cell at ~19 cells/s, on top
of the serial origin-marker walk (#722 reports the same loop).

The walk now resolves the wire directory and each distinct row directory
once and joins every cell's recorded wire/render paths from those roots;
the per-cell escape check is the symlink test on the wire itself (the
filename is already validated as a leaf, so its lexical parent is the wire
directory). Three properties pin that, and none may bend the others:

* **Equivalent records** -- the recorded wire/render strings are exactly
  what the old fully-resolved joins produced, so downstream consumers
  (prepare, handoff, resume fences) read the same bytes from the same names.
* **Bounded resolves** -- during the walk the wire directory and each owner
  root are resolved exactly once, and no resolved path is a file under
  either root: zero per-cell ``resolve()`` calls however many cells run.
* **Fail-closed escape** -- a symlinked wire is still refused.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_tessera_joint_aura import fixture


def _spy_resolves(monkeypatch):
    seen = []
    real_resolve = Path.resolve

    def counting(self, *args, **kwargs):
        seen.append(str(self))
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counting)
    return seen


def test_recorded_paths_match_the_legacy_fully_resolved_joins(tmp_path):
    """The join-from-resolved-roots records the same strings ``resolve()`` did."""
    from prismaquant import tessera_joint_aura as bridge
    from prismaquant.production_weight_cache import _cache_weight_filename

    config, names, fmt, _payload, _states = fixture(tmp_path)
    data = bridge.load_measured_anchor_input(config, verify_payloads=False)
    wire_dir = (tmp_path / "merged/cache/wire").resolve()
    rowdir = (tmp_path / "campaign/rows/row-0000").resolve()
    assert len(data.cells) == 2 * 1
    for name in names:
        cell = data.cells[name, fmt]
        assert cell["wire"] == str(wire_dir / (name + ".tessera"))
        assert cell["render"] == str(
            rowdir / "cache" / _cache_weight_filename(name, fmt))


def test_shared_directories_resolve_once_and_no_cell_file_resolves(tmp_path, monkeypatch):
    """The per-cell ``resolve()`` storm is gone; shared roots resolve once."""
    from prismaquant import tessera_joint_aura as bridge

    config, _names, _fmt, _payload, _states = fixture(tmp_path)
    seen = _spy_resolves(monkeypatch)
    start = len(seen)
    bridge.load_measured_anchor_input(config, verify_payloads=False)
    walked = seen[start:]
    wire_root = str(tmp_path / "merged/cache/wire")
    owner_root = str(tmp_path / "campaign/rows/row-0000")
    # Each shared directory is resolved exactly once for the whole walk,
    # however many cells it holds.
    assert walked.count(wire_root) == 1
    assert walked.count(owner_root) == 1
    # And no per-cell file path under either root is ever resolved: the old
    # loop resolved the wire twice and the render once per cell.
    cell_files = [path for path in walked
                  if path.startswith(wire_root + "/") or path.startswith(owner_root + "/")]
    assert cell_files == []


def test_a_symlinked_wire_is_still_refused(tmp_path):
    """Resolving once does not open the escape the per-cell resolve closed."""
    from prismaquant import tessera_joint_aura as bridge

    config, names, fmt, _payload, _states = fixture(tmp_path)
    target = tmp_path / "merged/cache/wire" / (names[0] + ".tessera")
    blob = target.read_bytes()
    target.unlink()
    target.symlink_to(tmp_path / "elsewhere.tessera")
    (tmp_path / "elsewhere.tessera").write_bytes(blob)
    with pytest.raises(ValueError, match="escaping wire path"):
        bridge.load_measured_anchor_input(config, verify_payloads=False,
                                          head_walk_workers=1)
