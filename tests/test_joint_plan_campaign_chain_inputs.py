"""A plan that cannot name its campaign chain is refused by name, before CUDA (PQ #1293).

``load_joint_anchor_plan`` admitted a plan with no ``inputs`` block, and the
``prepare`` GPU action then died on a bare ``KeyError: 'inputs'`` at
``tessera_joint_aura`` -- after the projection prewarm had already allocated
on the device, exactly the failure mode ``_config_device_envelope`` exists
because of: a config the admission gate should have refused, raised after the
device was touched, indistinguishable from an admitted plan that failed
later. Preserved evidence: run-01 S3 of the #1293 non-release pilot
(``issues-pq-1293-nonrelease-20261001-v1/run-01``, 2026-10-04).

The named refusal lives at the plan's own admission, the grammar owner:
``inputs`` present as a mapping, every bound campaign chain key shaped
``{path, sha256}``, and a ``canonical_capture`` binding. The chain loader
names its missing keys instead of subscripting them, and the standalone
synthesis census read goes through the same named seam. A plan that binds a
subset (the Stage B quantum shape, PQ #1024) stays loadable.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tests", ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from test_joint_projection_backend import _plan  # noqa: E402


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(tmp_path, config):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(config))
    return path


def test_plan_without_inputs_is_refused_at_admission_by_name(tmp_path):
    """The run-01 S3 shape: no ``inputs`` block, refused at load, not in CUDA."""
    from prismaquant.tessera_joint_aura import load_joint_anchor_plan

    config = _plan(tmp_path / "run")
    del config["inputs"]
    path = _write(tmp_path, config)
    with pytest.raises(ValueError, match="campaign chain"):
        load_joint_anchor_plan(path, _sha(path))


def test_plan_without_canonical_capture_is_refused_at_admission_by_name(tmp_path):
    """Refused by name at load, not as a bare KeyError inside prepare.

    This admits nothing new: every prepare arm reads
    ``config["canonical_capture"]`` bare (``_prepare_source_owner``,
    ``prepare_cache``), and a campaign scope's artifacts always name it
    (``joint_catalog_extension.SCOPE_ARTIFACT_BINDINGS``), so a plan without
    the binding never executed -- it died later, after the prewarm.
    """
    from prismaquant.tessera_joint_aura import load_joint_anchor_plan

    config = _plan(tmp_path / "run")
    del config["canonical_capture"]
    path = _write(tmp_path, config)
    with pytest.raises(ValueError, match="canonical capture"):
        load_joint_anchor_plan(path, _sha(path))


@pytest.mark.parametrize("defer_pool_reads", [False, True])
def test_malformed_chain_binding_is_refused_at_admission_by_shape(
        tmp_path, defer_pool_reads):
    """A bound chain key that is not a path/SHA256 pair refuses by name.

    Shape-only: the admission reads the parsed plan and never touches the
    bound file, so the deferred Stage B load still stats nothing (PQ #1024).
    """
    from prismaquant.tessera_joint_aura import load_joint_anchor_plan

    config = _plan(tmp_path / "run")
    config["inputs"]["census"] = {"path": "census.json"}
    path = _write(tmp_path, config)
    with pytest.raises(ValueError, match="census.*path/SHA256"):
        load_joint_anchor_plan(path, _sha(path), defer_pool_reads=defer_pool_reads)


def test_chain_loader_names_missing_chain_keys(tmp_path):
    """The walk's own admission names every missing key, once, in one refusal."""
    from prismaquant.tessera_joint_aura import (
        HEAD_WALK_INPUT_KEYS, load_measured_anchor_input)

    with pytest.raises(ValueError) as exc:
        load_measured_anchor_input({})
    message = str(exc.value)
    for key in HEAD_WALK_INPUT_KEYS:
        assert key in message, f"{key} is not named in the refusal: {message}"


def test_synthesize_census_read_names_a_missing_binding(tmp_path):
    """The standalone synthesis census read refuses by name, not by KeyError."""
    from prismaquant.tessera_joint_aura import synthesize_renders

    config = _plan(tmp_path / "run")
    with pytest.raises(ValueError, match="census"):
        synthesize_renders(config, plan_sha256=_sha(_write(tmp_path, config)),
                           mirror_root=str(tmp_path / "mirror"))


def test_plan_binding_a_subset_still_admits(tmp_path):
    """The Stage B quantum shape (PQ #1024) keeps loading, both modes."""
    from prismaquant.tessera_joint_aura import load_joint_anchor_plan

    path, config, _pool = _pool_plan(tmp_path)
    assert load_joint_anchor_plan(path, _sha(path)) == config
    assert load_joint_anchor_plan(path, _sha(path), defer_pool_reads=True)


def _pool_plan(tmp_path):
    """The test_load_plan_readset_1024 pool plan: ``inputs: {}``, capture bound."""
    pool = tmp_path / "pool"
    (pool / "boundaries").mkdir(parents=True)
    config = _plan(tmp_path / "run")
    # Exercise the empty subset independently of the imported fixture default.
    config["inputs"] = {}
    config["execution"]["boundary_storage"] = {
        "schema": "prismaquant.aura.boundary_storage.v1",
        "directory": str(pool / "boundaries"),
        "max_resident_bytes": 1, "max_auxiliary_bytes": 1,
        "max_artifact_bytes": 1, "prefetch_batches": 1}
    path = _write(tmp_path, config)
    return path, config, pool
