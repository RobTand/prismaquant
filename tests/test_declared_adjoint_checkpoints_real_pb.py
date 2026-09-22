"""Real PrismaBuild export of a declared adjoint checkpoint's own files.

Kept in its own file, so its process imports PrismaBuild only from the
qualified local-spool bundle (``stagea_local_spool_pb_pin.json``): a test
that bound the other candidate first would shadow it and this test would
skip. The private fleet runs PrismaBuild's actual claim, export and finish
path; nothing on the PrismaBuild side is replaced.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import test_stage_a_produced_boundary_chain as chain
from test_stage_a_produced_boundary_chain import _isolated_launch_context  # noqa: F401
from test_declared_adjoint_checkpoints import (
    BOUNDARY, _same, _session, _shared_adjoint, _shared_pass, _tensor)
from prismaquant.joint_adjoint_checkpoints import (
    adjoint_space, checkpoint_directory, load_adjoint_checkpoint,
    write_adjoint_checkpoint)


def test_real_prismabuild_exports_the_checkpoint_group(tmp_path, monkeypatch):
    """PrismaBuild's own export action copies the checkpoint-class group."""
    from fleet_sdk import require_prismabuild_sdk
    require_prismabuild_sdk()
    pin_path = Path(__file__).with_name("stagea_local_spool_pb_pin.json")
    pin = json.loads(pin_path.read_text())
    root = Path(pin["bundle_root"])
    assert root.is_dir(), "the qualified PB source bundle must be provisioned"
    for name, digest in pin["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    monkeypatch.setattr(chain, "PIN_PATH", pin_path)
    from prismaquant.produced_output_spool import MAX_ENV, ROOT_ENV
    owner, publication, queue, _env, _repo = chain._bound_owner(
        tmp_path, staging_timeout_s=90, published=True,
        claim_capacity={"cpu": 4, "mem_gb": 4}, checkpoint_max_bytes=1 << 20,
        producer_environment={ROOT_ENV: str(tmp_path / "local-output"),
                              MAX_ENV: str(1 << 20)})
    spool = owner._local_output_spool
    assert spool is not None
    plane = {(0, batch): owner.write(_tensor(0, batch), probe_index=0,
                                     batch_index=batch, boundary_index=BOUNDARY)
             for batch in range(chain.GROUP_SIZE)}
    space = adjoint_space(tmp_path / "outputs")
    shared_adjoint, shared_pass = (_shared_adjoint(1, chain.GROUP_SIZE),
                                   _shared_pass(chain.GROUP_SIZE))
    record = write_adjoint_checkpoint(
        space, boundary=BOUNDARY, session=_session(owner), cotangents=plane,
        shared_adjoint=shared_adjoint, shared_pass=shared_pass, owner=owner,
        declared=True)
    directory = checkpoint_directory(space, BOUNDARY)
    assert not directory.exists()
    batch_id = publication.batch_id_for(kind="checkpoint", boundary_index=BOUNDARY,
                                        group_index=0)
    rows = spool._groups[batch_id]["references"]
    assert rows and {artifact_class for _l, _c, artifact_class in rows} == {"checkpoint"}
    sources = [Path(local.path) for local, _c, _a in rows]
    assert all(path.is_file() for path in sources)
    with chain._fleet(queue, tmp_path, capacity={"cpu": 4, "mem_gb": 4}) as fleet:
        owner.settle_local_output()
    outcomes = [json.loads(line) for line in fleet.stdout.splitlines()
                if line.startswith("{")]
    assert len(outcomes) >= 2 and all(row["rc"] == 0 for row in outcomes), outcomes
    for local, canonical, _a in rows:
        data = Path(canonical.path).read_bytes()
        assert hashlib.sha256(data).hexdigest() == local.sha256
    assert not any(path.exists() for path in sources)
    assert spool.report()["pending_groups"] == 0
    loaded, loaded_shared, loaded_pass = load_adjoint_checkpoint(space, record)
    for key in plane:
        _same(loaded[key], _tensor(*key))
    _same(loaded_shared, shared_adjoint)
    _same(loaded_pass, shared_pass)
