"""The default adjoint checkpoint path writes what origin/main wrote.

``write_adjoint_checkpoint`` gained a declared writer behind
``declared=True`` (``PRISMAQUANT_STAGE_A_DECLARED_CHECKPOINTS=1`` in Stage A).
With it unset, the record, the file set and every file's bytes must be the
ones the writer produced before that change, on the unwatched path
(``owner=None``) and on the owner path, together with the owner's commitment
and ledgers.

The expected fingerprints were produced by running this file unchanged
against origin/main 613ffb68a0 through PrismaBuild. The file imports nothing
origin/main lacks, so the same test runs on both trees; the PR records both
action keys.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.cost_streaming import BOUNDARY_STORAGE_SCHEMA, StreamedBoundaryArtifacts
from prismaquant.joint_adjoint_checkpoints import adjoint_space, write_adjoint_checkpoint

BOUNDARY = 5


def _session():
    return {"generation": "ab" * 16, "kind": "adjoint_checkpoint",
            "run_identity_sha256": "cd" * 32}


def _plane():
    return {(probe, batch): torch.arange(64, dtype=torch.float32).reshape(8, 8)
            * (probe + 1) + batch
            for probe in range(2) for batch in range(3)}


# Shared state carries no tensor here on purpose. Pickling a tensor goes
# through torch's legacy storage serializer, whose storage key is the
# storage's C address, so the same tensor pickles to different bytes from
# one allocation to the next -- on origin/main exactly as on this branch.
# Plain values pickle deterministically, which lets every file be compared
# byte for byte; the cotangent ``.pt`` files are tensors and are deterministic.
def _shared_adjoint():
    return {(probe, batch): {"w0": 0.5 * probe, "w1": float(batch), "n": probe * 10 + batch}
            for probe in range(2) for batch in range(3)}


def _shared_pass():
    return {batch: {"tag": f"batch-{batch}", "n": batch} for batch in range(3)}


def _fingerprint(space: Path, record: dict) -> dict:
    """Hashes of the record and every file, independent of the pytest path.

    Paths inside ``space`` are spelled ``<space>``. ``cotangent_sha256`` is a
    digest over the record's own fields, absolute paths included, so it
    changes with the temporary directory: it is checked here to be exactly
    ``canonical_json_sha256`` of those fields and then spelled ``<digest>``.
    """
    digest = record["cotangent_sha256"]
    assert digest == canonical_json_sha256(
        {key: record[key] for key in
         ("schema", "boundary", "session", "activation_entries", "shared_state_entries")},
        where="adjoint checkpoint")
    root = str(space)
    text = (json.dumps(record, sort_keys=True)
            .replace(root, "<space>").replace(digest, "<digest>"))
    files = {}
    for path in sorted(space.rglob("*")):
        if path.is_file():
            data = (path.read_bytes().replace(root.encode(), b"<space>")
                    .replace(digest.encode(), b"<digest>"))
            files[str(path.relative_to(space))] = hashlib.sha256(data).hexdigest()
    return {"record_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "files": files}


#: Produced on origin/main 613ffb68a0 by PrismaBuild action 4d2404bdd702
#: (see the module docstring).
EXPECTED = {'owner': {'files': {'checkpoints/boundary-005/checkpoint.json': 'dffecdeee48e1dc8c5e98cb6460ef0739b99419ae4d95705294f2395714c9bb6',
                     'checkpoints/boundary-005/entries/cotangent-0-0.pt': '7967f18b256dd93f70a7f3ca097eab2ff92824a484d1b602e309ebbcc7a8f969',
                     'checkpoints/boundary-005/entries/cotangent-0-1.pt': 'f62c23f3c2439f8d5ffac304c7c250f49f71d66e2ce55cdbc82663a36b7a7c5f',
                     'checkpoints/boundary-005/entries/cotangent-0-2.pt': '3e2f8032bc4fab20e889da2b24fb9fd0880e419c5dfecc349ce6342b0297ad56',
                     'checkpoints/boundary-005/entries/cotangent-1-0.pt': 'a99853b6dfb0dd471689872a9c295a24b3460bef312e888f89ab7ced21c1dbe5',
                     'checkpoints/boundary-005/entries/cotangent-1-1.pt': 'ae8a721dc7745cdde04f02c482bdb965c30e4197dee2d65b53bfc9b8a10f8aed',
                     'checkpoints/boundary-005/entries/cotangent-1-2.pt': 'ff54f7a270cb5abb49a2e122877ab05c6c529c70c070ff47e5f1655762cca9a9',
                     'checkpoints/boundary-005/entries/shared-adjoint-0-0.pkl': '6d53da002640ef917fdaee7e6d721293865e58cee7887fe2966c220bad4d9ae1',
                     'checkpoints/boundary-005/entries/shared-adjoint-0-1.pkl': 'e9eaad61645650d8f96429984167525f5d726c3af20e979f3ff9ed2c39e83bc7',
                     'checkpoints/boundary-005/entries/shared-adjoint-0-2.pkl': 'ecaf1900bde89c3b05a349d8e01c560dca36fcdbfdeff0fdb607d6156016b09a',
                     'checkpoints/boundary-005/entries/shared-adjoint-1-0.pkl': '1a9baf92860ee83b19f8b35fd760b2be436a99ddf08854d630eebedacdd9bf54',
                     'checkpoints/boundary-005/entries/shared-adjoint-1-1.pkl': '0ad443bc36d51b9060b6581bf88346e2a50cbd7422402e33f2ac671973f1eda6',
                     'checkpoints/boundary-005/entries/shared-adjoint-1-2.pkl': '7eb4acfd62bab0b8ba90d457d86d278df42ead29c23d44af09ebdd93e81a6265',
                     'checkpoints/boundary-005/entries/shared-pass-0.pkl': '23b6ccecfeec0aeb768bef566c3b191fe330361c9c2217b9d5cb1305bc8ac36c',
                     'checkpoints/boundary-005/entries/shared-pass-1.pkl': 'd66e518a13bd05835d2a426103bb6c074ffb69c5d3c53f2720196e6c99e942b9',
                     'checkpoints/boundary-005/entries/shared-pass-2.pkl': '9c48bd2d25381c41540df6c8fb0f896ed75ac358a84c5da6e704d767d6f70a41'},
           'record_sha256': '2f01160c573fd1b1cb5797c2cb78dd5a70ae1a55fc494c60402283a00475acca'},
 'owner_commitment': {'actual_bytes': 24675,
                      'checkpoint_dir': '<space>/checkpoints/boundary-005',
                      'envelope_bytes': 476868,
                      'receipt_digest': '<digest>',
                      'reservation': 1,
                      'unused_bytes': 452193},
 'owner_ledgers': {'checkpoint_envelope_unused_bytes': 452193,
                   'live_artifact_bytes': 14454,
                   'live_checkpoint_bytes': 24675,
                   'retired_entries': 0,
                   'written_entries': 6},
 'unwatched': {'files': {'checkpoints/boundary-005/checkpoint.json': 'dffecdeee48e1dc8c5e98cb6460ef0739b99419ae4d95705294f2395714c9bb6',
                         'checkpoints/boundary-005/entries/cotangent-0-0.pt': '7967f18b256dd93f70a7f3ca097eab2ff92824a484d1b602e309ebbcc7a8f969',
                         'checkpoints/boundary-005/entries/cotangent-0-1.pt': 'f62c23f3c2439f8d5ffac304c7c250f49f71d66e2ce55cdbc82663a36b7a7c5f',
                         'checkpoints/boundary-005/entries/cotangent-0-2.pt': '3e2f8032bc4fab20e889da2b24fb9fd0880e419c5dfecc349ce6342b0297ad56',
                         'checkpoints/boundary-005/entries/cotangent-1-0.pt': 'a99853b6dfb0dd471689872a9c295a24b3460bef312e888f89ab7ced21c1dbe5',
                         'checkpoints/boundary-005/entries/cotangent-1-1.pt': 'ae8a721dc7745cdde04f02c482bdb965c30e4197dee2d65b53bfc9b8a10f8aed',
                         'checkpoints/boundary-005/entries/cotangent-1-2.pt': 'ff54f7a270cb5abb49a2e122877ab05c6c529c70c070ff47e5f1655762cca9a9',
                         'checkpoints/boundary-005/entries/shared-adjoint-0-0.pkl': '6d53da002640ef917fdaee7e6d721293865e58cee7887fe2966c220bad4d9ae1',
                         'checkpoints/boundary-005/entries/shared-adjoint-0-1.pkl': 'e9eaad61645650d8f96429984167525f5d726c3af20e979f3ff9ed2c39e83bc7',
                         'checkpoints/boundary-005/entries/shared-adjoint-0-2.pkl': 'ecaf1900bde89c3b05a349d8e01c560dca36fcdbfdeff0fdb607d6156016b09a',
                         'checkpoints/boundary-005/entries/shared-adjoint-1-0.pkl': '1a9baf92860ee83b19f8b35fd760b2be436a99ddf08854d630eebedacdd9bf54',
                         'checkpoints/boundary-005/entries/shared-adjoint-1-1.pkl': '0ad443bc36d51b9060b6581bf88346e2a50cbd7422402e33f2ac671973f1eda6',
                         'checkpoints/boundary-005/entries/shared-adjoint-1-2.pkl': '7eb4acfd62bab0b8ba90d457d86d278df42ead29c23d44af09ebdd93e81a6265',
                         'checkpoints/boundary-005/entries/shared-pass-0.pkl': '23b6ccecfeec0aeb768bef566c3b191fe330361c9c2217b9d5cb1305bc8ac36c',
                         'checkpoints/boundary-005/entries/shared-pass-1.pkl': 'd66e518a13bd05835d2a426103bb6c074ffb69c5d3c53f2720196e6c99e942b9',
                         'checkpoints/boundary-005/entries/shared-pass-2.pkl': '9c48bd2d25381c41540df6c8fb0f896ed75ac358a84c5da6e704d767d6f70a41'},
               'record_sha256': '2f01160c573fd1b1cb5797c2cb78dd5a70ae1a55fc494c60402283a00475acca'}}


def _unwatched(tmp_path):
    space = adjoint_space(tmp_path / "unwatched")
    record = write_adjoint_checkpoint(
        space, boundary=BOUNDARY, session=_session(), cotangents=_plane(),
        shared_adjoint=_shared_adjoint(), shared_pass=_shared_pass())
    return _fingerprint(space, record)


def _owner_path(tmp_path):
    owner = StreamedBoundaryArtifacts({
        "schema": BOUNDARY_STORAGE_SCHEMA, "directory": str(tmp_path / "entries"),
        "max_resident_bytes": 1 << 16, "max_auxiliary_bytes": 1 << 20,
        "max_artifact_bytes": 1 << 24, "prefetch_batches": 2})
    owner.bind({"fixture": "default-path-identity"}, n_probes=2, published=True)
    references = {key: owner.write(tensor, probe_index=key[0], batch_index=key[1],
                                   boundary_index=BOUNDARY)
                  for key, tensor in _plane().items()}
    space = adjoint_space(tmp_path / "owner")
    record = write_adjoint_checkpoint(
        space, boundary=BOUNDARY, session=_session(), cotangents=references,
        shared_adjoint=_shared_adjoint(), shared_pass=_shared_pass(), owner=owner)
    commitment = owner.checkpoint_commitment(record["cotangent_sha256"])
    commitment["checkpoint_dir"] = commitment["checkpoint_dir"].replace(str(space), "<space>")
    assert commitment["receipt_digest"] == record["cotangent_sha256"]
    commitment["receipt_digest"] = "<digest>"
    ledgers = {key: owner.telemetry[key] for key in (
        "live_artifact_bytes", "live_checkpoint_bytes", "written_entries",
        "retired_entries", "checkpoint_envelope_unused_bytes")}
    # Every referenced entry is still an ordinary live file after the copy.
    assert all(Path(reference.path).is_file() for reference in references.values())
    return _fingerprint(space, record), commitment, ledgers


def test_default_path_matches_origin_main(tmp_path):
    observed = {"unwatched": _unwatched(tmp_path)}
    observed["owner"], observed["owner_commitment"], observed["owner_ledgers"] = (
        _owner_path(tmp_path))
    assert observed == EXPECTED, json.dumps(observed, sort_keys=True)
