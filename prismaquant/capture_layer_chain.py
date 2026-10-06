"""Streamed calibration capture as retryable layer-chain quanta (PQ #1885).

The streamed capture visits every source layer once, in order, and writes
each layer's units through :class:`~prismaquant.tessera_calibration_cache.CaptureWriter`.
On a large model that is one long row: an interruption can require replaying
the traversal. Newly written entries are sealed without readback. This module cuts the
same traversal at layer boundaries into PrismaBuild rows that run one after
another, each retryable on its own:

* **A prep row, run once.** It computes the capture's traversal identity
  (everything but the source digests, PQ #1896) without reading a payload,
  records the stat fingerprint of every source file, mints the boundary
  generation the quanta share, and seals the prep record
  (``<capture>/chain/prep.json``): the identity, the layer ranges, the
  calibration batch count, the boundary storage policy and session, and the
  source fingerprints. It runs no forward and hashes no source payload.
* **Quanta, one per layer range ``[a, b)``, in order.** Each rebinds the
  generation as its own owner (``owners/capture-AAA-BBB.json``) and reads its
  source through a recording
  :class:`~prismaquant.tessera_calibration_cache.CaptureSourceAuthentication`
  bound to the prep: every file it consumes (the head shards and its own
  layers' shards, plus the small metadata files) is hashed once through the
  held descriptor its tensors are then read through, before its first tensor
  reaches the capture, and held to the prep's stat fingerprint. A census
  that declares producer digests is compared there. Nothing it does not read
  is hashed, and no stat or path record stands in for a digest; the prep's
    fingerprints refuse in certified mode; dev mode stamps their drift. It starts
    from boundary ``a`` (the predecessor's
  hidden states, read through the generation's verified windows) and runs
  ``[a, b)`` with the unchanged capture visitor, so its units are written by
  the same writer and journal as a monolithic capture's, each hashed as it
  is written. It writes boundary ``b`` for its successor and records its
  fragment: the boundary ``b`` entries, its selected initialization witness,
  each unit's record with the stat fingerprint taken when it was written,
  and the source digests it recorded. A quantum never deletes its input
  boundary.
* **One join.** It requires the prep's ranges to tile the source layers and
  every owner to be complete, merges the selected witnesses into the full
  traversal's initialization contract and holds it to the census contract,
  and unions the quanta's recorded source digests: two quanta that read one
  file and recorded different digests refuse. It hashes only the files no
  quantum read (MTP sidecars, tokenizer assets), binds the complete roster
  into the sealed identity, and publishes the manifest from the units'
  records: each entry is held to the fingerprint its writer took, and no
  entry is read. Then it retires the interior boundaries.

The capture written this way is the monolith's, entry for entry: the same
identity, the same journal, the same per-unit files and the same manifest.
Each owner hashes a consumed file once through its held descriptor (Refs #1896).
Head shards and shards spanning ranges can recur across quanta; physical tensor
rereads also remain possible when the kernel reclaims retained clean pages.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import re

from .cost_stage_checkpoint import (
    atomic_write_bytes,
    canonical_json,
    canonical_json_sha256,
    publish_new_bytes,
)
from .digests import bytes_sha256hex, canonical_json_bytes, indent2_json_file_bytes
from .qnames import DOTTED_LAYER_QNAME

#: v2 (PQ #1896): the prep seals the traversal identity, which binds no
#: source digests; the join binds the digests the quanta recorded. A v1 prep
#: (whose identity the prep had hashed) is refused, never resumed.
PREP_SCHEMA = "prismaquant.capture_layer_chain.prep.v2"
FRAGMENT_SCHEMA = "prismaquant.capture_layer_chain.fragment.v1"
JOIN_SCHEMA = "prismaquant.capture_layer_chain.join.v1"
BIND_SCHEMA = "prismaquant.capture_layer_chain.bind.v1"
PREP_OWNER_LABEL = "capture-prep"
ROLES = ("prep", "quantum", "join")


class CaptureChainRefused(RuntimeError):
    """The capture chain cannot run or join without losing or corrupting bytes."""


# -- layout --------------------------------------------------------------------

def chain_root(capture_root) -> Path:
    return Path(capture_root) / "chain"


def prep_path(capture_root) -> Path:
    return chain_root(capture_root) / "prep.json"


def join_path(capture_root) -> Path:
    return chain_root(capture_root) / "join.json"


def range_label(start: int, stop: int) -> str:
    return f"capture-{start:03d}-{stop:03d}"


def capture_fragment_path(capture_root, start: int, stop: int) -> Path:
    return chain_root(capture_root) / f"{range_label(start, stop)}.fragment.json"


fragment_path = capture_fragment_path


def capture_generation_directory(prep) -> Path:
    return Path(prep["boundary_storage"]["directory"]) / str(prep["session"]["generation"])


generation_directory = capture_generation_directory


def capture_owner_status_path(prep, start: int, stop: int) -> Path:
    return generation_directory(prep) / "owners" / f"{range_label(start, stop)}.json"


owner_status_path = capture_owner_status_path


# -- layer ranges --------------------------------------------------------------

_RANGE = re.compile(r"(\d+):(\d+)")


def parse_layer_range(text) -> tuple[int, int]:
    """``A:B`` with ``0 <= A < B``: the source layers ``[A, B)``."""
    match = _RANGE.fullmatch(str(text).strip())
    if match is None:
        raise CaptureChainRefused(f"a capture layer range is A:B, not {text!r}")
    start, stop = int(match[1]), int(match[2])
    if not 0 <= start < stop:
        raise CaptureChainRefused(f"a capture layer range needs 0 <= A < B, not {text!r}")
    return start, stop


def parse_layer_ranges(text) -> list[tuple[int, int]]:
    """``A:B,B:C,...`` in order; tiling is :func:`require_layer_tiling`'s."""
    parts = [part for part in str(text).split(",") if part.strip()]
    if not parts:
        raise CaptureChainRefused("a capture chain names at least one layer range")
    return [parse_layer_range(part) for part in parts]


def require_layer_tiling(ranges, *, num_layers=None) -> list[tuple[int, int]]:
    """Refuse ranges that do not cover ``[0, num_layers)`` exactly once, in order.

    A gap would leave a layer's units uncaptured; an overlap would let two
    quanta write one unit and two boundaries claim one layer.
    """
    pairs = []
    for pair in ranges:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(type(value) is not int for value in pair)):
            raise CaptureChainRefused(f"a capture layer range is two integers, not {pair!r}")
        pairs.append((pair[0], pair[1]))
    if not pairs:
        raise CaptureChainRefused("a capture chain names at least one layer range")
    expected = 0
    for start, stop in pairs:
        if start < expected:
            raise CaptureChainRefused(f"capture layer range {start}:{stop} overlaps its predecessor")
        if start > expected:
            raise CaptureChainRefused(f"capture layer ranges leave layers {expected}:{start} uncovered")
        if stop <= start:
            raise CaptureChainRefused(f"capture layer range {start}:{stop} is empty")
        expected = stop
    if num_layers is not None and expected != num_layers:
        raise CaptureChainRefused(
            f"capture layer ranges cover {expected} layers; the source has {num_layers}")
    return pairs


# -- source fingerprints -------------------------------------------------------

def source_fingerprints(source_root) -> dict:
    """The stat fingerprint of every file a capture identity hashes."""
    from .cost_streaming import _streamed_identity_stat_fingerprint
    from .tessera_calibration_cache import capture_source_files
    return {path.name: _streamed_identity_stat_fingerprint(path)
            for path in capture_source_files(source_root)}


def require_source_fingerprints(recorded, source_root, *, where) -> None:
    """Compare recorded prep/source metadata, never authenticate payload bytes.

    Each quantum records digests from its own reads under same-descriptor
    integrity checks. D32 prep-vs-running stat or roster drift stamps and
    continues; certified mode retains the original refusal.
    """
    from .cost_streaming import stat_fingerprint_reusable
    from .dev_mode import seal_check
    live = source_fingerprints(source_root)
    seal_check('capture prep source roster', set(recorded), set(live), where=where,
        refusal=lambda: CaptureChainRefused(f'{where}: the source file roster changed since the prep'))
    changed = sorted(name for name, value in live.items()
                     if name not in recorded or not stat_fingerprint_reusable(value, recorded[name]))
    seal_check('capture prep source stat', recorded, live, where=where, same=not changed,
        refusal=lambda: CaptureChainRefused(
            f'{where}: source files changed since the prep hashed them: {changed[:8]}'))


# -- the prep record -----------------------------------------------------------

def bind_identity(identity, ranges, n_batches) -> dict:
    """What the boundary generation's session seals."""
    return {"schema": BIND_SCHEMA,
            "capture_identity_sha256": canonical_json_sha256(identity, where="capture identity"),
            "ranges": [list(pair) for pair in ranges], "n_batches": int(n_batches)}


def _seal_capture_document(body, *, where, field):
    body = canonical_json(body, where=where)
    return {**body, field: canonical_json_sha256(body, where=where)}


_seal = _seal_capture_document


def _read_capture_document(path, *, schema, field, what):
    try:
        document = json.loads(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        raise CaptureChainRefused(f"no readable {what} at {path}") from exc
    if not isinstance(document, dict) or document.get("schema") != schema:
        raise CaptureChainRefused(f"{path} is not a {schema} document")
    body = {key: value for key, value in document.items() if key != field}
    if canonical_json_sha256(body, where=what) != document.get(field):
        raise CaptureChainRefused(f"{path} does not seal its own content")
    return document


_read_sealed = _read_capture_document


def read_prep(capture_root) -> dict:
    """The prep record, its own seal checked."""
    return _read_sealed(prep_path(capture_root), schema=PREP_SCHEMA,
                        field="prep_sha256", what="capture chain prep")


def authenticate_quantum_source(capture_root, *, census_path, model, resource_check=None,
                                release_read_pages=False):
    """The quantum's recording source owner, bound to the prep (PQ #1896).

    Header inspection may open any shard. A payload read hashes its shard
    once, through the held descriptor its tensors are then read through, and
    the owner records the digest; every file it opens must still be the object
    the prep stat. The join binds what the quanta recorded.
    """
    from .tessera_calibration_cache import record_capture_source
    prep = read_prep(capture_root)
    census = json.loads(Path(census_path).read_text())
    if census.get("model") != str(model) or prep["source_root"] != str(model):
        raise CaptureChainRefused("the quantum's model is not the source the census and the prep name")
    return record_capture_source(census_path, model=model, binding_sha256=prep["prep_sha256"],
        fingerprints=prep["source_fingerprints"], resource_check=resource_check,
        release_read_pages=release_read_pages)


def prepare_capture_chain(capture_root, *, census_path, ranges, n_batches, boundary_storage, identity):
    """The prep row: seal the traversal identity, the ranges and the source's stat once.

    ``identity`` is called once, between two fingerprint passes over the
    source; both passes must agree. It returns the traversal identity, which
    binds no source digests: the quanta record those from their own reads
    and the join binds them (PQ #1896), so the prep reads no payload. Refuses
    when the chain is already prepped.
    """
    from .cost_streaming import StreamedBoundaryArtifacts
    ranges = require_layer_tiling(ranges)
    if type(n_batches) is not int or n_batches < 1:
        raise CaptureChainRefused("a capture chain needs at least one calibration batch")
    root = Path(capture_root).resolve()
    if prep_path(root).exists():
        raise CaptureChainRefused(f"{prep_path(root)} exists: a capture chain is prepped once")
    source_root = str(json.loads(Path(census_path).read_text())["model"])
    before = source_fingerprints(source_root)
    captured = identity()
    if source_fingerprints(source_root) != before:
        raise CaptureChainRefused("the source changed while the prep sealed it")
    if "source_files" in captured:
        raise CaptureChainRefused(
            "a capture chain prep seals the traversal identity; its quanta record the source")
    session_identity = bind_identity(captured, ranges, n_batches)
    storage = StreamedBoundaryArtifacts(boundary_storage)
    with storage:
        storage.bind(session_identity, n_probes=0, published=True, owner_label=PREP_OWNER_LABEL)
        document = _seal({"schema": PREP_SCHEMA, "identity": captured,
                          "ranges": [list(pair) for pair in ranges], "n_batches": n_batches,
                          "boundary_storage": storage.config, "session": storage.session,
                          "bind_identity": session_identity, "source_root": source_root,
                          "source_fingerprints": before},
                         where="capture chain prep", field="prep_sha256")
        path = prep_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = indent2_json_file_bytes(document)
        if not publish_new_bytes(path, payload):
            raise CaptureChainRefused(f"{path} exists: a capture chain is prepped once")
    return {"path": str(path), "sha256": bytes_sha256hex(payload), "document": document}


prepare = prepare_capture_chain


# -- fragments and owners ------------------------------------------------------

def _owner_status(prep, start, stop):
    path = owner_status_path(prep, start, stop)
    try:
        return json.loads(path.read_bytes())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise CaptureChainRefused(f"unreadable owner status at {path}") from exc


def require_owner_complete(prep, start, stop) -> None:
    status = _owner_status(prep, start, stop)
    if (status is None or status.get("session") != prep["session"]
            or status.get("status") != "complete"):
        state = None if status is None else status.get("status")
        raise CaptureChainRefused(
            f"capture layers {start}:{stop} have owner status {state!r}, not a complete "
            "owner of this chain's generation")


def read_fragment(capture_root, prep, start, stop) -> dict:
    """One quantum's fragment, sealed and bound to this prep."""
    document = _read_sealed(fragment_path(capture_root, start, stop), schema=FRAGMENT_SCHEMA,
                            field="fragment_sha256", what="capture chain fragment")
    if (document.get("prep_sha256") != prep["prep_sha256"]
            or document.get("layers") != [start, stop]
            or document.get("session") != prep["session"]
            or document.get("n_batches") != prep["n_batches"]):
        raise CaptureChainRefused(
            f"{fragment_path(capture_root, start, stop)} is not layers {start}:{stop} of this chain")
    return document


class _BoundaryFrontier:
    """The hidden states at boundary ``layer``, read in the generation's windows.

    Each window is one verified prefetch of ``prefetch_batches`` entries; a
    tensor handed out stays valid after its window closes, because the
    reader copies every tensor out of its scratch buffer.
    """

    def __init__(self, storage, references, *, layer):
        self.storage = storage
        self.references = tuple(references)
        self.layer = layer

    def hidden_batches(self):
        size = self.storage.config["prefetch_batches"]
        for first in range(0, len(self.references), size):
            window_references = self.references[first:first + size]
            with self.storage.prefetch(window_references) as window:
                for reference in window_references:
                    yield self.storage.get(window, reference)


def quantum_range_requires_units(identity: Mapping, start: int, stop: int) -> bool:
    """Whether layers ``[start, stop)`` hold units this capture must record.

    A **full-scope** capture (no declared unit scope) tiles every source
    layer, so every range must verify units. A selected capture's quanta
    still tile every source layer, but a range none of its declared units
    live in verifies an empty unit map and that is its complete record --
    only when every declared unit names exactly one decoder layer: a unit
    with no ``.layers.N.`` component, or more than one, names no single
    layer a quantum could own, so the chain refuses rather than assume one;
    any declared unit inside the range demands its record.
    """
    selected = identity.get("unit_scope") == "selected"
    requires = bool(not selected and identity.get("units"))
    for name in identity.get("units", {}):
        first = DOTTED_LAYER_QNAME.search(name)
        # Components can share a separator: .layers.0.layers.1. has two.
        second = None if first is None else DOTTED_LAYER_QNAME.search(name, first.start() + 1)
        if first is None or second is not None:
            if selected:
                raise CaptureChainRefused(
                    f"selected unit {name!r} names no unambiguous decoder layer; "
                    "the chain cannot tell which quantum records it")
            requires = True
            continue
        if start <= int(first.group(1)) < stop:
            requires = True
    return requires


def require_prep_identity(prep, identity, *, n_batches, label):
    """One comparability rule for fresh traversal and completed-result adoption."""
    recorded = prep["identity"]
    if (not isinstance(identity, dict) or identity.keys() != recorded.keys()
            or any(value != identity[key] for key, value in recorded.items()
                   if key != "capture_runtime")):
        raise CaptureChainRefused("this quantum's capture identity differs from the prep's")
    if n_batches != prep["n_batches"]:
        raise CaptureChainRefused(
            f"this quantum draws {n_batches} calibration batches; the prep sealed {prep['n_batches']}")
    from .dev_mode import seal_check
    seal_check("capture quantum runtime", recorded.get("capture_runtime"),
               identity.get("capture_runtime"), where=label,
               refusal=lambda: CaptureChainRefused(
                   "this quantum's capture runtime differs from the prep's"))
    return recorded


class ChainQuantum:
    """One layer range's owner of the chain's boundary generation.

    Construction checks everything a quantum can check before its forward:
    the sealed prep, its own range in the prep's tiling of this source, its
    own owner not already complete, its predecessor's owner complete with a
    fragment, the source files' stat fences, and that its runner reads
    through a recording descriptor owner bound to this prep.
    """

    def __init__(self, capture_root, layers, *, num_layers, source_authentication):
        self.root = Path(capture_root).resolve()
        self.prep = read_prep(self.root)
        if (not getattr(source_authentication, "is_recording", False) or
                getattr(source_authentication, "manifest_sha256", None) != self.prep["prep_sha256"]):
            raise CaptureChainRefused(
                "a capture chain quantum reads its source through the prep's recording owner")
        self.source_authentication = source_authentication
        self.start, self.stop = (int(value) for value in layers)
        self.num_layers = int(num_layers)
        ranges = require_layer_tiling([tuple(pair) for pair in self.prep["ranges"]],
                                      num_layers=self.num_layers)
        if (self.start, self.stop) not in ranges:
            raise CaptureChainRefused(
                f"capture layers {self.start}:{self.stop} are not a range of this chain")
        self.label = range_label(self.start, self.stop)
        status = _owner_status(self.prep, self.start, self.stop)
        if status is not None and status.get("status") == "complete":
            raise CaptureChainRefused(
                f"capture layers {self.start}:{self.stop} are already complete; a quantum runs once")
        self.inputs = None
        index = ranges.index((self.start, self.stop))
        if index:
            before = ranges[index - 1]
            require_owner_complete(self.prep, *before)
            records = read_fragment(self.root, self.prep, *before)["boundary"]
            if not isinstance(records, list) or len(records) != self.prep["n_batches"]:
                raise CaptureChainRefused(
                    f"capture layers {before[0]}:{before[1]} left no boundary {self.start}")
            from .joint_adjoint_checkpoints import reference_from_record
            self.inputs = [reference_from_record(record) for record in records]
            for batch, reference in enumerate(self.inputs):
                identity = json.loads(reference.metadata_json)["identity"]
                if (identity.get("session") != self.prep["session"]
                        or identity.get("slot") != f"boundary-{batch}-{self.start}"):
                    raise CaptureChainRefused(
                        f"boundary {self.start} entry {batch} is not this chain's")
        require_source_fingerprints(self.prep["source_fingerprints"], self.prep["source_root"],
                                    where=f"capture layers {self.start}:{self.stop}")
        self.storage = None
        self.outputs = []

    @property
    def last(self) -> bool:
        return self.stop == self.num_layers

    def require_identity(self, identity, *, n_batches) -> dict:
        return require_prep_identity(self.prep, identity, n_batches=n_batches,
                                     label=range_label(self.start, self.stop))

    def _remove_stale_outputs(self):
        """A failed attempt's boundary ``stop`` entries: this owner's, never its input."""
        if self.last:
            return
        from .perturbed_x_cache import activation_cache_filename
        entries = generation_directory(self.prep) / "entries"
        for batch in range(self.prep["n_batches"]):
            path = entries / activation_cache_filename(f"boundary-{batch}-{self.stop}-at-{self.stop}")
            for candidate in (path, path.with_suffix(".pt.tmp")):
                candidate.unlink(missing_ok=True)

    @contextmanager
    def owner(self):
        """Own the generation for this range; the status is ``complete`` on a clean exit."""
        from .cost_streaming import StreamedBoundaryArtifacts
        storage = StreamedBoundaryArtifacts(self.prep["boundary_storage"])
        with storage:
            storage.rebind(self.prep["session"], identity=self.prep["bind_identity"],
                           n_probes=0, owner_label=self.label)
            self._remove_stale_outputs()
            if self.inputs is not None:
                storage.authorize_forward_inputs(self.inputs)
            self.storage = storage
            try:
                yield self
            finally:
                self.storage = None

    def frontier(self):
        """The traversal's start, or ``None`` for the range that starts at layer 0."""
        if self.inputs is None:
            return None
        return _BoundaryFrontier(self.storage, self.inputs, layer=self.start)

    def boundary_consumer(self):
        """Write boundary ``stop`` for the successor, in batch order; none for the last range."""
        if self.last:
            return None

        def consume(index, hidden):
            if index != len(self.outputs):
                raise CaptureChainRefused("capture chain boundary batches arrived out of order")
            self.outputs.append(self.storage.write(hidden, batch_index=index,
                                                   boundary_index=self.stop))
        return consume

    def complete(self, *, witness, verified) -> Path:
        """Record this range's fragment; call inside :meth:`owner`."""
        from .joint_adjoint_checkpoints import exact_entry_record
        if self.storage is None:
            raise CaptureChainRefused("a quantum records its fragment while it owns the generation")
        if len(self.outputs) != (0 if self.last else self.prep["n_batches"]):
            raise CaptureChainRefused(
                f"capture layers {self.start}:{self.stop} wrote {len(self.outputs)} boundary "
                f"entries for {self.prep['n_batches']} batches")
        if witness.get("observed_layers") != list(range(self.start, self.stop)):
            raise CaptureChainRefused("a quantum's witness names other layers than its range")
        if not verified and quantum_range_requires_units(
                self.prep["identity"], self.start, self.stop):
            raise CaptureChainRefused(f"capture layers {self.start}:{self.stop} verified no unit")
        document = _seal({"schema": FRAGMENT_SCHEMA, "prep_sha256": self.prep["prep_sha256"],
                          "session": self.prep["session"], "layers": [self.start, self.stop],
                          "num_layers": self.num_layers, "n_batches": self.prep["n_batches"],
                          "boundary": None if self.last else
                              [exact_entry_record(reference) for reference in self.outputs],
                          "witness": witness, "units": verified,
                          "source_authentication": self.source_authentication.receipt()},
                         where="capture chain fragment", field="fragment_sha256")
        path = fragment_path(self.root, self.start, self.stop)
        atomic_write_bytes(path, canonical_json_bytes(document, where="capture chain fragment") + b"\n")
        return path


# -- the join ------------------------------------------------------------------

def recorded_source_digests(prep, fragments) -> dict:
    """The union of the source digests the quanta recorded, ``{file name: sha256}``.

    Each quantum hashed what it read through the held descriptor it read it
    through (PQ #1896). Two quanta that read one file and recorded different
    digests read different bytes under one capture, and the join refuses.
    """
    from .tessera_calibration_cache import RECORDING_RECEIPT_SCHEMA
    recorded, reader = {}, {}
    for fragment in fragments:
        receipt = fragment.get("source_authentication")
        label = range_label(*fragment["layers"])
        if (not isinstance(receipt, dict) or receipt.get("schema") != RECORDING_RECEIPT_SCHEMA
                or receipt.get("binding_sha256") != prep["prep_sha256"]
                or not isinstance(receipt.get("verified_files"), list)):
            raise CaptureChainRefused(f"capture layers {label} recorded no source receipt of this prep")
        for row in receipt["verified_files"]:
            name, digest = row.get("name"), row.get("sha256")
            if recorded.setdefault(name, digest) != digest:
                raise CaptureChainRefused(
                    f"{name}: capture quanta {reader[name]} and {label} read different source bytes")
            reader.setdefault(name, label)
    return recorded


def join(capture_root, *, census_path) -> dict:
    """Publish the complete capture from the quanta's records.

    Reads no entry: each is held to the stat fingerprint its writer took. The
    source digests the quanta recorded are unioned and bound; only the files
    no quantum read are hashed (PQ #1896). The interior boundaries are retired
    only after the manifest is published, so a failed join leaves every input
    of a retry.
    """

    from .streaming_model import merge_selected_initialization_witnesses
    from .tessera_calibration_cache import (
        CaptureWriter, record_capture_source, require_capture_initialization_contract, sha256)
    root = Path(capture_root).resolve()
    prep = read_prep(root)
    identity = prep["identity"]
    if sha256(census_path) != identity["census_sha256"]:
        raise CaptureChainRefused("the join's census is not the one the prep sealed")
    ranges = require_layer_tiling(prep["ranges"])
    for start, stop in ranges:
        require_owner_complete(prep, start, stop)
    fragments = [read_fragment(root, prep, start, stop) for start, stop in ranges]
    counts = {fragment["num_layers"] for fragment in fragments}
    if len(counts) != 1:
        raise CaptureChainRefused("the quanta ran over sources of different depth")
    require_layer_tiling(ranges, num_layers=counts.pop())
    require_source_fingerprints(prep["source_fingerprints"], prep["source_root"],
                                where="capture chain join")
    merged = merge_selected_initialization_witnesses(
        [fragment["witness"] for fragment in fragments])
    try:
        require_capture_initialization_contract(identity["model_load_contract"], merged)
    except RuntimeError as error:
        raise CaptureChainRefused(str(error)) from error
    verified = {}
    for fragment in fragments:
        repeated = sorted(set(verified) & set(fragment["units"]))
        if repeated:
            raise CaptureChainRefused(f"two quanta verified one unit: {repeated[:8]}")
        verified.update(fragment["units"])
    if set(verified) != set(identity["units"]):
        missing = sorted(set(identity["units"]) - set(verified))
        raise CaptureChainRefused(f"the quanta verified no record for {missing[:8]}")
    recorded = recorded_source_digests(prep, fragments)
    # The join reads no payload a quantum read: it binds their digests to the
    # objects the prep stat, and hashes only what no quantum consumed.
    with record_capture_source(census_path, model=prep["source_root"],
                               binding_sha256=prep["prep_sha256"],
                               fingerprints=prep["source_fingerprints"],
                               release_read_pages=True) as source:
        source.adopt_recorded_digests(recorded)
        source_receipt = source.authenticate_complete_source()
        source_files = source.recorded_source_files()
    writer = CaptureWriter(root, census_path=census_path, identity=identity)
    receipt = writer.finish(model_load_contract=merged, verified=verified,
                            source_files=source_files)
    retired = retire_interior_boundaries(prep, fragments[:-1])
    document = {"schema": JOIN_SCHEMA, "prep_sha256": prep["prep_sha256"],
                "manifest": receipt, "source_authentication": source_receipt,
                "retired_boundary_entries": retired}
    atomic_write_bytes(join_path(root), indent2_json_file_bytes(document))
    return document


def retire_interior_boundaries(prep, fragments) -> int:
    """Unlink the boundary entries the quanta passed along; the join's alone.

    The generation's owners never unlink a published generation's entries,
    so the join removes them by exact path, each held to the generation's own
    entries directory first. Missing files are a retried join's.
    """
    entries = (generation_directory(prep) / "entries").resolve()
    retired = 0
    for fragment in fragments:
        for record in fragment["boundary"] or ():
            path = Path(record["path"])
            if path.parent.resolve() != entries:
                raise CaptureChainRefused(f"{path} is outside the chain's boundary generation")
            if path.exists():
                path.unlink()
                retired += 1
    return retired
