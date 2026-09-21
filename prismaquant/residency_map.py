"""Read declared manifest bytes from PrismaBuild's stage tier when it has them.

PrismaBuild's tiered caching (RobTand/prismabuild#583) stages the byte ranges a
consumer's data manifest names onto an SSD stage tier, composes a residency map
from its movers' fragments and injects the map's path as
``PRISMABUILD_RESIDENCY_MAP``. This module is the consumer half. Without a
reader the stage is a copy nobody reads, and the before/after says exactly that.

The schema is PrismaBuild's, not ours. These field rules mirror
``src/prismabuild/residency_map.py`` at ``182166a04d`` on
``flash/583-movers-egress-20260918``, field for field; nothing here imports
``prismabuild``, so the rules are re-stated rather than shared, and a drift
between the two is a refusal here rather than a wrong read.

One resolver, one store (principle 8). Every read site that can be served from
the stage asks this module, and the bytes on either side of the redirect are
counted here, so the run's ``results.json`` can say what the stage actually
served. The resolver owns no bytes: it answers "where do I open this", and the
caller's own digest decides whether the answer was good.

The identity rules, in order:

* Without the variable the resolver is inert. No map is read, no path is
  rewritten and the accounting block is absent, so the bytes and the behaviour
  are what they are today.
* A map is bound to one read set. Its ``manifest_sha256`` has to equal the
  digest of the data manifest this process was submitted with, and a process
  that bound no manifest digest gets no redirect at all. A map naming another
  action's bytes is the failure this rule exists for.
* A map that is missing, unreadable, not this schema, or carrying a field this
  reader does not know is refused **whole**, with a reason, and every read
  falls back to its declared path. The variable is injected at claim time while
  movers are still running, so an absent map is ordinary: it is retried on the
  next resolve rather than being fatal.
* The entry key is ``(path, offset)``, spelled ``"<offset>:<path>"``, because a
  data manifest may name one path at several offsets. A reader of whole files
  asks ``staged_read`` for offset 0 and is refused an entry covering less than
  the declared file. A reader of one tensor's span out of a shard asks
  ``staged_range`` for that span and is served the single entry covering it
  outright, or none: a span straddling two staged ranges is a miss here, and
  the caller reads its declared path rather than a concatenation nobody fenced.
* An entry is used only after a per-entry check: the digest the map publishes
  has to equal the digest the caller already requires (when it has one), the
  staged file has to be a regular file of exactly the entry's byte count, and
  the declared file has to be that length too. The caller then verifies the
  bytes it actually read against the same digest and refuses the entry on a
  mismatch.
* A refused entry falls back to the declared path and is recorded with a
  reason. Nothing here is silent.
* A map of the ram-overlay generation (RobTand/prismabuild#640) adds the
  optional halves: the header may announce ``ram_tier_id``, ``ram_root`` and
  ``ram_epoch``, and an entry may name a ``ram_path`` under that root -- the
  tmpfs copy of a range the stage entry already vouches for. A tmpfs empties
  on reboot while the map survives on the shared mount, so the ram half is
  offered only while the map's ``ram_epoch`` equals the epoch the pool's tier
  record announces for ``ram_tier_id``. No record, an unreadable record, a
  missing epoch or a mismatched one fails closed **on the ram half only**: the
  entry still answers from its stage copy, then the declared path, exactly the
  chain above. The ram file passes the same pre-open fence as the stage copy,
  and a caller whose ram read refuses falls back to the stage copy before the
  declared path -- a dead tmpfs is a cache miss, never an ENOENT.

The map is replaced atomically as movers finish, so it is re-read when its
identity changes rather than loaded once. Reads start on the first resident
entry; the resolver never waits for the map to be complete.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import os
from pathlib import Path
import stat
import threading

SCHEMA = "prismaquant.prismabuild.residency_map.v1"
FRAGMENT_SCHEMA = "prismaquant.prismabuild.residency_map_fragment.v1"
ENV_VAR = "PRISMABUILD_RESIDENCY_MAP"
#: Where the pool's tier records are read from, ``<queue>/tiers`` by default
#: (the map lives at ``<queue>/residency/<consumer>.map.json``, so the tiers
#: directory is its sibling's sibling). The variable names it outright for a
#: layout -- or a test -- that does not match the queue's own.
TIERS_DIR_ENV_VAR = "PRISMABUILD_RESIDENCY_TIERS_DIR"
#: The schema of the tier records in that directory (PrismaBuild's
#: ``storage_tiers.TIER_RECORD_SCHEMA_V1``, restated rather than imported).
TIER_RECORD_SCHEMA = "prismabuild.storage_tier.v1"
# A whole-manifest map for the GLM-5.3-Flash prepare is ~469k entries; at this
# schema's per-entry size that is under 128 MiB. The bound exists so a wrong
# path cannot be read without limit, not to describe an expected size.
MAX_MAP_BYTES = 256 * 1024 * 1024
MAX_RECORDED_FALLBACKS = 256

#: ``ResidencyResolver.staged_range_outcome`` verdicts. An entry covering the
#: span that passed every check; no entry covering it (the map's ordinary
#: mid-flight state, which says nothing about whether the span is declared);
#: an entry covering it that then failed a check, already recorded as a
#: fallback. Named rather than spelled inline so a caller's branch reads as
#: the distinction it is making.
#: Larger than any byte offset a declared file can hold, so a bisect key of
#: ``(start, _INFINITE_OFFSET)`` orders after every real span beginning at
#: ``start``. Not a magic constant: 2**64 is past the addressable range of
#: any file the manifest can name.
_INFINITE_OFFSET = 1 << 64


def _merge_spans(rows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sorted, non-overlapping ``[start, end)`` spans.

    The manifest may declare one path at several offsets, and adjacent
    declarations are common (a shard's header entry abutting its payload
    entry). Merging keeps membership a single bisect rather than a scan,
    and makes a read that spans two abutting declarations declared -- which
    it is: the bytes are all named.
    """
    merged: list[tuple[int, int]] = []
    for low, high in sorted(rows):
        if merged and low <= merged[-1][1]:
            if high > merged[-1][1]:
                merged[-1] = (merged[-1][0], high)
        else:
            merged.append((low, high))
    return merged


RANGE_HIT = "hit"
RANGE_UNCOVERED = "uncovered"
RANGE_UNDECLARED = "undeclared"
RANGE_REFUSED = "refused"


_ROOT_KEYS = {"schema", "tier_id", "stage_root", "manifest_sha256", "leads",
              "generation", "entries"}
#: The ram overlay's optional header: which tier, under which root, in which
#: epoch. A map naming ram paths must announce all three (PrismaBuild's
#: ``validate_map`` refuses less), and a map naming none may carry them anyway.
_RAM_ROOT_KEYS = {"ram_tier_id", "ram_root", "ram_epoch"}
_ENTRY_KEYS = {"stage_path", "bytes", "offset", "sha256", "ram_path"}
_HEX = frozenset("0123456789abcdef")


class ResidencyMapRefused(Exception):
    """The map named by the environment is not a map this reader can use."""


class StagedReadRefused(Exception):
    """The staged copy failed its identity check. Read the declared path."""


def _is_hex64(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(char in _HEX for char in value))


def residency_map_key(path: str, offset: int = 0) -> str:
    """One manifest entry's identity, spelled as PrismaBuild's map spells it.

    Decimal offset, then a colon, then the path. The split is on the first
    colon and the offset is digits, so a path containing colons cannot collide
    with another entry.
    """
    return f"{int(offset)}:{path}"


def _normal(path: object) -> str:
    """The lookup key: the manifest's own spelling, normalized, never resolved.

    Data-manifest entry paths are absolute and already equal to their own
    ``posixpath.normpath``. Resolving here would be wrong in the common case;
    the resolved spelling is handled by a second index, built only if a lookup
    misses (see ``_real_key``).
    """
    return os.path.normpath(os.fspath(path))


def _identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


class ResidencyResolver:
    """Where to open a declared path, and what each tier actually served."""

    def __init__(self, map_path: str | Path, tiers_dir: str | Path | None = None):
        self._map_path = str(map_path)
        override = os.environ.get(TIERS_DIR_ENV_VAR)
        self._tiers_dir = str(tiers_dir if tiers_dir is not None else
                              override if override else
                              Path(self._map_path).parent.parent / "tiers")
        self._lock = threading.Lock()
        self._manifest_sha256: str | None = None
        self._identity: tuple[int, int, int, int] | None = None
        self._entries: dict[str, dict] = {}
        self._real_entries: dict[str, dict] | None = None
        self._real_dirs: dict[str, str] = {}
        self._map_sha256: str | None = None
        self._tier_id: str | None = None
        self._stage_root: str | None = None
        self._leads: tuple[str, ...] = ()
        self._generation: int | None = None
        self._refused: str | None = None
        # The ram half's state: what the map announced, and (identity-cached)
        # what the pool's tier record answers for the announced epoch.
        self._ram_tier_id: str | None = None
        self._ram_root: str | None = None
        self._ram_epoch: str | None = None
        self._ram_refusal: str | None = None
        self._tier_record_identity: tuple[int, int, int, int] | None = None
        self._tier_record: tuple[str | None, str | None] | None = None
        self._hits = 0
        self._misses = 0
        self._range_hits = 0
        self._range_misses = 0
        # Polls a strict reader spent waiting for a DECLARED range PrismaBuild
        # had published but not yet moved. Each poll is a real uncovered-span
        # lookup and is counted in ``range_misses`` like any other, so the true
        # number of distinct reads that found nothing is
        # ``range_misses - range_wait_polls``. Kept separate rather than
        # excluded so neither number is quietly redefined.
        # PB's sealed readset, read once, lazily, from the action's own pool
        # row (see _load_declared_readset). None until read; a dict of
        # {normalized path: sorted merged [start, end) spans} once bound.
        # A FAILED read is cached too, deliberately: the claim row and the
        # CAS blob live as long as the action does, so a failure is a
        # standing fact about this run, not a transient one, and retrying
        # it per uncovered span would re-walk the pool queue under the
        # resolver's lock. The cache is bounded by the binding it belongs
        # to -- bind_manifest_sha256 drops all three on a digest change.
        self._declared: dict[str, list[tuple[int, int]]] | None = None
        self._declared_reason: str | None = None
        self._declared_attempted = False
        self._range_wait_polls = 0
        self._range_wait_seconds = 0.0
        self._range_waits_served = 0
        self._range_waits_refused = 0
        self._intervals: dict[str, list[dict]] = {}
        self._real_intervals: dict[str, list[dict]] = {}
        self._intervals_for: dict[str, dict] | None = None
        self._bytes_from_stage = 0
        self._bytes_from_ram = 0
        self._ram_hits = 0
        self._ram_fallbacks: list[dict] = []
        self._ram_fallback_count = 0
        self._bytes_from_pool = 0
        self._fallbacks: list[dict] = []
        self._fallback_count = 0
        # The serving-tier record (ID-07/INV-04): where bytes were actually
        # served from, under which tier/epoch, noted at open time before
        # payload trust. Counters observe; this record authorizes nothing.
        # ``lease_id`` is None until the PB reader-lease API (RNG-02/SM-03)
        # lands — the gap is explicit, not hidden.
        self._serving_tiers: list[dict] = []
        self._serving_tier_count = 0

    # -- binding ---------------------------------------------------------

    def bind_manifest_sha256(self, digest: str) -> None:
        """Name the read set this process was submitted with.

        A map is a statement about one manifest. Until this is called the
        resolver refuses every map, so a launcher that injected the wrong
        action's map cannot redirect a single read.

        The sealed readset is a statement about one manifest too, so it is
        dropped here with the map identity. Keeping it would be worse than
        a stale cache: membership from manifest A, consulted for manifest
        B's paths, answers "not declared" for every one of them -- a clean,
        deliberate-looking immediate refusal that is exactly the failure
        this binding was added to prevent.
        """
        if not _is_hex64(digest):
            raise ValueError("a data manifest digest is 64 lowercase hex characters")
        with self._lock:
            if self._manifest_sha256 != digest:
                self._manifest_sha256 = digest
                self._identity = None
                self._refused = None
                self._declared = None
                self._declared_reason = None
                self._declared_attempted = False

    # -- the map ---------------------------------------------------------

    def _read_map(self) -> None:
        """Re-read the map when its identity changed. Caller holds the lock."""
        if self._manifest_sha256 is None:
            self._forget("no data manifest digest is bound to this process")
            return
        try:
            before = os.lstat(self._map_path)
        except OSError as error:
            self._forget(f"residency map is unreadable: {error.strerror}")
            return
        if not stat.S_ISREG(before.st_mode):
            self._forget("residency map is not a regular file")
            return
        identity = _identity(before)
        if identity == self._identity:
            return
        if before.st_size > MAX_MAP_BYTES:
            self._identity = identity
            self._forget("residency map exceeds the reader's byte bound",
                         keep_identity=True)
            return
        try:
            with open(self._map_path, "rb") as handle:
                raw = handle.read(before.st_size + 1)
                after = os.fstat(handle.fileno())
        except OSError as error:
            self._forget(f"residency map is unreadable: {error.strerror}")
            return
        if len(raw) != before.st_size or _identity(after) != identity:
            # A composer replaced it mid-read. Forget the identity so the next
            # resolve reads the replacement instead of trusting a torn copy.
            self._forget("residency map changed during its read")
            return
        self._identity = identity
        try:
            self._adopt(json.loads(raw.decode("utf-8")), raw)
        except (UnicodeError, json.JSONDecodeError):
            self._forget("residency map is not valid UTF-8 JSON", keep_identity=True)
        except ResidencyMapRefused as refusal:
            self._forget(str(refusal), keep_identity=True)

    def _forget(self, reason: str, *, keep_identity: bool = False) -> None:
        if not keep_identity:
            self._identity = None
        self._entries = {}
        self._real_entries = None
        self._map_sha256 = None
        self._tier_id = None
        self._stage_root = None
        self._leads = ()
        self._generation = None
        self._forget_ram_half()
        if self._refused != reason:
            self._refused = reason
            print(f"[residency] refused {self._map_path}: {reason}", flush=True)

    def _forget_ram_half(self) -> None:
        """Drop the ram half's state. Caller holds the lock."""
        self._ram_tier_id = None
        self._ram_root = None
        self._ram_epoch = None
        self._ram_refusal = None
        self._tier_record_identity = None
        self._tier_record = None

    def _adopt(self, payload: object, raw: bytes) -> None:
        if type(payload) is not dict:
            raise ResidencyMapRefused(
                f"residency map must be an object, not {type(payload).__name__}")
        unknown = sorted(set(payload) - _ROOT_KEYS - _RAM_ROOT_KEYS)
        if unknown:
            raise ResidencyMapRefused(f"unknown residency map fields: {unknown}")
        missing = sorted(_ROOT_KEYS - set(payload))
        if missing:
            raise ResidencyMapRefused(f"residency map is missing {missing}")
        if payload["schema"] != SCHEMA:
            raise ResidencyMapRefused(
                f"residency map declares schema {payload['schema']!r}, not {SCHEMA}")
        if not _is_hex64(payload["manifest_sha256"]):
            raise ResidencyMapRefused("residency map manifest_sha256 is not a digest")
        if payload["manifest_sha256"] != self._manifest_sha256:
            raise ResidencyMapRefused(
                "residency map names data manifest "
                f"{payload['manifest_sha256'][:12]}, this run reads "
                f"{self._manifest_sha256[:12]}")
        stage_root = payload["stage_root"]
        if (type(stage_root) is not str or not stage_root.startswith("/")
                or os.path.normpath(stage_root) != stage_root):
            raise ResidencyMapRefused("residency map stage_root is not a normalized absolute path")
        tier_id = payload["tier_id"]
        if type(tier_id) is not str or not tier_id or "/" in tier_id:
            raise ResidencyMapRefused("residency map tier_id is not a tier id")
        leads = payload["leads"]
        if type(leads) is not list or any(not _is_hex64(lead) for lead in leads):
            raise ResidencyMapRefused("residency map leads must be 64-character action keys")
        if len(set(leads)) != len(leads):
            raise ResidencyMapRefused("residency map leads repeat a key")
        generation = payload["generation"]
        if type(generation) is not int or isinstance(generation, bool) or generation < 0:
            raise ResidencyMapRefused("residency map generation must be a count")
        ram_tier_id, ram_root, ram_epoch = self._ram_header(payload)
        entries = payload["entries"]
        if type(entries) is not dict:
            raise ResidencyMapRefused(
                "residency map entries must be an object keyed by '<offset>:<path>', not "
                f"{type(entries).__name__} (a ranged roster of rows is not this schema)")
        prefix = stage_root.rstrip("/") + "/"
        adopted: dict[str, dict] = {}
        for key, row in entries.items():
            adopted[str(key)] = self._entry(str(key), row, stage_root, prefix, ram_root)
        if any("ram_path" in entry for entry in adopted.values()) and (
                ram_tier_id is None or ram_root is None or ram_epoch is None):
            # A ram copy nobody can date is not resident (#640): the tmpfs
            # empties on reboot while the map survives, so an entry naming a
            # ram path without the tier, root and epoch that place it in time
            # is refused the way any other entry that says too little is.
            raise ResidencyMapRefused(
                "a residency map naming ram paths must announce its "
                "ram tier, root and epoch")
        self._entries = adopted
        self._real_entries = None
        self._map_sha256 = hashlib.sha256(raw).hexdigest()
        self._tier_id = tier_id
        self._stage_root = stage_root
        self._leads = tuple(leads)
        self._generation = generation
        self._ram_tier_id = ram_tier_id
        self._ram_root = ram_root
        self._ram_epoch = ram_epoch
        self._ram_refusal = None
        self._tier_record_identity = None
        self._tier_record = None
        if self._refused is not None:
            print(f"[residency] adopted {self._map_path}: {len(adopted)} entries "
                  f"on {tier_id}, generation {generation}", flush=True)
        self._refused = None

    @staticmethod
    def _ram_header(payload: dict) -> tuple[str | None, str | None, str | None]:
        """The map's optional ram announcement, with PrismaBuild's own rules.

        The three fields are independent optionals in the writer's schema --
        only an entry naming a ``ram_path`` requires all three -- and each is
        validated the way ``prismabuild.residency_map.validate_map`` validates
        it: the root a normalized absolute path, the tier id and the epoch
        non-empty strings without ``/``, because the epoch is a filename-safe
        identity the tier loop mints.
        """

        ram_root = payload.get("ram_root")
        if ram_root is not None and (
                type(ram_root) is not str or not ram_root.startswith("/")
                or os.path.normpath(ram_root) != ram_root):
            raise ResidencyMapRefused(
                "residency map ram_root is not a normalized absolute path")
        ram_tier_id = payload.get("ram_tier_id")
        if ram_tier_id is not None and (
                type(ram_tier_id) is not str or not ram_tier_id or "/" in ram_tier_id):
            raise ResidencyMapRefused("residency map ram_tier_id is not a tier id")
        ram_epoch = payload.get("ram_epoch")
        if ram_epoch is not None and (
                type(ram_epoch) is not str or not ram_epoch or "/" in ram_epoch):
            raise ResidencyMapRefused("residency map ram_epoch is not an epoch")
        return ram_tier_id, ram_root, ram_epoch

    @staticmethod
    def _entry(key: str, row: object, stage_root: str, prefix: str,
               ram_root: str | None = None) -> dict:
        head, separator, path = key.partition(":")
        if not separator or not path or not head.isdigit():
            raise ResidencyMapRefused(f"malformed residency map key {key!r}")
        offset = int(head)
        if type(row) is not dict:
            raise ResidencyMapRefused(f"residency map entry {key!r} must be an object")
        unknown = sorted(set(row) - _ENTRY_KEYS)
        if unknown:
            raise ResidencyMapRefused(f"unknown residency map entry fields: {unknown}")
        stage_path = row.get("stage_path")
        if (type(stage_path) is not str or not stage_path.startswith("/")
                or os.path.normpath(stage_path) != stage_path):
            raise ResidencyMapRefused(
                f"residency map entry {key!r} stage_path is not a normalized absolute path")
        if not (stage_path == stage_root or stage_path.startswith(prefix)):
            raise ResidencyMapRefused(
                f"residency map entry {key!r} is staged outside {stage_root!r}")
        size = row.get("bytes")
        if type(size) is not int or isinstance(size, bool) or size <= 0:
            raise ResidencyMapRefused(f"residency map entry {key!r} has no positive size")
        declared = row.get("offset", offset)
        if type(declared) is not int or isinstance(declared, bool) or declared < 0:
            raise ResidencyMapRefused(f"residency map entry {key!r} offset is not a count")
        if declared != offset:
            raise ResidencyMapRefused(
                f"residency map entry {key!r} offset {declared} disagrees with its key")
        if not _is_hex64(row.get("sha256")):
            raise ResidencyMapRefused(f"residency map entry {key!r} has no SHA-256 digest")
        checked = {"stage_path": stage_path, "bytes": size, "offset": offset,
                   "sha256": row["sha256"], "declared_path": path}
        ram_path = row.get("ram_path")
        if ram_path is not None:
            if ram_root is None:
                # PrismaBuild's own compose refuses this shape; a map that
                # carries it is a map this reader refuses whole, like every
                # other entry that does not say what the schema requires.
                raise ResidencyMapRefused(
                    f"residency map entry {key!r} names a ram_path, "
                    "but the map announces no ram root")
            if (type(ram_path) is not str or not ram_path.startswith("/")
                    or os.path.normpath(ram_path) != ram_path):
                raise ResidencyMapRefused(
                    f"residency map entry {key!r} ram_path is not a normalized absolute path")
            ram_prefix = ram_root.rstrip("/") + "/"
            if not (ram_path == ram_root or ram_path.startswith(ram_prefix)):
                # A map that could name a path outside the announced ram tier
                # is a map that could redirect a consumer's read anywhere.
                raise ResidencyMapRefused(
                    f"residency map entry {key!r} ram_path is outside "
                    f"the map's ram root {ram_root!r}")
            checked["ram_path"] = ram_path
        return checked

    def _real_key(self, key: str) -> dict | None:
        """Second index, for a caller that resolved symlinks and the map did not.

        The manifest's spelling is not resolved; several PrismaQuant readers
        spell the same file with ``Path.resolve()``. Building this index costs
        one ``realpath`` per distinct directory in the map, and it is built only
        after a lookup has already missed, so a run whose spellings agree never
        pays for it.
        """
        if self._real_entries is None:
            index: dict[str, dict] = {}
            for entry in self._entries.values():
                path = entry["declared_path"]
                directory, name = os.path.split(path)
                real = self._real_dirs.get(directory)
                if real is None:
                    try:
                        real = os.path.realpath(directory)
                    except OSError:
                        real = directory
                    self._real_dirs[directory] = real
                index[residency_map_key(os.path.join(real, name), entry["offset"])] = entry
            self._real_entries = index
        return self._real_entries.get(key)

    # -- the ram half -----------------------------------------------------

    def _announced_ram_epoch(self) -> tuple[str | None, str | None]:
        """The epoch the pool currently announces for the map's ram tier.

        Caller holds the lock. The record is the tier loop's own announcement
        (``<tiers>/<ram tier id>.json``, ``prismabuild.storage_tier.v1``), and
        its ``epoch`` is what the map's ``ram_epoch`` is compared against --
        the marker file it dates from dies with the tmpfs on reboot, so a
        record the map's generation predates announces an epoch the map's
        bytes cannot be in. The record is re-read when its stat identity
        changes rather than per lookup, so noticing a reboot costs one lstat.

        Any failure is a refusal naming why, never a guess: an epoch nobody
        announces is an epoch the ram half is not resident in.
        """

        record_path = Path(self._tiers_dir) / f"{self._ram_tier_id}.json"
        try:
            before = os.lstat(record_path)
        except OSError as error:
            return None, f"ram tier record is unreadable: {error.strerror}"
        if not stat.S_ISREG(before.st_mode):
            return None, "ram tier record is not a regular file"
        identity = _identity(before)
        if identity == self._tier_record_identity and self._tier_record is not None:
            return self._tier_record
        try:
            with open(record_path, "rb") as handle:
                raw = handle.read(MAX_MAP_BYTES + 1)
                after = os.fstat(handle.fileno())
        except OSError as error:
            return None, f"ram tier record is unreadable: {error.strerror}"
        if _identity(after) != identity:
            # Announced by atomic replace; a torn read is not a verdict.
            return None, "ram tier record changed during its read"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            return None, "ram tier record is not valid UTF-8 JSON"
        if type(payload) is not dict or payload.get("schema") != TIER_RECORD_SCHEMA:
            return None, "ram tier record is not a storage tier record"
        if payload.get("tier_id") != self._ram_tier_id:
            return None, "ram tier record names another tier than the map announced"
        epoch = payload.get("epoch")
        if type(epoch) is not str or not epoch or "/" in epoch:
            return None, "ram tier record announces no epoch"
        self._tier_record_identity = identity
        self._tier_record = (epoch, None)
        return self._tier_record

    def _ram_live(self) -> bool:
        """Is the map's ram half resident in the epoch the pool announces?

        Caller holds the lock. Only entries that carry a ``ram_path`` ask;
        a map that announces no ram half, or whose entries name none, never
        reads a tier record at all.
        """

        if self._ram_tier_id is None or self._ram_epoch is None:
            return False
        epoch, refusal = self._announced_ram_epoch()
        if epoch is None:
            reason = refusal
        elif epoch != self._ram_epoch:
            reason = (f"ram epoch {self._ram_epoch} is stale: "
                      f"the pool announces {epoch}")
        else:
            reason = None
        if reason != self._ram_refusal:
            self._ram_refusal = reason
            if reason is None:
                print(f"[residency] ram half resident on {self._ram_tier_id}, "
                      f"epoch {epoch}", flush=True)
            else:
                print(f"[residency] ram half not resident: {reason}", flush=True)
        return reason is None

    def _ram_offer(self, entry: dict, path: str) -> str | None:
        """The live ram copy of a bound entry, or None. Caller holds the lock.

        The pre-open fence is the stage copy's own: a regular file of exactly
        the entry's byte count. The digest need not be re-checked here -- it is
        the same entry, and the caller verifies the bytes it reads against the
        same digest either way. A refusal is recorded and falls through to the
        stage copy; it never fails the read.
        """

        ram_path = entry.get("ram_path")
        if ram_path is None or not self._ram_live():
            return None
        try:
            info = os.lstat(ram_path)
        except OSError as error:
            self._record_ram_fallback(path, f"ram copy is unreadable: {error.strerror}")
            return None
        if not stat.S_ISREG(info.st_mode):
            self._record_ram_fallback(path, "ram copy is not a regular file")
            return None
        if info.st_size != entry["bytes"]:
            self._record_ram_fallback(path, "ram copy size differs from the map")
            return None
        return ram_path

    # -- the answer ------------------------------------------------------

    def staged_read(self, declared: str | Path, *, offset: int = 0,
                    expected_sha256: str | None = None) -> dict | None:
        """Return where to open ``declared`` on the stage, or None for the pool.

        The pre-open checks are the cheap half of the identity: the digest the
        map publishes has to be the digest the caller already requires, the
        staged file has to be a regular file of exactly the entry's size, and
        the declared file has to be that size too, which is what binds the
        entry to the file it claims to stand for and is the only check
        available to a caller that has no digest yet. The caller still verifies
        the bytes it reads; this decides which copy is worth opening.
        """
        path = _normal(declared)
        key = residency_map_key(path, offset)
        with self._lock:
            self._read_map()
            entry = self._entries.get(key)
            if entry is None and self._entries:
                entry = self._real_key(key)
            if entry is None:
                self._misses += 1
                return None
            if expected_sha256 is not None and entry["sha256"] != expected_sha256:
                self._record_fallback(
                    path, "map digest differs from the digest this read requires")
                return None
            try:
                declared_info = os.lstat(path)
            except OSError:
                # The declared file's own length is what binds a whole-file
                # entry to this caller's read; without it the entry could be a
                # byte range served as a whole file, and a pool path that has
                # gone away would succeed from the stage where reading it
                # directly fails closed. The map redirects a read; it does not
                # substitute for one.
                self._record_fallback(
                    path, "declared file is unstatable, cannot bind the entry to it")
                return None
            if declared_info.st_size != entry["bytes"]:
                # These readers read whole files. An entry covering part of one
                # is a legitimate map entry and a wrong answer for this caller.
                self._record_fallback(
                    path, "map entry is a byte range, not the whole declared file")
                return None
            ram_path = self._ram_offer(entry, path)
            stage_path = entry["stage_path"]
            try:
                info = os.lstat(stage_path)
            except OSError as error:
                stage_refusal = f"staged copy is unreadable: {error.strerror}"
            else:
                if not stat.S_ISREG(info.st_mode):
                    stage_refusal = "staged copy is not a regular file"
                elif info.st_size != entry["bytes"]:
                    stage_refusal = "staged copy size differs from the map"
                else:
                    stage_refusal = None
            if stage_refusal is not None:
                if ram_path is None:
                    self._record_fallback(path, stage_refusal)
                    return None
                # The stage copy failed its fence but the entry is bound and
                # the ram copy is current: the ram half answers on its own,
                # and the caller that cannot use it falls back to the declared
                # path through the stage_path it still holds.
                return {"declared_path": path, "stage_path": stage_path,
                        "ram_path": ram_path, "bytes": entry["bytes"],
                        "offset": entry["offset"], "sha256": entry["sha256"]}
            answer = {"declared_path": path, "stage_path": stage_path,
                      "bytes": entry["bytes"], "offset": entry["offset"],
                      "sha256": entry["sha256"]}
            if ram_path is not None:
                answer["ram_path"] = ram_path
            return answer

    def _interval_index(self) -> tuple[dict, dict]:
        """The map's entries grouped by declared path. Caller holds the lock.

        The map is keyed by an exact offset, which answers "is this whole file
        staged" and not "which staged range covers these bytes". This is that
        second question's index, built once per adopted map -- ``_adopt`` and
        ``_forget`` both install a new ``_entries``, so the identity check below
        is what invalidates it -- and built for both spellings at once, since
        the realpath pass costs one ``realpath`` per distinct directory and
        ``_real_dirs`` already caches those.
        """
        if self._intervals_for is self._entries:
            return self._intervals, self._real_intervals
        declared_index: dict[str, list[dict]] = {}
        real_index: dict[str, list[dict]] = {}
        for entry in self._entries.values():
            path = entry["declared_path"]
            declared_index.setdefault(path, []).append(entry)
            directory, name = os.path.split(path)
            real = self._real_dirs.get(directory)
            if real is None:
                try:
                    real = os.path.realpath(directory)
                except OSError:
                    real = directory
                self._real_dirs[directory] = real
            if real != directory:
                real_index.setdefault(os.path.join(real, name), []).append(entry)
        for rows in declared_index.values():
            rows.sort(key=lambda row: row["offset"])
        for rows in real_index.values():
            rows.sort(key=lambda row: row["offset"])
        self._intervals = declared_index
        self._real_intervals = real_index
        self._intervals_for = self._entries
        return declared_index, real_index

    def stages(self, declared: str | Path) -> bool:
        """Does the map name this declared file at any offset?

        The question a reader asks before it wraps anything. A file the map
        never names is read exactly as it is read without a map, so nothing
        pays for a redirect that cannot happen.
        """
        path = _normal(declared)
        with self._lock:
            self._read_map()
            index, real = self._interval_index()
            return bool(index.get(path) or real.get(path))

    # -- PB's sealed readset ---------------------------------------------

    def _load_declared_readset(self) -> None:
        """Read the sealed readset PrismaBuild published for this action.

        Caller holds the lock. Sets ``self._declared`` on success and
        ``self._declared_reason`` on every failure; runs at most once.

        All the PB-facing work -- resolving the action's claim row, the
        ``cas_root`` and manifest digest on it, reading and hashing the CAS
        blob, and decoding it through PB's own validating reader -- lives in
        :func:`staged_lease.load_sealed_readset`, next to the lease
        context it is resolved from. This holds the result and answers
        questions about spans.

        Every failure is silent about the range and loud once in the log.
        An unbound readset means this process cannot tell a declared range
        from an undeclared one, which is a reason to do nothing extra --
        never a reason to refuse a read, and never a licence to wait.

        Cost, stated because it is paid under the resolver's lock: one
        claim-row read, one bounded blob read and SHA-256, and PB's own
        validation of the entry list -- 36,600 entries on the
        GLM-5.3-Flash campaign. It happens once per binding, before any
        wait it might authorize, and concurrent ``staged_range`` callers
        wait behind it for that one pass. Loading outside the lock would
        trade that for several threads each doing the same work.
        """
        self._declared_attempted = True
        if self._manifest_sha256 is None:
            self._declared_reason = "no data manifest digest is bound"
            return
        try:
            from .staged_lease import ReadsetUnbound, load_sealed_readset
        except ImportError as error:  # pragma: no cover - import-time only
            self._declared_reason = f"lease module unavailable: {error}"
            return
        try:
            spans = load_sealed_readset(self._manifest_sha256)
        except ReadsetUnbound as unbound:
            self._declared_reason = str(unbound)
            return
        except Exception as error:  # never let this path fail a read
            self._declared_reason = f"sealed readset error: {error}"
            return
        self._declared = {_normal(path): _merge_spans(rows)
                          for path, rows in spans.items()}
        self._declared_reason = None
        print(f"[residency] sealed readset bound: {len(self._declared)} "
              f"declared path(s), manifest {self._manifest_sha256[:12]}",
              flush=True)

    def _declares(self, path: str, start: int, end: int) -> bool | None:
        """Does PB's sealed readset declare ``[start, end)`` of ``path``?

        ``None`` when the readset is not bound -- *unknown*, which is not
        the same answer as False and must never be spelled as one.
        Caller holds the lock.
        """
        if not self._declared_attempted:
            self._load_declared_readset()
        if self._declared is None:
            return None
        rows = self._declared.get(path)
        if not rows:
            return False
        index = bisect.bisect_right(rows, (start, _INFINITE_OFFSET)) - 1
        if index < 0:
            return False
        low, high = rows[index]
        return low <= start and end <= high

    def declared_readset(self) -> dict:
        """Whether the sealed readset is bound, and why not when it is not."""
        with self._lock:
            if not self._declared_attempted:
                self._load_declared_readset()
            if self._declared is None:
                return {"state": "unbound", "reason": self._declared_reason}
            return {"state": "bound", "paths": len(self._declared)}

    def staged_range(self, declared: str | Path, start: int, end: int, *,
                     declared_size: int | None = None) -> dict | None:
        """Where to open the staged copy of ``declared``'s ``[start, end)``.

        The entry alone, which is all most callers need.
        :meth:`staged_range_outcome` carries the same answer plus *why* a
        None is a None -- a different question, with a different right
        response. This spelling and its ``dict | None`` contract are
        unchanged for every existing caller.
        """
        return self.staged_range_outcome(
            declared, start, end, declared_size=declared_size)[0]

    def staged_range_outcome(
            self, declared: str | Path, start: int, end: int, *,
            declared_size: int | None = None) -> tuple[dict | None, str]:
        """``(entry, outcome)`` for ``declared``'s ``[start, end)``.

        ``outcome`` is one of:

        ``RANGE_HIT``
            an entry covers the span and passed every pre-open check.
        ``RANGE_UNCOVERED``
            no entry covers the span, and PrismaBuild's sealed readset
            declares it. The map holds only what a mover has already
            written, so a range PB published but has not moved yet lands
            here: it is *not staged yet*, which is not *not staged*.
            Counted in ``range_misses``, silent.
        ``RANGE_UNDECLARED``
            no entry covers the span and the sealed readset does not
            declare it -- nothing was ever asked to stage these bytes, so
            no amount of waiting will produce them. Counted in
            ``range_misses`` like any other uncovered span.

            When the sealed readset is not bound at all (see
            :meth:`declared_readset`) the two are indistinguishable and
            every uncovered span reports ``RANGE_UNCOVERED``. Unknown is
            reported as unknown; callers decide what to do with it, and
            the reader's pre-flight declines to wait on one.
        ``RANGE_REFUSED``
            an entry covers the span and then failed a check: the declared
            file is unstatable, the entry runs past it, or the staged copy
            is unreadable, not a regular file, or a different size than the
            map says, with no RAM offer. Counted and printed as a fallback.

        The kinds exist because the right response differs. Re-asking can
        turn ``RANGE_UNCOVERED`` into a hit, because the map changes under
        its readers; it cannot improve ``RANGE_REFUSED``, which is evidence
        already in hand -- an entry that runs past the declared file runs
        past it on every look.

        This method reports the kind and never waits. Re-asking belongs to
        the caller -- ``layer_streaming._await_layer_readset``, before a
        layer's gather is submitted -- which does not hold this lock.

        ``staged_read``'s pre-open checks, asked of a byte range. One entry has
        to cover the span outright; the staged copy has to be a regular file of
        exactly the entry's length; and the entry has to fit inside the declared
        file, which is what binds it to the file it stands for and is the check
        ``staged_read`` spells as an equality because its callers read the whole
        thing. A caller that already holds the declared file's length passes it
        as ``declared_size`` rather than making this stat a shard's worth of
        NFS getattrs.

        A span no entry covers is a miss, counted and silent: a half-staged
        shard is the ordinary mid-flight state, not a refusal worth a line per
        tensor. An entry that covers it and then fails a check is a fallback,
        counted and printed, exactly as a whole-file refusal is.

        The read position inside the staged file is ``start - entry["offset"]``:
        PrismaBuild's mover writes a range as a file of its own, from byte 0.
        """
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end:
            raise ValueError("a staged range is a non-negative [start, end) span")
        path = _normal(declared)
        with self._lock:
            self._read_map()
            index, real = self._interval_index()
            rows = index.get(path) or real.get(path)
            entry = None
            for row in rows or ():
                if row["offset"] <= start and end <= row["offset"] + row["bytes"]:
                    entry = row
                    break
            if entry is None:
                self._range_misses += 1
                if self._declares(path, start, end) is False:
                    return None, RANGE_UNDECLARED
                return None, RANGE_UNCOVERED
            if declared_size is None:
                try:
                    declared_size = os.lstat(path).st_size
                except OSError:
                    self._record_fallback(
                        path, "declared file is unstatable, cannot bind the entry to it")
                    return None, RANGE_REFUSED
            if entry["offset"] + entry["bytes"] > declared_size:
                self._record_fallback(path, "map entry runs past the declared file")
                return None, RANGE_REFUSED
            ram_path = self._ram_offer(entry, path)
            stage_path = entry["stage_path"]
            try:
                info = os.lstat(stage_path)
            except OSError as error:
                stage_refusal = f"staged copy is unreadable: {error.strerror}"
            else:
                if not stat.S_ISREG(info.st_mode):
                    stage_refusal = "staged copy is not a regular file"
                elif info.st_size != entry["bytes"]:
                    stage_refusal = "staged copy size differs from the map"
                else:
                    stage_refusal = None
            if stage_refusal is not None:
                if ram_path is None:
                    self._record_fallback(path, stage_refusal)
                    return None, RANGE_REFUSED
                return ({"declared_path": path, "stage_path": stage_path,
                         "ram_path": ram_path, "bytes": entry["bytes"],
                         "offset": entry["offset"],
                         "sha256": entry["sha256"]}, RANGE_HIT)
            answer = {"declared_path": path, "stage_path": stage_path,
                      "bytes": entry["bytes"], "offset": entry["offset"],
                      "sha256": entry["sha256"]}
            if ram_path is not None:
                answer["ram_path"] = ram_path
            return answer, RANGE_HIT

    # -- the accounting --------------------------------------------------

    def _record_fallback(self, path: str, reason: str) -> None:
        """Caller holds the lock. The list is bounded; the count is not."""
        self._fallback_count += 1
        if len(self._fallbacks) < MAX_RECORDED_FALLBACKS:
            self._fallbacks.append({"path": path, "reason": reason})
        print(f"[residency] fallback {path}: {reason}", flush=True)

    def _record_ram_fallback(self, path: str, reason: str) -> None:
        """Caller holds the lock. Bounded like the stage's own."""
        self._ram_fallback_count += 1
        if len(self._ram_fallbacks) < MAX_RECORDED_FALLBACKS:
            self._ram_fallbacks.append({"path": path, "reason": reason})
        print(f"[residency] ram fallback {path}: {reason}", flush=True)

    def record_fallback(self, declared: str | Path, reason: str) -> None:
        with self._lock:
            self._record_fallback(_normal(declared), reason)

    def record_range_wait(self, declared: str | Path, *, polls: int,
                          seconds: float, served: bool) -> None:
        """One strict read that waited for a range the map did not hold yet.

        The resolver itself still never waits (module docstring): this only
        records what a reader chose to do with an ordinary mid-flight miss,
        so ``results.json`` shows the waiting as waiting instead of leaving it
        inside ``range_misses`` as a burst of failed lookups.
        """
        if type(polls) is not int or isinstance(polls, bool) or polls < 0:
            raise ValueError("a range wait polls a non-negative number of times")
        if type(seconds) not in (int, float) or seconds < 0:
            raise ValueError("a range wait lasts a non-negative number of seconds")
        with self._lock:
            self._range_wait_polls += polls
            self._range_wait_seconds += float(seconds)
            if served:
                self._range_waits_served += 1
            else:
                self._range_waits_refused += 1
            print(f"[residency] range wait {_normal(declared)}: "
                  f"{polls} poll(s) over {seconds:.1f}s -> "
                  f"{'staged' if served else 'still not staged, refusing'}",
                  flush=True)

    def record_ram_fallback(self, declared: str | Path, reason: str) -> None:
        """One ram copy that did not serve the read it was offered for.

        The read continues from the stage copy or the declared path; this is
        the record of the ram half's own miss, kept beside ``fallbacks`` so
        the two chains stay separately readable in ``results.json``.
        """
        with self._lock:
            self._record_ram_fallback(_normal(declared), reason)

    def record_stage_read(self, declared: str | Path, nbytes: int) -> None:
        with self._lock:
            self._hits += 1
            self._bytes_from_stage += int(nbytes)

    def record_ram_read(self, declared: str | Path, nbytes: int) -> None:
        """One read the ram tier served, counted beside the stage's own.

        Not inside ``hits``: ``hits`` stays the number of reads the stage
        copy served, so a run's accounting can say what each tier served
        without subtracting. ``ram_hits`` is the ram tier's own count.
        """
        with self._lock:
            self._ram_hits += 1
            self._bytes_from_ram += int(nbytes)

    def record_stage_range_read(self, declared: str | Path, nbytes: int) -> None:
        """One read the stage served out of a byte range.

        Counted in ``hits`` too, so ``hits`` stays the number of reads the
        stage served and ``range_hits`` says how many of those came out of a
        range rather than a whole staged file. A run whose journal shows
        ``range_hits`` above zero read shard bytes off the stage.
        """
        with self._lock:
            self._hits += 1
            self._range_hits += 1
            self._bytes_from_stage += int(nbytes)

    def record_pool_read(self, declared: str | Path, nbytes: int) -> None:
        with self._lock:
            self._bytes_from_pool += int(nbytes)

    def record_serving_tier(self, declared: str | Path, tier: str,
                            detail: str = "", *, pin_id: str | None = None,
                            range_ref: str | None = None) -> None:
        """Note where an open actually served from, before payload trust.

        ``tier`` is ``ram`` or ``stage``. The tier/epoch halves are the
        resolver's own; ``lease_id`` is None until the PB reader-lease API
        (RNG-02/SM-03) exists. Bounded like the fallback lists; the count
        is not. Counters observe — this record authorizes nothing.

        Under a lifetime pin, callers pass the SDK serving record's
        ``pin_id``/``range_ref``: the record is emitted at the successful
        actual open, never when a path candidate merely passes ``lstat``.
        """
        path = _normal(declared)
        with self._lock:
            self._serving_tier_count += 1
            if len(self._serving_tiers) < MAX_RECORDED_FALLBACKS:
                record: dict = {"path": path, "serving_tier": tier,
                                "lease_id": None}
                if tier == "ram":
                    record["tier_id"] = self._ram_tier_id
                    record["epoch"] = self._ram_epoch
                else:
                    record["tier_id"] = self._tier_id
                if pin_id is not None:
                    record["pin_id"] = pin_id
                    record["lease_id"] = pin_id
                if range_ref is not None:
                    record["range_ref"] = range_ref
                if detail:
                    record["detail"] = detail
                self._serving_tiers.append(record)

    def lease_identity(self) -> dict:
        """The composed map's own identity for lease covers.

        Returns ``tier_id``, ``leads`` (the mover keys that vouched this
        read set), ``manifest_sha256``, and the ram half's
        ``ram_tier_id``/``ram_epoch``. The leads are heritage, not a
        covering set: both tier legs resolve minimal per-key covers
        through PB's ``covers_for_keys`` (RAM movers are not in the
        composed map at all — see ``staged_lease.ram_covers``).
        """
        with self._lock:
            return {"tier_id": self._tier_id,
                    "leads": list(self._leads),
                    "manifest_sha256": self._manifest_sha256,
                    "ram_tier_id": self._ram_tier_id,
                    "ram_epoch": self._ram_epoch,
                    "residency_root": str(Path(self._map_path).parent)}

    def report(self) -> dict:
        """What each tier served this run, for ``results.json``."""
        with self._lock:
            report = {
                "map_path": self._map_path,
                "map_sha256": self._map_sha256,
                "manifest_sha256": self._manifest_sha256,
                "tier_id": self._tier_id,
                "stage_root": self._stage_root,
                "leads": list(self._leads),
                "generation": self._generation,
                "entries": len(self._entries),
                "hits": self._hits,
                "misses": self._misses,
                "range_hits": self._range_hits,
                "range_misses": self._range_misses,
                "range_wait_polls": self._range_wait_polls,
                "range_wait_seconds": round(self._range_wait_seconds, 3),
                "range_waits_served": self._range_waits_served,
                "range_waits_refused": self._range_waits_refused,
                "declared_readset": (
                    {"state": "bound", "paths": len(self._declared)}
                    if self._declared is not None else
                    {"state": "unbound", "reason": self._declared_reason}
                    if self._declared_attempted else
                    {"state": "unread",
                     "reason": "no read asked whether a range was declared"}),
                "fallbacks": [dict(row) for row in self._fallbacks],
                "fallback_count": self._fallback_count,
                "bytes_from_stage": self._bytes_from_stage,
                "ram_hits": self._ram_hits,
                "ram_fallbacks": [dict(row) for row in self._ram_fallbacks],
                "ram_fallback_count": self._ram_fallback_count,
                "bytes_from_ram": self._bytes_from_ram,
                "bytes_from_pool": self._bytes_from_pool,
                "serving_tiers": [dict(row) for row in self._serving_tiers],
                "serving_tier_count": self._serving_tier_count,
            }
            if self._ram_tier_id is not None:
                # The ram half's own header and verdict, present only when a
                # map of the overlay generation was adopted at all.
                report["ram_tier_id"] = self._ram_tier_id
                report["ram_root"] = self._ram_root
                report["ram_epoch"] = self._ram_epoch
                if self._ram_refusal is not None:
                    report["ram_refused"] = self._ram_refusal
            if self._refused is not None:
                report["refused"] = self._refused
            return report


_RESOLVER_LOCK = threading.Lock()
_RESOLVER_FOR: tuple[str | None, ResidencyResolver | None] = (None, None)


def residency_resolver() -> ResidencyResolver | None:
    """The process's resolver, or None when the environment names no map.

    One resolver per map path, so every read site and the run's accounting see
    the same store. Unset means inert: no file is stat'ed and no path moves.
    """
    named = os.environ.get(ENV_VAR)
    if not named:
        return None
    global _RESOLVER_FOR
    with _RESOLVER_LOCK:
        path, resolver = _RESOLVER_FOR
        if path != named or resolver is None:
            resolver = ResidencyResolver(named)
            _RESOLVER_FOR = (named, resolver)
        return resolver


def bind_residency_manifest(digest: str | None) -> None:
    """Bind this process's read set, so a map for other bytes cannot apply."""
    resolver = residency_resolver()
    if resolver is not None and digest is not None:
        resolver.bind_manifest_sha256(digest)


def residency_report() -> dict | None:
    """The accounting block, or None when the resolver is inert."""
    resolver = residency_resolver()
    return None if resolver is None else resolver.report()


def reset_residency_resolver_for_tests() -> None:
    global _RESOLVER_FOR
    with _RESOLVER_LOCK:
        _RESOLVER_FOR = (None, None)
