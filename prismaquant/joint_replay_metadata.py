"""Geometry-bounded uint64 metadata for the existing Stage B spill owner.

No payload, cache, IO or schedule is owned here. Chunk queries expose frozen
views of the same ordered integer fields the tuple plan retained.
"""
from array import array
from collections.abc import Mapping, Sequence
import operator

_UNSET = (1 << 64) - 1


class UInt64Rows(Sequence):
    """One bounded array owner, shared by firing and read-plan metadata."""

    def __init__(self, width, *, max_rows):
        if type(width) is not int or width <= 0:
            raise ValueError("spill metadata row width must be positive")
        if type(max_rows) is not int or not 0 <= max_rows < 1 << 64:
            raise ValueError("spill metadata geometry must be unsigned uint64")
        self.width, self.max_rows = width, max_rows
        self.raw = array("Q")
        self._frozen = False

    def __len__(self):
        return len(self.raw) // self.width

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("spill metadata row index out of range")
        start = index * self.width
        return tuple(self.raw[start:start + self.width])

    def append(self, row):
        if self._frozen:
            raise RuntimeError("spill metadata rows are frozen")
        if len(self) >= self.max_rows:
            raise RuntimeError("spill metadata exceeds its geometry")
        if len(row) != self.width or any(type(v) is not int or not 0 <= v < 1 << 64 for v in row):
            raise ValueError("spill metadata fields must be unsigned uint64")
        self.raw.extend(row)

    def repeat(self, value, count):
        if self._frozen or self.width != 1 or type(count) is not int or count < 0:
            raise ValueError("spill metadata repetition requires mutable scalar rows")
        if len(self) + count > self.max_rows:
            raise RuntimeError("spill metadata exceeds its geometry")
        if type(value) is not int or not 0 <= value < 1 << 64:
            raise ValueError("spill metadata fields must be unsigned uint64")
        self.raw.extend(array("Q", [value]) * count)

    def freeze(self):
        self._frozen = True


class _RowsView(Sequence):
    """A chunk's integer section, reconstructed one row at a time."""

    def __init__(self, rows, start, count):
        self._rows, self._start, self._count = rows, start, count

    def __len__(self):
        return self._count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("spill metadata view index out of range")
        row = self._rows[self._start + index]
        return row[0] if self._rows.width == 1 else row

    def __eq__(self, other):
        return (isinstance(other, Sequence) and len(self) == len(other)
                and all(left == right for left, right in zip(self, other)))


class PackedLastUses(Mapping):
    """The existing (owner, entry) lookup over one bounded last-use array."""

    def __init__(self, entries, *, max_parts):
        count = sum(len(rows) for rows in entries.values())
        if count > max_parts:
            raise RuntimeError("spill last-use entries exceed their geometry")
        self._segments = {}
        self._values = UInt64Rows(1, max_rows=max_parts)
        for owner, rows in entries.items():
            self._segments[owner] = (len(self._values), len(rows))
            self._values.repeat(_UNSET, len(rows))

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        for owner, (_start, count) in self._segments.items():
            for entry in range(count):
                yield owner, entry

    def _index(self, key):
        owner, entry = key
        start, count = self._segments[owner]
        entry = operator.index(entry)
        if not 0 <= entry < count:
            raise KeyError(key)
        return start + entry

    def __getitem__(self, key):
        value = self._values.raw[self._index(key)]
        if value == _UNSET:
            raise KeyError(key)
        return value

    def __setitem__(self, key, position):
        if self._values._frozen:
            raise RuntimeError("spill last uses are frozen")
        if type(position) is not int or not 0 <= position < _UNSET:
            raise ValueError("spill last-use position must be unsigned uint64")
        self._values.raw[self._index(key)] = position

    def freeze(self):
        if _UNSET in self._values.raw:
            raise RuntimeError("spill last-use plan has an unread entry")
        self._values.freeze()


class PackedReadPlan(Sequence):
    """Frozen chunk fields with the legacy ordered (owner, chunk) interface.

    Each scalar/pair section and chunk header is capped by the existing
    conservative part geometry. Section queries retain thin views, not a
    tuple per record. IO-engine descriptors still scale with chunk count.
    """

    def __init__(self, names, *, max_parts):
        self._names = tuple(names)
        self._name_ids = {name: index for index, name in enumerate(self._names)}
        self._headers = UInt64Rows(8, max_rows=max_parts)
        self._records = UInt64Rows(1, max_rows=max_parts)
        self._inputs = UInt64Rows(2, max_rows=max_parts)
        self._gradients = UInt64Rows(2, max_rows=max_parts)

    def __len__(self):
        return len(self._headers)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        owner, rs, rn, xs, xn, gs, gn, used = self._headers[index]
        return self._names[owner], (
            _RowsView(self._records, rs, rn), _RowsView(self._inputs, xs, xn),
            _RowsView(self._gradients, gs, gn), used)

    def used_bytes(self, index):
        return self._headers[index][7]

    def append(self, item):
        owner, (records, inputs, gradients, used) = item
        if owner not in self._name_ids:
            raise ValueError("spill chunk owner is outside its roster")
        if not records or len(records) != len(gradients):
            raise ValueError("spill chunk record/gradient count differs")
        sections = ((self._records, records), (self._inputs, inputs), (self._gradients, gradients))
        if self._headers._frozen or any(rows._frozen for rows, _values in sections):
            raise RuntimeError("spill read plan is frozen")
        if len(self._headers) >= self._headers.max_rows or any(
                len(rows) + len(values) > rows.max_rows for rows, values in sections):
            raise RuntimeError("spill read-plan metadata exceeds its geometry")
        rs, xs, gs = len(self._records), len(self._inputs), len(self._gradients)
        for position in records:
            self._records.append((position,))
        for pair in inputs:
            self._inputs.append(pair)
        for pair in gradients:
            self._gradients.append(pair)
        self._headers.append((self._name_ids[owner], rs, len(records), xs, len(inputs),
                              gs, len(gradients), used))

    def freeze(self):
        for rows in (self._headers, self._records, self._inputs, self._gradients):
            rows.freeze()
