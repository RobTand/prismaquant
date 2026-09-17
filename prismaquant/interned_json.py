"""Read a JSON document with the stdlib reader, sharing its repeated strings.

``json.loads(path.read_text())`` holds the decoded text and the object graph at
the same time. On the 7,198,621,019-byte merged joint checkpoint that is a
20.73 GiB peak and 14.03 GiB retained, of which 12.39 GiB is
``identity.units``: 205,243,544 Tessera format-name occurrences that are 5,635
distinct strings, because every unit's menu is a fresh list of fresh ``str``
objects (``joint-aura-resume/ckpt-subtree-probe-0{1,2}.txt``).

The reader here is the standard library's -- ``json.load`` over the file's text
-- and the only thing this module adds is the ``object_pairs_hook`` that makes
those repeats cost one ``str`` each:

* ``object_pairs_hook`` sees every object as the decoder completes it, which is
  where sharing happens: equal strings become the same ``str``, and a list
  whose members are *all exactly* ``str`` has those members replaced by the
  canonical ones.
* Nothing else is shared. A list is never shared as a whole, so two units whose
  menus are equal get their own ``list`` of the same ``str`` objects and
  mutating one unit's menu cannot be seen through another. A list holding
  anything but ``str`` keeps its own members, types and identity: ``['x', 1]``,
  ``['x', True]`` and ``['x', 1.0]`` are one *value* as a memo key, so a
  value-keyed memo must never see them.
* Both memos are bounded, so a document whose strings are all distinct does not
  grow a second copy of itself; past the bound a string is left unshared, which
  changes identity and never value.

Measured on dl380g10, 2026-09-16, one process per phase
(``tools/checkpoint_parse_probe.py``; receipts under
``joint-aura-resume/parser-repair/``). The parse of the merged checkpoint peaks
at 13.812 GiB and retains 2.116 GiB, and the full
``load_measured_anchor_input`` metadata intake over the real campaign -- 36,423
units, 197,990 cells, 0 synthesized -- peaks at 15.830 GiB inside the 21 GiB
CPU envelope (PB ``c9650e1bd63f``, receipt ``d688cb3e``). Both runs recompute
the recorded identity seal ``fcdc6734310ce583fb436f72da0099a79edad9d10b92dd32dbbf18cfe0cf3157``
exactly, which is the gate a value this hook moved would fail.

Nothing here is a cache or a sidecar: the file is read on every call, no byte
of it is written anywhere, and the joint loader recomputes the checkpoint's
canonical identity digest from the assembled graph and compares it with the
recorded seal. ``tests/test_interned_json.py`` holds this reader equal to
``json.loads`` over escapes, Unicode, number formats, duplicate keys,
whitespace, refusal cases, exact value types and the sharing itself.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: Bounds on the sharing memos. Distinct strings stop being interned past the
#: limit (they are still returned unchanged), and a document with more than
#: this many distinct menus keeps its later menus unshared.
STRING_MEMO_LIMIT = 1 << 16
LIST_MEMO_LIMIT = 1 << 10


def _all_exactly_str(items: list) -> bool:
    """True when every member is exactly ``str``, at C speed.

    ``tuple(map(type, ...))`` is one C-level pass and one tuple comparison; the
    obvious ``all(type(item) is str for item in items)`` is one Python call per
    member, and the checkpoint's menus hold 205M of them. Exactness is the
    point: a member that is merely *equal* to a ``str`` (a subclass, an ``int``
    beside a ``bool``, a ``float``) must never be substituted for another.
    """
    return tuple(map(type, items)) == (str,) * len(items)


class _Interner:
    """Share equal strings and equal lists of strings within one parse.

    The string memo is the general one; the list memo exists because a menu is
    a list of thousands of names, and hashing the *tuple* of that list is a
    C-level scan, while interning every occurrence would be one Python call per
    occurrence -- 205M of them on the checkpoint.
    """

    __slots__ = ("_strings", "_lists", "_string_limit", "_list_limit", "_stats")

    def __init__(self, *, stats: dict | None, string_limit: int, list_limit: int) -> None:
        self._strings: dict[str, str] = {}
        self._lists: dict[tuple, tuple] = {}
        self._string_limit = string_limit
        self._list_limit = list_limit
        self._stats = stats

    def string(self, text: str) -> str:
        memo = self._strings
        found = memo.get(text)
        if found is not None:
            if self._stats is not None:
                self._stats["strings_shared"] = self._stats.get("strings_shared", 0) + 1
            return found
        if len(memo) < self._string_limit:
            memo[text] = text
            if self._stats is not None:
                self._stats["strings_seen"] = self._stats.get("strings_seen", 0) + 1
        return text

    def sequence(self, items: list) -> None:
        """Replace ``items`` in place with the canonical members of an equal list.

        Only a list whose members are *all exactly* ``str`` is shared, because
        the memo is keyed by value: ``['x', 1]``, ``['x', True]`` and
        ``['x', 1.0]`` are the same tuple key, so a value-keyed memo would
        substitute one list's member for another's and change a unit's data,
        and ``['x', {}]`` cannot be a key at all. Every other list -- a nested
        container, a number beside a string, an unhashable member -- keeps its
        own members, types and mutability. In place and per list: the caller's
        list object stays its own, so no two units ever share a mutable
        container.
        """
        if len(items) < 2 or not _all_exactly_str(items):
            return
        key = tuple(items)
        memo = self._lists
        found = memo.get(key)   # exact ``str`` members: hashable by construction
        if found is None:
            if len(memo) >= self._list_limit:
                return
            found = tuple(self.string(item) for item in items)
            memo[key] = found
            self._note("lists_memoized")
        else:
            self._note("lists_shared")
        items[:] = found

    def _note(self, counter: str) -> None:
        if self._stats is not None:
            self._stats[counter] = self._stats.get(counter, 0) + 1


def _interning_pairs_hook(user_hook, interner: _Interner):
    """The hook the decoder builds objects with: share, then hand over."""

    def hook(pairs):
        for index, (key, value) in enumerate(pairs):
            if type(key) is str:
                key = interner.string(key)
            if type(value) is str:
                value = interner.string(value)
            elif type(value) is list:
                interner.sequence(value)
            pairs[index] = (key, value)
        if user_hook is not None:
            return user_hook(pairs)
        return dict(pairs)

    return hook


def _sharing_hook(object_pairs_hook, *, intern_strings: bool, string_memo_limit: int,
                  list_memo_limit: int, stats: dict | None):
    """``(object_pairs_hook, interner)``: the sharing the reader is built on."""
    interner = (_Interner(stats=stats, string_limit=string_memo_limit,
                          list_limit=list_memo_limit) if intern_strings else None)
    if interner is None:
        return object_pairs_hook, None
    return _interning_pairs_hook(object_pairs_hook, interner), interner


def interning_pairs_hook(*, object_pairs_hook=None, intern_strings: bool = True,
                         string_memo_limit: int = STRING_MEMO_LIMIT,
                         list_memo_limit: int = LIST_MEMO_LIMIT,
                         stats: dict | None = None):
    """The ``object_pairs_hook`` that shares equal strings, for ``json.load``.

    ``json.load(handle, object_pairs_hook=interning_pairs_hook())`` is the
    whole reader; :func:`load_json_file` is that call with a path.
    """
    hook, _interner = _sharing_hook(
        object_pairs_hook, intern_strings=intern_strings,
        string_memo_limit=string_memo_limit, list_memo_limit=list_memo_limit,
        stats=stats)
    return hook


def load_json_file(source, *, object_pairs_hook=None, intern_strings: bool = True,
                   string_memo_limit: int = STRING_MEMO_LIMIT,
                   list_memo_limit: int = LIST_MEMO_LIMIT,
                   stats: dict | None = None) -> Any:
    """``json.load`` with the sharing hook: the joint checkpoint's reader.

    ``source`` is a path or an open text handle. The file's text is UTF-8
    explicitly, which keeps the refusal of a leading BOM that the previous
    ``json.loads(path.read_text())`` path had. ``object_pairs_hook`` is the one
    decoder option this forwards, exactly as the stdlib applies it; the
    stdlib's own refusal covers every other option. ``intern_strings`` turns
    the sharing off, and ``stats`` is an optional diagnostics dict.
    """
    if hasattr(source, "read"):
        handle, owned = source, False
    else:
        handle, owned = Path(source).open("r", encoding="utf-8"), True
    try:
        return json.load(handle, object_pairs_hook=interning_pairs_hook(
            object_pairs_hook=object_pairs_hook, intern_strings=intern_strings,
            string_memo_limit=string_memo_limit, list_memo_limit=list_memo_limit,
            stats=stats))
    finally:
        if owned:
            handle.close()
