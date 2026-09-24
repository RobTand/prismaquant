"""Read the row table PrismaBuild's ``pbwait`` prints.

``pbwait`` reports each action it waited for as one row of a
``key status transport job host elapsed rc receipt note`` table and returns
its verdict as the exit code: 0 when every action's work is done
(``executed`` or ``cache_hit``). It prints no JSON. The Tessera campaign
dispatcher reads the table for its receipts, and the joint-quanta dispatcher
reads it to decide whether a band-serial producer has run (PQ #1197), so both
use this one parser.
"""
from __future__ import annotations

#: The statuses ``pbwait.verdict`` counts as done: a memoized result is the
#: same result.
DONE_STATUSES = frozenset({"executed", "cache_hit"})


def parse_pbwait_table(text: str) -> list[dict]:
    """The ``key status transport job host elapsed rc receipt note`` table.

    Read by the header's own column offsets rather than by splitting on
    whitespace: ``pbwait`` left-justifies every cell to a common width, and a
    cell can hold a space -- ``rc`` renders ``1 (action 137)`` when the
    launcher's status and the action's differ, which is exactly the failing
    row a whitespace split would drop.
    """
    rows: list[dict] = []
    header: list[tuple[str, int, int]] | None = None
    for line in text.splitlines():
        if header is None:
            if line.split()[:2] != ["key", "status"]:
                continue
            names = line.split()
            starts = []
            cursor = 0
            for name in names:
                cursor = line.index(name, cursor)
                starts.append(cursor)
                cursor += len(name)
            ends = starts[1:] + [1 << 20]
            header = list(zip(names, starts, ends))
            continue
        if not line.strip():
            continue
        rows.append({name: line[start:end].strip()
                     for name, start, end in header})
    return rows
