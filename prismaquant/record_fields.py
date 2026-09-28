"""Read a record another project publishes, tolerating what it adds.

PrismaQuant reads several records it does not write: Tessera's packaged
``runtime_contract.json``, its full-engine resource report, and the native
runtime record a panel carries. Each reader used to refuse any field it did not
know. That refused additive changes too, so every producer release that added a
field or a block broke a reader that had nothing to decide about it (#926,
#958, #927). One rule replaces those closed field sets, and it lives here so
that every reader applies it the same way (#1548):

* A field or block the reader does not know is accepted. The reader reads only
  what it names, so an added field cannot change a value it uses.
* A field the reader consumes must be present. Its type and value are still
  checked where the reader reads it.
* A producer that adds a field an old reader must not skip, because skipping it
  would change what the record means, lists that field's name in the object's
  ``must_understand`` array. A reader that does not know a listed name refuses
  the record. This is the producer's way to force a refusal on the one change a
  tolerant reader would otherwise get wrong.

Keyed tables are not field sets and do not use this helper. When the keys are
values the reader decides on, such as a lane's ``requires`` predicate, a
tensor-parallel unit's ``loader_axes``, or the report's domain and term tables,
an unknown key is a condition the reader cannot evaluate, and those readers
still refuse it.
"""
from __future__ import annotations

from typing import Any, Collection, Mapping

__all__ = ["MUST_UNDERSTAND", "admit_fields"]

#: The member of a published object that lists the fields a reader may not skip.
MUST_UNDERSTAND = "must_understand"


def admit_fields(payload: Any, where: str, *, required: Collection[str],
                 optional: Collection[str] = (), error: type[Exception]) -> Mapping[str, Any]:
    """Admit one published object, or refuse it with ``error``.

    ``required`` are the fields the reader consumes and cannot do without.
    ``optional`` are the other fields the reader understands. Together they are
    what the reader knows, and a ``must_understand`` name outside them refuses.
    Any other field is accepted and left unread. Returns ``payload``.
    """
    if not isinstance(payload, Mapping):
        raise error(f"{where}: expected a JSON object, got {type(payload).__name__}")
    missing = sorted(set(required) - set(payload))
    if missing:
        raise error(f"{where}: missing field(s) {missing}")
    if MUST_UNDERSTAND in payload:
        marked = payload[MUST_UNDERSTAND]
        if (not isinstance(marked, list)
                or any(not isinstance(name, str) or not name for name in marked)
                or len(set(marked)) != len(marked)):
            raise error(
                f"{where}.{MUST_UNDERSTAND} must be a JSON array of unique, non-empty "
                f"field names, got {marked!r}")
        unknown = sorted(set(marked) - set(required) - set(optional))
        if unknown:
            raise error(
                f"{where}: the producer marks {unknown} must-understand, and this reader "
                "does not understand them. The producer added a field an old reader may "
                "not skip; teach the reader the field before reading this record.")
    return payload
