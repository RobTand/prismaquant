"""Strict JSON and refusal vocabulary, independent of numerical artifact schemas."""
from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import PurePosixPath
import re
from typing import Any, NoReturn



def unique_json_object(
    duplicate: Callable[[str], BaseException],
) -> Callable[[list[tuple[str, Any]]], dict[str, Any]]:
    """An ``object_pairs_hook`` that refuses a repeated key.

    ``json`` keeps the last of two equal keys, so one document would have two
    readings. The hook raises ``duplicate(key)``, the caller's own exception.
    """

    def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise duplicate(key)
            result[key] = value
        return result

    return object_from_pairs


def strict_json_loads(
    text: str | bytes,
    *,
    duplicate: Callable[[str], BaseException],
    constant: Callable[[str], BaseException] | None = None,
) -> Any:
    """``json.loads`` that refuses a repeated key and, given ``constant``, NaN.

    ``duplicate(key)`` and ``constant(name)`` build the caller's exceptions;
    ``name`` is ``NaN``, ``Infinity`` or ``-Infinity``. Without ``constant``
    those three parse as floats, as ``json.loads`` parses them. ``text`` is
    ``str`` or ``bytes``, as for ``json.loads``. A malformed document raises
    ``json.JSONDecodeError`` (a ``ValueError``) and undecodable bytes raise
    ``UnicodeDecodeError``; the caller wraps both in its own refusal.
    """
    if constant is None:
        return json.loads(text, object_pairs_hook=unique_json_object(duplicate))

    def reject_constant(name: str) -> NoReturn:
        raise constant(name)

    return json.loads(text, object_pairs_hook=unique_json_object(duplicate),
                      parse_constant=reject_constant)


_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
# Bazel canonical repository names use +, including rules_python++python+pq.
_PATH_COMPONENT = re.compile(r"[A-Za-z0-9._+-]+\Z")


class Contract:
    """One module's refusal vocabulary: the exception it raises and its prefix.

    ``fail`` and ``require`` carry any message. The typed checks speak the
    ``"{where} must be ..."`` vocabulary that the cluster-campaign and
    quality-prefill contracts share word for word, so a module that binds
    them refuses exactly as its own copies did.
    """

    __slots__ = ("error", "prefix")

    def __init__(self, error: type[BaseException] = ValueError, prefix: str = "") -> None:
        self.error = error
        self.prefix = prefix

    def exception(self, message: str) -> BaseException:
        """The refusal ``fail`` raises, for a caller that raises it itself."""
        return self.error(f"{self.prefix}{message}")

    def fail(self, message: str) -> NoReturn:
        raise self.exception(message)

    def require(self, condition: object, message: str) -> None:
        if not condition:
            raise self.exception(message)

    def mapping(self, value: object, *, where: str) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            self.fail(f"{where} must be an object")
        return value

    def exact_mapping(
        self, value: object, *, keys: frozenset[str], where: str,
    ) -> Mapping[str, object]:
        """An object whose keys are strings and exactly ``keys``."""
        self.mapping(value, where=where)
        if any(type(key) is not str for key in value):
            self.fail(f"{where} keys must be strings")
        actual, expected = set(value), set(keys)
        if actual != expected:
            self.fail(f"{where} fields differ: missing={sorted(expected - actual)}, "
                      f"extra={sorted(actual - expected)}")
        return value

    def integer(
        self, value: object, *, where: str, minimum: int, maximum: int = 2**63 - 1,
    ) -> int:
        """``type(value) is int``: a ``bool`` never satisfies an integer field."""
        if type(value) is not int or not minimum <= value <= maximum:
            self.fail(f"{where} must be an integer in [{minimum}, {maximum}]")
        return value

    def string(
        self, value: object, *, where: str, pattern: re.Pattern[str] | None = None,
    ) -> str:
        """A non-empty string with no padding or control characters."""
        if type(value) is not str or not value:
            self.fail(f"{where} must be a non-empty string")
        if value != value.strip() or any(ord(char) < 32 for char in value):
            self.fail(f"{where} contains whitespace padding or control characters")
        if pattern is not None and pattern.fullmatch(value) is None:
            self.fail(f"{where} has an invalid value")
        return value

    def sha256(self, value: object, *, where: str) -> str:
        """A lowercase hex SHA-256 digest."""
        return self.string(value, where=where, pattern=_SHA256_HEX)

    def absolute_posix_path(self, value: object, *, where: str) -> str:
        """A non-root absolute POSIX path of plain, traversal-free components."""
        raw = self.string(value, where=where)
        if not raw.startswith("/") or raw == "/":
            self.fail(f"{where} must be a non-root absolute POSIX path")
        components = raw.split("/")[1:]
        if (
            not components
            or any(
                not component
                or component in {".", ".."}
                or _PATH_COMPONENT.fullmatch(component) is None
                for component in components
            )
            or str(PurePosixPath(raw)) != raw
        ):
            self.fail(f"{where} must be normalized and traversal-free")
        return raw
