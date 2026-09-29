"""Export-lane sorted-json keeps: each site differs from every profile (issue #1695).

All four gated sorted-json spellings in ``tessera_export_lane.py`` were
checked against the digests profiles (quoting the profile definitions from
``prismaquant/digests.py``):

- ``DIRECT_UTF8_STRICT`` / ``DIRECT_ASCII_STRICT`` / ``DIRECT_ASCII_LAX`` /
  ``DIRECT_ASCII_LAX_DEFAULT_STR`` are all one ``json.dumps`` with
  ``sort_keys=True``, ``separators=(",", ":")`` (compact).
- ``indent2_json_file_bytes`` is ``json.dumps(value, sort_keys=True,
  indent=2, allow_nan=False) + "\\\\n"``.

Deltas pinned here:
1. ``hessian_capture_sha256`` dumps with default (spaced) separators,
   lax NaN and ``default=str``, inside a NUL-framed multi-update seal --
   no single-dumps profile reproduces it.
2. ``main``'s build report dumps indent-2 WITHOUT the ``allow_nan``
   pin: the owner refuses NaN payloads the site emits.
3. The ``main`` / ``read_cached_unit_bundle`` prints and the
   ``require_platform_executes_derived_from_contract`` error texts are
   diagnostics, not digests.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from prismaquant import digests as profiles  # noqa: E402


def test_hessian_dumps_differ_from_every_compact_profile():
    value = {"b": 1, "a": [1, 2]}
    spaced = json.dumps(value, sort_keys=True).encode()
    assert spaced == b'{"a": [1, 2], "b": 1}'
    for profile in (profiles.DIRECT_UTF8_STRICT,
                    profiles.DIRECT_ASCII_STRICT,
                    profiles.DIRECT_ASCII_LAX,
                    profiles.DIRECT_ASCII_LAX_DEFAULT_STR):
        assert profile.encoded(value) != spaced


def test_build_bytes_laxity_differs_from_indent2_owner():
    nan_payload = {"x": float("nan")}
    site = (json.dumps(nan_payload, indent=2, sort_keys=True) + "\n").encode()
    assert b"NaN" in site
    with pytest.raises(ValueError):
        profiles.indent2_json_file_bytes(nan_payload)


def test_hessian_site_differs_in_separators():
    # The site dumps with default (spaced) separators inside its NUL-framed
    # multi-update seal; no compact-separator profile reproduces even the
    # dumps half, let alone the framing.
    value = {"dtype": "float32", "shape": [4]}
    site = json.dumps(value, sort_keys=True).encode()
    assert site == b'{"dtype": "float32", "shape": [4]}'
    assert (profiles.DIRECT_ASCII_STRICT.encoded(value)
            == b'{"dtype":"float32","shape":[4]}')
    assert site != profiles.DIRECT_ASCII_STRICT.encoded(value)
