"""Literal prefill bytes survive retirement of the PB document adapter."""
from __future__ import annotations

import hashlib

import pytest

from prismaquant.digests import canonical_json_bytes, canonical_json_sha256


@pytest.mark.parametrize(('value', 'encoded'), [
    ({}, b'{}'),
    ({'z': True, 'a': None}, b'{"a":null,"z":true}'),
    ({10: 'ten', 2: 'two'}, b'{"10":"ten","2":"two"}'),
    ({'tuple': (False, 1, 'x')}, b'{"tuple":[false,1,"x"]}'),
    ({'é': '雪\x00\n'}, b'{"\xc3\xa9":"\xe9\x9b\xaa\\u0000\\n"}'),
    ({'float': 1e-7, 'negative_zero': -0.0}, b'{"float":1e-07,"negative_zero":-0.0}'),
    (['a', True, None, {'z': 2, 'a': 1}], b'["a",true,null,{"a":1,"z":2}]'),
    ('', b'""'),
    ('\r\n', b'"\\r\\n"'),
])
def test_literal_canonical_bytes_single_publication_lf_and_digest(value, encoded):
    actual = canonical_json_bytes(value, where='prefill')
    assert actual == encoded
    assert actual + b'\n' == encoded + b'\n'
    assert not (actual + b'\n').endswith(b'\n\n')
    assert canonical_json_sha256(value, where='prefill') == hashlib.sha256(encoded).hexdigest()


def cyclic():
    value = []
    value.append(value)
    return value


@pytest.mark.parametrize('value', [
    float('nan'), float('inf'), float('-inf'), object(), {1, 2},
    {(1, 2): 'tuple key'}, cyclic(), {'bad': '\ud800'},
])
def test_noncanonical_and_invalid_utf8_values_are_refused(value):
    with pytest.raises(ValueError):
        canonical_json_bytes(value, where='prefill')
