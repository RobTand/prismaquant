"""Campaign metadata keeps its inherited comparison and sampling byte recipes."""
import hashlib
import math

import pytest

from prismaquant.digests import DIRECT_ASCII_SPACED_LAX_DEFAULT_STR
from prismaquant.tessera_campaign_table import MergeRefused, _keystream, _require_equal


def test_comparison_profile_keeps_spaces_unicode_nonfinite_and_fallback():
    class Label:
        def __str__(self):
            return 'héllo'

    value = {'z': float('nan'), 'a': Label(), 'n': {10: 1, 2: 3}}
    assert DIRECT_ASCII_SPACED_LAX_DEFAULT_STR.text(value) == (
        '{"a": "h\\u00e9llo", "n": {"2": 3, "10": 1}, "z": NaN}')
    assert _require_equal('metadata', {'a': value, 'b': value}) is value
    with pytest.raises(MergeRefused, match=r'rows disagree on metadata: a=.*b='):
        _require_equal('metadata', {'a': value, 'b': {'z': 0}})


def test_sampling_stream_keeps_sha256_counter_word_order():
    stream = _keystream(42, 'audit:stack')
    for counter in range(3):
        expected = hashlib.sha256(f'42:audit:stack:{counter}'.encode()).digest()
        for offset in range(0, 32, 8):
            assert next(stream) == int.from_bytes(expected[offset:offset + 8], 'big') / (1 << 64)
