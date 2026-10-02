"""Strict shared-array section admission, storage identity and frozen views."""
import pytest

from prismaquant.joint_replay_metadata import UInt64Rows


@pytest.mark.parametrize("width,rows", [(1, [(0,), ((1 << 64) - 1,)]), (2, [(0, 3), (5, 9)]), (6, [(1, 2, 3, 4, 5, 6)])])
def test_section_extends_the_existing_array_in_order(width, rows):
    owner = UInt64Rows(width, max_rows=len(rows) + 1)
    raw = owner.raw
    owner.append(tuple(0 for _ in range(width)))
    owner.extend(rows)
    assert owner.raw is raw
    assert list(owner) == [tuple(0 for _ in range(width)), *rows]


@pytest.mark.parametrize("scalar", [False, True])
def test_section_geometry_refuses_before_growth(scalar):
    owner = UInt64Rows(1 if scalar else 2, max_rows=2)
    owner.append((1,) if scalar else (1, 2))
    before = owner.raw.tobytes()
    with pytest.raises(RuntimeError, match="geometry"):
        (owner.extend_scalars([1, 2]) if scalar else owner.extend([(1, 2), (3, 4)]))
    assert owner.raw.tobytes() == before
    owner.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        (owner.extend_scalars([]) if scalar else owner.extend([]))


@pytest.mark.parametrize("value", [True, 1.0, -1, 1 << 64])
@pytest.mark.parametrize("scalar", [False, True])
def test_invalid_late_field_refuses_before_section_growth(value, scalar):
    owner = UInt64Rows(1 if scalar else 2, max_rows=10)
    before = owner.raw.tobytes()
    with pytest.raises(ValueError, match="unsigned uint64"):
        (owner.extend_scalars([1, value]) if scalar else owner.extend([(1, 2), (3, value)]))
    assert owner.raw.tobytes() == before


def test_wrong_width_refuses_without_growth():
    owner = UInt64Rows(2, max_rows=10)
    with pytest.raises(ValueError, match="unsigned uint64"):
        owner.extend([(1, 2), (3,)])
    with pytest.raises(ValueError, match="width one"):
        owner.extend_scalars([1, 2])
    assert len(owner) == 0
