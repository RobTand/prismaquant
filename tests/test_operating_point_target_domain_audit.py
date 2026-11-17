import pytest

from tools import operating_point


@pytest.mark.parametrize('target', ['nan', 'inf', '-inf', '0', '-0.1', '0.05'])
def test_explicit_target_requires_a_finite_positive_value(tmp_path, capsys, target):
    points = tmp_path / 'points.csv'
    points.write_text('4,0.1\n5,0.01\n6,0.001\n')
    argv = ['--csv', str(points), '--kl-target=' + target]
    if target == '0.05':
        assert operating_point.main(argv) == 0
        assert 'KL target 0.05 -> bpp 4.30' in capsys.readouterr().out
    else:
        with pytest.raises(SystemExit) as refused:
            operating_point.main(argv)
        assert refused.value.code == 2
        assert 'finite and positive' in capsys.readouterr().err


def test_selection_fit_retains_vetoed_and_validation_measurements(tmp_path):
    import json

    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps({"results": [{"achieved_bits": 6, "kl_mean": 0.001}]}))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "frontier": [{"bpp": 4, "kl": 0.1, "kl_max": 0.2}, {"bpp": 3, "kl": 0}],
        "vetoed_rows": [{"row": {"bpp": 5, "kl": 0.01}}, {"bpp": 4, "kl": 9}],
        "validation_json": str(validation),
    }))
    assert operating_point._collect_selection(str(selection)) == [(4.0, 0.1, 0.2), (5.0, 0.01, None), (6.0, 0.001, None)]
