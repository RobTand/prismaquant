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
