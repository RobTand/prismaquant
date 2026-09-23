"""Sample for ``tests/test_fleet_data_selection.py``: never collected alone."""
import pytest


@pytest.mark.fleet_data
def test_reads_fleet_data():
    pass


def test_reads_nothing():
    pass
