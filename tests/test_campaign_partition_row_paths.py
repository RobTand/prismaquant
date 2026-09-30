"""The existing PB readset owner recognizes strict partition row filenames."""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import dispatch_tessera_campaign as dispatch  # pyright: ignore[reportMissingImports]


@pytest.mark.parametrize("row_id", ["row-0000", "row-0075", "row-0001-p0000", "row-0001-p0287"])
def test_shared_readset_owner_recognizes_exact_row_path(row_id):
    producer = dispatch._manifest_producer()
    row = {"argv": ["python", "--units", f"/workspace/units/{row_id}.json"]}
    assert producer.row_id_of(row) == row_id


@pytest.mark.parametrize("row_id", ["row-0001-p", "row-0001-p001", "row-0001-p00000", "row-0001-part0000"])
def test_shared_readset_owner_does_not_guess_malformed_partition_id(row_id):
    producer = dispatch._manifest_producer()
    row = {"argv": ["python", "--units", f"/workspace/units/{row_id}.json"]}
    assert producer.row_id_of(row) is None


def test_shared_readset_owner_refuses_ambiguous_partition_paths():
    producer = dispatch._manifest_producer()
    row = {"argv": ["python", "--units", "/workspace/units/row-0001-p0000.json",
                    "--other", "/workspace/units/row-0001-p0001.json"]}
    assert producer.row_id_of(row) is None


def test_shared_selection_metadata_imports_without_site_packages():
    path = Path(__file__).resolve().parents[1] / "prismaquant/tessera_campaign_selection.py"
    script = ("import importlib.util,sys; "
              "spec=importlib.util.spec_from_file_location('selection',sys.argv[1]); "
              "owner=importlib.util.module_from_spec(spec); spec.loader.exec_module(owner); "
              "owner.validate_unit_selection({'schema':owner.UNITS_SCHEMA,'groups':"
              "[{'key':'u:a','members':['a']}]}); "
              "assert 'torch' not in sys.modules")
    completed = subprocess.run([sys.executable, "-S", "-c", script, str(path)],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
