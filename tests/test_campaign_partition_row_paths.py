"""The existing PB readset owner recognizes strict partition row filenames."""
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


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
