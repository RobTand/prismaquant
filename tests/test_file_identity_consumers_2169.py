"""Load-bearing file identities and caller-owned fallback/classification."""
import hashlib
import subprocess
from pathlib import Path
import sys

import pytest

from prismaquant.cluster_campaign import _receipt_observation
from prismaquant.fisher_col_weights import _dataset_sha256


@pytest.mark.parametrize("size", [0, 1, 1048575, 1048576, 1048577, 8388607, 8388608, 8388609])
def test_file_identities_keep_exact_bytes_across_both_read_windows(tmp_path, size):
    raw = (bytes(range(256)) * ((size + 255) // 256))[:size]
    path = tmp_path / "calibration-receipt.bin"
    path.write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()
    assert _dataset_sha256(str(path)) == expected
    receipt = {"path": str(path), "sha256": expected}
    assert _receipt_observation([receipt]) == ([], [])
    path.write_bytes(raw + b"\x00")
    assert _receipt_observation([receipt]) == ([], [str(path)])


@pytest.mark.parametrize("identifier", [None, "", "fixture/nonexistent-dataset", "数据集/δ"])
def test_dataset_identifier_keeps_optional_strict_utf8_profile(identifier):
    expected = None if not identifier else hashlib.sha256(identifier.encode("utf-8")).hexdigest()
    assert _dataset_sha256(identifier) == expected


def test_nonregular_receipts_and_dataset_fallback_keep_different_policies(tmp_path):
    missing = tmp_path / "missing"
    directory = tmp_path / "directory"
    directory.mkdir()
    payload = tmp_path / "real"
    payload.write_bytes(b"payload")
    link = tmp_path / "link"
    link.symlink_to(payload)
    broken = tmp_path / "broken"
    broken.symlink_to(missing)
    expected = hashlib.sha256(b"payload").hexdigest()
    receipts = [{"path": str(p), "sha256": expected} for p in (missing, directory, link, broken)]
    assert _receipt_observation(receipts) == ([str(missing)], [str(directory), str(link), str(broken)])
    assert _dataset_sha256(str(link)) == expected
    for p in (missing, directory, broken):
        assert _dataset_sha256(str(p)) == hashlib.sha256(str(p).encode()).hexdigest()


def test_dataset_lone_surrogate_keeps_native_encode_refusal():
    with pytest.raises(UnicodeEncodeError):
        _dataset_sha256("\ud800")


@pytest.mark.parametrize("module", ["prismaquant.cluster_campaign", "prismaquant.fisher_col_weights"])
def test_public_cli_import_and_argument_surface(module):
    result = subprocess.run([sys.executable, "-m", module, "--help"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_campaign_worker_direct_script_keeps_stdlib_only_bootstrap():
    script = Path(__file__).resolve().parents[1] / "prismaquant" / "cluster_campaign.py"
    result = subprocess.run([sys.executable, str(script), "--help"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
