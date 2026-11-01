import pytest

from prismaquant.staged_lease import release_retry_horizon_s


@pytest.mark.parametrize("point", ["/mnt/shared", "/stage/prewarm", "/ram/prewarm"])
def test_pin_release_uses_nfs_above_its_autofs_trigger(tmp_path, point):
    table = tmp_path / "mountinfo"
    table.write_text(
        "1 0 0:1 / / rw - ext4 /dev/root rw\n"
        f"2 1 0:2 / {point} rw - autofs systemd-1 rw\n"
        f"3 2 0:3 / {point} rw - nfs4 host:/export rw,timeo=600,retrans=2\n"
    )
    assert release_retry_horizon_s(point + "/action/pin", str(table)) == 180.0
