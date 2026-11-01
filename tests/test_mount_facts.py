import pytest

from prismaquant.mount_facts import find_mount


@pytest.mark.parametrize("kind", ["mounts", "mountinfo"])
def test_topmost_containing_mount_owns_options(tmp_path, kind):
    table = tmp_path / kind
    if kind == "mountinfo":
        rows = (
            r"1 0 0:1 / / rw - ext4 /dev/root rw",
            r"2 1 0:2 / /stage/prewarm rw - autofs systemd-1 rw",
            r"3 2 0:3 / /stage/prewarm rw - nfs4 host:/export rw,timeo=600,retrans=2,nconnect=8",
            r"4 1 0:4 / /stage/prewarm/local\040disk rw - tmpfs tmpfs rw,size=100k",
        )
    else:
        rows = (
            r"/dev/root / ext4 rw 0 0",
            r"systemd-1 /stage/prewarm autofs rw 0 0",
            r"host:/export /stage/prewarm nfs4 rw,timeo=600,retrans=2,nconnect=8 0 0",
            r"tmpfs /stage/prewarm/local\040disk tmpfs rw,size=100k 0 0",
        )
    table.write_text("\n".join(rows))
    mounted = find_mount("/stage/prewarm/model/weights", table=table)
    assert mounted.point == "/stage/prewarm"
    assert mounted.filesystem == "nfs4"
    assert mounted.options["timeo"] == "600"
    assert mounted.options["nconnect"] == "8"
    assert find_mount("/stage/prewarm/local disk/a", table=table).filesystem == "tmpfs"
    assert find_mount("/stage/prewarm-other/a", table=table).filesystem == "ext4"


def test_missing_or_malformed_mount_table_has_no_facts(tmp_path):
    table = tmp_path / "mountinfo"
    assert find_mount("/a", table=table) is None
    table.write_text("1 2 -\n1 2 3 4 - nfs4\n")
    assert find_mount("/a", table=table) is None
