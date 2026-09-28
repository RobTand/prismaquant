"""Golden byte tests for the tessera_joint_aura digest consolidation (PQ #1593).

The fixture records the OLD outputs of every consolidated site, captured on the
pre-change tree (origin/main 145ddd28) by running the old spellings verbatim.
Each test recomputes the same row through the NEW code path and asserts
equality, so wire receipts, marker fences, head-walk rosters, pretty bank
files and qualification seals recorded before the change still verify.
"""

import hashlib
import json
from pathlib import Path

from prismaquant.digests import (
    DIRECT_UTF8_STRICT,
    bytes_sha256hex,
    file_digest_sha256hex,
    indent2_json_file_bytes,
    newline_utf8_sha256,
)
from prismaquant.tessera_joint_aura import (
    QUALIFICATION_CELLS_SCHEMA,
    _json,
    _qualification_cells_sha256,
    _qualification_file_sha,
)

FIXTURE = Path(__file__).parent / "fixtures" / "digest_joint_aura_1593.json"


def _table():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    inputs, rows = document["inputs"], document["old_rows"]
    blobs = [bytes.fromhex(h) for h in inputs["blobs_hex"]]
    cells = {
        tuple(key.split("|", 1)): value
        for key, value in inputs["cells"].items()
    }
    return inputs, rows, blobs, cells


def test_wire_blob_digests_match_the_old_sha256_spelling():
    _, rows, blobs, _ = _table()
    for blob, old in zip(blobs, rows["blob_sha256"]):
        assert bytes_sha256hex(blob) == old  # was hashlib.sha256(blob).hexdigest()
        assert hashlib.sha256(blob).hexdigest() == old


def test_marker_digests_match_the_old_read_bytes_spelling(tmp_path):
    inputs, rows, _, _ = _table()
    marker = tmp_path / "render-origin.marker"
    marker.write_bytes(bytes.fromhex(inputs["marker_hex"]))
    # both call sites read the marker's bytes then hash them in one shot
    assert bytes_sha256hex(marker.read_bytes()) == rows["marker_sha256"]
    assert hashlib.sha256(marker.read_bytes()).hexdigest() == rows["marker_sha256"]


def test_head_walk_roster_digest_matches_the_old_newline_join_spelling():
    inputs, rows, _, _ = _table()
    for roster, old in zip(inputs["rosters"], rows["roster_sha256"]):
        # the head-walk roster keeps caller order; it is never sorted here
        assert newline_utf8_sha256(roster) == old
        assert hashlib.sha256("\n".join(roster).encode("utf-8")).hexdigest() == old
    roster = inputs["rosters"][2]
    assert newline_utf8_sha256(reversed(roster)) != rows["roster_sha256"][2]


def test_the_pretty_bank_file_matches_the_old_indent2_spelling(tmp_path):
    inputs, rows, _, _ = _table()
    value = inputs["json_file_value"]
    assert indent2_json_file_bytes(value).hex() == rows["pretty_file_hex"]
    target = tmp_path / "bank.json"
    _json(target, value)
    assert target.read_bytes().hex() == rows["pretty_file_hex"]
    # old spelling, verbatim: json.dumps(indent=2, sort_keys=True,
    # allow_nan=False) + "\n", encoded with the default UTF-8
    assert (json.dumps(value, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode().hex() == rows["pretty_file_hex"]


def test_qualification_rows_match_the_old_direct_utf8_spelling():
    _, rows, _, cells = _table()
    for (name, fmt), old in zip(sorted(cells), rows["cell_row_hex"]):
        cell = cells[name, fmt]
        assert DIRECT_UTF8_STRICT.encoded((name, fmt, {
            'anchor': cell['anchor'], 'record': cell['record'],
            'render': cell['render'], 'wire': cell['wire'],
            'render_origin': cell['render_origin'],
        })).hex() == old


def test_the_qualification_seal_is_unchanged_end_to_end():
    _, rows, _, cells = _table()
    assert _qualification_cells_sha256(cells) == rows["cells_seal_sha256"]
    # old spelling, verbatim: schema line, sorted rows, 8-byte big-endian
    # lengths -- the framing this seal is identified by
    digest = hashlib.sha256(QUALIFICATION_CELLS_SCHEMA.encode() + b"\n")
    for name, fmt in sorted(cells):
        cell = cells[name, fmt]
        row = json.dumps((name, fmt, {
            'anchor': cell['anchor'], 'record': cell['record'],
            'render': cell['render'], 'wire': cell['wire'],
            'render_origin': cell['render_origin'],
        }), sort_keys=True, separators=(',', ':'), ensure_ascii=False,
            allow_nan=False).encode('utf-8')
        digest.update(len(row).to_bytes(8, 'big'))
        digest.update(row)
    assert digest.hexdigest() == rows["cells_seal_sha256"]


def test_the_fenced_file_digest_matches_the_old_file_digest_spelling(tmp_path):
    path = tmp_path / "qualification-input.bin"
    payload = bytes(range(256)) * 41 + "锚点".encode("utf-8")
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    # old spelling: hashlib.file_digest(handle, 'sha256').hexdigest() over the
    # stat-fenced descriptor
    assert _qualification_file_sha(path) == expected
    with open(path, "rb") as handle:
        assert file_digest_sha256hex(handle) == expected
    # reads from the handle's current position, like the old call did
    with open(path, "rb") as handle:
        handle.seek(256)
        assert file_digest_sha256hex(handle) == hashlib.sha256(payload[256:]).hexdigest()
