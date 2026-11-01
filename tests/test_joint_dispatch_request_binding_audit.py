import hashlib
import json

import pytest

from test_dispatch_joint_quanta import (
    _argv, _portable_spec, _receipt, _stamp_receipt, _write_receipt,
    campaign, records_dir,
)
from dispatch_joint_quanta import FakeGateway, main
from prismaquant.cost_stage_checkpoint import canonical_json_sha256


@pytest.mark.parametrize('changed', [False, True])
def test_terminal_reuse_binds_the_current_request(tmp_path, campaign, records_dir, changed):
    receipt = tmp_path / 'receipt.json'
    _write_receipt(receipt, _receipt(campaign))
    _stamp_receipt(records_dir, receipt)
    gateway = FakeGateway(terminal=True)
    output = tmp_path / 'out'
    assert main(_argv(records_dir, output, receipt), _gateway=gateway) == 0
    gateway.submitted.clear()
    if changed:
        path = records_dir / 'layer-000.json'
        record = json.loads(path.read_text())
        manifest = record['read_set']
        source = tmp_path / 'slices' / 'layer-000.data-manifest.json'
        source.write_text(json.dumps({'slice': 'layer-000', 'revision': 2}))
        manifest['manifest_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        record.pop('identity_sha256')
        record['identity_sha256'] = canonical_json_sha256(record, where="revised request fixture")
        path.write_text(json.dumps(record))
    assert main(_argv(records_dir, output, receipt), _gateway=gateway) == (3 if changed else 0)
    assert gateway.submitted == []
