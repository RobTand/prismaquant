"""Explicit Python 3.14 scan contracts; not part of the Python 3.12 test population."""
from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ci import bazel_artifacts


class ScanRuntimeContract(unittest.TestCase):
    def test_native_zstd_secret_detection(self):
        from compression import zstd
        self.assertEqual(sys.version_info[:2], (3, 14))
        with tempfile.TemporaryDirectory(prefix='pq-scan-runtime-') as directory:
            path = Path(directory) / 'private.binpb.zst'
            path.write_bytes(zstd.compress(b'prefix compressed-scan-sentinel suffix'))
            output = io.StringIO()
            with patch.dict(os.environ, {'PQ_RUNTIME_SCAN_KEY': 'compressed-scan-sentinel'}):
                with contextlib.redirect_stdout(output):
                    code = bazel_artifacts.scanner.main(['--env', 'PQ_RUNTIME_SCAN_KEY', str(path)])
            self.assertEqual(code, 1)
            self.assertIn('secret detected', output.getvalue())
            self.assertNotIn('not scanned', output.getvalue())
            self.assertNotIn('compressed-scan-sentinel', output.getvalue())

    def test_json_publication_requires_the_native_runtime(self):
        from compression import zstd
        self.assertTrue(bazel_artifacts.scan_runtime_available())
        with tempfile.TemporaryDirectory(prefix='pq-scan-runtime-') as directory:
            upload = Path(directory) / 'upload'
            upload.mkdir()
            with patch.dict(os.environ, {'PQ_RUNTIME_SCAN_KEY': 'native-upload-sentinel'}):
                bazel_artifacts.publish_json(upload, {'result.json': b'{"verdict":"pass"}'},
                                             ['PQ_RUNTIME_SCAN_KEY'])
                self.assertEqual((upload / 'result.json').read_bytes(), b'{"verdict":"pass"}')
                # Even fully decodable non-JSON logs must remain private.
                (upload / 'private.zst').write_bytes(zstd.compress(b'no secret'))
                self.assertEqual(bazel_artifacts.scan_upload(upload, ['PQ_RUNTIME_SCAN_KEY']), 75)


if __name__ == '__main__':
    unittest.main()
