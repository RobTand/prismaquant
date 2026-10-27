"""Focused tests for the no-publish release rehearsal."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from release_dry_run import dry_run, isolated_environment, validate_tag


class ReleaseDryRunTests(unittest.TestCase):
    def test_environment_drops_credentials_and_user_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            with patch.dict(os.environ, {
                "GH_TOKEN": "sentinel", "TWINE_PASSWORD": "sentinel",
                "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "sentinel",
                "PAPERCLIP_API_KEY": "sentinel", "PYTHONPATH": "sentinel",
                "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_VALUE_0": "sentinel",
            }):
                environment = isolated_environment(home)
            child = subprocess.run(
                [sys.executable, "-c", "import json, os; print(json.dumps(dict(os.environ)))"],
                env=environment, text=True, capture_output=True, check=True,
            )
            observed = json.loads(child.stdout)
            self.assertNotIn("sentinel", observed.values())
            self.assertEqual(observed["HOME"], str(home))
            self.assertEqual(observed["PIP_NO_INDEX"], "1")
            self.assertEqual(observed["GIT_CONFIG_GLOBAL"], os.devnull)

    def test_version_and_artifact_validation(self):
        cases = [
            ("v0.16.2", "prismaquant-0.16.2-py3-none-any.whl", "prismaquant-0.16.2.tar.gz", True),
            ("v0.16.2rc1", "prismaquant-0.16.2rc1-py3-none-any.whl", "prismaquant-0.16.2rc1.tar.gz", True),
            ("v0.16.3", "prismaquant-0.16.2-py3-none-any.whl", "prismaquant-0.16.2.tar.gz", False),
            ("v0.16.2", "prismaquant-0.16.2-py3-none-any.whl", "prismaquant-0.16.3.tar.gz", False),
            ("v0.16.2", "other-0.16.2-py3-none-any.whl", "prismaquant-0.16.2.tar.gz", False),
            ("v0.16.2", "prismaquant-0.16.2-py3-none-any.whl", "other-0.16.2.tar.gz", False),
            ("0.16.2", "prismaquant-0.16.2-py3-none-any.whl", "prismaquant-0.16.2.tar.gz", False),
            ("--help", "prismaquant-0.16.2-py3-none-any.whl", "prismaquant-0.16.2.tar.gz", False),
            ("v0.16.2/remote", "prismaquant-0.16.2-py3-none-any.whl", "prismaquant-0.16.2.tar.gz", False),
            ("v0.16.2", None, "prismaquant-0.16.2.tar.gz", False),
            ("v0.16.2", "prismaquant-0.16.2-py3-none-any.whl", None, False),
        ]
        for tag, wheel, sdist, accepted in cases:
            with self.subTest(tag=tag, wheel=wheel, sdist=sdist):
                with tempfile.TemporaryDirectory() as directory:
                    dist = Path(directory)
                    for name in (wheel, sdist):
                        if name:
                            (dist / name).touch()
                    if accepted:
                        self.assertEqual(validate_tag(tag, dist), tag[1:])
                    else:
                        with self.assertRaises(ValueError):
                            validate_tag(tag, dist)

    def test_duplicate_artifacts_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            dist = Path(directory)
            for name in (
                "prismaquant-0.16.2-py3-none-any.whl",
                "prismaquant-0.16.3-py3-none-any.whl",
                "prismaquant-0.16.2.tar.gz",
            ):
                (dist / name).touch()
            with self.assertRaisesRegex(ValueError, "exactly one"):
                validate_tag("v0.16.2", dist)

    def test_bad_revision_writes_failure_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "evidence"
            with self.assertRaises(RuntimeError):
                dry_run(root, "not-a-commit", "v0.16.2", output)
            receipt = json.loads((output / "receipt.json").read_text())
            self.assertEqual(receipt["result"], "FAIL")
            self.assertNotIn("artifacts", receipt)
            self.assertTrue((output / "validation.log").is_file())

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            marker = output / "receipt.json"
            marker.write_text("previous evidence")
            with self.assertRaises(FileExistsError):
                dry_run(output, "HEAD", "v0.16.2", output)
            self.assertEqual(marker.read_text(), "previous evidence")


if __name__ == "__main__":
    unittest.main()
