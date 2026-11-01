"""Provisioning controls test authentication, not canonical admission qualification."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

import _rung_allowability_producer as adapter


ENVIRONMENT = ("TESSERA_RUNG_ALLOWABILITY_MODULE",
               "TESSERA_RUNG_ALLOWABILITY_MODULE_SHA256", "TESSERA_PRODUCER_PYTHON")
DECLARATION = Path(__file__).parent / "fixtures/rung_allowability_v3/producer.json"
SOURCE = b'''
def validate_index(payload):
    if "refusal" in payload:
        raise ValueError(payload["refusal"])
    return payload

def validate_table(payload):
    return payload

def admit_rung(payload, **kwargs):
    return {"payload": payload, "kwargs": kwargs}
'''


class TestExternalProducer(unittest.TestCase):
    def setUp(self):
        self._clear_resolution()
        self.addCleanup(self._clear_resolution)
        self.directory = tempfile.TemporaryDirectory(prefix="pq-producer-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.declared = json.loads(DECLARATION.read_text())
        self.declared["sha256"] = hashlib.sha256(SOURCE).hexdigest()
        original_read = Path.read_text

        def read_text(path, *args, **kwargs):
            if path == DECLARATION:
                return json.dumps(self.declared)
            return original_read(path, *args, **kwargs)

        reader = patch.object(Path, "read_text", read_text)
        reader.start()
        self.addCleanup(reader.stop)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        for name in ENVIRONMENT:
            os.environ.pop(name, None)
        transport = patch.object(adapter, "urlopen", side_effect=lambda *a, **k: io.BytesIO(SOURCE))
        self.transport = transport.start()
        self.addCleanup(transport.stop)

    @staticmethod
    def _clear_resolution():
        for name in ("metadata_source", "_download_source"):
            function = getattr(adapter, name, None)
            if hasattr(function, "cache_clear"):
                function.cache_clear()

    def _supply(self, raw=SOURCE, name="producer.py"):
        path = self.root / name
        path.write_bytes(raw)
        os.environ[ENVIRONMENT[0]] = str(path)
        os.environ[ENVIRONMENT[1]] = hashlib.sha256(raw).hexdigest()
        return path

    def _assert_dependency(self, error):
        message = str(error)
        for key in ("repository", "commit", "path", "sha256"):
            self.assertIn(self.declared[key], message)

    def test_unset_environment_resolves_declared_source_and_current_interpreter(self):
        producer = adapter.ExternalProducer()
        with patch.object(adapter.subprocess, "run", wraps=adapter.subprocess.run) as run:
            self.assertEqual(producer.validate_index({"index": 1}), {"index": 1})
            self.assertEqual(producer.validate_table({"table": 2}), {"table": 2})
            self.assertEqual(producer.admit_rung({"table": 2}, rung=1024),
                             {"payload": {"table": 2}, "kwargs": {"rung": 1024}})
        for call in run.call_args_list:
            self.assertEqual(call.args[0][0], sys.executable)
        url = (f"https://raw.githubusercontent.com/{self.declared['repository']}/"
               f"{self.declared['commit']}/{self.declared['path']}")
        self.transport.assert_called_once_with(url, timeout=60)
        self.assertEqual(producer.evidence["source_sha256"], self.declared["sha256"])

    def test_downloaded_source_digest_mismatch_refuses_before_execution(self):
        self.transport.side_effect = lambda *a, **k: io.BytesIO(b"raise RuntimeError('executed')")
        with patch.object(adapter.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "SHA-256") as error:
                adapter.ExternalProducer().validate_table({})
        run.assert_not_called()
        self._assert_dependency(error.exception)

    def test_download_failure_names_declared_dependency(self):
        self.transport.side_effect = URLError("controlled transport failure")
        with patch.object(adapter.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "controlled transport failure") as error:
                adapter.ExternalProducer().validate_table({})
        run.assert_not_called()
        self._assert_dependency(error.exception)
        self.transport.side_effect = lambda *a, **k: io.BytesIO(SOURCE)
        self.assertEqual(adapter.ExternalProducer().validate_table({}), {})

    def test_authenticated_explicit_source_runs_without_network(self):
        self._supply()
        self.transport.side_effect = AssertionError("network forbidden")
        interpreter = str(Path(sys.executable).resolve())
        os.environ[ENVIRONMENT[2]] = interpreter
        with patch.object(adapter.subprocess, "run", wraps=adapter.subprocess.run) as run:
            self.assertEqual(adapter.ExternalProducer().validate_table({"local": True}), {"local": True})
        self.assertEqual(run.call_args.args[0][0], interpreter)
        self.transport.assert_not_called()

    def test_incomplete_or_conflicting_override_refuses(self):
        cases = ("missing_digest", "missing_module", "empty_digest", "empty_module",
                 "conflicting_digest", "different_source", "unreadable", "changed")
        for case in cases:
            with self.subTest(case=case):
                path = self._supply()
                if case == "missing_digest":
                    os.environ.pop(ENVIRONMENT[1])
                elif case == "missing_module":
                    os.environ.pop(ENVIRONMENT[0])
                elif case == "empty_digest":
                    os.environ[ENVIRONMENT[1]] = ""
                elif case == "empty_module":
                    os.environ[ENVIRONMENT[0]] = ""
                elif case == "conflicting_digest":
                    os.environ[ENVIRONMENT[1]] = "0" * 64
                elif case == "different_source":
                    self._supply(SOURCE + b"\n# different producer\n")
                elif case == "unreadable":
                    path.unlink()
                else:
                    path.write_bytes(SOURCE + b"\n# changed\n")
                with patch.object(adapter.subprocess, "run") as run:
                    with self.assertRaises(ValueError) as error:
                        adapter.ExternalProducer().validate_table({})
                run.assert_not_called()
                self._assert_dependency(error.exception)
                self.assertIn(ENVIRONMENT[0], str(error.exception))
        self.transport.assert_not_called()

    def test_changed_source_refuses_on_subsequent_invocation(self):
        for supplied in (False, True):
            with self.subTest(supplied=supplied):
                for name in ENVIRONMENT:
                    os.environ.pop(name, None)
                if supplied:
                    self._supply()
                producer = adapter.ExternalProducer()
                producer.validate_table({})
                path, _ = adapter.metadata_source()
                Path(path).write_bytes(SOURCE + b"\n# changed\n")
                with self.assertRaisesRegex(ValueError, "SHA-256") as error:
                    producer.validate_table({})
                self._assert_dependency(error.exception)
                self.assertIsNone(producer.evidence)
                self._clear_resolution()
        path = self._supply()
        original_run = adapter.subprocess.run

        def change_before_child(*args, **kwargs):
            path.write_bytes(SOURCE + b"\n# changed before child read\n")
            return original_run(*args, **kwargs)

        with patch.object(adapter.subprocess, "run", side_effect=change_before_child):
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                producer.validate_table({})
        self.assertIsNone(producer.evidence)

    def test_source_selection_changes_do_not_reuse_stale_resolution(self):
        producer = adapter.ExternalProducer()
        producer.validate_table({})
        first = self._supply(name="first.py")
        self.assertEqual(adapter.metadata_source()[0], str(first))
        second = self._supply(name="second.py")
        first.unlink()
        self.assertEqual(adapter.metadata_source()[0], str(second))
        self.assertEqual(producer.validate_table({"second": True}), {"second": True})
        os.environ[ENVIRONMENT[1]] = "0" * 64
        with self.assertRaises(ValueError):
            producer.validate_table({})
        for name in ENVIRONMENT:
            os.environ.pop(name, None)
        path, digest = adapter.metadata_source()
        self.assertNotEqual(path, str(second))
        self.assertEqual(digest, self.declared["sha256"])
        producer.validate_table({})
        self.transport.assert_called_once()

    def test_child_import_isolation_and_digest_evidence(self):
        self._supply()
        producer = adapter.ExternalProducer()
        producer.validate_table({})
        self.assertEqual(producer.evidence, {
            "source_sha256": self.declared["sha256"],
            "interpreter_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
            "forbidden_imports": [],
        })
        stdlib_source = b'''
from __future__ import annotations
from dataclasses import asdict, dataclass

@dataclass
class Payload:
    value: str

def validate_table(payload):
    return asdict(Payload(payload["value"]))
'''
        self.declared["sha256"] = hashlib.sha256(stdlib_source).hexdigest()
        self._supply(stdlib_source)
        self.assertEqual(producer.validate_table({"value": "stdlib"}), {"value": "stdlib"})
        self.assertEqual(producer.evidence["forbidden_imports"], [])
        for module in ("tessera.serving", "torch", "triton", "vllm"):
            for phase in ("load", "call"):
                for mechanism in ("__import__", "importlib"):
                    with self.subTest(module=module, phase=phase, mechanism=mechanism):
                        expression = (f"__import__({module!r})" if mechanism == "__import__" else
                                      f"__import__('importlib').import_module({module!r})")
                        attempt = f"try:\n    {expression}\nexcept ImportError:\n    pass\n"
                        code = (attempt + "def validate_table(payload):\n    return payload\n"
                                if phase == "load" else
                                "def validate_table(payload):\n" +
                                "".join("    " + line + "\n" for line in attempt.splitlines()) +
                                "    return payload\n")
                        raw = code.encode()
                        self.declared["sha256"] = hashlib.sha256(raw).hexdigest()
                        self._supply(raw)
                        root_module = module.split(".")[0]
                        with self.assertRaisesRegex(RuntimeError, f"forbidden producer import.*{root_module}"):
                            producer.validate_table({})
                        self.assertIsNone(producer.evidence)

    def test_canonical_value_error_is_preserved(self):
        self._supply()
        producer = adapter.ExternalProducer()
        message = "canonical refusal: unmeasured rung"
        with self.assertRaises(ValueError) as error:
            producer.validate_index({"refusal": message})
        self.assertEqual(str(error.exception), message)
        self.assertEqual(producer.evidence["source_sha256"], self.declared["sha256"])

    def test_publication_restores_consumer_producer_binding(self):
        from pytest import MonkeyPatch
        from prismaquant import rung_allowability as consumer
        from test_rung_allowability import publication

        original = consumer._producer_api
        with MonkeyPatch.context() as monkeypatch:
            root = publication.__wrapped__(self.root, monkeypatch)
            self.assertTrue(root.is_dir())
            self.assertIsInstance(consumer._producer_api(), adapter.ExternalProducer)
            self.assertIsNot(consumer._producer_api, original)
        self.assertIs(consumer._producer_api, original)


if __name__ == "__main__":
    unittest.main()
