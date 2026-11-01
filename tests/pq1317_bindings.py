"""Resolve package-local harness bytes without relabeling recorded launches."""
import hashlib
import json
from pathlib import Path


def harness_binding(package, selector):
    package = Path(package).resolve()
    bindings = json.loads((package / "results/harness-bindings.json").read_bytes())
    binding = bindings[selector]
    relative = Path(binding["root"])
    assert not relative.is_absolute()
    root = (package / relative).resolve()
    assert root.is_relative_to(package) and root != package
    actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.glob("*.py")}
    assert actual and actual == binding["sha256"], selector
    if selector == "current_validation":
        assert root == package / "harness"
    else:
        assert root != package / "harness"
    return root, actual
