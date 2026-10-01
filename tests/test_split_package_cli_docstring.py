"""The manifest builder's argument help does not require Python docstrings."""
from __future__ import annotations

import pytest

from tools import build_stagea_split_package as builder


@pytest.mark.parametrize("doc", [None, "Description.\n\nDetails."])
def test_argument_help_survives_stripped_module_docstring(monkeypatch, capsys, doc):
    monkeypatch.setattr(builder, "__doc__", doc)
    with pytest.raises(SystemExit) as exc:
        builder.main(["--help"])
    assert exc.value.code == 0
    assert "--original-manifest" in capsys.readouterr().out
