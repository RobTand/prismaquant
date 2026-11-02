"""Real native DSv4 import left by the unsupported AutoModel config refusal."""
import sys
from pathlib import Path

import pytest
import transformers


def test_native_dsv4_import_still_refuses_the_vendored_override():
    from prismaquant import vendored
    from prismaquant.model_profiles.registry import (
        DeadVendoredOverrideError,
        profile_from_config,
    )

    native = sys.modules["transformers.models.deepseek_v4"]
    assert transformers.__file__ is not None
    assert native.__file__ is not None
    installed = Path(transformers.__file__).resolve().parent
    assert Path(native.__file__).resolve() == (
        installed / "models/deepseek_v4/__init__.py")
    with pytest.raises(vendored.VendoredOverrideError, match="already imported"):
        vendored.register_deepseek_v4()
    with pytest.raises(DeadVendoredOverrideError, match="already imported"):
        profile_from_config({"model_type": "deepseek_v4"})
