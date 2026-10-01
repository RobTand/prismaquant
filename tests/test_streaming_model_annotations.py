"""Private streaming helper annotations resolve in their declaring module."""
from typing import get_type_hints

import pytest
import torch

from prismaquant import streaming_model


@pytest.mark.parametrize("name,parameter", [
    ("_init_rotary_inplace", "base_model"),
    ("_module_has_meta_tensors", "module"),
])
def test_streaming_module_annotation_resolves(name, parameter):
    hints = get_type_hints(getattr(streaming_model, name))
    assert hints[parameter] is torch.nn.Module
