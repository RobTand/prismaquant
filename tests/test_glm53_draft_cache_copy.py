"""CPU regression of the draft copy using the pinned validator field shape."""
from dataclasses import field, replace
import importlib.util
from pathlib import Path
from typing import ClassVar

from pydantic import Field, field_validator, model_validator
from pydantic.dataclasses import dataclass
import pytest

PATCH = (Path(__file__).resolve().parents[1] / "prismaquant" /
         "serving_runtime_patches/glm53_draft_cache_copy/patch_draft_cache_copy.py")
spec = importlib.util.spec_from_file_location("draft_cache_copy_patch", PATCH)
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)


@dataclass
class CacheConfig:
    # These fields and validators reproduce the pinned config/cache.py shape.
    DEFAULT_BLOCK_SIZE: ClassVar[int] = 16
    block_size: int = Field(default=None, gt=0)
    cache_dtype: str = "auto"
    user_specified_block_size: bool = field(default=False, init=False)
    _block_size_resolved: bool = field(default=False, init=False)

    @field_validator("block_size", mode="wrap")
    @classmethod
    def _skip_none_validation(cls, value, handler):
        return value if value is None else handler(value)

    @model_validator(mode="after")
    def _apply_block_size_default(self):
        if self._block_size_resolved:
            return self
        self._block_size_resolved = True
        if self.block_size is None:
            self.block_size = self.DEFAULT_BLOCK_SIZE
        else:
            self.user_specified_block_size = True
        return self


def draft_copy():
    namespace = {"replace": replace}
    exec(patch.HELPER, namespace)
    return namespace["_draft_cache_config"]


@pytest.mark.parametrize("resolved_block", [16, 64, 256])
def test_draft_dtype_copy_keeps_automatic_backend_selection(resolved_block):
    target = CacheConfig()
    target.block_size = resolved_block  # The target backend may resolve it.
    draft = draft_copy()(target, "fp8_ds_mla")
    assert draft is not target
    assert draft.cache_dtype == "fp8_ds_mla"
    assert draft.block_size == resolved_block
    assert draft.user_specified_block_size is False
    assert target.cache_dtype == "auto"
    assert target.user_specified_block_size is False


@pytest.mark.parametrize("block", [16, 64, 256])
def test_explicit_block_size_stays_a_backend_constraint(block):
    target = CacheConfig(block_size=block)
    draft = draft_copy()(target, "fp8_ds_mla")
    assert draft.block_size == block
    assert draft.user_specified_block_size is True


def test_nested_revalidation_preserves_restored_provenance():
    @dataclass
    class Config:
        cache_config: CacheConfig
    target = CacheConfig()
    draft = draft_copy()(target, "fp8_ds_mla")
    nested = Config(cache_config=draft)
    assert nested.cache_config.user_specified_block_size is False


def test_candidate_refuses_an_unrelated_or_already_edited_runtime():
    with pytest.raises(ValueError, match="not the base image"):
        patch.patched_source(b"unrelated serving runtime")
